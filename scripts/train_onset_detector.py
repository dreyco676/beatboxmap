#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train a small CNN onset detector on AVP Personal corpus and export to ONNX.

Usage
-----
    python scripts/train_onset_detector.py
    python scripts/train_onset_detector.py --epochs 50 --output models/onset_cnn.onnx
    python scripts/train_onset_detector.py --loso        # LOSO eval before final export

Requires (training only, not runtime)
--------------------------------------
    pip install torch librosa

The exported ONNX model is used by voxkit.dsp.onsets.OnsetDetector at runtime.
onnxruntime and librosa are the only runtime dependencies; torch is not required.

ONNX interface
--------------
    Input:  mel_spectrogram  (1, 1, T, 40)  float32  log-mel, pre-normalised
    Output: onset_logits     (1, T)          float32  pre-sigmoid logits
"""

from __future__ import annotations

import argparse
import csv
import warnings
from math import gcd
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
_AVP_DIR = _REPO_ROOT / "data" / "avp" / "AVP_Dataset" / "Personal"
_DEFAULT_OUT = _REPO_ROOT / "models" / "onset_cnn.onnx"

# ---------------------------------------------------------------
# Mel-spectrogram parameters — must match OnsetDetector._compute_mel()
# ---------------------------------------------------------------
SR = 16_000
HOP = 80         # 5 ms  (matches OnsetDetector._HOP; max timing error ≤ 2.5 ms)
N_FFT = 512      # 32 ms
N_MELS = 40
FMIN = 27.5
FMAX = 8_000.0

# ---------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------
EPOCHS = 50
LR = 1e-3
POS_WEIGHT = 10.0    # upweight onset frames (class ratio ~1:43)
LABEL_RADIUS = 1     # frames either side of onset labeled positive (±10 ms)
AUG_DB = 6.0         # ±dB amplitude augmentation in log-mel domain


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------

def _load_wav(path: Path) -> np.ndarray:
    import scipy.io.wavfile as wv
    sr, data = wv.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if sr != SR:
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        data = resample_poly(data, SR // g, sr // g).astype(np.float32)
    return data


def _load_onsets(path: Path) -> list[float]:
    onsets: list[float] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row:
                try:
                    onsets.append(float(row[0]))
                except ValueError:
                    pass
    return onsets


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Return log-mel spectrogram (T, N_MELS) float32."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    return librosa.power_to_db(mel + 1e-8).T.astype(np.float32)


def make_labels(onsets: list[float], n_frames: int) -> np.ndarray:
    """Binary onset labels, 1 within LABEL_RADIUS frames of each onset."""
    y = np.zeros(n_frames, dtype=np.float32)
    for t in onsets:
        f = int(round(t * SR / HOP))
        for d in range(-LABEL_RADIUS, LABEL_RADIUS + 1):
            fi = f + d
            if 0 <= fi < n_frames:
                y[fi] = 1.0
    return y


def load_corpus(avp_dir: Path, verbose: bool = True) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Return list of (mel_frames, binary_labels, participant_id)."""
    items: list[tuple[np.ndarray, np.ndarray, str]] = []
    participant_dirs = sorted(
        [d for d in avp_dir.iterdir() if d.is_dir() and d.name.startswith("Participant_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for pdir in participant_dirs:
            for wav_path in sorted(pdir.glob("*.wav")):
                if "Improvisation" in wav_path.stem:
                    continue
                csv_path = wav_path.with_suffix(".csv")
                if not csv_path.exists():
                    continue
                audio = _load_wav(wav_path)
                onsets = _load_onsets(csv_path)
                if not onsets:
                    continue
                mel = compute_mel(audio)
                labels = make_labels(onsets, len(mel))
                items.append((mel, labels, pdir.name))
    if verbose:
        n_frames = sum(len(m) for m, _, _ in items)
        n_onset = sum(int(l.sum()) for _, l, _ in items)
        print(f"Loaded {len(items)} files · {n_frames:,} frames · "
              f"{n_onset:,} onset frames ({n_onset/n_frames*100:.1f}% positive)")
    return items


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

def build_model(mel_mean: float, mel_std: float) -> "torch.nn.Module":
    import torch
    import torch.nn as nn

    class OnsetCNN(nn.Module):
        """
        Fully-convolutional onset detector.

        Input:  (1, 1, T, 40)  log-mel spectrogram
        Output: (1, T)         pre-sigmoid onset logits

        Architecture compresses the frequency axis through three Conv2d layers
        (40→33→26→19 mel bins) then collapses it entirely while adding ±70 ms
        temporal context, producing one logit per 10 ms frame.
        """

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mel_mean", torch.tensor(mel_mean, dtype=torch.float32))
            self.register_buffer("mel_std",  torch.tensor(mel_std,  dtype=torch.float32))
            self.layers = nn.Sequential(
                nn.Conv2d(1,  16, kernel_size=(3,  8), padding=(1, 0)), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(3,  8), padding=(1, 0)), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(3,  8), padding=(1, 0)), nn.ReLU(),
                # Collapse remaining 19 mel bins; add ±70 ms context (7 frames each side)
                nn.Conv2d(32, 16, kernel_size=(7, 19), padding=(3, 0)), nn.ReLU(),
                nn.Conv2d(16,  1, kernel_size=(1,  1)),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = (x - self.mel_mean) / (self.mel_std + 1e-8)
            h = self.layers(x)                  # (1, 1, T, 1)
            return h.squeeze(1).squeeze(-1)     # (1, T)

    return OnsetCNN()


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------

def train(
    corpus: list[tuple[np.ndarray, np.ndarray, str]],
    epochs: int = EPOCHS,
    verbose: bool = True,
) -> "torch.nn.Module":
    import torch
    import torch.nn as nn

    # Global normalisation constants from training data
    all_mel = np.vstack([m for m, _, _ in corpus])
    mel_mean = float(all_mel.mean())
    mel_std  = float(all_mel.std())
    if verbose:
        print(f"Mel stats: mean={mel_mean:.2f}  std={mel_std:.2f}")

    model = build_model(mel_mean, mel_std)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(POS_WEIGHT, dtype=torch.float32)
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(0)

    model.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(corpus))
        epoch_loss = 0.0
        for idx in order:
            mel, labels, _ = corpus[idx]
            # Amplitude augmentation: random ±AUG_DB shift in log-mel domain
            mel_aug = mel + rng.uniform(-AUG_DB, AUG_DB)
            x = torch.from_numpy(mel_aug[np.newaxis, np.newaxis]).float()  # (1,1,T,40)
            y = torch.from_numpy(labels[np.newaxis]).float()               # (1,T)
            logits = model(x)   # (1,T)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss/len(corpus):.4f}")

    return model


# ---------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------

def loso_eval(
    corpus: list[tuple[np.ndarray, np.ndarray, str]],
    epochs: int = EPOCHS,
    tol_ms: float = 50.0,
) -> float:
    """Run Leave-One-Subject-Out eval; return mean F-measure across all folds."""
    import torch
    from voxkit.dsp.onset_eval import _align_pairs

    participants = sorted(set(p for _, _, p in corpus))
    tol = tol_ms / 1000.0
    fold_f: list[float] = []

    for held_out in participants:
        train_data = [(m, l, p) for m, l, p in corpus if p != held_out]
        test_data  = [(m, l, p) for m, l, p in corpus if p == held_out]

        model = train(train_data, epochs=epochs, verbose=False)
        model.eval()

        for mel, labels, _ in test_data:
            with torch.no_grad():
                x = torch.from_numpy(mel[np.newaxis, np.newaxis]).float()
                logits = model(x)[0].numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            pred = _peak_pick(probs)
            ref = [i * HOP / SR for i, v in enumerate(labels) if v > 0.5]
            # Deduplicate labels that span radius
            ref_dedup: list[float] = []
            for t in ref:
                if not ref_dedup or t - ref_dedup[-1] > tol:
                    ref_dedup.append(t)

            pairs = _align_pairs(pred, ref_dedup, tol)
            tp = len(pairs)
            p = tp / len(pred) if pred else 0.0
            r = tp / len(ref_dedup) if ref_dedup else 0.0
            fold_f.append(2 * p * r / (p + r) if p + r else 0.0)

    mean_f = float(np.mean(fold_f))
    print(f"LOSO mean F = {mean_f:.3f}  median = {float(np.median(fold_f)):.3f}  "
          f"({len(fold_f)} files across {len(participants)} participants)")
    return mean_f


# ---------------------------------------------------------------
# Peak picker (used in LOSO eval and by OnsetDetector at runtime)
# ---------------------------------------------------------------

def _peak_pick(
    probs: np.ndarray,
    threshold: float = 0.5,
    min_gap_s: float = 0.050,
) -> list[float]:
    min_gap = max(1, int(min_gap_s * SR / HOP))
    onsets: list[float] = []
    i = 0
    while i < len(probs):
        if probs[i] > threshold:
            w_end = min(i + min_gap, len(probs))
            peak_i = i + int(np.argmax(probs[i:w_end]))
            onsets.append(float(peak_i * HOP / SR))
            i = peak_i + min_gap
        else:
            i += 1
    return onsets


# ---------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------

def export_onnx(model: "torch.nn.Module", out_path: Path) -> None:
    import torch

    model.eval()
    dummy = torch.zeros(1, 1, 100, N_MELS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["mel_spectrogram"],
        output_names=["onset_logits"],
        dynamic_axes={"mel_spectrogram": {2: "time"}, "onset_logits": {1: "time"}},
        opset_version=17,
    )
    print(f"Exported ONNX model → {out_path}")

    # Verify round-trip
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    dummy_np = dummy.numpy()
    ort_out = sess.run(None, {"mel_spectrogram": dummy_np})[0]
    torch_out = model(dummy).detach().numpy()
    max_diff = float(np.abs(ort_out - torch_out).max())
    print(f"ONNX round-trip max diff: {max_diff:.2e}  {'OK' if max_diff < 1e-4 else 'WARNING'}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN onset detector → ONNX")
    p.add_argument("--avp-dir", type=Path, default=_AVP_DIR)
    p.add_argument("--output",  type=Path, default=_DEFAULT_OUT)
    p.add_argument("--epochs",  type=int,  default=EPOCHS)
    p.add_argument("--loso",    action="store_true",
                   help="Run LOSO evaluation before final training (slow)")
    p.add_argument("--quiet",   action="store_true")
    return p.parse_args()


def main() -> None:
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "src"))

    args = _parse_args()
    verbose = not args.quiet

    if not args.avp_dir.exists():
        sys.exit(f"AVP dataset not found at {args.avp_dir}")

    print(f"Loading AVP corpus from {args.avp_dir} ...")
    corpus = load_corpus(args.avp_dir, verbose=verbose)

    if args.loso:
        print(f"\nRunning LOSO evaluation ({len(set(p for _,_,p in corpus))} folds) ...")
        loso_eval(corpus, epochs=args.epochs)

    print(f"\nTraining final model on all {len(corpus)} files for {args.epochs} epochs ...")
    model = train(corpus, epochs=args.epochs, verbose=verbose)

    export_onnx(model, args.output)
    print("\nDone. Run `python scripts/train_onset_detector.py --loso` to see LOSO metrics.")


if __name__ == "__main__":
    main()
