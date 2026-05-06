#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract embeddings from AVP-LVT v4 and run Leave-One-Subject-Out classifier eval.

Usage
-----
    # PANNs only (fastest)
    python scripts/run_avp_loso.py --substrate panns

    # BEATs only
    python scripts/run_avp_loso.py --substrate beats

    # Both substrates → bake-off decision (Q33)
    python scripts/run_avp_loso.py --substrate both

    # Re-use cached embeddings (skip extraction)
    python scripts/run_avp_loso.py --substrate panns --use-cache

Dataset expected at:
    data/avp/AVP_Dataset/Personal/Participant_N/PN_<Sound>_Personal.wav + .csv

Outputs:
    data/avp_loso_<substrate>.json   — LOSO results + bootstrap CI
    data/avp_embeddings_<substrate>.npz  — embedding cache
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data" / "avp" / "AVP_Dataset" / "Personal"
_MODELS_DIR = _REPO_ROOT / "models"

_ONNX_PATHS = {
    "panns": _MODELS_DIR / "panns_cnn14_16k.onnx",
    "beats": _MODELS_DIR / "beats_iter3plus_as2m.onnx",
}

# substrate CLI name → EmbeddingExtractor substrate_id
_SUBSTRATE_IDS = {
    "panns": "panns_cnn14",
    "beats": "beats",
}

# AVP filename stem → taxonomy class_id
_SOUND_TO_CLASS = {
    "Kick":     "kick",
    "HHclosed": "closed_hat",
    "HHopened": "open_hat",
    "Snare":    "snare",
}

_TARGET_SR = 16_000


# ---------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------

def _load_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    """Return (mono float32 audio, sample_rate)."""
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
    return data, sr


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _load_onset_times(csv_path: Path) -> list[float]:
    """Read onset times (seconds) from AVP CSV (no header, col 0 = time)."""
    times = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if row:
                try:
                    times.append(float(row[0]))
                except ValueError:
                    pass
    return times


# ---------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------

def extract_embeddings_for_substrate(
    substrate: str,
    data_dir: Path,
    *,
    verbose: bool = True,
) -> dict:
    """Return dict with keys: embeddings (N,D), labels (N,), subjects (N,).

    Scans all Personal/Participant_N directories and all 4 sound files.
    """
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from voxkit.classifier.embeddings import EmbeddingExtractor

    onnx_path = _ONNX_PATHS[substrate]
    if not onnx_path.exists():
        script = "convert_panns_to_onnx.py" if substrate == "panns" else "convert_beats_to_onnx.py"
        sys.exit(f"ERROR: ONNX model not found: {onnx_path}\nRun:  python scripts/{script}")

    extractor = EmbeddingExtractor(onnx_path, substrate_id=_SUBSTRATE_IDS[substrate])
    if verbose:
        print(f"\n[{substrate}] Loading ONNX model: {onnx_path.name}")
        print(f"[{substrate}] Embedding dim: {extractor.embedding_dim}")

    participant_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Participant_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not participant_dirs:
        sys.exit(f"ERROR: No Participant_N directories found in {data_dir}")

    if verbose:
        print(f"[{substrate}] Found {len(participant_dirs)} participants")

    all_embeddings: list[np.ndarray] = []
    all_labels: list[str] = []
    all_subjects: list[str] = []

    for p_dir in participant_dirs:
        subject_id = p_dir.name  # e.g. "Participant_1"
        p_num = subject_id.split("_")[1]
        n_emb_before = len(all_embeddings)

        for sound_key, class_id in _SOUND_TO_CLASS.items():
            # e.g. P1_Kick_Personal.wav / P1_Kick_Personal.csv
            wav_path = p_dir / f"P{p_num}_{sound_key}_Personal.wav"
            csv_path = p_dir / f"P{p_num}_{sound_key}_Personal.csv"

            if not wav_path.exists() or not csv_path.exists():
                if verbose:
                    print(f"  WARNING: missing {wav_path.name} or {csv_path.name}, skipping")
                continue

            audio_raw, src_sr = _load_wav_mono_float32(wav_path)
            audio = _resample(audio_raw, src_sr, _TARGET_SR)
            onsets = _load_onset_times(csv_path)

            if not onsets:
                if verbose:
                    print(f"  WARNING: no onsets in {csv_path.name}, skipping")
                continue

            # Filter out onsets beyond audio length
            max_t = len(audio) / _TARGET_SR
            onsets = [t for t in onsets if 0 < t < max_t]

            try:
                embs = extractor.extract_at_onsets(audio, onsets, _TARGET_SR)
            except Exception as exc:
                if verbose:
                    print(f"  WARNING: extraction failed for {wav_path.name}: {exc}")
                continue

            for emb in embs:
                all_embeddings.append(emb)
                all_labels.append(class_id)
                all_subjects.append(subject_id)

        n_new = len(all_embeddings) - n_emb_before
        if verbose:
            print(f"  {subject_id}: {n_new} embeddings")

    if not all_embeddings:
        sys.exit("ERROR: No embeddings extracted. Check dataset path and ONNX model.")

    embeddings = np.stack(all_embeddings).astype(np.float32)
    labels = np.array(all_labels)
    subjects = np.array(all_subjects)

    if verbose:
        print(f"\n[{substrate}] Total: {len(embeddings)} embeddings from "
              f"{len(set(subjects))} participants across {len(_SOUND_TO_CLASS)} classes")
        for cls in sorted(set(labels)):
            print(f"  {cls}: {(labels == cls).sum()}")

    return {"embeddings": embeddings, "labels": labels, "subjects": subjects}


# ---------------------------------------------------------------
# LOSO eval
# ---------------------------------------------------------------

# PCA components for the LR head. Reduces the LR problem from the full
# embedding dim (2048 for PANNs, 768 for BEATs) to a tractable size.
# Mahalanobis gate always operates on full-dim embeddings (Q34, Q52).
_PCA_N_COMPONENTS = 256


def run_loso(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    substrate: str,
    verbose: bool = True,
) -> dict:
    """Run Leave-One-Subject-Out eval. Return results dict."""
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from sklearn.decomposition import PCA
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from voxkit.classifier.classifier import Classifier
    from voxkit.core.taxonomy import TaxonomyConfig

    taxonomy = TaxonomyConfig.default_v1_0()
    dim = embeddings.shape[1]
    unique_subjects = sorted(set(subjects))
    n_folds = len(unique_subjects)

    # Cap PCA components to what the data can support
    n_components = min(_PCA_N_COMPONENTS, dim)

    if verbose:
        print(f"\n[{substrate}] Running LOSO ({n_folds} folds, PCA→{n_components}d)...")

    fold_f1s: list[float] = []
    fold_results: list[dict] = []

    for fold_idx, held_out in enumerate(unique_subjects):
        test_mask = subjects == held_out
        train_mask = ~test_mask

        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        s_train = subjects[train_mask]
        X_test = embeddings[test_mask]
        y_test = labels[test_mask]

        # StandardScaler + PCA on training split.
        # Scaling is baked into pca_matrix so the Classifier applies
        # (X - mean) / scale @ pca.T implicitly via X @ pca_matrix.T.
        # The LR intercept absorbs the centering offset; Mahalanobis still
        # operates on the original full-dim embeddings (Q34).
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(X_train_s)
        pca_matrix = pca.components_ / scaler.scale_[np.newaxis, :]  # (n_components, dim)

        clf = Classifier.untrained(taxonomy, dim)
        try:
            clf.fit(X_train, y_train, s_train, pca_matrix=pca_matrix)
        except Exception as exc:
            if verbose:
                print(f"  Fold {fold_idx+1}/{n_folds} ({held_out}): FIT FAILED — {exc}")
            continue

        # Use argmax of predict_proba to evaluate raw class discriminability.
        # clf.predict() also applies the Mahalanobis OOD gate, which is calibrated
        # for within-training spread and rejects all BEATs test samples (3x the
        # threshold) because BEATs encodes subject identity strongly. The bake-off
        # measures embedding discriminability, not OOD detection.
        probs = clf.predict_proba(X_test)
        preds = [clf._classes[int(np.argmax(row))] for row in probs]
        f1 = float(f1_score(y_test, preds, average="macro",
                             labels=list(clf._classes), zero_division=0))
        fold_f1s.append(f1)

        ordered_classes = list(clf._classes)
        per_class_arr = f1_score(
            y_test, preds, labels=ordered_classes,
            average=None, zero_division=0,
        )
        per_class = {cls: float(per_class_arr[i])
                     for i, cls in enumerate(ordered_classes)}

        fold_results.append({
            "subject": held_out,
            "macro_f1": round(f1, 4),
            "per_class_f1": {k: (round(v, 4) if v is not None else None)
                             for k, v in per_class.items()},
            "n_test": int(test_mask.sum()),
        })

        if verbose:
            pc = "  ".join(f"{k}={v:.3f}" if v is not None else f"{k}=N/A"
                           for k, v in per_class.items())
            print(f"  Fold {fold_idx+1:2d}/{n_folds} ({held_out:15s}): "
                  f"macro_F1={f1:.4f}  [{pc}]")

    if not fold_f1s:
        sys.exit("ERROR: All folds failed.")

    scores = np.array(fold_f1s)
    mean_f1 = float(scores.mean())
    std_f1 = float(scores.std())

    # Bootstrap CI (Q74)
    from voxkit.eval.substrate_bakeoff import bootstrap_ci_macro_f1
    ci_low, ci_high = bootstrap_ci_macro_f1(scores, n_resamples=1000, seed=0)

    if verbose:
        print(f"\n[{substrate}] LOSO macro-F1: {mean_f1:.4f} ± {std_f1:.4f}  "
              f"95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    return {
        "substrate": substrate,
        "n_folds": n_folds,
        "mean_macro_f1": round(mean_f1, 4),
        "std_macro_f1": round(std_f1, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "fold_f1s": [round(f, 4) for f in fold_f1s],
        "folds": fold_results,
    }


# ---------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------

def _cache_path(substrate: str) -> Path:
    return _REPO_ROOT / "data" / f"avp_embeddings_{substrate}.npz"


def _save_cache(substrate: str, data: dict) -> None:
    path = _cache_path(substrate)
    np.savez_compressed(str(path), **data)
    print(f"[cache] Saved embeddings → {path}")


def _load_cache(substrate: str) -> dict | None:
    path = _cache_path(substrate)
    if not path.exists():
        return None
    print(f"[cache] Loading embeddings from {path}")
    npz = np.load(str(path), allow_pickle=True)
    return {k: npz[k] for k in npz.files}


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AVP-LVT v4 LOSO embedding eval + substrate bake-off"
    )
    p.add_argument(
        "--substrate", choices=["panns", "beats", "both"], default="panns",
        help="Substrate to evaluate (default: panns)",
    )
    p.add_argument(
        "--data-dir", type=Path, default=_DATA_DIR,
        help=f"Path to AVP Personal modality dir (default: {_DATA_DIR})",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_REPO_ROOT / "data",
        help="Directory for result JSON files (default: data/)",
    )
    p.add_argument(
        "--use-cache", action="store_true",
        help="Load cached embeddings instead of re-extracting",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-fold output",
    )
    return p.parse_args()


def _run_one(substrate: str, args: argparse.Namespace) -> dict:
    verbose = not args.quiet

    # Embedding extraction (or cache load)
    data = None
    if args.use_cache:
        data = _load_cache(substrate)
    if data is None:
        t0 = time.time()
        data = extract_embeddings_for_substrate(
            substrate, args.data_dir, verbose=verbose
        )
        print(f"[{substrate}] Extraction took {time.time()-t0:.1f}s")
        _save_cache(substrate, data)

    # LOSO
    t0 = time.time()
    results = run_loso(
        data["embeddings"], data["labels"], data["subjects"],
        substrate=substrate, verbose=verbose,
    )
    print(f"[{substrate}] LOSO took {time.time()-t0:.1f}s")

    # Save per-substrate JSON
    out_path = args.output_dir / f"avp_loso_{substrate}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[{substrate}] Results → {out_path}")

    return results


def main() -> None:
    args = _parse_args()

    if not args.data_dir.exists():
        sys.exit(
            f"ERROR: Dataset directory not found: {args.data_dir}\n"
            f"Unzip AVP_Dataset.zip into data/avp/ first."
        )

    substrates = ["panns", "beats"] if args.substrate == "both" else [args.substrate]
    all_results: dict[str, dict] = {}

    for substrate in substrates:
        all_results[substrate] = _run_one(substrate, args)

    # Substrate bake-off (Q33) when both are run
    if len(substrates) == 2:
        from voxkit.eval.substrate_bakeoff import substrate_decision
        import sys; sys.path.insert(0, str(_REPO_ROOT / "src"))

        panns_f1s = np.array(all_results["panns"]["fold_f1s"])
        beats_f1s = np.array(all_results["beats"]["fold_f1s"])
        decision = substrate_decision(panns_f1s, beats_f1s, seed=0)

        print(f"\n{'='*60}")
        print(f"SUBSTRATE BAKE-OFF DECISION (Q33)")
        print(f"  Winner  : {decision.winner.upper()}")
        print(f"  Rationale: {decision.rationale}")
        print(f"  Tiebreaker used: {decision.tiebreaker_used}")
        print(f"{'='*60}")

        bakeoff = {
            "winner": decision.winner,
            "tiebreaker_used": decision.tiebreaker_used,
            "rationale": decision.rationale,
            "panns": all_results["panns"],
            "beats": all_results["beats"],
        }
        out_path = args.output_dir / "avp_bakeoff.json"
        out_path.write_text(json.dumps(bakeoff, indent=2))
        print(f"Bake-off results → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
