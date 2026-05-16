# SPDX-License-Identifier: GPL-3.0-or-later
"""Onset evaluation metrics: F-measure, alignment MAE, release gate (Component 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------
# Internal: greedy nearest-neighbour pairing (Tidy First before T14)
# ---------------------------------------------------------------

def _align_pairs(
    detected: list[float],
    reference: list[float],
    tol_s: float,
) -> list[tuple[float, float]]:
    """Greedy nearest-neighbour pairing within tolerance.

    Returns list of (detected_time, reference_time) matched pairs.
    Each reference point is used at most once.
    """
    used_ref: set[int] = set()
    pairs: list[tuple[float, float]] = []
    for d in sorted(detected):
        best_i = None
        best_dist = tol_s + 1.0
        for i, r in enumerate(sorted(reference)):
            if i not in used_ref and abs(d - r) < best_dist:
                best_dist = abs(d - r)
                best_i = i
        if best_i is not None and best_dist <= tol_s:
            pairs.append((d, sorted(reference)[best_i]))
            used_ref.add(best_i)
    return pairs


# ---------------------------------------------------------------
# F-measure
# ---------------------------------------------------------------

def f_measure(
    detected: list[float],
    reference: list[float],
    iou_ms: float = 50.0,
) -> float:
    """Standard onset F-measure with tolerance window iou_ms milliseconds.

    Both empty → 1.0 (vacuous: no FPs, no FNs).
    One side empty → 0.0.
    """
    if not detected and not reference:
        return 1.0
    if not detected or not reference:
        return 0.0

    tol = iou_ms / 1000.0
    pairs = _align_pairs(detected, reference, tol)
    tp = len(pairs)
    precision = tp / len(detected)
    recall = tp / len(reference)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------
# Alignment MAE
# ---------------------------------------------------------------

def alignment_mae(
    detected: list[float],
    reference: list[float],
    iou_ms: float = 50.0,
) -> float:
    """Median absolute timing error over matched (true-positive) pairs, in ms.

    Ignores unmatched detections and unmatched reference events.
    Returns 0.0 if there are no matched pairs.
    """
    tol = iou_ms / 1000.0
    pairs = _align_pairs(detected, reference, tol)
    if not pairs:
        return 0.0
    errors_ms = [abs(d - r) * 1000.0 for d, r in pairs]
    return float(np.median(errors_ms))


# ---------------------------------------------------------------
# Release gate (Q70, §7.8)
# ---------------------------------------------------------------

_THRESHOLDS = {
    "AVP": {"f_min": 0.92, "mae_max_ms": 15.0},
    "OOD": {"f_min": 0.88, "mae_max_ms": 25.0},
}


@dataclass
class ReleaseGateResult:
    passed: bool
    failed_tier: str   # "" if passed; "detection" or "alignment" if failed


def _load_avp_corpus() -> list[tuple[np.ndarray, list[float]]]:
    """Load AVP Personal corpus from the standard project data path."""
    import csv
    import scipy.io.wavfile as wv
    from pathlib import Path as _Path
    from math import gcd

    repo_root = _Path(__file__).parent.parent.parent.parent
    personal_dir = repo_root / "data" / "avp" / "AVP_Dataset" / "Personal"
    if not personal_dir.exists():
        raise FileNotFoundError(f"AVP dataset not found at {personal_dir}")

    pairs: list[tuple[np.ndarray, list[float]]] = []
    for pdir in sorted(
        (d for d in personal_dir.iterdir() if d.is_dir() and d.name.startswith("Participant_")),
        key=lambda d: int(d.name.split("_")[1]),
    ):
        for wav_path in sorted(pdir.glob("*.wav")):
            if "Improvisation" in wav_path.stem:
                continue
            csv_path = wav_path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            sr, data = wv.read(str(wav_path))
            if data.ndim == 2:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype != np.float32:
                data = data.astype(np.float32)
            if sr != 16_000:
                from scipy.signal import resample_poly
                g = gcd(sr, 16_000)
                data = resample_poly(data, 16_000 // g, sr // g).astype(np.float32)
            onsets: list[float] = []
            with open(csv_path, newline="", encoding="utf-8") as f_csv:
                for row in csv.reader(f_csv):
                    if row:
                        try:
                            onsets.append(float(row[0]))
                        except ValueError:
                            pass
            pairs.append((data, onsets))
    return pairs


_NAMED_CORPORA = {"AVP": _load_avp_corpus}


def evaluate_corpus(
    detector,
    corpus,
    iou_ms: float = 50.0,
) -> tuple[float, float]:
    """Evaluate onset detection over a corpus.

    Parameters
    ----------
    detector : object with .detect(audio) -> list[float]
    corpus   : either a named corpus string (e.g. "AVP") or a list of
               (audio_float32, reference_onset_times_s) pairs
    iou_ms   : match tolerance in milliseconds

    Returns
    -------
    (f_measure, mae_ms) computed across the whole corpus
    """
    if isinstance(corpus, str):
        if corpus not in _NAMED_CORPORA:
            raise ValueError(f"Unknown named corpus {corpus!r}; known: {list(_NAMED_CORPORA)}")
        corpus = _NAMED_CORPORA[corpus]()

    tol = iou_ms / 1000.0
    all_f: list[float] = []
    mae_sum = 0.0
    n_matched = 0

    for audio, ref_times in corpus:
        pred_times = detector.detect(audio)
        pairs = _align_pairs(pred_times, ref_times, tol)
        tp = len(pairs)
        precision = tp / len(pred_times) if pred_times else 0.0
        recall = tp / len(ref_times) if ref_times else 0.0
        if precision + recall > 0:
            all_f.append(2.0 * precision * recall / (precision + recall))
        else:
            all_f.append(0.0)
        mae_sum += sum(abs(d - r) * 1000.0 for d, r in pairs)
        n_matched += tp

    mean_f = float(np.mean(all_f)) if all_f else 0.0
    mean_mae = mae_sum / n_matched if n_matched > 0 else 0.0
    return mean_f, mean_mae


def release_gate(f: float, mae_ms: float, dataset: str) -> ReleaseGateResult:
    """Two-tier release gate (Q70).

    Returns ReleaseGateResult with .passed and .failed_tier.
    """
    thresholds = _THRESHOLDS[dataset]
    if f < thresholds["f_min"]:
        return ReleaseGateResult(passed=False, failed_tier="detection")
    if mae_ms > thresholds["mae_max_ms"]:
        return ReleaseGateResult(passed=False, failed_tier="alignment")
    return ReleaseGateResult(passed=True, failed_tier="")
