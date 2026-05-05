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
