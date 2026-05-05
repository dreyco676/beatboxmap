# SPDX-License-Identifier: GPL-3.0-or-later
"""Onset evaluation metrics: F-measure, alignment MAE (Component 4 stub)."""

from __future__ import annotations


def f_measure(
    predicted: list[float],
    reference: list[float],
    iou_ms: float = 50.0,
) -> float:
    """Standard onset F-measure with tolerance window iou_ms milliseconds."""
    tol = iou_ms / 1000.0
    if not predicted or not reference:
        return 0.0

    used_ref: set[int] = set()
    tp = 0
    for p in sorted(predicted):
        for i, r in enumerate(sorted(reference)):
            if i not in used_ref and abs(p - r) <= tol:
                tp += 1
                used_ref.add(i)
                break

    precision = tp / len(predicted)
    recall = tp / len(reference)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
