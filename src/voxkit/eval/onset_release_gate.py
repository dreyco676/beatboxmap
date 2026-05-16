# SPDX-License-Identifier: GPL-3.0-or-later
"""Onset detection release-gate evaluator (Q70, §7.8)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReleaseGateResult:
    passed: bool
    f: float
    mae_ms: float
    dataset: str


_THRESHOLDS: dict[str, dict] = {
    "AVP": {"f_min": 0.92, "mae_ms_max": 15.0},
    "OOD": {"f_min": 0.88, "mae_ms_max": 25.0},
}


def _match_onsets(
    pred: list[float],
    ref: list[float],
    tolerance_s: float,
) -> tuple[int, int, int, float, int]:
    """Greedy match detections to references within tolerance_s seconds.

    Returns (tp, fp, fn, mae_sum_ms, n_matched).
    """
    pred_s = sorted(pred)
    ref_s = sorted(ref)
    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    mae_sum_ms = 0.0

    for j, r in enumerate(ref_s):
        best_i = None
        best_dist = tolerance_s
        for i, p in enumerate(pred_s):
            if i in matched_pred:
                continue
            dist = abs(p - r)
            if dist <= best_dist:
                best_dist = dist
                best_i = i
        if best_i is not None:
            matched_pred.add(best_i)
            matched_ref.add(j)
            mae_sum_ms += best_dist * 1000.0

    tp = len(matched_ref)
    fp = len(pred_s) - len(matched_pred)
    fn = len(ref_s) - len(matched_ref)
    return tp, fp, fn, mae_sum_ms, tp


def evaluate_corpus(
    detector,
    ground_truth: list[tuple[np.ndarray, list[float]]],
    iou_ms: float = 50.0,
) -> tuple[float, float]:
    """Evaluate onset detection over a corpus.

    Parameters
    ----------
    detector     : object with .detect(audio) → list[float] onset times
    ground_truth : list of (audio, reference_onset_times) pairs
    iou_ms       : match tolerance in milliseconds

    Returns
    -------
    (f_measure, mae_ms)
    """
    tolerance_s = iou_ms / 1000.0
    tp_total = fp_total = fn_total = 0
    mae_sum = 0.0
    n_matched = 0

    for audio, ref_times in ground_truth:
        pred_times = detector.detect(audio)
        tp, fp, fn, mae_ms, matched = _match_onsets(pred_times, ref_times, tolerance_s)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        mae_sum += mae_ms
        n_matched += matched

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mae = mae_sum / n_matched if n_matched > 0 else 0.0
    return (f, mae)


def release_gate_check(f: float, mae_ms: float, dataset: str) -> ReleaseGateResult:
    """Check onset metrics against release-gate thresholds for the given dataset."""
    thresh = _THRESHOLDS.get(dataset, {"f_min": 0.90, "mae_ms_max": 30.0})
    passed = f >= thresh["f_min"] and mae_ms <= thresh["mae_ms_max"]
    return ReleaseGateResult(passed=passed, f=f, mae_ms=mae_ms, dataset=dataset)


def release_gate_main(args: list[str]) -> int:
    """CLI entry point. Parses args, checks gate, writes JSON, returns exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Onset release-gate check")
    parser.add_argument("--avp-f", type=float, required=True)
    parser.add_argument("--avp-mae-ms", type=float, required=True)
    parser.add_argument("--ood-f", type=float, required=True)
    parser.add_argument("--ood-mae-ms", type=float, required=True)
    parser.add_argument("--output", type=str, default=None)
    parsed = parser.parse_args(args)

    avp = release_gate_check(parsed.avp_f, parsed.avp_mae_ms, "AVP")
    ood = release_gate_check(parsed.ood_f, parsed.ood_mae_ms, "OOD")

    output = {
        "avp": {"f": avp.f, "mae_ms": avp.mae_ms, "passed": avp.passed},
        "ood": {"f": ood.f, "mae_ms": ood.mae_ms, "passed": ood.passed},
    }

    if parsed.output:
        Path(parsed.output).write_text(json.dumps(output, indent=2, sort_keys=True))

    return 0 if (avp.passed and ood.passed) else 1
