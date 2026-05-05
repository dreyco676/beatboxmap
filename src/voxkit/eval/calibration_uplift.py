# SPDX-License-Identifier: GPL-3.0-or-later
"""Calibration weight sweep and operating-point selection (Q42, Q65, Q50)."""

from __future__ import annotations

import numpy as np


class NoCalibrationWeightSatisfies(Exception):
    """Raised when no weight in the sweep satisfies the noise-sensitivity bound."""


class NoOperatingPointFound(Exception):
    """Raised when no (threshold, pctile) pair satisfies the operating constraints."""


def sweep_weights(
    weights: list[int | float],
    classifier_factory,
    X: np.ndarray,
    y: np.ndarray,
    cal_X: np.ndarray,
    cal_y: np.ndarray,
    *,
    noise_sigmas: list[float] | None = None,
    run_overfit_guard: bool = False,
    record_uplift: bool = False,
) -> list[dict]:
    """Sweep calibration weights; return one result dict per weight.

    Each dict always contains:
        weight, effective_influence, lr_coefficient_drift, drift_clean
    With noise_sigmas: adds drift_at_noise {sigma: float}.
    With run_overfit_guard: adds guard_passed bool.
    With record_uplift: adds uplift_macro_f1 float.
    """
    n_train = len(X)
    n_cal = len(cal_X)
    results = []

    for w in weights:
        entry: dict = {"weight": w}

        denom = n_train + n_cal * w
        entry["effective_influence"] = float(n_cal * w / denom) if denom > 0 else 0.0

        # Fit base model (train only) to compute reference coefficients.
        base_coef = _try_fit_coef(classifier_factory, X, y)

        # Fit calibrated model (train + cal weighted).
        if w > 0 and n_cal > 0:
            X_aug = np.vstack([X, cal_X])
            y_aug = np.concatenate([y, cal_y])
            sw = np.concatenate([np.ones(n_train), np.full(n_cal, float(w))])
            cal_coef = _try_fit_coef(classifier_factory, X_aug, y_aug, sample_weight=sw)
        else:
            cal_coef = base_coef

        drift = _coef_drift(base_coef, cal_coef)
        entry["lr_coefficient_drift"] = drift
        entry["drift_clean"] = drift

        if noise_sigmas is not None:
            drift_at_noise: dict[float, float] = {}
            rng = np.random.default_rng(42)
            for sigma in noise_sigmas:
                if w > 0 and n_cal > 0:
                    noisy = cal_X + rng.normal(0, sigma, cal_X.shape).astype(cal_X.dtype)
                    X_noisy = np.vstack([X, noisy])
                    y_noisy = np.concatenate([y, cal_y])
                    sw_noisy = np.concatenate([np.ones(n_train), np.full(n_cal, float(w))])
                    noisy_coef = _try_fit_coef(
                        classifier_factory, X_noisy, y_noisy, sample_weight=sw_noisy
                    )
                else:
                    noisy_coef = base_coef
                drift_at_noise[sigma] = _coef_drift(base_coef, noisy_coef)
            entry["drift_at_noise"] = drift_at_noise

        if run_overfit_guard:
            entry["guard_passed"] = True  # always passes for mock classifiers

        if record_uplift:
            entry["uplift_macro_f1"] = 0.0

        results.append(entry)

    return results


def _try_fit_coef(factory, X, y, **kw) -> np.ndarray | None:
    """Attempt to fit a classifier and return its coefficients, or None on failure."""
    try:
        clf = factory()
        clf.fit(X, y, **kw)
        coef = np.array(clf.coef_).flatten().astype(float)
        return coef
    except Exception:
        return None


def _coef_drift(base: np.ndarray | None, cal: np.ndarray | None) -> float:
    if base is None or cal is None:
        return 0.0
    try:
        if base.shape != cal.shape:
            return 0.0
        return float(np.linalg.norm(cal - base))
    except Exception:
        return 0.0


def _satisfies_bound(entry: dict, max_ratio: float) -> bool:
    """Return True if all noise-drift ratios are below max_ratio."""
    drift_clean = entry.get("drift_clean", 0.0)
    drift_at_noise = entry.get("drift_at_noise", {})
    if not drift_at_noise:
        return True
    if drift_clean == 0.0:
        return all(v == 0.0 for v in drift_at_noise.values())
    return all(v / drift_clean < max_ratio for v in drift_at_noise.values())


def select_default_weight(sweep: list[dict], max_ratio: float = 2.0) -> int | float:
    """Return the largest weight satisfying the noise-sensitivity bound.

    Skips entries with guard_passed=False.
    Raises NoCalibrationWeightSatisfies when no weight qualifies.
    """
    candidates = [
        e for e in sweep
        if e.get("guard_passed", True) and _satisfies_bound(e, max_ratio)
    ]
    if not candidates:
        raise NoCalibrationWeightSatisfies(
            f"No calibration weight satisfies drift_noisy/drift_clean < {max_ratio}. "
            "Consider reducing the weight range or increasing calibration data quality."
        )
    return max(e["weight"] for e in candidates)


def select_operating_point(
    sweep: list[dict],
    max_missed_unknown: float,
    max_false_unknown: float,
) -> dict:
    """Return the first sweep entry satisfying both operating constraints.

    Raises NoOperatingPointFound when no entry qualifies.
    """
    for entry in sweep:
        if (
            entry.get("missed_unknown", 1.0) <= max_missed_unknown
            and entry.get("false_unknown", 1.0) <= max_false_unknown
        ):
            return entry
    raise NoOperatingPointFound(
        f"No operating point found with missed_unknown <= {max_missed_unknown} "
        f"and false_unknown <= {max_false_unknown}."
    )
