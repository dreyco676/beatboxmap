# SPDX-License-Identifier: GPL-3.0-or-later
"""Temperature scaling, overfit guard, and operating-point selection (Q26, Q50, Q71, Q75)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class NoOperatingPointFound(Exception):
    pass


# ---------------------------------------------------------------
# Numerically-stable softmax with temperature
# ---------------------------------------------------------------

def softmax_with_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Row-wise softmax(logits / T) with log-sum-exp stability."""
    scaled = logits / T
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------
# Temperature fitting (Q75)
# ---------------------------------------------------------------

def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    indices: list[int] | None = None,
    T_min: float = 0.1,
    T_max: float = 10.0,
) -> float:
    """Fit scalar temperature T by minimising NLL on (logits, labels).

    Parameters
    ----------
    logits  : (N, C) raw logits from the LR head
    labels  : (N,) integer class indices
    indices : original sample indices (passed through for T45 spying)
    """
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_to_idx[l] for l in labels])

    def nll(T):
        probs = softmax_with_temperature(logits, T)
        probs = np.clip(probs, 1e-15, 1.0)
        return -float(np.mean(np.log(probs[np.arange(len(y_int)), y_int])))

    result = minimize_scalar(nll, bounds=(T_min, T_max), method="bounded")
    T = float(np.clip(result.x, T_min, T_max))
    return T


# ---------------------------------------------------------------
# LR head fitting (Q26)
# ---------------------------------------------------------------

def fit_lr_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    indices: list[int] | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a logistic regression head; return (coefficients, intercepts).

    Selects C ∈ {10, 1, 0.1, 0.01} via 3-fold CV (accuracy). When `groups`
    (subject IDs) are provided and there are ≥ 3 unique groups, inner CV folds
    are subject-disjoint (GroupKFold), so C is selected for cross-subject
    generalisation rather than within-subject performance. Cs are tried
    largest-first so ties break in favour of less regularisation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score

    n_unique_groups = len(set(groups)) if groups is not None else 0
    cv = GroupKFold(n_splits=3) if n_unique_groups >= 3 else 3

    best_c, best_score = 10.0, -1.0
    for C in [10.0, 1.0, 0.1, 0.01]:
        lr = LogisticRegression(
            C=C, solver="saga", penalty="l2", max_iter=5000, random_state=0
        )
        scores = cross_val_score(
            lr, X_train, y_train, cv=cv, scoring="accuracy",
            groups=groups if n_unique_groups >= 3 else None,
        )
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_c = C

    lr_final = LogisticRegression(
        C=best_c, solver="saga", penalty="l2", max_iter=5000, random_state=0
    )
    lr_final.fit(X_train, y_train)
    return lr_final.coef_, lr_final.intercept_


# ---------------------------------------------------------------
# Self-test overfit guard (Q71)
# ---------------------------------------------------------------

def self_test_overfit_guard(
    f1_calibrated: float,
    f1_baseline: float,
) -> tuple[bool, dict[str, Any]]:
    """Return (passed, diagnostics). Passes if drop ≤ 0.01."""
    delta = f1_calibrated - f1_baseline
    passed = delta >= -0.01 - 1e-9
    return passed, {
        "f1_calibrated": f1_calibrated,
        "f1_baseline": f1_baseline,
        "delta": delta,
    }


# ---------------------------------------------------------------
# Operating-point selection (Q50, §7.3)
# ---------------------------------------------------------------

def select_operating_point(
    sweep: list[dict],
    max_missed_unknown: float,
    max_false_unknown: float,
) -> dict:
    """Return first sweep entry satisfying both Q50 bounds, else raise."""
    for entry in sweep:
        if (entry["missed_unknown"] <= max_missed_unknown and
                entry["false_unknown"] <= max_false_unknown):
            return entry
    raise NoOperatingPointFound(
        f"No operating point satisfies missed_unknown≤{max_missed_unknown} "
        f"and false_unknown≤{max_false_unknown}"
    )
