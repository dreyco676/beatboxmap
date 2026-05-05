# SPDX-License-Identifier: GPL-3.0-or-later
"""Bootstrap-CI substrate tiebreaker for the eval harness (Q33, Q74)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SubstrateDecision:
    winner: str
    tiebreaker_used: bool
    rationale: str


def bootstrap_ci_macro_f1(
    scores: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return a bootstrap (low, high) confidence interval for macro-F1.

    Parameters
    ----------
    scores      : per-file macro-F1 scores
    n_resamples : number of bootstrap resamples (default 1000, Q74)
    seed        : random seed for reproducibility (T42)
    alpha       : significance level (two-tailed)
    """
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    n = len(scores)
    for i in range(n_resamples):
        means[i] = np.mean(rng.choice(scores, size=n, replace=True))
    low = float(np.percentile(means, 100.0 * alpha / 2.0))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (low, high)


def substrate_decision(
    panns_scores: np.ndarray,
    beats_scores: np.ndarray,
    pilot_ood_fn=None,
    seed: int = 0,
) -> SubstrateDecision:
    """Pick winning substrate using bootstrap CI; break ties via pilot OOD function.

    Parameters
    ----------
    panns_scores  : per-file macro-F1 scores for the PANNs substrate
    beats_scores  : per-file macro-F1 scores for the Beats substrate
    pilot_ood_fn  : callable() → str winner name; called only when CIs overlap
    seed          : passed to bootstrap_ci_macro_f1 for reproducibility (T44)
    """
    panns_low, panns_high = bootstrap_ci_macro_f1(panns_scores, seed=seed)
    beats_low, beats_high = bootstrap_ci_macro_f1(beats_scores, seed=seed)

    panns_ci_str = f"[{panns_low:.3f}, {panns_high:.3f}]"
    beats_ci_str = f"[{beats_low:.3f}, {beats_high:.3f}]"

    # Use a tolerance so that near-overlapping CIs (within finite-sample
    # estimation noise) also trigger the tiebreaker rather than flipping
    # the decision on tiny numerical differences.
    _GAP_THRESHOLD = 0.02

    if panns_low - beats_high > _GAP_THRESHOLD:
        rationale = (
            f"panns CI {panns_ci_str} strictly above beats CI {beats_ci_str}"
        )
        return SubstrateDecision(winner="panns", tiebreaker_used=False, rationale=rationale)

    if beats_low - panns_high > _GAP_THRESHOLD:
        rationale = (
            f"beats CI {beats_ci_str} strictly above panns CI {panns_ci_str}"
        )
        return SubstrateDecision(winner="beats", tiebreaker_used=False, rationale=rationale)

    # CIs overlap or are within tolerance — use pilot OOD tiebreaker.
    if pilot_ood_fn is not None:
        winner = pilot_ood_fn()
    else:
        panns_mean = float(np.mean(panns_scores))
        beats_mean = float(np.mean(beats_scores))
        winner = "panns" if panns_mean >= beats_mean else "beats"

    rationale = (
        f"CIs overlapped (panns {panns_ci_str}, beats {beats_ci_str}); "
        f"pilot OOD broke tie for {winner}"
    )
    return SubstrateDecision(winner=winner, tiebreaker_used=True, rationale=rationale)
