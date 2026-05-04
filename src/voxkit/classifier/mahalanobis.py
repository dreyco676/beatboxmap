# SPDX-License-Identifier: GPL-3.0-or-later
"""Mahalanobis distance via Cholesky factor (Q34, Q68)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def mahalanobis_sq_via_cholesky(
    x: np.ndarray,
    centroid: np.ndarray,
    L: np.ndarray,
) -> float:
    """Squared Mahalanobis distance using lower-triangular Cholesky L.

    d²(x, μ) = (x - μ)ᵀ Σ⁻¹ (x - μ)  where  Σ = L Lᵀ
    """
    diff = x - centroid
    y = solve_triangular(L, diff, lower=True)
    return float(y @ y)
