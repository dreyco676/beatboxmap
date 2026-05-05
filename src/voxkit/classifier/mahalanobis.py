# SPDX-License-Identifier: GPL-3.0-or-later
"""Mahalanobis distance via Cholesky factor (Q34, Q68)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf


def fit_mahalanobis_full_dim(
    avp_embeddings: np.ndarray,
    avp_labels: np.ndarray,
    calibration_embeddings: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_weight: float,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit pooled Mahalanobis parameters from AVP (+ optional calibration) data.

    Returns
    -------
    centroids : (n_classes, D) weighted centroids (AVP + cal)
    L         : (D, D) lower-triangular Cholesky of pooled AVP covariance
    thresholds: (n_classes,) per-class p95 Mahalanobis distance (AVP only)
    """
    D = avp_embeddings.shape[1]
    n_classes = len(classes)

    # AVP-only centroids (for covariance and thresholds, Q52)
    avp_centroids = np.zeros((n_classes, D), dtype=np.float64)
    for i, c in enumerate(classes):
        mask = avp_labels == c
        avp_centroids[i] = avp_embeddings[mask].astype(np.float64).mean(axis=0)

    # Weighted centroids (for predict)
    weighted_centroids = np.zeros((n_classes, D), dtype=np.float64)
    for i, c in enumerate(classes):
        avp_mask = avp_labels == c
        avp_vecs = avp_embeddings[avp_mask].astype(np.float64)
        n_avp = len(avp_vecs)
        avp_sum = avp_vecs.sum(axis=0)
        cal_sum = np.zeros(D, dtype=np.float64)
        n_cal_eff = 0.0
        if len(calibration_labels) > 0:
            cal_mask = calibration_labels == c
            cal_vecs = calibration_embeddings[cal_mask].astype(np.float64)
            cal_sum = cal_vecs.sum(axis=0)
            n_cal_eff = calibration_weight * len(cal_vecs)
        denom = n_avp + n_cal_eff
        if denom == 0:
            weighted_centroids[i] = avp_centroids[i]
        else:
            weighted_centroids[i] = (avp_sum + calibration_weight * cal_sum) / denom

    # Pooled covariance from AVP deviations from AVP centroids (Q52)
    deviations = []
    for i, c in enumerate(classes):
        mask = avp_labels == c
        vecs = avp_embeddings[mask].astype(np.float64)
        deviations.append(vecs - avp_centroids[i])
    all_devs = np.vstack(deviations)  # (N_avp, D)

    N = len(all_devs)
    if N <= D:
        lw = LedoitWolf(assume_centered=True)
        lw.fit(all_devs)
        cov = lw.covariance_
    else:
        cov = (all_devs.T @ all_devs) / max(N - 1, 1)
        # Light diagonal regularization for numerical safety
        cov += 1e-6 * np.eye(D)

    L = np.linalg.cholesky(cov)

    # Per-class p95 distance thresholds using AVP centroids + pooled L (Q52)
    thresholds = np.zeros(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = avp_labels == c
        vecs = avp_embeddings[mask].astype(np.float64)
        dists = np.array([
            np.sqrt(mahalanobis_sq_via_cholesky(v, avp_centroids[i], L))
            for v in vecs
        ])
        thresholds[i] = float(np.percentile(dists, 95))

    return (
        weighted_centroids.astype(np.float64),
        L.astype(np.float64),
        thresholds.astype(np.float64),
    )


def mahalanobis_sq_via_cholesky(
    x: np.ndarray,
    centroid: np.ndarray,
    L: np.ndarray,
) -> float:
    """Squared Mahalanobis distance using lower-triangular Cholesky L.

    d²(x, μ) = (x - μ)ᵀ Σ⁻¹ (x - μ)  where  Σ = L Lᵀ
    """
    if not np.allclose(L, np.tril(L)):
        raise ValueError("L must be lower-triangular")
    diff = x - centroid
    y = solve_triangular(L, diff, lower=True)
    return float(y @ y)
