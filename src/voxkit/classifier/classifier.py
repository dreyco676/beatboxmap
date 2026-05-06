# SPDX-License-Identifier: GPL-3.0-or-later
"""Classifier: composite gate (LR head + Mahalanobis OOD) (Q26, Q34, Q66, Q81)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from voxkit.classifier.calibration import (
    fit_lr_head,
    fit_temperature,
    self_test_overfit_guard,
    softmax_with_temperature,
)
from voxkit.classifier.mahalanobis import (
    fit_mahalanobis_full_dim,
    mahalanobis_sq_via_cholesky,
)
from voxkit.core.taxonomy import TaxonomyConfig


# ---------------------------------------------------------------
# Q81 dialog text
# ---------------------------------------------------------------

Q81_DIALOG_TEXT = (
    "Calibration didn't improve classification. "
    "The previous calibration has been restored. "
    "Try recording more or quieter samples for better results."
)


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class NotFittedError(Exception):
    pass


class CalibrationRejected(Exception):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


# ---------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------

class Classifier:
    def __init__(self, taxonomy: TaxonomyConfig, embedding_dim: int) -> None:
        self.taxonomy = taxonomy
        self.embedding_dim = embedding_dim
        self._fitted = False

        # Set after fit
        self.lr_coefficients_: np.ndarray | None = None
        self._lr_intercepts: np.ndarray | None = None
        self.T: float = 1.0
        self.class_centroids_full_dim: np.ndarray | None = None
        self._avp_centroids: np.ndarray | None = None
        self.pooled_cov_cholesky_full_dim: np.ndarray | None = None
        self.distance_thresholds: np.ndarray | None = None
        self.softmax_threshold: float = 0.5
        self._pca_matrix: np.ndarray | None = None
        self._classes: list[str] | None = None
        # Stored AVP data for recalibration without re-providing AVP
        self._stored_avp_X: np.ndarray | None = None
        self._stored_avp_y: np.ndarray | None = None
        self._stored_avp_subjects: np.ndarray | None = None

    @classmethod
    def untrained(cls, taxonomy: TaxonomyConfig | None, embedding_dim: int) -> "Classifier":
        if taxonomy is None:
            taxonomy = TaxonomyConfig.default_v1_0()
        return cls(taxonomy=taxonomy, embedding_dim=embedding_dim)

    # ------------------------------------------------------------------
    # Fold splitting (subject-based LOSO)
    # ------------------------------------------------------------------

    def _split_subjects(
        self, subjects: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lr_mask, t_mask) — one held-out subject group for T-fit."""
        unique_subjects = sorted(set(subjects))
        # Hold out the last subject for temperature fitting
        holdout_subject = unique_subjects[-1]
        t_mask = subjects == holdout_subject
        lr_mask = ~t_mask
        return lr_mask, t_mask

    # ------------------------------------------------------------------
    # Core fit
    # ------------------------------------------------------------------

    def _do_fit(
        self,
        avp_embeddings: np.ndarray,
        avp_labels: np.ndarray,
        avp_subjects: np.ndarray,
        calibration_embeddings: np.ndarray,
        calibration_labels: np.ndarray,
        calibration_weight: float,
        pca_matrix: np.ndarray | None,
    ) -> None:
        classes = sorted(set(avp_labels))
        self._classes = classes

        # Full-dim Mahalanobis (always on original embeddings, Q34)
        centroids, L, thresholds = fit_mahalanobis_full_dim(
            avp_embeddings=avp_embeddings,
            avp_labels=avp_labels,
            calibration_embeddings=calibration_embeddings,
            calibration_labels=calibration_labels,
            calibration_weight=calibration_weight,
            classes=classes,
        )
        self.class_centroids_full_dim = centroids
        # AVP-only centroids: used for Mahalanobis gate (thresholds were fit on these)
        D = avp_embeddings.shape[1]
        avp_centroids = np.zeros((len(classes), D), dtype=np.float64)
        for i, c in enumerate(classes):
            mask = avp_labels == c
            avp_centroids[i] = avp_embeddings[mask].astype(np.float64).mean(axis=0)
        self._avp_centroids = avp_centroids
        self.pooled_cov_cholesky_full_dim = L
        self.distance_thresholds = thresholds

        # Subject-based LOSO split for LR / T-fit
        lr_mask, t_mask = self._split_subjects(avp_subjects)
        lr_idx = np.where(lr_mask)[0].tolist()
        t_idx = np.where(t_mask)[0].tolist()

        X_lr = avp_embeddings[lr_mask]
        y_lr = avp_labels[lr_mask]
        s_lr = avp_subjects[lr_mask]
        X_t = avp_embeddings[t_mask]
        y_t = avp_labels[t_mask]

        # PCA projection for LR head only
        self._pca_matrix = pca_matrix
        if pca_matrix is not None:
            X_lr_head = X_lr @ pca_matrix.T
            X_t_head = X_t @ pca_matrix.T
        else:
            X_lr_head = X_lr
            X_t_head = X_t

        # LR head — pass subject IDs so inner CV uses subject-disjoint folds
        coef, intercept = fit_lr_head(X_lr_head, y_lr, indices=lr_idx, groups=s_lr)
        self.lr_coefficients_ = coef
        self._lr_intercepts = intercept

        # Temperature fitting on held-out subject
        logits_t = X_t_head @ coef.T + intercept
        self.T = float(np.clip(
            fit_temperature(logits_t, y_t, indices=t_idx),
            0.1, 10.0,
        ))
        self._fitted = True

    def _store_avp(
        self,
        avp_embeddings: np.ndarray,
        avp_labels: np.ndarray,
        avp_subjects: np.ndarray,
    ) -> None:
        self._stored_avp_X = avp_embeddings
        self._stored_avp_y = avp_labels
        self._stored_avp_subjects = avp_subjects

    def fit(
        self,
        avp_embeddings: np.ndarray,
        avp_labels: np.ndarray,
        avp_subjects: np.ndarray,
        pca_matrix: np.ndarray | None = None,
    ) -> None:
        D = avp_embeddings.shape[1]
        self._store_avp(avp_embeddings, avp_labels, avp_subjects)
        self._do_fit(
            avp_embeddings=avp_embeddings,
            avp_labels=avp_labels,
            avp_subjects=avp_subjects,
            calibration_embeddings=np.zeros((0, D), dtype=avp_embeddings.dtype),
            calibration_labels=np.array([]),
            calibration_weight=0.0,
            pca_matrix=pca_matrix,
        )

    def fit_with_calibration(
        self,
        avp_embeddings: np.ndarray | None = None,
        avp_labels: np.ndarray | None = None,
        avp_subjects: np.ndarray | None = None,
        calibration_embeddings: np.ndarray | None = None,
        calibration_labels: np.ndarray | None = None,
        calibration_weight: float = 1.0,
        pca_matrix: np.ndarray | None = None,
    ) -> None:
        if avp_embeddings is None:
            avp_embeddings = self._stored_avp_X
            avp_labels = self._stored_avp_y
            avp_subjects = self._stored_avp_subjects
        # Snapshot pre-calibration state via a plain fit
        snapshot = Classifier.untrained(self.taxonomy, self.embedding_dim)
        snapshot._do_fit(
            avp_embeddings=avp_embeddings,
            avp_labels=avp_labels,
            avp_subjects=avp_subjects,
            calibration_embeddings=np.zeros(
                (0, avp_embeddings.shape[1]), dtype=avp_embeddings.dtype
            ),
            calibration_labels=np.array([]),
            calibration_weight=0.0,
            pca_matrix=pca_matrix,
        )
        baseline_f1 = self._estimate_f1(snapshot, avp_embeddings, avp_labels)

        self._do_fit(
            avp_embeddings=avp_embeddings,
            avp_labels=avp_labels,
            avp_subjects=avp_subjects,
            calibration_embeddings=calibration_embeddings,
            calibration_labels=calibration_labels,
            calibration_weight=calibration_weight,
            pca_matrix=pca_matrix,
        )
        calibrated_f1 = self._estimate_f1(self, avp_embeddings, avp_labels)

        passed, diagnostics = self_test_overfit_guard(calibrated_f1, baseline_f1)
        if not passed:
            raise CalibrationRejected(Q81_DIALOG_TEXT, diagnostics)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _compute_logits(self, X: np.ndarray) -> np.ndarray:
        if self._pca_matrix is not None:
            head_input = X @ self._pca_matrix.T
        else:
            head_input = X
        return head_input @ self.lr_coefficients_.T + self._lr_intercepts

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("Call fit() before predict_proba()")
        logits = self._compute_logits(X)
        return softmax_with_temperature(logits, self.T)

    def predict(self, X: np.ndarray) -> list[tuple[str, float]]:
        if not self._fitted:
            raise NotFittedError("Call fit() before predict()")
        probs = self.predict_proba(X)
        results = []
        for i, row in enumerate(X):
            prob_row = probs[i]
            max_prob = float(prob_row.max())
            pred_idx = int(prob_row.argmax())

            # Softmax gate
            if max_prob < self.softmax_threshold:
                results.append((self.taxonomy.unknown_class_id, max_prob))
                continue

            # Mahalanobis gate
            pred_class = self._classes[pred_idx]
            d_sq = mahalanobis_sq_via_cholesky(
                row.astype(np.float64),
                self._avp_centroids[pred_idx],
                self.pooled_cov_cholesky_full_dim,
            )
            d = float(np.sqrt(d_sq))
            if d > self.distance_thresholds[pred_idx]:
                results.append((self.taxonomy.unknown_class_id, max_prob))
                continue

            results.append((pred_class, max_prob))
        return results

    # ------------------------------------------------------------------
    # Snapshot / restore (CalibrationManager rollback)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        def _copy(v):
            return v.copy() if isinstance(v, np.ndarray) else v
        return {
            "_fitted": self._fitted,
            "lr_coefficients_": _copy(self.lr_coefficients_),
            "_lr_intercepts": _copy(self._lr_intercepts),
            "T": self.T,
            "class_centroids_full_dim": _copy(self.class_centroids_full_dim),
            "_avp_centroids": _copy(self._avp_centroids),
            "pooled_cov_cholesky_full_dim": _copy(self.pooled_cov_cholesky_full_dim),
            "distance_thresholds": _copy(self.distance_thresholds),
            "softmax_threshold": self.softmax_threshold,
            "_pca_matrix": _copy(self._pca_matrix),
            "_classes": list(self._classes) if self._classes is not None else None,
            "_stored_avp_X": _copy(self._stored_avp_X),
            "_stored_avp_y": _copy(self._stored_avp_y),
            "_stored_avp_subjects": _copy(self._stored_avp_subjects),
        }

    def restore(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Distribution-shift threshold (Q45)
    # ------------------------------------------------------------------

    def get_distribution_shift_threshold(self) -> float:
        return float(np.mean(self.distance_thresholds))

    # ------------------------------------------------------------------
    # Serialization (T27, T47)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        state = {
            "taxonomy": self.taxonomy,
            "embedding_dim": self.embedding_dim,
            "lr_coefficients_": self.lr_coefficients_,
            "_lr_intercepts": self._lr_intercepts,
            "T": self.T,
            "class_centroids_full_dim": self.class_centroids_full_dim,
            "_avp_centroids": self._avp_centroids,
            "pooled_cov_cholesky_full_dim": self.pooled_cov_cholesky_full_dim,
            "distance_thresholds": self.distance_thresholds,
            "softmax_threshold": self.softmax_threshold,
            "_pca_matrix": self._pca_matrix,
            "_classes": self._classes,
            "_fitted": self._fitted,
            "_stored_avp_X": self._stored_avp_X,
            "_stored_avp_y": self._stored_avp_y,
            "_stored_avp_subjects": self._stored_avp_subjects,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "Classifier":
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        for k, v in state.items():
            setattr(obj, k, v)
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_f1(clf: "Classifier", X: np.ndarray, y: np.ndarray) -> float:
        """Macro F1 on the training set (used for overfit guard)."""
        from sklearn.metrics import f1_score
        preds = [cls for cls, _ in clf.predict(X)]
        return float(f1_score(y, preds, average="macro", zero_division=0))
