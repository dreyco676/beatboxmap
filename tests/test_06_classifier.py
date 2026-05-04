# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 6: Classifier (composite gate).

Drives implementation of `voxkit.classifier.classifier`,
`voxkit.classifier.mahalanobis`, and `voxkit.classifier.calibration`.

Spec refs: §11 Component 6; Q26 (LR head), Q34 (full-dim Mahalanobis),
Q43 (PCA-or-not for LR), Q52 (AVP-only covariance), Q66 (TaxonomyConfig),
Q68 (Cholesky storage), Q71 (self-test overfit guard), Q75 (T held-out),
Q81 (CalibrationRejected wording).

This component has the most behavior; the test list is correspondingly
larger. Strict TDD ordering: each test drives one piece. Refactor only
on Green. Tidy First markers indicate where a structural change should
land in its own commit before the next behavioral test.

============================================================
TEST LIST (implement strictly in order)
============================================================

Mahalanobis-via-Cholesky building block (Q68)
  T01  mahalanobis_sq_via_cholesky on identity covariance == squared L2
  T02  Distance from a point to itself is 0
  T03  Distance is symmetric in (x, mu)
  T04  Distance via Cholesky agrees with explicit-inverse form (1e-8)
  T05  Cholesky round-trip (cov → cholesky → reconstruct) within 1e-10
  T06  Function rejects non-lower-triangular L

Pooled covariance fitting (Q34, Q52)
  T07  fit_mahalanobis_full_dim returns centroids, L, thresholds
  T08  Centroids use AVP + weighted calibration (sum-of-vectors check)
  T09  Pooled covariance uses AVP only, unweighted (Q52)
  T10  Per-class distance thresholds use AVP only (Q52)
  T11  L is lower-triangular
  T12  Distance thresholds approximate per-class 95th percentile

  -- TIDY FIRST before T13: extract `_softmax_with_temperature` to
     `voxkit.classifier.calibration` as a pure function. Used by both
     classifier and eval. Structural-only commit.

Logistic head + temperature scaling (Q26, Q75)
  T13  fit() trains LR on AVP and exposes coefficients
  T14  predict_proba sums to 1.0 per row
  T15  Untrained classifier raises on predict
  T16  Temperature T is fit on a held-out fold disjoint from training
       (verified by mocking the fit function and inspecting the data)
  T17  Temperature T is never < 0.1 or > 10 (sanity bounds)
  T18  Setting T=1.0 reduces to standard softmax

Composite unknown gate (Q34)
  T19  predict returns (class_id, score) per input
  T20  predict returns 'unknown' if max(softmax) < softmax_threshold
  T21  predict returns 'unknown' if Mahalanobis dist > distance_threshold
  T22  predict returns 'unknown' if both gates fire
  T23  predict returns trained class when both gates pass
  T24  Unknown class id is taken from TaxonomyConfig.unknown_class_id (Q66)

TaxonomyConfig parameterization (Q66)
  T25  Classifier with default config has 4 trained classes
  T26  Classifier with 5-class custom config trains and predicts
  T27  Classifier persists taxonomy in serialized form

PCA-64 path (Q43)
  T28  predict with PCA matrix uses LR_input = PCA @ embedding
  T29  predict without PCA matrix uses LR_input = embedding
  T30  Mahalanobis ALWAYS uses full-dim regardless of PCA presence (Q34)

Calibration path with weighting (Q42, Q65)
  T31  fit_with_calibration accepts AVP + calibration data
  T32  Centroids reflect the calibration_weight in the weighted sum
  T33  Pooled covariance ignores calibration data (AVP only, Q52)
  T34  fit_with_calibration with empty calibration data == fit() result

Self-test overfit guard (Q71)
  T35  Guard passes when calibrated F1 >= baseline F1
  T36  Guard passes when calibrated F1 within 1 point of baseline
  T37  Guard rejects when calibrated F1 drops by > 1 point
  T38  Rejection raises CalibrationRejected with diagnostics dict
  T39  Diagnostics dict contains f1_calibrated, f1_baseline, delta
  T40  CalibrationRejected message matches Q81 wording

  -- TIDY FIRST before T41: rename `LR_input` to `head_input` in private
     helper signatures (Q43 evolved meaning). Internal-only rename;
     no public API change. Structural commit, tests stay green.

Distribution-shift threshold (Q45)
  T41  get_distribution_shift_threshold returns AVP-derived value
  T42  Threshold is stable across two fits with same RNG seed

Operating-point selection (Q50, §7.3)
  T43  select_operating_point finds (sft_threshold, dist_pctile) satisfying
       both Q50 bounds (≤ 25% missed-unknown, ≤ 5% false-unknown)
  T44  select_operating_point raises if no pair satisfies bounds (loud-fail)

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

T16 strengthened: disjoint, not just smaller (Priya, Sam, Alex,
Casey, Riley, Lin, Marco: 7/9 — strong)
  T45  Q75: T-fit holdout is DISJOINT from LR training data and from
       covariance-fit data, and respects the subject grouping (no LOSO
       leakage). T16 only verifies n_T_samples < n_total — a holdout
       that overlaps with training would still pass T16. T45 captures
       the index sets and asserts no intersection.

Reproducibility (Priya, Sam, Alex, Casey, Riley, Marco, Dana: 7/9)
  T46  Same RNG seed → bit-exact LR coefficients, Cholesky factor, and
       distance thresholds across two fits. Without this, "the model
       changed" debugging is a nightmare.
  T47  Save/load round-trip preserves predictions exactly on a held-out
       batch (T27 only checks taxonomy persistence; nothing currently
       guards LR weights, Cholesky, T, or thresholds round-tripping).

Numerical stability (Priya, Lin, Sam, Alex, Casey, Marco: 6/9)
  T48  When D > N (high-dim embedding, few samples), pooled covariance
       is regularized (e.g., shrinkage to identity) so Cholesky succeeds
       and Mahalanobis is well-defined. The PANNs path with full-dim
       2048 + ~80 calibration samples is the realistic case.
  T49  Logits with large magnitudes (e.g., [200, -200, 0, 0]) softmax
       safely (no NaN/Inf); the LR head's predict_proba returns valid
       probabilities. Catches missing log-sum-exp in the head.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS
============================================================

OQ-1  Per-class F1 in classifier output (open question 22 in spec).
      [Priya, Marco, Alex: 3/9 — defer per spec disposition.]
OQ-2  1-sample-per-class calibration edge case behavior. [Priya, Casey:
      2/9 — defer; UI requires 3 minimum.]
OQ-3  Class-imbalance simulation guardrail. [Priya: 1/9 — REJECTED;
      AVP is verified balanced (spec §8 rejection list).]
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _synth_avp(n_classes=4, samples_per_class=20, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_classes, dim)) * 5.0
    X, y, subjects = [], [], []
    for c in range(n_classes):
        Xc = centroids[c] + rng.standard_normal((samples_per_class, dim))
        X.append(Xc)
        y.extend([f"class_{c}"] * samples_per_class)
        # 4 subjects, evenly distributed
        for i in range(samples_per_class):
            subjects.append(f"subj_{i % 4}")
    return np.vstack(X), np.array(y), np.array(subjects), centroids


# ---------------------------------------------------------------
# Mahalanobis-via-Cholesky building block (Q68)
# ---------------------------------------------------------------

def test_T01_identity_covariance_reduces_to_squared_l2():
    from voxkit.classifier.mahalanobis import mahalanobis_sq_via_cholesky
    L = np.eye(4)
    x = np.array([1.0, 2.0, 0.5, -1.0])
    mu = np.zeros(4)
    assert mahalanobis_sq_via_cholesky(x, mu, L) == pytest.approx(np.sum(x ** 2))


def test_T02_distance_from_point_to_itself_is_zero():
    from voxkit.classifier.mahalanobis import mahalanobis_sq_via_cholesky
    L = np.tril(np.array([[2.0, 0.0], [0.5, 1.5]]))
    x = np.array([1.5, -0.7])
    assert mahalanobis_sq_via_cholesky(x, x, L) == pytest.approx(0.0, abs=1e-12)


def test_T03_distance_symmetric_in_x_and_mu():
    from voxkit.classifier.mahalanobis import mahalanobis_sq_via_cholesky
    rng = np.random.default_rng(3)
    cov = np.eye(8) + 0.1 * np.ones((8, 8))
    L = np.linalg.cholesky(cov)
    x = rng.standard_normal(8)
    mu = rng.standard_normal(8)
    a = mahalanobis_sq_via_cholesky(x, mu, L)
    b = mahalanobis_sq_via_cholesky(mu, x, L)
    assert a == pytest.approx(b, rel=1e-12)


def test_T04_cholesky_form_agrees_with_inverse_form():
    from voxkit.classifier.mahalanobis import mahalanobis_sq_via_cholesky
    rng = np.random.default_rng(4)
    A = rng.standard_normal((16, 16))
    cov = A @ A.T + np.eye(16)
    L = np.linalg.cholesky(cov)
    inv_cov = np.linalg.inv(cov)

    x = rng.standard_normal(16)
    mu = rng.standard_normal(16)
    diff = x - mu
    via_chol = mahalanobis_sq_via_cholesky(x, mu, L)
    via_inv = float(diff @ inv_cov @ diff)
    assert via_chol == pytest.approx(via_inv, rel=1e-8)


def test_T05_cholesky_roundtrip_within_1e_10():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((20, 20))
    cov = A @ A.T + np.eye(20)
    L = np.linalg.cholesky(cov)
    reconstructed = L @ L.T
    assert np.max(np.abs(reconstructed - cov)) < 1e-10


def test_T06_non_lower_triangular_L_rejected():
    from voxkit.classifier.mahalanobis import mahalanobis_sq_via_cholesky
    bad_L = np.array([[1.0, 0.5], [0.0, 1.0]])  # upper-triangular
    with pytest.raises(ValueError, match="lower"):
        mahalanobis_sq_via_cholesky(np.zeros(2), np.zeros(2), bad_L)


# ---------------------------------------------------------------
# Pooled covariance fitting (Q34, Q52)
# ---------------------------------------------------------------

def test_T07_fit_returns_centroids_L_thresholds():
    from voxkit.classifier.mahalanobis import fit_mahalanobis_full_dim
    X, y, _, _ = _synth_avp()
    centroids, L, thresholds = fit_mahalanobis_full_dim(
        avp_embeddings=X, avp_labels=y,
        calibration_embeddings=np.zeros((0, X.shape[1])),
        calibration_labels=np.array([]),
        calibration_weight=1.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    assert centroids.shape == (4, X.shape[1])
    assert L.shape == (X.shape[1], X.shape[1])
    assert thresholds.shape == (4,)


def test_T08_centroids_use_weighted_calibration():
    from voxkit.classifier.mahalanobis import fit_mahalanobis_full_dim
    X, y, _, _ = _synth_avp(samples_per_class=10)
    cal_X = np.full((4, X.shape[1]), 100.0)   # outliers
    cal_y = np.array(["class_0", "class_1", "class_2", "class_3"])
    centroids_w0, _, _ = fit_mahalanobis_full_dim(
        X, y, cal_X, cal_y, calibration_weight=0.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    centroids_w5, _, _ = fit_mahalanobis_full_dim(
        X, y, cal_X, cal_y, calibration_weight=5.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    # Higher weight → centroids closer to the outlier (positive direction).
    assert np.all(np.linalg.norm(centroids_w5, axis=1) > np.linalg.norm(centroids_w0, axis=1))


def test_T09_covariance_ignores_calibration_data():
    """Q52: calibration data MUST NOT feed pooled covariance."""
    from voxkit.classifier.mahalanobis import fit_mahalanobis_full_dim
    X, y, _, _ = _synth_avp()
    cal_X = np.full((4, X.shape[1]), 1000.0)
    cal_y = np.array(["class_0", "class_1", "class_2", "class_3"])
    _, L_no_cal, _ = fit_mahalanobis_full_dim(
        X, y, np.zeros((0, X.shape[1])), np.array([]), calibration_weight=0.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    _, L_with_cal, _ = fit_mahalanobis_full_dim(
        X, y, cal_X, cal_y, calibration_weight=100.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    np.testing.assert_allclose(L_no_cal, L_with_cal, atol=1e-9)


def test_T10_thresholds_use_avp_only():
    """Q52: distance thresholds use AVP, unweighted."""
    from voxkit.classifier.mahalanobis import fit_mahalanobis_full_dim
    X, y, _, _ = _synth_avp()
    cal_X = np.full((4, X.shape[1]), 1000.0)
    cal_y = np.array(["class_0", "class_1", "class_2", "class_3"])
    _, _, t_no_cal = fit_mahalanobis_full_dim(
        X, y, np.zeros((0, X.shape[1])), np.array([]), calibration_weight=0.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    _, _, t_with_cal = fit_mahalanobis_full_dim(
        X, y, cal_X, cal_y, calibration_weight=10.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    np.testing.assert_allclose(t_no_cal, t_with_cal, atol=1e-9)


def test_T11_L_is_lower_triangular():
    from voxkit.classifier.mahalanobis import fit_mahalanobis_full_dim
    X, y, _, _ = _synth_avp()
    _, L, _ = fit_mahalanobis_full_dim(
        X, y, np.zeros((0, X.shape[1])), np.array([]), calibration_weight=1.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    # Upper-triangular elements must be exactly zero.
    upper = np.triu(L, k=1)
    assert np.max(np.abs(upper)) == 0.0


def test_T12_distance_thresholds_approximate_p95():
    from voxkit.classifier.mahalanobis import fit_mahalanobis_full_dim, mahalanobis_sq_via_cholesky
    X, y, _, _ = _synth_avp(samples_per_class=200, dim=8, seed=12)
    centroids, L, thresholds = fit_mahalanobis_full_dim(
        X, y, np.zeros((0, X.shape[1])), np.array([]), calibration_weight=1.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    # For class 0, ~95% of AVP distances should be at or below threshold.
    in_class = X[y == "class_0"]
    dists = np.array([np.sqrt(mahalanobis_sq_via_cholesky(x, centroids[0], L)) for x in in_class])
    frac_under = float(np.mean(dists <= thresholds[0]))
    assert 0.92 < frac_under < 0.98


# ----- TIDY FIRST checkpoint -----
# Extract `_softmax_with_temperature(logits, T)` to
# `voxkit.classifier.calibration`. Pure structural change.


# ---------------------------------------------------------------
# Logistic head + temperature scaling (Q26, Q75)
# ---------------------------------------------------------------

def test_T13_fit_trains_lr_and_exposes_coefficients():
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp()
    clf = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)
    assert clf.lr_coefficients_.shape[1] == X.shape[1]


def test_T14_predict_proba_sums_to_one():
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp()
    clf = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)
    probs = clf.predict_proba(X[:10])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_T15_untrained_classifier_raises_on_predict():
    from voxkit.classifier.classifier import Classifier, NotFittedError
    clf = Classifier.untrained(taxonomy=None, embedding_dim=32)
    with pytest.raises(NotFittedError):
        clf.predict(np.zeros((1, 32)))


def test_T16_temperature_fit_uses_disjoint_holdout():
    """Q75: T fit on a held-out subject inside the training fold; never
    overlaps with LR training data or covariance estimation data."""
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp()

    captured = {}
    real_fit_T = __import__("voxkit.classifier.calibration", fromlist=["fit_temperature"]).fit_temperature

    def spy(logits, labels, **kwargs):
        captured["n_T_samples"] = len(labels)
        return real_fit_T(logits, labels, **kwargs)

    with patch("voxkit.classifier.classifier.fit_temperature", side_effect=spy):
        clf = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
        clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)

    # T must be fit on strictly fewer samples than the LR training set.
    assert captured["n_T_samples"] < len(y)


def test_T17_temperature_within_sanity_bounds():
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp()
    clf = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)
    assert 0.1 <= clf.T <= 10.0


def test_T18_T_equal_one_reduces_to_standard_softmax():
    from voxkit.classifier.calibration import softmax_with_temperature
    logits = np.array([[1.0, 2.0, 3.0]])
    standard = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    np.testing.assert_allclose(
        softmax_with_temperature(logits, T=1.0), standard, atol=1e-10,
    )


# ---------------------------------------------------------------
# Composite unknown gate (Q34)
# ---------------------------------------------------------------

def _trained_classifier(seed=42):
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp(seed=seed)
    clf = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)
    return clf, X


def test_T19_predict_returns_class_id_and_score():
    clf, X = _trained_classifier()
    out = clf.predict(X[:1])
    assert isinstance(out, list)
    cls, score = out[0]
    assert isinstance(cls, str)
    assert 0.0 <= score <= 1.0


def test_T20_unknown_when_softmax_below_threshold():
    clf, _ = _trained_classifier()
    clf.softmax_threshold = 0.99   # force gate
    # Synthetic mixture: equal probability across classes.
    far_point = np.zeros((1, clf.embedding_dim))
    out = clf.predict(far_point)
    assert out[0][0] == "unknown"


def test_T21_unknown_when_distance_above_threshold():
    clf, _ = _trained_classifier()
    # Inflate softmax_threshold low; force the OOD path.
    clf.softmax_threshold = 0.01
    # Force distance_thresholds to near-zero so anything triggers OOD.
    clf.distance_thresholds = np.zeros_like(clf.distance_thresholds) + 1e-9
    far_point = np.full((1, clf.embedding_dim), 1000.0)
    out = clf.predict(far_point)
    assert out[0][0] == "unknown"


def test_T22_unknown_when_both_gates_fire():
    clf, _ = _trained_classifier()
    clf.softmax_threshold = 0.99
    clf.distance_thresholds = np.zeros_like(clf.distance_thresholds) + 1e-9
    far_point = np.full((1, clf.embedding_dim), 1000.0)
    assert clf.predict(far_point)[0][0] == "unknown"


def test_T23_trained_class_when_gates_pass():
    clf, X = _trained_classifier()
    clf.softmax_threshold = 0.0
    clf.distance_thresholds = np.full_like(clf.distance_thresholds, 1e9)
    out = clf.predict(X[:5])
    for cls, _ in out:
        assert cls != "unknown"


def test_T24_unknown_class_id_from_taxonomy():
    """Q66: unknown name comes from TaxonomyConfig, not a hardcoded literal."""
    from voxkit.classifier.classifier import Classifier
    from voxkit.core.taxonomy import TaxonomyConfig
    tax = TaxonomyConfig(
        classes=("a", "b", "c", "d"),
        midi_mapping={"a": 36, "b": 38, "c": 42, "d": 46},
        unknown_class_id="OOD",
    )
    X, y, subjects, _ = _synth_avp()
    y = np.array([{"class_0": "a", "class_1": "b", "class_2": "c", "class_3": "d"}[v] for v in y])
    clf = Classifier.untrained(taxonomy=tax, embedding_dim=X.shape[1])
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)
    clf.softmax_threshold = 0.99
    far = np.zeros((1, X.shape[1]))
    assert clf.predict(far)[0][0] == "OOD"


# ---------------------------------------------------------------
# TaxonomyConfig parameterization (Q66)
# ---------------------------------------------------------------

def test_T25_default_classifier_has_4_trained_classes():
    from voxkit.classifier.classifier import Classifier
    clf = Classifier.untrained(taxonomy=None, embedding_dim=32)
    assert len(clf.taxonomy.classes) == 4


def test_T26_5_class_custom_config_works_end_to_end():
    from voxkit.classifier.classifier import Classifier
    from voxkit.core.taxonomy import TaxonomyConfig
    tax = TaxonomyConfig(
        classes=("a", "b", "c", "d", "e"),
        midi_mapping={"a": 36, "b": 38, "c": 42, "d": 46, "e": 50},
    )
    X, y, subjects, _ = _synth_avp(n_classes=5)
    y = np.array([{"class_0": "a", "class_1": "b", "class_2": "c",
                   "class_3": "d", "class_4": "e"}[v] for v in y])
    clf = Classifier.untrained(taxonomy=tax, embedding_dim=X.shape[1])
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)
    out = clf.predict(X[:5])
    assert all(cls in tax.classes or cls == tax.unknown_class_id for cls, _ in out)


def test_T27_classifier_persists_taxonomy(tmp_path):
    from voxkit.classifier.classifier import Classifier
    clf, _ = _trained_classifier()
    p = tmp_path / "model.bundle"
    clf.save(p)
    reloaded = Classifier.load(p)
    assert reloaded.taxonomy == clf.taxonomy


# ---------------------------------------------------------------
# PCA-64 path (Q43)
# ---------------------------------------------------------------

def test_T28_predict_with_pca_uses_projected_input_for_lr():
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp(dim=64)
    pca = np.eye(32, 64)   # truncate to 32 dims for the test
    clf = Classifier.untrained(taxonomy=None, embedding_dim=64)
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects, pca_matrix=pca)
    clf.softmax_threshold = 0.0
    clf.distance_thresholds = np.full_like(clf.distance_thresholds, 1e9)
    # Smoke: predict runs with a PCA path.
    assert clf.predict(X[:1])[0][0] != "unknown"


def test_T29_predict_without_pca_uses_full_dim_for_lr():
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp(dim=64)
    clf = Classifier.untrained(taxonomy=None, embedding_dim=64)
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects, pca_matrix=None)
    assert clf.lr_coefficients_.shape[1] == 64


def test_T30_mahalanobis_always_full_dim_regardless_of_pca():
    """Q34: Mahalanobis ships full-dim pooled. PCA only affects LR head."""
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp(dim=64)
    pca = np.eye(32, 64)
    clf = Classifier.untrained(taxonomy=None, embedding_dim=64)
    clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects, pca_matrix=pca)
    assert clf.class_centroids_full_dim.shape[1] == 64
    assert clf.pooled_cov_cholesky_full_dim.shape == (64, 64)


# ---------------------------------------------------------------
# Calibration path with weighting (Q42, Q65)
# ---------------------------------------------------------------

def test_T31_fit_with_calibration_accepts_avp_plus_calibration():
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp()
    cal_X = np.zeros((8, X.shape[1]))
    cal_y = np.array(["class_0"] * 2 + ["class_1"] * 2 + ["class_2"] * 2 + ["class_3"] * 2)
    clf = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf.fit_with_calibration(
        avp_embeddings=X, avp_labels=y, avp_subjects=subjects,
        calibration_embeddings=cal_X, calibration_labels=cal_y,
        calibration_weight=5.0,
    )
    assert clf.lr_coefficients_ is not None


def test_T32_centroids_reflect_calibration_weight():
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp()
    cal_X = np.full((4, X.shape[1]), 50.0)   # far in positive direction
    cal_y = np.array(["class_0", "class_1", "class_2", "class_3"])
    clf_low = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf_low.fit_with_calibration(X, y, subjects, cal_X, cal_y, calibration_weight=0.5)
    clf_high = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf_high.fit_with_calibration(X, y, subjects, cal_X, cal_y, calibration_weight=20.0)
    # Higher weight pulls centroids further toward 50.
    assert np.linalg.norm(clf_high.class_centroids_full_dim) > np.linalg.norm(clf_low.class_centroids_full_dim)


def test_T33_pooled_covariance_ignores_calibration_data():
    """Q52: covariance is AVP-only regardless of weight."""
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp()
    cal_X = np.full((4, X.shape[1]), 1000.0)
    cal_y = np.array(["class_0", "class_1", "class_2", "class_3"])
    clf_low = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf_low.fit_with_calibration(X, y, subjects, cal_X, cal_y, calibration_weight=0.0)
    clf_high = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    clf_high.fit_with_calibration(X, y, subjects, cal_X, cal_y, calibration_weight=100.0)
    np.testing.assert_allclose(
        clf_low.pooled_cov_cholesky_full_dim, clf_high.pooled_cov_cholesky_full_dim, atol=1e-9,
    )


def test_T34_empty_calibration_equals_plain_fit():
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp()
    plain = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    plain.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)
    cal = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    cal.fit_with_calibration(
        X, y, subjects,
        calibration_embeddings=np.zeros((0, X.shape[1])),
        calibration_labels=np.array([]),
        calibration_weight=5.0,
    )
    np.testing.assert_allclose(plain.class_centroids_full_dim, cal.class_centroids_full_dim, atol=1e-10)


# ---------------------------------------------------------------
# Self-test overfit guard (Q71)
# ---------------------------------------------------------------

def test_T35_guard_passes_when_calibrated_f1_at_least_baseline():
    from voxkit.classifier.calibration import self_test_overfit_guard
    passed, diag = self_test_overfit_guard(
        f1_calibrated=0.85, f1_baseline=0.85,
    )
    assert passed
    assert diag["delta"] == pytest.approx(0.0)


def test_T36_guard_passes_within_one_point():
    from voxkit.classifier.calibration import self_test_overfit_guard
    passed, _ = self_test_overfit_guard(f1_calibrated=0.84, f1_baseline=0.85)   # 1 point drop
    assert passed


def test_T37_guard_rejects_above_one_point_drop():
    from voxkit.classifier.calibration import self_test_overfit_guard
    passed, _ = self_test_overfit_guard(f1_calibrated=0.83, f1_baseline=0.85)   # 2 points
    assert not passed


def test_T38_rejection_raises_calibration_rejected_with_diagnostics():
    from voxkit.classifier.classifier import Classifier, CalibrationRejected
    X, y, subjects, _ = _synth_avp()
    clf = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
    # Bad calibration: random labels that hurt the model.
    bad_cal = np.full((20, X.shape[1]), 0.0)
    bad_y = np.tile(["class_3"], 20)
    with patch("voxkit.classifier.classifier.self_test_overfit_guard",
               return_value=(False, {"f1_calibrated": 0.7, "f1_baseline": 0.85, "delta": -0.15})):
        with pytest.raises(CalibrationRejected) as exc:
            clf.fit_with_calibration(X, y, subjects, bad_cal, bad_y, calibration_weight=10.0)
    assert exc.value.diagnostics["delta"] == pytest.approx(-0.15)


def test_T39_diagnostics_dict_contains_required_fields():
    from voxkit.classifier.classifier import CalibrationRejected
    e = CalibrationRejected(message="...", diagnostics={
        "f1_calibrated": 0.7, "f1_baseline": 0.85, "delta": -0.15,
    })
    for k in ("f1_calibrated", "f1_baseline", "delta"):
        assert k in e.diagnostics


def test_T40_calibration_rejected_message_matches_q81():
    from voxkit.classifier.classifier import Q81_DIALOG_TEXT
    # Anchors that must appear in the Q81-compliant text:
    for snippet in (
        "didn't improve classification",
        "previous calibration has been restored",
        "more or quieter samples",
    ):
        assert snippet in Q81_DIALOG_TEXT


# ----- TIDY FIRST checkpoint -----
# Rename `LR_input` → `head_input` in private helpers for clarity given
# the PCA-or-not branching. Public API unchanged.


# ---------------------------------------------------------------
# Distribution-shift threshold (Q45)
# ---------------------------------------------------------------

def test_T41_distribution_shift_threshold_returns_avp_derived_value():
    clf, _ = _trained_classifier()
    t = clf.get_distribution_shift_threshold()
    assert t > 0.0


def test_T42_threshold_stable_across_seeded_fits():
    a, _ = _trained_classifier(seed=42)
    b, _ = _trained_classifier(seed=42)
    assert a.get_distribution_shift_threshold() == pytest.approx(
        b.get_distribution_shift_threshold(), rel=1e-6
    )


# ---------------------------------------------------------------
# Operating-point selection (Q50, §7.3)
# ---------------------------------------------------------------

def test_T43_operating_point_satisfies_q50_bounds():
    """§7.3: select (softmax_threshold, distance_percentile) satisfying
    Q50 bounds (≤ 25% missed-unknown, ≤ 5% false-unknown)."""
    from voxkit.classifier.calibration import select_operating_point
    # Synthetic sweep: provide enough headroom that some pair satisfies bounds.
    sweep = [
        {"softmax_threshold": 0.45, "distance_pctile": 95,
         "missed_unknown": 0.20, "false_unknown": 0.04},
        {"softmax_threshold": 0.55, "distance_pctile": 95,
         "missed_unknown": 0.15, "false_unknown": 0.06},
    ]
    op = select_operating_point(sweep, max_missed_unknown=0.25, max_false_unknown=0.05)
    assert op["softmax_threshold"] == 0.45
    assert op["distance_pctile"] == 95


def test_T44_operating_point_raises_when_no_pair_satisfies():
    from voxkit.classifier.calibration import (
        select_operating_point, NoOperatingPointFound,
    )
    sweep = [
        {"softmax_threshold": 0.45, "distance_pctile": 95,
         "missed_unknown": 0.30, "false_unknown": 0.10},
    ]
    with pytest.raises(NoOperatingPointFound):
        select_operating_point(sweep, max_missed_unknown=0.25, max_false_unknown=0.05)


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T45_temperature_holdout_is_disjoint_and_respects_subjects():
    """Q75 strengthened: T16 only verifies n_T_samples < n_total. A
    holdout that overlaps with LR training data would still satisfy
    T16 but violate Q75. Capture the actual index sets and assert
    (a) no overlap and (b) subject groups don't bleed across the fold."""
    from voxkit.classifier.classifier import Classifier
    X, y, subjects, _ = _synth_avp(samples_per_class=40, seed=45)

    captured = {"T_indices": None, "LR_indices": None, "T_subjects": None,
                "LR_subjects": None}

    def spy_T(logits, labels, *, indices=None, **kwargs):
        captured["T_indices"] = set(indices) if indices is not None else set()
        return 1.0

    def spy_LR(X_train, y_train, *, indices=None, **kwargs):
        captured["LR_indices"] = set(indices) if indices is not None else set()
        # mimic sklearn return shape minimally
        n_classes = len(set(y_train))
        return np.zeros((n_classes, X_train.shape[1])), np.zeros(n_classes)

    with patch("voxkit.classifier.classifier.fit_temperature", side_effect=spy_T), \
         patch("voxkit.classifier.classifier.fit_lr_head", side_effect=spy_LR):
        clf = Classifier.untrained(taxonomy=None, embedding_dim=X.shape[1])
        clf.fit(avp_embeddings=X, avp_labels=y, avp_subjects=subjects)

    overlap = captured["T_indices"] & captured["LR_indices"]
    assert overlap == set(), (
        f"T-fit and LR-fit folds overlap on {len(overlap)} samples; "
        "Q75 requires disjoint folds"
    )
    # Subject-grouped holdout: every sample's subject in T's fold must
    # NOT appear in LR's fold (LOSO at the subject level).
    T_subjects = {subjects[i] for i in captured["T_indices"]}
    LR_subjects = {subjects[i] for i in captured["LR_indices"]}
    assert T_subjects.isdisjoint(LR_subjects), (
        f"subjects bleed across T/LR folds: {T_subjects & LR_subjects}"
    )


def test_T46_same_seed_produces_bit_exact_classifier():
    """Reproducibility is the foundation of "did the model change?"
    debugging. If two fits with the same seed differ at all in LR
    coefficients or Cholesky factor, downstream eval-vs-baseline
    comparison becomes unreliable."""
    a, _ = _trained_classifier(seed=46)
    b, _ = _trained_classifier(seed=46)
    np.testing.assert_array_equal(a.lr_coefficients_, b.lr_coefficients_)
    np.testing.assert_array_equal(
        a.pooled_cov_cholesky_full_dim, b.pooled_cov_cholesky_full_dim,
    )
    np.testing.assert_array_equal(a.distance_thresholds, b.distance_thresholds)
    assert a.T == b.T


def test_T47_save_load_preserves_predictions_exactly(tmp_path):
    """T27 verifies taxonomy persists across save/load. T47 verifies the
    NUMERIC parts (LR weights, Cholesky, T, thresholds) — without which
    a saved bundle is correct in shape but predicts differently than
    the in-memory classifier did at fit time."""
    clf, X = _trained_classifier(seed=47)
    p = tmp_path / "clf.bundle"
    clf.save(p)
    from voxkit.classifier.classifier import Classifier
    reloaded = Classifier.load(p)
    a = clf.predict(X[:20])
    b = reloaded.predict(X[:20])
    assert a == b, "save/load drift in predictions; numeric state not round-tripping"


def test_T48_high_dim_low_n_covariance_is_regularized():
    """The realistic PANNs case: D=2048, N≈80 calibration samples. A
    plain sample covariance is rank-deficient and Cholesky will fail.
    The implementation must regularize (shrinkage to identity, or
    diagonal loading); this test confirms the path doesn't crash."""
    from voxkit.classifier.mahalanobis import fit_mahalanobis_full_dim
    rng = np.random.default_rng(48)
    D, n_per_class = 2048, 5   # 4 classes * 5 = 20 samples << 2048 dims
    X = rng.standard_normal((4 * n_per_class, D)).astype(np.float32)
    y = np.array(sum(([f"class_{c}"] * n_per_class for c in range(4)), []))
    centroids, L, thresholds = fit_mahalanobis_full_dim(
        avp_embeddings=X, avp_labels=y,
        calibration_embeddings=np.zeros((0, D), dtype=np.float32),
        calibration_labels=np.array([]),
        calibration_weight=1.0,
        classes=["class_0", "class_1", "class_2", "class_3"],
    )
    assert L.shape == (D, D)
    # The diagonal should be strictly positive (regularization succeeded).
    assert np.all(np.diag(L) > 0)


def test_T49_extreme_logits_do_not_overflow_softmax():
    """A Classifier whose LR head outputs huge logits (e.g., near a
    decision boundary with very confident embeddings) must still
    produce valid probabilities. Log-sum-exp / numerically-stable
    softmax is the contract."""
    from voxkit.classifier.calibration import softmax_with_temperature
    huge = np.array([[200.0, -200.0, 0.0, 0.0]])
    out = softmax_with_temperature(huge, T=1.0)
    assert np.all(np.isfinite(out))
    assert out.sum() == pytest.approx(1.0, abs=1e-6)
    # The first class wins; its prob should be ~1.0 (not NaN, not 0).
    assert out[0, 0] > 0.99
