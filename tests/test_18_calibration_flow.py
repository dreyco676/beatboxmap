# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD tests for the end-to-end calibration flow (record → embed → fit).

Drives implementation of:
  - voxkit.ui.calibration_flow.CalibrationFlow
  - voxkit.ui.calibration_flow.NotReadyForPreview

============================================================
TEST LIST
============================================================

  T01  CalibrationFlow constructs with extractor, manager, classifier
  T02  add_sample(class_id, audio) succeeds without error
  T03  add_sample increases the per-class count in status()
  T04  can_preview() returns False before all classes have ≥1 sample
  T05  can_preview() returns True once all classes have ≥1 sample
  T06  preview() raises NotReadyForPreview before can_preview() is True
  T07  preview() returns (class_id: str, score: float) when ready
  T08  status() returns per-class count dict keyed by class name
  T09  commit() raises IncompleteCalibration with < MIN_SAMPLES_PER_CLASS
  T10  commit() returns CommitHandle when all classes have ≥3 samples
  T11  commit() raises CalibrationRejected on adversarial samples; classifier restored
  T12  record_abandon_event() emits calibration_abandoned to telemetry
  T13  add_sample with silent audio raises SilentSample
  T14  End-to-end flow: add → can_preview → preview → commit succeeds
"""

from __future__ import annotations

import numpy as np
import pytest

from voxkit.core.taxonomy import TaxonomyConfig
from voxkit.classifier.classifier import Classifier
from voxkit.classifier.calibration_manager import CalibrationManager, IncompleteCalibration
from voxkit.classifier.classifier import CalibrationRejected
from voxkit.telemetry.local_sink import LocalDiagnosticSink


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

_SR = 16_000
_DIM = 16
_TAXONOMY = TaxonomyConfig.default_v1_0()
_CLASSES = _TAXONOMY.classes   # ("kick", "snare", "closed_hat", "open_hat")


# ---------------------------------------------------------------
# Fake extractor
# ---------------------------------------------------------------

class _FakeExtractor:
    """Returns a fixed embedding for all calls (kick centroid = e_0)."""
    embedding_dim = _DIM
    required_sample_rate = _SR
    input_length = _SR

    def __init__(self, return_embedding: np.ndarray | None = None):
        if return_embedding is None:
            v = np.zeros(_DIM, dtype=np.float32)
            v[0] = 1.0   # kick centroid
            return_embedding = v
        self._emb = return_embedding
        self.calls: list = []

    def extract_at_onsets(self, audio, onset_times_s, sample_rate):
        self.calls.append((audio, list(onset_times_s)))
        return np.stack([self._emb.copy() for _ in onset_times_s])


# ---------------------------------------------------------------
# Synthetic fitted classifier
# ---------------------------------------------------------------

def _make_fitted_classifier(dim: int = _DIM) -> Classifier:
    """Return a Classifier fitted on synthetic AVP data (class k → e_k)."""
    rng = np.random.RandomState(0)
    classes = list(_CLASSES)
    n_subjects = 4
    samples_per = 3   # per class per subject

    avp_emb, avp_labels, avp_subjects = [], [], []
    for s_idx in range(n_subjects):
        for c_idx, cls in enumerate(classes):
            for _ in range(samples_per):
                base = np.zeros(dim, dtype=np.float64)
                base[c_idx] = 1.0
                emb = base + rng.randn(dim) * 0.02
                avp_emb.append(emb)
                avp_labels.append(cls)
                avp_subjects.append(f"s{s_idx}")

    avp_emb = np.array(avp_emb, dtype=np.float32)
    avp_labels = np.array(avp_labels)
    avp_subjects = np.array(avp_subjects)

    clf = Classifier.untrained(_TAXONOMY, dim)
    clf.fit(avp_emb, avp_labels, avp_subjects)
    return clf


def _make_flow(calibration_weight: float = 1.0, telemetry=None):
    """Return (flow, manager, classifier) ready for use."""
    from voxkit.ui.calibration_flow import CalibrationFlow
    clf = _make_fitted_classifier()
    mgr = CalibrationManager(clf, calibration_weight=calibration_weight,
                              telemetry=telemetry)
    extractor = _FakeExtractor()
    flow = CalibrationFlow(extractor, mgr, clf)
    return flow, mgr, clf


def _loud_audio(n: int = _SR) -> np.ndarray:
    """Non-silent float32 audio at amplitude 0.5."""
    return np.full(n, 0.5, dtype=np.float32)


def _add_one_per_class(flow) -> None:
    """Add one sample per class using loud audio."""
    for cls in _CLASSES:
        flow.add_sample(cls, _loud_audio())


def _add_three_per_class(flow, embedding: np.ndarray | None = None) -> None:
    """Add 3 samples per class directly to session (bypasses extractor).

    When embedding is provided, each sample uses that exact embedding so
    the classifier F1 after calibration is predictable.
    """
    from voxkit.classifier.calibration_manager import CalibrationSession
    for c_idx, cls in enumerate(_CLASSES):
        if embedding is not None:
            emb = embedding
        else:
            v = np.zeros(_DIM, dtype=np.float32)
            v[c_idx] = 1.0   # on-class centroid
            emb = v
        for _ in range(3):
            flow._session.add_sample(cls, emb)


# ---------------------------------------------------------------
# T01  Construction
# ---------------------------------------------------------------

def test_T01_calibration_flow_constructs():
    flow, _, _ = _make_flow()
    assert flow is not None


# ---------------------------------------------------------------
# T02  add_sample succeeds
# ---------------------------------------------------------------

def test_T02_add_sample_does_not_raise():
    flow, _, _ = _make_flow()
    flow.add_sample("kick", _loud_audio())  # must not raise


# ---------------------------------------------------------------
# T03  add_sample increases per-class count
# ---------------------------------------------------------------

def test_T03_add_sample_increments_count():
    flow, _, _ = _make_flow()
    assert flow.status()["kick"] == 0
    flow.add_sample("kick", _loud_audio())
    assert flow.status()["kick"] == 1
    flow.add_sample("kick", _loud_audio())
    assert flow.status()["kick"] == 2


# ---------------------------------------------------------------
# T04  can_preview returns False before all classes have samples
# ---------------------------------------------------------------

def test_T04_can_preview_false_before_all_classes():
    flow, _, _ = _make_flow()
    assert flow.can_preview() is False
    flow.add_sample("kick", _loud_audio())
    assert flow.can_preview() is False   # snare/hat still missing


# ---------------------------------------------------------------
# T05  can_preview returns True when all classes have ≥1
# ---------------------------------------------------------------

def test_T05_can_preview_true_after_one_each():
    flow, _, _ = _make_flow()
    _add_one_per_class(flow)
    assert flow.can_preview() is True


# ---------------------------------------------------------------
# T06  preview raises NotReadyForPreview before can_preview
# ---------------------------------------------------------------

def test_T06_preview_raises_if_not_ready():
    from voxkit.ui.calibration_flow import NotReadyForPreview
    flow, _, _ = _make_flow()
    with pytest.raises(NotReadyForPreview):
        flow.preview(_loud_audio())


# ---------------------------------------------------------------
# T07  preview returns (class_id, score) when ready
# ---------------------------------------------------------------

def test_T07_preview_returns_class_and_score():
    flow, _, _ = _make_flow()
    _add_one_per_class(flow)

    class_id, score = flow.preview(_loud_audio())
    assert isinstance(class_id, str)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # class_id must be a known class or unknown
    valid = set(_CLASSES) | {_TAXONOMY.unknown_class_id}
    assert class_id in valid


# ---------------------------------------------------------------
# T08  status() returns per-class count dict
# ---------------------------------------------------------------

def test_T08_status_returns_count_dict():
    flow, _, _ = _make_flow()
    s = flow.status()
    assert isinstance(s, dict)
    assert set(s.keys()) == set(_CLASSES)
    assert all(v == 0 for v in s.values())
    flow.add_sample("snare", _loud_audio())
    assert flow.status()["snare"] == 1


# ---------------------------------------------------------------
# T09  commit() raises IncompleteCalibration with too few samples
# ---------------------------------------------------------------

def test_T09_commit_raises_incomplete():
    flow, _, _ = _make_flow()
    # Add only 2 per class (need 3)
    for cls in _CLASSES:
        v = np.zeros(_DIM, dtype=np.float32); v[0] = 1.0
        flow._session.add_sample(cls, v)
        flow._session.add_sample(cls, v)
    with pytest.raises(IncompleteCalibration):
        flow.commit()


# ---------------------------------------------------------------
# T10  commit() returns CommitHandle on success
# ---------------------------------------------------------------

def test_T10_commit_returns_handle():
    from voxkit.classifier.calibration_manager import CommitHandle
    flow, _, _ = _make_flow(calibration_weight=1.0)
    # Add 3 good samples per class (embeddings near class centroids)
    _add_three_per_class(flow)
    handle = flow.commit()
    assert isinstance(handle, CommitHandle)


# ---------------------------------------------------------------
# T11  commit() propagates CalibrationRejected from manager
# ---------------------------------------------------------------

def test_T11_commit_propagates_calibration_rejected():
    """CalibrationFlow.commit() must re-raise CalibrationRejected from the manager."""
    flow, mgr, _ = _make_flow()
    _add_three_per_class(flow)

    # Patch manager.commit to simulate the guard firing
    def _raise(_session):
        raise CalibrationRejected(
            "test rejection",
            {"f1_calibrated": 0.5, "f1_baseline": 0.6, "delta": -0.1},
        )
    mgr.commit = _raise

    with pytest.raises(CalibrationRejected):
        flow.commit()


# ---------------------------------------------------------------
# T12  record_abandon_event emits telemetry
# ---------------------------------------------------------------

def test_T12_record_abandon_event_emits_telemetry():
    sink = LocalDiagnosticSink(None)
    flow, _, _ = _make_flow(telemetry=sink)
    flow.add_sample("kick", _loud_audio())
    flow.record_abandon_event()

    events = [e for e in sink.events if e.get("event") == "calibration_abandoned"]
    assert len(events) == 1
    assert "kick" in events[0]["details"]["counts"]


# ---------------------------------------------------------------
# T13  add_sample with silent audio raises SilentSample
# ---------------------------------------------------------------

def test_T13_silent_audio_raises():
    from voxkit.classifier.calibration_manager import SilentSample
    flow, _, _ = _make_flow()
    silent = np.zeros(_SR, dtype=np.float32)
    with pytest.raises(SilentSample):
        flow.add_sample("kick", silent)


# ---------------------------------------------------------------
# T14  Full end-to-end: add → can_preview → preview → commit
# ---------------------------------------------------------------

def test_T14_end_to_end_flow():
    from voxkit.classifier.calibration_manager import CommitHandle
    from voxkit.ui.calibration_flow import CalibrationFlow

    clf = _make_fitted_classifier()
    mgr = CalibrationManager(clf, calibration_weight=1.0)
    # Use a class-specific extractor so preview returns the right class
    kick_emb = np.zeros(_DIM, dtype=np.float32); kick_emb[0] = 1.0
    extractor = _FakeExtractor(return_embedding=kick_emb)
    flow = CalibrationFlow(extractor, mgr, clf)

    # Phase 1: add 1 sample per class to unlock preview
    assert not flow.can_preview()
    for cls in _CLASSES:
        flow.add_sample(cls, _loud_audio())
    assert flow.can_preview()

    # Phase 2: live preview
    class_id, score = flow.preview(_loud_audio())
    assert class_id == "kick"   # extractor returns kick centroid → classifier says kick
    assert score > 0.5

    # Phase 3: add enough for commit (need 3 total; already have 1)
    for cls in _CLASSES:
        v = np.zeros(_DIM, dtype=np.float32); v[list(_CLASSES).index(cls)] = 1.0
        flow._session.add_sample(cls, v)
        flow._session.add_sample(cls, v)
    # Now 3 per class

    handle = flow.commit()
    assert isinstance(handle, CommitHandle)
