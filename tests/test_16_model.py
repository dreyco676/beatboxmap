# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD tests for the concrete Model class connecting EmbeddingExtractor + Classifier.

Drives implementation of:
  - voxkit.ui.model.Model
  - voxkit.ui.model.NotPreparedError

============================================================
TEST LIST
============================================================

  T01  Model can be constructed with an extractor and a classifier
  T02  model.taxonomy delegates to classifier.taxonomy
  T03  model.embed() raises NotPreparedError before prepare()
  T04  prepare() enables embed(); subsequent embed() does not raise
  T05  embed() calls extractor.extract_at_onsets with the given onset_t
  T06  embed() returns a 1-D ndarray of shape (embedding_dim,)
  T07  predict() calls classifier.predict with stacked embeddings as ndarray
  T08  predict() returns a list of Event objects
  T09  predict() sets Event.t from the onset_t passed to embed()
  T10  predict() sets Event.class_id and Event.score from classifier output
  T11  predict([]) returns []
  T12  prepare() resets onset tracking so a second session works cleanly
  T13  Model integrates with InferenceWorker end-to-end
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from voxkit.core.session import Event
from voxkit.core.taxonomy import TaxonomyConfig


# ---------------------------------------------------------------
# Fake collaborators (no ONNX, no fitting needed)
# ---------------------------------------------------------------

_DIM = 16
_SR = 16_000


class _FakeExtractor:
    embedding_dim = _DIM
    required_sample_rate = _SR
    input_length = _SR

    def __init__(self):
        self.calls: list[tuple[np.ndarray, list[float], int]] = []

    def extract_at_onsets(
        self,
        audio: np.ndarray,
        onset_times_s: list[float],
        sample_rate: int,
    ) -> np.ndarray:
        self.calls.append((audio, onset_times_s, sample_rate))
        n = len(onset_times_s)
        if n == 0:
            return np.zeros((0, _DIM), dtype=np.float32)
        # Return deterministic embeddings based on onset time
        rows = []
        for t in onset_times_s:
            rows.append(np.full(_DIM, t, dtype=np.float32))
        return np.stack(rows)


class _FakeClassifier:
    taxonomy = TaxonomyConfig.default_v1_0()

    def __init__(self):
        self.calls: list[np.ndarray] = []
        self._class_ids = list(self.taxonomy.classes)

    def predict(self, X: np.ndarray) -> list[tuple[str, float]]:
        self.calls.append(X.copy())
        results = []
        for i in range(len(X)):
            cls = self._class_ids[i % len(self._class_ids)]
            score = float(0.9 - 0.05 * i)
            results.append((cls, score))
        return results


# ---------------------------------------------------------------
# T01  Construction
# ---------------------------------------------------------------

def test_T01_model_constructed():
    from voxkit.ui.model import Model
    extractor = _FakeExtractor()
    classifier = _FakeClassifier()
    model = Model(extractor, classifier)
    assert model is not None


# ---------------------------------------------------------------
# T02  taxonomy delegation
# ---------------------------------------------------------------

def test_T02_taxonomy_delegates_to_classifier():
    from voxkit.ui.model import Model
    extractor = _FakeExtractor()
    classifier = _FakeClassifier()
    model = Model(extractor, classifier)
    assert model.taxonomy is classifier.taxonomy


# ---------------------------------------------------------------
# T03  NotPreparedError before prepare()
# ---------------------------------------------------------------

def test_T03_embed_raises_before_prepare():
    from voxkit.ui.model import Model, NotPreparedError
    model = Model(_FakeExtractor(), _FakeClassifier())
    with pytest.raises(NotPreparedError):
        model.embed(0.0)


# ---------------------------------------------------------------
# T04  prepare() enables embed()
# ---------------------------------------------------------------

def test_T04_embed_succeeds_after_prepare():
    from voxkit.ui.model import Model
    audio = np.zeros(_SR * 2, dtype=np.float32)
    model = Model(_FakeExtractor(), _FakeClassifier())
    model.prepare(audio)
    # Should not raise
    emb = model.embed(0.5)
    assert emb is not None


# ---------------------------------------------------------------
# T05  embed() forwards onset_t to extractor
# ---------------------------------------------------------------

def test_T05_embed_calls_extractor_with_onset_t():
    from voxkit.ui.model import Model
    extractor = _FakeExtractor()
    audio = np.zeros(_SR * 3, dtype=np.float32)
    model = Model(extractor, _FakeClassifier())
    model.prepare(audio)
    model.embed(1.0)

    assert len(extractor.calls) == 1
    _, onset_times, sr = extractor.calls[0]
    assert onset_times == [1.0]
    assert sr == _SR


# ---------------------------------------------------------------
# T06  embed() returns (embedding_dim,) shaped array
# ---------------------------------------------------------------

def test_T06_embed_returns_1d_array():
    from voxkit.ui.model import Model
    audio = np.zeros(_SR * 2, dtype=np.float32)
    model = Model(_FakeExtractor(), _FakeClassifier())
    model.prepare(audio)
    emb = model.embed(0.0)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (_DIM,)


# ---------------------------------------------------------------
# T07  predict() calls classifier with stacked ndarray
# ---------------------------------------------------------------

def test_T07_predict_calls_classifier_with_stacked_ndarray():
    from voxkit.ui.model import Model
    extractor = _FakeExtractor()
    classifier = _FakeClassifier()
    audio = np.zeros(_SR * 3, dtype=np.float32)
    model = Model(extractor, classifier)
    model.prepare(audio)

    emb0 = model.embed(0.0)
    emb1 = model.embed(1.0)
    model.predict([emb0, emb1])

    assert len(classifier.calls) == 1
    X = classifier.calls[0]
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, _DIM)


# ---------------------------------------------------------------
# T08  predict() returns list of Event objects
# ---------------------------------------------------------------

def test_T08_predict_returns_events():
    from voxkit.ui.model import Model
    audio = np.zeros(_SR * 2, dtype=np.float32)
    model = Model(_FakeExtractor(), _FakeClassifier())
    model.prepare(audio)
    emb = model.embed(0.0)
    events = model.predict([emb])
    assert isinstance(events, list)
    assert len(events) == 1
    assert isinstance(events[0], Event)


# ---------------------------------------------------------------
# T09  predict() sets Event.t from onset_t
# ---------------------------------------------------------------

def test_T09_event_t_matches_onset_t():
    from voxkit.ui.model import Model
    audio = np.zeros(_SR * 3, dtype=np.float32)
    model = Model(_FakeExtractor(), _FakeClassifier())
    model.prepare(audio)
    emb0 = model.embed(0.25)
    emb1 = model.embed(1.75)
    events = model.predict([emb0, emb1])
    assert events[0].t == pytest.approx(0.25)
    assert events[1].t == pytest.approx(1.75)


# ---------------------------------------------------------------
# T10  predict() sets Event.class_id and Event.score from classifier
# ---------------------------------------------------------------

def test_T10_event_class_id_and_score_from_classifier():
    from voxkit.ui.model import Model
    classifier = _FakeClassifier()
    audio = np.zeros(_SR * 2, dtype=np.float32)
    model = Model(_FakeExtractor(), classifier)
    model.prepare(audio)
    emb = model.embed(0.0)
    events = model.predict([emb])

    expected_class, expected_score = classifier._class_ids[0], 0.9
    assert events[0].class_id == expected_class
    assert events[0].score == pytest.approx(expected_score)


# ---------------------------------------------------------------
# T11  predict([]) returns []
# ---------------------------------------------------------------

def test_T11_predict_empty_returns_empty():
    from voxkit.ui.model import Model
    audio = np.zeros(_SR, dtype=np.float32)
    model = Model(_FakeExtractor(), _FakeClassifier())
    model.prepare(audio)
    result = model.predict([])
    assert result == []


# ---------------------------------------------------------------
# T12  prepare() resets onset tracking
# ---------------------------------------------------------------

def test_T12_prepare_resets_onset_tracking():
    from voxkit.ui.model import Model
    extractor = _FakeExtractor()
    classifier = _FakeClassifier()
    audio = np.zeros(_SR * 3, dtype=np.float32)
    model = Model(extractor, classifier)

    # First session: two onsets
    model.prepare(audio)
    emb0 = model.embed(0.0)
    emb1 = model.embed(1.0)
    events_first = model.predict([emb0, emb1])
    assert len(events_first) == 2

    # Second session: one onset — must not carry over first session's data
    model.prepare(audio)
    emb2 = model.embed(2.0)
    events_second = model.predict([emb2])
    assert len(events_second) == 1
    assert events_second[0].t == pytest.approx(2.0)


# ---------------------------------------------------------------
# T13  End-to-end with InferenceWorker
# ---------------------------------------------------------------

def test_T13_integrates_with_inference_worker():
    from voxkit.ui.model import Model
    from voxkit.ui.inference_worker import InferenceWorker

    extractor = _FakeExtractor()
    classifier = _FakeClassifier()
    audio = np.zeros(_SR * 2, dtype=np.float32)
    model = Model(extractor, classifier)
    model.prepare(audio)

    completed_events: list = []
    failed_msgs: list = []

    worker = InferenceWorker(audio, model)
    worker.completed.connect(lambda evts: completed_events.extend(evts))
    worker.failed.connect(lambda msg: failed_msgs.append(msg))

    worker.start()
    done = worker.wait_for_completion(timeout=5.0)

    assert done, "InferenceWorker did not finish in time"
    assert failed_msgs == [], f"Worker failed: {failed_msgs}"
    assert len(completed_events) >= 1
    event = completed_events[0]
    assert isinstance(event, Event)
    assert isinstance(event.t, float)
    assert isinstance(event.class_id, str)
