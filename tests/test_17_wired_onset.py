# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD tests for wiring the real OnsetDetector into InferenceWorker and
run_pipeline, replacing the [0.0] placeholder.

Drives changes to:
  - voxkit.ui.inference_worker.InferenceWorker.__init__  (onset_detector kwarg)
  - voxkit.ui.inference_worker.InferenceWorker._detect_onsets  (delegate to it)
  - voxkit.ui.inference_pipeline.run_pipeline             (detect_onsets kwarg)

============================================================
TEST LIST
============================================================

  T01  InferenceWorker accepts onset_detector kwarg without error
  T02  With onset_detector=None (default), stub behaviour preserved
  T03  With onset_detector provided, _detect_onsets delegates to detector.detect(audio)
  T04  Real OnsetDetector(16_000) is accepted as onset_detector
  T05  End-to-end: audio with a clear transient → worker detects onset and emits events
  T06  run_pipeline accepts detect_onsets callable kwarg
  T07  With detect_onsets=None, run_pipeline uses the stub (empty→[], nonempty→[0.0])
  T08  With detect_onsets provided, run_pipeline uses that callable
  T09  Two onsets detected → two embed() calls → two events
  T10  Empty audio with real OnsetDetector → no onsets → empty event list
"""

from __future__ import annotations

import numpy as np
import pytest

_SR = 16_000


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _transient(t_s: float, duration_s: float = 4.0, amplitude: float = 0.8) -> np.ndarray:
    """Silent audio with a single loud burst at t_s (width 5 ms)."""
    n = int(duration_s * _SR)
    audio = np.zeros(n, dtype=np.float32)
    center = int(t_s * _SR)
    width = int(0.005 * _SR)   # 5 ms
    start = max(0, center - width // 2)
    end = min(n, start + width)
    audio[start:end] = amplitude
    return audio


def _two_transients(t1: float, t2: float, duration_s: float = 4.0) -> np.ndarray:
    n = int(duration_s * _SR)
    audio = np.zeros(n, dtype=np.float32)
    for t in (t1, t2):
        center = int(t * _SR)
        width = int(0.005 * _SR)
        start = max(0, center - width // 2)
        end = min(n, start + width)
        audio[start:end] = 0.8
    return audio


class _FakeModel:
    """Minimal model stub: embed returns a fixed vector, predict returns Events."""
    def __init__(self):
        from voxkit.core.taxonomy import TaxonomyConfig
        self.taxonomy = TaxonomyConfig.default_v1_0()
        self._audio = None
        self._onsets = []

    def prepare(self, audio):
        self._audio = audio
        self._onsets = []

    def embed(self, onset_t: float):
        self._onsets.append(onset_t)
        return np.ones(16, dtype=np.float32)

    def predict(self, embeddings):
        from voxkit.core.session import Event
        return [
            Event(t=t, class_id="kick", score=0.9)
            for t in self._onsets
        ]


class _SpyDetector:
    """Records calls; returns a preset list of onset times."""
    def __init__(self, returns: list[float]):
        self._returns = returns
        self.calls: list[np.ndarray] = []

    def detect(self, audio: np.ndarray) -> list[float]:
        self.calls.append(audio)
        return list(self._returns)


# ---------------------------------------------------------------
# T01  InferenceWorker accepts onset_detector kwarg
# ---------------------------------------------------------------

def test_T01_worker_accepts_onset_detector_kwarg():
    from voxkit.ui.inference_worker import InferenceWorker
    audio = np.zeros(_SR, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    worker = InferenceWorker(audio, model, onset_detector=None)
    assert worker is not None


# ---------------------------------------------------------------
# T02  Default (None) preserves stub behaviour
# ---------------------------------------------------------------

def test_T02_default_stub_empty_audio_returns_no_onsets():
    from voxkit.ui.inference_worker import InferenceWorker
    audio = np.zeros(0, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    worker = InferenceWorker(audio, model)
    assert worker._detect_onsets(audio) == []


def test_T02b_default_stub_nonempty_audio_returns_zero():
    from voxkit.ui.inference_worker import InferenceWorker
    audio = np.zeros(_SR, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    worker = InferenceWorker(audio, model)
    assert worker._detect_onsets(audio) == [0.0]


# ---------------------------------------------------------------
# T03  With detector, _detect_onsets delegates
# ---------------------------------------------------------------

def test_T03_detect_onsets_delegates_to_detector():
    from voxkit.ui.inference_worker import InferenceWorker
    audio = np.zeros(_SR, dtype=np.float32)
    spy = _SpyDetector(returns=[0.1, 0.5])
    model = _FakeModel()
    model.prepare(audio)
    worker = InferenceWorker(audio, model, onset_detector=spy)
    result = worker._detect_onsets(audio)
    assert result == [0.1, 0.5]
    assert len(spy.calls) == 1
    assert np.array_equal(spy.calls[0], audio)


# ---------------------------------------------------------------
# T04  Real OnsetDetector accepted
# ---------------------------------------------------------------

def test_T04_real_onset_detector_accepted():
    from voxkit.ui.inference_worker import InferenceWorker
    from voxkit.dsp.onsets import OnsetDetector
    audio = np.zeros(_SR, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    detector = OnsetDetector(sample_rate=_SR)
    worker = InferenceWorker(audio, model, onset_detector=detector)
    assert worker is not None


# ---------------------------------------------------------------
# T05  End-to-end: transient audio → events emitted
# ---------------------------------------------------------------

def test_T05_end_to_end_transient_produces_events():
    from voxkit.ui.inference_worker import InferenceWorker
    from voxkit.dsp.onsets import OnsetDetector

    audio = _transient(t_s=1.0)
    model = _FakeModel()
    model.prepare(audio)
    detector = OnsetDetector(sample_rate=_SR)

    completed: list = []
    failed: list = []
    worker = InferenceWorker(audio, model, onset_detector=detector)
    worker.completed.connect(lambda evts: completed.extend(evts))
    worker.failed.connect(lambda msg: failed.append(msg))
    worker.start()
    done = worker.wait_for_completion(timeout=5.0)

    assert done
    assert failed == [], f"Worker failed: {failed}"
    assert len(completed) >= 1
    event = completed[0]
    assert isinstance(event.t, float)
    assert abs(event.t - 1.0) < 0.05  # within 50 ms of the planted transient


# ---------------------------------------------------------------
# T06  run_pipeline accepts detect_onsets kwarg
# ---------------------------------------------------------------

def test_T06_run_pipeline_accepts_detect_onsets_kwarg():
    from voxkit.ui.inference_pipeline import run_pipeline
    audio = np.zeros(_SR, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    result = run_pipeline(audio, model, detect_onsets=None)
    assert result is not None


# ---------------------------------------------------------------
# T07  run_pipeline default stub behaviour
# ---------------------------------------------------------------

def test_T07_run_pipeline_default_stub_nonempty():
    from voxkit.ui.inference_pipeline import run_pipeline
    audio = np.zeros(_SR, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    result = run_pipeline(audio, model)
    assert not result.cancelled
    # stub returns [0.0] → one event
    assert len(result.events) == 1


def test_T07b_run_pipeline_default_stub_empty():
    from voxkit.ui.inference_pipeline import run_pipeline
    audio = np.zeros(0, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    result = run_pipeline(audio, model)
    assert not result.cancelled
    assert result.events == []


# ---------------------------------------------------------------
# T08  run_pipeline uses provided detect_onsets callable
# ---------------------------------------------------------------

def test_T08_run_pipeline_uses_provided_callable():
    from voxkit.ui.inference_pipeline import run_pipeline
    audio = np.zeros(_SR, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    calls: list = []

    def _spy_detect(a):
        calls.append(a)
        return [0.5]

    result = run_pipeline(audio, model, detect_onsets=_spy_detect)
    assert len(calls) == 1
    assert len(result.events) == 1
    assert result.events[0].t == pytest.approx(0.5)


# ---------------------------------------------------------------
# T09  Two onsets → two events
# ---------------------------------------------------------------

def test_T09_two_onsets_two_events():
    from voxkit.ui.inference_worker import InferenceWorker
    from voxkit.dsp.onsets import OnsetDetector

    audio = _two_transients(0.5, 2.0)
    model = _FakeModel()
    model.prepare(audio)
    detector = OnsetDetector(sample_rate=_SR)

    completed: list = []
    worker = InferenceWorker(audio, model, onset_detector=detector)
    worker.completed.connect(lambda evts: completed.extend(evts))
    worker.start()
    worker.wait_for_completion(timeout=5.0)

    assert len(completed) == 2
    times = sorted(e.t for e in completed)
    assert abs(times[0] - 0.5) < 0.05
    assert abs(times[1] - 2.0) < 0.05


# ---------------------------------------------------------------
# T10  Empty audio with real detector → empty events
# ---------------------------------------------------------------

def test_T10_empty_audio_real_detector_no_events():
    from voxkit.ui.inference_worker import InferenceWorker
    from voxkit.dsp.onsets import OnsetDetector

    audio = np.zeros(0, dtype=np.float32)
    model = _FakeModel()
    model.prepare(audio)
    detector = OnsetDetector(sample_rate=_SR)

    completed: list = []
    failed: list = []
    worker = InferenceWorker(audio, model, onset_detector=detector)
    worker.completed.connect(lambda evts: completed.extend(evts))
    worker.failed.connect(lambda msg: failed.append(msg))
    worker.start()
    worker.wait_for_completion(timeout=5.0)

    assert failed == []
    assert completed == []
