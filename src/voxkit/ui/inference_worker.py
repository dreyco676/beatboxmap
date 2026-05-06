# SPDX-License-Identifier: GPL-3.0-or-later
"""InferenceWorker — Qt-coupled thread adapter for the inference pipeline (Q76)."""

from __future__ import annotations

import threading

import numpy as np


class WorkerAlreadyStarted(Exception):
    pass


class _Signal:
    """Minimal signal-like object: connect callbacks, emit to all of them."""

    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for cb in self._callbacks:
            cb(*args)


class InferenceWorker:
    """Runs the three-phase inference pipeline on a dedicated thread.

    Signals (connect via .connect(callable)):
        phase_changed(str)   — "onset" | "embedding" | "classify"
        progress(float)      — values in [0.0, 1.0]
        completed(list)      — list of events on success
        failed(str)          — error message on unhandled exception
        cancelled()          — emitted instead of completed/failed after cancel
    """

    def __init__(self, audio: np.ndarray, model, *, onset_detector=None) -> None:
        self._audio = audio
        self._model = model
        self._onset_detector = onset_detector
        self._cancel_flag = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

        self.phase_changed = _Signal()
        self.progress = _Signal()
        self.completed = _Signal()
        self.failed = _Signal()
        self.cancelled = _Signal()

    # ------------------------------------------------------------------
    # Overridable hooks (for testing / subclassing)
    # ------------------------------------------------------------------

    def _detect_onsets(self, audio: np.ndarray) -> list[float]:
        if self._onset_detector is not None:
            return self._onset_detector.detect(audio)
        if len(audio) == 0:
            return []
        return [0.0]

    def _embed_one(self, onset_t: float):
        return self._model.embed(onset_t)

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            if self._cancel_flag.is_set():
                self.cancelled.emit()
                return

            self.phase_changed.emit("onset")
            onsets = self._detect_onsets(self._audio)

            if self._cancel_flag.is_set():
                self.cancelled.emit()
                return

            self.phase_changed.emit("embedding")
            embeddings: list = []
            n = max(len(onsets), 1)
            for i, onset_t in enumerate(onsets):
                if self._cancel_flag.is_set():
                    self.cancelled.emit()
                    return
                emb = self._embed_one(onset_t)
                embeddings.append(emb)
                self.progress.emit(0.5 * (i + 1) / n)

            if self._cancel_flag.is_set():
                self.cancelled.emit()
                return

            self.phase_changed.emit("classify")
            self.progress.emit(1.0)
            events = list(self._model.predict(embeddings))
            self.completed.emit(events)

        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self._done.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise WorkerAlreadyStarted(
                "InferenceWorker.start() called twice; create a new instance to re-run"
            )
        t = threading.Thread(target=self._run, name="InferenceWorker", daemon=True)
        self._thread = t
        t.start()

    def cancel(self) -> None:
        self._cancel_flag.set()

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Block until the worker finishes. Returns True if done, False on timeout."""
        return self._done.wait(timeout=timeout)
