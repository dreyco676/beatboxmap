# SPDX-License-Identifier: GPL-3.0-or-later
"""Concrete Model: wires EmbeddingExtractor + Classifier for InferenceWorker."""

from __future__ import annotations

import numpy as np

from voxkit.core.session import Event


class NotPreparedError(RuntimeError):
    pass


class Model:
    def __init__(self, extractor, classifier, sample_rate: int = 16_000) -> None:
        self._extractor = extractor
        self._classifier = classifier
        self._sample_rate = sample_rate
        self.taxonomy = classifier.taxonomy
        self._audio: np.ndarray | None = None
        self._pending_onsets: list[float] = []

    def prepare(self, audio: np.ndarray) -> None:
        self._audio = audio
        self._pending_onsets = []

    def embed(self, onset_t: float) -> np.ndarray:
        if self._audio is None:
            raise NotPreparedError("Call model.prepare(audio) before inference.")
        self._pending_onsets.append(onset_t)
        embeddings = self._extractor.extract_at_onsets(
            self._audio, [onset_t], self._sample_rate
        )
        return embeddings[0]

    def predict(self, embeddings: list) -> list[Event]:
        if not embeddings:
            self._pending_onsets.clear()
            return []
        X = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
        class_results = self._classifier.predict(X)
        events = [
            Event(t=t, class_id=cls, score=float(score))
            for (cls, score), t in zip(class_results, self._pending_onsets)
        ]
        self._pending_onsets.clear()
        return events
