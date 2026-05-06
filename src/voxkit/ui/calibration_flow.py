# SPDX-License-Identifier: GPL-3.0-or-later
"""CalibrationFlow: audio → embedding → calibration session → commit."""

from __future__ import annotations

import numpy as np


class NotReadyForPreview(RuntimeError):
    pass


class CalibrationFlow:
    """Orchestrates the end-to-end calibration loop.

    audio snippet → EmbeddingExtractor → CalibrationSession → CalibrationManager.commit()

    Usage::

        flow = CalibrationFlow(extractor, manager, classifier)
        for cls in taxonomy.classes:
            audio = record_snippet()
            flow.add_sample(cls, audio)
        if flow.can_preview():
            class_id, score = flow.preview(new_audio)
        handle = flow.commit()
    """

    def __init__(self, extractor, manager, classifier) -> None:
        self._extractor = extractor
        self._manager = manager
        self._classifier = classifier
        self._session = manager.start_session()

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def add_sample(self, class_id: str, audio: np.ndarray) -> None:
        """Extract embedding from audio and add to the calibration session."""
        if len(audio) == 0:
            raise ValueError("audio must be non-empty")
        sr = self._extractor.required_sample_rate
        center_t = len(audio) / (2 * sr)
        embs = self._extractor.extract_at_onsets(audio, [center_t], sr)
        embedding = embs[0]
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        self._session.add_sample(class_id, embedding, source_audio_rms=rms)

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------

    def can_preview(self) -> bool:
        """True once every class has at least one sample."""
        return all(
            self._session.count_for(c) >= 1 for c in self._session.classes
        )

    def preview(self, audio: np.ndarray) -> tuple[str, float]:
        """Return (class_id, score) for a single audio snippet.

        Raises NotReadyForPreview if can_preview() is False.
        """
        if not self.can_preview():
            raise NotReadyForPreview(
                "Record at least one sample of each sound to enable live preview."
            )
        if len(audio) == 0:
            raise ValueError("audio must be non-empty")
        sr = self._extractor.required_sample_rate
        center_t = len(audio) / (2 * sr)
        embs = self._extractor.extract_at_onsets(audio, [center_t], sr)
        embedding = embs[0]
        results = self._classifier.predict(embedding[np.newaxis, :])
        class_id, score = results[0]
        return class_id, score

    # ------------------------------------------------------------------
    # Status and commit
    # ------------------------------------------------------------------

    def status(self) -> dict[str, int]:
        """Return {class_id: sample_count} for all classes."""
        return {c: self._session.count_for(c) for c in self._session.classes}

    def commit(self):
        """Fit classifier with calibration data.

        Returns CommitHandle on success.
        Raises IncompleteCalibration or CalibrationRejected.
        """
        return self._manager.commit(self._session)

    def record_abandon_event(self) -> None:
        """Emit a calibration_abandoned telemetry event with per-class counts."""
        from voxkit.telemetry.local_sink import build_event
        self._manager._emit_safe(build_event(
            event="calibration_abandoned",
            details={"counts": self.status()},
        ))
