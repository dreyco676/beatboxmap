# SPDX-License-Identifier: GPL-3.0-or-later
"""CalibrationManager: orchestrates calibration sample collection and commit (Q71, §5.6)."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from voxkit.classifier.classifier import CalibrationRejected
from voxkit.telemetry.local_sink import build_event

_SILENCE_RMS_THRESHOLD = 1e-4
_CLIP_FRACTION_THRESHOLD = 0.001
_NEAR_DUPLICATE_COSINE = 0.999


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class IncompleteCalibration(Exception):
    pass


class SessionAlreadyCommitted(Exception):
    pass


class SilentSample(Exception):
    pass


class AllSamplesSilent(Exception):
    pass


# ---------------------------------------------------------------
# CommitHandle
# ---------------------------------------------------------------

@dataclass
class CommitHandle:
    classifier: Any
    diagnostics: dict


# ---------------------------------------------------------------
# CalibrationSession
# ---------------------------------------------------------------

class CalibrationSession:
    def __init__(
        self,
        classes: tuple[str, ...],
        embedding_dim: int,
        required_per_class: int = 3,
    ) -> None:
        self.classes = classes
        self._embedding_dim = embedding_dim
        self.required_per_class = required_per_class
        self._buckets: dict[str, list[np.ndarray]] = {c: [] for c in classes}
        self.state: str = "open"
        self.near_duplicate_count: int = 0
        self._sample_rms: list[float | None] = []

    def count_for(self, cls: str) -> int:
        return len(self._buckets[cls])

    def is_complete(self) -> bool:
        return all(len(self._buckets[c]) >= self.required_per_class for c in self.classes)

    def has_all_silent(self) -> bool:
        total = sum(len(self._buckets[c]) for c in self.classes)
        rms_provided = [r for r in self._sample_rms if r is not None]
        if total == 0 or len(rms_provided) != total:
            return False
        return all(r < _SILENCE_RMS_THRESHOLD for r in rms_provided)

    def add_sample(
        self,
        class_id: str,
        embedding: np.ndarray,
        *,
        source_audio_rms: float | None = None,
        source_audio_clipped_fraction: float | None = None,
        skip_silence_check: bool = False,
    ) -> None:
        if self.state == "committed":
            raise SessionAlreadyCommitted("Cannot add samples to a committed session")
        if class_id not in self.classes:
            raise ValueError(f"Unknown class: {class_id!r}")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("embedding must be finite (no NaN or Inf)")
        if len(embedding) != self._embedding_dim:
            raise ValueError(
                f"embedding dim {len(embedding)} != expected dim {self._embedding_dim}"
            )
        if not skip_silence_check and source_audio_rms is not None:
            if source_audio_rms < _SILENCE_RMS_THRESHOLD:
                raise SilentSample(
                    f"Sample is silent (rms={source_audio_rms:.2e}); "
                    "check that the microphone is not muted"
                )
        if (source_audio_clipped_fraction is not None and
                source_audio_clipped_fraction > _CLIP_FRACTION_THRESHOLD):
            warnings.warn(
                f"Sample audio has clip fraction {source_audio_clipped_fraction:.3f}; "
                "consider re-recording at a lower level",
                UserWarning,
                stacklevel=2,
            )

        # Near-duplicate check (log only, no warning)
        bucket = self._buckets[class_id]
        if bucket:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                emb_unit = embedding / norm
                for prev in bucket:
                    prev_norm = np.linalg.norm(prev)
                    if prev_norm > 0:
                        cosine = float(np.dot(emb_unit, prev / prev_norm))
                        if cosine > _NEAR_DUPLICATE_COSINE:
                            self.near_duplicate_count += 1
                            break

        bucket.append(embedding.copy())
        self._sample_rms.append(source_audio_rms)

    def get_embeddings_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        embeddings, labels = [], []
        for cls in self.classes:
            for emb in self._buckets[cls]:
                embeddings.append(emb)
                labels.append(cls)
        return np.array(embeddings), np.array(labels)


# ---------------------------------------------------------------
# CalibrationManager
# ---------------------------------------------------------------

class CalibrationManager:
    def __init__(
        self,
        classifier: Any,
        *,
        calibration_weight: float = 1.0,
        telemetry: Any | None = None,
    ) -> None:
        self._classifier = classifier
        self._calibration_weight = calibration_weight
        self._telemetry = telemetry

    def start_session(self) -> CalibrationSession:
        classes = self._classifier.taxonomy.classes
        dim = self._classifier.embedding_dim
        return CalibrationSession(classes=classes, embedding_dim=dim)

    def commit(self, session: CalibrationSession) -> CommitHandle:
        if not session.is_complete():
            raise IncompleteCalibration(
                f"Each class needs {session.required_per_class} samples; "
                f"session has: {[(c, session.count_for(c)) for c in session.classes]}"
            )

        if session.has_all_silent():
            raise AllSamplesSilent(
                "All calibration samples are silent; check that the microphone is active"
            )

        cal_emb, cal_labels = session.get_embeddings_and_labels()
        snapshot = self._classifier.snapshot()

        try:
            self._classifier.fit_with_calibration(
                calibration_embeddings=cal_emb,
                calibration_labels=cal_labels,
                calibration_weight=self._calibration_weight,
            )
        except CalibrationRejected as exc:
            self._classifier.restore(snapshot)
            session.state = "rejected"
            self._emit_safe(build_event(
                event="calibration_overfit_guard_triggered",
                details=dict(exc.diagnostics),
            ))
            raise

        session.state = "committed"
        self._emit_safe(build_event(
            event="calibration_committed",
            details={"calibration_weight": self._calibration_weight},
        ))
        return CommitHandle(
            classifier=self._classifier,
            diagnostics={"calibration_weight": self._calibration_weight},
        )

    def _emit_safe(self, event: dict) -> None:
        if self._telemetry is None:
            return
        try:
            self._telemetry.emit(event)
        except Exception as e:
            sys.stderr.write(
                f"WARNING: telemetry/diagnostic write failed: {e}\n"
            )
