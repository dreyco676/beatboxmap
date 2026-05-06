# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-Python inference pipeline driver (Q76). No Qt imports allowed."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PipelineResult:
    events: list
    cancelled: bool
    audio: np.ndarray


def _default_detect_onsets(audio: np.ndarray) -> list[float]:
    if len(audio) == 0:
        return []
    return [0.0]


def run_pipeline(
    audio: np.ndarray,
    model,
    *,
    on_phase=None,
    on_progress=None,
    cancel_flag: threading.Event | None = None,
    detect_onsets=None,
) -> PipelineResult:
    """Run the three-phase inference pipeline and return a PipelineResult.

    Phases: onset → embedding → classify.
    Checks cancel_flag between and within phases; returns immediately when set.
    No Qt imports are made here — callers may block Qt if they wish.
    """

    def _cancelled() -> bool:
        return cancel_flag is not None and cancel_flag.is_set()

    def _notify(p: str) -> None:
        if on_phase is not None:
            on_phase(p)

    if _cancelled():
        return PipelineResult(events=[], cancelled=True, audio=audio)

    _notify("onset")
    onsets = detect_onsets(audio) if detect_onsets is not None else _default_detect_onsets(audio)

    if _cancelled():
        return PipelineResult(events=[], cancelled=True, audio=audio)

    _notify("embedding")
    embeddings: list = []
    n = max(len(onsets), 1)
    for i, onset_t in enumerate(onsets):
        if _cancelled():
            return PipelineResult(events=[], cancelled=True, audio=audio)
        emb = model.embed(onset_t)
        embeddings.append(emb)
        if on_progress is not None:
            on_progress(0.5 * (i + 1) / n)

    if _cancelled():
        return PipelineResult(events=[], cancelled=True, audio=audio)

    _notify("classify")
    if on_progress is not None:
        on_progress(1.0)
    events = list(model.predict(embeddings))

    return PipelineResult(events=events, cancelled=False, audio=audio)
