# SPDX-License-Identifier: GPL-3.0-or-later
"""Energy-based onset detector with click-guard suppression (Component 4)."""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class AudioContainsNonFinite(Exception):
    pass


# ---------------------------------------------------------------
# OnsetDetector
# ---------------------------------------------------------------

_REQUIRED_FS = 16_000
_CLICK_GUARD_MS = 15.0   # ±15 ms suppression window around each click


class OnsetDetector:
    """Energy-flux onset detector.

    Uses non-overlapping 5 ms frames (hop = fs // 200) so detected
    onset times are within ±5 ms of the true onset position.
    """

    def __init__(self, sample_rate: int) -> None:
        if sample_rate != _REQUIRED_FS:
            raise ValueError(
                f"OnsetDetector requires sample_rate={_REQUIRED_FS}; "
                f"got {sample_rate}"
            )
        self._fs = sample_rate
        self._hop = sample_rate // 200   # 5 ms at 16 kHz = 80 samples

    # ------------------------------------------------------------------
    # Public: detect
    # ------------------------------------------------------------------

    def detect(
        self,
        audio: np.ndarray,
        click_track: list[float] | None = None,
    ) -> list[float]:
        """Return onset times in seconds.

        Parameters
        ----------
        audio:
            Mono float32 audio at 16 kHz.
        click_track:
            Optional list of click times (seconds) to suppress.
            Onsets within ±15 ms of a click are removed.
        """
        if audio.ndim != 1:
            raise ValueError("OnsetDetector requires mono (1-D) audio")
        if not np.all(np.isfinite(audio)):
            raise AudioContainsNonFinite("Audio contains NaN or Inf")

        hop = self._hop
        n = len(audio)
        if n < hop:
            return []

        n_frames = n // hop
        # Non-overlapping energy frames → onset time = frame_start.
        energies = np.array([
            float(np.sum(audio[i * hop: (i + 1) * hop].astype(np.float64) ** 2))
            for i in range(n_frames)
        ])

        # Positive first-difference onset detection function.
        odf = np.diff(energies, prepend=energies[0])
        odf = np.maximum(odf, 0.0)

        peak = odf.max()
        if peak == 0:
            return []
        threshold = 0.2 * peak

        min_gap = max(1, int(0.05 * self._fs / hop))  # 50 ms in frames

        raw_onsets_s: list[float] = []
        i = 0
        while i < len(odf):
            if odf[i] > threshold:
                w_end = min(i + min_gap, len(odf))
                peak_idx = i + int(np.argmax(odf[i:w_end]))
                raw_onsets_s.append(float(peak_idx * hop / self._fs))
                i = peak_idx + min_gap
            else:
                i += 1

        return self._suppress_clicks(raw_onsets_s, click_track)

    # ------------------------------------------------------------------
    # Public: click_guard_fire_rate
    # ------------------------------------------------------------------

    def click_guard_fire_rate(
        self,
        click_positions: list[int],
        raw_onsets: list[int],
        window_bars: int,
        bpm: float,
        sample_rate: int,
    ) -> float:
        """Fraction of click positions (within the most recent window_bars bars)
        that had a raw onset within ±15 ms."""
        if not click_positions:
            return 0.0

        bar_dur_s = 60.0 / bpm
        window_samples = int(window_bars * bar_dur_s * sample_rate)
        end_sample = max(click_positions)
        start_sample = end_sample - window_samples

        recent = [c for c in click_positions if c >= start_sample]
        if not recent:
            return 0.0

        guard = int(0.015 * sample_rate)   # ±15 ms in samples
        hits = sum(
            1 for c in recent
            if any(abs(c - o) <= guard for o in raw_onsets)
        )
        return hits / len(recent)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _suppress_clicks(
        self,
        onsets_s: list[float],
        click_track: list[float] | None,
    ) -> list[float]:
        if not click_track:
            return onsets_s
        guard = _CLICK_GUARD_MS / 1000.0
        return [
            t for t in onsets_s
            if not any(abs(t - c) <= guard for c in click_track)
        ]
