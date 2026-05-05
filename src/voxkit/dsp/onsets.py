# SPDX-License-Identifier: GPL-3.0-or-later
"""Energy-based onset detector (Component 4 stub; used by T32 cross-component test)."""

from __future__ import annotations

import numpy as np


class OnsetDetector:
    """Simple energy-flux onset detector."""

    def __init__(self, sample_rate: int) -> None:
        self._fs = sample_rate

    def detect(
        self,
        audio: np.ndarray,
        click_track: list[float] | None = None,
    ) -> list[float]:
        """Return onset times in seconds.

        click_track: optional list of known click times (seconds) to suppress
        false positives caused by residual bleed energy.
        """
        hop = max(1, self._fs // 100)  # 10 ms
        frame_size = min(512, len(audio))

        if len(audio) < frame_size:
            return []

        n_frames = (len(audio) - frame_size) // hop + 1
        energies = np.array([
            np.sum(audio[i * hop : i * hop + frame_size].astype(np.float64) ** 2)
            for i in range(n_frames)
        ])

        # Positive first difference = onset strength function.
        odf = np.diff(energies, prepend=energies[0])
        odf = np.maximum(odf, 0.0)

        peak = odf.max()
        if peak == 0:
            return []
        threshold = 0.2 * peak

        min_gap = max(1, int(0.05 * self._fs / hop))  # 50 ms minimum between onsets

        onsets: list[float] = []
        i = 0
        while i < len(odf):
            if odf[i] > threshold:
                # Find the local peak within the minimum-gap window.
                w_end = min(i + min_gap, len(odf))
                peak_idx = i + int(np.argmax(odf[i:w_end]))
                t = float(peak_idx * hop / self._fs)
                onsets.append(t)
                i = peak_idx + min_gap
            else:
                i += 1

        return onsets
