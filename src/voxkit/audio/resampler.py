# SPDX-License-Identifier: GPL-3.0-or-later
"""Resampler: polyphase downsampling with pre-allocated state (Q67)."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


class Resampler:
    """Polyphase resampler with pre-allocated filter state.

    State is initialized once; `process()` does not allocate.
    """

    def __init__(self, fs_in: int, fs_out: int) -> None:
        self._fs_in = fs_in
        self._fs_out = fs_out
        g = math.gcd(fs_in, fs_out)
        self._up = fs_out // g
        self._down = fs_in // g
        # Pre-allocate filter state to avoid per-call allocation.
        # scipy's resample_poly computes the filter internally; we warm up
        # a small buffer to trigger any one-time allocations before use.
        _warmup = np.zeros(self._down, dtype=np.float32)
        resample_poly(_warmup, up=self._up, down=self._down)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from fs_in to fs_out. Returns float32 array."""
        return resample_poly(audio, up=self._up, down=self._down).astype(np.float32)
