# SPDX-License-Identifier: GPL-3.0-or-later
"""Tempo grid construction and event quantization (§11 Component 8)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from voxkit.core.session import Event  # re-export so callers can import from here


# ---------------------------------------------------------------
# Supported subdivisions
# ---------------------------------------------------------------

_SUPPORTED_SUBDIVISIONS = ("1/4", "1/8", "1/16", "1/32")


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class EmptyGrid(Exception):
    pass


# ---------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------

def build_grid(
    bpm: float,
    time_signature: tuple[int, int],
    bars: int,
    subdivision: str,
) -> np.ndarray:
    """Return an array of grid positions in seconds.

    Parameters
    ----------
    bpm           : beats per minute (> 0)
    time_signature: (numerator, denominator) e.g. (4, 4) or (3, 4)
    bars          : number of bars (>= 0)
    subdivision   : one of "1/4", "1/8", "1/16", "1/32"
    """
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    if bars < 0:
        raise ValueError(f"bars must be >= 0, got {bars}")
    if subdivision not in _SUPPORTED_SUBDIVISIONS:
        supported = ", ".join(_SUPPORTED_SUBDIVISIONS)
        raise ValueError(
            f"subdivision {subdivision!r} not supported; "
            f"try {supported}"
        )
    if bars == 0:
        return np.zeros(0, dtype=np.float64)

    numerator, denominator = time_signature
    beat_duration = 60.0 / bpm                          # seconds per beat
    bar_duration = numerator * beat_duration             # seconds per bar
    grid_n = int(subdivision.split("/")[1])              # e.g. 16 for "1/16"
    steps_per_bar = numerator * grid_n // denominator   # integer subdivision count

    step_duration = bar_duration / steps_per_bar
    total_steps = bars * steps_per_bar
    # Integer-scaled multiply avoids float accumulation duplicates (T26).
    return np.arange(total_steps, dtype=np.float64) * step_duration


# ---------------------------------------------------------------
# Nearest grid index (pure helper — Tidy First extraction, T24)
# ---------------------------------------------------------------

def _nearest_grid_index(t: float, grid: np.ndarray) -> int:
    """Return index of the grid position nearest to t.

    Tie-breaking: snap to the EARLIER (lower-index) position per v0.11
    panel decision (T33) — vocal-percussion players tend to play behind
    the beat, so snapping backward preserves the groove.
    """
    if t <= grid[0]:
        return 0
    if t >= grid[-1]:
        return len(grid) - 1
    idx = int(np.searchsorted(grid, t, side="left"))
    # grid[idx-1] < t <= grid[idx]
    d_left = t - grid[idx - 1]
    d_right = grid[idx] - t
    # <= means ties snap to the earlier (left) position.
    if d_left <= d_right:
        return idx - 1
    return idx


# ---------------------------------------------------------------
# Single-event quantization
# ---------------------------------------------------------------

def quantize_time(t: float, grid: np.ndarray, strength: float) -> float:
    """Quantize a single time value toward the nearest grid position.

    Parameters
    ----------
    t        : raw event time in seconds
    grid     : sorted array of grid positions
    strength : 0 = no movement, 1 = full snap to grid; clamped to [0, 1].
               NaN raises ValueError.
    """
    if math.isnan(strength):
        raise ValueError("strength must not be NaN")
    strength = float(np.clip(strength, 0.0, 1.0))
    idx = _nearest_grid_index(t, grid)
    target = float(grid[idx])
    return t + strength * (target - t)


# ---------------------------------------------------------------
# Batch quantization
# ---------------------------------------------------------------

def quantize_events(
    events: list[Event],
    grid: np.ndarray,
    strength: float,
) -> list[Event]:
    """Quantize a list of events, preserving order, class_id, and score.

    Raises EmptyGrid if the grid is empty and events is non-empty.
    """
    if not events:
        return []
    if len(grid) == 0:
        raise EmptyGrid(
            "Cannot quantize events with an empty grid (0-bar session); "
            "the quantize button would appear to do nothing"
        )
    return [
        dataclasses.replace(e, t=quantize_time(e.t, grid, strength))
        for e in events
    ]


# ---------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------

def quantize_session(session):
    """Quantize all events in a Session using its bpm, time_signature,
    bars, quantize_grid, and quantize_strength. Returns a new Session."""
    grid = build_grid(
        bpm=session.bpm,
        time_signature=(
            session.time_signature.numerator,
            session.time_signature.denominator,
        ),
        bars=session.bars,
        subdivision=session.quantize_grid,
    )
    quantized = quantize_events(
        session.events, grid, strength=session.quantize_strength
    )
    return dataclasses.replace(session, events=quantized)
