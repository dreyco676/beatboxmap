# SPDX-License-Identifier: GPL-3.0-or-later
"""Resampler worker budget computation (Q67)."""

from __future__ import annotations


def compute_budget_ms(buffer_duration_ms: float) -> float:
    """Return the resampler-worker time budget in milliseconds.

    Rules (spec §11 Component 2 / Q67):
      buffer < 3 ms  → 3 × buffer   (relax for tiny buffers)
      buffer > 30 ms → 1.5 × buffer (tighten for large buffers)
      otherwise      → 10 ms        (floor for typical 5–10 ms buffers)
    """
    if buffer_duration_ms < 3.0:
        return 3.0 * buffer_duration_ms
    if buffer_duration_ms > 30.0:
        return 1.5 * buffer_duration_ms
    return 10.0
