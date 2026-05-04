# SPDX-License-Identifier: GPL-3.0-or-later
"""Drop-rate monitor with sliding window UX hook (Q67)."""

from __future__ import annotations

import collections
from typing import Callable


class DropRateMonitor:
    """Tracks buffer drop rate over a rolling window; calls handler when threshold exceeded."""

    def __init__(
        self,
        window_seconds: float,
        threshold: float,
        handler: Callable[[], None],
    ) -> None:
        self._window = window_seconds
        self._threshold = threshold
        self._handler = handler
        self._events: collections.deque[tuple[float, bool]] = collections.deque()
        self._warned = False

    def observe(self, *, dropped: bool, now: float) -> None:
        self._events.append((now, dropped))
        # Evict events outside the rolling window.
        cutoff = now - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

        # Only evaluate once the window is ≥90% populated.
        # A partially-filled window produces artificially high drop rates
        # that would false-trigger the warning.
        if len(self._events) < 2:
            return
        span = self._events[-1][0] - self._events[0][0]
        if span < 0.9 * self._window:
            return

        total = len(self._events)
        n_dropped = sum(1 for _, d in self._events if d)
        rate = n_dropped / total
        if rate > self._threshold and not self._warned:
            self._warned = True
            self._handler()
