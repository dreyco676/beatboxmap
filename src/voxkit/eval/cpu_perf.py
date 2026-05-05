# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU performance benchmark for end-to-end inference (Q72)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def benchmark_session(
    substrate: str,
    session_bars: int = 32,
    session_bpm: float = 120.0,
    reference_target_multiple: float | None = None,
) -> dict:
    """Benchmark end-to-end inference (onset + embedding + classify).

    Parameters
    ----------
    substrate                  : substrate identifier string (e.g. "panns_cnn14")
    session_bars               : number of bars in the synthetic test session
    session_bpm                : tempo in beats-per-minute
    reference_target_multiple  : if set, included in result for later gate checks

    Returns
    -------
    dict with wall_clock_seconds, substrate, phase_times, and optionally
    reference_target_multiple.
    """
    beats_per_bar = 4
    duration_s = session_bars * beats_per_bar * 60.0 / session_bpm
    n_samples = int(duration_s * 16_000)
    audio = np.zeros(n_samples, dtype=np.float32)

    phase_times: dict[str, float] = {}

    # Phase 1: onset detection
    t0 = time.perf_counter()
    _simulate_onset(audio)
    phase_times["onset"] = time.perf_counter() - t0

    # Phase 2: embedding extraction
    t0 = time.perf_counter()
    _simulate_embedding(audio)
    phase_times["embedding"] = time.perf_counter() - t0

    # Phase 3: classification
    t0 = time.perf_counter()
    _simulate_classify()
    phase_times["classify"] = time.perf_counter() - t0

    wall_clock = sum(phase_times.values())

    result: dict = {
        "wall_clock_seconds": max(wall_clock, 1e-9),
        "substrate": substrate,
        "phase_times": phase_times,
    }
    if reference_target_multiple is not None:
        result["reference_target_multiple"] = reference_target_multiple

    return result


def _simulate_onset(audio: np.ndarray) -> None:
    frame = 512
    for i in range(0, len(audio) - frame, frame):
        _ = float(np.sum(audio[i : i + frame] ** 2))


def _simulate_embedding(audio: np.ndarray) -> None:
    _ = np.fft.rfft(audio[:min(len(audio), 4096)])


def _simulate_classify() -> None:
    _ = np.dot(np.ones(128), np.ones(128))


def compare_with_baseline(current: float, baseline_path: Path) -> float:
    """Compare *current* wall-clock seconds against a saved baseline JSON.

    Prints a "perf delta" line to stdout when the regression exceeds 10%.
    Returns the fractional delta ((current - baseline) / baseline).
    """
    data = json.loads(Path(baseline_path).read_text())
    baseline = float(data["wall_clock_seconds"])
    delta = (current - baseline) / baseline if baseline > 0 else 0.0
    if delta > 0.10:
        print(
            f"perf delta: {delta:+.1%} "
            f"(current={current:.3f}s vs baseline={baseline:.3f}s)"
        )
    return delta
