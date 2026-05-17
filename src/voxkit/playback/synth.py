# SPDX-FileCopyrightText: 2026 John Hogue
# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic drum sounds and MIDI-event-to-audio renderer for preview playback."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------
# Drum sound generators
# ---------------------------------------------------------------

def _kick(sr: int) -> np.ndarray:
    dur = 0.30
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    freq = 120.0 * np.exp(-t * 22) + 45.0   # pitch sweep 120 → 45 Hz
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 10)
    transient = np.exp(-t * 250) * 0.3       # click attack
    return ((np.sin(phase) * env + transient) * 0.82).astype(np.float32)


def _snare(sr: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    dur = 0.18
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 32)
    noise = rng.standard_normal(len(t)) * np.exp(-t * 20)
    return ((body * 0.35 + noise * 0.65) * 0.68).astype(np.float32)


def _closed_hat(sr: int) -> np.ndarray:
    rng = np.random.default_rng(2)
    dur = 0.04
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    noise = rng.standard_normal(len(t)) * np.exp(-t * 90)
    return (noise * 0.42).astype(np.float32)


def _open_hat(sr: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    dur = 0.28
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    noise = rng.standard_normal(len(t)) * np.exp(-t * 13)
    return (noise * 0.44).astype(np.float32)


_GENERATORS = {
    "kick":       _kick,
    "snare":      _snare,
    "closed_hat": _closed_hat,
    "open_hat":   _open_hat,
}


# ---------------------------------------------------------------
# Click track generator (count-in)
# ---------------------------------------------------------------

def generate_click_track(
    bpm: float,
    bars: int = 1,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Return a float32 audio buffer with metronome clicks at *bpm*.

    Beat 1 of each bar uses a higher-pitched accent click.
    """
    beat_samples = int(sample_rate * 60.0 / bpm)
    n_beats = bars * 4
    out = np.zeros(beat_samples * n_beats, dtype=np.float32)
    click_n = int(0.013 * sample_rate)
    t_c = np.arange(click_n, dtype=np.float64) / sample_rate
    env = np.exp(-t_c * 280)
    for beat in range(n_beats):
        freq = 1100.0 if beat % 4 == 0 else 750.0
        click = (np.sin(2 * np.pi * freq * t_c) * env).astype(np.float32)
        pos = beat * beat_samples
        end = min(len(out), pos + click_n)
        out[pos:end] += click[: end - pos]
    return out


# ---------------------------------------------------------------
# MIDI event renderer
# ---------------------------------------------------------------

def render_events(
    events: list,
    bpm: float,
    bars: int,
    sample_rate: int = 16_000,
    muted: "set[str] | None" = None,
) -> np.ndarray:
    """Render *events* to a float32 audio buffer using synthesised drum sounds.

    Events whose ``class_id`` is in *muted* are skipped.
    The output is normalised to ±0.9 to prevent clipping.
    """
    if muted is None:
        muted = set()

    duration_s = bars * 4 * 60.0 / bpm
    n_samples = int(duration_s * sample_rate)
    out = np.zeros(n_samples, dtype=np.float32)

    sounds = {k: fn(sample_rate) for k, fn in _GENERATORS.items()}

    for ev in events:
        cls = ev.class_id
        if cls in muted or cls not in sounds:
            continue
        sound = sounds[cls]
        pos = int(ev.t * sample_rate)
        end = min(n_samples, pos + len(sound))
        if end > pos:
            out[pos:end] += sound[: end - pos]

    peak = float(np.abs(out).max())
    if peak > 0.9:
        out *= 0.9 / peak

    return out
