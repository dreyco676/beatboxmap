# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 8: Tempo & grid engine.

Drives implementation of `voxkit.tempo.grid`.

Spec refs: §11 Component 8; v0.9 design (BPM-driven grid construction,
quantize_strength interpolation between raw and grid time).

The tempo & grid engine takes a session's BPM and time signature, builds
a grid of beat/sub-beat positions, and quantizes onset events toward the
nearest grid position by a configurable strength.

============================================================
TEST LIST (implement strictly in order)
============================================================

Grid construction
  T01  build_grid for 1 bar of 4/4 at 120 BPM with 1/16 grid has 16 positions
  T02  Grid positions span exactly the bar duration (last == bar_duration)
  T03  Grid is monotonically increasing
  T04  120 BPM 4/4 1/16 grid: position[1] - position[0] == 0.125 s
  T05  60 BPM 4/4 1/16 grid: position[1] - position[0] == 0.25 s
  T06  3/4 time signature gives correct number of positions per bar
  T07  Grid for N bars has N × (positions_per_bar) positions
  T08  Empty grid (0 bars) returns empty array

Pre-conditions
  T09  build_grid with bpm <= 0 raises
  T10  build_grid with negative bars raises
  T11  build_grid with unknown grid spec ("1/3") raises
  T12  build_grid supports 1/4, 1/8, 1/16, 1/32 grid specs

Quantization (single event)
  T13  Event exactly on a grid position is unchanged at strength=1.0
  T14  Event 5 ms after grid position: strength=0 → unchanged
  T15  Event 5 ms after grid position: strength=1 → exactly on grid
  T16  Event 5 ms after grid position: strength=0.5 → 2.5 ms after grid
  T17  Event between two grid positions snaps to the nearest one (strength=1)
  T18  Event before the first grid position snaps to position[0] at strength=1
  T19  Event after the last grid position snaps to position[-1] at strength=1

Quantization (batch of events)
  T20  Quantizing an empty list of events returns an empty list
  T21  Quantizing preserves event order (no reordering)
  T22  Quantizing preserves event class_id and score (not just timestamps)
  T23  Quantize is deterministic: same inputs → same outputs

  -- TIDY FIRST before T24: extract `_nearest_grid_index(t, grid)` to a
     pure helper so tempo curves (future v1.1) can reuse it. Also lets
     us unit-test the pure function without quantize_strength noise.

Edge cases / numerical stability
  T24  Quantize handles event time exactly between two grid positions
       (deterministic tie-breaking: round-half-to-even or round-up,
       documented in spec)
  T25  Floating-point error: event at grid + 1e-12 quantizes to grid
  T26  Very dense grid (1/32 at 200 BPM) does not produce duplicate
       grid positions due to float accumulation

Strength interpolation
  T27  strength=0 is the identity (events unchanged)
  T28  strength=1 places events exactly on grid
  T29  Linearity: at strength=s, displacement = s × (grid - raw)
  T30  Strength outside [0, 1] is clamped (0.5 → 0.5; 1.5 → 1.0; -0.1 → 0)

Integration with Session
  T31  quantize_session uses session.bpm, time_signature, and bars
  T32  quantize_session writes results to session.events
       (or returns a new Session — implementer's choice; test asserts the
       returned object's events have grid-aligned timestamps)

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

Tie-breaking honesty (Marco, Lin, Sam, Alex, Casey, Jordan: 6/9)
  T33  T24 currently asserts "snap to LATER position on a tie", but the
       spec text in T24's docstring says "round-half-up". Vocal-percussion
       performers tend to play slightly behind the beat, not ahead;
       round-half-DOWN (snap to earlier) is the percussion-correct
       choice. T33 documents the panel decision: snap-to-earlier on ties
       is the v1.0 default; the spec text §11 Component 8 must be
       updated to match. If the implementer keeps round-half-up, T33
       fails loudly so the disagreement surfaces in code review.

Triplet support gap (Marco, Jordan, Casey: 3/9 — RECORDED AS OQ)
  -- vocal-percussion grooves often use 1/8 triplets and 1/16 triplets;
     T11 explicitly rejects "1/3" but the spec doesn't mention triplets.
     OPEN QUESTION OQ-1: ship triplet support in v1.0 (cheap, ~1 day) or
     defer to v1.1?

Edge-case integrity (Sam, Lin, Alex, Casey, Riley, Marco: 6/9)
  T34  Quantize with strength=NaN raises ValueError (does not silently
       map to 0 or 1).
  T35  Empty grid (0-bar session) + non-empty events: quantize_events
       raises EmptyGrid (does not silently return events unchanged,
       which would surface to the user as "quantize button does nothing").

Performance sanity (Sam, Lin, Casey, Riley: 4/9 — WEAK; recorded
as OQ-2)
  -- A 32-bar session at 1/32 grid has ~1024 grid positions and can
     have hundreds of events. quantize_events should be O(n_events *
     log(n_grid)) via binary search, not O(n_events * n_grid) via
     linear scan. Bench under tracemalloc is overkill for v1.0; defer.

============================================================
v0.12 PANEL ADDITIONS (principal-engineer + Marco synthesis;
Marco-equivalent reviewer rate-limited)
============================================================

User-facing error quality (STRONG — vocal percussion uses triplets;
T11 rejecting "1/3" is correct for v1.0 but the error message must
help the user, not just say "subdivision invalid")
  T36  Unsupported subdivision error names the supported set. v0.11
       T11 accepts any error message containing "subdivision"; v0.12
       requires the message to enumerate the supported subdivisions
       so a user trying "1/3" sees "1/3 not supported; try 1/4, 1/8,
       1/16, or 1/32".

Quantize integrity (STRONG — currently no test asserts that
quantize_events doesn't lose or duplicate events)
  T37  quantize_events preserves event count exactly. T20 covers
       empty input; T21/T22 cover ordering and per-event fidelity but
       neither pins count for the multi-event case. A buggy quantizer
       that drops events on grid-overlap collisions would pass v0.11
       and surface to the user as "missing notes after quantize".

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  Triplet subdivisions (1/8t, 1/16t) for v1.0? v0.12 (Marco): if
      deferred, T36 above ensures the user gets a helpful error.
OQ-2  Quantize big-O complexity guard (deferred per above).
OQ-3  Tempo curves / variable BPM (Q-future, defer to v1.1).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------

def test_T01_one_bar_4_4_120bpm_sixteenth_grid_has_16_positions():
    from voxkit.tempo.grid import build_grid
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    assert len(grid) == 16


def test_T02_grid_spans_exactly_one_bar():
    from voxkit.tempo.grid import build_grid
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    bar_duration = 4 * (60.0 / 120.0)   # 4 beats × 0.5s per beat = 2.0s
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(bar_duration - bar_duration / 16, abs=1e-9)


def test_T03_grid_is_monotonically_increasing():
    from voxkit.tempo.grid import build_grid
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=2, subdivision="1/16")
    diffs = np.diff(grid)
    assert np.all(diffs > 0)


def test_T04_120bpm_4_4_sixteenth_step_is_125ms():
    from voxkit.tempo.grid import build_grid
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    assert grid[1] - grid[0] == pytest.approx(0.125, abs=1e-9)


def test_T05_60bpm_4_4_sixteenth_step_is_250ms():
    from voxkit.tempo.grid import build_grid
    grid = build_grid(bpm=60, time_signature=(4, 4), bars=1, subdivision="1/16")
    assert grid[1] - grid[0] == pytest.approx(0.25, abs=1e-9)


def test_T06_three_four_time_signature():
    from voxkit.tempo.grid import build_grid
    grid = build_grid(bpm=120, time_signature=(3, 4), bars=1, subdivision="1/16")
    assert len(grid) == 12   # 3 beats × 4 sixteenths/beat


def test_T07_n_bars_has_n_times_positions_per_bar():
    from voxkit.tempo.grid import build_grid
    one_bar = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    four_bars = build_grid(bpm=120, time_signature=(4, 4), bars=4, subdivision="1/16")
    assert len(four_bars) == 4 * len(one_bar)


def test_T08_zero_bars_returns_empty_grid():
    from voxkit.tempo.grid import build_grid
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=0, subdivision="1/16")
    assert len(grid) == 0


# ---------------------------------------------------------------
# Pre-conditions
# ---------------------------------------------------------------

def test_T09_zero_bpm_rejected():
    from voxkit.tempo.grid import build_grid
    with pytest.raises(ValueError, match="bpm"):
        build_grid(bpm=0, time_signature=(4, 4), bars=1, subdivision="1/16")


def test_T10_negative_bars_rejected():
    from voxkit.tempo.grid import build_grid
    with pytest.raises(ValueError, match="bars"):
        build_grid(bpm=120, time_signature=(4, 4), bars=-1, subdivision="1/16")


def test_T11_unknown_subdivision_rejected():
    from voxkit.tempo.grid import build_grid
    with pytest.raises(ValueError, match="subdivision"):
        build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/3")


def test_T12_supported_subdivisions():
    from voxkit.tempo.grid import build_grid
    for sub in ("1/4", "1/8", "1/16", "1/32"):
        grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision=sub)
        assert len(grid) > 0


# ---------------------------------------------------------------
# Quantization (single event)
# ---------------------------------------------------------------

def test_T13_event_on_grid_unchanged_at_full_strength():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    out = quantize_time(grid[3], grid, strength=1.0)
    assert out == pytest.approx(grid[3])


def test_T14_strength_zero_does_not_move_event():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    raw = grid[3] + 0.005   # 5 ms after
    assert quantize_time(raw, grid, strength=0.0) == pytest.approx(raw)


def test_T15_strength_one_snaps_to_grid():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    raw = grid[3] + 0.005
    assert quantize_time(raw, grid, strength=1.0) == pytest.approx(grid[3])


def test_T16_strength_half_moves_halfway():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    raw = grid[3] + 0.005
    out = quantize_time(raw, grid, strength=0.5)
    assert out == pytest.approx(grid[3] + 0.0025, abs=1e-9)


def test_T17_event_snaps_to_nearest_grid():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    midpoint_offset = (grid[1] - grid[0]) * 0.4   # less than half = snap to grid[0]+1 nope, grid[3]
    raw = grid[3] + midpoint_offset
    out = quantize_time(raw, grid, strength=1.0)
    assert out == pytest.approx(grid[3])


def test_T18_event_before_first_grid_snaps_to_first():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    out = quantize_time(-0.5, grid, strength=1.0)
    assert out == pytest.approx(grid[0])


def test_T19_event_after_last_grid_snaps_to_last():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    out = quantize_time(100.0, grid, strength=1.0)
    assert out == pytest.approx(grid[-1])


# ---------------------------------------------------------------
# Quantization (batch of events)
# ---------------------------------------------------------------

def test_T20_empty_event_list_returns_empty():
    from voxkit.tempo.grid import build_grid, quantize_events
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    assert quantize_events([], grid, strength=1.0) == []


def test_T21_quantize_preserves_order():
    from voxkit.tempo.grid import build_grid, quantize_events, Event
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    events = [
        Event(t=0.05, class_id="kick", score=0.9),
        Event(t=0.15, class_id="snare", score=0.8),
        Event(t=0.30, class_id="kick", score=0.85),
    ]
    out = quantize_events(events, grid, strength=1.0)
    assert [e.t for e in out] == sorted(e.t for e in out)


def test_T22_quantize_preserves_class_id_and_score():
    from voxkit.tempo.grid import build_grid, quantize_events, Event
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    events = [Event(t=0.07, class_id="snare", score=0.83)]
    out = quantize_events(events, grid, strength=1.0)
    assert out[0].class_id == "snare"
    assert out[0].score == 0.83


def test_T23_quantize_deterministic():
    from voxkit.tempo.grid import build_grid, quantize_events, Event
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    events = [Event(t=0.07, class_id="snare", score=0.83)]
    out_a = quantize_events(events, grid, strength=0.7)
    out_b = quantize_events(events, grid, strength=0.7)
    assert [e.t for e in out_a] == [e.t for e in out_b]


# ----- TIDY FIRST checkpoint -----
# Extract `_nearest_grid_index(t, grid)` to a pure helper. Used by both
# `quantize_time` and (future v1.1) tempo-curve mapping. Pure structural
# change; no behavior delta.


# ---------------------------------------------------------------
# Edge cases / numerical stability
# ---------------------------------------------------------------

def test_T24_event_exactly_between_two_grid_positions_deterministic():
    """Determinism test only — see T33 for the panel-decided tie direction.

    v0.10 sketch was 'round-half-up' (snap to LATER position). The v0.11
    panel (Marco/Lin/Sam consensus) flipped this to round-half-down for
    musical reasons (T33). T24 retained as a determinism guard: same
    input must always produce the same output.
    """
    from voxkit.tempo.grid import quantize_time
    grid = np.array([0.0, 0.1, 0.2])
    raw = 0.05
    a = quantize_time(raw, grid, strength=1.0)
    b = quantize_time(raw, grid, strength=1.0)
    assert a == b   # deterministic, regardless of which side


def test_T25_tiny_floating_point_error_quantizes_to_grid():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    out = quantize_time(grid[3] + 1e-12, grid, strength=1.0)
    assert out == pytest.approx(grid[3], abs=1e-10)


def test_T26_dense_grid_no_duplicate_positions():
    from voxkit.tempo.grid import build_grid
    grid = build_grid(bpm=200, time_signature=(4, 4), bars=8, subdivision="1/32")
    assert len(np.unique(grid)) == len(grid)


# ---------------------------------------------------------------
# Strength interpolation
# ---------------------------------------------------------------

def test_T27_strength_zero_is_identity():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    raw = 0.07
    assert quantize_time(raw, grid, strength=0.0) == pytest.approx(raw)


def test_T28_strength_one_places_on_grid():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    out = quantize_time(0.07, grid, strength=1.0)
    assert any(abs(out - g) < 1e-9 for g in grid)


def test_T29_strength_linearity():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    raw = 0.07
    target = quantize_time(raw, grid, strength=1.0)
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = quantize_time(raw, grid, strength=s)
        assert out == pytest.approx(raw + s * (target - raw), abs=1e-9)


def test_T30_strength_outside_unit_clamped():
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    raw = 0.07
    target = quantize_time(raw, grid, strength=1.0)
    assert quantize_time(raw, grid, strength=1.5) == pytest.approx(target)
    assert quantize_time(raw, grid, strength=-0.1) == pytest.approx(raw)


# ---------------------------------------------------------------
# Integration with Session
# ---------------------------------------------------------------

def test_T31_quantize_session_uses_session_bpm_and_signature():
    from voxkit.tempo.grid import quantize_session
    from voxkit.core.session import Session, TimeSignature, Event
    s = Session(
        bpm=120,
        time_signature=TimeSignature(4, 4),
        bars=1,
        sample_rate=16_000,
        recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000 * 2, dtype=np.float32),
        events=[Event(t=0.07, class_id="kick", score=0.9)],
        quantize_grid="1/16",
        quantize_strength=1.0,
    )
    out = quantize_session(s)
    assert out.events[0].t == pytest.approx(0.125, abs=1e-9)


def test_T32_quantize_session_returns_session_with_grid_aligned_events():
    from voxkit.tempo.grid import quantize_session, build_grid
    from voxkit.core.session import Session, TimeSignature, Event
    s = Session(
        bpm=120,
        time_signature=TimeSignature(4, 4),
        bars=2,
        sample_rate=16_000,
        recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000 * 4, dtype=np.float32),
        events=[
            Event(t=0.07, class_id="kick", score=0.9),
            Event(t=0.31, class_id="snare", score=0.8),
        ],
        quantize_grid="1/16",
        quantize_strength=1.0,
    )
    out = quantize_session(s)
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=2, subdivision="1/16")
    for e in out.events:
        assert any(abs(e.t - g) < 1e-9 for g in grid)


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T33_tie_breaking_snaps_to_earlier_grid_position():
    """v0.11 panel decision (Marco/Lin/Sam consensus, 6/9): vocal-
    percussion performers tend to play slightly BEHIND the beat. Snapping
    forward (later) on a tie yanks them out of feel; snapping backward
    (earlier) preserves the human-played groove.

    Spec text §11 Component 8 is being amended to "round-half-down on
    ties". If you're updating quantize_time and this fails because you
    kept round-half-up, escalate to the panel before changing this test.
    """
    from voxkit.tempo.grid import quantize_time
    grid = np.array([0.0, 0.1, 0.2])
    raw = 0.05   # exactly midway
    out = quantize_time(raw, grid, strength=1.0)
    assert out == pytest.approx(0.0), (
        "tie should snap to earlier grid position (round-half-down) "
        "per v0.11 panel; got snap to later"
    )


def test_T34_strength_nan_raises():
    """A NaN strength is almost always a bug upstream (uninitialized
    UI slider state). Silently mapping to 0 or 1 hides the bug."""
    from voxkit.tempo.grid import build_grid, quantize_time
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/16")
    with pytest.raises(ValueError, match="strength"):
        quantize_time(0.07, grid, strength=float("nan"))


def test_T35_quantize_events_with_empty_grid_raises():
    """A 0-bar session has an empty grid. quantize_events on a non-empty
    event list with an empty grid should raise EmptyGrid, not silently
    return the events unchanged (which would surface to the user as
    "the quantize button does nothing")."""
    from voxkit.tempo.grid import quantize_events, EmptyGrid, Event
    events = [Event(t=0.5, class_id="kick", score=0.9)]
    empty_grid = np.zeros(0)
    with pytest.raises(EmptyGrid):
        quantize_events(events, empty_grid, strength=1.0)


# ---------------------------------------------------------------
# v0.12 panel additions (principal-engineer + Marco synthesis)
# ---------------------------------------------------------------

def test_T36_unsupported_subdivision_error_lists_supported_set():
    """v0.12 (Marco): vocal-percussion grooves often use 1/8 triplets;
    v1.0 doesn't ship triplet support (OQ-1) but the user trying '1/3'
    deserves an error message that names the supported set, not just
    'subdivision invalid'. The match against 'supported' is loose to
    let the implementer choose phrasing."""
    from voxkit.tempo.grid import build_grid
    with pytest.raises(ValueError) as exc:
        build_grid(bpm=120, time_signature=(4, 4), bars=1, subdivision="1/3")
    msg = str(exc.value)
    # Must enumerate the supported subdivisions.
    for known in ("1/4", "1/8", "1/16", "1/32"):
        assert known in msg, (
            f"unsupported-subdivision error must enumerate supported set; "
            f"missing {known} in message: {msg!r}"
        )


def test_T37_quantize_events_preserves_event_count_exactly():
    """v0.12: T20 covers empty input; T21/T22 cover ordering and per-
    event fidelity. Nothing pins the count invariant for the multi-
    event case. A buggy quantizer that drops events when two snap to
    the same grid position (silent dedup) would surface as 'missing
    notes after quantize' — wrong layer, wrong fix."""
    from voxkit.tempo.grid import build_grid, quantize_events, Event
    grid = build_grid(bpm=120, time_signature=(4, 4), bars=2, subdivision="1/16")
    # Three events that will all snap to NEAR the same grid position
    # (within 5 ms of grid[2]); a dedup bug would collapse to 1.
    events = [
        Event(t=grid[2] - 0.002, class_id="kick", score=0.9),
        Event(t=grid[2] + 0.001, class_id="kick", score=0.85),
        Event(t=grid[2] + 0.003, class_id="kick", score=0.8),
        Event(t=grid[5], class_id="snare", score=0.7),
    ]
    out = quantize_events(events, grid, strength=1.0)
    assert len(out) == len(events), (
        f"quantize_events lost or duplicated events: "
        f"in={len(events)}, out={len(out)}"
    )
