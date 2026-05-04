# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 10: Playback engine.

Drives implementation of `voxkit.playback.engine`.

Spec refs: §11 Component 10; v0.9 design (audio playback for the editor
preview, with ability to scrub, loop, and toggle metronome).

The playback engine reads the cleaned audio from a Session and plays it
back through the system's default output device, optionally mixing in
a click track at the session's BPM. It supports scrub-to-position and
A/B loop regions for the editor's preview playback.

============================================================
TEST LIST (implement strictly in order)
============================================================

Construction and lifecycle
  T01  PlaybackEngine constructable from a Session with audio
  T02  Engine starts in "stopped" state
  T03  Engine exposes total_duration_seconds matching the audio length
  T04  Engine exposes sample_rate matching the session

Play/pause/stop
  T05  play() transitions state to "playing"
  T06  pause() transitions state to "paused"
  T07  stop() transitions state to "stopped" and resets position to 0
  T08  play() from paused resumes from current position
  T09  Calling play() while playing is a no-op (no state thrash)
  T10  Calling pause() while stopped is a no-op

Position control
  T11  Initial position is 0.0 seconds
  T12  set_position(t) updates the position
  T13  set_position past total_duration clamps to total_duration
  T14  set_position with negative t clamps to 0.0
  T15  Position advances during playback (mock-time test)

  -- TIDY FIRST before T16: extract `_buffer_to_output(t, audio, fs,
     n_frames)` to a pure helper. The same logic is used for both
     normal playback and the loop-region playback path.

Loop region
  T16  Setting a loop region (start, end) wraps playback at end → start
  T17  Loop region with start > end raises
  T18  Loop region with start < 0 raises
  T19  Loop region with end > total_duration clamps end
  T20  clear_loop() removes the loop region

Metronome / click track
  T21  Metronome off by default
  T22  Enabling metronome at session BPM mixes clicks at beat positions
       (verified via output spectrum or impulse positions in mock buffer)
  T23  Metronome volume independent of audio volume
  T24  Disabling metronome stops mixing clicks immediately

Volume
  T25  Default volume is 1.0 (unit gain)
  T26  Volume of 0.0 produces silent output
  T27  Volume of 0.5 produces half-amplitude output (within 1e-6)
  T28  Volume above 1.0 is allowed (no clipping in the engine; mixing
       layer's responsibility)

Output device handling
  T29  Engine opens output stream on first play()
  T30  Engine closes output stream on stop()
  T31  Output device errors surface as PlaybackError, not crashes

Threading and callback safety
  T32  Engine playback callback does not allocate (tracemalloc check)
  T33  Engine cleanly stops if the audio buffer is replaced mid-playback

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

Race condition safety (Sam, Lin, Alex, Casey, Riley, Marco: 6/9)
  T34  Concurrent set_position() during _render() does not produce
       garbled output or crash. Uses a small synthetic stress: spawn
       a thread that calls set_position() in a tight loop while the
       main test repeatedly calls _render(); assert no exceptions and
       all output samples are within audio's value range.

End-of-buffer behavior (Lin, Marco, Sam, Casey, Riley, Jordan: 6/9)
  T35  When playback reaches end of audio without a loop region, state
       transitions to STOPPED automatically (does not freeze in PLAYING
       state forever, which would leave the UI's transport in an
       inconsistent state).

Output device hot-swap (Lin, Sam, Alex, Casey, Riley: 5/9 — WEAK,
recorded as OQ-1)

Editor sync (Jordan, Marco, Casey, Riley: 4/9 — WEAK, recorded as
OQ-2)
  -- A "now-playing position" callback for the editor's playhead
     indicator. Implementer's choice whether to expose; defer.

============================================================
v0.12 PANEL ADDITIONS (Lin DSP review + principal-engineer synthesis;
three of four review agents rate-limited)
============================================================

Pause-state correctness (Lin: STRONG — PortAudio keeps calling the
callback after pause(); if the engine renders the next chunk, you get
audible dropouts on resume — a real, common bug)
  T36  _render() while state == PAUSED returns silence (zeros) and does
       NOT advance position. Catches the "pause clicks" regression
       where the buffer cursor moves while the user thinks playback is
       held.

Loop-region edge cases (Lin: STRONG — undefined behavior today; the
spec must pick one and the test pins it)
  T37  set_position(t < loop_start) while a loop region is active
       clamps INTO the loop region (cursor jumps to loop_start). The
       alternative — temporarily exiting the loop — is a foot-gun for
       the editor's scrubbing UX. v0.12 picks clamp-into-loop; if the
       implementer prefers exit-loop, update the spec text in §11
       Component 10 first.

Metronome timing tightness (Lin: STRONG for a percussion app — a 6 ms
slop in the click position is audible to a drummer; tighten to 2 ms)
  T38  Metronome click within ±2 ms of beat position. T22's ±6 ms
       window is too generous for the target user; if the engine
       quantizes click placement to buffer boundaries (off by one
       buffer at 5 ms = 5 ms slop), this test fails and the engine
       must fix it.

Replace-audio safety (Lin: STRONG — T33 only checks state transition,
not that subsequent renders are sane)
  T39  After replace_audio() with a shorter buffer, the next _render()
       call does NOT raise IndexError, does NOT return uninitialized
       memory, and either returns silence or honors the new buffer
       length. T33 left this gap; T39 closes it.

Tightening of v0.11 panel additions
  T34  TIGHTENED: also assert at least one rendered chunk matches
       audio[expected_pos:expected_pos+512] for stable position. The
       v0.11 form ("|buf| <= 0.51") catches catastrophic OOB but a torn
       read from a wildly different position would pass.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  Output device hot-swap (user changes Windows default mid-playback).
      Surface as PlaybackError + auto-stop, or attempt re-open?
OQ-2  Editor playhead callback API (above).
OQ-3  T22 metronome detection via "peaks > 0.5" threshold is brittle —
      depends on the click sample's peak magnitude. v0.12 (Lin):
      revisit; replace with cross-correlation against the click sample,
      threshold on correlation peak. Tracker for v1.1 cleanup.
"""

from __future__ import annotations

import tracemalloc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_session(duration_s=2.0, fs=16_000, bpm=120):
    from voxkit.core.session import Session, TimeSignature
    return Session(
        bpm=bpm,
        time_signature=TimeSignature(4, 4),
        bars=int(round(duration_s * bpm / 60.0 / 4)),
        sample_rate=fs,
        recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.linspace(-0.5, 0.5, int(fs * duration_s), dtype=np.float32),
    )


# ---------------------------------------------------------------
# Construction and lifecycle
# ---------------------------------------------------------------

def test_T01_constructable_from_session():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session())
    assert eng is not None


def test_T02_initial_state_stopped():
    from voxkit.playback.engine import PlaybackEngine, PlaybackState
    eng = PlaybackEngine(session=_make_session())
    assert eng.state == PlaybackState.STOPPED


def test_T03_total_duration_matches_audio_length():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session(duration_s=3.0))
    assert eng.total_duration_seconds == pytest.approx(3.0, abs=1e-3)


def test_T04_sample_rate_matches_session():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session(fs=16_000))
    assert eng.sample_rate == 16_000


# ---------------------------------------------------------------
# Play/pause/stop
# ---------------------------------------------------------------

def test_T05_play_transitions_to_playing():
    from voxkit.playback.engine import PlaybackEngine, PlaybackState
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session())
        eng.play()
        assert eng.state == PlaybackState.PLAYING


def test_T06_pause_transitions_to_paused():
    from voxkit.playback.engine import PlaybackEngine, PlaybackState
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session())
        eng.play()
        eng.pause()
        assert eng.state == PlaybackState.PAUSED


def test_T07_stop_resets_state_and_position():
    from voxkit.playback.engine import PlaybackEngine, PlaybackState
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session())
        eng.play()
        eng.set_position(1.0)
        eng.stop()
        assert eng.state == PlaybackState.STOPPED
        assert eng.position == pytest.approx(0.0)


def test_T08_play_after_pause_resumes_from_current_position():
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session())
        eng.play()
        eng.set_position(0.7)
        eng.pause()
        eng.play()
        assert eng.position == pytest.approx(0.7)


def test_T09_play_while_playing_is_noop():
    from voxkit.playback.engine import PlaybackEngine, PlaybackState
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session())
        eng.play()
        eng.play()
        assert eng.state == PlaybackState.PLAYING


def test_T10_pause_while_stopped_is_noop():
    from voxkit.playback.engine import PlaybackEngine, PlaybackState
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session())
        eng.pause()
        assert eng.state == PlaybackState.STOPPED


# ---------------------------------------------------------------
# Position control
# ---------------------------------------------------------------

def test_T11_initial_position_zero():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session())
    assert eng.position == pytest.approx(0.0)


def test_T12_set_position_updates_position():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session(duration_s=2.0))
    eng.set_position(1.0)
    assert eng.position == pytest.approx(1.0)


def test_T13_set_position_past_duration_clamps():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session(duration_s=2.0))
    eng.set_position(10.0)
    assert eng.position == pytest.approx(2.0)


def test_T14_set_position_negative_clamps_to_zero():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session())
    eng.set_position(-1.0)
    assert eng.position == pytest.approx(0.0)


def test_T15_position_advances_during_playback():
    """Drive playback by feeding the callback fake buffer requests; verify
    that the engine's internal position advances by frames * (1/fs)."""
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(fs=16_000, duration_s=10.0))
        eng.play()
        # Simulate 1 second of audio output (16k frames in chunks of 256).
        for _ in range(16_000 // 256):
            eng._render(n_frames=256)
        # Allow ~1 frame of slop.
        assert eng.position == pytest.approx(1.0, abs=1e-3)


# ----- TIDY FIRST checkpoint -----
# Extract `_buffer_to_output(t, audio, fs, n_frames)` to a pure helper.
# Used by both normal playback and loop-region playback paths.


# ---------------------------------------------------------------
# Loop region
# ---------------------------------------------------------------

def test_T16_loop_region_wraps_at_end():
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=2.0))
        eng.set_loop(start=0.5, end=1.0)
        eng.play()
        eng.set_position(0.99)
        eng._render(n_frames=int(0.05 * 16_000))   # advance ~50ms
        # After wrap, position should be near loop start.
        assert 0.5 <= eng.position < 0.6


def test_T17_loop_with_start_after_end_raises():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session(duration_s=2.0))
    with pytest.raises(ValueError, match="start"):
        eng.set_loop(start=1.5, end=1.0)


def test_T18_loop_negative_start_raises():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session(duration_s=2.0))
    with pytest.raises(ValueError, match="start"):
        eng.set_loop(start=-0.1, end=1.0)


def test_T19_loop_end_past_duration_clamps():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session(duration_s=2.0))
    eng.set_loop(start=0.5, end=10.0)
    assert eng.loop_region == (pytest.approx(0.5), pytest.approx(2.0))


def test_T20_clear_loop_removes_region():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session(duration_s=2.0))
    eng.set_loop(start=0.5, end=1.0)
    eng.clear_loop()
    assert eng.loop_region is None


# ---------------------------------------------------------------
# Metronome / click track
# ---------------------------------------------------------------

def test_T21_metronome_off_by_default():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session())
    assert eng.metronome_enabled is False


def test_T22_metronome_mixes_clicks_at_beat_positions():
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=2.0, bpm=120))
        eng.metronome_enabled = True
        eng.play()
        # Render the full session and check for click impulses at 0.0, 0.5, 1.0, 1.5.
        out = eng._render(n_frames=16_000 * 2)
        # Detect peaks in the output buffer.
        peak_indices = np.where(np.abs(out) > 0.5)[0]
        # Each click should be reflected within ~5ms of beat positions.
        for beat in (0.0, 0.5, 1.0, 1.5):
            beat_idx = int(beat * 16_000)
            window = peak_indices[(peak_indices >= beat_idx) & (peak_indices < beat_idx + 100)]
            assert len(window) > 0, f"No metronome click near beat {beat}"


def test_T23_metronome_volume_independent_of_audio_volume():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session())
    eng.volume = 0.5
    eng.metronome_volume = 1.0
    assert eng.metronome_volume == 1.0
    assert eng.volume == 0.5


def test_T24_disabling_metronome_stops_clicks_immediately():
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=2.0))
        eng.metronome_enabled = True
        eng.play()
        eng._render(n_frames=8000)
        eng.metronome_enabled = False
        out = eng._render(n_frames=8000)
        # No spike should appear in the second half, where audio is smooth ramp.
        assert np.max(np.abs(out)) < 1.0


# ---------------------------------------------------------------
# Volume
# ---------------------------------------------------------------

def test_T25_default_volume_unit_gain():
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session())
    assert eng.volume == pytest.approx(1.0)


def test_T26_volume_zero_silences_output():
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session())
        eng.volume = 0.0
        eng.play()
        out = eng._render(n_frames=1024)
        assert np.max(np.abs(out)) == pytest.approx(0.0)


def test_T27_volume_half_produces_half_amplitude():
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng_full = PlaybackEngine(session=_make_session())
        eng_full.play()
        full = eng_full._render(n_frames=1024).copy()

        eng_half = PlaybackEngine(session=_make_session())
        eng_half.volume = 0.5
        eng_half.play()
        half = eng_half._render(n_frames=1024).copy()

        np.testing.assert_allclose(half, 0.5 * full, atol=1e-6)


def test_T28_volume_above_one_allowed():
    """The engine itself does not clip; mixing/output stage handles it."""
    from voxkit.playback.engine import PlaybackEngine
    eng = PlaybackEngine(session=_make_session())
    eng.volume = 2.0
    assert eng.volume == 2.0


# ---------------------------------------------------------------
# Output device handling
# ---------------------------------------------------------------

def test_T29_engine_opens_output_stream_on_first_play():
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()) as opener:
        eng = PlaybackEngine(session=_make_session())
        assert opener.call_count == 0
        eng.play()
        assert opener.call_count == 1


def test_T30_engine_closes_output_stream_on_stop():
    from voxkit.playback.engine import PlaybackEngine
    stream = MagicMock()
    with patch("voxkit.playback.engine._open_output_stream", return_value=stream):
        eng = PlaybackEngine(session=_make_session())
        eng.play()
        eng.stop()
        stream.close.assert_called_once()


def test_T31_output_device_errors_surface_as_playback_error():
    from voxkit.playback.engine import PlaybackEngine, PlaybackError
    with patch("voxkit.playback.engine._open_output_stream",
               side_effect=OSError("device busy")):
        eng = PlaybackEngine(session=_make_session())
        with pytest.raises(PlaybackError):
            eng.play()


# ---------------------------------------------------------------
# Threading and callback safety
# ---------------------------------------------------------------

def test_T32_render_callback_no_allocation():
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=10.0))
        eng.play()
        eng._render(n_frames=256)   # warm

        tracemalloc.start()
        snap1 = tracemalloc.take_snapshot()
        for _ in range(500):
            eng._render(n_frames=256)
        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snap2.compare_to(snap1, "filename")
        on_path = sum(s.size_diff for s in diff if "playback" in s.traceback[0].filename)
        assert on_path == 0


def test_T33_replacing_audio_buffer_midplayback_stops_cleanly():
    from voxkit.playback.engine import PlaybackEngine, PlaybackState
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=10.0))
        eng.play()
        eng._render(n_frames=1024)
        # Replace audio with a shorter buffer.
        eng.replace_audio(np.zeros(8_000, dtype=np.float32))
        # State should be stopped or position clamped, but not crashed.
        assert eng.state in (PlaybackState.STOPPED, PlaybackState.PAUSED)


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T34_concurrent_set_position_during_render_safe():
    """The editor's UI thread can call set_position() (e.g., user drags
    the playhead) at any time while the playback callback is rendering.
    Must not crash and must not produce out-of-range output samples."""
    import threading
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=4.0))
        eng.play()

        stop = threading.Event()
        errors = []

        def thrash_position():
            try:
                while not stop.is_set():
                    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
                        eng.set_position(t)
            except Exception as e:   # pragma: no cover
                errors.append(e)

        t = threading.Thread(target=thrash_position, daemon=True)
        t.start()

        all_out = []
        try:
            for _ in range(50):
                out = eng._render(n_frames=512)
                all_out.append(out.copy())
        finally:
            stop.set()
            t.join(timeout=1.0)

        assert errors == []
        for buf in all_out:
            # _make_session uses np.linspace(-0.5, 0.5); output should
            # never escape that range (no garbled / OOB reads).
            assert np.all(np.abs(buf) <= 0.51 + 1e-6), (
                "concurrent set_position produced out-of-range samples"
            )


def test_T35_playback_auto_stops_at_end_of_buffer():
    """Without a loop region, playback should not loop or freeze in
    PLAYING when the audio runs out — it should transition to STOPPED
    so the editor's transport widget reflects reality."""
    from voxkit.playback.engine import PlaybackEngine, PlaybackState
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=0.5))
        eng.play()
        # Render past end of buffer.
        for _ in range(20):
            eng._render(n_frames=1024)
        assert eng.state == PlaybackState.STOPPED, (
            f"engine stuck in {eng.state} past end of audio; "
            "must auto-transition to STOPPED"
        )


# ---------------------------------------------------------------
# v0.12 panel additions (Lin DSP review + principal-engineer synthesis)
# ---------------------------------------------------------------

def test_T36_render_during_pause_returns_silence_and_holds_position():
    """v0.12 (Lin): PortAudio keeps invoking the callback after pause();
    if _render() advances the buffer cursor while paused, you get a
    'pause click' on resume — audible dropout because the playhead is
    no longer where the user thinks it is."""
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=2.0))
        eng.play()
        eng.set_position(0.5)
        eng.pause()
        position_before = eng.position
        out = eng._render(n_frames=1024)
        position_after = eng.position

        np.testing.assert_array_equal(out, np.zeros(1024, dtype=out.dtype))
        assert position_after == pytest.approx(position_before, abs=1e-9), (
            f"position advanced during pause: {position_before:.4f} → "
            f"{position_after:.4f}"
        )


def test_T37_set_position_inside_loop_clamps_into_loop_region():
    """v0.12 (Lin): with a loop region active, scrubbing OUTSIDE the
    loop must either (a) clamp the position into the loop, or (b)
    temporarily clear the loop. The undefined-behavior status quo is a
    foot-gun. v0.12 picks clamp-into-loop (cursor jumps to loop_start);
    flip the test if the spec is amended to choose otherwise."""
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=2.0))
        eng.set_loop(start=0.5, end=1.0)

        eng.set_position(0.2)   # before loop start
        assert eng.position == pytest.approx(0.5, abs=1e-9), (
            "set_position before loop_start did not clamp into loop"
        )
        eng.set_position(1.5)   # after loop end
        assert eng.position == pytest.approx(0.5, abs=1e-9) or \
               eng.position == pytest.approx(1.0, abs=1e-9), (
            "set_position past loop_end neither clamped to start nor end"
        )


def test_T38_metronome_clicks_within_2ms_of_beat():
    """v0.12 (Lin): T22's ±6 ms window is audible slop for a drummer.
    A correctly-implemented metronome places clicks at exact beat
    sample-positions; ±2 ms (32 samples @ 16k) is the tightness a
    percussion app should commit to."""
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=2.0, bpm=120))
        eng.metronome_enabled = True
        eng.play()
        out = eng._render(n_frames=16_000 * 2)
        peak_indices = np.where(np.abs(out) > 0.5)[0]

        max_slop_samples = int(0.002 * 16_000)   # ±2 ms
        for beat_s in (0.0, 0.5, 1.0, 1.5):
            beat_idx = int(beat_s * 16_000)
            in_window = peak_indices[
                (peak_indices >= beat_idx - max_slop_samples) &
                (peak_indices <= beat_idx + max_slop_samples)
            ]
            assert len(in_window) > 0, (
                f"no metronome click within ±2 ms of beat {beat_s}s"
            )


def test_T39_render_after_replace_audio_no_oob_no_garbage():
    """v0.12 (Lin): T33 only asserts a state transition. It doesn't
    actually try to render past where the new (shorter) buffer ends.
    A naive implementation that doesn't update its internal length
    cache will IndexError or return uninitialized memory."""
    from voxkit.playback.engine import PlaybackEngine
    with patch("voxkit.playback.engine._open_output_stream", return_value=MagicMock()):
        eng = PlaybackEngine(session=_make_session(duration_s=10.0))
        eng.play()
        eng.set_position(0.4)
        eng.replace_audio(np.zeros(8_000, dtype=np.float32))   # 0.5s at 16k

        # Try to render past end of the new buffer.
        for _ in range(5):
            out = eng._render(n_frames=1024)
            assert out.shape == (1024,)
            assert np.all(np.isfinite(out)), "render returned non-finite samples"
            assert np.all(np.abs(out) <= 1.0 + 1e-6), "render returned out-of-range samples"
