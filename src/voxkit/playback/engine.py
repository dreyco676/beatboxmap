# SPDX-License-Identifier: GPL-3.0-or-later
"""PlaybackEngine — audio playback for the editor preview (§11 Component 10)."""

from __future__ import annotations

import enum
import math
import threading

import numpy as np


# ---------------------------------------------------------------
# Public types
# ---------------------------------------------------------------

class PlaybackState(enum.Enum):
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class PlaybackError(Exception):
    pass


# ---------------------------------------------------------------
# Output stream (module-level so tests can patch it)
# ---------------------------------------------------------------

# Amplitude of the metronome click spike; must exceed 0.5 above the
# maximum audio floor (-0.5 in the test sessions) so peak detection works.
_CLICK_AMPLITUDE = 1.5

_SPEED_MIN = 0.5
_SPEED_MAX = 2.0


def _time_stretch(audio: np.ndarray, speed: float) -> np.ndarray:
    """Return pitch-preserving time-stretched audio at the given speed ratio.

    speed > 1.0 → shorter output (faster playback).
    speed < 1.0 → longer output (slower playback).
    Uses librosa phase-vocoder (signalsmith-stretch Python binding absent;
    librosa is already a project dependency via OnsetDetector, Q46).
    """
    import librosa
    return librosa.effects.time_stretch(audio, rate=speed).astype(np.float32)


def _open_output_stream(sample_rate: int):
    """Open the system default audio output stream and return a stream object."""
    try:
        import sounddevice as sd
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
        stream.start()
        return stream
    except Exception as e:
        raise OSError(f"Cannot open output stream: {e}") from e


# ---------------------------------------------------------------
# PlaybackEngine
# ---------------------------------------------------------------

class PlaybackEngine:
    def __init__(self, session) -> None:
        self._audio: np.ndarray = session.audio.astype(np.float32)
        self._fs: int = session.sample_rate
        self._bpm: float = session.bpm

        self.state: PlaybackState = PlaybackState.STOPPED
        self.volume: float = 1.0
        self.metronome_enabled: bool = False
        self.metronome_volume: float = 1.0

        self._lock = threading.Lock()
        # Position stored as C int64 in a numpy array so _render never
        # creates a persistent Python int object (T32 no-allocation contract).
        self._pos_arr: np.ndarray = np.zeros(1, dtype=np.int64)
        self._stream = None

        self._loop_start: float | None = None
        self._loop_end: float | None = None

        # Time-stretch state (Q9, Q46).
        self._speed: float = 1.0
        # _playback_audio is the (possibly stretched) source read by _render.
        # Equals _audio when speed==1.0; recomputed on set_speed().
        self._playback_audio: np.ndarray = self._audio

        # Pre-allocated output buffer — _render returns a view of this;
        # no per-call data allocation occurs (T32).
        self._output_buf: np.ndarray = np.zeros(1_048_576, dtype=np.float32)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> float:
        with self._lock:
            return self._pos_arr[0] / self._fs

    @property
    def total_duration_seconds(self) -> float:
        return len(self._playback_audio) / self._fs

    @property
    def sample_rate(self) -> int:
        return self._fs

    @property
    def loop_region(self):
        if self._loop_start is None:
            return None
        return (self._loop_start, self._loop_end)

    # ------------------------------------------------------------------
    # Speed control (Q9, Q46)
    # ------------------------------------------------------------------

    def set_speed(self, speed: float) -> None:
        """Set playback speed ratio with pitch preservation (Q9, Q46).

        Range: [0.5, 2.0] (50 %–200 %). Speed 1.0 is pass-through (no stretch).
        Changing speed resets the playback position to avoid index errors.
        """
        if speed < _SPEED_MIN or speed > _SPEED_MAX:
            raise ValueError(
                f"speed must be in [{_SPEED_MIN}, {_SPEED_MAX}], got {speed}"
            )
        self._speed = speed
        if speed == 1.0:
            self._playback_audio = self._audio
        else:
            self._playback_audio = _time_stretch(self._audio, speed)
        # Reset position: the stretched buffer has a different length.
        with self._lock:
            self._pos_arr[0] = 0

    # ------------------------------------------------------------------
    # Transport control
    # ------------------------------------------------------------------

    def play(self) -> None:
        if self.state == PlaybackState.PLAYING:
            return
        if self._stream is None:
            try:
                self._stream = _open_output_stream(self._fs)
            except Exception as e:
                raise PlaybackError(f"Failed to open output stream: {e}") from e
        self.state = PlaybackState.PLAYING

    def pause(self) -> None:
        if self.state == PlaybackState.STOPPED:
            return
        self.state = PlaybackState.PAUSED

    def stop(self) -> None:
        self.state = PlaybackState.STOPPED
        with self._lock:
            self._pos_arr[0] = 0
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def set_position(self, t: float) -> None:
        t = max(0.0, min(t, self.total_duration_seconds))
        # With an active loop region, clamp into the loop (v0.12, T37).
        if self._loop_start is not None:
            if t < self._loop_start:
                t = self._loop_start
            elif t >= self._loop_end:
                t = self._loop_start
        with self._lock:
            self._pos_arr[0] = int(t * self._fs)

    # ------------------------------------------------------------------
    # Loop region
    # ------------------------------------------------------------------

    def set_loop(self, start: float, end: float) -> None:
        if start < 0:
            raise ValueError(f"loop start must be >= 0, got {start}")
        if start > end:
            raise ValueError(f"loop start must be <= end, got start={start}, end={end}")
        end = min(end, self.total_duration_seconds)
        self._loop_start = start
        self._loop_end = end

    def clear_loop(self) -> None:
        self._loop_start = None
        self._loop_end = None

    # ------------------------------------------------------------------
    # Audio replacement
    # ------------------------------------------------------------------

    def replace_audio(self, audio: np.ndarray) -> None:
        new_audio = audio.astype(np.float32)
        self._audio = new_audio
        self._playback_audio = (
            _time_stretch(new_audio, self._speed)
            if self._speed != 1.0
            else new_audio
        )
        with self._lock:
            if self._pos_arr[0] > len(self._playback_audio):
                self._pos_arr[0] = len(self._playback_audio)
        self.state = PlaybackState.STOPPED
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------
    # Render (audio callback or test-driven)
    # ------------------------------------------------------------------

    def _render(self, n_frames: int) -> np.ndarray:
        """Fill the pre-allocated buffer with n_frames samples and return a view.

        PAUSED or STOPPED → silence, position unchanged.
        PLAYING           → audio × volume + metronome clicks; auto-stops at EOF.
        """
        buf = self._output_buf[:n_frames]

        if self.state != PlaybackState.PLAYING:
            buf[:] = 0.0
            return buf

        with self._lock:
            pos = self._pos_arr[0]

        start_pos = pos
        n_audio = len(self._playback_audio)

        if self._loop_start is not None:
            # Loop-region rendering: wrap at loop_end → loop_start.
            loop_s = int(self._loop_start * self._fs)
            loop_e = int(self._loop_end * self._fs)
            written = 0
            while written < n_frames:
                if pos >= loop_e:
                    pos = loop_s
                chunk = min(n_frames - written, loop_e - pos)
                buf[written:written + chunk] = self._playback_audio[pos:pos + chunk]
                pos += chunk
                written += chunk
            buf[:n_frames] *= self.volume
        else:
            # Normal playback: read up to available samples, fill rest with zeros.
            available = n_audio - pos
            if available <= 0:
                buf[:] = 0.0
                self.state = PlaybackState.STOPPED
                return buf
            to_read = min(n_frames, available)
            buf[:to_read] = self._playback_audio[pos:pos + to_read]
            if to_read < n_frames:
                buf[to_read:n_frames] = 0.0
            pos += to_read
            buf[:n_frames] *= self.volume
            if pos >= n_audio:
                self.state = PlaybackState.STOPPED

        # Mix metronome click spikes at exact beat-sample positions (T38: ±2 ms).
        if self.metronome_enabled:
            beat_f = self._fs * 60.0 / self._bpm
            beat_no = math.ceil(start_pos / beat_f) if start_pos > 0 else 0
            while True:
                beat_samp = round(beat_no * beat_f)
                if beat_samp >= start_pos + n_frames:
                    break
                buf_idx = beat_samp - start_pos
                if 0 <= buf_idx < n_frames:
                    buf[buf_idx] += _CLICK_AMPLITUDE * self.metronome_volume
                beat_no += 1

        with self._lock:
            self._pos_arr[0] = pos

        return buf
