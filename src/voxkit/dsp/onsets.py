# SPDX-License-Identifier: GPL-3.0-or-later
"""CNN onset detector backed by ONNX runtime (Component 4).

At runtime the detector loads models/onset_cnn.onnx and uses it for all
detection.  If the model file is absent (e.g. a fresh clone before training),
it falls back to the legacy energy-flux detector with a warning so tests and
development workflows continue to work.

Public interface is unchanged:
    detector = OnsetDetector(sample_rate=16_000)
    onset_times = detector.detect(audio)              # list[float], seconds
    onset_times = detector.detect(audio, click_track) # with click suppression
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

_REQUIRED_FS = 16_000
_CLICK_GUARD_MS = 15.0

# Mel-spectrogram parameters — must match scripts/train_onset_detector.py
_HOP = 80         # 5 ms at 16 kHz — max timing quantisation error ≤ 2.5 ms
_N_FFT = 512
_N_MELS = 40
_FMIN = 27.5
_FMAX = 8_000.0

_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent.parent / "models" / "onset_cnn.onnx"
)


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class AudioContainsNonFinite(Exception):
    pass


# ---------------------------------------------------------------
# Mel computation (shared by detector and training script)
# ---------------------------------------------------------------

def _compute_mel(audio: np.ndarray) -> np.ndarray:
    """Return log-mel spectrogram (T, _N_MELS) float32."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=_REQUIRED_FS, n_fft=_N_FFT, hop_length=_HOP,
        n_mels=_N_MELS, fmin=_FMIN, fmax=_FMAX,
    )
    return librosa.power_to_db(mel + 1e-8).T.astype(np.float32)


# ---------------------------------------------------------------
# Peak picker (used by CNN path)
# ---------------------------------------------------------------

def _peak_pick(
    probs: np.ndarray,
    threshold: float = 0.5,
    min_gap_s: float = 0.050,
) -> list[float]:
    min_gap = max(1, int(min_gap_s * _REQUIRED_FS / _HOP))
    onsets: list[float] = []
    i = 0
    while i < len(probs):
        if probs[i] > threshold:
            w_end = min(i + min_gap, len(probs))
            peak_i = i + int(np.argmax(probs[i:w_end]))
            onsets.append(float(peak_i * _HOP / _REQUIRED_FS))
            i = peak_i + min_gap
        else:
            i += 1
    return onsets


# ---------------------------------------------------------------
# Noise gate (shared by CNN path; energy-flux has it built-in)
# ---------------------------------------------------------------

def _apply_noise_gate(onsets_s: list[float], audio: np.ndarray) -> list[float]:
    """Drop onsets whose local peak is below 6 dB above the 200 ms noise floor.

    Noise floor is the minimum hop-frame RMS within the first 200 ms.  Using
    the minimum (not mean) makes the estimate robust to the first onset landing
    inside the 200 ms window: only quiet background frames contribute.
    """
    n = len(audio)
    n_noise = min(n, int(_NOISE_FLOOR_MS / 1000.0 * _REQUIRED_FS))
    if n_noise < _HOP:
        return onsets_s
    frame_rms = [
        float(np.sqrt(np.mean(audio[i:i + _HOP].astype(np.float64) ** 2)))
        for i in range(0, n_noise - _HOP + 1, _HOP)
    ]
    noise_rms = min(frame_rms)
    gate = noise_rms * _NOISE_GATE_AMP
    if gate == 0.0:
        return onsets_s
    filtered: list[float] = []
    for t in onsets_s:
        sample = int(round(t * _REQUIRED_FS))
        start = max(0, sample - _HOP)
        end = min(n, sample + _HOP)
        if float(np.max(np.abs(audio[start:end].astype(np.float64)))) >= gate:
            filtered.append(t)
    return filtered


# ---------------------------------------------------------------
# Click suppression (shared by both paths)
# ---------------------------------------------------------------

def _refine_timing(onsets_s: list[float], audio: np.ndarray, window_s: float = 0.005) -> list[float]:
    """Snap each CNN-detected onset to the nearest waveform energy peak.

    The mel spectrogram has 5 ms frame resolution; this step finds the true
    transient peak in the original audio within ±window_s (default ±5 ms),
    giving sub-millisecond timing precision without retraining.
    """
    window = int(window_s * _REQUIRED_FS)
    refined: list[float] = []
    for t in onsets_s:
        center = int(round(t * _REQUIRED_FS))
        start = max(0, center - window)
        end = min(len(audio), center + window + 1)
        seg = np.abs(audio[start:end].astype(np.float64))
        peak = int(np.argmax(seg))
        refined.append(float((start + peak) / _REQUIRED_FS))
    return refined


def _suppress_clicks(onsets_s: list[float], click_track: list[float] | None) -> list[float]:
    if not click_track:
        return onsets_s
    guard = _CLICK_GUARD_MS / 1000.0
    return [t for t in onsets_s if not any(abs(t - c) <= guard for c in click_track)]


# ---------------------------------------------------------------
# Legacy energy-flux detector (fallback when ONNX model absent)
# ---------------------------------------------------------------

_NOISE_FLOOR_MS = 200.0
_NOISE_GATE_DB = 6.0
_NOISE_GATE_AMP = 10.0 ** (_NOISE_GATE_DB / 20.0)


def _energy_flux_detect(audio: np.ndarray, hop: int = 80) -> list[float]:
    """Energy-flux onset detector used as fallback (no model required)."""
    n = len(audio)
    if n < hop:
        return []
    n_frames = n // hop
    energies = np.array([
        float(np.sum(audio[i * hop:(i + 1) * hop].astype(np.float64) ** 2))
        for i in range(n_frames)
    ])
    odf = np.maximum(np.diff(energies, prepend=energies[0]), 0.0)
    peak = odf.max()
    if peak == 0:
        return []
    threshold = 0.2 * peak
    min_gap = max(1, int(0.05 * _REQUIRED_FS / hop))
    raw: list[float] = []
    i = 0
    while i < len(odf):
        if odf[i] > threshold:
            w_end = min(i + min_gap, len(odf))
            peak_idx = i + int(np.argmax(odf[i:w_end]))
            raw.append(float(peak_idx * hop / _REQUIRED_FS))
            i = peak_idx + min_gap
        else:
            i += 1

    # Noise gate
    n_noise = min(n, int(_NOISE_FLOOR_MS / 1000.0 * _REQUIRED_FS))
    if n_noise > 0:
        noise_rms = float(np.sqrt(np.mean(audio[:n_noise].astype(np.float64) ** 2)))
        gate = noise_rms * _NOISE_GATE_AMP
        if gate > 0:
            filtered: list[float] = []
            for t in raw:
                sample = int(round(t * _REQUIRED_FS))
                start = max(0, sample - hop)
                end = min(n, sample + hop)
                if float(np.max(np.abs(audio[start:end].astype(np.float64)))) >= gate:
                    filtered.append(t)
            raw = filtered

    return raw


# ---------------------------------------------------------------
# OnsetDetector
# ---------------------------------------------------------------

class OnsetDetector:
    """Onset detector: CNN (ONNX) when model is present, energy-flux otherwise.

    Parameters
    ----------
    sample_rate : int
        Must be 16 000 Hz.
    onnx_path : Path | None
        Path to the onset_cnn.onnx model.  Defaults to models/onset_cnn.onnx
        relative to the repo root.  Pass a Path explicitly in tests that
        provide a mock session.
    """

    def __init__(
        self,
        sample_rate: int,
        onnx_path: Path | str | None = None,
    ) -> None:
        if sample_rate != _REQUIRED_FS:
            raise ValueError(
                f"OnsetDetector requires sample_rate={_REQUIRED_FS}; got {sample_rate}"
            )
        self._fs = sample_rate
        self._session = None
        self._fallback = False

        model_path = Path(onnx_path) if onnx_path is not None else _DEFAULT_MODEL_PATH
        if model_path.exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"OnsetDetector: failed to load ONNX model ({exc}); "
                    "falling back to energy-flux detector.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fallback = True
        else:
            import warnings
            warnings.warn(
                f"OnsetDetector: model not found at {model_path}; "
                "falling back to energy-flux detector. "
                "Run scripts/train_onset_detector.py to generate the model.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fallback = True

    # ------------------------------------------------------------------
    # Public: detect
    # ------------------------------------------------------------------

    def detect(
        self,
        audio: np.ndarray,
        click_track: list[float] | None = None,
    ) -> list[float]:
        """Return onset times in seconds.

        Parameters
        ----------
        audio       : mono float32 at 16 kHz
        click_track : optional click times (seconds); onsets within ±15 ms suppressed
        """
        if audio.ndim != 1:
            raise ValueError("OnsetDetector requires mono (1-D) audio")
        if not np.all(np.isfinite(audio)):
            raise AudioContainsNonFinite("Audio contains NaN or Inf")

        if self._fallback:
            onsets = _energy_flux_detect(audio, hop=_REQUIRED_FS // 200)
        else:
            onsets = self._cnn_detect(audio)

        return _suppress_clicks(onsets, click_track)

    # ------------------------------------------------------------------
    # Public: click_guard_fire_rate
    # ------------------------------------------------------------------

    def click_guard_fire_rate(
        self,
        click_positions: list[int],
        raw_onsets: list[int],
        window_bars: int,
        bpm: float,
        sample_rate: int,
    ) -> float:
        """Fraction of recent click positions that had a raw onset within ±15 ms."""
        if not click_positions:
            return 0.0
        bar_dur_s = 60.0 / bpm
        window_samples = int(window_bars * bar_dur_s * sample_rate)
        end_sample = max(click_positions)
        start_sample = end_sample - window_samples
        recent = [c for c in click_positions if c >= start_sample]
        if not recent:
            return 0.0
        guard = int(0.015 * sample_rate)
        hits = sum(
            1 for c in recent
            if any(abs(c - o) <= guard for o in raw_onsets)
        )
        return hits / len(recent)

    # ------------------------------------------------------------------
    # Private: CNN inference
    # ------------------------------------------------------------------

    def _cnn_detect(self, audio: np.ndarray) -> list[float]:
        hop = _REQUIRED_FS // 200  # 80 samples = 5 ms (energy-flux hop, unused here)
        if len(audio) < _HOP:
            return []
        mel = _compute_mel(audio)          # (T, 40)
        if len(mel) == 0:
            return []
        inp = mel[np.newaxis, np.newaxis]  # (1, 1, T, 40)
        logits = self._session.run(None, {"mel_spectrogram": inp})[0][0]  # (T,)
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        onsets = _peak_pick(probs.astype(np.float32))
        onsets = _refine_timing(onsets, audio)
        return _apply_noise_gate(onsets, audio)
