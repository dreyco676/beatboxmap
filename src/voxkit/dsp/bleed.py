# SPDX-License-Identifier: GPL-3.0-or-later
"""ClickBleedHandler: FIR subtraction, IR estimation, re-estimation, history (§5.1.1, Q79, Q80)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class NoCalibrationData(Exception):
    pass


class SampleRateMismatch(Exception):
    pass


class AudioContainsNonFinite(Exception):
    pass


class NonFiniteIR(Exception):
    pass


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-30))


# ---------------------------------------------------------------
# IR estimation (Wiener deconvolution in frequency domain)
# ---------------------------------------------------------------

def estimate_ir(
    reference: np.ndarray,
    observed: np.ndarray,
    ir_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Estimate a causal FIR IR via Wiener-Hopf normal equations (Toeplitz solve).

    Handles the np.convolve(signal, ir, mode='same') convention by trimming the
    first ir_length samples from both ref and obs, ensuring the region used for
    correlation is free from mode='same' boundary clipping artifacts.
    """
    from scipy.linalg import solve_toeplitz as _solve_toeplitz

    ref = reference.astype(np.float64)
    obs = observed.astype(np.float64)

    # Trim the first (ir_length-1)//2 samples — the exact mode='same' shift.
    # This removes the first click from ref (which has its response clipped in obs),
    # making Rxx and Rxy consistent over the same set of clicks.
    shift = (ir_length - 1) // 2
    if len(ref) > 2 * shift:
        ref = ref[shift:]
        obs = obs[shift:]

    Rxx_full = np.correlate(ref, ref, mode="full")
    center = len(ref) - 1
    rxx_vec = Rxx_full[center : center + ir_length]

    if rxx_vec[0] < 1e-30:
        return np.zeros(ir_length, dtype=np.float32)

    Rxy_full = np.correlate(obs, ref, mode="full")
    rxy_vec = Rxy_full[center - shift : center - shift + ir_length]

    try:
        h = _solve_toeplitz((rxx_vec, rxx_vec), rxy_vec)
    except Exception:
        h = rxy_vec / (rxx_vec[0] + 1e-30)

    return h.astype(np.float32)


# ---------------------------------------------------------------
# ClickBleedHandler
# ---------------------------------------------------------------

class ClickBleedHandler:
    """Subtract headphone-bleed from audio using a pre-estimated FIR IR.

    Active callback path: Python-default (Q67 amended / Q76 GIL contract).
    Streaming: fixed-size chunks (CHUNK_SAMPLES) to bound peak RSS.
    Thread safety: IR swap is protected by _ir_lock; pointer replacement is
    atomic so concurrent clean() + reestimate_in_silent_window() is safe.
    """

    CHUNK_SAMPLES: int = 16_000 * 4  # 4 s at 16 kHz

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._ir: np.ndarray | None = None
        self._ir_lock = threading.Lock()
        self._cal_reference: np.ndarray | None = None
        self._cal_observed: np.ndarray | None = None
        # Transient per-chunk state; only valid inside clean().
        self._current_ref_chunk: np.ndarray | None = None

    # ------------------------------------------------------------------
    # IR management
    # ------------------------------------------------------------------

    def set_ir(self, ir: np.ndarray) -> None:
        if not np.all(np.isfinite(ir)):
            raise NonFiniteIR("IR contains NaN or Inf")
        with self._ir_lock:
            self._ir = ir.astype(np.float32).copy()

    def get_ir(self) -> np.ndarray | None:
        with self._ir_lock:
            return self._ir.copy() if self._ir is not None else None

    def _get_ir(self) -> np.ndarray | None:
        """Thread-safe IR reference snapshot (no copy; caller must not mutate)."""
        with self._ir_lock:
            return self._ir

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def set_calibration(self, reference: np.ndarray, observed: np.ndarray) -> None:
        self._cal_reference = reference.astype(np.float32)
        self._cal_observed = observed.astype(np.float32)

    def get_quality_attenuation_db(self) -> float:
        """Post-subtraction click residual ratio in dB (Q79).

        More positive = better attenuation.
        Raises NoCalibrationData if set_calibration() has not been called.
        """
        if self._cal_reference is None:
            raise NoCalibrationData("Call set_calibration() before get_quality_attenuation_db()")
        from voxkit.dsp.bleed_metrics import residual_ratio_db
        cleaned = self.clean(self._cal_observed, click_reference=self._cal_reference)
        return residual_ratio_db(cleaned=cleaned, calibration=self._cal_observed)

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(
        self,
        audio: np.ndarray,
        click_reference: np.ndarray | None = None,
        input_sample_rate: int | None = None,
    ) -> np.ndarray:
        """Subtract bleed from audio; process in fixed-size chunks (streaming)."""
        if not np.all(np.isfinite(audio)):
            raise AudioContainsNonFinite("Audio contains NaN or Inf")
        if input_sample_rate is not None and input_sample_rate != self._sample_rate:
            raise SampleRateMismatch(
                f"IR estimated at {self._sample_rate} Hz; "
                f"audio at {input_sample_rate} Hz — silent resampling refused"
            )
        if len(audio) == 0:
            return audio.copy()

        result = np.empty_like(audio)
        for start in range(0, len(audio), self.CHUNK_SAMPLES):
            end = min(start + self.CHUNK_SAMPLES, len(audio))
            if click_reference is not None:
                self._current_ref_chunk = click_reference[start:end]
            else:
                self._current_ref_chunk = None
            result[start:end] = self._process_chunk(audio[start:end])
        self._current_ref_chunk = None
        return result

    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process one chunk. Uses self._current_ref_chunk set by clean()."""
        ir = self._get_ir()
        ref = self._current_ref_chunk
        if ir is None or ref is None:
            return chunk.copy()
        bleed = np.convolve(ref, ir, mode="same")
        return (chunk - bleed).astype(np.float32)

    # ------------------------------------------------------------------
    # Re-estimation
    # ------------------------------------------------------------------

    def reestimate_in_silent_window(
        self, audio: np.ndarray, click_reference: np.ndarray
    ) -> None:
        """Estimate a new IR from a silent window; update only if it improves residual."""
        current_ir = self.get_ir()
        ir_length = len(current_ir) if current_ir is not None else 256

        new_ir = estimate_ir(
            reference=click_reference,
            observed=audio,
            ir_length=ir_length,
            sample_rate=self._sample_rate,
        )

        if current_ir is None:
            self.set_ir(new_ir)
            return

        current_residual = _rms(audio - np.convolve(click_reference, current_ir, "same"))
        new_residual = _rms(audio - np.convolve(click_reference, new_ir, "same"))

        # Require meaningful improvement to avoid accepting overfitted-to-noise IRs.
        # A correct Wiener estimate can marginally reduce residual on any input;
        # 1% guards against that without affecting genuine bleed windows (>10% improvement).
        if new_residual < current_residual * 0.99:
            self.set_ir(new_ir)


# ---------------------------------------------------------------
# Silent-window detectors
# ---------------------------------------------------------------

class ActiveSilentWindowDetector:
    """Detects silent regions by local RMS below a threshold."""

    def __init__(self, rms_threshold: float, sample_rate: int) -> None:
        self._threshold = rms_threshold
        self._fs = sample_rate

    def find_silent_regions(
        self, audio: np.ndarray, min_duration_ms: float = 200.0
    ) -> list[tuple[float, float]]:
        """Return list of (start_s, end_s) silent regions."""
        hop = max(1, self._fs // 100)  # 10 ms hop
        min_frames = max(1, int(min_duration_ms / 10))

        is_silent = []
        for i in range(0, len(audio), hop):
            chunk = audio[i : i + hop]
            rms_val = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            is_silent.append(rms_val < self._threshold)

        regions: list[tuple[float, float]] = []
        in_silent = False
        start_idx = 0
        for i, silent in enumerate(is_silent):
            if silent and not in_silent:
                start_idx = i
                in_silent = True
            elif not silent and in_silent:
                if i - start_idx >= min_frames:
                    regions.append((start_idx * hop / self._fs, i * hop / self._fs))
                in_silent = False
        if in_silent and len(is_silent) - start_idx >= min_frames:
            regions.append((start_idx * hop / self._fs, len(audio) / self._fs))
        return regions


class PassiveSilentWindowDetector:
    """Detects silent regions from an external VAD signal (frame-rate)."""

    def __init__(self, sample_rate: int, frame_size: int) -> None:
        self._fs = sample_rate
        self._frame_size = frame_size

    def find_silent_regions_from_vad(
        self, vad: np.ndarray, min_frames: int = 2
    ) -> list[tuple[float, float]]:
        """Return list of (start_s, end_s) from runs of 0 in the VAD."""
        regions: list[tuple[float, float]] = []
        in_silent = False
        start_idx = 0
        for i, v in enumerate(vad):
            if v == 0 and not in_silent:
                start_idx = i
                in_silent = True
            elif v != 0 and in_silent:
                if i - start_idx >= min_frames:
                    regions.append((
                        start_idx * self._frame_size / self._fs,
                        i * self._frame_size / self._fs,
                    ))
                in_silent = False
        if in_silent and len(vad) - start_idx >= min_frames:
            regions.append((
                start_idx * self._frame_size / self._fs,
                len(vad) * self._frame_size / self._fs,
            ))
        return regions


# ---------------------------------------------------------------
# BleedIRHistory (v0.9 carry: two protected slots)
# ---------------------------------------------------------------

@dataclass
class IREntry:
    ir: np.ndarray
    protected: bool


class BleedIRHistory:
    """Rolling history of IR estimates with protected (never-evicted) slots."""

    def __init__(self, unprotected_capacity: int = 10) -> None:
        self._unprotected_capacity = unprotected_capacity
        self._entries: list[IREntry] = []

    @property
    def entries(self) -> list[IREntry]:
        return list(self._entries)

    def append(self, ir: np.ndarray, *, protected: bool) -> None:
        if not protected:
            unprotected = [e for e in self._entries if not e.protected]
            if len(unprotected) >= self._unprotected_capacity:
                # Evict oldest unprotected (use identity, not ==, because
                # @dataclass __eq__ on numpy arrays raises ValueError).
                oldest = unprotected[0]
                for i, e in enumerate(self._entries):
                    if e is oldest:
                        del self._entries[i]
                        break
        self._entries.append(IREntry(ir=ir.copy(), protected=protected))


# ---------------------------------------------------------------
# Bleed banner
# ---------------------------------------------------------------

def should_show_bleed_banner(attenuation_db: float, override: bool) -> bool:
    """Return True if the bleed banner should be shown (Q79 < 10 dB threshold)."""
    if override:
        return False
    return attenuation_db < 10.0
