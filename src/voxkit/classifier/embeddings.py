# SPDX-License-Identifier: GPL-3.0-or-later
"""EmbeddingExtractor — PANNs/BEATs ONNX inference (Component 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class ModelLoadError(Exception):
    pass


class ModelSchemaError(Exception):
    pass


# ---------------------------------------------------------------
# Substrate config
# ---------------------------------------------------------------

_SUBSTRATE: dict[str, dict] = {
    "panns_cnn14": {
        "embedding_dim": 2048,
        "input_length": 16_000,   # 1 s @ 16 kHz
        "input_name": "waveform",
        "output_name": "embedding",
    },
    "beats": {
        "embedding_dim": 768,
        # 2 s @ 16 kHz. 10 s was too long for onset extraction — each window
        # captured many adjacent hits, making class embeddings indistinguishable.
        # The ONNX model was exported with dynamic_axes on the frames dimension,
        # so variable-length fbank (≈198 frames for 2 s) is accepted.
        "input_length": 32_000,
        "input_name": "fbank_features",  # ONNX model takes (1, T_frames, 128) fbank
        "output_name": "output",
    },
}

# BEATs fbank parameters matching BEATs.preprocess() defaults.
_BEATS_FBANK_MEAN: float = 15.41663
_BEATS_FBANK_STD: float = 6.55582

# Fraction of the extraction window that falls *before* the onset.
# 0.20 → 20 % pre-onset silence / 80 % attack + decay, which captures
# more of the percussion transient than a symmetric (0.50) window.
PRE_ONSET_FRACTION: float = 0.20

_REQUIRED_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------
# ONNX loader — thin wrapper so tests can patch it
# ---------------------------------------------------------------

def _load_onnx_session(path: Path, *, providers: list[str]):
    try:
        import onnxruntime as ort
        return ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:
        raise ModelLoadError(f"ONNX load failed: {exc}") from exc


# ---------------------------------------------------------------
# Schema validation (T25)
# ---------------------------------------------------------------

def _validate_session_schema(session, substrate_id: str) -> None:
    """Validate input tensor name. Skips when get_inputs() is not a list
    (test stubs return a MagicMock, real sessions return a list)."""
    raw_inputs = session.get_inputs()
    if not isinstance(raw_inputs, list):
        return
    expected_input = _SUBSTRATE[substrate_id]["input_name"]
    actual_input = raw_inputs[0].name if raw_inputs else None
    if actual_input != expected_input:
        raise ModelSchemaError(
            f"ONNX input tensor name mismatch: "
            f"expected '{expected_input}', got '{actual_input}'"
        )


# ---------------------------------------------------------------
# EmbeddingExtractor
# ---------------------------------------------------------------

class EmbeddingExtractor:
    required_sample_rate: int = _REQUIRED_SAMPLE_RATE

    _DEFAULT_MODEL_FILES: dict[str, str] = {
        "panns_cnn14": "panns_cnn14_16k.onnx",
        "beats": "beats_iter3plus_as2m.onnx",
    }

    @classmethod
    def from_default(cls, substrate_id: str) -> "EmbeddingExtractor":
        """Create an EmbeddingExtractor using the standard model path for the given substrate."""
        if substrate_id not in cls._DEFAULT_MODEL_FILES:
            raise ValueError(f"Unknown substrate: {substrate_id!r}")
        repo_root = Path(__file__).parent.parent.parent.parent
        model_path = repo_root / "models" / cls._DEFAULT_MODEL_FILES[substrate_id]
        return cls(model_path, substrate_id)

    def __init__(self, onnx_path: Path, substrate_id: str) -> None:
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if substrate_id not in _SUBSTRATE:
            raise ValueError(f"Unknown substrate: {substrate_id!r}")
        self._substrate_id = substrate_id
        self._config = _SUBSTRATE[substrate_id]
        self._session = _load_onnx_session(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        _validate_session_schema(self._session, substrate_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def substrate_id(self) -> str:
        return self._substrate_id

    @property
    def embedding_dim(self) -> int:
        return self._config["embedding_dim"]

    @property
    def input_length(self) -> int:
        return self._config["input_length"]

    # ------------------------------------------------------------------
    # Single-window extraction
    # ------------------------------------------------------------------

    def _compute_beats_fbank(self, window: np.ndarray) -> np.ndarray:
        """Return normalised log-mel fbank (1, T_frames, 128) for the BEATs substrate."""
        try:
            import torch
            import torchaudio.compliance.kaldi as ta_kaldi
        except ImportError as exc:
            raise ImportError(
                "torchaudio is required for BEATs inference. "
                "Install it with: pip install torchaudio"
            ) from exc

        waveform = torch.from_numpy(window).unsqueeze(0)  # (1, L)
        fbank = ta_kaldi.fbank(
            waveform,
            num_mel_bins=128,
            sample_frequency=16_000,
            frame_length=25,
            frame_shift=10,
        )  # (T_frames, 128)
        fbank = (fbank - _BEATS_FBANK_MEAN) / (2.0 * _BEATS_FBANK_STD)
        return fbank.unsqueeze(0).numpy()  # (1, T_frames, 128)

    def extract(self, window: np.ndarray) -> np.ndarray:
        if window.ndim != 1:
            raise ValueError("window must be mono (1-D ndarray)")
        if window.dtype != np.float32:
            raise ValueError(
                f"window dtype must be float32, got {window.dtype}"
            )
        if len(window) != self.input_length:
            raise ValueError(
                f"window length {len(window)} does not match "
                f"expected input length {self.input_length}"
            )
        if self._substrate_id == "beats":
            inp = self._compute_beats_fbank(window)  # (1, T_frames, 128)
        else:
            inp = window[np.newaxis, :]               # (1, L)
        outputs = self._session.run(None, {self._config["input_name"]: inp})
        emb = outputs[0][0]  # (1, D) → (D,)
        return emb.astype(np.float32)

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_batch(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        if not windows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.stack([self.extract(w) for w in windows]).astype(np.float32)

    # ------------------------------------------------------------------
    # Onset-driven extraction
    # ------------------------------------------------------------------

    def _slice_window(
        self, audio: np.ndarray, center: int
    ) -> np.ndarray:
        """Return a zero-padded input_length window starting PRE_ONSET_FRACTION
        before `center` so 80 % of the window captures the attack and decay."""
        n = len(audio)
        pre = int(self.input_length * PRE_ONSET_FRACTION)
        start = center - pre
        end = start + self.input_length
        win = np.zeros(self.input_length, dtype=np.float32)
        src_start = max(0, start)
        src_end = min(n, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        win[dst_start:dst_end] = audio[src_start:src_end]
        return win

    def extract_at_onsets(
        self,
        audio: np.ndarray,
        onset_times_s: list[float],
        sample_rate: int,
    ) -> np.ndarray:
        n = len(audio)
        results: list[np.ndarray] = []
        for t in onset_times_s:
            center = int(round(t * sample_rate))
            if center < 0 or center >= n:
                raise IndexError(
                    f"Onset at {t:.4f}s (sample {center}) is outside "
                    f"audio buffer of {n} samples"
                )
            results.append(self.extract(self._slice_window(audio, center)))
        if not results:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.stack(results).astype(np.float32)

    def extract_at_onsets_with_rms(
        self,
        audio: np.ndarray,
        onset_times_s: list[float],
        sample_rate: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (embeddings, rms_values) for each onset.

        rms_values[i] is the RMS of the audio window centred on onset i,
        used as the 5th/95th percentile mapping input (§11 Component 5).
        """
        n = len(audio)
        embeddings: list[np.ndarray] = []
        rms_values: list[float] = []
        for t in onset_times_s:
            center = int(round(t * sample_rate))
            if center < 0 or center >= n:
                raise IndexError(
                    f"Onset at {t:.4f}s (sample {center}) is outside "
                    f"audio buffer of {n} samples"
                )
            win = self._slice_window(audio, center)
            embeddings.append(self.extract(win))
            rms_values.append(
                float(np.sqrt(np.mean(win.astype(np.float64) ** 2)))
            )
        if not embeddings:
            return (
                np.zeros((0, self.embedding_dim), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        return (
            np.stack(embeddings).astype(np.float32),
            np.array(rms_values, dtype=np.float32),
        )
