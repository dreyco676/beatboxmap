# SPDX-License-Identifier: GPL-3.0-or-later
"""
Validation tests for converted ONNX model files.

These tests confirm that the ONNX files produced by the conversion scripts
(scripts/convert_beats_to_onnx.py, scripts/convert_panns_to_onnx.py) are
correct and compatible with EmbeddingExtractor.

All tests skip automatically when the model file is absent from models/
(the files are gitignored; they must be generated locally first).

============================================================
TEST LIST
============================================================

PANNs CNN14 (models/panns_cnn14_16k.onnx)
  T01  Model file exists (prerequisite; other PANNs tests skip if absent)
  T02  ONNX session loads without error
  T03  ONNX input tensor is named "waveform"
  T04  ONNX has exactly one input (no stray tensors)
  T05  ONNX output tensor is named "embedding"
  T06  Running inference returns shape (1, 2048)
  T07  Output dtype is float32
  T08  EmbeddingExtractor accepts the model (schema validation passes)
  T09  EmbeddingExtractor.extract() returns a (2048,) vector
  T10  Embedding values are all finite (no NaN / Inf)
  T11  Two distinct audio windows produce different embeddings (model is live)

BEATs (models/beats_iter3plus_as2m.onnx)
  T12  Model file exists (prerequisite; other BEATs tests skip if absent)
  T13  ONNX session loads without error
  T14  ONNX input tensor is named "audio_data"
  T15  ONNX has exactly one input (padding_mask folded as constant in export)
  T16  ONNX output tensor is named "output"
  T17  Running inference returns shape (1, 768)
  T18  Output dtype is float32
  T19  EmbeddingExtractor accepts the model (schema validation passes)
  T20  EmbeddingExtractor.extract() returns a (768,) vector
  T21  Embedding values are all finite (no NaN / Inf)
  T22  Two distinct audio windows produce different embeddings (model is live)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------
# Paths (relative to repo root; tests run from that directory)
# ---------------------------------------------------------------

_MODELS_DIR = Path(__file__).parent.parent / "models"
_PANNS_PATH = _MODELS_DIR / "panns_cnn14_16k.onnx"
_BEATS_PATH = _MODELS_DIR / "beats_iter3plus_as2m.onnx"

# ---------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------

def _skip_if_missing(path: Path) -> Path:
    """Return path or skip the test if the file does not exist."""
    if not path.exists():
        pytest.skip(
            f"Model file not found: {path}\n"
            f"Run the conversion script first:\n"
            f"  python scripts/convert_{'panns' if 'panns' in path.name else 'beats'}"
            f"_to_onnx.py --help"
        )
    return path


def _ort_session(path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------
# PANNs CNN14 — T01-T11
# ---------------------------------------------------------------

def test_T01_panns_file_exists():
    """Prerequisite: the ONNX file must be present before other tests run."""
    _skip_if_missing(_PANNS_PATH)


def test_T02_panns_onnx_loads():
    _skip_if_missing(_PANNS_PATH)
    sess = _ort_session(_PANNS_PATH)
    assert sess is not None


def test_T03_panns_input_named_waveform():
    _skip_if_missing(_PANNS_PATH)
    sess = _ort_session(_PANNS_PATH)
    input_names = [i.name for i in sess.get_inputs()]
    assert "waveform" in input_names, (
        f"Expected input named 'waveform'; got {input_names}"
    )


def test_T04_panns_exactly_one_input():
    _skip_if_missing(_PANNS_PATH)
    sess = _ort_session(_PANNS_PATH)
    inputs = sess.get_inputs()
    assert len(inputs) == 1, (
        f"Expected 1 input tensor; got {len(inputs)}: {[i.name for i in inputs]}"
    )


def test_T05_panns_output_named_embedding():
    _skip_if_missing(_PANNS_PATH)
    sess = _ort_session(_PANNS_PATH)
    output_names = [o.name for o in sess.get_outputs()]
    assert "embedding" in output_names, (
        f"Expected output named 'embedding'; got {output_names}"
    )


def test_T06_panns_output_shape_1x2048():
    _skip_if_missing(_PANNS_PATH)
    sess = _ort_session(_PANNS_PATH)
    audio = np.zeros((1, 16_000), dtype=np.float32)
    out = sess.run(None, {"waveform": audio})
    assert out[0].shape == (1, 2048), (
        f"Expected shape (1, 2048); got {out[0].shape}"
    )


def test_T07_panns_output_dtype_float32():
    _skip_if_missing(_PANNS_PATH)
    sess = _ort_session(_PANNS_PATH)
    audio = np.zeros((1, 16_000), dtype=np.float32)
    out = sess.run(None, {"waveform": audio})
    assert out[0].dtype == np.float32, (
        f"Expected float32; got {out[0].dtype}"
    )


def test_T08_panns_embedding_extractor_accepts_model():
    _skip_if_missing(_PANNS_PATH)
    from voxkit.classifier.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(
        onnx_path=_PANNS_PATH, substrate_id="panns_cnn14"
    )
    assert extractor.embedding_dim == 2048
    assert extractor.input_length == 16_000


def test_T09_panns_extract_returns_2048_vector():
    _skip_if_missing(_PANNS_PATH)
    from voxkit.classifier.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(onnx_path=_PANNS_PATH, substrate_id="panns_cnn14")
    window = np.random.default_rng(0).standard_normal(16_000).astype(np.float32)
    emb = extractor.extract(window)
    assert emb.shape == (2048,), f"Expected (2048,); got {emb.shape}"


def test_T10_panns_embedding_is_finite():
    _skip_if_missing(_PANNS_PATH)
    from voxkit.classifier.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(onnx_path=_PANNS_PATH, substrate_id="panns_cnn14")
    window = np.random.default_rng(1).standard_normal(16_000).astype(np.float32)
    emb = extractor.extract(window)
    assert np.all(np.isfinite(emb)), "Embedding contains NaN or Inf"


def test_T11_panns_different_inputs_different_embeddings():
    """The model must be sensitive to its input — not a trivial constant function."""
    _skip_if_missing(_PANNS_PATH)
    from voxkit.classifier.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(onnx_path=_PANNS_PATH, substrate_id="panns_cnn14")
    rng = np.random.default_rng(42)
    w1 = rng.standard_normal(16_000).astype(np.float32)
    w2 = rng.standard_normal(16_000).astype(np.float32)
    e1 = extractor.extract(w1)
    e2 = extractor.extract(w2)
    assert not np.allclose(e1, e2), (
        "Two different audio windows produced identical embeddings; "
        "model may not have loaded correctly"
    )


# ---------------------------------------------------------------
# BEATs — T12-T22
# ---------------------------------------------------------------

def test_T12_beats_file_exists():
    """Prerequisite: the ONNX file must be present before other tests run."""
    _skip_if_missing(_BEATS_PATH)


def test_T13_beats_onnx_loads():
    _skip_if_missing(_BEATS_PATH)
    sess = _ort_session(_BEATS_PATH)
    assert sess is not None


def test_T14_beats_input_named_fbank_features():
    _skip_if_missing(_BEATS_PATH)
    sess = _ort_session(_BEATS_PATH)
    input_names = [i.name for i in sess.get_inputs()]
    assert "fbank_features" in input_names, (
        f"Expected input named 'fbank_features'; got {input_names}. "
        f"Re-run convert_beats_to_onnx.py — the model must export the "
        f"post-fbank encoder, not the raw-audio path."
    )


def test_T15_beats_exactly_one_input():
    """padding_mask must be folded as a constant during export, not left as
    a runtime input — otherwise EmbeddingExtractor.extract() would fail because
    it only passes audio_data."""
    _skip_if_missing(_BEATS_PATH)
    sess = _ort_session(_BEATS_PATH)
    inputs = sess.get_inputs()
    assert len(inputs) == 1, (
        f"Expected 1 input (audio_data only); got {len(inputs)}: "
        f"{[i.name for i in inputs]}. "
        f"Re-run convert_beats_to_onnx.py — the wrapper must compute "
        f"padding_mask internally, not expose it as a second input."
    )


def test_T16_beats_output_named_output():
    _skip_if_missing(_BEATS_PATH)
    sess = _ort_session(_BEATS_PATH)
    output_names = [o.name for o in sess.get_outputs()]
    assert "output" in output_names, (
        f"Expected output named 'output'; got {output_names}"
    )


def test_T17_beats_output_shape_1x768():
    _skip_if_missing(_BEATS_PATH)
    sess = _ort_session(_BEATS_PATH)
    # 998 fbank frames = 10 s @ 16 kHz with 10 ms frame shift, 128 mel bins
    fbank = np.zeros((1, 998, 128), dtype=np.float32)
    out = sess.run(None, {"fbank_features": fbank})
    assert out[0].shape == (1, 768), (
        f"Expected shape (1, 768); got {out[0].shape}"
    )


def test_T18_beats_output_dtype_float32():
    _skip_if_missing(_BEATS_PATH)
    sess = _ort_session(_BEATS_PATH)
    fbank = np.zeros((1, 998, 128), dtype=np.float32)
    out = sess.run(None, {"fbank_features": fbank})
    assert out[0].dtype == np.float32, (
        f"Expected float32; got {out[0].dtype}"
    )


def test_T19_beats_embedding_extractor_accepts_model():
    _skip_if_missing(_BEATS_PATH)
    from voxkit.classifier.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(onnx_path=_BEATS_PATH, substrate_id="beats")
    assert extractor.embedding_dim == 768
    assert extractor.input_length == 32_000


def test_T20_beats_extract_returns_768_vector():
    _skip_if_missing(_BEATS_PATH)
    from voxkit.classifier.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(onnx_path=_BEATS_PATH, substrate_id="beats")
    window = np.random.default_rng(0).standard_normal(32_000).astype(np.float32)
    emb = extractor.extract(window)
    assert emb.shape == (768,), f"Expected (768,); got {emb.shape}"


def test_T21_beats_embedding_is_finite():
    _skip_if_missing(_BEATS_PATH)
    from voxkit.classifier.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(onnx_path=_BEATS_PATH, substrate_id="beats")
    window = np.random.default_rng(1).standard_normal(32_000).astype(np.float32)
    emb = extractor.extract(window)
    assert np.all(np.isfinite(emb)), "Embedding contains NaN or Inf"


def test_T22_beats_different_inputs_different_embeddings():
    """The model must be sensitive to its input — not a trivial constant function."""
    _skip_if_missing(_BEATS_PATH)
    from voxkit.classifier.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor(onnx_path=_BEATS_PATH, substrate_id="beats")
    rng = np.random.default_rng(42)
    w1 = rng.standard_normal(32_000).astype(np.float32)
    w2 = rng.standard_normal(32_000).astype(np.float32)
    e1 = extractor.extract(w1)
    e2 = extractor.extract(w2)
    assert not np.allclose(e1, e2), (
        "Two different audio windows produced identical embeddings; "
        "model may not have loaded correctly"
    )
