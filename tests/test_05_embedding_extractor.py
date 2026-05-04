# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 5: Embedding extractor.

Drives implementation of `voxkit.classifier.embeddings`.

Spec refs: §11 Component 5; §4.2 (PANNs CNN14 ONNX OR BEATs ONNX),
Q33 (substrate decision week 2), Q72 (CPU performance target).

============================================================
TEST LIST (implement strictly in order)
============================================================

Construction and substrate identification
  T01  Extractor can be constructed from an ONNX path
  T02  Extractor exposes `embedding_dim` matching its substrate
  T03  Extractor exposes `substrate_id` ("panns_cnn14" or "beats")
  T04  Construction fails clearly if the ONNX file is missing
  T05  Construction fails clearly if the ONNX file is corrupt

Single-onset extraction
  T06  Extracting from a window of expected length returns a 1D vector
  T07  Vector dtype is float32
  T08  Vector length equals embedding_dim

Pre-conditions
  T09  Window must be at the inference sample rate (16 kHz)
  T10  Window length must match the model's expected input length
  T11  Window must be mono (1D ndarray)

Determinism and stability
  T12  Same input → identical embedding (bit-exact across two calls)
  T13  Same input across two extractor instances → identical embedding
       (no hidden global state)
  T14  Embedding for silence is the same vector across runs
  T15  Different inputs produce different embeddings (sanity)

Batch extraction
  T16  extract_batch on N windows returns array of shape (N, embedding_dim)
  T17  extract_batch result equals stacking single extracts (within 1e-6)
  T18  extract_batch on empty list returns shape (0, embedding_dim)

  -- TIDY FIRST before T19: extract `_window_around(audio, onset_sample,
     pre_pad, post_pad)` into `voxkit.classifier.windowing`. The same
     windowing is needed by both inference and the eval harness.

Onset-driven extraction
  T19  extract_at_onsets returns one embedding per onset
  T20  Onsets near the audio boundary are zero-padded, not skipped
  T21  Onsets outside the audio buffer raise IndexError

ONNX session caching
  T22  Repeated calls do not reload the ONNX session
  T23  ONNX session uses CPU execution provider only (offline guarantee)

CPU performance (Q72)
  T24  Single-window extraction on the reference rig completes in
       under 50 ms (~64 onsets in 16s session × 0.5x = 8s budget total)

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

Schema-drift catches (Lin, Sam, Alex, Priya, Casey, Riley: 6/9)
  T25  ONNX session input/output tensor names match the substrate's
       expected schema; an upstream model swap with renamed tensors
       fails fast with a clear error, not a silent shape mismatch.
  T26  Substrate-specific input_length contract: PANNs CNN14 expects
       ~1s @ 16k = 16000 samples; BEATs expects 10s @ 16k = 160000
       samples. Test asserts each substrate exposes the right value.

Memory & batch hygiene (Lin, Sam, Alex, Casey, Riley, Marco, Priya: 7/9)
  T27  extract_batch on 1000 windows does not OOM and does not exceed
       a documented per-batch memory budget. Sentinel: peak RSS during
       batch < 4 × (1000 × embedding_dim × 4 bytes). Marked @slow.

============================================================
v0.12 PANEL ADDITIONS (principal-engineer ML synthesis;
Priya-equivalent reviewer rate-limited — synthesis only)
============================================================

Privacy / offline guarantee strengthening (STRONG)
  T28  ONNX session is loaded with providers=["CPUExecutionProvider"]
       EXPLICITLY (not just by accident of CPU-only environment). T23
       checks the provider is set; T28 also asserts no GPU/CUDA
       provider is in the providers list and that the session has no
       implicit fallback to providers it didn't request.

Dtype hygiene (STRONG — the no-allocation contract relies on dtype
being float32 end-to-end; an upstream bug that delivers float64 would
silently double the working set)
  T29  extract() with a float64 window raises ValueError("dtype",
       "float32") rather than silently downcasting. The downcast is
       behaviorally correct but masks an upstream bug in the recorder
       or resampler. Loud-fail.

Onset-driven extraction edge (STRONG — onsets near the END of the
buffer are the symmetric case to T20's near-START; if pad logic is
asymmetric, tests miss the regression)
  T30  Onset positioned at audio_length - 1 sample is zero-padded on
       the RIGHT (post-pad) and produces a valid embedding, mirroring
       T20's near-start case. Asymmetric pad logic is a real failure
       mode in window-around helpers.

Removals
  T09  REMOVE: asserts ext.input_length % 16 == 0 — a multiple-of-16
       check that any sample rate satisfies for round-millisecond
       windows. Pins NO contract about 16 kHz; the comment claims
       "contract enforced at recorder boundary" but the test doesn't
       enforce that. v0.12: replace with a real assertion that the
       extractor exposes its required sample_rate as a class-level
       constant (or remove entirely; the recorder boundary IS where
       this is tested).

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  PANNs vs BEATs equivalence sanity (cosine similarity floor on
      AVP samples). [Priya, Marco: 2/9 — defer; Q33 substrate decision
      week 2 already exercises both.]
OQ-2  INT8 quantized model handling. [Casey: 1/9 — defer; out of v1.0
      scope per Q33.]
OQ-3  Window normalization (peak vs RMS): is the input expected to be
      pre-normalized? Implicit currently. [Lin: 1/9 — open question
      for week 2 implementation; not test-driven yet.]
OQ-4  v0.11 T26 hardcodes BEATs input_length = 160000. If the canonical
      BEATs config differs, T26 fails without revealing the truth.
      v0.12 (synthesis): pull the number from a substrate-config
      module, not from the test. Defer to a tidy commit.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

@pytest.fixture
def fake_onnx_path(tmp_path) -> Path:
    """A placeholder; real tests use a small fixture model checked into
    the test data tier (Q63 minimum-reproducible). Here, we create an
    empty file and patch the ONNX loader."""
    p = tmp_path / "fake_model.onnx"
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------
# Construction and substrate identification
# ---------------------------------------------------------------

def test_T01_constructable_from_onnx_path(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session", return_value=MagicMock()):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
    assert ext is not None


def test_T02_exposes_embedding_dim(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session", return_value=MagicMock()):
        panns = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        beats = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="beats")
    assert panns.embedding_dim == 2048
    assert beats.embedding_dim == 768


def test_T03_exposes_substrate_id(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session", return_value=MagicMock()):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
    assert ext.substrate_id == "panns_cnn14"


def test_T04_missing_onnx_path_raises(tmp_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with pytest.raises(FileNotFoundError):
        EmbeddingExtractor(onnx_path=tmp_path / "nope.onnx", substrate_id="panns_cnn14")


def test_T05_corrupt_onnx_path_raises_clearly(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor, ModelLoadError
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               side_effect=ModelLoadError("not a valid ONNX file")):
        with pytest.raises(ModelLoadError, match="ONNX"):
            EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")


# ---------------------------------------------------------------
# Single-onset extraction
# ---------------------------------------------------------------

def _stub_session(output_dim: int):
    sess = MagicMock()
    sess.run.side_effect = lambda outs, inputs: [
        np.full((1, output_dim), 0.1, dtype=np.float32),
    ]
    return sess


def test_T06_window_returns_1d_vector(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        emb = ext.extract(window=np.zeros(ext.input_length, dtype=np.float32))
    assert emb.ndim == 1


def test_T07_vector_dtype_float32(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        emb = ext.extract(window=np.zeros(ext.input_length, dtype=np.float32))
    assert emb.dtype == np.float32


def test_T08_vector_length_matches_embedding_dim(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        emb = ext.extract(window=np.zeros(ext.input_length, dtype=np.float32))
    assert len(emb) == ext.embedding_dim


# ---------------------------------------------------------------
# Pre-conditions
# ---------------------------------------------------------------

def test_T09_extractor_exposes_required_sample_rate(fake_onnx_path):
    """v0.12 REWRITE: the v0.11 form asserted input_length % 16 == 0
    which any sane window length satisfies — pinned no contract. The
    extractor's contract IS 16 kHz; expose it as a class-level constant
    so downstream code can assert against it rather than guess."""
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
    assert ext.required_sample_rate == 16_000


def test_T10_wrong_window_length_raises(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        with pytest.raises(ValueError, match="length"):
            ext.extract(window=np.zeros(ext.input_length - 1, dtype=np.float32))


def test_T11_stereo_input_rejected(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        with pytest.raises(ValueError, match="mono"):
            ext.extract(window=np.zeros((ext.input_length, 2), dtype=np.float32))


# ---------------------------------------------------------------
# Determinism and stability
# ---------------------------------------------------------------

def test_T12_same_input_bit_exact(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        rng = np.random.default_rng(12)
        win = rng.standard_normal(ext.input_length).astype(np.float32)
        a = ext.extract(win)
        b = ext.extract(win)
    np.testing.assert_array_equal(a, b)


def test_T13_two_instances_produce_identical_embeddings(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               side_effect=lambda *a, **k: _stub_session(2048)):
        e1 = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        e2 = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        rng = np.random.default_rng(13)
        win = rng.standard_normal(e1.input_length).astype(np.float32)
    np.testing.assert_array_equal(e1.extract(win), e2.extract(win))


def test_T14_silence_embedding_consistent(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        sil = np.zeros(ext.input_length, dtype=np.float32)
        np.testing.assert_array_equal(ext.extract(sil), ext.extract(sil))


def test_T15_different_inputs_different_embeddings(fake_onnx_path):
    """This requires a non-stub session that actually responds to input.
    Stubbed: assert the contract is invoked with different inputs."""
    sess = MagicMock()
    counter = {"i": 0}

    def fake_run(outs, inputs):
        counter["i"] += 1
        return [np.array([[float(counter["i"])] * 2048], dtype=np.float32)]

    sess.run.side_effect = fake_run
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session", return_value=sess):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        a = ext.extract(np.zeros(ext.input_length, dtype=np.float32))
        b = ext.extract(np.ones(ext.input_length, dtype=np.float32))
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------

def test_T16_extract_batch_returns_2d_array(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        windows = [np.zeros(ext.input_length, dtype=np.float32) for _ in range(5)]
        out = ext.extract_batch(windows)
    assert out.shape == (5, 2048)


def test_T17_extract_batch_matches_stacked_singles(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        rng = np.random.default_rng(17)
        windows = [rng.standard_normal(ext.input_length).astype(np.float32) for _ in range(3)]
        batched = ext.extract_batch(windows)
        singles = np.stack([ext.extract(w) for w in windows])
    np.testing.assert_allclose(batched, singles, atol=1e-6)


def test_T18_extract_batch_empty_returns_empty(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        out = ext.extract_batch([])
    assert out.shape == (0, 2048)


# ----- TIDY FIRST checkpoint -----
# Extract `_window_around(audio, onset_sample, pre_pad, post_pad)` to
# `voxkit.classifier.windowing` so the eval harness shares the same
# windowing logic. Pure structural change; no behavior delta.


# ---------------------------------------------------------------
# Onset-driven extraction
# ---------------------------------------------------------------

def test_T19_extract_at_onsets_returns_one_embedding_per_onset(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        audio = np.zeros(16_000 * 5, dtype=np.float32)   # 5 seconds
        onsets = [0.5, 1.0, 2.5, 4.0]
        out = ext.extract_at_onsets(audio, onset_times_s=onsets, sample_rate=16_000)
    assert out.shape == (4, 2048)


def test_T20_onsets_near_boundary_zero_padded(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        audio = np.zeros(16_000, dtype=np.float32)
        onsets = [0.001]   # right at the very start
        out = ext.extract_at_onsets(audio, onset_times_s=onsets, sample_rate=16_000)
    assert out.shape == (1, 2048)


def test_T21_onset_outside_audio_buffer_raises(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        audio = np.zeros(16_000, dtype=np.float32)   # 1 second
        with pytest.raises(IndexError):
            ext.extract_at_onsets(audio, onset_times_s=[5.0], sample_rate=16_000)


# ---------------------------------------------------------------
# ONNX session caching
# ---------------------------------------------------------------

def test_T22_repeated_calls_do_not_reload_session(fake_onnx_path):
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)) as loader:
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        for _ in range(10):
            ext.extract(np.zeros(ext.input_length, dtype=np.float32))
    assert loader.call_count == 1


def test_T23_session_uses_cpu_execution_provider_only(fake_onnx_path):
    """Offline guarantee: no GPU dependency, no network call."""
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session") as loader:
        loader.return_value = _stub_session(2048)
        EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        kwargs = loader.call_args.kwargs
    assert kwargs.get("providers") == ["CPUExecutionProvider"]


# ---------------------------------------------------------------
# CPU performance (Q72)
# ---------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.dataset_required("real_onnx_model")
def test_T24_single_window_under_50ms_on_reference_rig():
    import time
    from voxkit.classifier.embeddings import EmbeddingExtractor

    ext = EmbeddingExtractor.from_default("panns_cnn14")
    win = np.zeros(ext.input_length, dtype=np.float32)
    ext.extract(win)   # warm

    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        ext.extract(win)
    per_call_ms = (time.perf_counter() - t0) * 1000 / n
    assert per_call_ms < 50.0, f"per-call {per_call_ms:.1f} ms exceeds 50 ms target"


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T25_onnx_tensor_schema_validated_at_load(fake_onnx_path):
    """A drop-in model swap with renamed input/output tensors should
    fail fast at load, not at the first inference call (which would be
    in the middle of a user's recording session)."""
    from voxkit.classifier.embeddings import EmbeddingExtractor, ModelSchemaError

    bad_session = MagicMock()
    bad_session.get_inputs.return_value = [MagicMock(name="wrong_input_name")]
    bad_session.get_outputs.return_value = [MagicMock(name="wrong_output_name")]

    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=bad_session):
        with pytest.raises(ModelSchemaError, match="input"):
            EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")


def test_T26_substrate_input_length_contracts(fake_onnx_path):
    """PANNs CNN14 and BEATs expect different window sizes. The extractor
    must surface the correct one for the substrate, since the windowing
    helper (extract_at_onsets) uses input_length to size the audio slice."""
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        panns = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        # PANNs CNN14: 1 second @ 16 kHz
        assert panns.input_length == 16_000
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(768)):
        beats = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="beats")
        # BEATs: 10 seconds @ 16 kHz (per substrate doc; adjust if the
        # canonical config differs)
        assert beats.input_length == 160_000


@pytest.mark.slow
def test_T27_extract_batch_memory_bounded(fake_onnx_path):
    """A 1000-window batch must not blow up memory. Real models would
    surface OOM; with the stub we only check the working-set sanity:
    output array dominates everything else."""
    pytest.importorskip("psutil")
    import psutil
    import os
    from voxkit.classifier.embeddings import EmbeddingExtractor

    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        windows = [np.zeros(ext.input_length, dtype=np.float32) for _ in range(1000)]

        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss
        out = ext.extract_batch(windows)
        rss_after = proc.memory_info().rss

    output_bytes = out.nbytes   # ~8 MB for 1000×2048 float32
    overhead = rss_after - rss_before
    assert overhead < 4 * output_bytes, (
        f"batch of 1000 grew RSS by {overhead / 1e6:.0f} MB; "
        f"output is only {output_bytes / 1e6:.0f} MB — possible buffering bug"
    )


# ---------------------------------------------------------------
# v0.12 panel additions (principal-engineer ML synthesis)
# ---------------------------------------------------------------

def test_T28_onnx_session_loaded_with_only_cpu_provider(fake_onnx_path):
    """v0.12: T23 checks providers=['CPUExecutionProvider'] is passed,
    but onnxruntime will silently use other providers if they're
    available unless we ALSO assert the negative — no GPU/CUDA/DML
    provider sneaks in. This is a privacy/offline-guarantee test, not
    a perf test."""
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session") as loader:
        loader.return_value = _stub_session(2048)
        EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        kwargs = loader.call_args.kwargs

    providers = kwargs.get("providers")
    assert providers == ["CPUExecutionProvider"]
    forbidden = {"CUDAExecutionProvider", "TensorrtExecutionProvider",
                 "DmlExecutionProvider", "ROCMExecutionProvider"}
    assert not (set(providers) & forbidden), (
        f"GPU/accelerator provider leaked into ONNX session: "
        f"{set(providers) & forbidden}"
    )


def test_T29_extract_with_float64_window_loud_fails(fake_onnx_path):
    """v0.12: a float64 window will silently work (numpy/onnxruntime
    will downcast) but doubles the working-set memory and signals an
    upstream bug in recorder/resampler. The no-allocation contract is
    a downstream consequence of dtype hygiene."""
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        bad_window = np.zeros(ext.input_length, dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            ext.extract(bad_window)


def test_T30_onset_at_buffer_end_zero_padded_right(fake_onnx_path):
    """v0.12: T20 covers near-START boundary (zero-pad on the LEFT).
    The symmetric near-END case (zero-pad on the RIGHT) needs equal
    coverage; an asymmetric pad implementation is a real failure mode
    for window-around helpers."""
    from voxkit.classifier.embeddings import EmbeddingExtractor
    with patch("voxkit.classifier.embeddings._load_onnx_session",
               return_value=_stub_session(2048)):
        ext = EmbeddingExtractor(onnx_path=fake_onnx_path, substrate_id="panns_cnn14")
        audio = np.zeros(16_000, dtype=np.float32)        # 1 second
        last_sample_t = (len(audio) - 1) / 16_000
        out = ext.extract_at_onsets(audio, onset_times_s=[last_sample_t],
                                     sample_rate=16_000)
    assert out.shape == (1, 2048)
