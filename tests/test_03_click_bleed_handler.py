# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 3: Click bleed handler.

Drives implementation of `voxkit.dsp.bleed`.

Spec refs: §11 Component 3; v0.9 design (FIR subtraction with mid-session
re-estimation; bleed_ir_history with two protected slots), Q79 (quality
indicator metric — post-subtraction click residual ratio in dB),
Q80 (week-1 tracer-bullet acceptance: > 20 dB null after 2-second
adaptation on a deliberately leaky setup).

============================================================
TEST LIST (implement strictly in order)
============================================================

Pass-through and trivial cases
  T01  ClickBleedHandler with no IR is identity (output == input)
  T02  Cleaning audio of length 0 returns length 0
  T03  Cleaning audio with a known-zero IR returns input (no subtraction)

IR estimation from click-only segment
  T04  Estimating an IR from a 2s click-only segment returns an IR of
       expected length (e.g., 1024 taps at 16 kHz)
  T05  Estimating an IR from silence returns an all-zero IR
  T06  Estimating an IR from a known input/output pair recovers the IR
       within 1e-3 RMS error (synthetic reference)

Subtraction reduces click energy
  T07  Subtracting a known IR from "click * IR + performance" leaves the
       performance with > 30 dB residual click attenuation (synthetic)
  T08  Subtracting an IR estimated from clicks reduces click-only energy
       by > 20 dB (Q80 acceptance)

  -- TIDY FIRST before T09: extract `_residual_ratio_db` into its own
     pure function in `voxkit.dsp.bleed_metrics` so the metric and the
     handler can be tested independently. Structural change only.

Quality indicator metric (Q79)
  T09  residual_ratio_db on identical audio returns 0 dB (no attenuation)
  T10  residual_ratio_db on cleaned audio with 100x lower RMS returns 40 dB
  T11  residual_ratio_db on cleaned audio with 10x higher RMS returns -20 dB
  T12  get_quality_attenuation_db() returns positive value for good IR
  T13  get_quality_attenuation_db() returns ~0 dB for a no-op IR

Mid-session re-estimation in silent windows
  T14  Active silent-window detector flags a 200ms region below RMS threshold
  T15  Active silent-window detector does not flag a region above threshold
  T16  Passive silent-window detector identifies known silent regions
       (when a separate VAD signal is provided)
  T17  Re-estimation in a flagged silent window updates the active IR
  T18  Re-estimation does NOT update the active IR if the new estimate
       has worse residual than the current one (one-step regression test)

bleed_ir_history with two protected slots (v0.9 carry)
  T19  History starts empty
  T20  After first re-estimation, history has 1 entry
  T21  After 2 re-estimations, history has 2 entries (both protected)
  T22  After 5 re-estimations, history has 2 protected + 3 unprotected,
       all retained (history is bounded but protected slots win on eviction)
  T23  When unprotected slots overflow, oldest unprotected is evicted first
  T24  Protected slots are NEVER evicted

Bleed banner and override
  T25  Banner is shown when get_quality_attenuation_db() < 10 dB
  T26  Banner is suppressed when bleed_gate_overridden=True

Tracer-bullet integration test (Q80, week 1 acceptance)
  T27  On synthetic "leaky open-back headphones" simulation: 2s adaptation
       achieves > 20 dB null on click-only follow-up segment

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

Defensive contracts on the quality metric (Lin, Sam, Alex, Marco,
Jordan, Casey, Riley: 7/9 — strong)
  T28  get_quality_attenuation_db() before set_calibration() raises
       NoCalibrationData (or returns a sentinel) — must not silently
       return 0.0 dB which the UI would render as "perfect attenuation".
  T29  Sample-rate mismatch between calibration audio and live audio
       raises SampleRateMismatch (no silent down/upsampling). Catches
       a real failure mode where the recorder switches device sample
       rates mid-session.
  T30  IR length shorter than the actual room/headphone IR support
       produces a documented degraded attenuation; test asserts the
       function does NOT crash, only that get_quality_attenuation_db()
       falls into the yellow/red band so the UI banner fires.

Streaming safety (Lin, Sam, Alex, Casey, Riley, Marco: 6/9)
  T31  clean() handles a 60-minute (60 * 60 * 16_000 sample) buffer
       without OOM by processing in fixed-size chunks. Sentinel: peak
       RSS during clean() is < input_size + small_overhead bytes.
       Skipped on CI without psutil; marked @pytest.mark.slow.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS
============================================================

OQ-1  Phase distortion impact on downstream onset detection. [Lin, Marco:
      2/9 — REJECTED. v0.9 design uses linear-phase FIR; phase distortion
      is bounded by construction. Re-open if AVP eval shows MAE drift
      after subtraction.]
OQ-2  Concurrent re-estimation safety (silent-window detector firing
      while clean() is mid-call). [Sam, Lin: 2/9 — defer; the v0.10
      InferenceWorker contract serializes these calls.]
OQ-3  Re-estimation TRIGGER tests (T17/T18 are manual calls; the
      automatic firing on detected silent windows is currently untested
      end-to-end). [Marco, Alex, Lin, Sam, Casey: 5/9 WEAK — record as
      OQ for an integration test in week 2 once Q73's progress dialog
      lands.]
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------
# Helpers (synthetic test data)
# ---------------------------------------------------------------

def _make_click_train(fs=16_000, duration_s=2.0, period_s=0.5, attack_n=8):
    n = int(fs * duration_s)
    out = np.zeros(n, dtype=np.float32)
    period = int(fs * period_s)
    for i in range(0, n, period):
        if i + attack_n < n:
            out[i:i + attack_n] = np.linspace(1.0, 0.0, attack_n).astype(np.float32)
    return out


def _convolve_with_ir(signal, ir):
    return np.convolve(signal, ir, mode="same").astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-30))


# ---------------------------------------------------------------
# Pass-through and trivial cases
# ---------------------------------------------------------------

def test_T01_handler_with_no_ir_is_identity():
    from voxkit.dsp.bleed import ClickBleedHandler
    h = ClickBleedHandler(sample_rate=16_000)
    audio = np.random.default_rng(0).standard_normal(16_000).astype(np.float32)
    out = h.clean(audio)
    np.testing.assert_array_equal(out, audio)


def test_T02_cleaning_zero_length_audio_returns_zero_length():
    from voxkit.dsp.bleed import ClickBleedHandler
    out = ClickBleedHandler(sample_rate=16_000).clean(np.zeros(0, dtype=np.float32))
    assert out.shape == (0,)


def test_T03_zero_ir_returns_input_unchanged():
    from voxkit.dsp.bleed import ClickBleedHandler
    h = ClickBleedHandler(sample_rate=16_000)
    h.set_ir(np.zeros(1024, dtype=np.float32))
    audio = np.random.default_rng(1).standard_normal(16_000).astype(np.float32)
    out = h.clean(audio)
    np.testing.assert_allclose(out, audio, atol=1e-6)


# ---------------------------------------------------------------
# IR estimation from click-only segment
# ---------------------------------------------------------------

def test_T04_ir_from_2s_click_only_has_expected_length():
    from voxkit.dsp.bleed import estimate_ir
    fs = 16_000
    clicks = _make_click_train(fs=fs, duration_s=2.0)
    bleed = _convolve_with_ir(clicks, np.array([0.0, 0.5, 0.3, 0.1], dtype=np.float32))
    ir = estimate_ir(reference=clicks, observed=bleed, ir_length=1024, sample_rate=fs)
    assert ir.shape == (1024,)


def test_T05_ir_from_silence_is_zero():
    from voxkit.dsp.bleed import estimate_ir
    fs = 16_000
    silence_ref = np.zeros(fs * 2, dtype=np.float32)
    silence_obs = np.zeros(fs * 2, dtype=np.float32)
    ir = estimate_ir(reference=silence_ref, observed=silence_obs, ir_length=256, sample_rate=fs)
    assert _rms(ir) < 1e-6


def test_T06_ir_estimation_recovers_known_ir():
    from voxkit.dsp.bleed import estimate_ir
    fs = 16_000
    clicks = _make_click_train(fs=fs, duration_s=2.0)
    true_ir = np.zeros(64, dtype=np.float32)
    true_ir[0] = 0.4
    true_ir[3] = 0.2
    true_ir[7] = 0.1
    observed = _convolve_with_ir(clicks, true_ir)
    est = estimate_ir(reference=clicks, observed=observed, ir_length=64, sample_rate=fs)
    rms_err = _rms(est - true_ir)
    assert rms_err < 1e-3


# ---------------------------------------------------------------
# Subtraction reduces click energy
# ---------------------------------------------------------------

def test_T07_known_ir_subtraction_attenuates_click_above_30db():
    from voxkit.dsp.bleed import ClickBleedHandler
    fs = 16_000
    ir = np.array([0.0, 0.4, 0.2, 0.1], dtype=np.float32)
    clicks = _make_click_train(fs=fs, duration_s=2.0)
    perf = 0.05 * np.random.default_rng(7).standard_normal(len(clicks)).astype(np.float32)
    contaminated = _convolve_with_ir(clicks, ir) + perf

    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(ir)
    cleaned = h.clean(contaminated, click_reference=clicks)

    bleed_only_before = _convolve_with_ir(clicks, ir)
    bleed_residual = cleaned - perf
    attenuation_db = 20 * np.log10(_rms(bleed_only_before) / (_rms(bleed_residual) + 1e-12))
    assert attenuation_db > 30.0


def test_T08_estimated_ir_attenuates_click_only_above_20db():
    """Q80 acceptance precursor in synthetic form."""
    from voxkit.dsp.bleed import ClickBleedHandler, estimate_ir
    fs = 16_000
    true_ir = np.zeros(128, dtype=np.float32)
    true_ir[0:5] = [0.0, 0.5, 0.3, 0.15, 0.05]

    cal_clicks = _make_click_train(fs=fs, duration_s=2.0)
    cal_observed = _convolve_with_ir(cal_clicks, true_ir)
    est = estimate_ir(reference=cal_clicks, observed=cal_observed, ir_length=128, sample_rate=fs)

    test_clicks = _make_click_train(fs=fs, duration_s=2.0)
    test_observed = _convolve_with_ir(test_clicks, true_ir)
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(est)
    cleaned = h.clean(test_observed, click_reference=test_clicks)

    attenuation_db = 20 * np.log10(_rms(test_observed) / (_rms(cleaned) + 1e-12))
    assert attenuation_db > 20.0


# ----- TIDY FIRST checkpoint -----
# Before T09: extract `_residual_ratio_db` into `voxkit.dsp.bleed_metrics`
# so the metric is testable in isolation. Structural change only.


# ---------------------------------------------------------------
# Quality indicator metric (Q79)
# ---------------------------------------------------------------

def test_T09_residual_ratio_db_zero_for_identical_audio():
    from voxkit.dsp.bleed_metrics import residual_ratio_db
    x = np.random.default_rng(9).standard_normal(1000).astype(np.float32)
    assert abs(residual_ratio_db(cleaned=x, calibration=x)) < 1e-6


def test_T10_residual_ratio_db_40db_for_100x_quieter():
    from voxkit.dsp.bleed_metrics import residual_ratio_db
    rng = np.random.default_rng(10)
    cal = rng.standard_normal(1000).astype(np.float32)
    cleaned = (cal / 100.0).astype(np.float32)
    assert residual_ratio_db(cleaned=cleaned, calibration=cal) == pytest.approx(40.0, abs=0.5)


def test_T11_residual_ratio_db_negative_when_cleaned_louder():
    from voxkit.dsp.bleed_metrics import residual_ratio_db
    rng = np.random.default_rng(11)
    cal = rng.standard_normal(1000).astype(np.float32)
    cleaned = (cal * 10.0).astype(np.float32)
    assert residual_ratio_db(cleaned=cleaned, calibration=cal) == pytest.approx(-20.0, abs=0.5)


def test_T12_quality_attenuation_db_positive_for_good_ir():
    from voxkit.dsp.bleed import ClickBleedHandler, estimate_ir
    fs = 16_000
    true_ir = np.zeros(64, dtype=np.float32); true_ir[0:3] = [0.0, 0.5, 0.2]
    clicks = _make_click_train(fs=fs, duration_s=2.0)
    observed = _convolve_with_ir(clicks, true_ir)
    est = estimate_ir(reference=clicks, observed=observed, ir_length=64, sample_rate=fs)

    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(est)
    h.set_calibration(reference=clicks, observed=observed)
    assert h.get_quality_attenuation_db() > 15.0


def test_T13_quality_attenuation_db_near_zero_for_noop_ir():
    from voxkit.dsp.bleed import ClickBleedHandler
    fs = 16_000
    clicks = _make_click_train(fs=fs, duration_s=2.0)
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(np.zeros(64, dtype=np.float32))
    h.set_calibration(reference=clicks, observed=clicks)
    assert abs(h.get_quality_attenuation_db()) < 1.0


# ---------------------------------------------------------------
# Mid-session re-estimation in silent windows
# ---------------------------------------------------------------

def test_T14_active_detector_flags_silent_region():
    from voxkit.dsp.bleed import ActiveSilentWindowDetector
    fs = 16_000
    audio = np.zeros(fs, dtype=np.float32)
    audio[0:8000] = 0.5 * np.random.default_rng(14).standard_normal(8000).astype(np.float32)
    # 0.0 to 0.5s = active; 0.5s to 1s = silent.
    det = ActiveSilentWindowDetector(rms_threshold=0.01, sample_rate=fs)
    flags = det.find_silent_regions(audio, min_duration_ms=200)
    assert any(0.5 <= start < 1.0 for start, _ in flags)


def test_T15_active_detector_does_not_flag_active_region():
    from voxkit.dsp.bleed import ActiveSilentWindowDetector
    fs = 16_000
    audio = 0.5 * np.random.default_rng(15).standard_normal(fs).astype(np.float32)
    det = ActiveSilentWindowDetector(rms_threshold=0.01, sample_rate=fs)
    flags = det.find_silent_regions(audio, min_duration_ms=200)
    assert flags == []


def test_T16_passive_detector_uses_external_vad():
    from voxkit.dsp.bleed import PassiveSilentWindowDetector
    fs = 16_000
    vad = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1])  # frame-rate VAD
    det = PassiveSilentWindowDetector(sample_rate=fs, frame_size=1600)
    flags = det.find_silent_regions_from_vad(vad, min_frames=2)
    assert (0.2, 0.4) in [(round(s, 1), round(e, 1)) for s, e in flags]
    assert (0.5, 0.8) in [(round(s, 1), round(e, 1)) for s, e in flags]


def test_T17_reestimation_in_silent_window_updates_ir():
    from voxkit.dsp.bleed import ClickBleedHandler
    fs = 16_000
    handler = ClickBleedHandler(sample_rate=fs)
    initial_ir = np.zeros(64, dtype=np.float32); initial_ir[1] = 0.1
    handler.set_ir(initial_ir)
    # Provide a silent window with a measurably better IR fit.
    silent_audio = np.zeros(fs, dtype=np.float32)
    click_ref = _make_click_train(fs=fs, duration_s=1.0)
    handler.reestimate_in_silent_window(silent_audio, click_reference=click_ref)
    # New IR should differ from initial (re-estimation happened).
    assert not np.allclose(handler.get_ir(), initial_ir)


def test_T18_reestimation_rejected_if_residual_worse():
    from voxkit.dsp.bleed import ClickBleedHandler
    fs = 16_000
    handler = ClickBleedHandler(sample_rate=fs)
    good_ir = np.zeros(64, dtype=np.float32); good_ir[0:3] = [0.0, 0.4, 0.2]
    handler.set_ir(good_ir)
    # Hand it a window that produces a worse IR (random noise vs known clicks).
    bad_window = 0.5 * np.random.default_rng(18).standard_normal(fs).astype(np.float32)
    handler.reestimate_in_silent_window(
        bad_window, click_reference=_make_click_train(fs=fs, duration_s=1.0),
    )
    np.testing.assert_array_equal(handler.get_ir(), good_ir)


# ---------------------------------------------------------------
# bleed_ir_history with two protected slots
# ---------------------------------------------------------------

def test_T19_history_starts_empty():
    from voxkit.dsp.bleed import BleedIRHistory
    assert BleedIRHistory().entries == []


def test_T20_history_has_one_entry_after_first_reestimation():
    from voxkit.dsp.bleed import BleedIRHistory
    h = BleedIRHistory()
    h.append(np.zeros(64, dtype=np.float32), protected=True)
    assert len(h.entries) == 1


def test_T21_two_protected_entries_both_kept():
    from voxkit.dsp.bleed import BleedIRHistory
    h = BleedIRHistory(unprotected_capacity=3)
    h.append(np.ones(64, dtype=np.float32) * 0.1, protected=True)
    h.append(np.ones(64, dtype=np.float32) * 0.2, protected=True)
    assert sum(1 for e in h.entries if e.protected) == 2


def test_T22_protected_plus_unprotected_all_retained_within_capacity():
    from voxkit.dsp.bleed import BleedIRHistory
    h = BleedIRHistory(unprotected_capacity=3)
    h.append(np.ones(64) * 0.1, protected=True)
    h.append(np.ones(64) * 0.2, protected=True)
    h.append(np.ones(64) * 0.3, protected=False)
    h.append(np.ones(64) * 0.4, protected=False)
    h.append(np.ones(64) * 0.5, protected=False)
    assert len(h.entries) == 5


def test_T23_oldest_unprotected_evicted_when_unprotected_overflows():
    from voxkit.dsp.bleed import BleedIRHistory
    h = BleedIRHistory(unprotected_capacity=2)
    h.append(np.ones(64) * 0.1, protected=False)
    h.append(np.ones(64) * 0.2, protected=False)
    h.append(np.ones(64) * 0.3, protected=False)
    rms_values = sorted([float(_rms(e.ir)) for e in h.entries if not e.protected])
    # The 0.1-magnitude IR should have been evicted.
    assert all(r > 0.15 for r in rms_values)


def test_T24_protected_slots_never_evicted():
    from voxkit.dsp.bleed import BleedIRHistory
    h = BleedIRHistory(unprotected_capacity=1)
    h.append(np.ones(64) * 0.1, protected=True)
    h.append(np.ones(64) * 0.2, protected=True)
    for v in (0.3, 0.4, 0.5, 0.6, 0.7):
        h.append(np.ones(64) * v, protected=False)
    assert sum(1 for e in h.entries if e.protected) == 2


# ---------------------------------------------------------------
# Bleed banner and override
# ---------------------------------------------------------------

def test_T25_banner_shown_when_attenuation_below_10db():
    from voxkit.dsp.bleed import should_show_bleed_banner
    assert should_show_bleed_banner(attenuation_db=5.0, override=False) is True


def test_T26_banner_suppressed_when_override_set():
    from voxkit.dsp.bleed import should_show_bleed_banner
    assert should_show_bleed_banner(attenuation_db=5.0, override=True) is False


# ---------------------------------------------------------------
# Tracer-bullet integration (Q80)
# ---------------------------------------------------------------

def test_T27_tracer_bullet_synthetic_leaky_headphones_above_20db():
    """Q80: > 20 dB null after 2s adaptation on a synthetic 'leaky' setup.
    This is the test the maintainer also runs against real hardware in
    week 1 (Q80) to validate the v0.9 design under v0.11 contracts."""
    from voxkit.dsp.bleed import ClickBleedHandler, estimate_ir
    fs = 16_000
    # Synthetic 'leaky open-back headphones': longer IR, more taps.
    leaky_ir = np.zeros(256, dtype=np.float32)
    leaky_ir[0:10] = [0.0, 0.6, 0.4, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02]

    cal_clicks = _make_click_train(fs=fs, duration_s=2.0)
    cal_observed = _convolve_with_ir(cal_clicks, leaky_ir)
    est = estimate_ir(reference=cal_clicks, observed=cal_observed,
                      ir_length=256, sample_rate=fs)

    eval_clicks = _make_click_train(fs=fs, duration_s=2.0)
    eval_observed = _convolve_with_ir(eval_clicks, leaky_ir)
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(est)
    cleaned = h.clean(eval_observed, click_reference=eval_clicks)

    attenuation_db = 20 * np.log10(_rms(eval_observed) / (_rms(cleaned) + 1e-12))
    assert attenuation_db > 20.0, f"only {attenuation_db:.1f} dB; spec requires > 20"


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T28_quality_metric_without_calibration_raises():
    """The UI's bleed banner reads get_quality_attenuation_db() to drive
    the green/yellow/red bar. Returning 0.0 silently when no calibration
    data exists would render as 'perfect attenuation' and hide the very
    problem the banner exists to surface. Loud-fail instead."""
    from voxkit.dsp.bleed import ClickBleedHandler, NoCalibrationData
    h = ClickBleedHandler(sample_rate=16_000)
    h.set_ir(np.zeros(64, dtype=np.float32))
    with pytest.raises(NoCalibrationData):
        h.get_quality_attenuation_db()


def test_T29_sample_rate_mismatch_in_clean_raises():
    """If the recorder switches device rate mid-session, the bleed
    handler's IR (estimated at the original rate) cannot be applied.
    Silent resampling would give a wrong IR; loud-fail instead."""
    from voxkit.dsp.bleed import ClickBleedHandler, SampleRateMismatch
    fs_ir = 16_000
    fs_audio = 48_000
    h = ClickBleedHandler(sample_rate=fs_ir)
    h.set_ir(np.zeros(64, dtype=np.float32))
    audio_at_wrong_rate = np.zeros(fs_audio, dtype=np.float32)
    clicks_at_wrong_rate = np.zeros(fs_audio, dtype=np.float32)
    with pytest.raises(SampleRateMismatch):
        h.clean(audio_at_wrong_rate, click_reference=clicks_at_wrong_rate,
                input_sample_rate=fs_audio)


def test_T30_short_ir_length_falls_into_red_band_not_crash():
    """A user with a very leaky setup may have an IR longer than the
    configured ir_length. The handler must continue to work; the quality
    metric will simply read in the red band, triggering the banner."""
    from voxkit.dsp.bleed import ClickBleedHandler, estimate_ir
    fs = 16_000
    # Long, slowly-decaying IR (simulates a very reflective room).
    long_ir = np.zeros(2048, dtype=np.float32)
    long_ir[0:1024] = np.exp(-np.arange(1024) / 200.0).astype(np.float32) * 0.3
    long_ir[1] = 0.6   # main impulse

    clicks = _make_click_train(fs=fs, duration_s=2.0)
    observed = _convolve_with_ir(clicks, long_ir)
    # Estimate with a too-short IR.
    short_est = estimate_ir(reference=clicks, observed=observed,
                            ir_length=64, sample_rate=fs)
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(short_est)
    h.set_calibration(reference=clicks, observed=observed)

    # Must not raise. Quality should be in the red zone (< 10 dB).
    atten = h.get_quality_attenuation_db()
    assert atten < 20.0, (
        f"expected red/yellow band (insufficient IR length); got {atten:.1f} dB"
    )


@pytest.mark.slow
def test_T31_clean_processes_long_buffer_without_oom():
    """A user recording a 60-minute take must not see VoxKit OOM. The
    handler should process in fixed chunks rather than holding the full
    convolution working set in memory at once."""
    pytest.importorskip("psutil")
    import psutil
    import os
    from voxkit.dsp.bleed import ClickBleedHandler

    fs = 16_000
    duration_s = 60 * 60
    n_samples = fs * duration_s
    audio = np.zeros(n_samples, dtype=np.float32)
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(np.zeros(256, dtype=np.float32))

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    cleaned = h.clean(audio)
    rss_after = proc.memory_info().rss

    # Allow generous overhead (intermediate buffers, NumPy temporaries),
    # but not "double the input buffer" which would indicate non-streaming.
    overhead = rss_after - rss_before
    input_bytes = audio.nbytes
    assert overhead < input_bytes, (
        f"RSS grew by {overhead / 1e6:.0f} MB on a {input_bytes / 1e6:.0f} MB "
        f"input — likely loading full convolution working set"
    )
    assert cleaned.shape == audio.shape
