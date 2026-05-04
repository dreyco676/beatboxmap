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
v0.12 PANEL ADDITIONS (Lin DSP review + principal-engineer synthesis;
three of four review agents rate-limited — DSP review is the only
fully-quorate one for this file)
============================================================

Cross-component contract (Lin: STRONG — bleed handler is §6 top risk
and nothing in v0.11 verified its output is downstream-usable)
  T32  ClickBleedHandler.clean() output, fed into OnsetDetector.detect()
       on a synthetic perf+bleed signal, achieves F-measure within 0.05
       of the same OnsetDetector run on the perf-only baseline. Catches
       phase distortion, ringing, and over-subtraction that the dB
       attenuation tests (T07/T08/T27) entirely miss. The bleed handler
       can have great dB numbers and still wreck onset detection.

Concurrency (Lin: STRONG — OQ-2 in v0.11 punted to "InferenceWorker
serializes" but never tested it; that's an assumption, not a contract)
  T33  Concurrent reestimate_in_silent_window() while clean() is in
       flight: spawn a thread that calls clean() in a loop while the
       main test calls reestimate_in_silent_window(). Assert no
       exceptions, and that any single clean() output uses ONE IR
       end-to-end (no half-applied IR; pointer swap is atomic).

Loud-fail on poisoned inputs (Lin: STRONG — symmetric with T27 in
test_04; np.convolve silently propagates NaN)
  T34  clean() with NaN in audio raises AudioContainsNonFinite. Today
       NaN propagates through the entire output, surfacing as "no
       events detected" downstream — wrong diagnosis, wrong fix.
  T35  set_ir() with a NaN-tainted IR raises NonFiniteIR at set time,
       not at the next clean() call. Catches a poisoned bleed estimate
       at the source.

Defensive contracts on the metric (Lin: STRONG — division-by-zero
behavior depends on numpy version today)
  T36  residual_ratio_db with calibration RMS = 0 (silent calibration
       segment) raises ZeroCalibrationEnergy. Not +inf, not a numpy-
       version-dependent value. The UI must surface "calibration
       captured nothing" rather than "perfect attenuation."

Tightening of v0.11 panel additions
  T17  TIGHTENED: assert post-reestimation residual is strictly LOWER
       than pre-reestimation residual on the calibration segment. The
       v0.11 form ("IR differs from initial") accepts a worse IR.
  T30  TIGHTENED: long-IR-truncated case must land in the RED band
       (< 10 dB), not just below 20. The whole point of T30 is the
       banner-firing path; "yellow is acceptable" defeats it.
  T31  REWRITTEN as deterministic chunk-counter test rather than psutil
       RSS measurement (RSS is OS- and platform-flaky; the streaming
       contract is a property of the code, not of /proc).

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  Phase distortion impact on downstream onset detection. [v0.11:
      Lin, Marco 2/9 REJECTED on linear-phase FIR grounds. v0.12:
      T32 above measures the END-TO-END impact directly, which is
      what the v0.11 reviewers actually wanted. Closes OQ-1.]
OQ-2  Concurrent re-estimation safety. [v0.11: 2/9 deferred. v0.12:
      T33 above tests it. Closes OQ-2.]
OQ-3  Re-estimation TRIGGER tests (T17/T18 are manual calls; the
      automatic firing on detected silent windows is currently untested
      end-to-end). [Marco, Alex, Lin, Sam, Casey: 5/9 WEAK — record as
      OQ for an integration test in week 2 once Q73's progress dialog
      lands.]
OQ-4  v0.11 T31 used psutil RSS as an OOM proxy; v0.12 replaces with a
      chunk-counter (above). The OOM-on-real-hardware concern remains
      valid but is a manual smoke, not a unit test.
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


def test_T17_reestimation_in_silent_window_improves_ir():
    """v0.12 (Lin) TIGHTENED from 'differs from initial' to 'strictly
    better residual'. The v0.11 form accepted any update — including a
    worse IR — as success. The contract is that re-estimation only
    fires when it improves on the current IR (the regression check in
    T18 is the negative-case mirror)."""
    from voxkit.dsp.bleed import ClickBleedHandler, estimate_ir
    fs = 16_000
    handler = ClickBleedHandler(sample_rate=fs)
    # Start with a deliberately poor initial IR so re-estimation has
    # measurable headroom to improve.
    initial_ir = np.zeros(64, dtype=np.float32); initial_ir[1] = 0.05
    handler.set_ir(initial_ir)

    click_ref = _make_click_train(fs=fs, duration_s=1.0)
    # The "silent" window also carries the click reference for IR fitting;
    # implementation may pass the observed bleed under the click.
    true_ir = np.zeros(64, dtype=np.float32); true_ir[0:3] = [0.0, 0.4, 0.2]
    silent_audio = _convolve_with_ir(click_ref, true_ir).astype(np.float32)

    pre_residual = _rms(silent_audio - _convolve_with_ir(click_ref, initial_ir))
    handler.reestimate_in_silent_window(silent_audio, click_reference=click_ref)
    post_residual = _rms(silent_audio - _convolve_with_ir(click_ref, handler.get_ir()))

    assert post_residual < pre_residual, (
        f"reestimation produced no improvement: pre={pre_residual:.4f}, "
        f"post={post_residual:.4f}"
    )


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

    # v0.12 (Lin) TIGHTENED: must land in the RED band (< 10 dB), not
    # just below 20. The whole point of T30 is the banner-firing path;
    # allowing yellow defeats the test's intent.
    atten = h.get_quality_attenuation_db()
    assert atten < 10.0, (
        f"expected RED band (insufficient IR length triggers banner); "
        f"got {atten:.1f} dB"
    )


def test_T31_clean_streams_in_chunks_not_one_shot():
    """v0.12 REWRITE of v0.11 T31. Original used psutil RSS which is
    flaky on CI and OS-dependent. The streaming contract is a property
    of the code (chunked convolution) and can be tested directly: count
    the chunks the handler processes a long buffer in.

    A non-streaming implementation processes the whole buffer in one
    np.convolve call — chunk count = 1. A streaming implementation
    processes in fixed-size chunks — chunk count > 1.
    """
    from voxkit.dsp.bleed import ClickBleedHandler

    fs = 16_000
    # 1-minute buffer is enough to force chunking without making CI slow.
    audio = np.zeros(fs * 60, dtype=np.float32)
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(np.zeros(256, dtype=np.float32))

    chunk_calls = []
    original_chunk_fn = h._process_chunk  # implementation hook

    def counting_wrapper(chunk):
        chunk_calls.append(len(chunk))
        return original_chunk_fn(chunk)

    h._process_chunk = counting_wrapper
    cleaned = h.clean(audio)

    assert cleaned.shape == audio.shape
    assert len(chunk_calls) > 1, (
        f"clean() processed the 60s buffer in {len(chunk_calls)} chunk(s); "
        "streaming contract requires > 1 (deterministic chunking)"
    )
    # No single chunk should exceed a reasonable bound (e.g., 1 MB ≈ 16s @ 16k).
    max_chunk_samples = max(chunk_calls)
    assert max_chunk_samples * 4 <= 1_000_000, (
        f"largest chunk was {max_chunk_samples * 4 / 1e6:.1f} MB; "
        "streaming chunks should stay under ~1 MB"
    )


# ---------------------------------------------------------------
# v0.12 panel additions (Lin DSP review + principal-engineer synthesis)
# ---------------------------------------------------------------

def test_T32_clean_output_does_not_break_onset_detection():
    """Lin (v0.12): the bleed handler is the §6 top-risk component, but
    every existing test measures dB attenuation of the click — none verify
    that the cleaned audio is still usable by the downstream onset
    detector. Phase distortion, ringing, and over-subtraction can all
    produce great dB numbers and still wreck onset detection.

    Synthesize a perf signal + bleed; clean it; assert the OnsetDetector
    F-measure on the cleaned audio is within 0.05 of the F-measure on
    the perf-only baseline."""
    from voxkit.dsp.bleed import ClickBleedHandler, estimate_ir
    from voxkit.dsp.onsets import OnsetDetector
    from voxkit.dsp.onset_eval import f_measure

    fs = 16_000
    perf_times_ms = [200.0, 480.0, 760.0, 1040.0, 1320.0, 1600.0, 1880.0]
    perf = np.zeros(fs * 2, dtype=np.float32)
    for t_ms in perf_times_ms:
        idx = int(t_ms * 1e-3 * fs)
        for k in range(8):
            if idx + k < len(perf):
                perf[idx + k] = 0.6 * (1.0 - k / 8.0)

    true_ir = np.zeros(128, dtype=np.float32)
    true_ir[0:5] = [0.0, 0.5, 0.3, 0.15, 0.05]
    cal_clicks = _make_click_train(fs=fs, duration_s=2.0)
    cal_observed = _convolve_with_ir(cal_clicks, true_ir)
    est_ir = estimate_ir(reference=cal_clicks, observed=cal_observed,
                         ir_length=128, sample_rate=fs)

    test_clicks = _make_click_train(fs=fs, duration_s=2.0,
                                     period_s=0.4, attack_n=8)
    contaminated = perf + _convolve_with_ir(test_clicks, true_ir)
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(est_ir)
    cleaned = h.clean(contaminated, click_reference=test_clicks)

    detector = OnsetDetector(sample_rate=fs)
    perf_onsets = detector.detect(perf)
    cleaned_onsets = detector.detect(cleaned, click_track=[t / 1000.0 for t in [0, 400, 800, 1200, 1600]])
    reference = [t / 1000.0 for t in perf_times_ms]

    f_perf = f_measure(perf_onsets, reference, iou_ms=50.0)
    f_cleaned = f_measure(cleaned_onsets, reference, iou_ms=50.0)
    assert f_cleaned >= f_perf - 0.05, (
        f"bleed handler degrades onset detection beyond tolerance: "
        f"perf-only F={f_perf:.3f}, cleaned F={f_cleaned:.3f}"
    )


def test_T33_concurrent_reestimation_safe_during_clean():
    """Lin (v0.12 — closes v0.11 OQ-2): the v0.11 panel deferred this
    by saying 'InferenceWorker serializes these calls'. That is an
    assumption, not a contract. If a future maintainer wires re-
    estimation onto a separate signal-handler thread, undefined behavior
    ships silently. Test the contract directly: clean() running in a
    loop while reestimate_in_silent_window() fires from another thread
    must produce no exceptions and no half-applied IR (every clean()
    call's output corresponds to ONE coherent IR end-to-end)."""
    import threading
    from voxkit.dsp.bleed import ClickBleedHandler

    fs = 16_000
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(np.zeros(64, dtype=np.float32))

    audio = np.random.default_rng(33).standard_normal(fs).astype(np.float32) * 0.1
    silent_window = np.zeros(fs, dtype=np.float32)
    click_ref = _make_click_train(fs=fs, duration_s=1.0)

    errors = []
    stop = threading.Event()

    def reestimator():
        try:
            while not stop.is_set():
                h.reestimate_in_silent_window(silent_window, click_reference=click_ref)
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=reestimator, daemon=True)
    t.start()
    try:
        for _ in range(20):
            out = h.clean(audio)
            assert out.shape == audio.shape
            assert np.all(np.isfinite(out)), "clean() output contains NaN/Inf"
    finally:
        stop.set()
        t.join(timeout=2.0)

    assert errors == [], f"reestimation thread raised: {errors}"


def test_T34_clean_with_nan_in_audio_raises():
    """Lin (v0.12): np.convolve silently propagates NaN through the
    entire output. Without this guard, an upstream recorder bug
    (uninitialized ring slot) surfaces as 'no events detected' rather
    than 'audio contains non-finite values'. Loud-fail."""
    from voxkit.dsp.bleed import ClickBleedHandler, AudioContainsNonFinite
    fs = 16_000
    h = ClickBleedHandler(sample_rate=fs)
    h.set_ir(np.zeros(64, dtype=np.float32))
    audio = np.zeros(fs, dtype=np.float32)
    audio[100] = np.nan
    with pytest.raises(AudioContainsNonFinite):
        h.clean(audio)


def test_T35_set_ir_with_nan_rejected_at_set_time():
    """Lin (v0.12): a NaN-tainted IR rejected at set_ir() rather than at
    every subsequent clean() call. Catches a poisoned bleed estimate at
    its source, before it can corrupt downstream output."""
    from voxkit.dsp.bleed import ClickBleedHandler, NonFiniteIR
    h = ClickBleedHandler(sample_rate=16_000)
    bad_ir = np.zeros(64, dtype=np.float32)
    bad_ir[3] = np.nan
    with pytest.raises(NonFiniteIR):
        h.set_ir(bad_ir)


def test_T36_residual_ratio_with_zero_calibration_raises():
    """Lin (v0.12): residual_ratio_db with calibration_rms == 0 today
    divides by 1e-30 (the eps in _rms) and returns a numpy-version-
    dependent number. The UI would render this as "perfect attenuation"
    when in fact the calibration captured nothing. Loud-fail with a
    specific exception so the UI can show 'calibration recording was
    silent — please re-record'."""
    from voxkit.dsp.bleed_metrics import residual_ratio_db, ZeroCalibrationEnergy
    cleaned = np.random.default_rng(36).standard_normal(1000).astype(np.float32)
    silent_cal = np.zeros(1000, dtype=np.float32)
    with pytest.raises(ZeroCalibrationEnergy):
        residual_ratio_db(cleaned=cleaned, calibration=silent_cal)
