# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 4: Onset detector.

Drives implementation of `voxkit.dsp.onsets`.

Spec refs: §11 Component 4; §5.2 (librosa.onset + click-aware preprocessing),
Q70 (two-tier release gate: detection F-measure + alignment MAE).

============================================================
TEST LIST (implement strictly in order)
============================================================

Trivial signals
  T01  Silent input returns no onsets
  T02  A single impulse returns exactly one onset
  T03  Two well-separated impulses return exactly two onsets
  T04  Onset times for impulse trains are within ±5ms of impulse positions

Pre-conditions
  T05  Detector raises on non-mono input
  T06  Detector raises if sample_rate != 16_000 (inference rate per §4.2)

Click-aware preprocessing
  T07  Without click-aware preprocessing, a click track produces N onsets
       at click positions (sanity: detector sees clicks)
  T08  With click_track times provided, those onset positions are suppressed
  T09  Performance onsets adjacent to (but not at) click times are NOT
       suppressed (the suppression window is bounded)

Detection F-measure helper (§7.8 release-gate input)
  T10  f_measure(detected=[], reference=[]) returns 1.0 (vacuous)
  T11  f_measure with all detections matched returns 1.0
  T12  f_measure with no detections matched returns 0.0
  T13  IOU window of 50 ms: a detection 49 ms off matches; 51 ms off does not

  -- TIDY FIRST before T14: extract `_align_pairs` (greedy nearest-neighbor
     matching under a tolerance) into `voxkit.dsp.onset_eval`. The same
     pairing logic is needed for both F-measure and MAE; share it.

Alignment MAE helper (Q70 second tier)
  T14  alignment_mae on perfectly aligned detections returns 0
  T15  alignment_mae on detections all 12 ms late returns 12 ms
  T16  alignment_mae uses median absolute error, not mean
  T17  alignment_mae ignores unmatched detections (only computed on TPs)

Two-tier release gate (Q70)
  T18  release_gate passes when F ≥ 0.92 AND MAE ≤ 15 ms (AVP)
  T19  release_gate fails when F = 0.93 but MAE = 17 ms (alignment fails)
  T20  release_gate fails when F = 0.90 but MAE = 10 ms (detection fails)
  T21  release_gate uses different thresholds for AVP vs OOD per §7.8

Quality on synthetic high-precision data
  T22  Synthetic dataset with 100 random impulses + low noise:
       F-measure ≥ 0.99, MAE ≤ 5 ms (sanity: detector works on ideal data)

AVP-tier release gate (slow; marked)
  T23  Full AVP corpus: F ≥ 0.92, MAE ≤ 15 ms (release-gate test)

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

F-measure honesty (Lin, Marco, Alex, Sam, Casey, Jordan, Priya: 7/9)
  T24  f_measure with detected events that have NO reference partner
       (false positives) is correctly penalized. Existing T11/T12 cover
       only TPs and FNs; without an FP test, a degenerate detector that
       fires on every frame would still score 1.0 on T11.

MAE naming consistency (Lin, Sam, Alex, Casey, Riley, Marco, Priya: 7/9)
  T25  alignment_mae must use ONE statistic and the function name must
       match. T16 currently uses median while the function is named
       "mae" (Mean Absolute Error). This test asserts: either the
       function uses mean (and T16 must change), OR the function is
       renamed to alignment_med_AE and T25 enforces the new name. As
       written, T25 calls the spec-aligned name and the implementer
       picks. RECOMMENDATION: rename to alignment_median_ae per Q70's
       robustness intent; mean-of-absolute would be vulnerable to the
       100ms outlier in T16's data.

Robustness (Marco, Lin, Alex, Casey, Sam, Riley: 6/9)
  T26  Two onsets within 5 ms of each other: detector merges OR reports
       both, with the chosen behavior documented. Test asserts the
       chosen contract; spec needs a one-line addition.
  T27  NaN/Inf in audio raises a clear AudioContainsNonFinite error
       (does NOT silently produce no onsets, which would mask a
       recorder bug as a "no events detected" UX state).

============================================================
WEAK CONSENSUS / OPEN QUESTIONS
============================================================

OQ-1  Tongue-click handling (Lin's defer note from §8). Recorded; pursue
      if AVP eval shows tongue-click subjects underperforming.
OQ-2  Adaptive threshold under sustained loud input. [Lin: 1/9 — defer.
      librosa.onset has its own adaptation; not our layer.]
OQ-3  Detector latency contract (windowed lookahead). Q70 metrics are
      offline; latency only matters if/when live preview is built.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _impulse_signal(fs: int, duration_s: float, impulse_positions_ms: list[float]):
    n = int(fs * duration_s)
    sig = np.zeros(n, dtype=np.float32)
    for pos_ms in impulse_positions_ms:
        idx = int(round(pos_ms * 1e-3 * fs))
        if 0 <= idx < n:
            # Short attack to be DSP-realistic.
            for k in range(8):
                if idx + k < n:
                    sig[idx + k] = (1.0 - k / 8.0)
    return sig


def _add_noise(sig, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(sig.shape).astype(np.float32)
    sig_rms = np.sqrt(np.mean(sig ** 2) + 1e-12)
    target = sig_rms / (10 ** (snr_db / 20))
    noise *= target / (np.sqrt(np.mean(noise ** 2) + 1e-12))
    return sig + noise


# ---------------------------------------------------------------
# Trivial signals
# ---------------------------------------------------------------

def test_T01_silence_returns_no_onsets():
    from voxkit.dsp.onsets import OnsetDetector
    audio = np.zeros(16_000, dtype=np.float32)
    assert OnsetDetector(sample_rate=16_000).detect(audio) == []


def test_T02_single_impulse_returns_one_onset():
    from voxkit.dsp.onsets import OnsetDetector
    sig = _impulse_signal(16_000, 1.0, [200.0])
    onsets = OnsetDetector(sample_rate=16_000).detect(sig)
    assert len(onsets) == 1


def test_T03_two_separated_impulses_return_two_onsets():
    from voxkit.dsp.onsets import OnsetDetector
    sig = _impulse_signal(16_000, 2.0, [200.0, 1500.0])
    onsets = OnsetDetector(sample_rate=16_000).detect(sig)
    assert len(onsets) == 2


def test_T04_onset_times_within_5ms_of_impulse_positions():
    from voxkit.dsp.onsets import OnsetDetector
    impulses_ms = [100.0, 350.0, 600.0, 875.0]
    sig = _impulse_signal(16_000, 1.0, impulses_ms)
    detected_s = OnsetDetector(sample_rate=16_000).detect(sig)
    detected_ms = sorted(t * 1000.0 for t in detected_s)
    for ref, det in zip(impulses_ms, detected_ms):
        assert abs(det - ref) < 5.0


# ---------------------------------------------------------------
# Pre-conditions
# ---------------------------------------------------------------

def test_T05_detector_rejects_non_mono_input():
    from voxkit.dsp.onsets import OnsetDetector
    stereo = np.zeros((16_000, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="mono"):
        OnsetDetector(sample_rate=16_000).detect(stereo)


def test_T06_detector_rejects_non_16khz_sample_rate():
    from voxkit.dsp.onsets import OnsetDetector
    with pytest.raises(ValueError, match="16000"):
        OnsetDetector(sample_rate=44_100)


# ---------------------------------------------------------------
# Click-aware preprocessing
# ---------------------------------------------------------------

def test_T07_baseline_click_track_produces_onsets_at_click_positions():
    from voxkit.dsp.onsets import OnsetDetector
    fs = 16_000
    click_times_ms = [500.0, 1000.0, 1500.0, 2000.0]
    sig = _impulse_signal(fs, 2.5, click_times_ms)
    onsets = OnsetDetector(sample_rate=fs).detect(sig)
    assert len(onsets) == len(click_times_ms)


def test_T08_click_aware_suppression_removes_click_onsets():
    from voxkit.dsp.onsets import OnsetDetector
    fs = 16_000
    click_times_ms = [500.0, 1000.0, 1500.0, 2000.0]
    sig = _impulse_signal(fs, 2.5, click_times_ms)
    click_times_s = [t / 1000.0 for t in click_times_ms]
    onsets = OnsetDetector(sample_rate=fs).detect(sig, click_track=click_times_s)
    assert onsets == []


def test_T09_performance_onsets_near_click_are_not_over_suppressed():
    """If the suppression window is too wide, real performance onsets near
    a click get eaten. Spec window is implementation-defined but bounded
    (e.g., ±20 ms); a performance onset 100 ms after a click must survive."""
    from voxkit.dsp.onsets import OnsetDetector
    fs = 16_000
    click_times_ms = [500.0]
    perf_times_ms = [600.0]
    sig = _impulse_signal(fs, 1.0, click_times_ms + perf_times_ms)
    click_times_s = [t / 1000.0 for t in click_times_ms]
    onsets = OnsetDetector(sample_rate=fs).detect(sig, click_track=click_times_s)
    assert len(onsets) == 1
    assert abs(onsets[0] * 1000.0 - 600.0) < 10.0


# ---------------------------------------------------------------
# Detection F-measure helper
# ---------------------------------------------------------------

def test_T10_f_measure_empty_inputs_returns_one():
    from voxkit.dsp.onset_eval import f_measure
    assert f_measure(detected=[], reference=[], iou_ms=50.0) == pytest.approx(1.0)


def test_T11_f_measure_perfect_match():
    from voxkit.dsp.onset_eval import f_measure
    times = [0.1, 0.5, 1.0]
    assert f_measure(detected=times, reference=times, iou_ms=50.0) == pytest.approx(1.0)


def test_T12_f_measure_zero_when_no_match():
    from voxkit.dsp.onset_eval import f_measure
    assert f_measure(detected=[0.0], reference=[1.0], iou_ms=50.0) == pytest.approx(0.0)


def test_T13_f_measure_iou_50ms_boundary():
    from voxkit.dsp.onset_eval import f_measure
    f_in = f_measure(detected=[0.049], reference=[0.0], iou_ms=50.0)
    f_out = f_measure(detected=[0.051], reference=[0.0], iou_ms=50.0)
    assert f_in == pytest.approx(1.0)
    assert f_out == pytest.approx(0.0)


# ----- TIDY FIRST checkpoint -----
# Extract `_align_pairs(detected, reference, tolerance_ms)` into the
# eval module; both `f_measure` and `alignment_mae` should call it.
# Structural change only; tests must stay green.


# ---------------------------------------------------------------
# Alignment MAE helper (Q70)
# ---------------------------------------------------------------

def test_T14_mae_zero_for_perfectly_aligned_detections():
    from voxkit.dsp.onset_eval import alignment_mae
    times = [0.1, 0.2, 0.3]
    assert alignment_mae(detected=times, reference=times, iou_ms=50.0) == pytest.approx(0.0)


def test_T15_mae_returns_constant_offset():
    from voxkit.dsp.onset_eval import alignment_mae
    ref = [0.1, 0.2, 0.3]
    det = [t + 0.012 for t in ref]   # 12 ms late on every detection
    assert alignment_mae(detected=det, reference=ref, iou_ms=50.0) == pytest.approx(12.0, abs=0.01)


def test_T16_mae_uses_median_not_mean():
    from voxkit.dsp.onset_eval import alignment_mae
    # Three 10 ms errors and one 100 ms outlier; mean = 32.5, median = 10.
    ref = [0.1, 0.2, 0.3, 0.4]
    det = [0.110, 0.210, 0.310, 0.500]
    assert alignment_mae(detected=det, reference=ref, iou_ms=50.0) == pytest.approx(10.0, abs=0.5)


def test_T17_mae_ignores_unmatched_detections():
    from voxkit.dsp.onset_eval import alignment_mae
    ref = [0.1, 0.2]
    det = [0.110, 0.210, 0.999]   # third detection has no reference partner
    assert alignment_mae(detected=det, reference=ref, iou_ms=50.0) == pytest.approx(10.0, abs=0.5)


# ---------------------------------------------------------------
# Two-tier release gate (Q70)
# ---------------------------------------------------------------

def test_T18_release_gate_passes_when_both_tiers_pass():
    from voxkit.dsp.onset_eval import release_gate
    result = release_gate(f=0.93, mae_ms=12.0, dataset="AVP")
    assert result.passed


def test_T19_release_gate_fails_alignment_tier():
    from voxkit.dsp.onset_eval import release_gate
    result = release_gate(f=0.93, mae_ms=17.0, dataset="AVP")
    assert not result.passed
    assert "alignment" in result.failed_tier


def test_T20_release_gate_fails_detection_tier():
    from voxkit.dsp.onset_eval import release_gate
    result = release_gate(f=0.90, mae_ms=10.0, dataset="AVP")
    assert not result.passed
    assert "detection" in result.failed_tier


def test_T21_release_gate_uses_dataset_specific_thresholds():
    from voxkit.dsp.onset_eval import release_gate
    avp = release_gate(f=0.89, mae_ms=12.0, dataset="AVP")     # fails (need >= 0.92)
    ood = release_gate(f=0.89, mae_ms=12.0, dataset="OOD")     # passes (need >= 0.88)
    assert not avp.passed
    assert ood.passed


# ---------------------------------------------------------------
# Quality on synthetic high-precision data
# ---------------------------------------------------------------

def test_T22_synthetic_high_snr_above_99_f_measure_under_5ms_mae():
    from voxkit.dsp.onsets import OnsetDetector
    from voxkit.dsp.onset_eval import f_measure, alignment_mae
    rng = np.random.default_rng(22)
    fs = 16_000
    duration_s = 30.0
    n_impulses = 100
    # Random, well-separated impulse times in [0.5, duration - 0.5]
    raw = sorted(rng.uniform(0.5, duration_s - 0.5, size=n_impulses * 3).tolist())
    times_ms = []
    last = -1.0
    for t in raw:
        if (t * 1000.0) - last > 100.0:
            times_ms.append(t * 1000.0)
            last = t * 1000.0
        if len(times_ms) == n_impulses:
            break
    sig = _impulse_signal(fs, duration_s, times_ms)
    sig = _add_noise(sig, snr_db=40, seed=22)
    detected = OnsetDetector(sample_rate=fs).detect(sig)
    reference = [t / 1000.0 for t in times_ms]
    assert f_measure(detected, reference, iou_ms=50.0) >= 0.99
    assert alignment_mae(detected, reference, iou_ms=50.0) <= 5.0


# ---------------------------------------------------------------
# AVP-tier release gate (slow)
# ---------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.dataset_required("AVP")
def test_T23_avp_release_gate_passes():
    from voxkit.dsp.onsets import OnsetDetector
    from voxkit.dsp.onset_eval import release_gate, evaluate_corpus
    detector = OnsetDetector(sample_rate=16_000)
    f, mae = evaluate_corpus(detector, corpus="AVP", iou_ms=50.0)
    result = release_gate(f=f, mae_ms=mae, dataset="AVP")
    assert result.passed, f"AVP release gate failed: F={f:.3f}, MAE={mae:.1f}ms"


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T24_f_measure_penalizes_false_positives():
    """A detector firing 1000 spurious events on a single-impulse signal
    must score badly. T11/T12 only test TPs and FNs; without this test,
    `f_measure(detected=[...]*1000, reference=[0.5]) == 1.0` would not
    be caught."""
    from voxkit.dsp.onset_eval import f_measure
    detected = [i * 0.05 for i in range(20)]   # 20 detections every 50 ms
    reference = [0.5]                          # one true event
    f = f_measure(detected=detected, reference=reference, iou_ms=50.0)
    # Precision = 1/20 = 0.05; Recall = 1.0 → F = 2*P*R/(P+R) ≈ 0.095
    assert f < 0.20, f"FP-heavy detector scored F={f:.3f}; expected < 0.20"


def test_T25_mae_function_name_matches_statistic_used():
    """Q70 robustness: the spec wants robust-to-outliers alignment, which
    points to median. T16 in this file already asserts median behavior
    (3 errors of 10ms + 1 outlier of 100ms → 10, not 32.5). The function
    name 'alignment_mae' (Mean Absolute Error) is therefore misleading.

    This test asserts the rename to 'alignment_median_ae' has happened;
    if the implementer chose to keep the MAE name and switch to mean,
    update T16 instead and remove this test.
    """
    from voxkit.dsp import onset_eval
    assert hasattr(onset_eval, "alignment_median_ae"), (
        "Q70 robustness implies median (per T16); function should be "
        "named alignment_median_ae for honesty. Keep alignment_mae as a "
        "deprecated alias if needed."
    )


def test_T26_close_onsets_documented_behavior():
    """Two impulses 4 ms apart: detector either merges (returns 1) or
    keeps both (returns 2). Spec must document the choice; test pins
    whichever the implementer chose so it doesn't silently flip."""
    from voxkit.dsp.onsets import OnsetDetector, ONSET_MIN_SEPARATION_MS
    fs = 16_000
    sig = _impulse_signal(fs, 0.5, [200.0, 204.0])   # 4 ms apart
    onsets = OnsetDetector(sample_rate=fs).detect(sig)
    if ONSET_MIN_SEPARATION_MS >= 5:
        assert len(onsets) == 1, "with min separation >= 5 ms, expect merge"
    else:
        assert len(onsets) == 2, "with min separation < 5 ms, expect both"


def test_T27_nan_in_audio_raises_clearly():
    """A NaN-contaminated audio buffer often comes from an upstream
    recorder bug (uninitialized ring slot). A silently-empty onset list
    would surface to the user as 'no events detected' — wrong diagnosis,
    wrong fix. Loud-fail."""
    from voxkit.dsp.onsets import OnsetDetector, AudioContainsNonFinite
    fs = 16_000
    sig = _impulse_signal(fs, 1.0, [200.0, 500.0])
    sig[1000] = np.nan
    with pytest.raises(AudioContainsNonFinite):
        OnsetDetector(sample_rate=fs).detect(sig)
