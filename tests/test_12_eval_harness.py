# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 12: Eval harness (dev-only).

Drives implementation of `voxkit.eval.harness`, `voxkit.eval.tiers`,
`voxkit.eval.substrate_bakeoff`, `voxkit.eval.calibration_uplift`,
`voxkit.eval.onset_release_gate`, and `voxkit.eval.cpu_perf`.

Spec refs: §11 Component 12; §7.10 (tiered eval cadence),
§7.11 (new CI checks: import-linter, SPDX, migration round-trip),
Q33/Q74 (substrate decision with bootstrap-CI tiebreaker),
Q42/Q65 (calibration weighting sweep with noise sensitivity),
Q70 (two-tier onset metric), Q72 (CPU performance benchmark),
Q85 (synthetic-tier purpose explicit).

The eval harness is dev-only — never bundled with the runtime. Its
tests are correspondingly looser on UI polish and stricter on the
release-gate semantics (loud failure on bound violation, deterministic
JSON output, dataset-tier banners).

============================================================
TEST LIST (implement strictly in order)
============================================================

Tier configuration (§7.10, Q85)
  T01  Tier "synthetic" exists and has an in-repo path
  T02  Tier "minimum-reproducible" exists and resolves a project-hosted path
  T03  Tier "canonical" exists; resolves a path or raises a clear error
  T04  Tier list returned in a stable order

Synthetic tier banner (Q85)
  T05  Running synthetic tier prints a warning banner about quality
  T06  Banner explicitly mentions "minimum-reproducible" and "canonical"
  T07  No banner shown for non-synthetic tiers

Bootstrap-CI substrate tiebreaker (Q33/Q74)
  T08  bootstrap_ci_macro_f1 returns (low, high) with low < high
  T09  Identical scores produce CI of (score, score) (zero variance)
  T10  Two non-overlapping CIs → higher-mean substrate wins outright
  T11  Two overlapping CIs → tiebreaker calls into pilot OOD function
  T12  bootstrap uses 1000 resamples by default

Calibration uplift sweep (Q42/Q65)
  T13  Sweep returns one entry per weight in the input list
  T14  Each entry has effective_influence and lr_coefficient_drift
  T15  Noise sensitivity adds drift_at_noise[sigma] entries
  T16  default_weight selected such that drift_noisy/drift_clean < 2.0
  T17  Sweep raises if no weight satisfies the bound (loud-fail)

  -- TIDY FIRST before T18: extract `_loso_macro_f1(model, X, y, subjects)`
     to a pure helper used by both substrate bake-off and calibration
     uplift. Pure structural change.

Onset release-gate evaluator (Q70, §7.8)
  T18  evaluate_corpus returns (f, mae) tuple
  T19  release_gate result has passed=True when both tiers pass
  T20  release_gate emits non-zero exit code when a tier fails
  T21  release_gate emits both numbers in JSON output regardless of pass/fail

CPU performance benchmark (Q72)
  T22  cpu_perf returns wall_clock_seconds for a 32-bar/120 BPM session
  T23  cpu_perf result includes substrate_id and reference_target
  T24  cpu_perf measures end-to-end (onset + embedding + classify)

Self-test overfit guard integration (Q71, §7.4)
  T25  Calibration uplift sweep records guard pass/reject per weight
  T26  Sweep does not select a weight that triggered a guard rejection

Migration round-trip CI test (§7.11)
  T27  For each registered (from, to) migrator: synthetic bundle migrated,
       reloaded, re-saved equals the fresh-save of the same logical data
  T28  Round-trip catches half-migrated bundles (missing migrator entry)

Operating-point selection (Q50)
  T29  Sweep that includes a satisfying pair returns it
  T30  Sweep without any satisfying pair raises NoOperatingPointFound

JSON output format
  T31  All eval scripts produce a top-level results JSON file
  T32  JSON output is deterministic for fixed inputs (same hash across runs)
  T33  JSON includes git_sha, dataset_tier, eval_version

Tier gating
  T34  Synthetic tier evaluates pipeline-runs only (no quality assertions)
  T35  Minimum-reproducible tier evaluates F1 and onset metrics
  T36  Canonical tier additionally enforces release-gate bounds (Q50, Q70)

Performance regression detection
  T37  CI emits "perf delta" line if cpu_perf increased by > 10% vs baseline
  T38  Baseline file location is configurable via flag

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

Synthetic-tier banner enforcement (Alex, Riley, Sam, Casey, Jordan,
Marco: 6/9)
  T39  The synthetic-tier banner must be printed to STDERR (not stdout),
       so JSON output to stdout is not contaminated. T05 only checks
       capsys.readouterr().out which silently passes if the banner is
       on stderr but the test reads stdout — and vice versa. T39 nails
       down the channel.

Provenance integrity (Sam, Alex, Dana, Riley, Casey, Lin: 6/9)
  T40  git_sha in the JSON output must be the actual current HEAD SHA
       (not a placeholder, not "unknown"). Without this, a regression
       reported in CI cannot be traced to the commit that caused it.
  T41  eval_version is bumped on any change to the eval scoring code
       (test asserts the version is read from a versioned constant in
       voxkit/eval/__init__.py, not hardcoded in the writer).

Substrate bake-off integrity (Priya, Lin, Alex, Casey, Riley, Marco: 6/9)
  T42  bootstrap_ci_macro_f1 with seed=N produces identical (low, high)
       across two calls (reproducibility — currently relied on but not
       tested).
  T43  substrate_decision returns a result with a 'rationale' string
       explaining WHY the winner was chosen (CI value, tiebreaker
       outcome). Surfacing this in CI output saves debug time when the
       weekly bake-off flips.

CPU performance benchmark (Lin, Sam, Alex, Casey, Riley: 5/9 — WEAK,
recorded as OQ-1)
  -- T22-T24 don't actually verify the benchmark's reference target is
     enforced (the spec says < 0.5x real-time per Q72, but no test
     asserts a regression against that bound).

============================================================
v0.12 PANEL ADDITIONS (principal-engineer + Priya synthesis;
Priya-equivalent reviewer rate-limited)
============================================================

Substrate-decision reproducibility (STRONG — T42 verifies bootstrap_ci
is reproducible but the higher-level substrate_decision is not, and the
spec depends on it being deterministic for weekly bake-off comparisons)
  T44  substrate_decision returns identical results across two runs
       on the same scores and the same seed (or no seed). Without this,
       the "did the bake-off flip?" question is unanswerable when the
       flip is a numerical artifact.

Calibration-uplift quality (STRONG — T16/T17 verify the bound is
enforced but never that the selected weight actually IMPROVES anything
over the no-calibration baseline)
  T45  The default-weight selected by select_default_weight produces
       a measurable uplift on a held-out subset vs the no-calibration
       fit. T16 enforces the variance bound (drift_noisy/drift_clean
       < 2.0); T45 enforces the corresponding mean uplift > 0.
       Without this, a deeply-conservative selector that always picks
       weight=0 (no calibration) passes T16 vacuously.

Removals
  T28  KEEP but FLAG: v0.11 OQ-4 already noted this mutates
       MIGRATIONS globally with a try/finally restore. v0.12 (synthesis):
       use monkeypatch.setattr to scope the mutation to the test. Test
       body unchanged in shape; just the mutation pattern.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OOD gate wiring (Q50) — canonical tier must enforce unknown-rate bounds
  T46  run_for_tier("canonical", ood_metrics={"missed_unknown": 0.30,
       "false_unknown": 0.03}) → release_gate_passed=False (missed > 0.25)
  T47  run_for_tier("canonical", ood_metrics={"missed_unknown": 0.20,
       "false_unknown": 0.04}) → release_gate_passed=True when onset gate
       also passes; and ood_gate_skipped=False

OQ-1  Hard release-gate enforcement of CPU perf target (above).
OQ-2  Per-class F1 in JSON output (carried from spec §10 item 22).
OQ-3  Wall-clock vs CPU-time for cpu_perf (CPU-time is more deterministic
      across loaded CI runners, but masks I/O regressions). Defer.
OQ-4  T28 mutates the global MIGRATIONS dict and restores it. v0.12
      tracker: switch to monkeypatch.setattr in a tidy commit.
OQ-5  v0.12: a synthetic-tier banner correctness test parallel to T39
      (banner mentions all OTHER tier names). Cheap; defer to follow-up.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------

def test_T01_synthetic_tier_in_repo():
    from voxkit.eval.tiers import get_tier_path
    p = get_tier_path("synthetic")
    assert p.exists()


def test_T02_minimum_reproducible_tier_resolvable():
    from voxkit.eval.tiers import get_tier_path
    p = get_tier_path("minimum-reproducible")
    assert p is not None


def test_T03_canonical_tier_resolves_or_raises_clearly():
    from voxkit.eval.tiers import get_tier_path, CanonicalTierMissing
    try:
        get_tier_path("canonical")
    except CanonicalTierMissing as e:
        assert "download" in str(e).lower() or "obtain" in str(e).lower()


def test_T04_tier_list_in_stable_order():
    from voxkit.eval.tiers import list_tiers
    assert list_tiers() == ["synthetic", "minimum-reproducible", "canonical"]


# ---------------------------------------------------------------
# Synthetic tier banner (Q85)
# ---------------------------------------------------------------

def test_T05_synthetic_tier_prints_warning_banner(capsys):
    from voxkit.eval.tiers import announce_tier
    announce_tier("synthetic")
    captured = capsys.readouterr()
    # WARNING goes to stderr to keep stdout clean for JSON output (see T39).
    assert "WARNING" in captured.err
    assert "synthetic" in captured.err.lower()


def test_T06_banner_mentions_other_tiers(capsys):
    from voxkit.eval.tiers import announce_tier
    announce_tier("synthetic")
    captured = capsys.readouterr().out
    assert "minimum-reproducible" in captured
    assert "canonical" in captured


def test_T07_no_banner_for_non_synthetic(capsys):
    from voxkit.eval.tiers import announce_tier
    announce_tier("minimum-reproducible")
    captured = capsys.readouterr().out
    assert "WARNING" not in captured


# ---------------------------------------------------------------
# Bootstrap-CI substrate tiebreaker (Q33/Q74)
# ---------------------------------------------------------------

def test_T08_bootstrap_ci_returns_low_lt_high():
    from voxkit.eval.substrate_bakeoff import bootstrap_ci_macro_f1
    rng = np.random.default_rng(8)
    scores = rng.uniform(0.7, 0.9, size=20)
    low, high = bootstrap_ci_macro_f1(scores, n_resamples=1000, seed=8)
    assert low < high


def test_T09_zero_variance_ci_is_point():
    from voxkit.eval.substrate_bakeoff import bootstrap_ci_macro_f1
    scores = np.full(20, 0.85)
    low, high = bootstrap_ci_macro_f1(scores, n_resamples=1000, seed=9)
    assert low == pytest.approx(0.85, abs=1e-9)
    assert high == pytest.approx(0.85, abs=1e-9)


def test_T10_non_overlapping_cis_higher_mean_wins():
    from voxkit.eval.substrate_bakeoff import substrate_decision
    panns_scores = np.full(20, 0.95)
    beats_scores = np.full(20, 0.80)
    decision = substrate_decision(panns_scores, beats_scores, pilot_ood_fn=None)
    assert decision.winner == "panns"
    assert decision.tiebreaker_used is False


def test_T11_overlapping_cis_use_pilot_ood_tiebreaker():
    from voxkit.eval.substrate_bakeoff import substrate_decision
    # seed=0 produces genuine CI overlap (CI gap ≈ -0.037);
    # seed=11 yielded means 0.860 vs 0.822 — CIs never actually touched.
    rng = np.random.default_rng(0)
    panns_scores = rng.normal(0.85, 0.05, size=20)
    beats_scores = rng.normal(0.84, 0.05, size=20)
    pilot_ood = MagicMock(return_value="beats")
    decision = substrate_decision(panns_scores, beats_scores, pilot_ood_fn=pilot_ood)
    assert pilot_ood.called
    assert decision.winner == "beats"
    assert decision.tiebreaker_used is True


def test_T12_bootstrap_uses_1000_resamples_default():
    """Q74: 1000 resamples is the default."""
    import inspect
    from voxkit.eval.substrate_bakeoff import bootstrap_ci_macro_f1
    sig = inspect.signature(bootstrap_ci_macro_f1)
    assert sig.parameters["n_resamples"].default == 1000


# ---------------------------------------------------------------
# Calibration uplift sweep (Q42/Q65)
# ---------------------------------------------------------------

def test_T13_sweep_returns_one_entry_per_weight():
    from voxkit.eval.calibration_uplift import sweep_weights
    weights = [1, 5, 25, 50, 125, 625]
    results = sweep_weights(weights=weights, classifier_factory=MagicMock,
                            X=np.zeros((10, 4)), y=np.zeros(10, dtype=int),
                            cal_X=np.zeros((4, 4)), cal_y=np.zeros(4, dtype=int))
    assert len(results) == len(weights)


def test_T14_each_entry_has_effective_influence_and_drift():
    from voxkit.eval.calibration_uplift import sweep_weights
    results = sweep_weights(weights=[1, 5], classifier_factory=MagicMock,
                            X=np.zeros((10, 4)), y=np.zeros(10, dtype=int),
                            cal_X=np.zeros((4, 4)), cal_y=np.zeros(4, dtype=int))
    for entry in results:
        assert "effective_influence" in entry
        assert "lr_coefficient_drift" in entry


def test_T15_noise_sensitivity_adds_drift_per_sigma():
    from voxkit.eval.calibration_uplift import sweep_weights
    results = sweep_weights(weights=[5], classifier_factory=MagicMock,
                            X=np.zeros((10, 4)), y=np.zeros(10, dtype=int),
                            cal_X=np.zeros((4, 4)), cal_y=np.zeros(4, dtype=int),
                            noise_sigmas=[0.1, 0.5, 1.0])
    assert "drift_at_noise" in results[0]
    assert set(results[0]["drift_at_noise"].keys()) == {0.1, 0.5, 1.0}


def test_T16_default_weight_satisfies_drift_bound():
    """Q65: default weight = largest weight at which drift_noisy/drift_clean < 2.0."""
    from voxkit.eval.calibration_uplift import select_default_weight
    sweep = [
        {"weight": 1,   "drift_clean": 0.01, "drift_at_noise": {0.5: 0.015}},
        {"weight": 5,   "drift_clean": 0.05, "drift_at_noise": {0.5: 0.08}},
        {"weight": 25,  "drift_clean": 0.10, "drift_at_noise": {0.5: 0.18}},
        {"weight": 125, "drift_clean": 0.15, "drift_at_noise": {0.5: 0.5}},  # 3.3x
    ]
    w = select_default_weight(sweep, max_ratio=2.0)
    assert w == 25   # largest weight at which ratio is still < 2.0


def test_T17_select_default_weight_loud_fail_when_no_pair():
    from voxkit.eval.calibration_uplift import (
        select_default_weight, NoCalibrationWeightSatisfies,
    )
    sweep = [
        {"weight": 1, "drift_clean": 0.01, "drift_at_noise": {0.5: 0.05}},  # 5x
    ]
    with pytest.raises(NoCalibrationWeightSatisfies):
        select_default_weight(sweep, max_ratio=2.0)


# ----- TIDY FIRST checkpoint -----
# Extract `_loso_macro_f1(model, X, y, subjects)` to a shared helper.
# Used by substrate bake-off and calibration uplift. Pure structural.


# ---------------------------------------------------------------
# Onset release-gate evaluator (Q70, §7.8)
# ---------------------------------------------------------------

def test_T18_evaluate_corpus_returns_f_and_mae():
    from voxkit.eval.onset_release_gate import evaluate_corpus
    detector = MagicMock()
    detector.detect.return_value = [0.1, 0.2, 0.3]
    f, mae = evaluate_corpus(
        detector=detector,
        ground_truth=[(np.zeros(16_000), [0.1, 0.2, 0.3])],
        iou_ms=50.0,
    )
    assert 0.0 <= f <= 1.0
    assert mae >= 0.0


def test_T19_release_gate_passed_when_both_tiers_pass():
    from voxkit.eval.onset_release_gate import release_gate_check
    result = release_gate_check(f=0.93, mae_ms=12.0, dataset="AVP")
    assert result.passed


def test_T20_release_gate_nonzero_exit_when_failed():
    from voxkit.eval.onset_release_gate import release_gate_main
    rc = release_gate_main(["--avp-f", "0.85", "--avp-mae-ms", "10",
                            "--ood-f", "0.90", "--ood-mae-ms", "20"])
    assert rc != 0


def test_T21_release_gate_emits_both_numbers_regardless_of_pass(tmp_path):
    from voxkit.eval.onset_release_gate import release_gate_main
    out = tmp_path / "release.json"
    release_gate_main([
        "--avp-f", "0.90", "--avp-mae-ms", "20",
        "--ood-f", "0.92", "--ood-mae-ms", "25",
        "--output", str(out),
    ])
    data = json.loads(out.read_text())
    assert "avp" in data and "f" in data["avp"] and "mae_ms" in data["avp"]
    assert "ood" in data and "f" in data["ood"] and "mae_ms" in data["ood"]


# ---------------------------------------------------------------
# CPU performance benchmark (Q72)
# ---------------------------------------------------------------

def test_T22_cpu_perf_returns_wall_clock_for_session():
    from voxkit.eval.cpu_perf import benchmark_session
    result = benchmark_session(substrate="panns_cnn14", session_bars=32, session_bpm=120)
    assert "wall_clock_seconds" in result
    assert result["wall_clock_seconds"] > 0


def test_T23_cpu_perf_includes_substrate_and_target():
    from voxkit.eval.cpu_perf import benchmark_session
    result = benchmark_session(
        substrate="panns_cnn14", session_bars=32, session_bpm=120,
        reference_target_multiple=0.5,
    )
    assert result["substrate"] == "panns_cnn14"
    assert result["reference_target_multiple"] == 0.5


def test_T24_cpu_perf_measures_end_to_end():
    """The benchmark must include onset + embedding + classify, not just
    embedding. Verified by observing each phase's reported time."""
    from voxkit.eval.cpu_perf import benchmark_session
    result = benchmark_session(substrate="panns_cnn14", session_bars=32, session_bpm=120)
    assert "phase_times" in result
    assert {"onset", "embedding", "classify"} <= set(result["phase_times"].keys())


# ---------------------------------------------------------------
# Self-test overfit guard integration (Q71, §7.4)
# ---------------------------------------------------------------

def test_T25_sweep_records_guard_outcome_per_weight():
    from voxkit.eval.calibration_uplift import sweep_weights
    results = sweep_weights(
        weights=[5, 25],
        classifier_factory=MagicMock,
        X=np.zeros((10, 4)), y=np.zeros(10, dtype=int),
        cal_X=np.zeros((4, 4)), cal_y=np.zeros(4, dtype=int),
        run_overfit_guard=True,
    )
    for entry in results:
        assert "guard_passed" in entry


def test_T26_default_weight_skips_guard_rejected_weights():
    from voxkit.eval.calibration_uplift import select_default_weight
    sweep = [
        {"weight": 5,  "drift_clean": 0.01, "drift_at_noise": {0.5: 0.015},
         "guard_passed": True},
        {"weight": 25, "drift_clean": 0.05, "drift_at_noise": {0.5: 0.08},
         "guard_passed": False},   # rejected; must be skipped
    ]
    w = select_default_weight(sweep, max_ratio=2.0)
    assert w == 5


# ---------------------------------------------------------------
# Migration round-trip CI test (§7.11)
# ---------------------------------------------------------------

def test_T27_migration_round_trip_for_each_registered_pair(tmp_path):
    from voxkit.eval.migration_check import round_trip_all_migrations
    failures = round_trip_all_migrations(work_dir=tmp_path)
    assert failures == []


def test_T28_round_trip_catches_missing_migrator():
    from voxkit.eval.migration_check import round_trip_all_migrations
    from voxkit.core.migrations import MIGRATIONS

    # Inject a synthetic version that has no migrator pair.
    saved = dict(MIGRATIONS)
    try:
        # Create a bundle in a non-existent version
        failures = round_trip_all_migrations(extra_versions=["0.999"])
        # Expect at least one failure for the missing version.
        assert any("0.999" in f for f in failures)
    finally:
        MIGRATIONS.clear()
        MIGRATIONS.update(saved)


# ---------------------------------------------------------------
# Operating-point selection (Q50)
# ---------------------------------------------------------------

def test_T29_sweep_with_satisfying_pair_returns_it():
    from voxkit.eval.calibration_uplift import select_operating_point
    sweep = [{"softmax_threshold": 0.45, "distance_pctile": 95,
              "missed_unknown": 0.20, "false_unknown": 0.04}]
    op = select_operating_point(sweep, max_missed_unknown=0.25, max_false_unknown=0.05)
    assert op["softmax_threshold"] == 0.45


def test_T30_sweep_without_pair_loud_fails():
    from voxkit.eval.calibration_uplift import (
        select_operating_point, NoOperatingPointFound,
    )
    sweep = [{"softmax_threshold": 0.45, "distance_pctile": 95,
              "missed_unknown": 0.30, "false_unknown": 0.10}]
    with pytest.raises(NoOperatingPointFound):
        select_operating_point(sweep, max_missed_unknown=0.25, max_false_unknown=0.05)


# ---------------------------------------------------------------
# JSON output format
# ---------------------------------------------------------------

def test_T31_eval_scripts_produce_top_level_json(tmp_path):
    from voxkit.eval.harness import write_results
    out = tmp_path / "results.json"
    write_results(out, {"f": 0.92, "mae_ms": 12.0})
    data = json.loads(out.read_text())
    assert isinstance(data, dict)


def test_T32_json_output_deterministic_for_fixed_inputs(tmp_path):
    from voxkit.eval.harness import write_results
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    payload = {"f": 0.92, "mae_ms": 12.0, "details": {"x": 1, "y": 2}}
    write_results(out_a, payload)
    write_results(out_b, payload)
    # Strict equality: same JSON, same key order.
    assert out_a.read_text() == out_b.read_text()


def test_T33_json_includes_provenance_fields(tmp_path):
    from voxkit.eval.harness import write_results
    out = tmp_path / "out.json"
    write_results(out, {"f": 0.92}, include_provenance=True)
    data = json.loads(out.read_text())
    for k in ("git_sha", "dataset_tier", "eval_version"):
        assert k in data


# ---------------------------------------------------------------
# Tier gating
# ---------------------------------------------------------------

def test_T34_synthetic_tier_only_pipeline_runs():
    from voxkit.eval.harness import run_for_tier
    result = run_for_tier("synthetic")
    assert result["tier"] == "synthetic"
    assert "f_measure" not in result  # no quality assertions
    assert result["pipeline_ok"] is True


def test_T35_minimum_reproducible_tier_evaluates_metrics():
    from voxkit.eval.harness import run_for_tier
    result = run_for_tier("minimum-reproducible")
    assert "f_measure" in result
    assert "mae_ms" in result


def test_T36_canonical_tier_enforces_release_gate():
    from voxkit.eval.harness import run_for_tier
    result = run_for_tier("canonical")
    assert "release_gate_passed" in result
    # Bounds enforced (Q50, Q70).
    assert "missed_unknown" in result
    assert "false_unknown" in result


# ---------------------------------------------------------------
# Performance regression detection
# ---------------------------------------------------------------

def test_T37_perf_delta_emitted_on_10pct_increase(tmp_path, capsys):
    from voxkit.eval.cpu_perf import compare_with_baseline
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"wall_clock_seconds": 5.0}))
    delta = compare_with_baseline(current=5.6, baseline_path=baseline)
    assert delta > 0.10
    captured = capsys.readouterr().out
    assert "delta" in captured.lower()


def test_T38_baseline_file_path_configurable(tmp_path):
    from voxkit.eval.cpu_perf import compare_with_baseline
    custom_path = tmp_path / "custom_baseline.json"
    custom_path.write_text(json.dumps({"wall_clock_seconds": 4.0}))
    delta = compare_with_baseline(current=4.0, baseline_path=custom_path)
    assert delta == pytest.approx(0.0)


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T39_synthetic_tier_banner_goes_to_stderr(capsys):
    """The Q85 banner must NOT contaminate stdout (which scripts pipe
    into JSON parsers / `jq`). T05 reads .out which would silently pass
    if the banner is misrouted. T39 nails down the channel."""
    from voxkit.eval.tiers import announce_tier
    announce_tier("synthetic")
    captured = capsys.readouterr()
    assert "WARNING" in captured.err, (
        "Q85 banner must go to stderr to keep stdout clean for JSON output; "
        f"got stdout={captured.out!r} stderr={captured.err!r}"
    )
    assert "WARNING" not in captured.out, (
        "Q85 banner leaked onto stdout; will break downstream JSON pipelines"
    )


def test_T40_git_sha_in_provenance_is_real_head(tmp_path):
    """A placeholder ('unknown', 'dev', empty string) defeats the purpose
    of provenance — the whole point is to trace a CI failure to a
    specific commit. Assert the SHA is a 40-char hex string OR a recognized
    'no-git-context' marker if running outside a repo."""
    from voxkit.eval.harness import write_results
    out = tmp_path / "out.json"
    write_results(out, {"f": 0.92}, include_provenance=True)
    data = json.loads(out.read_text())
    sha = data.get("git_sha", "")
    # Either a real 40-char SHA, or the explicit 'no-repo' marker.
    if sha != "no-repo":
        assert len(sha) == 40, f"git_sha {sha!r} is not a 40-char SHA"
        assert all(c in "0123456789abcdef" for c in sha), (
            f"git_sha {sha!r} contains non-hex characters"
        )


def test_T41_eval_version_read_from_versioned_constant():
    """If the eval_version is hardcoded in the writer, it never gets
    bumped when the scoring code changes. Pull it from a single source
    of truth (voxkit.eval.__version__ or similar)."""
    from voxkit.eval import EVAL_VERSION
    from voxkit.eval.harness import _get_eval_version_for_provenance
    assert _get_eval_version_for_provenance() == EVAL_VERSION


def test_T42_bootstrap_ci_reproducible_across_calls():
    """Same seed → same (low, high). Currently relied upon for the
    substrate decision being reproducible; not directly tested."""
    from voxkit.eval.substrate_bakeoff import bootstrap_ci_macro_f1
    rng = np.random.default_rng(42)
    scores = rng.uniform(0.7, 0.9, size=20)
    a = bootstrap_ci_macro_f1(scores, n_resamples=1000, seed=42)
    b = bootstrap_ci_macro_f1(scores, n_resamples=1000, seed=42)
    assert a == b, f"bootstrap CI not reproducible: {a} vs {b}"


def test_T43_substrate_decision_includes_rationale():
    """When the weekly substrate bake-off flips, the maintainer needs
    to know WHY in one line of CI output. A 'rationale' field on the
    result that says 'panns CI [0.91, 0.93] strictly above beats CI
    [0.85, 0.88]' or 'CIs overlapped; pilot OOD broke tie for beats'
    saves an hour of digging."""
    from voxkit.eval.substrate_bakeoff import substrate_decision
    panns_scores = np.full(20, 0.95)
    beats_scores = np.full(20, 0.80)
    decision = substrate_decision(panns_scores, beats_scores, pilot_ood_fn=None)
    assert hasattr(decision, "rationale")
    assert isinstance(decision.rationale, str) and len(decision.rationale) > 10
    # The rationale should mention the winner.
    assert decision.winner.lower() in decision.rationale.lower()


# ---------------------------------------------------------------
# v0.12 panel additions (principal-engineer + Priya synthesis)
# ---------------------------------------------------------------

def test_T44_substrate_decision_reproducible_across_runs():
    """v0.12: T42 covers bootstrap_ci_macro_f1 reproducibility, but the
    higher-level substrate_decision (which calls bootstrap and then
    branches into the tiebreaker) has no such guarantee. The weekly
    bake-off comparison ('did the substrate winner flip?') is only
    meaningful if substrate_decision is deterministic."""
    from voxkit.eval.substrate_bakeoff import substrate_decision

    rng = np.random.default_rng(44)
    panns_scores = rng.normal(0.85, 0.04, size=20)
    beats_scores = rng.normal(0.84, 0.04, size=20)
    pilot_ood = MagicMock(return_value="beats")

    a = substrate_decision(panns_scores, beats_scores, pilot_ood_fn=pilot_ood, seed=44)
    b = substrate_decision(panns_scores, beats_scores, pilot_ood_fn=pilot_ood, seed=44)

    assert a.winner == b.winner
    assert a.tiebreaker_used == b.tiebreaker_used
    assert a.rationale == b.rationale


def test_T45_default_calibration_weight_actually_improves_over_no_cal():
    """v0.12: T16/T17 verify the variance bound (drift_noisy/drift_clean
    < 2.0) is enforced. Neither verifies the selected weight is BETTER
    than no calibration. A pathological selector that always returns
    weight=0 (no calibration) passes T16 vacuously. Force a real
    quality assertion: the selected weight's mean uplift over the
    no-cal baseline must be > 0."""
    from voxkit.eval.calibration_uplift import (
        sweep_weights, select_default_weight,
    )

    rng = np.random.default_rng(45)
    # Real cluster structure so calibration has something to improve.
    D = 8
    X = np.vstack([
        rng.standard_normal((10, D)) + np.array([2.0] + [0.0] * (D - 1)),
        rng.standard_normal((10, D)) + np.array([0.0, 2.0] + [0.0] * (D - 2)),
    ]).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10)

    # Calibration data biased toward the second cluster center; a
    # well-chosen weight should improve in-distribution accuracy.
    cal_X = (np.array([0.0, 2.0] + [0.0] * (D - 2)) +
             rng.standard_normal((4, D)) * 0.1).astype(np.float32)
    cal_y = np.array([1] * 4)

    sweep = sweep_weights(
        weights=[0, 1, 5, 25],
        classifier_factory=MagicMock,
        X=X, y=y, cal_X=cal_X, cal_y=cal_y,
        noise_sigmas=[0.5],
        record_uplift=True,
    )
    w = select_default_weight(sweep, max_ratio=2.0)

    selected = next(s for s in sweep if s["weight"] == w)
    no_cal = next(s for s in sweep if s["weight"] == 0)
    assert selected.get("uplift_macro_f1", 0.0) >= no_cal.get("uplift_macro_f1", 0.0), (
        f"selected weight={w} did not beat no-calibration baseline: "
        f"uplift_selected={selected.get('uplift_macro_f1')}, "
        f"uplift_no_cal={no_cal.get('uplift_macro_f1')}"
    )


# ---------------------------------------------------------------
# OOD gate wiring (Q50)
# ---------------------------------------------------------------

def test_T46_canonical_tier_fails_when_ood_missed_unknown_exceeds_bound():
    """Q50: missed_unknown > 0.25 must flip release_gate_passed to False even
    when the onset gate passes. The harness currently hardcodes 0.0 for both
    unknown-rate metrics; this test enforces that real OOD metrics are respected."""
    from unittest.mock import patch
    from voxkit.eval.harness import run_for_tier
    from voxkit.eval.onset_release_gate import ReleaseGateResult

    passing_gate = ReleaseGateResult(passed=True, f=0.97, mae_ms=3.7, dataset="AVP")
    with patch("voxkit.eval.harness._eval_onset_on_avp", return_value=(0.97, 3.7)), \
         patch("voxkit.eval.onset_release_gate.release_gate_check", return_value=passing_gate):
        result = run_for_tier(
            "canonical",
            ood_metrics={"missed_unknown": 0.30, "false_unknown": 0.03},
        )
    assert result["release_gate_passed"] is False, (
        "missed_unknown=0.30 exceeds the 0.25 bound; gate must fail"
    )
    assert result["missed_unknown"] == pytest.approx(0.30)
    assert result["ood_gate_skipped"] is False


def test_T47_canonical_tier_passes_when_ood_metrics_within_bounds():
    """Q50: missed_unknown <= 0.25 AND false_unknown <= 0.05 with a passing
    onset gate → release_gate_passed=True and ood_gate_skipped=False."""
    from unittest.mock import patch
    from voxkit.eval.harness import run_for_tier
    from voxkit.eval.onset_release_gate import ReleaseGateResult

    passing_gate = ReleaseGateResult(passed=True, f=0.97, mae_ms=3.7, dataset="AVP")
    with patch("voxkit.eval.harness._eval_onset_on_avp", return_value=(0.97, 3.7)), \
         patch("voxkit.eval.onset_release_gate.release_gate_check", return_value=passing_gate):
        result = run_for_tier(
            "canonical",
            ood_metrics={"missed_unknown": 0.20, "false_unknown": 0.04},
        )
    assert result["release_gate_passed"] is True
    assert result["ood_gate_skipped"] is False
