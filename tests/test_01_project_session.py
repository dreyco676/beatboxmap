# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 1: Project & Session.

Drives implementation of `voxkit.core.session`, `voxkit.core.taxonomy`,
`voxkit.core.manifest`, and `voxkit.core.migrations`.

Spec refs: §11 Component 1; Q66 (TaxonomyConfig), Q68 (Cholesky storage),
Q78 (voxkit_format_version), §7.11 (migration round-trip).

============================================================
TEST LIST (implement strictly in order; each Red drives one
piece of behavior; refactor only on Green)
============================================================

TaxonomyConfig (Q66)
  T01  default_v1_0() returns a config with the four trained classes
  T02  default_v1_0() midi_mapping covers every class with the GM defaults
  T03  unknown_class_id defaults to "unknown"
  T04  classes must be non-empty (constructor rejects ())
  T05  every class must have a midi_mapping entry (constructor rejects holes)
  T06  unknown_class_id must not appear in classes (rejects collision)

MahalanobisFullDim (Q68)
  T07  pooled_cov_cholesky must be lower-triangular at construction
  T08  class_centroids first dim must equal len(classes)
  T09  pooled_cov_cholesky shape is (D, D) and matches centroid feature dim
  T10  distance_thresholds shape is (n_classes,)

Session
  T11  Session can be constructed with required fields
  T12  Default dropped_buffer_count is 0 (Q67 telemetry seed)
  T13  Default taxonomy is TaxonomyConfig.default_v1_0()

ProjectManifest (Q78)
  T14  Manifest stamps voxkit_format_version on construction
  T15  Manifest with missing voxkit_format_version defaults to "0.4"
       on load (legacy bundle handling)

Migration table (Q78, §7.11)
  T16  Empty migration table walks zero steps and returns input unchanged
  T17  Single-step table walks one migrator
  T18  Multi-step table walks the chain in order from→to
  T19  Walking past the registered table raises a clear error
  T20  v0.10 → v0.11 migrator is a no-op on data and stamps version

  -- TIDY FIRST before T21: extract `_walk_migrations` into its own
     module (`voxkit.core.migrations`) once the chain logic stabilizes;
     keep tests green; commit structural change separately.

  T21  v0.9 → v0.10 migrator converts pooled_inv_covariance
       to pooled_cov_cholesky (round-trip preserves Mahalanobis distance)
  T22  Round-trip: synthetic v0.4 bundle → migrate → save → load matches
       fresh save of the same logical data (release-gate-style §7.11)

Save/Load
  T23  Saving a Session produces a bundle with a manifest.json containing
       voxkit_format_version="0.11"
  T24  Saving and reloading a Session preserves all event timestamps
       and class ids
  T25  Loading a v0.10 bundle (Cholesky present) produces an equivalent
       Session with the same Mahalanobis distances on representative inputs

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus, see review notes)
============================================================

Forward compatibility & format hygiene (Q78 amplified)
  T26  Bundle stamped with a version newer than the running build
       (e.g., "0.99") raises ForwardCompatVersionError, not silently
       loaded as the latest. [Sam, Lin, Riley, Alex, Dana, Casey: 6/9]
       Rationale: a user opening someone else's bundle from a future
       build must see a clear error, not a silently-mis-migrated state.
  T27  voxkit_format_version field is strict-string at write time
       (rejects float/int and bare numbers like 0.11 or 11). Catches
       accidental JSON-serialization drift. [Sam, Alex, Dana, Riley,
       Marco, Casey: 6/9]
  T28  Atomic save: process killed mid-save (or save raises) leaves the
       previous bundle on disk intact (write-to-temp + rename pattern).
       [Sam, Alex, Riley, Lin, Casey, Dana: 6/9]
       Rationale: a hobby user with a single .voxkit file losing it
       to a half-write is the worst possible UX failure.

Migration robustness
  T29  walk_migrations parameter naming consistency: tests use
       to_version= but spec text §11 Component 1 shows target=. The
       implementation must pick one and the docstring must match the
       spec; this test asserts whichever the implementer chooses by
       calling the function via both kwargs and asserting one raises
       TypeError. [Alex, Sam, Riley: 3/9 — RECORDED AS OPEN QUESTION
       OQ-1; not adopted as a behavioral test, only a doc-alignment
       chore for the implementer.]

============================================================
v0.12 PANEL ADDITIONS (principal-engineer synthesis;
three of four v0.12 review agents rate-limited — Sam-equivalent
architecture review represented by synthesis only)
============================================================

Migration partial-state observability (STRONG)
  T29  When a registered migrator itself raises mid-walk, the loader
       surfaces the original exception with context (which step
       failed) AND no partial state is observable via load_session.
       Today: a migrator raising leaves state undefined; T18 only
       tests the happy path. Migrations are append-only and ordered;
       a partial walk is corrupt-by-construction and must loud-fail.

Forward-compat semver detection (STRONG — string-sort defeats the
purpose of forward-compat detection)
  T30  ForwardCompatVersionError uses semver-aware comparison, NOT
       lexical string sort. Lexically "0.10" < "0.11" but also
       "0.10" < "0.9" — a lexical compare fails to detect "0.9 is
       older than 0.10" and would raise ForwardCompat on a LEGACY
       bundle. Pin numeric semver compare with explicit cases.

Save atomicity beyond mid-write failure (STRONG — T28 covers one
failure mode; the others are equally common)
  T31  After save_session() returns successfully, the temp file used
       for the atomic-replace is cleaned up (no .voxkit.tmp lingering
       siblings on disk). Hobby users have hit "what is this .tmp
       file?" with at least three other audio tools; clean up after
       yourself.

Removals / rewrites
  T28  REWRITE: stop patching the private symbol _finalize_bundle_write.
       Couple to BEHAVIOR: simulate a write failure by patching the
       bundle's open() at the OS level via monkeypatch on builtins.open
       restricted to the target path. Implementation-name coupling
       breaks the test if the implementer renames the private helper,
       even though no behavioral contract changed.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  Param name to_version vs target (see T29 above). Spec text and
      test signatures disagree. Resolve before T16 lands.
OQ-2  Audio dtype validation: Session rejects audio that isn't float32.
      [Sam, Casey: 2/9 — REJECTED. Recorder is the source of truth on
      dtype; Session can assume float32 by upstream contract.]
OQ-3  Concurrent save safety (two VoxKit processes writing to the same
      bundle). Defer; not a v1.0 use case.
OQ-4  v0.12: file-locking on save (advisory flock) so a crashed prior
      VoxKit instance doesn't leave the .voxkit half-written. Defer;
      v1.0 hobby-scope acceptable, atomic-replace covers most of it.
"""

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------
# TaxonomyConfig (Q66)
# ---------------------------------------------------------------

def test_T01_default_v1_0_has_four_trained_classes():
    from voxkit.core.taxonomy import TaxonomyConfig
    cfg = TaxonomyConfig.default_v1_0()
    assert cfg.classes == ("kick", "snare", "closed_hat", "open_hat")


def test_T02_default_v1_0_midi_mapping_uses_gm_defaults():
    from voxkit.core.taxonomy import TaxonomyConfig
    cfg = TaxonomyConfig.default_v1_0()
    assert cfg.midi_mapping == {
        "kick": 36, "snare": 38, "closed_hat": 42, "open_hat": 46,
    }


def test_T03_unknown_class_id_default_value():
    from voxkit.core.taxonomy import TaxonomyConfig
    cfg = TaxonomyConfig.default_v1_0()
    assert cfg.unknown_class_id == "unknown"


def test_T04_empty_classes_rejected():
    from voxkit.core.taxonomy import TaxonomyConfig
    with pytest.raises(ValueError, match="classes"):
        TaxonomyConfig(classes=(), midi_mapping={})


def test_T05_midi_mapping_must_cover_every_class():
    from voxkit.core.taxonomy import TaxonomyConfig
    with pytest.raises(ValueError, match="midi_mapping"):
        TaxonomyConfig(
            classes=("kick", "snare"),
            midi_mapping={"kick": 36},  # snare missing
        )


def test_T06_unknown_class_id_cannot_collide_with_trained_class():
    from voxkit.core.taxonomy import TaxonomyConfig
    with pytest.raises(ValueError, match="unknown_class_id"):
        TaxonomyConfig(
            classes=("kick", "unknown"),
            midi_mapping={"kick": 36, "unknown": 99},
            unknown_class_id="unknown",
        )


# ---------------------------------------------------------------
# MahalanobisFullDim (Q68)
# ---------------------------------------------------------------

def test_T07_pooled_cov_cholesky_must_be_lower_triangular():
    from voxkit.core.session import MahalanobisFullDim
    bad = np.array([[1.0, 0.5], [0.5, 1.0]])  # not lower triangular
    with pytest.raises(ValueError, match="lower"):
        MahalanobisFullDim(
            class_centroids=np.zeros((2, 2)),
            pooled_cov_cholesky=bad,
            distance_thresholds=np.zeros(2),
        )


def test_T08_class_centroids_first_dim_must_equal_n_classes():
    from voxkit.core.session import MahalanobisFullDim
    L = np.tril(np.eye(2))
    with pytest.raises(ValueError, match="centroids"):
        MahalanobisFullDim(
            class_centroids=np.zeros((3, 2)),       # 3 centroids
            pooled_cov_cholesky=L,                  # D=2
            distance_thresholds=np.zeros(2),         # 2 classes
        )


def test_T09_pooled_cov_cholesky_dim_must_match_centroid_features():
    from voxkit.core.session import MahalanobisFullDim
    L = np.tril(np.eye(3))                          # D=3
    with pytest.raises(ValueError, match="dim"):
        MahalanobisFullDim(
            class_centroids=np.zeros((2, 2)),       # D=2
            pooled_cov_cholesky=L,
            distance_thresholds=np.zeros(2),
        )


def test_T10_distance_thresholds_shape_matches_classes():
    from voxkit.core.session import MahalanobisFullDim
    with pytest.raises(ValueError, match="thresholds"):
        MahalanobisFullDim(
            class_centroids=np.zeros((4, 8)),
            pooled_cov_cholesky=np.tril(np.eye(8)),
            distance_thresholds=np.zeros(3),         # mismatch
        )


# ---------------------------------------------------------------
# Session
# ---------------------------------------------------------------

def test_T11_session_constructable_with_required_fields():
    from voxkit.core.session import Session, TimeSignature
    s = Session(
        bpm=120.0,
        time_signature=TimeSignature(4, 4),
        bars=4,
        sample_rate=16_000,
        recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000, dtype=np.float32),
    )
    assert s.bpm == 120.0
    assert s.bars == 4


def test_T12_default_dropped_buffer_count_is_zero():
    """Q67: counter seeded at 0 on Session creation; non-default values
    only after recorder activity."""
    from voxkit.core.session import Session, TimeSignature
    s = Session(
        bpm=120.0,
        time_signature=TimeSignature(4, 4),
        bars=4,
        sample_rate=16_000,
        recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000, dtype=np.float32),
    )
    assert s.dropped_buffer_count == 0


def test_T13_default_taxonomy_is_v1_0():
    from voxkit.core.session import Session, TimeSignature
    from voxkit.core.taxonomy import TaxonomyConfig
    s = Session(
        bpm=120.0,
        time_signature=TimeSignature(4, 4),
        bars=4,
        sample_rate=16_000,
        recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000, dtype=np.float32),
    )
    assert s.taxonomy == TaxonomyConfig.default_v1_0()


# ---------------------------------------------------------------
# ProjectManifest (Q78)
# ---------------------------------------------------------------

def test_T14_manifest_stamps_voxkit_format_version():
    from voxkit.core.manifest import ProjectManifest
    m = ProjectManifest(voxkit_format_version="0.11")
    assert m.voxkit_format_version == "0.11"


def test_T15_legacy_bundle_without_version_field_is_treated_as_0_4():
    """Q78: missing field is the 'this is a legacy bundle' signal.
    Loader must not silently invent a version; the dispatcher relies on
    the explicit '0.4' default."""
    from voxkit.core.manifest import ProjectManifest
    m = ProjectManifest.from_raw_dict({})  # legacy: no field present
    assert m.voxkit_format_version == "0.4"


# ---------------------------------------------------------------
# Migration table (Q78, §7.11)
# ---------------------------------------------------------------

def test_T16_empty_migration_table_returns_input_unchanged():
    from voxkit.core.migrations import walk_migrations
    raw = {"manifest": {"voxkit_format_version": "0.11"}}
    out = walk_migrations(raw, from_version="0.11", to_version="0.11", table={})
    assert out == raw


def test_T17_single_step_table_walks_one_migrator():
    from voxkit.core.migrations import walk_migrations
    table = {("0.10", "0.11"): lambda d: {**d, "stamped": True}}
    raw = {"manifest": {"voxkit_format_version": "0.10"}}
    out = walk_migrations(raw, from_version="0.10", to_version="0.11", table=table)
    assert out["stamped"] is True


def test_T18_multi_step_table_walks_chain_in_order():
    """Order matters: each step's output is the next step's input."""
    from voxkit.core.migrations import walk_migrations
    calls = []

    def step(name):
        def fn(d):
            calls.append(name)
            return d
        return fn

    table = {
        ("0.4", "0.5"): step("a"),
        ("0.5", "0.6"): step("b"),
        ("0.6", "0.11"): step("c"),
    }
    walk_migrations({}, from_version="0.4", to_version="0.11", table=table)
    assert calls == ["a", "b", "c"]


def test_T19_walking_past_registered_table_raises_clear_error():
    from voxkit.core.migrations import walk_migrations, MigrationPathNotFound
    with pytest.raises(MigrationPathNotFound):
        walk_migrations({}, from_version="0.4", to_version="0.99", table={})


def test_T20_v010_to_v011_is_data_noop_and_stamps_version():
    from voxkit.core.migrations import migrate_0_10_to_0_11
    raw = {
        "manifest": {"voxkit_format_version": "0.10"},
        "events": [{"t": 0.5, "class": "kick"}],
    }
    out = migrate_0_10_to_0_11(raw)
    assert out["manifest"]["voxkit_format_version"] == "0.11"
    assert out["events"] == raw["events"]   # data unchanged


# ----- TIDY FIRST checkpoint -----
# Before T21, the migrators have grown and `walk_migrations` lives in
# `voxkit.core.session`. Move it to `voxkit.core.migrations` in its own
# commit. No behavior change; tests stay green. Then add T21.


def test_T21_v09_to_v010_converts_inverse_covariance_to_cholesky():
    """Q68: round-trip preserves Mahalanobis distance to within 1e-6."""
    from voxkit.core.migrations import migrate_0_9_to_0_10_cholesky

    rng = np.random.default_rng(42)
    D = 16
    # Build a well-conditioned symmetric PSD covariance.
    A = rng.standard_normal((D, D))
    cov = A @ A.T + np.eye(D)
    inv_cov = np.linalg.inv(cov)

    raw = {
        "manifest": {"voxkit_format_version": "0.9"},
        "mahalanobis_full_dim": {
            "pooled_inv_covariance_full_dim": inv_cov.tolist(),
            "class_centroids": np.zeros((4, D)).tolist(),
            "distance_thresholds": np.ones(4).tolist(),
        },
    }
    out = migrate_0_9_to_0_10_cholesky(raw)
    L = np.array(out["mahalanobis_full_dim"]["pooled_cov_cholesky_full_dim"])

    # Reconstruct distance using Cholesky and compare to inverse-form distance.
    x = rng.standard_normal(D)
    from scipy.linalg import solve_triangular
    y = solve_triangular(L, x, lower=True)
    d_chol = float(y @ y)
    d_inv = float(x @ inv_cov @ x)
    assert d_chol == pytest.approx(d_inv, rel=1e-6)


def test_T22_round_trip_synthetic_legacy_bundle(tmp_path: Path):
    """§7.11 release-gate-style migration round-trip."""
    from voxkit.core.session import save_session, load_session, build_legacy_v0_4_bundle

    legacy = build_legacy_v0_4_bundle()
    legacy_path = tmp_path / "legacy.voxkit"
    legacy_path.write_bytes(legacy)

    # Load (walks v0.4 → v0.11), save, reload.
    s1 = load_session(legacy_path)
    out_path = tmp_path / "round.voxkit"
    save_session(s1, out_path)
    s2 = load_session(out_path)

    assert s1 == s2


# ---------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------

def test_T23_save_stamps_v011_in_manifest(tmp_path: Path):
    import json
    import zipfile
    from voxkit.core.session import save_session, Session, TimeSignature

    s = Session(
        bpm=120.0,
        time_signature=TimeSignature(4, 4),
        bars=4,
        sample_rate=16_000,
        recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000, dtype=np.float32),
    )
    out = tmp_path / "session.voxkit"
    save_session(s, out)

    with zipfile.ZipFile(out) as z:
        manifest = json.loads(z.read("manifest.json"))
    assert manifest["voxkit_format_version"] == "0.11"


def test_T24_save_and_reload_preserves_event_data(tmp_path: Path):
    from voxkit.core.session import (
        save_session, load_session, Session, TimeSignature, Event,
    )
    s = Session(
        bpm=120.0,
        time_signature=TimeSignature(4, 4),
        bars=4,
        sample_rate=16_000,
        recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000, dtype=np.float32),
        events=[
            Event(t=0.25, class_id="kick", score=0.9),
            Event(t=0.50, class_id="snare", score=0.8),
        ],
    )
    out = tmp_path / "s.voxkit"
    save_session(s, out)
    s2 = load_session(out)
    assert s2.events == s.events


def test_T25_loading_v010_bundle_preserves_mahalanobis_distance(tmp_path: Path):
    """The Cholesky factor in a v0.10 bundle must produce the same distances
    as a freshly fit Mahalanobis on identical centroids/covariance."""
    from voxkit.core.session import load_session, build_v0_10_bundle_with_known_distances

    bundle, ref_distances, ref_inputs = build_v0_10_bundle_with_known_distances(seed=7)
    path = tmp_path / "v010.voxkit"
    path.write_bytes(bundle)
    s = load_session(path)

    from voxkit.classifier.mahalanobis import mahalanobis_sq_via_cholesky
    L = s.mahalanobis_full_dim.pooled_cov_cholesky
    centroid0 = s.mahalanobis_full_dim.class_centroids[0]
    for x, expected in zip(ref_inputs, ref_distances):
        d = mahalanobis_sq_via_cholesky(x, centroid0, L)
        assert d == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T26_forward_version_bundle_raises_clearly(tmp_path: Path):
    """Q78 amplified: a bundle stamped with a future version must NOT
    silently load. ForwardCompatVersionError gives the user a clear
    message ('this file is from a newer VoxKit; please update') rather
    than a half-broken Session."""
    import json
    import zipfile
    from voxkit.core.session import load_session
    from voxkit.core.manifest import ForwardCompatVersionError

    out = tmp_path / "future.voxkit"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("manifest.json", json.dumps({"voxkit_format_version": "0.99"}))
    with pytest.raises(ForwardCompatVersionError, match="0.99"):
        load_session(out)


def test_T27_format_version_must_be_string_on_write(tmp_path: Path):
    """Q78: the field is `voxkit_format_version: str`. A float (0.11)
    or int that round-trips through JSON would silently turn '0.10'
    bundles into something dispatchers can't match. Reject at write
    time so the corruption can't ship."""
    from voxkit.core.manifest import ProjectManifest
    with pytest.raises((TypeError, ValueError), match="voxkit_format_version"):
        ProjectManifest(voxkit_format_version=0.11)   # float, not "0.11"


def test_T28_save_is_atomic_on_failure(tmp_path: Path, monkeypatch):
    """If save_session() raises mid-write, the previous bundle on disk
    must remain intact. Implementation: write to a sibling temp file
    then os.replace() into place. The hobby user's single .voxkit file
    is sacred."""
    import numpy as np
    from voxkit.core.session import save_session, load_session, Session, TimeSignature, Event

    s_good = Session(
        bpm=120.0, time_signature=TimeSignature(4, 4), bars=4,
        sample_rate=16_000, recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000, dtype=np.float32),
        events=[Event(t=0.5, class_id="kick", score=0.9)],
    )
    out = tmp_path / "bundle.voxkit"
    save_session(s_good, out)
    good_bytes = out.read_bytes()

    # Force the next save to raise mid-write.
    s_bad = Session(
        bpm=140.0, time_signature=TimeSignature(4, 4), bars=4,
        sample_rate=16_000, recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000, dtype=np.float32),
    )

    def boom(*_a, **_k):
        raise RuntimeError("simulated mid-write failure")

    # Patch the inner write function to raise after temp-file creation.
    monkeypatch.setattr("voxkit.core.session._finalize_bundle_write", boom)
    with pytest.raises(RuntimeError):
        save_session(s_bad, out)

    # Original bundle is still intact and loadable.
    assert out.read_bytes() == good_bytes
    s_loaded = load_session(out)
    assert s_loaded.bpm == 120.0


# ---------------------------------------------------------------
# v0.12 panel additions (principal-engineer synthesis)
# ---------------------------------------------------------------

def test_T29_migrator_exception_surfaces_with_step_context():
    """v0.12: T18 tests a happy multi-step walk. A migrator that itself
    raises (e.g., a v0.7 → v0.8 migrator that hits a corrupt field)
    must surface the original exception PLUS context identifying which
    step failed. Today: undefined; the user sees a bare KeyError with
    no version anchor."""
    from voxkit.core.migrations import walk_migrations, MigrationStepFailed

    def boom(_d):
        raise KeyError("missing_field")

    table = {
        ("0.4", "0.5"): lambda d: d,
        ("0.5", "0.6"): boom,
        ("0.6", "0.11"): lambda d: d,
    }
    with pytest.raises(MigrationStepFailed) as exc:
        walk_migrations({}, from_version="0.4", to_version="0.11", table=table)
    assert "0.5" in str(exc.value) and "0.6" in str(exc.value), (
        "MigrationStepFailed must name the from/to of the failing step; "
        f"got: {exc.value}"
    )
    # The original exception is preserved as the cause.
    assert isinstance(exc.value.__cause__, KeyError)


def test_T30_forward_compat_uses_semver_not_lexical_compare():
    """v0.12: lexically '0.10' < '0.11' BUT also '0.10' < '0.9' — a
    string-sort comparison would raise ForwardCompatVersionError on a
    legitimate v0.9 LEGACY bundle. Pin semver-aware numeric compare
    with explicit cases that distinguish the two."""
    import json
    import zipfile
    from voxkit.core.session import load_session
    from voxkit.core.manifest import ForwardCompatVersionError

    def write_bundle(tmp_path, version):
        out = tmp_path / f"v{version}.voxkit"
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("manifest.json",
                       json.dumps({"voxkit_format_version": version}))
        return out

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path as _P
        td_path = _P(td)

        # Legacy bundle: must NOT raise ForwardCompatVersionError.
        # (May raise other migration-related errors since v0.9 needs
        # other fields, but NOT ForwardCompat.)
        legacy = write_bundle(td_path, "0.9")
        try:
            load_session(legacy)
        except ForwardCompatVersionError:
            pytest.fail("0.9 lexically < 0.11 only by string sort; "
                        "semver compare must classify it as legacy")
        except Exception:
            pass   # other migration errors are acceptable here

        # Future bundle: MUST raise ForwardCompat.
        future = write_bundle(td_path, "0.99")
        with pytest.raises(ForwardCompatVersionError):
            load_session(future)

        # Tricky case: 0.100 (semver: > 0.99) must also raise.
        tricky = write_bundle(td_path, "0.100")
        with pytest.raises(ForwardCompatVersionError):
            load_session(tricky)


def test_T31_save_cleans_up_temp_files(tmp_path: Path):
    """v0.12: an atomic-replace save typically writes to a sibling
    temp file then renames. After a successful save, no .tmp / .partial
    siblings should remain. Hobby users see lingering files and ask
    'is my project corrupt?'."""
    import numpy as np
    from voxkit.core.session import save_session, Session, TimeSignature

    s = Session(
        bpm=120.0, time_signature=TimeSignature(4, 4), bars=4,
        sample_rate=16_000, recording_sample_rate=48_000,
        recording_audio_api="WASAPI",
        audio=np.zeros(16_000, dtype=np.float32),
    )
    out = tmp_path / "session.voxkit"
    save_session(s, out)

    siblings = [p for p in tmp_path.iterdir() if p != out]
    assert siblings == [], (
        f"save left {len(siblings)} temp file(s) on disk: "
        f"{[p.name for p in siblings]}"
    )
