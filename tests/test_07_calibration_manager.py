# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 7: Calibration manager.

Drives implementation of `voxkit.classifier.calibration_manager`.

Spec refs: §11 Component 7; §5.6 (calibration weighting from Q42/Q65),
Q71 (self-test overfit guard integration), Q61 (local-file diagnostics),
Q81 (CalibrationRejected wording propagation).

The CalibrationManager is the orchestration layer between the user
("record me 3 of each") and the Classifier's fit_with_calibration.
It owns the recording flow, sample-quality checks, retry semantics,
the CommitHandle returned from fit_with_calibration, and the diagnostic
events emitted on success/rejection.

============================================================
TEST LIST (implement strictly in order)
============================================================

CalibrationSession lifecycle
  T01  start_session() returns a session with all classes "incomplete"
  T02  Newly started session has no recordings
  T03  required_per_class defaults to 3 samples per trained class

Recording samples into a session
  T04  add_sample(class_id, embedding) appends to that class's bucket
  T05  add_sample with unknown class_id raises
  T06  Adding samples does not mutate other classes' buckets
  T07  is_complete() == True only when every class has >= required count
  T08  is_complete() == False when any class is short

Quality checks per sample
  T09  add_sample rejects a non-finite (NaN/Inf) embedding with a clear error
  T10  add_sample rejects an embedding of wrong dimensionality
  T11  add_sample warns (does not reject) when sample is a near-duplicate
       of an existing sample (cosine sim > 0.999)

Commit flow with the Classifier
  T12  commit() refuses if session not complete (raises IncompleteCalibration)
  T13  commit() calls Classifier.fit_with_calibration with collected samples
  T14  commit() returns a CommitHandle with classifier reference + diagnostics
  T15  commit() uses calibration_weight from session config

  -- TIDY FIRST before T16: extract `_collect_samples_to_arrays(session)`
     into a pure helper. Used by both commit() and the test fixtures.
     Structural-only commit; tests stay green.

Self-test overfit guard integration (Q71)
  T16  When fit_with_calibration raises CalibrationRejected, commit()
       restores the previous classifier state
  T17  Rejection emits a "calibration_overfit_guard_triggered" diagnostic
  T18  Rejection raises CalibrationRejected to the caller (UI layer)
  T19  After rejection, the session can be modified and re-committed

Successful commit
  T20  Successful commit emits a "calibration_committed" diagnostic
  T21  Successful commit advances the session state to "committed"
  T22  A committed session cannot have samples added

Diagnostic event format (Q61)
  T23  Diagnostic events are JSON-serializable dicts
  T24  Events have ts (ISO timestamp), event, details fields
  T25  Diagnostic sink writes events to the local file (Q61)
  T26  Diagnostic file path is under user profile directory

Rollback semantics
  T27  After rejection, classifier predicts identically to its pre-fit state
  T28  After rejection, the failed sample bundle is preserved on the
       session for the user's next attempt

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

Snapshot/restore must be load-bearing (Sam, Alex, Priya, Casey, Riley,
Lin, Marco: 7/9 — strong)
  T29  T16/T27 only check that .restore() WAS CALLED on the mock with
       the snapshot dict. They do not verify that restore actually
       returns the classifier to the pre-fit state. T29 uses a real
       Classifier (not a mock) to assert: snapshot, mutate, restore →
       predictions equal pre-snapshot predictions on a held-out batch.
       Without this, a no-op restore() would pass the existing tests.

Sample integrity (Marco, Jordan, Alex, Sam, Casey, Lin: 6/9)
  T30  add_sample with audio that is silent (RMS below floor) is
       rejected as "this didn't capture anything"; protects the user
       from committing 3 silent kicks because their mic was muted.
  T31  add_sample with clipped audio (peak ≥ 1.0 for > 0.1% of samples)
       emits a warning but accepts; users still need to commit. The UI
       layer surfaces the warning.

Diagnostic durability (Riley, Dana, Alex, Sam, Casey, Jordan: 6/9)
  T32  Diagnostic file write failure (disk full / permission denied)
       does NOT crash commit(); calibration succeeds; the failure is
       logged to stderr at WARNING level. Telemetry is supplementary
       and must never block primary functionality.
  T33  Diagnostic events are append-only (existing log preserved across
       process restarts). T25 currently writes a single event to a
       fresh file; T33 writes, closes, reopens, writes again, and
       asserts both events are present in order.

============================================================
v0.12 PANEL ADDITIONS (principal-engineer + Casey/Riley/Marco synthesis;
those reviewers rate-limited — synthesis only)
============================================================

User-facing failure modes (STRONG)
  T34  Whole-session-silent commit rejected. v0.11 T30 covers ONE
       silent sample; the more dangerous case is all 12 samples silent
       (user flipped the wrong mic switch and didn't notice). The
       commit() must surface AllSamplesSilent before the LR fit even
       runs — the fit would either fail with a degenerate covariance
       or produce garbage thresholds that Q71's overfit guard might
       not catch.

Diagnostic durability symmetry (STRONG — T32 only covers the rejection
path; diagnostic failure on the SUCCESS path is a different code branch
and a different blast radius)
  T35  Successful commit + diagnostic-write failure: the calibration
       still committed (classifier state advanced) AND the user's
       calibration is not lost. Mirror of T32 but with the success
       path; a different code branch with different blast radius.

Removals / softening
  T11  REWRITE: near-duplicate WARN at cosine > 0.999 will fire on
       legitimately-similar consecutive samples (two crisp kicks from
       the same person, captured back-to-back, ARE near-duplicates in
       embedding space). v0.12 (Casey/Marco): downgrade from
       UserWarning to a debug-log entry and a session-attribute count
       the UI can choose to surface OR ignore. Test asserts the
       behavior the team chooses; v0.12 picks "log only", T11 updated
       accordingly. The user-facing warning is too noisy for the
       intended audience.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  Calibration sample audio (not just embedding) preserved on the
      session for re-extraction with a different substrate. [Marco,
      Sam: 2/9 — defer to v1.1; substrate is fixed at fit time per Q33.]
OQ-2  Cross-session calibration carry-over (load yesterday's calibration
      into today's session). [Jordan: 1/9 — defer; session-scoped per
      v0.10 design.]
OQ-3  Diagnostic event PII review. [Dana: 2/9 — defer; events are
      numeric metrics + class IDs, not user content.]
OQ-4  v0.12: snapshot/restore idempotency (calling restore() twice
      doesn't double-restore). T29 verifies restore actually works
      once; the idempotency case is plausible (UI double-fires the
      cancel button) but low-impact. Defer.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_classifier_stub(embedding_dim=32):
    """A Classifier stub that records its fit_with_calibration calls."""
    clf = MagicMock()
    clf.embedding_dim = embedding_dim
    clf.taxonomy = MagicMock()
    clf.taxonomy.classes = ("kick", "snare", "closed_hat", "open_hat")
    clf.taxonomy.unknown_class_id = "unknown"
    clf.fit_with_calibration = MagicMock()
    clf.snapshot = MagicMock(return_value={"snapshot": True})
    clf.restore = MagicMock()
    return clf


# ---------------------------------------------------------------
# CalibrationSession lifecycle
# ---------------------------------------------------------------

def test_T01_start_session_marks_all_classes_incomplete():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    for cls in session.classes:
        assert session.count_for(cls) == 0


def test_T02_new_session_has_no_recordings():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    assert sum(session.count_for(c) for c in session.classes) == 0


def test_T03_required_per_class_defaults_to_3():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    assert session.required_per_class == 3


# ---------------------------------------------------------------
# Recording samples into a session
# ---------------------------------------------------------------

def test_T04_add_sample_appends_to_class_bucket():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    session.add_sample("kick", np.zeros(32))
    assert session.count_for("kick") == 1


def test_T05_add_sample_unknown_class_raises():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    with pytest.raises(ValueError, match="class"):
        session.add_sample("not_a_class", np.zeros(32))


def test_T06_add_sample_does_not_affect_other_buckets():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    session.add_sample("kick", np.zeros(32))
    assert session.count_for("snare") == 0
    assert session.count_for("closed_hat") == 0


def test_T07_is_complete_true_when_all_classes_have_required():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    for cls in session.classes:
        for _ in range(session.required_per_class):
            session.add_sample(cls, np.zeros(32))
    assert session.is_complete()


def test_T08_is_complete_false_when_any_class_short():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    for cls in ("kick", "snare", "closed_hat"):  # open_hat missing
        for _ in range(session.required_per_class):
            session.add_sample(cls, np.zeros(32))
    assert not session.is_complete()


# ---------------------------------------------------------------
# Quality checks per sample
# ---------------------------------------------------------------

def test_T09_nonfinite_embedding_rejected():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    bad = np.full(32, np.nan)
    with pytest.raises(ValueError, match="finite"):
        session.add_sample("kick", bad)


def test_T10_wrong_dim_embedding_rejected():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    with pytest.raises(ValueError, match="dim"):
        session.add_sample("kick", np.zeros(31))   # one short


def test_T11_near_duplicate_sample_logged_not_warned():
    """v0.12 (Casey/Marco) REWRITTEN: the v0.11 form raised a UserWarning
    on cosine > 0.999, which fires on legitimate consecutive samples
    (two crisp kicks from the same person ARE near-duplicate embeddings).
    The warning was too noisy for users doing the right thing.

    v0.12 contract: the duplicate is silently accepted, but the session
    increments a near_duplicate_count attribute the UI may choose to
    surface in a single end-of-session summary (less disruptive than
    a warning per sample)."""
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    rng = np.random.default_rng(11)
    v = rng.standard_normal(32)
    session.add_sample("kick", v)
    session.add_sample("kick", v + 1e-6)   # near-duplicate, no warning
    assert session.count_for("kick") == 2
    assert session.near_duplicate_count == 1


# ---------------------------------------------------------------
# Commit flow with the Classifier
# ---------------------------------------------------------------

def _completed_session(mgr):
    session = mgr.start_session()
    rng = np.random.default_rng(0)
    for cls in session.classes:
        for _ in range(session.required_per_class):
            session.add_sample(cls, rng.standard_normal(32))
    return session


def test_T12_commit_refuses_incomplete_session():
    from voxkit.classifier.calibration_manager import (
        CalibrationManager, IncompleteCalibration,
    )
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    session.add_sample("kick", np.zeros(32))   # only one sample
    with pytest.raises(IncompleteCalibration):
        mgr.commit(session)


def test_T13_commit_calls_classifier_with_collected_samples():
    from voxkit.classifier.calibration_manager import CalibrationManager
    clf = _make_classifier_stub()
    mgr = CalibrationManager(classifier=clf)
    session = _completed_session(mgr)
    mgr.commit(session)
    assert clf.fit_with_calibration.called
    kwargs = clf.fit_with_calibration.call_args.kwargs
    # 4 classes × 3 samples each
    assert kwargs["calibration_embeddings"].shape == (12, 32)
    assert len(kwargs["calibration_labels"]) == 12


def test_T14_commit_returns_handle_with_classifier_and_diagnostics():
    from voxkit.classifier.calibration_manager import CalibrationManager
    clf = _make_classifier_stub()
    mgr = CalibrationManager(classifier=clf)
    handle = mgr.commit(_completed_session(mgr))
    assert handle.classifier is clf
    assert isinstance(handle.diagnostics, dict)


def test_T15_commit_uses_session_calibration_weight():
    from voxkit.classifier.calibration_manager import CalibrationManager
    clf = _make_classifier_stub()
    mgr = CalibrationManager(classifier=clf, calibration_weight=7.5)
    mgr.commit(_completed_session(mgr))
    kwargs = clf.fit_with_calibration.call_args.kwargs
    assert kwargs["calibration_weight"] == 7.5


# ----- TIDY FIRST checkpoint -----
# Extract `_collect_samples_to_arrays(session)` to a pure helper so the
# session-to-arrays serialization is testable in isolation. No behavior
# change.


# ---------------------------------------------------------------
# Self-test overfit guard integration (Q71)
# ---------------------------------------------------------------

def test_T16_rejection_restores_previous_classifier_state():
    from voxkit.classifier.calibration_manager import CalibrationManager
    from voxkit.classifier.classifier import CalibrationRejected
    clf = _make_classifier_stub()
    clf.fit_with_calibration.side_effect = CalibrationRejected(
        message="x", diagnostics={"delta": -0.05},
    )
    mgr = CalibrationManager(classifier=clf)
    with pytest.raises(CalibrationRejected):
        mgr.commit(_completed_session(mgr))
    clf.restore.assert_called_once()


def test_T17_rejection_emits_overfit_guard_diagnostic():
    from voxkit.classifier.calibration_manager import CalibrationManager
    from voxkit.classifier.classifier import CalibrationRejected
    sink = MagicMock()
    clf = _make_classifier_stub()
    clf.fit_with_calibration.side_effect = CalibrationRejected(
        message="x", diagnostics={"f1_calibrated": 0.7, "f1_baseline": 0.85, "delta": -0.15},
    )
    mgr = CalibrationManager(classifier=clf, telemetry=sink)
    with pytest.raises(CalibrationRejected):
        mgr.commit(_completed_session(mgr))
    events = [c.args[0] for c in sink.emit.call_args_list]
    assert any(e["event"] == "calibration_overfit_guard_triggered" for e in events)


def test_T18_rejection_raised_to_caller():
    from voxkit.classifier.calibration_manager import CalibrationManager
    from voxkit.classifier.classifier import CalibrationRejected
    clf = _make_classifier_stub()
    clf.fit_with_calibration.side_effect = CalibrationRejected(
        message="x", diagnostics={"delta": -0.05},
    )
    mgr = CalibrationManager(classifier=clf)
    with pytest.raises(CalibrationRejected):
        mgr.commit(_completed_session(mgr))


def test_T19_after_rejection_session_can_be_modified_and_recommitted():
    from voxkit.classifier.calibration_manager import CalibrationManager
    from voxkit.classifier.classifier import CalibrationRejected
    clf = _make_classifier_stub()
    # First call rejects, second succeeds.
    clf.fit_with_calibration.side_effect = [
        CalibrationRejected(message="x", diagnostics={"delta": -0.05}),
        None,
    ]
    mgr = CalibrationManager(classifier=clf)
    session = _completed_session(mgr)
    with pytest.raises(CalibrationRejected):
        mgr.commit(session)
    # Add an extra sample and re-commit.
    session.add_sample("kick", np.zeros(32))
    handle = mgr.commit(session)
    assert handle is not None


# ---------------------------------------------------------------
# Successful commit
# ---------------------------------------------------------------

def test_T20_successful_commit_emits_committed_diagnostic():
    from voxkit.classifier.calibration_manager import CalibrationManager
    sink = MagicMock()
    mgr = CalibrationManager(classifier=_make_classifier_stub(), telemetry=sink)
    mgr.commit(_completed_session(mgr))
    events = [c.args[0] for c in sink.emit.call_args_list]
    assert any(e["event"] == "calibration_committed" for e in events)


def test_T21_successful_commit_marks_session_committed():
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = _completed_session(mgr)
    mgr.commit(session)
    assert session.state == "committed"


def test_T22_committed_session_rejects_new_samples():
    from voxkit.classifier.calibration_manager import (
        CalibrationManager, SessionAlreadyCommitted,
    )
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = _completed_session(mgr)
    mgr.commit(session)
    with pytest.raises(SessionAlreadyCommitted):
        session.add_sample("kick", np.zeros(32))


# ---------------------------------------------------------------
# Diagnostic event format (Q61)
# ---------------------------------------------------------------

def test_T23_diagnostic_events_are_json_serializable():
    from voxkit.telemetry.local_sink import LocalDiagnosticSink
    sink = LocalDiagnosticSink(path=None)   # in-memory mode
    sink.emit({"ts": "2025-01-01T00:00:00Z", "event": "x", "details": {"a": 1}})
    for evt in sink.events:
        json.dumps(evt)   # must not raise


def test_T24_events_have_required_fields():
    from voxkit.telemetry.local_sink import build_event
    e = build_event(event="calibration_committed", details={"weight": 5.0})
    assert "ts" in e and "event" in e and "details" in e


def test_T25_diagnostic_sink_writes_to_local_file(tmp_path: Path):
    from voxkit.telemetry.local_sink import LocalDiagnosticSink
    p = tmp_path / "diag.jsonl"
    sink = LocalDiagnosticSink(path=p)
    sink.emit({"ts": "2025-01-01T00:00:00Z", "event": "x", "details": {}})
    sink.flush()
    contents = p.read_text().splitlines()
    assert len(contents) == 1
    assert json.loads(contents[0])["event"] == "x"


def test_T26_default_diagnostic_path_under_user_profile():
    from voxkit.telemetry.local_sink import default_diagnostic_path
    p = default_diagnostic_path()
    assert ".voxkit" in p.parts
    assert "diagnostics" in p.parts


# ---------------------------------------------------------------
# Rollback semantics
# ---------------------------------------------------------------

def test_T27_after_rejection_classifier_predicts_identically_to_prefit():
    """Q71 + Q81: rejected calibration is fully rolled back. The classifier's
    .restore() restores its snapshot; therefore predictions on identical
    inputs match the pre-commit state bit-for-bit."""
    from voxkit.classifier.calibration_manager import CalibrationManager
    from voxkit.classifier.classifier import CalibrationRejected
    clf = _make_classifier_stub()
    clf.fit_with_calibration.side_effect = CalibrationRejected(
        message="x", diagnostics={"delta": -0.05},
    )
    mgr = CalibrationManager(classifier=clf)
    with pytest.raises(CalibrationRejected):
        mgr.commit(_completed_session(mgr))
    # Snapshot was taken; restore was called with that snapshot.
    args, _ = clf.restore.call_args
    assert args[0] == {"snapshot": True}


def test_T28_failed_sample_bundle_preserved_on_session():
    from voxkit.classifier.calibration_manager import CalibrationManager
    from voxkit.classifier.classifier import CalibrationRejected
    clf = _make_classifier_stub()
    clf.fit_with_calibration.side_effect = CalibrationRejected(
        message="x", diagnostics={"delta": -0.05},
    )
    mgr = CalibrationManager(classifier=clf)
    session = _completed_session(mgr)
    with pytest.raises(CalibrationRejected):
        mgr.commit(session)
    assert session.state == "rejected"
    assert session.count_for("kick") == 3   # samples preserved


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T29_real_classifier_restore_recovers_predictions_exactly():
    """T16/T27 only check that restore() was CALLED. They don't verify
    restore actually works. Use a real Classifier with snapshot/restore
    and assert predictions on a held-out batch match the pre-fit state
    after rejection. A no-op restore() passes T16/T27 but fails this."""
    from voxkit.classifier.calibration_manager import CalibrationManager
    from voxkit.classifier.classifier import Classifier, CalibrationRejected

    rng = np.random.default_rng(29)
    D = 32
    centroids = rng.standard_normal((4, D)) * 5.0
    classes = ("kick", "snare", "closed_hat", "open_hat")
    X_train, y_train, subjects = [], [], []
    for c, name in enumerate(classes):
        Xc = centroids[c] + rng.standard_normal((20, D))
        X_train.append(Xc)
        y_train.extend([name] * 20)
        for i in range(20):
            subjects.append(f"subj_{i % 4}")
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    subjects = np.array(subjects)

    clf = Classifier.untrained(taxonomy=None, embedding_dim=D)
    clf.fit(avp_embeddings=X_train, avp_labels=y_train, avp_subjects=subjects)

    held_out = X_train[::5]
    preds_before = clf.predict(held_out)

    # Force a calibration rejection mid-flight via the overfit guard.
    bad_cal_X = rng.standard_normal((12, D)) * 100.0   # outliers
    bad_cal_y = np.array(list(classes) * 3)

    mgr = CalibrationManager(classifier=clf)
    session = mgr.start_session()
    for c in classes:
        for k in range(3):
            session.add_sample(c, bad_cal_X[list(classes).index(c) * 3 + k])

    from unittest.mock import patch as _patch
    with _patch("voxkit.classifier.classifier.self_test_overfit_guard",
                return_value=(False, {"f1_calibrated": 0.5,
                                       "f1_baseline": 0.85, "delta": -0.35})):
        with pytest.raises(CalibrationRejected):
            mgr.commit(session)

    preds_after = clf.predict(held_out)
    assert preds_after == preds_before, (
        "restore() did not return classifier to pre-fit predictions; "
        "snapshot/restore is not load-bearing"
    )


def test_T30_silent_audio_sample_rejected():
    """A user with a muted microphone would otherwise commit 3 silent
    'kicks' per class, then see horrible predictions and not understand
    why. Reject the silent sample with a clear message."""
    from voxkit.classifier.calibration_manager import CalibrationManager, SilentSample
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    # Silent embedding (zero vector) is the embedding-space proxy for
    # silent audio. Implementation may also receive raw audio and check
    # RMS upstream; either layer should reject.
    with pytest.raises(SilentSample):
        session.add_sample("kick", np.zeros(32, dtype=np.float32),
                           source_audio_rms=1e-6)


def test_T31_clipped_audio_sample_warns_but_accepted():
    """Clipped audio is suboptimal but recoverable; the user shouldn't
    be blocked. Surface a warning so the UI can show a hint."""
    from voxkit.classifier.calibration_manager import CalibrationManager
    mgr = CalibrationManager(classifier=_make_classifier_stub())
    session = mgr.start_session()
    rng = np.random.default_rng(31)
    emb = rng.standard_normal(32).astype(np.float32)
    with pytest.warns(UserWarning, match="clip"):
        session.add_sample("kick", emb,
                           source_audio_clipped_fraction=0.005)   # 0.5% clipped
    assert session.count_for("kick") == 1


def test_T32_diagnostic_write_failure_does_not_block_commit(tmp_path, capsys):
    """Telemetry is supplementary. A disk-full or permission-denied
    error on the diagnostic sink must NOT cause commit() to fail; the
    calibration that the user just spent two minutes recording succeeds."""
    from voxkit.classifier.calibration_manager import CalibrationManager
    sink = MagicMock()
    sink.emit.side_effect = OSError("disk full")
    mgr = CalibrationManager(classifier=_make_classifier_stub(), telemetry=sink)
    handle = mgr.commit(_completed_session(mgr))
    assert handle is not None
    err = capsys.readouterr().err
    assert "diagnostic" in err.lower() or "telemetry" in err.lower()


def test_T33_diagnostic_log_is_append_only_across_sessions(tmp_path: Path):
    """Diagnostic file must survive process restarts. Two writes
    separated by a sink close → reopen must both appear in order."""
    from voxkit.telemetry.local_sink import LocalDiagnosticSink
    p = tmp_path / "diag.jsonl"

    s1 = LocalDiagnosticSink(path=p)
    s1.emit({"ts": "2025-01-01T00:00:00Z", "event": "first", "details": {}})
    s1.flush()
    s1.close()

    s2 = LocalDiagnosticSink(path=p)
    s2.emit({"ts": "2025-01-01T00:00:01Z", "event": "second", "details": {}})
    s2.flush()
    s2.close()

    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "first"
    assert json.loads(lines[1])["event"] == "second"


# ---------------------------------------------------------------
# v0.12 panel additions (principal-engineer + Casey/Marco synthesis)
# ---------------------------------------------------------------

def test_T34_all_silent_session_commit_rejected_before_fit():
    """v0.12: T30 catches one silent SAMPLE; the more dangerous case is
    every sample silent (user flipped the wrong mic switch and never
    noticed). The fit either degenerates or produces garbage thresholds.
    Reject the commit BEFORE fit so the user gets a clear error rather
    than a 'classifier rejected' diagnostic that points at the wrong
    layer."""
    from voxkit.classifier.calibration_manager import (
        CalibrationManager, AllSamplesSilent,
    )
    clf = _make_classifier_stub()
    mgr = CalibrationManager(classifier=clf)
    session = mgr.start_session()
    # Fill with silent embeddings (zero vectors, plus the source-audio
    # RMS attribute the v0.11 T30 contract uses).
    for cls in session.classes:
        for _ in range(session.required_per_class):
            session.add_sample(cls, np.zeros(32, dtype=np.float32),
                               source_audio_rms=1e-6,
                               skip_silence_check=True)
    with pytest.raises(AllSamplesSilent):
        mgr.commit(session)
    # Fit must NOT have been called — the fail-fast check is upstream.
    assert not clf.fit_with_calibration.called


def test_T35_diagnostic_failure_on_success_path_does_not_lose_calibration(
    tmp_path, capsys,
):
    """v0.12: T32 covers diagnostic failure on the REJECTION path. The
    success path is a different code branch — and arguably more
    important: a user who just spent 2 minutes recording a working
    calibration must NOT lose it because the diagnostic file is full.

    Note this overlaps somewhat with T32 but exercises the
    'calibration_committed' emit path specifically rather than the
    'overfit_guard_triggered' emit path."""
    from voxkit.classifier.calibration_manager import CalibrationManager
    sink = MagicMock()
    sink.emit.side_effect = OSError("disk full")
    clf = _make_classifier_stub()
    mgr = CalibrationManager(classifier=clf, telemetry=sink)
    handle = mgr.commit(_completed_session(mgr))

    assert handle is not None
    assert handle.classifier is clf
    # The fit DID run; the committed state is real.
    assert clf.fit_with_calibration.called
    err = capsys.readouterr().err
    assert "diagnostic" in err.lower() or "telemetry" in err.lower()
