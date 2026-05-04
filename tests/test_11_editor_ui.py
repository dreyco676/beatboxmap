# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 11: Editor UI.

Drives implementation of `voxkit.ui.editor`, `voxkit.ui.inference_worker`,
`voxkit.ui.dialogs`, and `voxkit.ui.banners`.

Spec refs: §11 Component 11; Q54 (first-run guided tour),
Q73 (recording-session inference UX with three-phase progress + cancel),
Q76 (InferenceWorker on dedicated thread; main thread is Qt event loop only),
Q79 (click-bleed quality indicator metric),
Q81 (CalibrationRejected dialog wording),
v0.10 item 17 (migration banner persistent until calibration runs).

UI tests are functional-state tests, not pixel tests. We assert the
state machines (banner state, dialog state, worker state) and the
contracts between widgets — not the exact pixel coordinates of any
button. The full Qt event loop is mocked or stubbed where reasonable.

============================================================
TEST LIST (implement strictly in order)
============================================================

InferenceWorker contract (Q76)
  T01  Worker is a separate thread (not the test/main thread)
  T02  Worker emits phase_changed signal for "onset", "embedding", "classify"
  T03  Worker emits progress signal with values in [0.0, 1.0]
  T04  Worker emits completed signal with list of events on success
  T05  Worker emits failed signal with an error message on exception
  T06  Worker.cancel() sets the cancel flag
  T07  cancel() during onset phase exits before embedding phase begins
  T08  cancel() during embedding phase exits within ~1 onset's latency
  T09  After cancel, worker emits cancelled signal (not completed/failed)

  -- TIDY FIRST before T10: split `voxkit.ui.inference_worker` into
     a Qt-coupled class and a pure-Python pipeline driver. The pure
     driver is what T10–T15 exercise; Qt signals are the adapter.

Pipeline driver (Q76 pure logic, no Qt)
  T10  Pipeline driver runs all three phases in order on input audio
  T11  Pipeline driver returns events with class_id and timestamp
  T12  Pipeline driver respects cancel flag between phases
  T13  Pipeline driver respects cancel flag inside embedding loop (per onset)
  T14  Pipeline driver does not block on Qt anything (importable without Qt)
  T15  Pipeline driver preserves audio buffer in returned context on cancel

Recording-session progress dialog (Q73)
  T16  Progress dialog shows three phase labels in correct order
  T17  Progress bar advances as worker emits progress signals
  T18  Cancel button calls worker.cancel() exactly once
  T19  Cancel + cancelled signal closes dialog
  T20  Successful completion closes dialog and lands user in editor
  T21  Failed completion shows error dialog (not silent)

Bleed-quality banner (Q79)
  T22  Banner shows when attenuation_db < 10
  T23  Banner shows yellow color when 10 <= attenuation_db < 20
  T24  Banner hidden when attenuation_db >= 20
  T25  Banner is suppressed when bleed_gate_overridden=True
  T26  Banner displays the numeric attenuation_db alongside the bar

PCA-Mahalanobis migration banner (v0.10 item 17)
  T27  Migration banner is shown when migration is required
  T28  Banner has NO "Remind me later" link (resolution from v0.10 item 17)
  T29  Banner persists until calibration completes
  T30  Banner is dismissed automatically when calibration commits
  T31  Banner state survives a session save/load (persistence flag)

CalibrationRejected dialog (Q81)
  T32  Dialog text matches Q81 wording exactly
  T33  Dialog has a "Try again" and a "Continue with previous" button
  T34  "Try again" returns to calibration recording flow
  T35  "Continue with previous" closes dialog without state change

First-run guided tour (Q54)
  T36  Tour fires on the first 'unknown' event encountered
  T37  Tour fires only once per user (persisted state)
  T38  Tour does not fire if there are no unknowns in the first session

Editor lane configuration (Q66)
  T39  Default 4-class taxonomy → 5 lanes (4 trained + unknown)
  T40  5-class custom taxonomy → 6 lanes
  T41  Lane labels match TaxonomyConfig.classes order

Main thread contract (Q76)
  T42  Main thread does not block during inference (event loop runs)
  T43  No long-running operations on the main thread (sentinel test)

============================================================
v0.11 PANEL ADDITIONS (≥6/9 consensus)
============================================================

T42 honesty (Sam, Lin, Alex, Casey, Riley, Jordan, Marco: 7/9 — strong)
  T44  T42 currently asserts that 1000 trivial additions complete in
       < 0.1s while a worker runs. This proves nothing about the Qt
       event loop — busy CPython math doesn't yield to anything. The
       real Q76 contract is "the main thread can dispatch a Qt signal
       within X ms while inference is in progress." T44 starts the
       worker, schedules a QTimer-equivalent callback, and asserts the
       callback fires within 50 ms. Mocked Qt.

CalibrationRejected dialog (Q81) — programmatic verification
(Priya, Jordan, Alex, Casey, Riley, Sam: 6/9)
  T45  Q81 wording must include the diagnostic delta (-0.05 macro-F1)
       in the diagnostic dict surfaced to the diagnostic file. T32
       checks message text only; T45 asserts diagnostics propagate
       from CalibrationRejected → CalibrationRejectedDialog →
       diagnostic sink without loss.

Bleed banner numerical-display correctness (Lin, Jordan, Marco, Alex,
Casey, Riley: 6/9)
  T46  T26 asserts "12.3" or "12" appears in banner text. The user-
       visible string must include a unit ("dB") otherwise the number
       is ambiguous. T46 enforces the unit.

InferenceWorker thread cleanup (Sam, Alex, Casey, Riley, Lin, Marco: 6/9)
  T47  Worker thread is joined within 1 second after wait_for_completion()
       returns. A leaked thread holds the model in memory and prevents
       clean shutdown — particularly visible on test runs that warn
       about non-daemon threads at exit.

============================================================
v0.12 PANEL ADDITIONS (principal-engineer + Sam synthesis;
Sam-equivalent reviewer rate-limited)
============================================================

Test-foundation correctness (STRONG — every InferenceWorker test relies
on wait_for_completion(timeout=5.0). If wait_for_completion is buggy
or doesn't actually block, T01-T09 pass spuriously)
  T48  wait_for_completion(timeout=0.001) returns False on a worker
       that hasn't yet completed (i.e., the timeout actually times
       out). Without this test, every existing InferenceWorker test's
       wait_for_completion(timeout=5.0) is unverified.

Lifecycle parity with Recorder (STRONG — Recorder T27 covers open-
twice; InferenceWorker has no equivalent and starting a worker twice
is a plausible mistake from a UI that forgot to disable the button)
  T49  InferenceWorker.start() called twice without intervening
       wait_for_completion raises WorkerAlreadyStarted. Mirror of
       Recorder T27.

Tightening of v0.11 panel additions
  T47  TIGHTEN: replace name-prefix "Inference" matching with a real
       reference to the worker's underlying thread (worker.thread or
       similar). The string-prefix test fails silently if the thread
       is named differently in subclasses.

============================================================
WEAK CONSENSUS / OPEN QUESTIONS (carry from v0.11 + v0.12)
============================================================

OQ-1  Re-run inference from editor without re-record (spec §10 item 23,
      already tracked).
OQ-2  Editor undo/redo for manual event reclassification. [Jordan: 1/9
      — defer to v1.1.]
OQ-3  Accessibility: keyboard navigation of lanes, screen-reader labels.
      [Riley, Dana: 2/9 — RECORDED. v1.0 hobby-scope acceptable but
      reviewer wants to flag for v1.1.]
OQ-4  T14 (pipeline does not import Qt) uses sys.modules manipulation
      that's brittle to test-runner state. Replace with subprocess
      isolation in v1.1. [Sam: 1/9 — defer.]
OQ-5  v0.12: T43 (import editor in < 0.5s) is flaky on cold CI Python
      with PySide/PyQt. Defer; downgrade to a soft assertion or move
      to a manual benchmark.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

@pytest.fixture
def fake_model():
    m = MagicMock()
    m.taxonomy = MagicMock()
    m.taxonomy.classes = ("kick", "snare", "closed_hat", "open_hat")
    m.taxonomy.unknown_class_id = "unknown"
    return m


@pytest.fixture
def short_audio():
    return np.zeros(16_000 * 2, dtype=np.float32)


# ---------------------------------------------------------------
# InferenceWorker contract (Q76)
# ---------------------------------------------------------------

def test_T01_worker_runs_on_separate_thread(fake_model, short_audio):
    from voxkit.ui.inference_worker import InferenceWorker
    main_thread_id = threading.get_ident()
    captured_thread_id = {}

    def on_phase(phase):
        captured_thread_id[phase] = threading.get_ident()

    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.phase_changed.connect(on_phase)
    worker.start()
    worker.wait_for_completion(timeout=5.0)
    assert any(tid != main_thread_id for tid in captured_thread_id.values())


def test_T02_worker_emits_three_phases_in_order(fake_model, short_audio):
    from voxkit.ui.inference_worker import InferenceWorker
    phases = []
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.phase_changed.connect(phases.append)
    worker.start()
    worker.wait_for_completion(timeout=5.0)
    assert phases == ["onset", "embedding", "classify"]


def test_T03_progress_values_in_unit_range(fake_model, short_audio):
    from voxkit.ui.inference_worker import InferenceWorker
    progresses = []
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.progress.connect(progresses.append)
    worker.start()
    worker.wait_for_completion(timeout=5.0)
    assert all(0.0 <= p <= 1.0 for p in progresses)


def test_T04_completed_signal_carries_events_on_success(fake_model, short_audio):
    from voxkit.ui.inference_worker import InferenceWorker
    captured = {}
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.completed.connect(lambda evts: captured.setdefault("evts", evts))
    worker.start()
    worker.wait_for_completion(timeout=5.0)
    assert "evts" in captured


def test_T05_failed_signal_emitted_on_exception(fake_model, short_audio):
    from voxkit.ui.inference_worker import InferenceWorker
    fake_model.predict = MagicMock(side_effect=RuntimeError("boom"))
    captured = {}
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.failed.connect(lambda msg: captured.setdefault("err", msg))
    worker.start()
    worker.wait_for_completion(timeout=5.0)
    assert "boom" in captured.get("err", "")


def test_T06_cancel_sets_flag(fake_model, short_audio):
    from voxkit.ui.inference_worker import InferenceWorker
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.cancel()
    assert worker._cancel_flag.is_set()


def test_T07_cancel_during_onset_skips_embedding_phase(fake_model, short_audio):
    """Drive the worker's pipeline directly so we control phase boundaries."""
    from voxkit.ui.inference_worker import InferenceWorker
    phases = []
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.phase_changed.connect(phases.append)

    def slow_onset(*_a, **_k):
        time.sleep(0.05)
        return [0.1, 0.2, 0.3]

    worker._detect_onsets = slow_onset
    worker.start()
    time.sleep(0.01)   # let onset phase start
    worker.cancel()
    worker.wait_for_completion(timeout=5.0)
    assert "embedding" not in phases


def test_T08_cancel_during_embedding_exits_within_one_onset_latency(fake_model, short_audio):
    """Spec: worst-case cancel latency = one embedding extraction (~50 ms)."""
    from voxkit.ui.inference_worker import InferenceWorker
    worker = InferenceWorker(audio=short_audio, model=fake_model)

    embedding_calls = []

    def slow_embed(*_a, **_k):
        embedding_calls.append(time.perf_counter())
        time.sleep(0.05)
        return np.zeros(2048)

    worker._embed_one = slow_embed
    worker.start()
    time.sleep(0.06)   # one embedding into the loop
    worker.cancel()
    t_cancel = time.perf_counter()
    worker.wait_for_completion(timeout=5.0)
    t_done = time.perf_counter()
    # Total time from cancel to completion must be at most ~one embedding.
    assert (t_done - t_cancel) < 0.2


def test_T09_cancel_emits_cancelled_signal(fake_model, short_audio):
    from voxkit.ui.inference_worker import InferenceWorker
    captured = {"cancelled": False}
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.cancelled.connect(lambda: captured.update(cancelled=True))
    worker._detect_onsets = lambda *_a, **_k: time.sleep(0.05) or []
    worker.start()
    time.sleep(0.01)
    worker.cancel()
    worker.wait_for_completion(timeout=5.0)
    assert captured["cancelled"]


# ----- TIDY FIRST checkpoint -----
# Split inference worker into:
#   - voxkit.ui.inference_worker.InferenceWorker (Qt-coupled adapter)
#   - voxkit.ui.inference_pipeline.run_pipeline (pure Python)
# T10–T15 exercise the pure pipeline; T01–T09 stay with the adapter.


# ---------------------------------------------------------------
# Pipeline driver (Q76 pure logic)
# ---------------------------------------------------------------

def test_T10_pipeline_runs_three_phases_in_order(fake_model, short_audio):
    from voxkit.ui.inference_pipeline import run_pipeline
    phases = []
    run_pipeline(
        audio=short_audio, model=fake_model,
        on_phase=phases.append,
    )
    assert phases == ["onset", "embedding", "classify"]


def test_T11_pipeline_returns_events_with_class_and_timestamp(fake_model, short_audio):
    from voxkit.ui.inference_pipeline import run_pipeline
    result = run_pipeline(audio=short_audio, model=fake_model)
    for e in result.events:
        assert hasattr(e, "t")
        assert hasattr(e, "class_id")


def test_T12_pipeline_respects_cancel_between_phases(fake_model, short_audio):
    from voxkit.ui.inference_pipeline import run_pipeline
    cancel_flag = threading.Event()
    cancel_flag.set()
    result = run_pipeline(audio=short_audio, model=fake_model, cancel_flag=cancel_flag)
    assert result.cancelled


def test_T13_pipeline_respects_cancel_inside_embedding_loop(fake_model, short_audio):
    from voxkit.ui.inference_pipeline import run_pipeline
    cancel_flag = threading.Event()

    def slow_embed(emb_input, **_):
        cancel_flag.set()   # set after first call
        return np.zeros(2048)

    fake_model.embed = slow_embed
    result = run_pipeline(audio=short_audio, model=fake_model, cancel_flag=cancel_flag)
    assert result.cancelled


def test_T14_pipeline_does_not_import_qt():
    """Q76: pure pipeline must be importable in a no-Qt environment."""
    import sys
    import importlib

    # Pretend Qt is not installed.
    saved = {k: v for k, v in sys.modules.items() if "PyQt" in k or "PySide" in k}
    for k in saved:
        sys.modules[k] = None
    try:
        importlib.import_module("voxkit.ui.inference_pipeline")
    finally:
        for k, v in saved.items():
            sys.modules[k] = v


def test_T15_cancelled_pipeline_preserves_audio_buffer(fake_model, short_audio):
    from voxkit.ui.inference_pipeline import run_pipeline
    cancel_flag = threading.Event()
    cancel_flag.set()
    result = run_pipeline(audio=short_audio, model=fake_model, cancel_flag=cancel_flag)
    np.testing.assert_array_equal(result.audio, short_audio)


# ---------------------------------------------------------------
# Recording-session progress dialog (Q73)
# ---------------------------------------------------------------

def test_T16_progress_dialog_shows_three_phase_labels():
    from voxkit.ui.dialogs import RecordingProgressDialog
    dlg = RecordingProgressDialog()
    assert dlg.phase_labels == ("Detecting onsets", "Extracting embeddings", "Classifying events")


def test_T17_progress_bar_advances_with_worker_signals():
    from voxkit.ui.dialogs import RecordingProgressDialog
    dlg = RecordingProgressDialog()
    dlg.on_phase("onset")
    dlg.on_progress(0.5)
    assert dlg.current_progress == pytest.approx(0.5)


def test_T18_cancel_button_calls_worker_cancel_once():
    from voxkit.ui.dialogs import RecordingProgressDialog
    worker = MagicMock()
    dlg = RecordingProgressDialog(worker=worker)
    dlg.click_cancel()
    dlg.click_cancel()   # idempotent
    assert worker.cancel.call_count == 1


def test_T19_cancelled_signal_closes_dialog():
    from voxkit.ui.dialogs import RecordingProgressDialog
    dlg = RecordingProgressDialog()
    dlg.on_cancelled()
    assert dlg.is_closed


def test_T20_completion_closes_dialog_and_returns_to_editor():
    from voxkit.ui.dialogs import RecordingProgressDialog
    dlg = RecordingProgressDialog()
    dlg.on_completed(events=[])
    assert dlg.is_closed
    assert dlg.completion_path == "editor"


def test_T21_failure_shows_error_dialog():
    from voxkit.ui.dialogs import RecordingProgressDialog
    dlg = RecordingProgressDialog()
    dlg.on_failed(message="Test error")
    assert dlg.error_shown
    assert "Test error" in dlg.error_text


# ---------------------------------------------------------------
# Bleed-quality banner (Q79)
# ---------------------------------------------------------------

def test_T22_banner_shown_below_10db():
    from voxkit.ui.banners import BleedQualityBanner
    b = BleedQualityBanner()
    b.update(attenuation_db=5.0, override=False)
    assert b.is_visible


def test_T23_banner_yellow_between_10_and_20():
    from voxkit.ui.banners import BleedQualityBanner
    b = BleedQualityBanner()
    b.update(attenuation_db=15.0, override=False)
    assert b.color == "yellow"


def test_T24_banner_hidden_above_20db():
    from voxkit.ui.banners import BleedQualityBanner
    b = BleedQualityBanner()
    b.update(attenuation_db=25.0, override=False)
    assert not b.is_visible


def test_T25_banner_suppressed_by_override():
    from voxkit.ui.banners import BleedQualityBanner
    b = BleedQualityBanner()
    b.update(attenuation_db=5.0, override=True)
    assert not b.is_visible


def test_T26_banner_displays_numeric_value():
    from voxkit.ui.banners import BleedQualityBanner
    b = BleedQualityBanner()
    b.update(attenuation_db=12.3, override=False)
    assert "12.3" in b.text or "12" in b.text


# ---------------------------------------------------------------
# PCA-Mahalanobis migration banner (v0.10 item 17)
# ---------------------------------------------------------------

def test_T27_migration_banner_shown_when_required():
    from voxkit.ui.banners import MigrationBanner
    b = MigrationBanner(migration_required=True)
    assert b.is_visible


def test_T28_migration_banner_has_no_remind_me_later_link():
    """v0.10 item 17 resolution: persistent until calibration runs."""
    from voxkit.ui.banners import MigrationBanner
    b = MigrationBanner(migration_required=True)
    actions = b.get_action_labels()
    assert all("remind" not in label.lower() for label in actions)
    assert all("later" not in label.lower() for label in actions)


def test_T29_migration_banner_persists_through_dismiss_attempts():
    from voxkit.ui.banners import MigrationBanner
    b = MigrationBanner(migration_required=True)
    b.attempt_dismiss()
    assert b.is_visible


def test_T30_banner_auto_dismissed_on_calibration_commit():
    from voxkit.ui.banners import MigrationBanner
    b = MigrationBanner(migration_required=True)
    b.on_calibration_committed()
    assert not b.is_visible


def test_T31_banner_state_survives_save_load():
    from voxkit.ui.banners import MigrationBanner
    b1 = MigrationBanner(migration_required=True)
    state = b1.serialize()
    b2 = MigrationBanner.deserialize(state)
    assert b2.is_visible == b1.is_visible


# ---------------------------------------------------------------
# CalibrationRejected dialog (Q81)
# ---------------------------------------------------------------

def test_T32_dialog_text_matches_q81_wording():
    from voxkit.ui.dialogs import CalibrationRejectedDialog
    from voxkit.classifier.classifier import Q81_DIALOG_TEXT
    dlg = CalibrationRejectedDialog(diagnostics={"delta": -0.05})
    assert dlg.message == Q81_DIALOG_TEXT


def test_T33_dialog_has_two_action_buttons():
    from voxkit.ui.dialogs import CalibrationRejectedDialog
    dlg = CalibrationRejectedDialog(diagnostics={"delta": -0.05})
    labels = dlg.action_labels()
    assert "Try again" in labels
    assert "Continue with previous" in labels


def test_T34_try_again_returns_to_calibration_flow():
    from voxkit.ui.dialogs import CalibrationRejectedDialog
    dlg = CalibrationRejectedDialog(diagnostics={"delta": -0.05})
    next_action = dlg.click("Try again")
    assert next_action == "calibration_flow"


def test_T35_continue_with_previous_closes_dialog():
    from voxkit.ui.dialogs import CalibrationRejectedDialog
    dlg = CalibrationRejectedDialog(diagnostics={"delta": -0.05})
    next_action = dlg.click("Continue with previous")
    assert next_action == "close"


# ---------------------------------------------------------------
# First-run guided tour (Q54)
# ---------------------------------------------------------------

def test_T36_tour_fires_on_first_unknown_event():
    from voxkit.ui.editor import EditorState
    state = EditorState(tour_completed=False)
    state.on_event_observed(class_id="unknown")
    assert state.tour_active


def test_T37_tour_fires_only_once_per_user():
    from voxkit.ui.editor import EditorState
    state = EditorState(tour_completed=False)
    state.on_event_observed(class_id="unknown")
    state.complete_tour()

    state2 = EditorState(tour_completed=True)
    state2.on_event_observed(class_id="unknown")
    assert not state2.tour_active


def test_T38_tour_does_not_fire_without_unknowns():
    from voxkit.ui.editor import EditorState
    state = EditorState(tour_completed=False)
    state.on_event_observed(class_id="kick")
    state.on_event_observed(class_id="snare")
    assert not state.tour_active


# ---------------------------------------------------------------
# Editor lane configuration (Q66)
# ---------------------------------------------------------------

def test_T39_default_taxonomy_5_lanes():
    from voxkit.ui.editor import build_lane_layout
    from voxkit.core.taxonomy import TaxonomyConfig
    layout = build_lane_layout(TaxonomyConfig.default_v1_0())
    assert len(layout.lanes) == 5   # 4 trained + unknown


def test_T40_5_class_taxonomy_6_lanes():
    from voxkit.ui.editor import build_lane_layout
    from voxkit.core.taxonomy import TaxonomyConfig
    tax = TaxonomyConfig(
        classes=("a", "b", "c", "d", "e"),
        midi_mapping={"a": 36, "b": 38, "c": 42, "d": 46, "e": 50},
    )
    layout = build_lane_layout(tax)
    assert len(layout.lanes) == 6


def test_T41_lane_labels_match_taxonomy_classes_order():
    from voxkit.ui.editor import build_lane_layout
    from voxkit.core.taxonomy import TaxonomyConfig
    tax = TaxonomyConfig.default_v1_0()
    layout = build_lane_layout(tax)
    labels = [lane.label for lane in layout.lanes]
    # First N labels match taxonomy classes; last is unknown.
    assert labels[:4] == list(tax.classes)
    assert labels[-1] == tax.unknown_class_id


# ---------------------------------------------------------------
# Main thread contract (Q76)
# ---------------------------------------------------------------

def test_T42_main_thread_does_not_block_during_inference(fake_model, short_audio):
    """Q76: main thread runs Qt event loop only. Test by running the
    worker and checking we can dispatch Qt events on the main thread
    while it runs."""
    from voxkit.ui.inference_worker import InferenceWorker
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.start()
    # Pretend to do main-thread work; should complete in trivial time.
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = 1 + 1
    elapsed = time.perf_counter() - t0
    worker.wait_for_completion(timeout=5.0)
    assert elapsed < 0.1


def test_T43_no_long_running_operations_on_main_thread():
    """Sentinel: import the editor module and confirm it does no expensive
    work at import time (Qt startup must be lightweight)."""
    import time
    import importlib
    t0 = time.perf_counter()
    importlib.import_module("voxkit.ui.editor")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5


# ---------------------------------------------------------------
# v0.11 panel additions (≥6/9 consensus)
# ---------------------------------------------------------------

def test_T44_main_thread_can_dispatch_callback_during_inference(
    fake_model, short_audio,
):
    """T42 measures CPython math throughput, which proves nothing
    about Qt event-loop responsiveness. T44 schedules a callback via
    a Timer (the test-friendly stand-in for QTimer.singleShot) and
    asserts it fires within 50 ms while the worker runs.

    A failing implementation: the worker thread uses GIL-heavy operations
    on data the main thread also touches, starving callbacks.
    """
    import threading
    from voxkit.ui.inference_worker import InferenceWorker

    callback_fired = threading.Event()
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.start()

    # Schedule a callback to fire 10 ms after worker starts.
    timer = threading.Timer(0.010, callback_fired.set)
    timer.start()
    fired = callback_fired.wait(timeout=0.05)   # 50 ms budget
    timer.cancel()

    worker.wait_for_completion(timeout=5.0)
    assert fired, (
        "main-thread callback did not fire within 50 ms while "
        "InferenceWorker was running; main thread may be blocked"
    )


def test_T45_calibration_dialog_diagnostics_reach_telemetry_intact():
    """The diagnostic file must record the macro-F1 delta that caused
    the rejection (Q81 + Q61). T32 checks dialog text. T45 closes the
    loop: CalibrationRejected → dialog → telemetry sink with the
    diagnostic dict intact."""
    from voxkit.ui.dialogs import CalibrationRejectedDialog
    sink = MagicMock()
    diag = {"f1_calibrated": 0.70, "f1_baseline": 0.85, "delta": -0.15}
    dlg = CalibrationRejectedDialog(diagnostics=diag, telemetry=sink)
    dlg.click("Continue with previous")

    emitted = [c.args[0] for c in sink.emit.call_args_list]
    overfit_events = [e for e in emitted
                      if e.get("event") == "calibration_overfit_guard_triggered"]
    assert overfit_events, "no calibration_overfit_guard_triggered event emitted"
    details = overfit_events[0].get("details", {})
    assert details.get("delta") == pytest.approx(-0.15)
    assert details.get("f1_calibrated") == pytest.approx(0.70)
    assert details.get("f1_baseline") == pytest.approx(0.85)


def test_T46_bleed_banner_displays_unit():
    """T26 only checks the number appears. The user-visible string must
    include 'dB' otherwise '12' is ambiguous (12 what? %?)."""
    from voxkit.ui.banners import BleedQualityBanner
    b = BleedQualityBanner()
    b.update(attenuation_db=12.3, override=False)
    assert "dB" in b.text, f"banner text missing 'dB' unit: {b.text!r}"


def test_T47_worker_thread_cleaned_up_after_completion(fake_model, short_audio):
    """A leaked worker thread holds the model in memory and surfaces as
    a 'non-daemon thread at exit' warning. Confirm the worker's thread
    has terminated within 1 s of wait_for_completion()."""
    import threading
    from voxkit.ui.inference_worker import InferenceWorker

    initial_threads = set(threading.enumerate())
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.start()
    worker.wait_for_completion(timeout=5.0)

    # Give the runtime a beat to clean up.
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        leaked = (set(threading.enumerate()) - initial_threads)
        if not any(t.is_alive() and "Inference" in t.name for t in leaked):
            break
        time.sleep(0.01)

    leaked_alive = [t for t in (set(threading.enumerate()) - initial_threads)
                    if t.is_alive() and "Inference" in t.name]
    assert leaked_alive == [], f"worker thread(s) leaked: {[t.name for t in leaked_alive]}"


# ---------------------------------------------------------------
# v0.12 panel additions (principal-engineer + Sam synthesis)
# ---------------------------------------------------------------

def test_T48_wait_for_completion_actually_times_out(fake_model):
    """v0.12: every existing InferenceWorker test relies on
    wait_for_completion(timeout=5.0). If wait_for_completion is buggy
    (e.g., always returns True without blocking) every test passes
    spuriously. Invert the contract: a worker that's mid-inference
    must NOT report completion at a 1ms timeout."""
    import threading
    import numpy as np
    from voxkit.ui.inference_worker import InferenceWorker

    long_audio = np.zeros(16_000 * 60, dtype=np.float32)   # 60s of audio
    started = threading.Event()
    block = threading.Event()

    fake_model_copy = fake_model
    original_predict = getattr(fake_model_copy, "predict", None)

    def slow_predict(*_a, **_k):
        started.set()
        block.wait(timeout=2.0)
        return []

    fake_model_copy.predict = slow_predict
    worker = InferenceWorker(audio=long_audio, model=fake_model_copy)
    worker.start()
    try:
        # Wait for the worker to enter slow_predict so we know it's
        # actually mid-flight when we test the timeout.
        assert started.wait(timeout=2.0), "worker never entered predict()"

        completed = worker.wait_for_completion(timeout=0.001)
        assert completed is False, (
            "wait_for_completion(timeout=0.001) returned True for a "
            "worker that hadn't completed; the timeout is broken and "
            "every other InferenceWorker test passes spuriously"
        )
    finally:
        block.set()
        worker.cancel()
        worker.wait_for_completion(timeout=5.0)
        if original_predict is not None:
            fake_model_copy.predict = original_predict


def test_T49_start_called_twice_raises(fake_model, short_audio):
    """v0.12: Recorder T27 covers open-twice. InferenceWorker has no
    equivalent and a UI that forgets to disable the start button is a
    plausible source of double-start. Loud-fail."""
    from voxkit.ui.inference_worker import InferenceWorker, WorkerAlreadyStarted
    worker = InferenceWorker(audio=short_audio, model=fake_model)
    worker.start()
    try:
        with pytest.raises(WorkerAlreadyStarted):
            worker.start()
    finally:
        worker.wait_for_completion(timeout=5.0)
