# SPDX-License-Identifier: GPL-3.0-or-later
"""
TDD test list for Component 13: Qt application shell + wired UI.

Drives implementation of `voxkit.ui.qt_widgets` — the PySide6 layer that
wires existing state machines (dialogs.py, banners.py, editor.py,
inference_worker.py) to actual Qt widgets.

All tests skip automatically when PySide6 is not installed (optional dep).
UI tests are functional-state and signal-contract tests, not pixel tests.
We assert widget visibility, text content, and action routing — not layout
coordinates.

============================================================
TEST LIST (implement strictly in order)
============================================================

QApplication smoke
  T01  QApplication can be created in offscreen mode (no display required)
  T02  MainWindow has window title "VoxKit"
  T03  MainWindow central widget contains recording and editor panels

InferenceProgressDialog (Q73 — wires state machine to PySide6)
  T04  Dialog exposes the three phase labels from RecordingProgressDialog
  T05  Progress value advances in dialog state when worker emits progress
  T06  Cancel button calls worker.cancel() exactly once
  T07  Cancelled signal from worker closes the dialog
  T08  Completed signal from worker closes the dialog
  T09  Failed signal propagates error message into dialog state
  T10  Dialog is modal (QDialog.isModal() is True)

BleedQualityBannerWidget (Q79 — wires banner state to PySide6)
  T11  Widget hidden when attenuation_db >= 20
  T12  Widget visible when attenuation_db < 10
  T13  Widget visible when 10 <= attenuation_db < 20
  T14  Banner text includes numeric value and "dB" unit
  T15  Widget hidden when override=True regardless of attenuation

MigrationBannerWidget (v0.10 item 17 — wires to PySide6)
  T16  Widget visible when migration_required=True
  T17  Widget hidden when migration_required=False
  T18  Only action label is "Recalibrate now" (no dismiss / remind-later)
  T19  attempt_dismiss() leaves banner visible (persistent)
  T20  on_calibration_committed() hides banner

PianoRollWidget (Q66 — wires lane layout to PySide6)
  T21  Default 4-class taxonomy creates 5 lane widgets (4 trained + unknown)
  T22  Lane labels match taxonomy.classes order; unknown is last
  T23  Unknown lane widget is marked is_unknown=True; trained lanes False

CalibrationRejectedQDialog (Q81 — wires dialog state to PySide6)
  T24  Dialog message text matches Q81 wording from classifier module
  T25  Dialog exposes "Try again" and "Continue with previous" buttons
  T26  Clicking "Try again" sets result_action to "calibration_flow"
  T27  Clicking "Continue with previous" sets result_action to "close"

TourOverlayWidget (Q54 — wires first-run tour state to PySide6)
  T28  Overlay hidden when tour_completed=True from the start
  T29  Overlay becomes visible after notify_event("unknown") (tour not done)
  T30  Clicking "Got it" hides overlay and marks tour complete

RecordingPanelWidget (Q24, Q73, Q76 — audio device → MainWindow wiring)
  T31  RecordingPanelWidget constructs with a fake recorder
  T32  Device picker is populated from recorder.list_devices()
  T33  Record button is disabled when recorder returns no devices
  T34  Record button is enabled when at least one device is available
  T35  Clicking Record calls recorder.open_stream(selected_device_id)
  T36  Record button text changes to "Stop Recording" after recording starts
  T37  Clicking Stop calls recorder.close_stream()
  T38  Stopping recording invokes on_recording_stopped callback
  T39  MainWindow created with a recorder has RecordingPanelWidget in the panel
  T40  Device picker shows exactly what recorder.list_devices() returns
"""

from __future__ import annotations

import os
import sys

import pytest

# Skip the entire module when PySide6 is not installed.
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


# ---------------------------------------------------------------
# QApplication singleton — offscreen so CI needs no display
# ---------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


# ---------------------------------------------------------------
# Shared helpers / fake collaborators
# ---------------------------------------------------------------

@pytest.fixture
def fake_taxonomy():
    from unittest.mock import MagicMock
    tax = MagicMock()
    tax.classes = ("kick", "snare", "closed_hat", "open_hat")
    tax.unknown_class_id = "unknown"
    return tax


class _FakeWorker:
    """Minimal worker stand-in with custom _Signal objects; no real thread."""

    def __init__(self):
        from voxkit.ui.inference_worker import _Signal
        self.phase_changed = _Signal()
        self.progress = _Signal()
        self.completed = _Signal()
        self.cancelled = _Signal()
        self.failed = _Signal()
        self.cancel_call_count = 0

    def cancel(self):
        self.cancel_call_count += 1


@pytest.fixture
def fake_worker():
    return _FakeWorker()


# ---------------------------------------------------------------
# QApplication smoke (T01-T03)
# ---------------------------------------------------------------

def test_T01_qapplication_created_offscreen(qapp):
    """QApplication is live in offscreen mode; no display required."""
    assert qapp is not None
    assert QApplication.instance() is qapp


def test_T02_main_window_title(qapp):
    from voxkit.ui.qt_widgets import MainWindow
    w = MainWindow()
    assert w.windowTitle() == "VoxKit"
    w.close()


def test_T03_main_window_has_recording_and_editor_panels(qapp):
    from PySide6.QtWidgets import QWidget
    from voxkit.ui.qt_widgets import MainWindow
    w = MainWindow()
    recording = w.findChild(QWidget, "recording_panel")
    editor = w.findChild(QWidget, "editor_panel")
    assert recording is not None, "recording_panel not found"
    assert editor is not None, "editor_panel not found"
    w.close()


# ---------------------------------------------------------------
# InferenceProgressDialog (T04-T10)
# ---------------------------------------------------------------

def test_T04_inference_dialog_shows_three_phase_labels(qapp, fake_worker):
    from voxkit.ui.dialogs import RecordingProgressDialog
    from voxkit.ui.qt_widgets import InferenceProgressDialog
    dlg = InferenceProgressDialog(worker=fake_worker)
    labels = dlg.phase_labels_text()
    assert labels == list(RecordingProgressDialog.phase_labels), (
        f"expected {list(RecordingProgressDialog.phase_labels)}, got {labels}"
    )


def test_T05_inference_dialog_progress_advances(qapp, fake_worker):
    from voxkit.ui.qt_widgets import InferenceProgressDialog
    dlg = InferenceProgressDialog(worker=fake_worker)
    fake_worker.progress.emit(0.42)
    assert dlg.progress_value() == pytest.approx(0.42)


def test_T06_cancel_button_calls_worker_cancel_once(qapp, fake_worker):
    from voxkit.ui.qt_widgets import InferenceProgressDialog
    dlg = InferenceProgressDialog(worker=fake_worker)
    dlg.simulate_cancel()
    dlg.simulate_cancel()  # second click must be idempotent
    assert fake_worker.cancel_call_count == 1


def test_T07_cancelled_signal_closes_dialog(qapp, fake_worker):
    from voxkit.ui.qt_widgets import InferenceProgressDialog
    dlg = InferenceProgressDialog(worker=fake_worker)
    assert not dlg.is_accepted
    fake_worker.cancelled.emit()
    assert dlg.is_accepted


def test_T08_completed_signal_closes_dialog(qapp, fake_worker):
    from voxkit.ui.qt_widgets import InferenceProgressDialog
    dlg = InferenceProgressDialog(worker=fake_worker)
    fake_worker.completed.emit([])
    assert dlg.is_accepted


def test_T09_failed_signal_propagates_error(qapp, fake_worker):
    from voxkit.ui.qt_widgets import InferenceProgressDialog
    dlg = InferenceProgressDialog(worker=fake_worker)
    fake_worker.failed.emit("boom")
    assert dlg.state.error_shown is True
    assert dlg.state.error_text == "boom"


def test_T10_inference_dialog_is_modal(qapp, fake_worker):
    from voxkit.ui.qt_widgets import InferenceProgressDialog
    dlg = InferenceProgressDialog(worker=fake_worker)
    assert dlg.isModal()


# ---------------------------------------------------------------
# BleedQualityBannerWidget (T11-T15)
# ---------------------------------------------------------------

def test_T11_bleed_banner_hidden_at_high_attenuation(qapp):
    from voxkit.ui.qt_widgets import BleedQualityBannerWidget
    b = BleedQualityBannerWidget()
    b.update_attenuation(25.0, override=False)
    assert not b.is_shown()


def test_T12_bleed_banner_visible_red_below_10(qapp):
    from voxkit.ui.qt_widgets import BleedQualityBannerWidget
    b = BleedQualityBannerWidget()
    b.update_attenuation(5.0, override=False)
    assert b.is_shown()
    assert b.current_color() == "red"


def test_T13_bleed_banner_visible_yellow_10_to_20(qapp):
    from voxkit.ui.qt_widgets import BleedQualityBannerWidget
    b = BleedQualityBannerWidget()
    b.update_attenuation(15.0, override=False)
    assert b.is_shown()
    assert b.current_color() == "yellow"


def test_T14_bleed_banner_text_has_value_and_unit(qapp):
    from voxkit.ui.qt_widgets import BleedQualityBannerWidget
    b = BleedQualityBannerWidget()
    b.update_attenuation(12.3, override=False)
    text = b.label_text()
    assert "12.3" in text
    assert "dB" in text


def test_T15_bleed_banner_hidden_when_override(qapp):
    from voxkit.ui.qt_widgets import BleedQualityBannerWidget
    b = BleedQualityBannerWidget()
    b.update_attenuation(5.0, override=True)
    assert not b.is_shown()


# ---------------------------------------------------------------
# MigrationBannerWidget (T16-T20)
# ---------------------------------------------------------------

def test_T16_migration_banner_visible_when_required(qapp):
    from voxkit.ui.qt_widgets import MigrationBannerWidget
    b = MigrationBannerWidget(migration_required=True)
    assert b.is_shown()


def test_T17_migration_banner_hidden_when_not_required(qapp):
    from voxkit.ui.qt_widgets import MigrationBannerWidget
    b = MigrationBannerWidget(migration_required=False)
    assert not b.is_shown()


def test_T18_migration_banner_action_label_only_recalibrate(qapp):
    from voxkit.ui.qt_widgets import MigrationBannerWidget
    b = MigrationBannerWidget(migration_required=True)
    labels = b.action_labels()
    assert labels == ["Recalibrate now"]
    for forbidden in ("Remind", "Later", "Dismiss", "later", "remind"):
        assert forbidden not in " ".join(labels), (
            f"Unexpected action label containing '{forbidden}': {labels}"
        )


def test_T19_migration_banner_attempt_dismiss_stays_visible(qapp):
    from voxkit.ui.qt_widgets import MigrationBannerWidget
    b = MigrationBannerWidget(migration_required=True)
    b.attempt_dismiss()
    assert b.is_shown()


def test_T20_migration_banner_hides_on_calibration_committed(qapp):
    from voxkit.ui.qt_widgets import MigrationBannerWidget
    b = MigrationBannerWidget(migration_required=True)
    b.on_calibration_committed()
    assert not b.is_shown()


# ---------------------------------------------------------------
# PianoRollWidget (T21-T23)
# ---------------------------------------------------------------

def test_T21_piano_roll_4_class_taxonomy_creates_5_lanes(qapp, fake_taxonomy):
    from voxkit.ui.qt_widgets import PianoRollWidget
    roll = PianoRollWidget(taxonomy=fake_taxonomy)
    assert roll.lane_count() == 5


def test_T22_piano_roll_lane_labels_match_taxonomy_order(qapp, fake_taxonomy):
    from voxkit.ui.qt_widgets import PianoRollWidget
    roll = PianoRollWidget(taxonomy=fake_taxonomy)
    labels = roll.lane_labels()
    assert labels[:4] == list(fake_taxonomy.classes)
    assert labels[4] == fake_taxonomy.unknown_class_id


def test_T23_piano_roll_unknown_lane_distinct_from_trained(qapp, fake_taxonomy):
    from voxkit.ui.qt_widgets import PianoRollWidget
    roll = PianoRollWidget(taxonomy=fake_taxonomy)
    unknown_widget = roll.unknown_lane_widget()
    assert unknown_widget is not None
    assert unknown_widget.is_unknown is True
    trained = [w for w in roll.lane_widgets() if not w.is_unknown]
    assert len(trained) == 4
    assert all(not w.is_unknown for w in trained)


# ---------------------------------------------------------------
# CalibrationRejectedQDialog (T24-T27)
# ---------------------------------------------------------------

def test_T24_calib_rejected_dialog_shows_q81_text(qapp):
    from voxkit.classifier.classifier import Q81_DIALOG_TEXT
    from voxkit.ui.qt_widgets import CalibrationRejectedQDialog
    dlg = CalibrationRejectedQDialog(diagnostics={"f1_delta": -0.05})
    assert Q81_DIALOG_TEXT in dlg.message_text()


def test_T25_calib_rejected_dialog_has_two_buttons(qapp):
    from voxkit.ui.qt_widgets import CalibrationRejectedQDialog
    dlg = CalibrationRejectedQDialog(diagnostics={})
    labels = dlg.button_labels()
    assert "Try again" in labels
    assert "Continue with previous" in labels


def test_T26_try_again_routes_to_calibration_flow(qapp):
    from voxkit.ui.qt_widgets import CalibrationRejectedQDialog
    dlg = CalibrationRejectedQDialog(diagnostics={})
    dlg.simulate_click("Try again")
    assert dlg.result_action() == "calibration_flow"


def test_T27_continue_with_previous_routes_to_close(qapp):
    from voxkit.ui.qt_widgets import CalibrationRejectedQDialog
    dlg = CalibrationRejectedQDialog(diagnostics={})
    dlg.simulate_click("Continue with previous")
    assert dlg.result_action() == "close"


# ---------------------------------------------------------------
# TourOverlayWidget (T28-T30)
# ---------------------------------------------------------------

def test_T28_tour_overlay_hidden_when_tour_already_completed(qapp):
    from voxkit.ui.editor import EditorState
    from voxkit.ui.qt_widgets import TourOverlayWidget
    state = EditorState(tour_completed=True)
    overlay = TourOverlayWidget(editor_state=state)
    assert not overlay.is_shown()


def test_T29_tour_overlay_visible_after_unknown_event(qapp):
    from voxkit.ui.editor import EditorState
    from voxkit.ui.qt_widgets import TourOverlayWidget
    state = EditorState(tour_completed=False)
    overlay = TourOverlayWidget(editor_state=state)
    assert not overlay.is_shown()
    overlay.notify_event("unknown")
    assert overlay.is_shown()


def test_T30_tour_overlay_dismiss_hides_and_completes_tour(qapp):
    from voxkit.ui.editor import EditorState
    from voxkit.ui.qt_widgets import TourOverlayWidget
    state = EditorState(tour_completed=False)
    overlay = TourOverlayWidget(editor_state=state)
    overlay.notify_event("unknown")
    assert overlay.is_shown()
    overlay.simulate_dismiss()
    assert not overlay.is_shown()
    assert state.tour_active is False


# ---------------------------------------------------------------
# Fake recorder for wiring tests (no real audio hardware needed)
# ---------------------------------------------------------------

class _FakeRecorder:
    """Minimal recorder stand-in; records calls without opening a stream."""

    def __init__(self, devices=None) -> None:
        from voxkit.audio.recorder import DeviceInfo
        self._devices = devices if devices is not None else [
            DeviceInfo(id="0", name="USB Mic", default_rate=16_000),
            DeviceInfo(id="1", name="Built-in Mic", default_rate=44_100),
        ]
        self.open_stream_calls: list[str] = []
        self.close_stream_calls: int = 0

    def list_devices(self):
        return list(self._devices)

    def open_stream(self, device_id: str) -> None:
        self.open_stream_calls.append(device_id)

    def close_stream(self) -> None:
        self.close_stream_calls += 1


# ---------------------------------------------------------------
# RecordingPanelWidget (T31-T40)
# ---------------------------------------------------------------

def test_T31_recording_panel_constructs(qapp):
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    rec = _FakeRecorder()
    w = RecordingPanelWidget(recorder=rec)
    assert w is not None


def test_T32_device_picker_populated_from_recorder(qapp):
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    rec = _FakeRecorder()
    w = RecordingPanelWidget(recorder=rec)
    assert w.device_count() == 2
    assert "USB Mic" in w.device_names()
    assert "Built-in Mic" in w.device_names()


def test_T33_record_button_disabled_when_no_devices(qapp):
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    rec = _FakeRecorder(devices=[])
    w = RecordingPanelWidget(recorder=rec)
    assert not w.is_record_enabled()


def test_T34_record_button_enabled_when_device_available(qapp):
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    rec = _FakeRecorder()
    w = RecordingPanelWidget(recorder=rec)
    assert w.is_record_enabled()


def test_T35_record_click_opens_stream_with_selected_device(qapp):
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    rec = _FakeRecorder()
    w = RecordingPanelWidget(recorder=rec)
    w.simulate_record_click()
    assert rec.open_stream_calls == ["0"]


def test_T36_record_button_text_changes_to_stop(qapp):
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    rec = _FakeRecorder()
    w = RecordingPanelWidget(recorder=rec)
    w.simulate_record_click()
    assert "Stop" in w.record_button_text()


def test_T37_stop_click_closes_stream(qapp):
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    rec = _FakeRecorder()
    w = RecordingPanelWidget(recorder=rec)
    w.simulate_record_click()   # start
    w.simulate_record_click()   # stop
    assert rec.close_stream_calls == 1


def test_T38_stop_invokes_callback(qapp):
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    rec = _FakeRecorder()
    fired: list = []
    w = RecordingPanelWidget(recorder=rec, on_recording_stopped=lambda: fired.append(1))
    w.simulate_record_click()   # start
    w.simulate_record_click()   # stop
    assert len(fired) == 1


def test_T39_main_window_with_recorder_has_recording_panel_widget(qapp):
    from voxkit.ui.qt_widgets import MainWindow, RecordingPanelWidget
    rec = _FakeRecorder()
    w = MainWindow(recorder=rec)
    rp = w.findChild(RecordingPanelWidget)
    assert rp is not None, "RecordingPanelWidget not found inside MainWindow"
    w.close()


def test_T40_device_picker_shows_only_recorder_list(qapp):
    from voxkit.audio.recorder import DeviceInfo
    from voxkit.ui.qt_widgets import RecordingPanelWidget
    devices = [DeviceInfo(id="5", name="Focusrite 2i2", default_rate=48_000)]
    rec = _FakeRecorder(devices=devices)
    w = RecordingPanelWidget(recorder=rec)
    assert w.device_count() == 1
    assert w.device_names() == ["Focusrite 2i2"]
