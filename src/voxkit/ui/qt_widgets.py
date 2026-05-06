# SPDX-License-Identifier: GPL-3.0-or-later
"""PySide6 widget layer wiring the UI state machines to actual Qt widgets.

All classes here are thin adapters: they hold a state machine instance and
keep Qt widget state in sync with it. Business logic lives in the state
machines (dialogs.py, banners.py, editor.py); this module owns only the
Qt coupling.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from voxkit.ui.banners import BleedQualityBanner, MigrationBanner
from voxkit.ui.dialogs import CalibrationRejectedDialog, RecordingProgressDialog
from voxkit.ui.editor import EditorState, build_lane_layout


# ---------------------------------------------------------------
# RecordingPanelWidget (Q24, Q73, Q76)
# ---------------------------------------------------------------

class RecordingPanelWidget(QWidget):
    """Device picker + Record/Stop button wired to Recorder (Q24, Q73)."""

    def __init__(
        self,
        recorder,
        on_recording_stopped=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._recorder = recorder
        self._on_stopped = on_recording_stopped
        self._recording = False
        self._setup_ui()
        self._populate_devices()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Input device:"))
        self._device_combo = QComboBox()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_row.addWidget(self._device_combo)
        layout.addLayout(device_row)

        self._record_btn = QPushButton("Record")
        self._record_btn.setEnabled(False)
        self._record_btn.clicked.connect(self._on_record_clicked)
        layout.addWidget(self._record_btn)

    def _populate_devices(self) -> None:
        self._device_combo.clear()
        for dev in self._recorder.list_devices():
            self._device_combo.addItem(dev.name, dev.id)
        self._on_device_changed()

    def _on_device_changed(self) -> None:
        has_device = self._device_combo.count() > 0
        if not self._recording:
            self._record_btn.setEnabled(has_device)

    def _on_record_clicked(self) -> None:
        if not self._recording:
            device_id = self._device_combo.currentData()
            self._recorder.open_stream(device_id)
            self._recording = True
            self._record_btn.setText("Stop Recording")
            self._device_combo.setEnabled(False)
        else:
            self._recorder.close_stream()
            self._recording = False
            self._record_btn.setText("Record")
            self._device_combo.setEnabled(True)
            if self._on_stopped is not None:
                self._on_stopped()

    # ---- test helpers ----

    def device_count(self) -> int:
        return self._device_combo.count()

    def device_names(self) -> list[str]:
        return [self._device_combo.itemText(i) for i in range(self._device_combo.count())]

    def is_record_enabled(self) -> bool:
        return self._record_btn.isEnabled()

    def record_button_text(self) -> str:
        return self._record_btn.text()

    def simulate_record_click(self) -> None:
        self._record_btn.click()


# ---------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------

class MainWindow(QMainWindow):
    """Application shell."""

    def __init__(self, recorder=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VoxKit")
        self._recorder = recorder
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        recording_container = QWidget()
        recording_container.setObjectName("recording_panel")
        recording_layout = QVBoxLayout(recording_container)
        recording_layout.setContentsMargins(0, 0, 0, 0)
        if self._recorder is not None:
            self._recording_panel_widget = RecordingPanelWidget(
                recorder=self._recorder,
                parent=recording_container,
            )
            recording_layout.addWidget(self._recording_panel_widget)
        layout.addWidget(recording_container)

        self._editor_panel = QWidget()
        self._editor_panel.setObjectName("editor_panel")
        layout.addWidget(self._editor_panel)


# ---------------------------------------------------------------
# InferenceProgressDialog
# ---------------------------------------------------------------

class InferenceProgressDialog(QDialog):
    """PySide6 dialog wiring InferenceWorker → RecordingProgressDialog (Q73)."""

    def __init__(self, worker=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = RecordingProgressDialog(worker=worker)
        self._worker = worker
        self.is_accepted = False
        self._setup_ui()
        self._wire_worker()

    def _setup_ui(self) -> None:
        self.setModal(True)
        layout = QVBoxLayout(self)

        self._phase_label_widgets: list[QLabel] = []
        for text in RecordingProgressDialog.phase_labels:
            lbl = QLabel(text)
            self._phase_label_widgets.append(lbl)
            layout.addWidget(lbl)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(self._cancel_btn)

    def _wire_worker(self) -> None:
        if self._worker is None:
            return
        self._worker.phase_changed.connect(self._on_phase)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.failed.connect(self._on_failed)

    def _on_phase(self, phase: str) -> None:
        self.state.on_phase(phase)

    def _on_progress(self, value: float) -> None:
        self.state.on_progress(value)
        self._progress_bar.setValue(int(value * 100))

    def _on_cancel_clicked(self) -> None:
        self.state.click_cancel()

    def _on_cancelled(self) -> None:
        self.state.on_cancelled()
        self.is_accepted = True

    def _on_completed(self, events) -> None:
        self.state.on_completed(events)
        self.is_accepted = True

    def _on_failed(self, message: str) -> None:
        self.state.on_failed(message)

    # ---- test helpers ----

    def phase_labels_text(self) -> list[str]:
        return [lbl.text() for lbl in self._phase_label_widgets]

    def progress_value(self) -> float:
        return self.state.current_progress

    def simulate_cancel(self) -> None:
        self._cancel_btn.click()


# ---------------------------------------------------------------
# BleedQualityBannerWidget
# ---------------------------------------------------------------

class BleedQualityBannerWidget(QWidget):
    """PySide6 widget wiring BleedQualityBanner state machine (Q79)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = BleedQualityBanner()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel()
        layout.addWidget(self._label)

    def update_attenuation(self, attenuation_db: float, override: bool = False) -> None:
        self._state.update(attenuation_db, override)
        self._sync_from_state()

    def _sync_from_state(self) -> None:
        self._label.setText(self._state.text)
        self._visible = self._state.is_visible
        self._color = self._state.color

    # ---- test helpers (avoid isVisible() parent-chain dependency) ----

    def is_shown(self) -> bool:
        return self._state.is_visible

    def current_color(self) -> str | None:
        return self._state.color

    def label_text(self) -> str:
        return self._state.text


# ---------------------------------------------------------------
# MigrationBannerWidget
# ---------------------------------------------------------------

class MigrationBannerWidget(QWidget):
    """PySide6 widget wiring MigrationBanner state machine (v0.10 item 17)."""

    def __init__(
        self, migration_required: bool, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = MigrationBanner(migration_required)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            QLabel(
                "VoxKit improved out-of-distribution detection in this version. "
                "Re-run a quick calibration to enable it. "
                "Until then, unknown detection uses the previous (less accurate) method."
            )
        )
        self._action_btn = QPushButton(self._state.get_action_labels()[0])
        layout.addWidget(self._action_btn)

    def is_shown(self) -> bool:
        return self._state.is_visible

    def action_labels(self) -> list[str]:
        return self._state.get_action_labels()

    def attempt_dismiss(self) -> None:
        self._state.attempt_dismiss()

    def on_calibration_committed(self) -> None:
        self._state.on_calibration_committed()


# ---------------------------------------------------------------
# Piano roll lane widgets
# ---------------------------------------------------------------

class LaneWidget(QWidget):
    """One horizontal lane in the piano roll."""

    def __init__(
        self, label: str, is_unknown: bool = False, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.lane_label = label
        self.is_unknown = is_unknown
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        display = f"? {self.lane_label}" if self.is_unknown else self.lane_label
        layout.addWidget(QLabel(display))


class PianoRollWidget(QWidget):
    """Multi-lane piano roll built from a LaneLayout (Q66)."""

    def __init__(self, taxonomy, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout_obj = build_lane_layout(taxonomy)
        self._lane_widgets: list[LaneWidget] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        unknown_label = self._layout_obj.lanes[-1].label
        for lane in self._layout_obj.lanes:
            is_unknown = lane.label == unknown_label
            w = LaneWidget(label=lane.label, is_unknown=is_unknown)
            self._lane_widgets.append(w)
            layout.addWidget(w)

    def lane_count(self) -> int:
        return len(self._lane_widgets)

    def lane_labels(self) -> list[str]:
        return [w.lane_label for w in self._lane_widgets]

    def lane_widgets(self) -> list[LaneWidget]:
        return list(self._lane_widgets)

    def unknown_lane_widget(self) -> LaneWidget | None:
        return next((w for w in self._lane_widgets if w.is_unknown), None)


# ---------------------------------------------------------------
# CalibrationRejectedQDialog
# ---------------------------------------------------------------

class CalibrationRejectedQDialog(QDialog):
    """PySide6 dialog wiring CalibrationRejectedDialog state machine (Q81)."""

    def __init__(
        self,
        diagnostics: dict,
        telemetry=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = CalibrationRejectedDialog(diagnostics, telemetry)
        self._result: str | None = None
        self._buttons: dict[str, QPushButton] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._msg_label = QLabel(self._state.message)
        self._msg_label.setWordWrap(True)
        layout.addWidget(self._msg_label)

        btn_row = QHBoxLayout()
        for label in self._state.action_labels():
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, lbl=label: self._on_click(lbl))
            self._buttons[label] = btn
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

    def _on_click(self, label: str) -> None:
        self._result = self._state.click(label)

    # ---- test helpers ----

    def message_text(self) -> str:
        return self._msg_label.text()

    def button_labels(self) -> list[str]:
        return list(self._buttons.keys())

    def result_action(self) -> str | None:
        return self._result

    def simulate_click(self, label: str) -> None:
        self._buttons[label].click()


# ---------------------------------------------------------------
# TourOverlayWidget
# ---------------------------------------------------------------

class TourOverlayWidget(QWidget):
    """First-run guided tour overlay (Q54); wires EditorState."""

    def __init__(
        self, editor_state: EditorState, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = editor_state
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Welcome! You've recorded an 'unknown' sound. "
                "Drag it into one of the trained lanes to reclassify."
            )
        )
        dismiss = QPushButton("Got it")
        dismiss.clicked.connect(self._on_dismiss)
        layout.addWidget(dismiss)
        self._dismiss_btn = dismiss

    def _on_dismiss(self) -> None:
        self._state.complete_tour()

    def notify_event(self, class_id: str) -> None:
        self._state.on_event_observed(class_id)

    def is_shown(self) -> bool:
        return self._state.tour_active

    def simulate_dismiss(self) -> None:
        self._dismiss_btn.click()
