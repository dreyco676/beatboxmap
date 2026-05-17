# SPDX-License-Identifier: GPL-3.0-or-later
"""PySide6 widget layer wiring the UI state machines to actual Qt widgets.

All classes here are thin adapters: they hold a state machine instance and
keep Qt widget state in sync with it. Business logic lives in the state
machines (dialogs.py, banners.py, editor.py); this module owns only the
Qt coupling.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QMenu,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter

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
        on_before_start=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._recorder = recorder
        self._on_stopped = on_recording_stopped
        self._on_before_start = on_before_start
        self._recording = False
        self._pending_start = False
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
        if not self._recording and not self._pending_start:
            self._record_btn.setEnabled(has_device)

    def _on_record_clicked(self) -> None:
        if self._recording:
            self._do_stop_recording()
        elif self._pending_start:
            # Cancel the count-in
            self._pending_start = False
            from voxkit.audio.recorder import stop_playback
            stop_playback()
            self._record_btn.setText("Record")
            self._device_combo.setEnabled(True)
        else:
            self._pending_start = True
            self._record_btn.setText("Cancel")
            self._device_combo.setEnabled(False)
            if self._on_before_start is not None:
                self._on_before_start(self._do_start_recording)
            else:
                self._do_start_recording()

    def _do_start_recording(self) -> None:
        if not self._pending_start:
            return  # cancelled during count-in
        self._pending_start = False
        self._recording = True
        device_id = self._device_combo.currentData()
        self._recorder.open_stream(device_id)
        self._record_btn.setText("Stop Recording")

    def _do_stop_recording(self) -> None:
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
    """Application shell — wires model loading, calibration, recording, inference, export."""

    def __init__(self, recorder=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VoxKit")

        from voxkit.audio.recorder import Recorder
        self._recorder = recorder if recorder is not None else Recorder()

        self._extractor = None
        self._classifier = None
        self._manager = None
        self._is_calibrated = False
        self._last_events: list = []
        self._last_audio = None

        self._setup_ui()
        # Defer model load until after window is shown so the UI appears immediately.
        QTimer.singleShot(100, self._load_model)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Export MIDI…", self._on_export)
        file_menu.addSeparator()
        file_menu.addAction("Save Calibration…", self._on_save_calibration)
        file_menu.addAction("Load Calibration…", self._on_load_calibration)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(6, 4, 6, 6)

        # ── Winamp-style LCD header ────────────────────────────────────
        header = QWidget()
        header.setStyleSheet("background-color: #001200; border: 1px solid #111111;")
        hlayout = QVBoxLayout(header)
        hlayout.setContentsMargins(6, 4, 6, 4)
        hlayout.setSpacing(2)

        title_lbl = QLabel("♪ VOXKIT")
        title_lbl.setObjectName("lcd_title")
        hlayout.addWidget(title_lbl)

        self._status_label = QLabel("Loading model…")
        self._status_label.setObjectName("lcd")
        self._status_label.setWordWrap(True)
        hlayout.addWidget(self._status_label)

        root.addWidget(header)

        # ---- controls row ----
        ctrl = QHBoxLayout()

        self._calibrate_btn = QPushButton("Calibrate…")
        self._calibrate_btn.setEnabled(False)
        self._calibrate_btn.clicked.connect(self._on_calibrate)
        ctrl.addWidget(self._calibrate_btn)

        ctrl.addWidget(QLabel("BPM:"))
        self._bpm_spin = QDoubleSpinBox()
        self._bpm_spin.setRange(40, 240)
        self._bpm_spin.setValue(120)
        self._bpm_spin.setSingleStep(1)
        self._bpm_spin.setDecimals(1)
        self._bpm_spin.valueChanged.connect(self._on_grid_changed)
        ctrl.addWidget(self._bpm_spin)

        ctrl.addWidget(QLabel("Bars:"))
        self._bars_spin = QSpinBox()
        self._bars_spin.setRange(1, 32)
        self._bars_spin.setValue(4)
        self._bars_spin.valueChanged.connect(self._on_grid_changed)
        ctrl.addWidget(self._bars_spin)

        self._countin_check = QCheckBox("Count-in")
        self._countin_check.setToolTip("Play a 1-bar click count-in before recording starts")
        ctrl.addWidget(self._countin_check)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ---- recording panel ----
        recording_container = QWidget()
        recording_container.setObjectName("recording_panel")
        rec_layout = QVBoxLayout(recording_container)
        rec_layout.setContentsMargins(0, 0, 0, 0)
        self._recording_panel_widget = RecordingPanelWidget(
            recorder=self._recorder,
            on_recording_stopped=self._on_recording_stopped,
            on_before_start=self._count_in_then_start,
            parent=recording_container,
        )
        rec_layout.addWidget(self._recording_panel_widget)
        root.addWidget(recording_container)

        # ---- playback controls ----
        play_row = QHBoxLayout()
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setEnabled(False)
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._on_play_toggled)
        play_row.addWidget(self._play_btn)
        play_row.addStretch()
        root.addLayout(play_row)

        # ---- events view (editor_panel objectName preserved for T03) ----
        editor_container = QWidget()
        editor_container.setObjectName("editor_panel")
        editor_layout = QVBoxLayout(editor_container)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        self._events_view = EventsViewWidget()
        self._events_view.events_changed.connect(self._on_events_changed)
        editor_layout.addWidget(self._events_view)
        root.addWidget(editor_container, stretch=1)

        # ---- mute row (populated in _load_model once taxonomy is known) ----
        self._mute_checks: dict[str, QCheckBox] = {}
        mute_row = QHBoxLayout()
        mute_row.addWidget(QLabel("Mute:"))
        self._mute_row = mute_row
        root.addLayout(mute_row)

        # ---- bottom action row ----
        bottom_row = QHBoxLayout()
        self._preview_btn = QPushButton("▶  Preview MIDI")
        self._preview_btn.setEnabled(False)
        self._preview_btn.setCheckable(True)
        self._preview_btn.setToolTip("Render events to synth drum sounds and play back")
        self._preview_btn.clicked.connect(self._on_preview_midi)
        bottom_row.addWidget(self._preview_btn)
        self._reinforce_btn = QPushButton("Reinforce Model")
        self._reinforce_btn.setEnabled(False)
        self._reinforce_btn.setToolTip(
            "Reclassify any wrong events above (right-click), then click here "
            "to add this recording to the training data and re-fit."
        )
        self._reinforce_btn.clicked.connect(self._on_reinforce)
        bottom_row.addWidget(self._reinforce_btn)
        bottom_row.addStretch()
        self._export_btn = QPushButton("Export MIDI…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        bottom_row.addWidget(self._export_btn)
        root.addLayout(bottom_row)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from voxkit.classifier.embeddings import EmbeddingExtractor
            from voxkit.classifier.classifier import Classifier
            from voxkit.classifier.calibration_manager import CalibrationManager
            from voxkit.core.taxonomy import TaxonomyConfig

            self._extractor = EmbeddingExtractor.from_default("beats")
            taxonomy = TaxonomyConfig.default_v1_0()
            self._classifier = Classifier(taxonomy, self._extractor.embedding_dim)
            self._manager = CalibrationManager(self._classifier)

            # Populate per-class mute checkboxes
            all_classes = list(taxonomy.classes) + [taxonomy.unknown_class_id]
            for cls in all_classes:
                chk = QCheckBox(cls.replace("_", " ").title())
                chk.stateChanged.connect(self._on_mute_changed)
                self._mute_row.addWidget(chk)
                self._mute_checks[cls] = chk
            self._mute_row.addStretch()

            self._calibrate_btn.setEnabled(True)
            self._status_label.setText(
                "Model loaded. Click Calibrate… to teach VoxKit your sounds."
            )
        except Exception as exc:
            self._status_label.setText(f"Model load failed: {exc}")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _on_calibrate(self) -> None:
        if self._extractor is None:
            return
        wizard = CalibrationWizardDialog(
            extractor=self._extractor,
            manager=self._manager,
            classifier=self._classifier,
            parent=self,
        )
        if wizard.exec() == QDialog.Accepted:
            self._is_calibrated = True
            self._status_label.setText("Calibrated. Select a device and click Record.")

    # ------------------------------------------------------------------
    # Recording → inference
    # ------------------------------------------------------------------

    def _on_recording_stopped(self) -> None:
        if not self._is_calibrated:
            QMessageBox.warning(
                self, "Not calibrated",
                "Please click Calibrate… and record examples of each sound first.",
            )
            return

        audio = self._recorder.get_recorded_audio()
        if len(audio) < 1600:  # less than 0.1 s at 16 kHz
            self._status_label.setText("Recording too short — try again.")
            return

        self._last_audio = audio
        self._play_btn.setEnabled(True)
        self._run_inference(audio)

    def _run_inference(self, audio) -> None:
        import numpy as np
        from voxkit.ui.model import Model
        from voxkit.ui.inference_pipeline import run_pipeline
        from voxkit.dsp.onsets import OnsetDetector

        model = Model(self._extractor, self._classifier)
        model.prepare(audio)

        onset_detector = OnsetDetector(sample_rate=16_000)

        self._status_label.setText("Processing… (detecting onsets and classifying)")
        QApplication.processEvents()

        try:
            result = run_pipeline(
                audio,
                model,
                detect_onsets=lambda a: onset_detector.detect(a),
            )
        except Exception as exc:
            self._status_label.setText(f"Inference failed: {exc}")
            return

        if result.cancelled:
            self._status_label.setText("Processing cancelled.")
            return

        self._last_events = result.events
        bpm = self._bpm_spin.value()
        bars = self._bars_spin.value()
        self._events_view.set_events(result.events, bpm, bars)
        self._export_btn.setEnabled(True)
        self._reinforce_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        n = len(result.events)
        self._status_label.setText(
            f"Done — {n} event{'s' if n != 1 else ''} detected. "
            "Export MIDI… to save."
        )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _count_in_then_start(self, start_fn) -> None:
        if not self._countin_check.isChecked():
            start_fn()
            return
        from voxkit.playback.synth import generate_click_track
        from voxkit.audio.recorder import play_nonblocking
        bpm = self._bpm_spin.value()
        click = generate_click_track(bpm, bars=1)
        play_nonblocking(click)
        bar_ms = int(4 * 60_000 / bpm) + 80
        QTimer.singleShot(bar_ms, start_fn)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _on_play_toggled(self, checked: bool) -> None:
        from voxkit.audio.recorder import play_nonblocking, stop_playback
        if checked:
            if self._last_audio is None:
                self._play_btn.setChecked(False)
                return
            if self._preview_btn.isChecked():
                self._preview_btn.setChecked(False)
                stop_playback()
                self._on_preview_finished()
            self._play_btn.setText("■  Stop")
            play_nonblocking(self._last_audio, sample_rate=16_000)
            duration_ms = int(len(self._last_audio) / 16_000 * 1000) + 300
            QTimer.singleShot(duration_ms, self._on_playback_finished)
        else:
            stop_playback()
            self._on_playback_finished()

    def _on_playback_finished(self) -> None:
        self._play_btn.setChecked(False)
        self._play_btn.setText("▶  Play")

    # ------------------------------------------------------------------
    # Grid / events callbacks
    # ------------------------------------------------------------------

    def _on_grid_changed(self) -> None:
        if self._last_events:
            self._events_view.set_events(
                self._last_events, self._bpm_spin.value(), self._bars_spin.value()
            )

    def _on_events_changed(self, events: list) -> None:
        self._last_events = events

    def _on_mute_changed(self) -> None:
        self._events_view.set_muted(self._muted_classes())

    def _muted_classes(self) -> set[str]:
        return {cls for cls, chk in self._mute_checks.items() if chk.isChecked()}

    # ------------------------------------------------------------------
    # MIDI preview
    # ------------------------------------------------------------------

    def _on_preview_midi(self, checked: bool) -> None:
        from voxkit.audio.recorder import play_nonblocking, stop_playback
        from voxkit.playback.synth import render_events
        if checked:
            if not self._last_events:
                self._preview_btn.setChecked(False)
                return
            if self._play_btn.isChecked():
                self._play_btn.setChecked(False)
                stop_playback()
                self._on_playback_finished()
            self._preview_btn.setText("■  Stop Preview")
            audio = render_events(
                self._last_events,
                bpm=self._bpm_spin.value(),
                bars=self._bars_spin.value(),
                muted=self._muted_classes(),
            )
            play_nonblocking(audio)
            duration_ms = int(self._bars_spin.value() * 4 * 60_000 / self._bpm_spin.value()) + 300
            QTimer.singleShot(duration_ms, self._on_preview_finished)
        else:
            stop_playback()
            self._on_preview_finished()

    def _on_preview_finished(self) -> None:
        self._preview_btn.setChecked(False)
        self._preview_btn.setText("▶  Preview MIDI")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        if not self._last_events:
            QMessageBox.information(self, "Nothing to export", "Record a beat first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDI", "", "MIDI files (*.mid)"
        )
        if not path:
            return
        from voxkit.export.midi import export_midi
        from voxkit.core.session import TimeSignature
        try:
            export_midi(
                self._last_events,
                Path(path),
                bpm=self._bpm_spin.value(),
                taxonomy=self._classifier.taxonomy,
            )
            self._status_label.setText(f"Exported to {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ------------------------------------------------------------------
    # Calibration persistence
    # ------------------------------------------------------------------

    def _on_save_calibration(self) -> None:
        if not self._is_calibrated or self._classifier is None:
            QMessageBox.information(self, "No calibration", "Calibrate first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "", "VoxKit Calibration (*.vkc)"
        )
        if not path:
            return
        try:
            self._classifier.save(Path(path))
            self._status_label.setText(f"Calibration saved to {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "VoxKit Calibration (*.vkc)"
        )
        if not path:
            return
        try:
            from voxkit.classifier.classifier import Classifier
            self._classifier = Classifier.load(Path(path))
            self._is_calibrated = True
            self._status_label.setText(
                f"Calibration loaded from {Path(path).name}. Ready to record."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    # ------------------------------------------------------------------
    # Reinforcement loop
    # ------------------------------------------------------------------

    def _on_reinforce(self) -> None:
        import numpy as np

        if self._last_audio is None or not self._last_events or self._classifier is None:
            return

        unknown_id = self._classifier.taxonomy.unknown_class_id
        known_events = [ev for ev in self._last_events if ev.class_id != unknown_id]
        if not known_events:
            QMessageBox.information(
                self, "Nothing to reinforce",
                "All events are classified as unknown. "
                "Reclassify some events first (right-click on a dot).",
            )
            return

        self._status_label.setText(
            f"Extracting embeddings for {len(known_events)} confirmed events…"
        )
        QApplication.processEvents()

        try:
            onset_times = [ev.t for ev in known_events]
            new_X = self._extractor.extract_at_onsets(
                self._last_audio, onset_times, sample_rate=16_000
            )
            new_y = np.array([ev.class_id for ev in known_events])

            existing_X = self._classifier._stored_avp_X
            existing_y = self._classifier._stored_avp_y
            existing_s = self._classifier._stored_avp_subjects

            next_subject = int(existing_s.max()) + 1 if len(existing_s) > 0 else 0
            new_s = np.arange(len(new_y)) + next_subject
            combined_X = np.vstack([existing_X, new_X])
            combined_y = np.concatenate([existing_y, new_y])
            combined_s = np.concatenate([existing_s, new_s])

            self._classifier.fit(combined_X, combined_y, combined_s)
            n = len(known_events)
            self._status_label.setText(
                f"Model reinforced with {n} event{'s' if n != 1 else ''}. "
                "Record again to see the improvement."
            )
        except Exception as exc:
            self._status_label.setText(f"Reinforcement failed: {exc}")


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


# ---------------------------------------------------------------
# EventsViewWidget — piano-roll timeline of classified hits
# ---------------------------------------------------------------

class EventsViewWidget(QWidget):
    """Paints classified percussion events as a colour-coded piano roll.

    Rows: one per class (taxonomy order + unknown at bottom).
    Columns: time from 0 to total duration (bars × beat length).
    Right-click any dot to reclassify or delete the event.
    """

    events_changed = Signal(list)

    _ROW_HEIGHT = 26
    _LABEL_W = 110
    _DOT_R = 6
    # Winamp spectrum-analyser palette: bright neons on near-black
    _CLASS_COLORS: dict[str, tuple[int, int, int]] = {
        "kick":        (0,   220, 0),    # bright green  (like Winamp peak bars)
        "snare":       (0,   200, 220),  # cyan
        "closed_hat":  (220, 200, 0),    # yellow
        "open_hat":    (220, 120, 0),    # orange
        "unknown":     (100, 100, 100),  # dim gray
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._events: list = []
        self._bpm: float = 120.0
        self._bars: int = 4
        self._classes: list[str] = []
        self._all_classes: list[str] = []
        self._muted: set[str] = set()
        self.setMinimumHeight(160)

    def set_muted(self, muted: set[str]) -> None:
        self._muted = set(muted)
        self.update()

    def set_events(self, events: list, bpm: float, bars: int) -> None:
        from voxkit.core.taxonomy import TaxonomyConfig
        taxonomy_order = list(TaxonomyConfig.default_v1_0().classes)
        self._all_classes = list(taxonomy_order) + ["unknown"]
        self._events = list(events)
        self._bpm = bpm
        self._bars = bars
        self._rebuild_classes()
        h = len(self._classes) * self._ROW_HEIGHT + 24
        self.setMinimumHeight(h)
        self.update()

    def _rebuild_classes(self) -> None:
        from voxkit.core.taxonomy import TaxonomyConfig
        taxonomy_order = list(TaxonomyConfig.default_v1_0().classes)
        seen = {e.class_id for e in self._events}
        self._classes = [c for c in taxonomy_order if c in seen]
        if "unknown" in seen:
            self._classes.append("unknown")
        if not self._classes:
            self._classes = taxonomy_order

    # ------------------------------------------------------------------
    # Mouse / context menu
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._events and event.button() in (
            Qt.MouseButton.RightButton, Qt.MouseButton.LeftButton
        ):
            ev = self._find_event_at(event.position())
            if ev is not None:
                self._show_event_menu(ev, event.globalPosition().toPoint())
                return
        super().mousePressEvent(event)

    def _find_event_at(self, pos):
        if not self._events or not self._classes:
            return None
        duration = self._bars * 4.0 * 60.0 / self._bpm
        timeline_w = max(1, self.width() - self._LABEL_W - 12)
        top_pad = 12
        px, py = pos.x(), pos.y()
        best_ev, best_dist = None, float(self._DOT_R + 6)
        for ev in self._events:
            if ev.class_id not in self._classes:
                continue
            row = self._classes.index(ev.class_id)
            cy = top_pad + row * self._ROW_HEIGHT + self._ROW_HEIGHT / 2.0
            frac = max(0.0, min(1.0, ev.t / duration if duration > 0 else 0.0))
            cx = self._LABEL_W + frac * timeline_w
            dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_ev = ev
        return best_ev

    def _show_event_menu(self, ev, global_pos) -> None:
        from voxkit.core.session import Event
        menu = QMenu(self)
        for cls in self._all_classes:
            action = menu.addAction(cls.replace("_", " ").title())
            action.setCheckable(True)
            action.setChecked(cls == ev.class_id)
            action.setData(cls)
        menu.addSeparator()
        delete_action = menu.addAction("Delete event")
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is delete_action:
            self._events = [e for e in self._events if e is not ev]
        else:
            new_cls = chosen.data()
            if new_cls != ev.class_id:
                new_ev = Event(t=ev.t, class_id=new_cls, score=ev.score)
                self._events = [new_ev if e is ev else e for e in self._events]
        self._rebuild_classes()
        self.update()
        self.events_changed.emit(list(self._events))

    def paintEvent(self, event) -> None:  # noqa: N802
        from PySide6.QtCore import QPointF, QRectF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Winamp LCD background — near-black green
        painter.fillRect(0, 0, w, h, QColor(0, 14, 0))

        if not self._events:
            painter.setPen(QColor(0, 100, 0))
            painter.drawText(
                0, 0, w, h, Qt.AlignmentFlag.AlignCenter,
                "RECORD A BEAT TO SEE EVENTS HERE",
            )
            return

        duration = self._bars * 4.0 * 60.0 / self._bpm
        timeline_w = max(1, w - self._LABEL_W - 12)
        top_pad = 12

        beat_count = self._bars * 4

        for row, cls in enumerate(self._classes):
            y = top_pad + row * self._ROW_HEIGHT
            # Alternating very-dark rows (LCD scanline effect)
            bg = QColor(0, 18, 0) if row % 2 == 0 else QColor(0, 12, 0)
            painter.fillRect(self._LABEL_W, y, timeline_w, self._ROW_HEIGHT, bg)

            # Class label in dim green monospace
            painter.setPen(QColor(0, 140, 0))
            painter.drawText(
                QRectF(4, y, self._LABEL_W - 8, self._ROW_HEIGHT),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                cls.replace("_", " ").upper(),
            )

            r, g, b = self._CLASS_COLORS.get(cls, (0, 180, 0))
            muted = cls in self._muted
            dim = 6 if muted else 1
            fill = QColor(r // dim, g // dim, b // dim)
            glow = QColor(r // (dim * 3), g // (dim * 3), b // (dim * 3))
            cy = y + self._ROW_HEIGHT / 2.0

            for ev in self._events:
                if ev.class_id != cls:
                    continue
                frac = ev.t / duration if duration > 0 else 0.0
                frac = max(0.0, min(1.0, frac))
                x = self._LABEL_W + frac * timeline_w
                # Outer glow ring
                painter.setBrush(glow)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(x, cy), self._DOT_R + 2, self._DOT_R + 2)
                # Bright centre dot
                painter.setBrush(fill)
                painter.drawEllipse(QPointF(x, cy), self._DOT_R, self._DOT_R)

        # Beat grid — dim green verticals, brighter on bar boundaries
        for b in range(beat_count + 1):
            x = self._LABEL_W + int(b / beat_count * timeline_w)
            on_bar = (b % 4 == 0)
            painter.setPen(QColor(0, 60, 0) if on_bar else QColor(0, 35, 0))
            painter.drawLine(x, top_pad, x, top_pad + len(self._classes) * self._ROW_HEIGHT)

    # ---- test helpers ----

    def event_count(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------
# CalibrationWizardDialog — per-class sample collection
# ---------------------------------------------------------------

class CalibrationWizardDialog(QDialog):
    """Step-by-step wizard: record N samples per class, then commit calibration.

    Uses sounddevice.rec() for simple 1-second blocking snippets so no
    Recorder bookkeeping is needed during calibration.
    """

    _SAMPLES_REQUIRED = 3
    _RECORD_SECONDS = 1.0
    _SAMPLE_RATE = 16_000

    def __init__(
        self,
        extractor,
        manager,
        classifier,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration Wizard")
        self.setMinimumWidth(400)

        self._extractor = extractor
        self._manager = manager
        self._classifier = classifier

        from voxkit.ui.calibration_flow import CalibrationFlow
        self._flow = CalibrationFlow(extractor, manager, classifier)
        self._classes: tuple[str, ...] = classifier.taxonomy.classes
        self._current_idx: int = 0
        self._counts: dict[str, int] = {c: 0 for c in self._classes}

        self._setup_ui()
        self._refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._step_label = QLabel()
        font = self._step_label.font()
        font.setPointSize(13)
        font.setBold(True)
        self._step_label.setFont(font)
        layout.addWidget(self._step_label)

        self._instruction_label = QLabel()
        self._instruction_label.setWordWrap(True)
        layout.addWidget(self._instruction_label)

        self._progress_label = QLabel()
        layout.addWidget(self._progress_label)

        # Device picker
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Input device:"))
        self._device_combo = QComboBox()
        self._populate_devices()
        dev_row.addWidget(self._device_combo, stretch=1)
        layout.addLayout(dev_row)

        self._record_btn = QPushButton(f"Record sample (1.5 s)")
        self._record_btn.clicked.connect(self._on_record)
        layout.addWidget(self._record_btn)

        self._error_label = QLabel()
        self._error_label.setStyleSheet("color: #e05050;")
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._next_btn = QPushButton("Next →")
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._on_next)
        btn_row.addWidget(self._next_btn)
        layout.addLayout(btn_row)

    def _populate_devices(self) -> None:
        try:
            from voxkit.audio.recorder import Recorder
            for dev in Recorder().list_devices():
                self._device_combo.addItem(dev.name, dev.id)
        except Exception:
            pass

    def _refresh(self) -> None:
        cls = self._classes[self._current_idx]
        n = self._counts[cls]
        total = len(self._classes)
        self._step_label.setText(
            f"Step {self._current_idx + 1} / {total}: {cls.replace('_', ' ')}"
        )
        self._instruction_label.setText(
            f"Click Record, then make ONE {cls.replace('_', ' ')} sound near the middle "
            f"of the {self._RECORD_SECONDS:.1f}-second clip.\n"
            f"Repeat {self._SAMPLES_REQUIRED} times (one sound per click)."
        )
        dots = "● " * n + "○ " * (self._SAMPLES_REQUIRED - n)
        self._progress_label.setText(f"Samples: {dots.strip()}")
        self._error_label.setText("")

        is_last = self._current_idx == total - 1
        self._next_btn.setEnabled(n >= self._SAMPLES_REQUIRED)
        self._next_btn.setText("Finish" if is_last else "Next →")

    def _on_record(self) -> None:
        from voxkit.audio.recorder import record_blocking

        device_id = self._device_combo.currentData()
        self._record_btn.setEnabled(False)
        self._record_btn.setText("Recording…")
        self._error_label.setText("")
        QApplication.processEvents()

        try:
            audio = record_blocking(
                self._RECORD_SECONDS,
                sample_rate=self._SAMPLE_RATE,
                device=int(device_id) if device_id is not None else None,
            )
        except Exception as exc:
            self._error_label.setText(f"Recording failed: {exc}")
            self._record_btn.setEnabled(True)
            self._record_btn.setText(f"Record sample ({self._RECORD_SECONDS:.0f} s)")
            return

        cls = self._classes[self._current_idx]
        try:
            self._flow.add_sample(cls, audio)
            self._counts[cls] += 1
        except Exception as exc:
            self._error_label.setText(f"Sample rejected: {exc}")
        finally:
            self._record_btn.setEnabled(True)
            self._record_btn.setText(f"Record sample ({self._RECORD_SECONDS:.0f} s)")
            self._refresh()

    def _on_next(self) -> None:
        is_last = self._current_idx == len(self._classes) - 1
        if is_last:
            try:
                import numpy as np
                cal_emb, cal_labels = self._flow._session.get_embeddings_and_labels()
                # No AVP pre-training in user-testing mode: use calibration samples
                # directly as training data. One synthetic subject per sample so
                # the LOSO split holds out only one sample rather than a whole class.
                subjects = np.arange(len(cal_labels))
                self._classifier.fit(cal_emb.astype(np.float32), cal_labels, subjects)
                self.accept()
            except Exception as exc:
                self._error_label.setText(f"Calibration failed: {exc}")
        else:
            self._current_idx += 1
            self._refresh()
