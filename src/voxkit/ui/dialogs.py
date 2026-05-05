# SPDX-License-Identifier: GPL-3.0-or-later
"""Dialog state machines for the Editor UI (Q73, Q81)."""

from __future__ import annotations


class RecordingProgressDialog:
    """State machine for the three-phase inference-progress dialog (Q73)."""

    phase_labels = (
        "Detecting onsets",
        "Extracting embeddings",
        "Classifying events",
    )

    def __init__(self, worker=None) -> None:
        self._worker = worker
        self._cancel_called = False

        self.current_progress: float = 0.0
        self.is_closed: bool = False
        self.completion_path: str | None = None
        self.error_shown: bool = False
        self.error_text: str | None = None

    def on_phase(self, phase: str) -> None:
        pass  # UI: update active phase label

    def on_progress(self, value: float) -> None:
        self.current_progress = value

    def click_cancel(self) -> None:
        if not self._cancel_called and self._worker is not None:
            self._worker.cancel()
            self._cancel_called = True

    def on_cancelled(self) -> None:
        self.is_closed = True

    def on_completed(self, events) -> None:
        self.is_closed = True
        self.completion_path = "editor"

    def on_failed(self, message: str) -> None:
        self.error_shown = True
        self.error_text = message


class CalibrationRejectedDialog:
    """Dialog shown when the overfit guard rejects a new calibration (Q81)."""

    def __init__(self, diagnostics: dict, telemetry=None) -> None:
        from voxkit.classifier.classifier import Q81_DIALOG_TEXT
        self.message = Q81_DIALOG_TEXT
        self._diagnostics = diagnostics
        self._telemetry = telemetry

    def action_labels(self) -> list[str]:
        return ["Try again", "Continue with previous"]

    def click(self, label: str) -> str:
        if self._telemetry is not None:
            self._telemetry.emit({
                "event": "calibration_overfit_guard_triggered",
                "details": self._diagnostics,
            })
        if label == "Try again":
            return "calibration_flow"
        return "close"
