# SPDX-License-Identifier: GPL-3.0-or-later
"""LocalDiagnosticSink + build_event per Q61."""

from __future__ import annotations

import datetime
import json
from pathlib import Path


def default_diagnostic_path() -> Path:
    return Path.home() / ".voxkit" / "diagnostics" / "voxkit_diag.jsonl"


def build_event(event: str, details: dict) -> dict:
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event": event,
        "details": details,
    }


class LocalDiagnosticSink:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._file = None
        self.events: list[dict] = []
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8")

    def emit(self, event: dict) -> None:
        self.events.append(event)
        if self._file is not None:
            self._file.write(json.dumps(event) + "\n")

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
