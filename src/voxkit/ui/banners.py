# SPDX-License-Identifier: GPL-3.0-or-later
"""Banner state machines for the Editor UI (Q79, v0.10 item 17)."""

from __future__ import annotations


class BleedQualityBanner:
    """Shows click-bleed quality feedback; hidden when attenuation >= 20 dB or overridden."""

    def __init__(self) -> None:
        self.is_visible: bool = False
        self.color: str | None = None
        self.text: str = ""

    def update(self, attenuation_db: float, override: bool) -> None:
        if override or attenuation_db >= 20.0:
            self.is_visible = False
            self.color = None
            self.text = ""
        else:
            self.is_visible = True
            self.color = "yellow" if attenuation_db >= 10.0 else "red"
            self.text = f"Bleed attenuation: {attenuation_db:.1f} dB"


class MigrationBanner:
    """Persistent migration notice (v0.10 item 17); dismissed only by calibration commit."""

    def __init__(self, migration_required: bool) -> None:
        self.is_visible: bool = migration_required

    def get_action_labels(self) -> list[str]:
        return ["Recalibrate now"]

    def attempt_dismiss(self) -> None:
        pass  # intentionally ignored — banner persists until calibration runs

    def on_calibration_committed(self) -> None:
        self.is_visible = False

    def serialize(self) -> dict:
        return {"migration_required": self.is_visible}

    @classmethod
    def deserialize(cls, state: dict) -> "MigrationBanner":
        return cls(migration_required=state["migration_required"])
