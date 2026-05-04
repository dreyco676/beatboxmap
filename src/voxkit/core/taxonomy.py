# SPDX-License-Identifier: GPL-3.0-or-later
"""TaxonomyConfig: percussion class taxonomy and GM MIDI mapping (Q66)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TaxonomyConfig:
    classes: tuple[str, ...]
    midi_mapping: Mapping[str, int]
    unknown_class_id: str = "unknown"

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("classes must be non-empty")
        missing = [c for c in self.classes if c not in self.midi_mapping]
        if missing:
            raise ValueError(
                f"midi_mapping missing entries for classes: {missing}"
            )
        if self.unknown_class_id in self.classes:
            raise ValueError(
                f"unknown_class_id '{self.unknown_class_id}' must not appear in classes"
            )

    @classmethod
    def default_v1_0(cls) -> "TaxonomyConfig":
        return cls(
            classes=("kick", "snare", "closed_hat", "open_hat"),
            midi_mapping={"kick": 36, "snare": 38, "closed_hat": 42, "open_hat": 46},
        )
