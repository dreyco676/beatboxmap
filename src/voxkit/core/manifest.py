# SPDX-License-Identifier: GPL-3.0-or-later
"""ProjectManifest + ForwardCompatVersionError (Q78)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ForwardCompatVersionError(Exception):
    """Raised when a bundle's format version is newer than this build."""


@dataclass(frozen=True)
class ProjectManifest:
    voxkit_format_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.voxkit_format_version, str):
            raise TypeError(
                f"voxkit_format_version must be str, "
                f"got {type(self.voxkit_format_version).__name__}"
            )

    @classmethod
    def from_raw_dict(cls, raw: dict[str, Any]) -> "ProjectManifest":
        version = raw.get("voxkit_format_version", "0.4")
        return cls(voxkit_format_version=version)
