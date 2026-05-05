# SPDX-License-Identifier: GPL-3.0-or-later
"""Tier configuration for the eval harness (§7.10, Q85)."""

from __future__ import annotations

import sys
from pathlib import Path


class CanonicalTierMissing(Exception):
    """Raised when the canonical dataset has not been downloaded locally."""


_TIER_ORDER = ["synthetic", "minimum-reproducible", "canonical"]

_SYNTHETIC_PATH = Path(__file__).parent / "data" / "synthetic"
_MIN_REPRO_PATH = Path(__file__).parent / "data" / "minimum-reproducible"


def list_tiers() -> list[str]:
    """Return the ordered tier list (stable order required by T04)."""
    return list(_TIER_ORDER)


def get_tier_path(tier: str) -> Path:
    """Return the local Path for *tier*.

    Raises CanonicalTierMissing (with download instructions) when the
    canonical dataset is not present on this machine.
    """
    if tier == "synthetic":
        return _SYNTHETIC_PATH
    if tier == "minimum-reproducible":
        return _MIN_REPRO_PATH
    if tier == "canonical":
        raise CanonicalTierMissing(
            "Canonical tier dataset is not available locally. "
            "Please download or obtain it from the project data store."
        )
    raise ValueError(f"Unknown tier {tier!r}; known tiers: {_TIER_ORDER}")


def announce_tier(tier: str) -> None:
    """Print a banner for the given tier.

    WARNING banner goes to stderr (keeps stdout clean for JSON pipelines).
    Tier-name mentions go to stdout so downstream tools can detect them.
    """
    if tier == "synthetic":
        print(
            f"WARNING: Running on synthetic tier. "
            f"Results may not reflect real-world performance. "
            f"Prefer minimum-reproducible or canonical tier for release evaluation.",
            file=sys.stderr,
        )
        print(
            f"Using synthetic tier. "
            f"Higher-quality tiers available: minimum-reproducible, canonical.",
            file=sys.stdout,
        )
