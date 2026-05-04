# SPDX-License-Identifier: GPL-3.0-or-later
"""
pytest configuration for the VoxKit test suite.

Implements the dataset-tier semantics from TDD_README.md:
  pytest -m "not slow"                              # synthetic tier (CI default)
  pytest -m "not slow" --dataset=minimum-reproducible
  pytest --dataset=canonical                        # release tier

Also implements the dataset_required(name) marker. A test marked
@pytest.mark.dataset_required("AVP") is skipped unless either:
  - --dataset=canonical is set (full tier — every dataset is required),
  - --dataset=minimum-reproducible is set AND the named dataset is present
    in the local dataset cache (resolved via voxkit.eval.tiers if available;
    falls back to checking $VOXKIT_DATASETS_DIR).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pytest


# ---------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------

_VALID_TIERS = ("synthetic", "minimum-reproducible", "canonical")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dataset",
        action="store",
        default="synthetic",
        choices=_VALID_TIERS,
        help=(
            "Dataset tier to run against. 'synthetic' (default) runs only "
            "in-repo synthetic data; 'minimum-reproducible' resolves the "
            "project-hosted reduced dataset; 'canonical' requires the full "
            "release-gate dataset. See spec §7.10 / Q85."
        ),
    )


# ---------------------------------------------------------------
# Marker registration is in pyproject.toml [tool.pytest.ini_options]
# but we also need to *interpret* dataset_required at collection time.
# ---------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    tier = config.getoption("--dataset")
    available_datasets = _discover_available_datasets()

    skip_for_tier = pytest.mark.skip(
        reason=f"dataset not available for --dataset={tier}"
    )

    for item in items:
        marker = item.get_closest_marker("dataset_required")
        if marker is None:
            continue
        if not marker.args:
            # Misuse: dataset_required with no name. Make it loud.
            item.add_marker(
                pytest.mark.skip(reason="dataset_required marker missing name arg")
            )
            continue

        dataset_name = marker.args[0]
        if tier == "canonical":
            # Canonical tier assumes everything is present; let the test run
            # and surface a real error if it isn't (release-gate semantics).
            continue
        if tier == "minimum-reproducible" and dataset_name in available_datasets:
            continue
        if tier == "synthetic":
            # Synthetic-tier CI never has external datasets; skip cleanly.
            item.add_marker(skip_for_tier)
            continue
        # minimum-reproducible tier but the named dataset isn't there.
        item.add_marker(skip_for_tier)


def _discover_available_datasets() -> set[str]:
    """Resolve which named datasets are present locally.

    Preference order:
      1. voxkit.eval.tiers.available_datasets() if importable (the in-tree
         source of truth once Component 12 lands),
      2. $VOXKIT_DATASETS_DIR/<name>/ existence check (lightweight CI shim).
    """
    try:
        from voxkit.eval.tiers import available_datasets  # type: ignore
        return set(available_datasets())
    except Exception:
        pass

    root = os.environ.get("VOXKIT_DATASETS_DIR")
    if not root:
        return set()
    base = Path(root)
    if not base.is_dir():
        return set()
    return {p.name for p in base.iterdir() if p.is_dir()}
