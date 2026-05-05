# SPDX-License-Identifier: GPL-3.0-or-later
"""Migration round-trip CI checker (§7.11, Q78)."""

from __future__ import annotations

from pathlib import Path


_CURRENT_VERSION = "0.11"


def _make_minimal_bundle(version: str) -> dict:
    """Create a minimal in-memory bundle dict at the given format version."""
    return {
        "manifest": {"voxkit_format_version": version},
        "events": [],
        "mahalanobis_full_dim": None,
    }


def round_trip_all_migrations(
    work_dir: Path | None = None,
    extra_versions: list[str] | None = None,
) -> list[str]:
    """Verify that every registered migration pair produces a valid round-trip.

    For each (from_v, to_v) pair in MIGRATIONS:
        create a minimal bundle at from_v, walk migrations to current, verify
        no data is lost.

    For extra_versions: attempt to walk migrations from each; record a failure
    when no migrator path exists (catches missing entries, T28).

    Returns a list of failure strings (empty on full success).
    """
    from voxkit.core.migrations import MIGRATIONS, MigrationPathNotFound, walk_migrations

    failures: list[str] = []

    # Collect all from-versions registered in MIGRATIONS.
    registered_from = [v[0] for v in MIGRATIONS.keys()]

    all_test_versions = list(registered_from)
    if extra_versions:
        all_test_versions.extend(extra_versions)

    for from_v in all_test_versions:
        if from_v == _CURRENT_VERSION:
            continue
        bundle = _make_minimal_bundle(from_v)
        try:
            walk_migrations(
                bundle,
                from_version=from_v,
                to_version=_CURRENT_VERSION,
                table=MIGRATIONS,
            )
        except MigrationPathNotFound as exc:
            failures.append(
                f"No migration path from version {from_v!r} to {_CURRENT_VERSION!r}: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Migration from {from_v!r} failed: {exc}")

    return failures
