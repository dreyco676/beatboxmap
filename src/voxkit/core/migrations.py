# SPDX-License-Identifier: GPL-3.0-or-later
"""Migration table and walk logic for .voxkit bundle format (Q78, §7.11)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class MigrationPathNotFound(Exception):
    """Raised when walk_migrations cannot reach to_version from from_version."""


class MigrationStepFailed(Exception):
    """Raised when a registered migrator raises; wraps original with step context."""


# ---------------------------------------------------------------
# Migration walk
# ---------------------------------------------------------------

Migrator = Callable[[dict[str, Any]], dict[str, Any]]
MigrationTable = dict[tuple[str, str], Migrator]


def walk_migrations(
    data: dict[str, Any],
    *,
    from_version: str,
    to_version: str,
    table: MigrationTable,
) -> dict[str, Any]:
    """Walk the migration table from from_version to to_version.

    Raises MigrationPathNotFound if the path cannot be completed.
    Raises MigrationStepFailed (chaining original) when a migrator raises.
    """
    current = from_version
    result = data

    while current != to_version:
        step_key = _find_step(current, table)
        if step_key is None:
            raise MigrationPathNotFound(
                f"No migration path from '{current}' to '{to_version}'. "
                f"Available steps: {list(table.keys())}"
            )
        from_v, to_v = step_key
        migrator = table[step_key]
        try:
            result = migrator(result)
        except Exception as exc:
            raise MigrationStepFailed(
                f"Migration step {from_v!r} → {to_v!r} failed: {exc}"
            ) from exc
        current = to_v

    return result


def _find_step(
    from_version: str, table: MigrationTable
) -> tuple[str, str] | None:
    for key in table:
        if key[0] == from_version:
            return key
    return None


# ---------------------------------------------------------------
# Registered migrators
# ---------------------------------------------------------------

def migrate_0_10_to_0_11(data: dict[str, Any]) -> dict[str, Any]:
    """v0.10 → v0.11: data no-op; stamp version."""
    manifest = dict(data.get("manifest", {}))
    manifest["voxkit_format_version"] = "0.11"
    return {**data, "manifest": manifest}


def migrate_0_9_to_0_10_cholesky(data: dict[str, Any]) -> dict[str, Any]:
    """v0.9 → v0.10: convert pooled_inv_covariance to lower-triangular Cholesky."""
    from scipy.linalg import cholesky

    manifest = dict(data.get("manifest", {}))
    manifest["voxkit_format_version"] = "0.10"
    result = {**data, "manifest": manifest}

    raw_mah = data.get("mahalanobis_full_dim")
    if raw_mah is None:
        return result

    mah = dict(raw_mah)
    if "pooled_inv_covariance_full_dim" not in mah:
        return result

    inv_cov = np.array(mah.pop("pooled_inv_covariance_full_dim"))
    cov = np.linalg.inv(inv_cov)
    L = cholesky(cov, lower=True)
    mah["pooled_cov_cholesky_full_dim"] = L.tolist()
    result["mahalanobis_full_dim"] = mah
    return result


def migrate_0_6_to_0_7(data: dict[str, Any]) -> dict[str, Any]:
    """v0.6 → v0.7: drop PlattCalibration; set banner flag for Editor."""
    manifest = dict(data.get("manifest", {}))
    manifest["voxkit_format_version"] = "0.7"
    return {
        **data,
        "manifest": manifest,
        "output_calibration": None,
        "output_calibration_migration_banner": True,
    }


def migrate_0_8_to_0_9(data: dict[str, Any]) -> dict[str, Any]:
    """v0.8 → v0.9: discard PCA-based Mahalanobis when pca_matrix_present;
    set banner flag so Editor renders the non-dismissable recalibration banner."""
    manifest = dict(data.get("manifest", {}))
    manifest["voxkit_format_version"] = "0.9"
    result = {**data, "manifest": manifest}
    if data.get("pca_matrix_present"):
        result["mahalanobis_full_dim"] = None
        result["pca_mahalanobis_migration_banner"] = True
    return result


def migrate_0_4_to_0_9(data: dict[str, Any]) -> dict[str, Any]:
    """v0.4 → v0.9: no structural changes; stamp version."""
    manifest = dict(data.get("manifest", {}))
    manifest["voxkit_format_version"] = "0.9"
    return {**data, "manifest": manifest}


MIGRATIONS: MigrationTable = {
    ("0.4", "0.9"): migrate_0_4_to_0_9,
    ("0.9", "0.10"): migrate_0_9_to_0_10_cholesky,
    ("0.10", "0.11"): migrate_0_10_to_0_11,
}
