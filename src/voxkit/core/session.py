# SPDX-License-Identifier: GPL-3.0-or-later
"""Session, MahalanobisFullDim, save/load — Component 1 (§11, Q66-Q68, Q78)."""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voxkit.core.taxonomy import TaxonomyConfig


# ---------------------------------------------------------------
# MahalanobisFullDim (Q68)
# ---------------------------------------------------------------

@dataclass
class MahalanobisFullDim:
    class_centroids: np.ndarray
    pooled_cov_cholesky: np.ndarray
    distance_thresholds: np.ndarray

    def __post_init__(self) -> None:
        L = self.pooled_cov_cholesky
        if not np.allclose(np.triu(L, k=1), 0):
            raise ValueError("pooled_cov_cholesky must be lower-triangular")
        D = L.shape[0]
        if L.shape != (D, D):
            raise ValueError(f"pooled_cov_cholesky dim mismatch: expected ({D},{D})")
        n_classes = self.class_centroids.shape[0]
        if self.distance_thresholds.shape[0] != n_classes:
            raise ValueError(
                f"class_centroids first dim ({n_classes}) must equal "
                f"distance_thresholds shape[0] ({self.distance_thresholds.shape[0]})"
            )
        if self.class_centroids.shape[1] != D:
            raise ValueError(
                f"centroid feature dim ({self.class_centroids.shape[1]}) "
                f"must equal Cholesky dim ({D})"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MahalanobisFullDim):
            return NotImplemented
        return (
            np.array_equal(self.class_centroids, other.class_centroids)
            and np.array_equal(self.pooled_cov_cholesky, other.pooled_cov_cholesky)
            and np.array_equal(self.distance_thresholds, other.distance_thresholds)
        )


# ---------------------------------------------------------------
# Session data types
# ---------------------------------------------------------------

@dataclass(frozen=True)
class TimeSignature:
    numerator: int
    denominator: int


@dataclass(frozen=True)
class Event:
    t: float
    class_id: str
    score: float


@dataclass
class Session:
    bpm: float
    time_signature: TimeSignature
    bars: int
    sample_rate: int
    recording_sample_rate: int
    recording_audio_api: str
    audio: np.ndarray
    events: list[Event] = field(default_factory=list)
    dropped_buffer_count: int = 0
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig.default_v1_0)
    mahalanobis_full_dim: MahalanobisFullDim | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Session):
            return NotImplemented
        return (
            self.bpm == other.bpm
            and self.time_signature == other.time_signature
            and self.bars == other.bars
            and self.sample_rate == other.sample_rate
            and self.recording_sample_rate == other.recording_sample_rate
            and self.recording_audio_api == other.recording_audio_api
            and np.array_equal(self.audio, other.audio)
            and self.events == other.events
            and self.dropped_buffer_count == other.dropped_buffer_count
            and self.taxonomy == other.taxonomy
            and self.mahalanobis_full_dim == other.mahalanobis_full_dim
        )


# ---------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------

CURRENT_FORMAT_VERSION = "0.11"


def _parse_version(v: str) -> tuple[int, ...]:
    """Numeric semver parse — "0.10" > "0.9"."""
    return tuple(int(x) for x in v.split("."))


def _is_future_version(v: str) -> bool:
    return _parse_version(v) > _parse_version(CURRENT_FORMAT_VERSION)


# ---------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------

def _build_bundle_bytes(s: Session) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        manifest = {
            "voxkit_format_version": CURRENT_FORMAT_VERSION,
            "bpm": s.bpm,
            "bars": s.bars,
            "sample_rate": s.sample_rate,
            "recording_sample_rate": s.recording_sample_rate,
            "recording_audio_api": s.recording_audio_api,
            "time_signature": {
                "numerator": s.time_signature.numerator,
                "denominator": s.time_signature.denominator,
            },
            "dropped_buffer_count": s.dropped_buffer_count,
            "taxonomy": {
                "classes": list(s.taxonomy.classes),
                "midi_mapping": dict(s.taxonomy.midi_mapping),
                "unknown_class_id": s.taxonomy.unknown_class_id,
            },
        }
        if not isinstance(manifest["voxkit_format_version"], str):
            raise TypeError(
                f"voxkit_format_version must be str, got "
                f"{type(manifest['voxkit_format_version']).__name__}"
            )
        z.writestr("manifest.json", json.dumps(manifest))

        events_data = [
            {"t": e.t, "class_id": e.class_id, "score": e.score}
            for e in s.events
        ]
        z.writestr("events.json", json.dumps(events_data))

        audio_buf = io.BytesIO()
        np.save(audio_buf, s.audio)
        z.writestr("audio.npy", audio_buf.getvalue())

        if s.mahalanobis_full_dim is not None:
            m = s.mahalanobis_full_dim
            mah_data = {
                "class_centroids": m.class_centroids.tolist(),
                "pooled_cov_cholesky_full_dim": m.pooled_cov_cholesky.tolist(),
                "distance_thresholds": m.distance_thresholds.tolist(),
            }
            z.writestr("mahalanobis.json", json.dumps(mah_data))

    return buf.getvalue()


def _finalize_bundle_write(bundle_bytes: bytes, path: Path) -> None:
    """Write bundle_bytes to path atomically via a sibling temp file."""
    tmp = path.with_suffix(".voxkit.tmp")
    try:
        tmp.write_bytes(bundle_bytes)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def save_session(s: Session, path: Path) -> None:
    bundle_bytes = _build_bundle_bytes(s)
    _finalize_bundle_write(bundle_bytes, path)


def _load_raw_bundle(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        events_raw = json.loads(z.read("events.json")) if "events.json" in z.namelist() else []
        audio_buf = io.BytesIO(z.read("audio.npy"))
        audio = np.load(audio_buf)
        mah_data = (
            json.loads(z.read("mahalanobis.json"))
            if "mahalanobis.json" in z.namelist()
            else None
        )
    return {
        "manifest": manifest,
        "events": events_raw,
        "audio": audio,
        "mahalanobis": mah_data,
    }


def load_session(path: Path) -> "Session":
    from voxkit.core.manifest import ForwardCompatVersionError, ProjectManifest
    from voxkit.core.migrations import walk_migrations, MIGRATIONS

    with zipfile.ZipFile(path) as z:
        raw_manifest = json.loads(z.read("manifest.json"))

    pm = ProjectManifest.from_raw_dict(raw_manifest)
    version = pm.voxkit_format_version

    if _is_future_version(version):
        raise ForwardCompatVersionError(
            f"Bundle version '{version}' is newer than this build "
            f"(supports up to '{CURRENT_FORMAT_VERSION}'). "
            "Please update VoxKit."
        )

    # Walk migrations to current version.
    with zipfile.ZipFile(path) as z:
        events_raw = (
            json.loads(z.read("events.json"))
            if "events.json" in z.namelist()
            else []
        )
        audio_buf = io.BytesIO(z.read("audio.npy"))
        audio = np.load(audio_buf)
        mah_data = (
            json.loads(z.read("mahalanobis.json"))
            if "mahalanobis.json" in z.namelist()
            else None
        )

    bundle_dict: dict[str, Any] = {
        "manifest": raw_manifest,
        "events": events_raw,
        "mahalanobis_full_dim": mah_data,
    }

    migrated = walk_migrations(
        bundle_dict,
        from_version=version,
        to_version=CURRENT_FORMAT_VERSION,
        table=MIGRATIONS,
    )

    mig_manifest = migrated.get("manifest", raw_manifest)
    taxonomy = TaxonomyConfig.default_v1_0()
    if "taxonomy" in mig_manifest:
        t = mig_manifest["taxonomy"]
        taxonomy = TaxonomyConfig(
            classes=tuple(t["classes"]),
            midi_mapping=t["midi_mapping"],
            unknown_class_id=t.get("unknown_class_id", "unknown"),
        )

    events = [
        Event(t=e["t"], class_id=e["class_id"], score=e["score"])
        for e in migrated.get("events", [])
    ]

    mah_out = migrated.get("mahalanobis_full_dim")
    mah_obj: MahalanobisFullDim | None = None
    if mah_out is not None:
        key = "pooled_cov_cholesky_full_dim"
        mah_obj = MahalanobisFullDim(
            class_centroids=np.array(mah_out["class_centroids"]),
            pooled_cov_cholesky=np.array(mah_out[key]),
            distance_thresholds=np.array(mah_out["distance_thresholds"]),
        )

    ts_raw = mig_manifest.get("time_signature", {"numerator": 4, "denominator": 4})
    return Session(
        bpm=mig_manifest.get("bpm", 120.0),
        time_signature=TimeSignature(
            ts_raw["numerator"], ts_raw["denominator"]
        ),
        bars=mig_manifest.get("bars", 4),
        sample_rate=mig_manifest.get("sample_rate", 16_000),
        recording_sample_rate=mig_manifest.get("recording_sample_rate", 48_000),
        recording_audio_api=mig_manifest.get("recording_audio_api", "WASAPI"),
        audio=audio,
        events=events,
        dropped_buffer_count=mig_manifest.get("dropped_buffer_count", 0),
        taxonomy=taxonomy,
        mahalanobis_full_dim=mah_obj,
    )


# ---------------------------------------------------------------
# Test helpers (used by T22, T25)
# ---------------------------------------------------------------

def build_legacy_v0_4_bundle() -> bytes:
    """Build a minimal synthetic v0.4 bundle (no mahalanobis, no events)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        manifest = {
            # Intentionally missing voxkit_format_version to simulate v0.4
        }
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("events.json", json.dumps([]))
        audio_buf = io.BytesIO()
        np.save(audio_buf, np.zeros(16_000, dtype=np.float32))
        z.writestr("audio.npy", audio_buf.getvalue())
    return buf.getvalue()


def build_v0_10_bundle_with_known_distances(seed: int = 7):
    """Build a synthetic v0.10 bundle with a known Mahalanobis structure.

    Returns (bundle_bytes, ref_distances, ref_inputs) so T25 can verify
    that loading preserves Mahalanobis distances to within 1e-6.
    """
    from scipy.linalg import cholesky, solve_triangular

    rng = np.random.default_rng(seed)
    D = 8
    n_classes = 4
    A = rng.standard_normal((D, D))
    cov = A @ A.T + np.eye(D)
    L = cholesky(cov, lower=True)

    centroids = rng.standard_normal((n_classes, D))
    thresholds = np.ones(n_classes) * 5.0

    ref_inputs = [rng.standard_normal(D) for _ in range(5)]
    centroid0 = centroids[0]
    ref_distances = []
    for x in ref_inputs:
        diff = x - centroid0
        y = solve_triangular(L, diff, lower=True)
        ref_distances.append(float(y @ y))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        manifest = {
            "voxkit_format_version": "0.10",
            "bpm": 120.0,
            "bars": 4,
            "sample_rate": 16_000,
            "recording_sample_rate": 48_000,
            "recording_audio_api": "WASAPI",
            "time_signature": {"numerator": 4, "denominator": 4},
            "dropped_buffer_count": 0,
        }
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("events.json", json.dumps([]))
        audio_buf = io.BytesIO()
        np.save(audio_buf, np.zeros(16_000, dtype=np.float32))
        z.writestr("audio.npy", audio_buf.getvalue())
        mah_data = {
            "class_centroids": centroids.tolist(),
            "pooled_cov_cholesky_full_dim": L.tolist(),
            "distance_thresholds": thresholds.tolist(),
        }
        z.writestr("mahalanobis.json", json.dumps(mah_data))

    return buf.getvalue(), ref_distances, ref_inputs
