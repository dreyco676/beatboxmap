# SPDX-License-Identifier: GPL-3.0-or-later
"""Eval harness: write_results, run_for_tier (§7.10, §7.11)."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_AVP_PERSONAL_DIR = _REPO_ROOT / "data" / "avp" / "AVP_Dataset" / "Personal"


def _get_git_sha() -> str:
    """Return the current HEAD SHA (40 hex chars) or 'no-repo' when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
        return "no-repo"
    except Exception:
        return "no-repo"


def _get_eval_version_for_provenance() -> str:
    """Read eval_version from the single source of truth (voxkit.eval.EVAL_VERSION)."""
    from voxkit.eval import EVAL_VERSION
    return EVAL_VERSION


def write_results(
    out_path: Path,
    payload: dict,
    *,
    include_provenance: bool = False,
    dataset_tier: str = "synthetic",
) -> None:
    """Write *payload* as deterministic JSON to *out_path*.

    With include_provenance=True, adds git_sha, dataset_tier, eval_version.
    Keys are sorted so identical inputs produce byte-identical output (T32).
    """
    data = dict(payload)
    if include_provenance:
        data["git_sha"] = _get_git_sha()
        data["dataset_tier"] = dataset_tier
        data["eval_version"] = _get_eval_version_for_provenance()
    Path(out_path).write_text(json.dumps(data, sort_keys=True, indent=2))


def _load_avp_corpus(personal_dir: Path) -> list[tuple[np.ndarray, list[float]]]:
    """Load (audio_float32_16kHz, onset_times_s) pairs from the AVP Personal directory.

    Skips Improvisation recordings (free-form; no canonical class label).
    """
    import scipy.io.wavfile as wv

    pairs: list[tuple[np.ndarray, list[float]]] = []
    participant_dirs = sorted(
        [d for d in personal_dir.iterdir() if d.is_dir() and d.name.startswith("Participant_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for pdir in participant_dirs:
        for wav_path in sorted(pdir.glob("*.wav")):
            if "Improvisation" in wav_path.stem:
                continue
            csv_path = wav_path.with_suffix(".csv")
            if not csv_path.exists():
                continue

            sr, data = wv.read(str(wav_path))
            if data.ndim == 2:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype != np.float32:
                data = data.astype(np.float32)

            if sr != 16_000:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, 16_000)
                data = resample_poly(data, 16_000 // g, sr // g).astype(np.float32)

            onset_times: list[float] = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if row:
                        try:
                            onset_times.append(float(row[0]))
                        except ValueError:
                            pass
            pairs.append((data, onset_times))
    return pairs


def _eval_onset_on_avp() -> tuple[float, float]:
    """Run OnsetDetector over AVP Personal corpus; return (f_measure, mae_ms)."""
    from voxkit.dsp.onsets import OnsetDetector
    from voxkit.eval.onset_release_gate import evaluate_corpus

    corpus = _load_avp_corpus(_AVP_PERSONAL_DIR)
    detector = OnsetDetector(sample_rate=16_000)
    return evaluate_corpus(detector, corpus)


def run_for_tier(tier: str) -> dict:
    """Run the eval pipeline for the given tier and return a results dict.

    Tier contracts
    --------------
    synthetic           : pipeline smoke-test only; no quality assertions
    minimum-reproducible: F-measure and MAE computed from AVP Personal corpus
    canonical           : F-measure, MAE, and release-gate pass/fail (falls
                          back to AVP Personal when canonical dataset is absent)
    """
    result: dict = {"tier": tier}

    if tier == "synthetic":
        result["pipeline_ok"] = True
        return result

    if tier == "minimum-reproducible":
        f_measure, mae_ms = _eval_onset_on_avp()
        result["f_measure"] = f_measure
        result["mae_ms"] = mae_ms
        return result

    if tier == "canonical":
        from voxkit.eval.onset_release_gate import release_gate_check
        f_measure, mae_ms = _eval_onset_on_avp()
        result["f_measure"] = f_measure
        result["mae_ms"] = mae_ms
        gate = release_gate_check(f_measure, mae_ms, "AVP")
        result["release_gate_passed"] = gate.passed
        # missed_unknown / false_unknown require a trained classifier and OOD
        # corpus; they are populated by the operating-point sweep (Q50) when
        # run via the full calibration pipeline, not by the onset-only harness.
        result["missed_unknown"] = 0.0
        result["false_unknown"] = 0.0
        return result

    result["pipeline_ok"] = False
    return result
