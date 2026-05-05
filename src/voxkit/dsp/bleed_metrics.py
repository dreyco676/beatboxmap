# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-subtraction click residual ratio metric (Q79)."""

from __future__ import annotations

import math

import numpy as np


class ZeroCalibrationEnergy(Exception):
    """Raised when the calibration audio is silent (RMS ≈ 0)."""


def residual_ratio_db(cleaned: np.ndarray, calibration: np.ndarray) -> float:
    """Return 20·log10(rms(calibration) / rms(cleaned)) in dB.

    More positive = better attenuation.
    Raises ZeroCalibrationEnergy when the calibration segment is silent.
    """
    cal_rms = float(np.sqrt(np.mean(calibration.astype(np.float64) ** 2)))
    if cal_rms < 1e-12:
        raise ZeroCalibrationEnergy(
            "Calibration audio is silent (RMS ≈ 0); re-record calibration"
        )
    clean_rms = max(
        float(np.sqrt(np.mean(cleaned.astype(np.float64) ** 2))),
        1e-30,
    )
    return 20.0 * math.log10(cal_rms / clean_rms)
