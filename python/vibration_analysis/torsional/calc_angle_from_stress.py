"""Angle-of-twist utilities from torsional shear stress inputs.

This module mirrors the MATLAB ``CalcAngleFromStress`` behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def calc_angle_from_stress(
    d_ro: float,
    d_L: float,
    d_taumax: Any,
    d_G: float,
) -> np.ndarray:
    """Return shaft twist angle in radians from shear stress.

    The implemented relationship is:
        phi = (L * tau_max) / (G * r_o)

    Args:
        d_ro: Outer radius of circular shaft section (positive scalar).
        d_L: Shaft section length (positive scalar).
        d_taumax: Torsional shear stress, scalar or array-like.
        d_G: Shear modulus of elasticity (positive scalar).

    Returns:
        A column-shaped ``numpy.ndarray`` of twist angle(s) in radians.
    """

    for value, name in ((d_ro, "d_ro"), (d_L, "d_L"), (d_G, "d_G")):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive scalar numeric quantity")

    tau_vector = np.asarray(d_taumax, dtype=float).reshape(-1, 1)
    if tau_vector.size == 0 or np.any(~np.isfinite(tau_vector)):
        raise ValueError("d_taumax must contain finite numeric value(s)")

    return (d_L * tau_vector) / (d_G * d_ro)
