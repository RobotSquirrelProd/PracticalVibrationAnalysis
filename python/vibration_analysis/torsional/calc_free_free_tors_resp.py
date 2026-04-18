"""Free-free torsional response utilities.

This module provides a Python translation of the MATLAB
CalcFreeFreeTorsResp function for the core eigen-analysis workflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _form_a(
    d_moip: np.ndarray,
    d_kt: np.ndarray,
    d_kt_ext: np.ndarray,
) -> np.ndarray:
    """Build the reduced A matrix used by the MATLAB implementation."""

    i_stat_nos = d_kt.size
    d_stiffness = np.zeros((i_stat_nos, i_stat_nos), dtype=float)

    for row in range(i_stat_nos):
        for col in range(i_stat_nos):
            if row == 0 and col == 0:
                d_stiffness[row, col] = d_kt[0] + d_kt_ext[0]
                continue

            if row == col:
                d_stiffness[row, col] = d_kt[row - 1] + d_kt[row] + d_kt_ext[row]

            if (row + 1) == col:
                d_stiffness[row, col] = -d_kt[row]
                d_stiffness[col, row] = -d_kt[row]

    d_moip_diag = np.diag(d_moip)
    return -np.linalg.solve(d_moip_diag, d_stiffness)


def calc_free_free_tors_resp(
    d_MoIp: Any,
    d_kt: Any,
    d_len: Any | None = None,
    i_station_skip: int = 1,
    y_max_tick_input: float = -1.0,
    *,
    d_kt_ext: Any | None = None,
    b_supr_degen: bool = True,
    b_no_plots: bool = False,
    str_plot_file: str = "",
    d_gear_ratio: Any | None = None,
    i_mode_max: int = 4,
    d_damp_int: Any | None = None,
    d_damp_ext: Any | None = None,
    d_obs: Any | None = None,
    b_supr_out: bool = False,
    d_u_MoIp: Any | None = None,
    d_u_kt: Any | None = None,
    d_u_kt_ext: Any | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, Any, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute free-free torsional modes from inertia and stiffness vectors.

    Returns a tuple that mirrors MATLAB output ordering:
    (h_mode_shapes, mat_eig_sorted, d_ft, h_mass_elastic, ss_rotor,
     d_MoIp, d_kt, d_damp_int, d_damp_ext, d_obs)

    The frequency vector d_ft is in cycles per minute (CPM).
    """

    d_moip = np.asarray(d_MoIp, dtype=float).reshape(-1)
    d_kt_vec = np.asarray(d_kt, dtype=float).reshape(-1)

    if d_moip.size == 0 or d_kt_vec.size == 0:
        raise ValueError("d_MoIp and d_kt must be non-empty vectors")
    if d_moip.size != d_kt_vec.size:
        raise ValueError("d_MoIp and d_kt must have the same length")
    if np.any(d_moip <= 0.0):
        raise ValueError("d_MoIp values must be strictly positive")

    n = d_kt_vec.size

    d_len_vec = np.ones(n, dtype=float) if d_len is None else np.asarray(d_len, dtype=float).reshape(-1)
    if d_len_vec.size != n:
        raise ValueError("d_len must have the same length as d_kt")

    if d_kt_ext is None:
        d_kt_ext_vec = np.zeros(n, dtype=float)
    else:
        d_kt_ext_vec = np.asarray(d_kt_ext, dtype=float).reshape(-1)
        if d_kt_ext_vec.size != n:
            raise ValueError("d_kt_ext must have the same length as d_kt")

    d_damp_int_vec = np.zeros(n, dtype=float) if d_damp_int is None else np.asarray(d_damp_int, dtype=float).reshape(-1)
    d_damp_ext_vec = np.zeros(n, dtype=float) if d_damp_ext is None else np.asarray(d_damp_ext, dtype=float).reshape(-1)
    if d_damp_int_vec.size != n or d_damp_ext_vec.size != n:
        raise ValueError("d_damp_int and d_damp_ext must match d_kt length")

    d_obs_vec = np.ones(2 * n, dtype=float) if d_obs is None else np.asarray(d_obs, dtype=float).reshape(-1)
    if d_obs_vec.size != (2 * n):
        raise ValueError("d_obs must have length 2 * len(d_kt)")

    if d_gear_ratio is not None:
        ratio = np.asarray(d_gear_ratio, dtype=float).reshape(-1)
        if np.any(ratio > 0.0):
            raise NotImplementedError("Positive d_gear_ratio handling is not yet implemented in Python version")

    # Uncertainty vectors are accepted for API compatibility with MATLAB.
    _ = (d_u_MoIp, d_u_kt, d_u_kt_ext)

    d_a = _form_a(d_moip, d_kt_vec, d_kt_ext_vec)
    eig_vals, eig_vecs = np.linalg.eig(d_a)

    d_ft_rad = np.sqrt(np.clip(-np.real(eig_vals), 0.0, None))
    order = np.argsort(d_ft_rad)
    d_ft_sorted = d_ft_rad[order]

    i_num_modes = min(n, int(i_mode_max))
    d_ft_plot = d_ft_sorted[:i_num_modes].copy()
    mat_eig_sorted = np.zeros((n, i_num_modes), dtype=float)

    # Keep MATLAB's frequency-to-eigenvector matching approach.
    for idx_mode in range(i_num_modes):
        freq_target = d_ft_sorted[idx_mode]
        for idx_station in range(n):
            if np.isclose(d_ft_rad[idx_station], freq_target, atol=1e-12, rtol=0.0):
                vec = np.real(eig_vecs[:, idx_station])
                vmin = float(np.min(vec))
                vmax = float(np.max(vec))
                delta = vmax - vmin
                slope = 1.0 if abs(delta) <= 1e-6 else (2.0 / delta)
                mat_eig_sorted[:, idx_mode] = (vec - vmin) * slope - 1.0
                break

    if b_supr_degen and d_ft_plot.size > 0:
        d_ft_plot = d_ft_plot[1:]
        mat_eig_sorted = mat_eig_sorted[:, 1:]

    # Convert from rad/s to CPM.
    d_ft_cpm = d_ft_plot * (30.0 / np.pi)

    d_len_sum = np.cumsum(d_len_vec)
    h_mass_elastic = None
    h_mode_shapes = None
    if not b_no_plots:
        h_mass_elastic = {
            "name": "Schematic",
            "station_positions": d_len_sum.copy(),
            "y_max_tick_input": float(y_max_tick_input),
            "i_station_skip": int(i_station_skip),
            "plot_file": str_plot_file,
        }
        h_mode_shapes = {
            "name": "Mode Shape",
            "station_positions": d_len_sum.copy(),
            "mode_shapes": mat_eig_sorted.copy(),
            "frequencies_cpm": d_ft_cpm.copy(),
        }

    # Lightweight state-space style payload matching MATLAB test intent.
    zero_n = np.zeros((n, n), dtype=float)
    ident_n = np.eye(n, dtype=float)
    m_inv = np.diag(1.0 / d_moip)
    stiffness = -np.diag(d_moip) @ d_a
    ss_rotor = {
        "A": np.block([[zero_n, ident_n], [-m_inv @ stiffness, zero_n]]),
        "B": np.zeros((2 * n, n), dtype=float),
        "C": np.eye(2 * n, dtype=float),
        "D": np.zeros((2 * n, n), dtype=float),
    }

    return (
        h_mode_shapes,
        mat_eig_sorted,
        d_ft_cpm,
        h_mass_elastic,
        ss_rotor,
        d_moip,
        d_kt_vec,
        d_damp_int_vec,
        d_damp_ext_vec,
        d_obs_vec,
    )
