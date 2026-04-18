"""Forced torsional response utilities.

This module provides a Python translation of MATLAB CalcForcedTorsResp.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .calc_free_free_tors_resp import calc_free_free_tors_resp


def calc_forced_tors_resp(
    d_MoIp: Any,
    d_kt: Any,
    d_len: Any | None = None,
    d_damp_int: Any | None = None,
    d_damp_ext: Any | None = None,
    d_obs: Any | None = None,
    i_station_skip: int = 1,
    y_max_tick_input: float = -1.0,
    *,
    d_kt_ext: Any | None = None,
    b_supr_degen: bool = True,
    b_no_plots: bool = False,
    str_plot_file: str = "",
    d_gear_ratio: Any | None = None,
    i_mode_max: int = 4,
    d_u_MoIp: Any | None = None,
    d_u_kt: Any | None = None,
    d_u_kt_ext: Any | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, Any, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute forced torsional model and free-free response summary.

    Returns a tuple in MATLAB output order:
    (h_plot, d_eigvec, d_ft, h_mass_elastic, ss_rotor,
     d_MoIp, d_kt, d_damp_int, d_damp_ext, d_obs)
    """

    d_moip_in = np.asarray(d_MoIp, dtype=float).reshape(-1)
    d_kt_in = np.asarray(d_kt, dtype=float).reshape(-1)
    if d_moip_in.size == 0 or d_kt_in.size == 0:
        raise ValueError("d_MoIp and d_kt must be non-empty vectors")
    if d_moip_in.size != d_kt_in.size:
        raise ValueError("d_MoIp and d_kt must have the same length")

    n = d_moip_in.size
    d_len_vec = np.ones(n, dtype=float) if d_len is None else np.asarray(d_len, dtype=float).reshape(-1)
    d_damp_int_vec = np.zeros(n, dtype=float) if d_damp_int is None else np.asarray(d_damp_int, dtype=float).reshape(-1)
    d_damp_ext_vec = np.zeros(n, dtype=float) if d_damp_ext is None else np.asarray(d_damp_ext, dtype=float).reshape(-1)
    d_obs_vec = np.ones(2 * n, dtype=float) if d_obs is None else np.asarray(d_obs, dtype=float).reshape(-1)

    if d_len_vec.size != n:
        raise ValueError("d_len must have the same length as d_MoIp")
    if d_damp_int_vec.size != n or d_damp_ext_vec.size != n:
        raise ValueError("d_damp_int and d_damp_ext must have same length as d_MoIp")
    if d_obs_vec.size != (2 * n):
        raise ValueError("Not enough observation values in O")

    free_free_out = calc_free_free_tors_resp(
        d_moip_in,
        d_kt_in,
        d_len_vec,
        i_station_skip,
        y_max_tick_input,
        d_kt_ext=d_kt_ext,
        b_supr_degen=b_supr_degen,
        b_no_plots=b_no_plots,
        str_plot_file=str_plot_file,
        d_gear_ratio=d_gear_ratio,
        i_mode_max=i_mode_max,
        d_damp_int=d_damp_int_vec,
        d_damp_ext=d_damp_ext_vec,
        d_obs=d_obs_vec,
        d_u_MoIp=d_u_MoIp,
        d_u_kt=d_u_kt,
        d_u_kt_ext=d_u_kt_ext,
    )

    (
        h_plot,
        d_eigvec,
        d_ft,
        h_mass_elastic,
        _ss_from_free,
        d_moip,
        d_kt_vec,
        d_damp_int_vec,
        d_damp_ext_vec,
        d_obs_vec,
    ) = free_free_out

    i_pmm = d_moip.size
    if d_obs_vec.size != (2 * i_pmm):
        raise ValueError("Not enough observation values in O")

    d_kt_ext_vec = np.zeros(i_pmm, dtype=float) if d_kt_ext is None else np.asarray(d_kt_ext, dtype=float).reshape(-1)
    if d_kt_ext_vec.size != i_pmm:
        raise ValueError("d_kt_ext must have same length as d_MoIp")

    a_mat = np.zeros((2 * i_pmm, 2 * i_pmm), dtype=float)
    b_mat = np.zeros((2 * i_pmm, i_pmm), dtype=float)
    c_mat = np.eye(2 * i_pmm, dtype=float)

    # Upper-right identity submatrix.
    for idx_row in range(i_pmm):
        a_mat[idx_row, i_pmm + idx_row] = 1.0

    # Diagonal mass-stiffness and mass-damping terms.
    idx_row = i_pmm
    for idx in range(i_pmm):
        if idx == 0:
            a_mat[idx_row, idx] = -(d_kt_vec[idx] + d_kt_ext_vec[idx]) / d_moip[idx]
            a_mat[idx_row, idx + i_pmm] = -(d_damp_int_vec[idx] + d_damp_ext_vec[idx]) / d_moip[idx]
        elif idx == (i_pmm - 1):
            a_mat[idx_row, idx] = -(d_kt_vec[idx - 1] + d_kt_ext_vec[idx]) / d_moip[idx]
            a_mat[idx_row, idx + i_pmm] = -(d_damp_int_vec[idx - 1] + d_damp_ext_vec[idx]) / d_moip[idx]
        else:
            a_mat[idx_row, idx] = -(d_kt_vec[idx - 1] + d_kt_vec[idx] + d_kt_ext_vec[idx]) / d_moip[idx]
            a_mat[idx_row, idx + i_pmm] = -(
                d_damp_int_vec[idx - 1] + d_damp_ext_vec[idx] + d_damp_int_vec[idx]
            ) / d_moip[idx]
        idx_row += 1

    # Off-diagonal mass-stiffness and mass-damping coupling terms.
    idx_row = i_pmm
    for idx in range(1, i_pmm):
        a_mat[idx_row, idx] = d_kt_vec[idx - 1] / d_moip[idx - 1]
        a_mat[idx_row + 1, idx - 1] = d_kt_vec[idx - 1] / d_moip[idx]

        a_mat[idx_row, idx + i_pmm] = d_damp_int_vec[idx - 1] / d_moip[idx - 1]
        a_mat[idx_row + 1, idx - 1 + i_pmm] = d_damp_int_vec[idx - 1] / d_moip[idx]
        idx_row += 1

    for idx in range(i_pmm):
        b_mat[idx + i_pmm, idx] = 1.0 / d_moip[idx]

    c_mat = c_mat[d_obs_vec != 0, :]
    d_mat = np.zeros((c_mat.shape[0], i_pmm), dtype=float)

    ss_rotor = {
        "A": a_mat,
        "B": b_mat,
        "C": c_mat,
        "D": d_mat,
    }

    return (
        h_plot,
        d_eigvec,
        d_ft,
        h_mass_elastic,
        ss_rotor,
        d_moip,
        d_kt_vec,
        d_damp_int_vec,
        d_damp_ext_vec,
        d_obs_vec,
    )
