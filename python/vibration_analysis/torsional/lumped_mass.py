"""Lumped-mass torsional vibration starter module.

Example:
    >>> import numpy as np
    >>> from vibration_analysis.torsional.lumped_mass import ShaftElement, natural_frequencies
    >>> inertias = np.array([0.10, 0.20])
    >>> shafts = [ShaftElement(0, 1, 1200.0)]
    >>> freqs_hz, modes = natural_frequencies(inertias, shafts)
    >>> freqs_hz.shape[0] >= 1
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ShaftElement:
    """Shaft coupling between two inertia stations.

    Attributes:
        node_i: First station index (0-based).
        node_j: Second station index (0-based).
        stiffness: Torsional stiffness in N*m/rad.
    """

    node_i: int
    node_j: int
    stiffness: float


def build_torsional_matrices(
    inertias: np.ndarray,
    shafts: Sequence[ShaftElement],
) -> tuple[np.ndarray, np.ndarray]:
    """Build mass (M) and stiffness (K) matrices for a torsional system.

    Args:
        inertias: 1D array of station inertias [kg*m^2].
        shafts: Sequence of shaft elements coupling station pairs.

    Returns:
        A tuple ``(M, K)`` where ``M`` and ``K`` are square ``(n, n)`` matrices.

    Raises:
        ValueError: If inertias or shaft definitions are invalid.
    """

    inertias = np.asarray(inertias, dtype=float)
    if inertias.ndim != 1 or inertias.size == 0:
        raise ValueError("inertias must be a non-empty 1D array")
    if np.any(inertias <= 0.0):
        raise ValueError("all inertias must be positive")

    n = inertias.size
    m_matrix = np.diag(inertias)
    k_matrix = np.zeros((n, n), dtype=float)

    for shaft in shafts:
        i = shaft.node_i
        j = shaft.node_j
        k = float(shaft.stiffness)
        if i == j:
            raise ValueError("shaft node_i and node_j must be different")
        if i < 0 or j < 0 or i >= n or j >= n:
            raise ValueError("shaft node indices must be within inertia array bounds")
        if k <= 0.0:
            raise ValueError("shaft stiffness must be positive")

        k_matrix[i, i] += k
        k_matrix[j, j] += k
        k_matrix[i, j] -= k
        k_matrix[j, i] -= k

    return m_matrix, k_matrix


def natural_frequencies(
    inertias: np.ndarray,
    shafts: Sequence[ShaftElement],
    rigid_body_tolerance_hz: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute undamped natural frequencies and mode shapes.

    The routine solves the generalized eigenproblem ``K phi = omega^2 M phi``.

    Args:
        inertias: 1D array of station inertias [kg*m^2].
        shafts: Sequence of shaft elements coupling station pairs.
        rigid_body_tolerance_hz: Frequency threshold below which modes are discarded.

    Returns:
        ``(frequencies_hz, mode_shapes)`` where frequencies are sorted ascending,
        and mode shape columns correspond to each frequency.
    """

    m_matrix, k_matrix = build_torsional_matrices(inertias, shafts)

    inv_sqrt_m = np.diag(1.0 / np.sqrt(np.diag(m_matrix)))
    normalized_k = inv_sqrt_m @ k_matrix @ inv_sqrt_m

    omega_squared, normalized_modes = np.linalg.eigh(normalized_k)
    omega_squared = np.clip(omega_squared, 0.0, None)

    frequencies_hz = np.sqrt(omega_squared) / (2.0 * np.pi)
    valid = frequencies_hz > rigid_body_tolerance_hz
    frequencies_hz = frequencies_hz[valid]
    mode_shapes = inv_sqrt_m @ normalized_modes[:, valid]

    order = np.argsort(frequencies_hz)
    return frequencies_hz[order], mode_shapes[:, order]
