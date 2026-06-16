"""Spatial smoothing and forward-backward averaging for decorrelation."""

import numpy as np
from typing import Tuple


def spatial_smoothing(
    R: np.ndarray,
    num_subarrays: int,
    mode: str = "forward_backward",
) -> np.ndarray:
    """Apply spatial smoothing to covariance matrix.

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    num_subarrays : int
        Number of subarrays.
    mode : str
        "forward", "backward", or "forward_backward".

    Returns
    -------
    R_smoothed : np.ndarray
    """
    M = R.shape[0]
    L = M - num_subarrays + 1

    Rf = np.zeros((L, L), dtype=complex)
    for i in range(num_subarrays):
        Rf += R[i : i + L, i : i + L]
    Rf /= num_subarrays

    if mode == "forward":
        return Rf

    J = np.fliplr(np.eye(L))
    Rb = J @ Rf.conj() @ J

    if mode == "backward":
        return Rb

    return (Rf + Rb) / 2.0


def forward_backward_averaging(R: np.ndarray) -> np.ndarray:
    """Forward-backward averaging for Hermitian persymmetry.

    R_fb = (R + J * conj(R) * J) / 2

    Parameters
    ----------
    R : np.ndarray, shape (M, M)

    Returns
    -------
    R_fb : np.ndarray
    """
    M = R.shape[0]
    J = np.fliplr(np.eye(M))
    return (R + J @ R.conj() @ J) / 2.0
