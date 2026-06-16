"""Conventional Beamforming (CBF) and Capon/MVDR beamforming.

References: Chapter 3, Section 3.3.
"""

import numpy as np
from typing import Optional


def cbf(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    **kwargs
) -> np.ndarray:
    """Conventional Beamforming (Bartlett) spatial spectrum.

    P_CBF(theta) = a^H(theta) * R * a(theta)

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        Array object with a steering_vector(theta) method.
    theta_scan : np.ndarray
        Scan angles in radians.

    Returns
    -------
    P : np.ndarray
        Spatial spectrum for each scan angle.
    """
    M = R.shape[0]
    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        P[ii] = np.abs(a.conj() @ R @ a)
    return P


def capon_mvdr(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    diagonal_loading: float = 0.0,
) -> np.ndarray:
    """Capon / MVDR (Minimum Variance Distortionless Response) beamforming.

    P_Capon(theta) = 1 / (a^H(theta) * R^{-1} * a(theta))

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        Array object with a steering_vector(theta) method.
    theta_scan : np.ndarray
        Scan angles in radians.
    diagonal_loading : float
        Diagonal loading factor for regularization (default 0).

    Returns
    -------
    P : np.ndarray
        Spatial spectrum for each scan angle.
    """
    M = R.shape[0]
    reg = diagonal_loading * np.eye(M)
    R_inv = np.linalg.inv(R + reg)
    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        P[ii] = 1.0 / np.abs(a.conj() @ R_inv @ a)
    return P
