"""Distributed signal source parameter estimation.

Handles spatially distributed sources with different angular distribution models:
- Uniform distribution
- Triangular distribution
- Gaussian distribution

References: Wang Yongliang, Chapter 10.
"""

import numpy as np
from typing import Tuple


def distributed_steering_vector(
    theta_center: float,
    delta: float,
    M: int,
    d: float = 0.5,
    wavelength: float = 1.0,
    distribution: str = "triangular",
) -> np.ndarray:
    """Compute the generalized steering vector for a distributed source.

    The effective steering vector accounts for angular spread around
    the center angle.

    Parameters
    ----------
    theta_center : float
        Center DOA angle in radians.
    delta : float
        Angular spread parameter in radians.
    M : int
        Number of array elements.
    d : float
        Element spacing.
    wavelength : float
    distribution : str
        Distribution shape: "uniform", "triangular", or "gaussian".

    Returns
    -------
    a_dist : np.ndarray, shape (M,), dtype=complex
    """
    a_idx = np.arange(M)
    a_center = np.exp(1j * 2.0 * np.pi * d / wavelength * a_idx * np.sin(theta_center))

    if distribution == "uniform":
        # sin(x)/x modulation: sinc(a * sqrt(3) * delta) / sinc(...)
        x = a_idx * delta
        # Avoid division by zero for the first element
        with np.errstate(divide="ignore", invalid="ignore"):
            mod_factor = np.where(a_idx == 0, 1.0, np.sin(np.sqrt(3) * x) / (np.sqrt(3) * x))
        return a_center * mod_factor

    elif distribution == "triangular":
        x = a_idx * delta
        with np.errstate(divide="ignore", invalid="ignore"):
            mod_factor = np.where(a_idx == 0, 1.0, 2.0 * (1.0 - np.cos(x)) / (x**2))
        return a_center * mod_factor

    elif distribution == "gaussian":
        mod_factor = np.exp(-0.5 * (a_idx * delta) ** 2)
        return a_center * mod_factor

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def distributed_source_music(
    R: np.ndarray,
    theta_scan: np.ndarray,
    delta_scan: np.ndarray,
    num_sources: int,
    d: float = 0.5,
    wavelength: float = 1.0,
    distribution: str = "triangular",
    mode: str = "eigenvalue",
) -> np.ndarray:
    """MUSIC-type estimator for distributed sources.

    Performs a 2D search over center angle theta and spread delta.

    Two modes:
    - "eigenvalue": Uses the minimum eigenvalue of G1 (Eq. 10.5.4)
    - "point_source": Treats as point source MUSIC for comparison

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    theta_scan : np.ndarray
        Center angle scan points in radians.
    delta_scan : np.ndarray
        Angular spread scan points in radians.
    num_sources : int
        Number of distributed sources.
    d : float
        Element spacing.
    wavelength : float
    distribution : str
        Distribution type.
    mode : str
        "eigenvalue" for distributed source, "point_source" for comparison.

    Returns
    -------
    P : np.ndarray
        2D spatial spectrum with shape (len(delta_scan), len(theta_scan)) for
        "eigenvalue" mode, or 1D (len(theta_scan),) for "point_source" mode.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    En = eigenvectors[:, idx][:, num_sources:]

    a_idx = np.arange(M)

    if mode == "point_source":
        P = np.zeros(len(theta_scan))
        for ii, th in enumerate(theta_scan):
            a = np.exp(1j * 2.0 * np.pi * d / wavelength * a_idx * np.sin(th))
            P[ii] = 1.0 / np.sum(np.abs(a.conj() @ En) ** 2)
        return P

    # Eigenvalue mode: 2D search
    P = np.zeros((len(delta_scan), len(theta_scan)))
    for kk1, th in enumerate(theta_scan):
        for kk2, delta in enumerate(delta_scan):
            a_dist = distributed_steering_vector(th, delta, M, d, wavelength, distribution)
            # G1 = Re{Psi^H * En * En^H * Psi}, where Psi = diag(a_dist)
            Psi = np.diag(a_dist)
            G1 = np.real(Psi.conj().T @ En @ En.conj().T @ Psi)
            # Minimum eigenvalue of G1
            ev_min = np.min(np.linalg.eigvalsh(G1))
            if ev_min > 1e-15:
                P[kk2, kk1] = 1.0 / ev_min
            else:
                P[kk2, kk1] = 1e15

    return P
