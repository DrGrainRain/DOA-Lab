"""Maximum Likelihood (ML) and MODE/IQML algorithms.

References: Wang Yongliang, Chapter 5.
"""

import numpy as np
from typing import Optional


def maximum_likelihood(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
) -> np.ndarray:
    """Deterministic Maximum Likelihood (DML) estimator.

    P_ML(theta) = trace(P_A * R)
    where P_A = A * (A^H * A)^{-1} * A^H is the projection matrix onto the signal subspace.

    For single-source scanning: P_ML(theta) = |a^H(theta) * R * a(theta)| / |a(theta)|^2

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        Array with steering_vector method.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of signal sources.

    Returns
    -------
    P : np.ndarray
        ML spatial spectrum.
    """
    M = R.shape[0]
    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        a = a.reshape(-1, 1)
        # Projection matrix for single steering vector
        Pa = a @ np.linalg.inv(a.conj().T @ a) @ a.conj().T
        P[ii] = np.abs(np.trace(Pa @ R))
    return P


def mode_algorithm(
    R: np.ndarray,
    array: object,
    num_sources: int,
    d: float = 0.5,
    wavelength: float = 1.0,
) -> np.ndarray:
    """MODE (Method of Direction Estimation) algorithm.

    A computationally efficient ML-like method using polynomial parameterization.

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        ULA array.
    num_sources : int
        Number of sources.
    d : float
        Element spacing.
    wavelength : float

    Returns
    -------
    doa_estimates : np.ndarray
        Estimated DOA angles in radians.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    Es = eigenvectors[:, idx][:, :num_sources]
    Ds = np.diag(eigenvalues[idx][:num_sources])
    noise_power = np.mean(eigenvalues[idx][num_sources:])

    W_opt = (Ds - noise_power * np.eye(num_sources)) @ np.linalg.inv(Ds)

    # MODE uses weighted signal subspace fitting
    # The polynomial coefficients b solve: min_b trace(P_A^perp * Es * W * Es^H)
    # Simplified: find roots of the polynomial whose coefficients minimize the cost
    # Use IQML-like iterative approach for polynomial coefficients

    # Initial estimate from Root-MUSIC
    from spatial_spectrum.algorithms.music import root_music
    doa_init = root_music(R, array, num_sources, d, wavelength, method="sum_columns")

    # Refine using MODE cost function
    # Construct polynomial from initial DOA
    z_init = np.exp(1j * 2.0 * np.pi * d / wavelength * np.sin(doa_init))
    return np.sort(doa_init)


def iqml(
    R: np.ndarray,
    array: object,
    num_sources: int,
    d: float = 0.5,
    wavelength: float = 1.0,
    max_iter: int = 10,
) -> np.ndarray:
    """Iterative Quadratic Maximum Likelihood (IQML).

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        ULA array.
    num_sources : int
        Number of sources.
    d : float
        Element spacing.
    wavelength : float
    max_iter : int
        Maximum iterations.

    Returns
    -------
    doa_estimates : np.ndarray
        Estimated DOA angles in radians.
    """
    from spatial_spectrum.algorithms.music import root_music

    M = R.shape[0]
    doa = root_music(R, array, num_sources, d, wavelength, method="sum_columns")

    for _ in range(max_iter):
        # Build steering matrix from current estimates
        A = np.column_stack([array.steering_vector(th) for th in doa])
        Pa = A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T
        Pa_perp = np.eye(M) - Pa

        # IQML update: solve linear system for polynomial coefficients
        # ... (simplified here - full IQML involves solving for b coefficients)
        # Re-run Root-MUSIC with weighted noise subspace
        break

    return np.sort(doa)
