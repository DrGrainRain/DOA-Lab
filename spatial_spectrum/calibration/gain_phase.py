"""Gain and phase error calibration for sensor arrays.

- Active calibration using known calibration sources
- Self-calibration (blind) using eigenstructure methods
- Taylor series expansion method

References: Wang Yongliang, Chapter 13, pp. 420-440.
"""

import numpy as np
from typing import Optional, Tuple


def active_gain_phase_calibration(
    X: np.ndarray,
    array: object,
    known_doa: float,
    num_sources: int = 1,
) -> np.ndarray:
    """Active calibration using a known-direction source.

    Estimates gain and phase errors by comparing the received steering vector
    with the ideal one.

    Parameters
    ----------
    X : np.ndarray, shape (M, snap)
        Received data from a known calibration source.
    array : object
        Array object.
    known_doa : float
        Known DOA of the calibration source in radians.
    num_sources : int
        Number of sources (1 for single calibration source).

    Returns
    -------
    errors : np.ndarray, shape (M,), dtype=complex
        Complex gain/phase error vector (diagonal of Gamma).
        Normalized so that errors[0] = 1.
    """
    M = X.shape[0]
    R = (X @ X.conj().T) / X.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    Es = eigenvectors[:, idx][:, :num_sources]

    # Ideal steering vector
    a_ideal = array.steering_vector(known_doa)

    # Signal eigenvector should be proportional to Gamma * a_ideal
    # Estimate: gamma_i = Es[i, 0] / a_ideal[i] * (a_ideal[0] / Es[0, 0])
    errors = Es[:, 0] / a_ideal
    errors /= errors[0]  # Normalize

    return errors


def self_calibration_gain_phase(
    X: np.ndarray,
    array: object,
    num_sources: int,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Self-calibration for gain/phase errors using eigenstructure method.

    Iteratively estimates DOA and array errors without known calibration sources.
    Suitable for UCA or when sources are uncorrelated.

    Algorithm (WSSF based, pp. 434-435):
        1. Initialize Gamma = I
        2. Estimate DOA using MUSIC with corrected array
        3. Update Gamma by minimizing the MUSIC cost
        4. Repeat until convergence

    Parameters
    ----------
    X : np.ndarray, shape (M, snap)
        Received data.
    array : object
        Array object (typically UCA).
    num_sources : int
        Number of signal sources.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    errors : np.ndarray, shape (M,), dtype=complex
        Estimated gain/phase errors.
    doa_estimates : np.ndarray
        Estimated DOA angles in radians.
    """
    M = X.shape[0]
    R = (X @ X.conj().T) / X.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    En = eigenvectors[:, idx][:, num_sources:]

    # Initialize
    gamma = np.ones(M, dtype=complex)
    J_prev = np.inf

    for iteration in range(max_iter):
        Gamma = np.diag(gamma)

        # DOA estimation step: find peak positions
        # Scan for DOA with current error correction
        theta_scan = np.linspace(-np.pi / 3, np.pi / 3, 360)
        P = np.zeros(len(theta_scan))
        for ii, th in enumerate(theta_scan):
            a = array.steering_vector(th)
            a_corrected = Gamma.conj().T @ a
            P[ii] = 1.0 / np.sum(np.abs(a_corrected.conj() @ En) ** 2)

        # Find peaks (simplified: pick the num_sources highest peaks)
        peak_indices = np.argsort(P)[-num_sources:]
        doa_estimates = theta_scan[peak_indices]

        # Update gamma
        O = np.zeros((M, M), dtype=complex)
        for jj in range(num_sources):
            a = array.steering_vector(doa_estimates[jj])
            E = np.diag(a)
            O += E.conj().T @ (En @ En.conj().T) @ E

        w = np.zeros(M)
        w[0] = 1.0
        gamma = np.linalg.solve(O, w)
        gamma /= gamma[0]

        J = gamma.conj() @ O @ gamma
        if np.abs(J_prev - J) < tol:
            break
        J_prev = J

    return gamma, np.sort(doa_estimates)


def taylor_series_calibration(
    X: np.ndarray,
    array: object,
    known_doas: np.ndarray,
    num_sources: int,
    max_iter: int = 10,
) -> np.ndarray:
    """Taylor series expansion method for gain/phase calibration.

    Uses multiple known calibration sources and a first-order Taylor
    expansion of the MUSIC null spectrum.

    Parameters
    ----------
    X : np.ndarray, shape (M, snap)
        Received data.
    array : object
        Array object.
    known_doas : np.ndarray
        Known DOA angles of calibration sources in radians.
    num_sources : int
        Number of sources.
    max_iter : int

    Returns
    -------
    errors : np.ndarray, shape (M,), dtype=complex
        Estimated errors.
    """
    M = X.shape[0]
    R = (X @ X.conj().T) / X.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    En = eigenvectors[:, idx][:, num_sources:]

    # First-order Taylor approximation
    errors = np.ones(M, dtype=complex)
    for _ in range(max_iter):
        Gamma = np.diag(errors)
        for doa in known_doas:
            a = array.steering_vector(doa)
            # Refine using gradient descent on MUSIC cost
            a_corrected = Gamma @ a
            grad = En @ En.conj().T @ a_corrected
            # Simple update
            errors -= 0.01 * grad * np.conj(a)
        errors /= errors[0]

    return errors
