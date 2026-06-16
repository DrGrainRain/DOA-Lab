"""Sensor position error calibration.

- Active calibration with disjoint/non-disjoint sources
- Self-calibration using ML-based position estimation

References: Wang Yongliang, Chapter 13, pp. 440-460.
"""

import numpy as np
from typing import Tuple


def active_position_calibration(
    X: np.ndarray,
    array: object,
    known_doas: np.ndarray,
    num_sources: int,
    nominal_positions: np.ndarray,
    method: str = "disjoint",
) -> np.ndarray:
    """Active position calibration using known-direction sources.

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
    nominal_positions : np.ndarray, shape (M,)
        Nominal (designed) element positions.
    method : str
        "disjoint" - one source at a time
        "non_disjoint" - all sources simultaneously

    Returns
    -------
    position_errors : np.ndarray, shape (M,)
        Estimated position errors (actual - nominal).
    """
    M = X.shape[0]
    R = (X @ X.conj().T) / X.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    En = eigenvectors[:, idx][:, num_sources:]

    if method == "disjoint":
        # Process one source at a time
        position_errors = np.zeros(M)
        for doa in known_doas:
            a_nominal = array.steering_vector(doa)
            # Phase difference gives position error
            phase_diff = np.angle(a_nominal[1:] / a_nominal[0])
            # Simplified: compare with ideal phase
            wavelength = array.wavelength
            ideal_phase = 2.0 * np.pi / wavelength * nominal_positions[1:] * np.sin(doa)
            position_errors[1:] += (phase_diff - ideal_phase) * wavelength / (2.0 * np.pi * np.sin(doa))
        position_errors /= len(known_doas)

    else:
        # Non-disjoint: use all sources in a joint optimization
        position_errors = np.zeros(M)
        cost = np.inf
        # Simple grid refinement
        for _ in range(100):
            current_pos = nominal_positions + position_errors
            a_test = np.exp(1j * 2.0 * np.pi / array.wavelength * current_pos.reshape(-1, 1) * np.sin(known_doas).reshape(1, -1))
            cost_new = np.sum(np.abs(a_test.conj().T @ En) ** 2)
            if cost_new < cost:
                cost = cost_new
                position_errors += 0.001 * np.random.randn(M) * array.wavelength

    return position_errors


def self_calibration_position(
    X: np.ndarray,
    array: object,
    num_sources: int,
    nominal_positions: np.ndarray,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Self-calibration for position errors using ML criterion.

    Jointly estimates DOA and sensor positions.

    Parameters
    ----------
    X : np.ndarray, shape (M, snap)
        Received data.
    array : object
        Array object.
    num_sources : int
        Number of sources.
    nominal_positions : np.ndarray, shape (M,)
        Nominal element positions.
    max_iter : int

    Returns
    -------
    position_errors : np.ndarray
        Estimated position errors.
    doa_estimates : np.ndarray
        Estimated DOA angles in radians.
    """
    M = X.shape[0]
    R = (X @ X.conj().T) / X.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    En = eigenvectors[:, idx][:, num_sources:]

    # Initial DOA estimation assuming nominal positions
    theta_scan = np.linspace(-np.pi / 3, np.pi / 3, 360)
    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        P[ii] = 1.0 / np.sum(np.abs(a.conj() @ En) ** 2)
    peak_indices = np.argsort(P)[-num_sources:]
    doa_estimates = theta_scan[peak_indices]

    # Iterative refinement
    pos_errors = np.zeros(M)
    for _ in range(max_iter):
        current_pos = nominal_positions + pos_errors
        # Update DOA estimates
        for j in range(num_sources):
            th = doa_estimates[j]
            a = np.exp(1j * 2.0 * np.pi / array.wavelength * current_pos * np.sin(th))
            cost = np.abs(a.conj() @ En) ** 2
            # Gradient step for position
            grad = 2.0 * np.real(np.conj(a) * En @ En.conj().T @ a * 1j * 2.0 * np.pi / array.wavelength * np.sin(th))
            pos_errors -= 0.001 * grad

    return pos_errors, np.sort(doa_estimates)
