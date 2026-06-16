"""Subspace Fitting algorithms: SSF, NSF, WSSF.

References: Wang Yongliang, Chapter 5.
"""

import numpy as np


def ssf(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
) -> np.ndarray:
    """Signal Subspace Fitting (SSF).

    P_SSF(theta) = |trace(P_A * Es * Es^H)|
    where P_A is the projection matrix onto a(theta).

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        Array with steering_vector method.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of sources.

    Returns
    -------
    P : np.ndarray
        SSF spatial spectrum.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    Es = eigenvectors[:, idx][:, :num_sources]

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th).reshape(-1, 1)
        Pa = a @ np.linalg.inv(a.conj().T @ a) @ a.conj().T
        P[ii] = np.abs(np.trace(Pa @ Es @ Es.conj().T))
    return P


def nsf(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
) -> np.ndarray:
    """Noise Subspace Fitting (NSF).

    P_NSF(theta) = 1 / |trace(En^H * a * a^H * En)|

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of sources.

    Returns
    -------
    P : np.ndarray
        NSF spatial spectrum.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    En = eigenvectors[:, idx][:, num_sources:]

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th).reshape(-1, 1)
        val = np.abs(np.trace(En.conj().T @ a @ a.conj().T @ En))
        if val > 1e-15:
            P[ii] = 1.0 / val
        else:
            P[ii] = 1e15
    return P


def wssf(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
) -> np.ndarray:
    """Weighted Signal Subspace Fitting (WSSF).

    Uses optimal weighting matrix W_opt = (Ds - sigma^2 * I)^2 * Ds^{-1}

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of sources.

    Returns
    -------
    P : np.ndarray
        WSSF spatial spectrum.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    Es = eigenvectors[:, :num_sources]
    Ds = np.diag(eigenvalues[:num_sources])
    noise_power = np.mean(eigenvalues[num_sources:])

    # Optimal weighting matrix
    W_opt = (Ds - noise_power * np.eye(num_sources)) @ np.linalg.inv(Ds)

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th).reshape(-1, 1)
        Pa = a @ np.linalg.inv(a.conj().T @ a) @ a.conj().T
        P[ii] = np.abs(np.trace(Pa @ Es @ W_opt @ Es.conj().T))
    return P
