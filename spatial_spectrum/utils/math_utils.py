"""Mathematical utilities for array signal processing."""

import numpy as np


def db(x: np.ndarray, reference: float = 0.0) -> np.ndarray:
    """Convert to dB scale: 10 * log10(x).

    Parameters
    ----------
    x : np.ndarray
        Input data.
    reference : float
        If > 0, normalize by reference first.

    Returns
    -------
    result : np.ndarray
    """
    x = np.abs(x)
    if reference > 0:
        x = x / reference
    x = np.maximum(x, 1e-15)
    return 10.0 * np.log10(x)


def hermitian(A: np.ndarray) -> np.ndarray:
    """Ensure matrix is Hermitian: (A + A^H) / 2."""
    return (A + A.conj().T) / 2.0


def projection_matrix(a: np.ndarray) -> np.ndarray:
    """Compute projection matrix P_A = a * (a^H * a)^{-1} * a^H.

    Parameters
    ----------
    a : np.ndarray, shape (M,) or (M, P)
        Steering vector or matrix.

    Returns
    -------
    P : np.ndarray
        Projection matrix.
    """
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a @ np.linalg.inv(a.conj().T @ a) @ a.conj().T


def steering_matrix(
    array: object,
    theta: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Build steering matrix for multiple DOA angles.

    Parameters
    ----------
    array : object
        Array with steering_vector method.
    theta : np.ndarray
        DOA angles in radians.

    Returns
    -------
    A : np.ndarray
        Steering matrix, shape (M, len(theta)).
    """
    M = array.num_elements
    P = len(theta)
    A = np.zeros((M, P), dtype=complex)
    for i, th in enumerate(theta):
        A[:, i] = array.steering_vector(th, **kwargs)
    return A
