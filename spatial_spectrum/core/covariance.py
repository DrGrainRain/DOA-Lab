"""Covariance matrix estimation and eigendecomposition utilities.

References: Chapter 2, Spatial Spectrum Estimation Basics.
"""

import numpy as np
from typing import Tuple


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Estimate spatial covariance matrix from snapshot data.

    R = X @ X^H / snap

    Parameters
    ----------
    X : np.ndarray, shape (M, snap), dtype=complex
        Received array data (M elements, snap snapshots).

    Returns
    -------
    R : np.ndarray, shape (M, M), dtype=complex
        Estimated covariance matrix.
    """
    M, snap = X.shape
    return (X @ X.conj().T) / snap


def eigen_decomposition(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform eigenvalue decomposition of a Hermitian matrix.

    Parameters
    ----------
    R : np.ndarray, shape (M, M), dtype=complex
        Covariance matrix (Hermitian).

    Returns
    -------
    eigenvalues : np.ndarray, shape (M,), dtype=float
        Eigenvalues sorted in descending order.
    eigenvectors : np.ndarray, shape (M, M), dtype=complex
        Corresponding eigenvectors as columns.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def noise_subspace(eigenvectors: np.ndarray, num_sources: int) -> np.ndarray:
    """Extract noise subspace from eigenvectors.

    Parameters
    ----------
    eigenvectors : np.ndarray, shape (M, M)
        All eigenvectors sorted by eigenvalue (descending).
    num_sources : int
        Number of signal sources P.

    Returns
    -------
    En : np.ndarray, shape (M, M-P)
        Noise subspace eigenvectors.
    """
    return eigenvectors[:, num_sources:]


def signal_subspace(eigenvectors: np.ndarray, num_sources: int) -> np.ndarray:
    """Extract signal subspace from eigenvectors.

    Parameters
    ----------
    eigenvectors : np.ndarray, shape (M, M)
        All eigenvectors sorted by eigenvalue (descending).
    num_sources : int
        Number of signal sources P.

    Returns
    -------
    Es : np.ndarray, shape (M, P)
        Signal subspace eigenvectors.
    """
    return eigenvectors[:, :num_sources]


def spatial_smoothing(R: np.ndarray, num_subarrays: int, mode: str = "forward_backward") -> np.ndarray:
    """Apply spatial smoothing to decorrelate coherent sources.

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Original covariance matrix.
    num_subarrays : int
        Number of subarrays (smoothing order).
    mode : str
        "forward"  - forward-only smoothing
        "backward" - backward-only smoothing
        "forward_backward" - forward-backward smoothing (default)

    Returns
    -------
    R_fb : np.ndarray, shape (L, L)
        Spatially smoothed covariance matrix, L = M - num_subarrays + 1.
    """
    M = R.shape[0]
    L = M - num_subarrays + 1
    if L <= 0:
        raise ValueError(f"num_subarrays ({num_subarrays}) must be < M ({M})")

    # Forward smoothing
    Rf = np.zeros((L, L), dtype=complex)
    for i in range(num_subarrays):
        Rf += R[i : i + L, i : i + L]
    Rf /= num_subarrays

    if mode == "forward":
        return Rf

    # Backward smoothing
    J = np.fliplr(np.eye(L))
    Rb = J @ Rf.conj() @ J

    if mode == "backward":
        return Rb

    # Forward-backward
    return (Rf + Rb) / 2.0
