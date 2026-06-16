"""MUSIC (MUltiple SIgnal Classification) family of algorithms.

Includes:
- Standard MUSIC
- Root-MUSIC (polynomial rooting and noise subspace projection methods)
- Beamspace MUSIC
- Spatial Smoothing MUSIC
- Minimum Norm Method (MNM)

References: Wang Yongliang, Chapter 4.
"""

import numpy as np
from typing import Tuple, Optional


def music(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
    mode: str = "noise",
    log_scale: bool = False,
) -> np.ndarray:
    """Standard MUSIC spatial spectrum estimation.

    P_MUSIC(theta) = 1 / || a^H(theta) * En ||^2

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        Array with steering_vector(theta) method.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of signal sources.
    mode : str
        "noise" - standard MUSIC using noise subspace (default).
        "signal" - using signal subspace (for comparison).
    log_scale : bool
        If True, return 10*log10(P).

    Returns
    -------
    P : np.ndarray
        Spatial spectrum.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    if mode == "noise":
        En = eigenvectors[:, num_sources:]
    else:
        Es = eigenvectors[:, :num_sources]
        En = Es

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        if mode == "noise":
            P[ii] = 1.0 / np.sum(np.abs(a.conj() @ En) ** 2)
        else:
            P[ii] = np.sum(np.abs(a.conj() @ En) ** 2)

    if log_scale:
        P = 10.0 * np.log10(P / np.max(np.abs(P)))
    return P


def root_music(
    R: np.ndarray,
    array: object,
    num_sources: int,
    d: float = 0.5,
    wavelength: float = 1.0,
    method: str = "sum_columns",
) -> np.ndarray:
    """Root-MUSIC algorithm for ULA.

    Two polynomial formation methods:
    1. "sum_columns" - sum along diagonals of En*En^H (Eq. 4.5.7)
    2. "subspace_fit" - using subspace partitioning

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        ULA array object.
    num_sources : int
        Number of signal sources.
    d : float
        Element spacing.
    wavelength : float
    method : str
        "sum_columns" or "subspace_fit".

    Returns
    -------
    doa_estimates : np.ndarray
        Estimated DOA angles in radians.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    En = eigenvectors[:, num_sources:]

    if method == "sum_columns":
        # Method 2: sum along diagonals of A = En * En^H
        A = En @ En.conj().T
        a_coeff = np.zeros(2 * M - 1, dtype=complex)
        for l in range(-(M - 1), M):
            s = 0.0 + 0.0j
            for m in range(M):
                n = m - l
                if 0 <= n < M:
                    s += A[m, n]
            a_coeff[l + M - 1] = s
        poly = a_coeff[::-1]
        roots_all = np.roots(poly)
    else:
        # Method 1: subspace fitting
        N = num_sources
        Un1 = En[:N, :]
        Un2 = En[N:, :]
        C = Un1 @ np.linalg.pinv(Un2) @ np.concatenate([[1], np.zeros(M - N - 1)])
        Cc = np.concatenate([[1], C[::-1]])
        roots_all = np.roots(Cc)

    # Select roots inside/near unit circle
    roots_inside = roots_all[np.abs(roots_all) <= 1.0]
    # Sort by distance from unit circle
    dist = np.abs(np.abs(roots_inside) - 1.0)
    idx_sort = np.argsort(dist)
    selected_roots = roots_inside[idx_sort][:num_sources]

    # Convert roots to DOA
    doa_estimates = np.arcsin(np.angle(selected_roots) * wavelength / (2.0 * np.pi * d))
    return np.sort(doa_estimates)


def beamspace_music(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
    beamforming_matrix: np.ndarray,
) -> np.ndarray:
    """Beamspace MUSIC.

    Transforms element-space data to beamspace before applying MUSIC,
    reducing computational complexity.

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Element-space covariance matrix.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of sources.
    beamforming_matrix : np.ndarray, shape (M, B)
        Beamforming matrix with B beams.

    Returns
    -------
    P : np.ndarray
        Beamspace MUSIC spectrum.
    """
    # Transform to beamspace
    W = beamforming_matrix
    R_bs = W.conj().T @ R @ W
    B = R_bs.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(R_bs)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    En_bs = eigenvectors[:, num_sources:]

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a_elem = array.steering_vector(th)
        a_bs = W.conj().T @ a_elem
        P[ii] = 1.0 / np.sum(np.abs(a_bs.conj() @ En_bs) ** 2)
    return P


def spatial_smoothing_music(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
    num_subarrays: int,
) -> np.ndarray:
    """Spatial Smoothing MUSIC for coherent source decorrelation.

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Original covariance matrix.
    array : object
        Array object (should have appropriate dimensions after smoothing).
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of sources.
    num_subarrays : int
        Number of subarrays for smoothing.

    Returns
    -------
    P : np.ndarray
        Spatially smoothed MUSIC spectrum.
    """
    from spatial_spectrum.core.covariance import spatial_smoothing

    R_fb = spatial_smoothing(R, num_subarrays, mode="forward_backward")
    L = R_fb.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(R_fb)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    En = eigenvectors[:, num_sources:]

    # Create a subarray object (ULA with L elements)
    from spatial_spectrum.core.array import UniformLinearArray

    subarray = UniformLinearArray(L, d=array.spacing, wavelength=array.wavelength)

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = subarray.steering_vector(th)
        P[ii] = 1.0 / np.sum(np.abs(a.conj() @ En) ** 2)
    return P


def mnm(
    R: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
    log_scale: bool = False,
) -> np.ndarray:
    """Minimum Norm Method (MNM).

    Finds the minimum-norm vector in the noise subspace.

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of signal sources.
    log_scale : bool

    Returns
    -------
    P : np.ndarray
        MNM spatial spectrum.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    En = eigenvectors[:, num_sources:]

    # Minimum norm weight vector
    c = En[0, :].conj()  # First row of noise subspace
    Enn = En[1:, :]
    w_mnm = np.concatenate([[1.0], Enn @ c / (c @ c.conj())])

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        P[ii] = 1.0 / np.abs(a.conj() @ w_mnm) ** 2

    if log_scale:
        P = 10.0 * np.log10(P / np.max(np.abs(P)))
    return P
