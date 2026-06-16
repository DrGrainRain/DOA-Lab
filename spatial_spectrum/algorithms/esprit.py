"""ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques).

Includes:
- LS-ESPRIT (Least Squares)
- TLS-ESPRIT (Total Least Squares)
- 2D ESPRIT for uniform rectangular arrays

References: Wang Yongliang, Chapter 6.
"""

import numpy as np
from typing import Tuple


def ls_esprit(
    R: np.ndarray, num_sources: int, d: float = 0.5, wavelength: float = 1.0
) -> np.ndarray:
    """LS-ESPRIT for ULA.

    Uses the shift-invariance property of ULA subarrays.

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    num_sources : int
        Number of signal sources.
    d : float
        Element spacing.
    wavelength : float
        Signal wavelength.

    Returns
    -------
    doa_estimates : np.ndarray
        Estimated DOA angles in radians, sorted.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    Us = eigenvectors[:, idx][:, :num_sources]

    # Subarray selection matrices (shift-invariance for ULA)
    Us1 = Us[:-1, :]  # First M-1 elements
    Us2 = Us[1:, :]   # Last M-1 elements

    # LS-ESPRIT: Psi = (Us1^H * Us1)^{-1} * Us1^H * Us2
    Psi_ls = np.linalg.inv(Us1.conj().T @ Us1) @ Us1.conj().T @ Us2

    # Eigenvalue decomposition of Psi
    eigenvalues_psi = np.linalg.eigvals(Psi_ls)

    # Convert to DOA
    doa_estimates = np.arcsin(np.angle(eigenvalues_psi) / np.pi * wavelength / (2.0 * d))

    return np.sort(doa_estimates)


def tls_esprit(
    R: np.ndarray, num_sources: int, d: float = 0.5, wavelength: float = 1.0
) -> np.ndarray:
    """TLS-ESPRIT for ULA.

    Uses total least squares for improved noise robustness.

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    num_sources : int
        Number of signal sources.
    d : float
        Element spacing.
    wavelength : float
        Signal wavelength.

    Returns
    -------
    doa_estimates : np.ndarray
        Estimated DOA angles in radians, sorted.
    """
    M = R.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    Us = eigenvectors[:, idx][:, :num_sources]

    Us1 = Us[:-1, :]
    Us2 = Us[1:, :]

    # TLS-ESPRIT: form Us12 = [Us1, Us2] and find null space of Us12^H * Us12
    Us12 = np.hstack([Us1, Us2])
    _, _, Vh = np.linalg.svd(Us12, full_matrices=False)
    # For TLS, take the P rightmost singular vectors of Vh
    # E = Vh[P:, P:] reshaped? Actually:
    # E = eigvecs corresponding to smallest eigenvalues of Us12^H * Us12
    # Using SVD of Us12: Us12^H * Us12 = V * S^2 * V^H
    # The eigenvectors for the P smallest eigenvalues are V[:, -P:]
    _, E_small_svd = np.linalg.eigh(Us12.conj().T @ Us12)
    E = E_small_svd[:, :num_sources]  # P smallest eigenvectors

    E11 = E[:num_sources, :]
    E21 = E[num_sources:, :]

    Psi_tls = -E11 @ np.linalg.inv(E21)

    eigenvalues_psi = np.linalg.eigvals(Psi_tls)
    doa_estimates = np.arcsin(np.angle(eigenvalues_psi) / np.pi * wavelength / (2.0 * d))

    return np.sort(np.real(doa_estimates))


def two_d_esprit(
    R: np.ndarray,
    nx: int,
    ny: int,
    num_sources: int,
    dx: float = 0.5,
    dy: float = 0.5,
    wavelength: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """2D ESPRIT for Uniform Rectangular Array.

    Estimates both azimuth and elevation angles.

    Parameters
    ----------
    R : np.ndarray, shape (nx*ny, nx*ny)
        Covariance matrix of the URA.
    nx : int
        Number of elements along x-axis.
    ny : int
        Number of elements along y-axis.
    num_sources : int
        Number of signal sources.
    dx : float
        Element spacing along x.
    dy : float
        Element spacing along y.
    wavelength : float

    Returns
    -------
    azimuth : np.ndarray
        Estimated azimuth angles in radians.
    elevation : np.ndarray
        Estimated elevation angles in radians.
    """
    M = nx * ny
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    Us = eigenvectors[:, idx][:, :num_sources]

    # Selection matrices for x- and y-direction shift invariance
    # Subarray 1: all elements except those on the last column (x-direction)
    sel_x1 = np.ones(nx * ny, dtype=bool)
    sel_x1.reshape(ny, nx)[:, -1] = False
    sel_x1 = sel_x1.ravel()

    sel_x2 = np.ones(nx * ny, dtype=bool)
    sel_x2.reshape(ny, nx)[:, 0] = False
    sel_x2 = sel_x2.ravel()

    # For y-direction
    sel_y1 = np.ones(nx * ny, dtype=bool)
    sel_y1.reshape(ny, nx)[-1, :] = False
    sel_y1 = sel_y1.ravel()

    sel_y2 = np.ones(nx * ny, dtype=bool)
    sel_y2.reshape(ny, nx)[0, :] = False
    sel_y2 = sel_y2.ravel()

    Us_x1 = Us[sel_x1, :]
    Us_x2 = Us[sel_x2, :]
    Us_y1 = Us[sel_y1, :]
    Us_y2 = Us[sel_y2, :]

    # LS-ESPRIT for each direction
    Psi_x = np.linalg.inv(Us_x1.conj().T @ Us_x1) @ Us_x1.conj().T @ Us_x2
    Psi_y = np.linalg.inv(Us_y1.conj().T @ Us_y1) @ Us_y1.conj().T @ Us_y2

    # Joint diagonalization (simplified: eigendecompose Psi_x + j*Psi_y)
    Psi = Psi_x + 1j * Psi_y
    eigenvalues_psi = np.linalg.eigvals(Psi)

    u = np.angle(eigenvalues_psi) * wavelength / (2.0 * np.pi * dx)
    v = np.imag(np.log(eigenvalues_psi)) * wavelength / (2.0 * np.pi * dy)

    # Avoid complex values from numerical errors
    u = np.real(u)
    v = np.real(v)

    azimuth = np.arctan2(v, u)
    elevation = np.arcsin(np.sqrt(u**2 + v**2))

    return azimuth, elevation
