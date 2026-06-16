"""High-Order Statistics based spatial spectrum estimation.

HO-MUSIC: Fourth-order cumulant based MUSIC for enhanced resolution
and robustness to Gaussian noise.

References: Wang Yongliang, Chapter 12.
"""

import numpy as np


def fourth_order_cumulant_matrix(X: np.ndarray) -> np.ndarray:
    """Compute the fourth-order cumulant matrix for array data.

    R4 = E{(x kron conj(x)) * (x kron conj(x))^H}
       - E{x kron conj(x)} * E{x kron conj(x)}^H
       - E{x * x^H} kron E{conj(x) * x^T}

    This formulation suppresses Gaussian noise contributions.

    Parameters
    ----------
    X : np.ndarray, shape (M, snap), dtype=complex
        Array received data.

    Returns
    -------
    R4 : np.ndarray, shape (M^2, M^2), dtype=complex
        Fourth-order cumulant matrix.
    """
    M, snap = X.shape
    M2 = M * M

    # Compute Kronecker products for each snapshot
    C = np.zeros((M2, snap), dtype=complex)
    for ii in range(snap):
        C[:, ii] = np.kron(X[:, ii], np.conj(X[:, ii]))

    # E{c * c^H}
    E1 = (C @ C.conj().T) / snap

    # E{c} * E{c}^H
    c_mean = np.mean(C, axis=1).reshape(-1, 1)
    E2 = c_mean @ c_mean.conj().T

    # E{x * x^H} kron E{conj(x) * x^T}
    Rx = (X @ X.conj().T) / snap
    E3 = np.kron(Rx, np.conj(Rx))

    R4 = E1 - E2 - E3
    return R4


def ho_music(
    X: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
) -> np.ndarray:
    """High-Order MUSIC using fourth-order cumulants.

    The effective array aperture is increased through the Kronecker product
    structure of the cumulant matrix, providing better resolution and handling
    more sources than elements.

    Parameters
    ----------
    X : np.ndarray, shape (M, snap), dtype=complex
        Array received data.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of signal sources.

    Returns
    -------
    P : np.ndarray
        HO-MUSIC spatial spectrum.
    """
    M = X.shape[0]
    M2 = M * M

    # Compute 4th-order cumulant matrix
    R4 = fourth_order_cumulant_matrix(X)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R4)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Noise subspace (cumulant matrix dimension is M^2)
    En = eigenvectors[:, num_sources:]

    # HO-MUSIC: use Kronecker steering vector b(theta) = a(theta) kron conj(a(theta))
    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        b = np.kron(a, np.conj(a))
        P[ii] = 1.0 / np.sum(np.abs(b.conj() @ En) ** 2)

    return P
