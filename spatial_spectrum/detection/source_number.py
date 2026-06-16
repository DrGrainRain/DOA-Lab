"""Source number estimation using information theoretic criteria and GDE.

Implements:
- AIC (Akaike Information Criterion)
- MDL (Minimum Description Length)
- HQ (Hannan-Quinn Criterion)
- EDC (Efficient Detection Criterion)
- GDE (Gerschgorin Disk Estimator)

References: Wang Yongliang, Chapter 2, Sections 2.4-2.5.
"""

import numpy as np
from typing import Tuple


def _eigenvalue_ratio(eigenvalues: np.ndarray, k: int) -> float:
    """Compute the ratio of arithmetic to geometric mean of noise eigenvalues."""
    M = len(eigenvalues)
    noise_ev = eigenvalues[k:]
    L = M - k
    if L == 0:
        return 1.0
    arith_mean = np.sum(noise_ev) / L
    geom_mean = np.prod(noise_ev) ** (1.0 / L)
    if geom_mean < 1e-300:
        geom_mean = 1e-300
    return arith_mean / geom_mean


def aic(eigenvalues: np.ndarray, num_snapshots: int) -> np.ndarray:
    """Akaike Information Criterion for source number estimation.

    AIC(k) = 2 * snap * (M - k) * log(rho(k)) + 2 * k * (2M - k)

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (M,)
        Eigenvalues sorted in descending order.
    num_snapshots : int
        Number of snapshots.

    Returns
    -------
    aic_values : np.ndarray, shape (M-1,)
        AIC values for k = 1, 2, ..., M-1.
    """
    M = len(eigenvalues)
    aic_values = np.zeros(M - 1)
    for k in range(1, M):
        ratio = _eigenvalue_ratio(eigenvalues, k)
        aic_values[k - 1] = 2.0 * num_snapshots * (M - k) * np.log(ratio) + 2.0 * k * (2.0 * M - k)
    return aic_values


def mdl(eigenvalues: np.ndarray, num_snapshots: int) -> np.ndarray:
    """Minimum Description Length criterion for source number estimation.

    MDL(k) = snap * (M - k) * log(rho(k)) + 0.5 * k * (2M - k) * log(snap)

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (M,)
        Eigenvalues sorted in descending order.
    num_snapshots : int
        Number of snapshots.

    Returns
    -------
    mdl_values : np.ndarray, shape (M-1,)
    """
    M = len(eigenvalues)
    mdl_values = np.zeros(M - 1)
    for k in range(1, M):
        ratio = _eigenvalue_ratio(eigenvalues, k)
        mdl_values[k - 1] = (
            num_snapshots * (M - k) * np.log(ratio) + 0.5 * k * (2.0 * M - k) * np.log(num_snapshots)
        )
    return mdl_values


def hq(eigenvalues: np.ndarray, num_snapshots: int) -> np.ndarray:
    """Hannan-Quinn Criterion.

    HQ(k) = snap * (M - k) * log(rho(k)) + 0.5 * k * (2M - k) * log(log(snap))
    """
    M = len(eigenvalues)
    hq_values = np.zeros(M - 1)
    for k in range(1, M):
        ratio = _eigenvalue_ratio(eigenvalues, k)
        hq_values[k - 1] = (
            num_snapshots * (M - k) * np.log(ratio)
            + 0.5 * k * (2.0 * M - k) * np.log(np.log(num_snapshots))
        )
    return hq_values


def edc(eigenvalues: np.ndarray, num_snapshots: int) -> np.ndarray:
    """Efficient Detection Criterion.

    EDC(k) = snap * (M - k) * log(rho(k)) + k * (2M - k) * 0.5 * log(log(snap))
    """
    M = len(eigenvalues)
    edc_values = np.zeros(M - 1)
    for k in range(1, M):
        ratio = _eigenvalue_ratio(eigenvalues, k)
        edc_values[k - 1] = (
            num_snapshots * (M - k) * np.log(ratio)
            + k * (2.0 * M - k) * 0.5 * np.log(np.log(num_snapshots))
        )
    return edc_values


def gde_criterion(R: np.ndarray, dl: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Gerschgorin Disk Estimator (GDE) for source number detection.

    Implements the GDE method based on unitary transformation of the covariance matrix.
    (Eq. 2.5.25 in the textbook)

    Parameters
    ----------
    R : np.ndarray, shape (M, M)
        Covariance matrix.
    dl : float
        Adjustment factor (default 0.5, typical range 0.25 to 0.8).

    Returns
    -------
    GDE : np.ndarray, shape (M, M-1)
        GDE values for each iteration n=0,...,M-1 and each k=1,...,M-1.
    K_est : np.ndarray, shape (M,)
        Estimated source counts for each iteration.
    """
    M = R.shape[0]
    R_pie = R[: M - 1, : M - 1]
    _, V = np.linalg.eigh(R_pie)
    # Unitary transformation matrix T
    T = np.zeros((M, M), dtype=complex)
    T[: M - 1, : M - 1] = V
    T[M - 1, M - 1] = 1.0
    Rt = T.conj().T @ R @ T
    Rou = np.abs(Rt[: M - 1, M - 1])
    temp0 = np.sum(Rou)

    GDE = np.zeros((M, M - 1))
    K_est = np.zeros(M)
    for n in range(M):
        for k in range(M - 1):
            GDE[n, k] = Rou[k] - dl / (M - 1) * temp0
        pos = np.where(GDE[n, :] >= 0)[0]
        if len(pos) > 0:
            K_est[n] = M - pos[0]
        else:
            K_est[n] = 0
    return GDE, K_est


def detect_sources(
    eigenvalues: np.ndarray,
    num_snapshots: int,
    method: str = "mdl",
) -> int:
    """Estimate the number of signal sources using the specified criterion.

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (M,)
        Eigenvalues sorted in descending order.
    num_snapshots : int
        Number of snapshots.
    method : str
        Criterion: "aic", "mdl", "hq", "edc".

    Returns
    -------
    num_sources : int
        Estimated number of sources (1 <= num_sources <= M-1).
    """
    methods = {"aic": aic, "mdl": mdl, "hq": hq, "edc": edc}
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Available: {list(methods.keys())}")

    values = methods[method](eigenvalues, num_snapshots)
    # The estimated number is the minimizer index + 1
    return int(np.argmin(values)) + 1
