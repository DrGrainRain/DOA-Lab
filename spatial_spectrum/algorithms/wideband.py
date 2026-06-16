"""Wideband signal spatial spectrum estimation.

- TCT (Two-sided Correlation Transformation)
- ISM (Incoherent Signal-subspace Method)

References: Wang Yongliang, Chapter 8.
"""

import numpy as np
from typing import Optional


def tct_doa(
    X_segments: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
    focusing_freq_idx: Optional[int] = None,
) -> np.ndarray:
    """Two-sided Correlation Transformation (TCT) for wideband DOA estimation.

    The TCT algorithm coherently combines frequency bins by finding
    focusing matrices that transform the signal subspace at each frequency
    to a common reference frequency.

    Parameters
    ----------
    X_segments : np.ndarray, shape (num_freqs, M, num_segments)
        Frequency-domain data for each frequency bin.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of signal sources.
    focusing_freq_idx : int, optional
        Index of the focusing (reference) frequency. If None, auto-select
        the frequency with median maximum eigenvalue.

    Returns
    -------
    P : np.ndarray
        Coherently combined MUSIC spatial spectrum.
    """
    num_freqs, M, num_segments = X_segments.shape

    # Compute covariance at each frequency
    R_freqs = np.zeros((num_freqs, M, M), dtype=complex)
    max_ev = np.zeros(num_freqs)
    eigvecs_freqs = np.zeros((num_freqs, M, M), dtype=complex)

    for jj in range(num_freqs):
        Xj = X_segments[jj]
        Rj = (Xj @ Xj.conj().T) / num_segments
        R_freqs[jj] = Rj
        ev, evec = np.linalg.eigh(Rj)
        idx = np.argsort(ev)[::-1]
        max_ev[jj] = ev[idx[0]]
        eigvecs_freqs[jj] = evec[:, idx]

    # Select focusing frequency (frequency with median max eigenvalue)
    if focusing_freq_idx is None:
        focusing_freq_idx = int(np.argsort(max_ev)[num_freqs // 2])

    Q0 = eigvecs_freqs[focusing_freq_idx]

    # Coherently combine covariance matrices
    R_coh = np.zeros((M, M), dtype=complex)
    for jj in range(num_freqs):
        Qf = eigvecs_freqs[jj]
        Tf = Q0 @ Qf.conj().T  # Focusing matrix: unitary
        R_coh += Tf @ R_freqs[jj] @ Tf.conj().T

    R_coh /= num_freqs

    # Apply narrowband MUSIC
    from spatial_spectrum.algorithms.music import music
    return music(R_coh, array, theta_scan, num_sources, mode="noise")


def ism_doa(
    X_segments: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    num_sources: int,
) -> np.ndarray:
    """Incoherent Signal-subspace Method (ISM) for wideband DOA.

    Averages MUSIC spectra across frequency bins (non-coherent combination).

    Parameters
    ----------
    X_segments : np.ndarray, shape (num_freqs, M, num_segments)
        Frequency-domain data.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    num_sources : int
        Number of sources.

    Returns
    -------
    P : np.ndarray
        ISM spatial spectrum (averaged across frequencies).
    """
    num_freqs = X_segments.shape[0]
    from spatial_spectrum.algorithms.music import music

    P_avg = np.zeros(len(theta_scan))
    for jj in range(num_freqs):
        Xj = X_segments[jj]
        Rj = (Xj @ Xj.conj().T) / Xj.shape[1]
        Pj = music(Rj, array, theta_scan, num_sources)
        P_avg += Pj

    return P_avg / num_freqs
