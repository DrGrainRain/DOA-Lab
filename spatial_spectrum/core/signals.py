"""Signal generation: LFM (chirp), CW, and array data synthesis.

References: Chapter 2, Spatial Spectrum Estimation Basics.
"""

import numpy as np
from typing import Optional


def lfm_signal(
    f0: float,
    u: float,
    t: np.ndarray,
    initial_phase: float = 0.0,
) -> np.ndarray:
    """Generate a Linear Frequency Modulated (LFM / chirp) signal.

    s(t) = exp(j * 2 * pi * (f0 * t + 0.5 * u * t^2))

    Parameters
    ----------
    f0 : float
        Carrier frequency in Hz.
    u : float
        Chirp rate (frequency sweep rate) in Hz/s.
    t : np.ndarray
        Time samples in seconds.
    initial_phase : float
        Initial phase in radians.

    Returns
    -------
    s : np.ndarray, dtype=complex
        Complex LFM signal.
    """
    return np.exp(1j * (2.0 * np.pi * (f0 * t + 0.5 * u * t**2) + initial_phase))


def cw_signal(f0: float, t: np.ndarray, initial_phase: float = 0.0) -> np.ndarray:
    """Generate a Continuous Wave (CW) tone.

    s(t) = exp(j * 2 * pi * f0 * t)

    Parameters
    ----------
    f0 : float
        Carrier frequency in Hz.
    t : np.ndarray
        Time samples in seconds.
    initial_phase : float
        Initial phase in radians.

    Returns
    -------
    s : np.ndarray, dtype=complex
    """
    return np.exp(1j * (2.0 * np.pi * f0 * t + initial_phase))


def generate_array_data(
    steering_matrix: np.ndarray,
    source_signals: np.ndarray,
    snr_db: float = np.inf,
    noise_type: str = "gaussian",
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Generate array received data X = A * S + N.

    Parameters
    ----------
    steering_matrix : np.ndarray, shape (M, P)
        Array steering matrix (M elements, P sources).
    source_signals : np.ndarray, shape (P, snap)
        Source signal matrix.
    snr_db : float
        Signal-to-noise ratio in dB per source (mean source power). Use inf for noiseless.
    noise_type : str
        Noise distribution: "gaussian" (default).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (M, snap), dtype=complex
        Received array data.
    """
    M, P = steering_matrix.shape
    if source_signals.shape[0] != P:
        raise ValueError(f"Source signal rows ({source_signals.shape[0]}) must match steering matrix cols ({P})")

    snap = source_signals.shape[1]
    X0 = steering_matrix @ source_signals

    if np.isinf(snr_db):
        return X0

    signal_power = np.mean(np.abs(X0) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    rng = np.random.RandomState(random_seed)
    if noise_type == "gaussian":
        noise = np.sqrt(noise_power / 2.0) * (rng.randn(M, snap) + 1j * rng.randn(M, snap))
    else:
        noise = np.sqrt(noise_power / 2.0) * (rng.randn(M, snap) + 1j * rng.randn(M, snap))

    return X0 + noise
