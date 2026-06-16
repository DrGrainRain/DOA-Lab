"""Linear Prediction (LP) algorithms for DOA estimation.

- Forward LP
- Backward LP
- Forward-Backward LP

References: Wang Yongliang, Chapter 3.
"""

import numpy as np
from typing import Tuple


def lp_forward(
    X: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Forward Linear Prediction.

    P_FLP(theta) = 1 / |a^H(theta) * w|^2
    where w = [1; -w_lp] and w_lp is the forward LP filter.

    Parameters
    ----------
    X : np.ndarray, shape (M, snap)
        Array received data.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    order : int
        LP order (default 1 for first-order LP).

    Returns
    -------
    P : np.ndarray
        Forward LP spatial spectrum in dB.
    """
    M, snap = X.shape
    L = M - order

    # Forward: predict X[M-1] from X[0:M-1]
    Xf = X[:L, :]
    xf_target = X[L:, :]
    Rf = (Xf @ Xf.conj().T) / snap
    rf = (Xf @ xf_target.conj().T) / snap

    w_lp = np.linalg.solve(Rf, rf).flatten()
    w = np.concatenate([[1.0], -w_lp])

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        P[ii] = 1.0 / np.abs(a.conj() @ w) ** 2

    return 10.0 * np.log10(P / np.max(P))


def lp_backward(
    X: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Backward Linear Prediction.

    P_BLP(theta) = 1 / |a^H(theta) * w|^2
    where w = [1; -w_blp] and w_blp is the backward LP filter.

    Parameters
    ----------
    X : np.ndarray, shape (M, snap)
        Array received data.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    order : int
        LP order.

    Returns
    -------
    P : np.ndarray
        Backward LP spatial spectrum in dB.
    """
    M, snap = X.shape
    L = M - order

    # Backward: predict X[0] from X[1:M]
    Xb = X[1:, :]
    xb_target = X[:1, :]
    Rb = (Xb @ Xb.conj().T) / snap
    rb = (Xb @ xb_target.conj().T) / snap

    w_lp = np.linalg.solve(Rb, rb).flatten()
    w = np.concatenate([[1.0], -w_lp])

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        P[ii] = 1.0 / np.abs(a.conj() @ w) ** 2

    return 10.0 * np.log10(P / np.max(P))


def lp_forward_backward(
    X: np.ndarray,
    array: object,
    theta_scan: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Forward-Backward Linear Prediction.

    Combines forward and backward prediction for improved performance
    with coherent sources.

    Parameters
    ----------
    X : np.ndarray, shape (M, snap)
        Array received data.
    array : object
        Array object.
    theta_scan : np.ndarray
        Scan angles in radians.
    order : int
        LP order.

    Returns
    -------
    P : np.ndarray
        Forward-backward LP spatial spectrum in dB.
    """
    M, snap = X.shape
    L = M - order

    Xf = X[:L, :]
    xf_target = X[L:, :]
    Xb = X[1:, :]
    xb_target = np.conj(X[:1, :])

    # Combined data
    Xfb = np.vstack([Xf, np.conj(Xb)])
    xfb_target = np.vstack([xf_target, xb_target])

    Rfb = (Xfb @ Xfb.conj().T) / snap
    rfb = (Xfb @ xfb_target.conj().T) / snap

    w_lp = np.linalg.solve(Rfb, rfb).flatten()
    w = np.concatenate([[1.0], -np.conj(w_lp)])

    P = np.zeros(len(theta_scan))
    for ii, th in enumerate(theta_scan):
        a = array.steering_vector(th)
        P[ii] = 1.0 / np.abs(a.conj() @ w) ** 2

    return 10.0 * np.log10(P / np.max(P))
