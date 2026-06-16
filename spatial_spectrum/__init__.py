"""
spatial_spectrum - A Python library for spatial spectrum estimation.

This library provides a comprehensive implementation of classical and modern
spatial spectrum estimation algorithms, based on Wang Yongliang's textbook
"Spatial Spectrum Estimation Theory and Algorithms".

Modules:
    core        - Array geometries, signal generation, covariance estimation
    algorithms  - DOA estimation algorithms (MUSIC, ESPRIT, ML, etc.)
    detection   - Source number estimation (AIC, MDL, GDE, etc.)
    calibration - Array error calibration (gain/phase, position)
    utils       - Helper utilities (smoothing, math functions)

Quick Start:
    >>> import numpy as np
    >>> from spatial_spectrum.core import UniformLinearArray, lfm_signal, generate_array_data
    >>> from spatial_spectrum.algorithms import music, root_music, ls_esprit
    >>> from spatial_spectrum.detection import detect_sources
    >>>
    >>> # Setup
    >>> M = 16
    >>> array = UniformLinearArray(M, wavelength=0.3)
    >>> theta_true = np.deg2rad([-10, 0, 20])
    >>> snap = 500
    >>> t = np.arange(snap) / 1000
    >>> s = lfm_signal(1e9, 1e6, t)
    >>> A = array.steering_matrix(theta_true)
    >>> S = np.tile(s.reshape(1, -1), (3, 1))
    >>> X = generate_array_data(A, S, snr_db=10)
    >>>
    >>> # DOA estimation
    >>> R = X @ X.conj().T / snap
    >>> theta_scan = np.deg2rad(np.arange(-90, 90, 0.1))
    >>> P = music(R, array, theta_scan, num_sources=3)
    >>>
    >>> # Source number detection
    >>> ev, evec = np.linalg.eigh(R)
    >>> num = detect_sources(np.sort(ev)[::-1], snap, method="mdl")

Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"

from spatial_spectrum.core import (
    UniformLinearArray,
    UniformCircularArray,
    UniformRectangularArray,
    steering_vector_ula,
    steering_vector_uca,
    steering_vector_ura,
    array_resolution,
    array_ambiguity,
    lfm_signal,
    cw_signal,
    generate_array_data,
    covariance_matrix,
    eigen_decomposition,
    noise_subspace,
    signal_subspace,
)

from spatial_spectrum.algorithms import (
    cbf,
    capon_mvdr,
    music,
    root_music,
    beamspace_music,
    spatial_smoothing_music,
    mnm,
    ls_esprit,
    tls_esprit,
    two_d_esprit,
    maximum_likelihood,
    mode_algorithm,
    iqml,
    ssf,
    nsf,
    wssf,
    lp_forward,
    lp_backward,
    lp_forward_backward,
    tct_doa,
    ism_doa,
    distributed_source_music,
    ho_music,
)

from spatial_spectrum.detection import (
    aic,
    mdl,
    hq,
    edc,
    gde_criterion,
    detect_sources,
)

from spatial_spectrum.calibration import (
    active_gain_phase_calibration,
    self_calibration_gain_phase,
    taylor_series_calibration,
    active_position_calibration,
    self_calibration_position,
)

__all__ = [
    # Core
    "UniformLinearArray", "UniformCircularArray", "UniformRectangularArray",
    "steering_vector_ula", "steering_vector_uca", "steering_vector_ura",
    "array_resolution", "array_ambiguity",
    "lfm_signal", "cw_signal", "generate_array_data",
    "covariance_matrix", "eigen_decomposition", "noise_subspace", "signal_subspace",
    # Algorithms
    "cbf", "capon_mvdr",
    "music", "root_music", "beamspace_music", "spatial_smoothing_music", "mnm",
    "ls_esprit", "tls_esprit", "two_d_esprit",
    "maximum_likelihood", "mode_algorithm", "iqml",
    "ssf", "nsf", "wssf",
    "lp_forward", "lp_backward", "lp_forward_backward",
    "tct_doa", "ism_doa",
    "distributed_source_music",
    "ho_music",
    # Detection
    "aic", "mdl", "hq", "edc", "gde_criterion", "detect_sources",
    # Calibration
    "active_gain_phase_calibration", "self_calibration_gain_phase",
    "taylor_series_calibration",
    "active_position_calibration", "self_calibration_position",
]
