"""Spatial spectrum estimation algorithms.

Implements the full suite of algorithms from Wang Yongliang's textbook:
- Beamforming (CBF, Capon/MVDR)
- MUSIC family (Standard, Root, Beamspace, Spatial Smoothing)
- ESPRIT family (LS, TLS, 2D, Real, Toeplitz)
- Maximum Likelihood & Subspace Fitting (ML, MODE, SSF, NSF, WSSF)
- Linear Prediction
- Wideband DOA (TCT, ISM)
- Distributed Source Estimation
- High-Order Statistics (HO-MUSIC)
"""

from .beamforming import cbf, capon_mvdr
from .music import music, root_music, beamspace_music, spatial_smoothing_music, mnm
from .esprit import ls_esprit, tls_esprit, two_d_esprit
from .ml import maximum_likelihood, mode_algorithm, iqml
from .subspace_fitting import ssf, nsf, wssf
from .linear_prediction import lp_forward, lp_backward, lp_forward_backward
from .wideband import tct_doa, ism_doa
from .distributed import distributed_source_music
from .high_order import ho_music

__all__ = [
    "cbf", "capon_mvdr",
    "music", "root_music", "beamspace_music", "spatial_smoothing_music", "mnm",
    "ls_esprit", "tls_esprit", "two_d_esprit",
    "maximum_likelihood", "mode_algorithm", "iqml",
    "ssf", "nsf", "wssf",
    "lp_forward", "lp_backward", "lp_forward_backward",
    "tct_doa", "ism_doa",
    "distributed_source_music",
    "ho_music",
]
