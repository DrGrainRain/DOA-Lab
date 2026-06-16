# DOA-Lab
A code repository for DOA (Direction-of-Arrival) estimation research. Implements classic array signal processing algorithms including MUSIC, ESPRIT, beamforming and subspace-based methods, as well as typical deep learning-based DOA estimation approaches. Maintained for academic study, algorithm verification and my first research paper.

# spatial-spectrum

**A comprehensive Python library for spatial spectrum estimation and array signal processing.**

Converted from MATLAB example code accompanying Wang Yongliang's textbook
*"Spatial Spectrum Estimation Theory and Algorithms"* (Chapters 2-14).

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/numpy-%3E%3D1.20-green)](https://numpy.org/)

---

## Features

- **15+ classical algorithms**: MUSIC, Root-MUSIC, ESPRIT (LS/TLS/2D), Capon/MVDR, ML, MODE, IQML, SSF, WSSF, Linear Prediction, MNM, HO-MUSIC, TCT wideband, distributed sources
- **Multiple array geometries**: ULA, UCA, URA with flexible parametrization
- **Source number detection**: AIC, MDL, HQ, EDC, GDE criteria
- **Array calibration**: Gain/phase and position error calibration (active & self-calibration)
- **Coherent sources**: Spatial smoothing, forward-backward averaging
- **Wideband signals**: TCT coherent and ISM incoherent processing
- **High-order statistics**: 4th-order cumulant based MUSIC
- **Distributed sources**: Uniform, triangular, and Gaussian angular distribution models

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/username/spatial-spectrum.git
cd spatial-spectrum
pip install -e .
```

### Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0 (optional, for advanced optimization)
- matplotlib >= 3.4.0 (optional, for visualization)

## Quick Start

### Example 1: Standard MUSIC DOA estimation

```python
import numpy as np
from spatial_spectrum import (
    UniformLinearArray,
    lfm_signal,
    generate_array_data,
    covariance_matrix,
    music,
    detect_sources,
)

# Array setup: 16-element ULA, half-wavelength spacing
M = 16
wavelength = 0.3  # meters
array = UniformLinearArray(M, wavelength=wavelength)

# Signal setup: 3 uncorrelated LFM sources
theta_true = np.deg2rad([-20, 5, 40])
snap = 500
fs = 1000
t = np.arange(snap) / fs
f0 = 1e9

# Generate source signals and array data
S = np.array([lfm_signal(f0, 5e6, t),
              lfm_signal(f0, 10e6, t),
              lfm_signal(f0, 3e6, t)])
A = array.steering_matrix(theta_true)
X = generate_array_data(A, S, snr_db=10)

# Covariance matrix and source detection
R = covariance_matrix(X)
eigenvalues, _ = np.linalg.eigh(R)
num_sources = detect_sources(eigenvalues[::-1], snap, method="mdl")
print(f"Estimated sources: {num_sources}")

# MUSIC spectrum
theta_scan = np.deg2rad(np.arange(-90, 90, 0.1))
P_music = music(R, array, theta_scan, num_sources=num_sources)

# Find peaks
peak_idx = np.argsort(P_music)[-num_sources:]
doa_est = np.rad2deg(theta_scan[peak_idx])
print(f"Estimated DOAs: {np.sort(doa_est)}")
```

### Example 2: Root-MUSIC (fast, no angle scanning)

```python
from spatial_spectrum import root_music

doa_est = root_music(R, array, num_sources=3,
                     d=array.spacing, wavelength=array.wavelength)
print(f"Root-MUSIC DOAs: {np.rad2deg(doa_est)}")
```

### Example 3: ESPRIT for fast computation

```python
from spatial_spectrum import ls_esprit, tls_esprit

doa_ls = ls_esprit(R, num_sources=3,
                   d=array.spacing, wavelength=array.wavelength)
doa_tls = tls_esprit(R, num_sources=3,
                     d=array.spacing, wavelength=array.wavelength)
print(f"LS-ESPRIT: {np.rad2deg(doa_ls)}")
print(f"TLS-ESPRIT: {np.rad2deg(doa_tls)}")
```

## Module Overview

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `spatial_spectrum.core` | Array geometries, signals, covariance | `UniformLinearArray`, `lfm_signal`, `covariance_matrix` |
| `spatial_spectrum.algorithms` | DOA estimation algorithms | `music`, `ls_esprit`, `root_music`, `cbf` |
| `spatial_spectrum.detection` | Source number estimation | `detect_sources`, `aic`, `mdl`, `gde_criterion` |
| `spatial_spectrum.calibration` | Array error calibration | `self_calibration_gain_phase`, `active_position_calibration` |
| `spatial_spectrum.utils` | Helper functions | `spatial_smoothing`, `db`, `projection_matrix` |

## Package Structure

```
spatial_spectrum/
├── __init__.py
├── pyproject.toml
├── README.md
├── API_REFERENCE.md
├── CONTRIBUTING.md
├── core/
│   ├── __init__.py
│   ├── array.py          # ULA, UCA, URA classes
│   ├── signals.py        # LFM, CW signal generation
│   └── covariance.py     # Covariance estimation, eigendecomposition
├── algorithms/
│   ├── __init__.py
│   ├── beamforming.py    # CBF, Capon/MVDR
│   ├── music.py          # MUSIC, Root-MUSIC, Beamspace, MNM
│   ├── esprit.py         # LS-ESPRIT, TLS-ESPRIT, 2D-ESPRIT
│   ├── ml.py             # ML, MODE, IQML
│   ├── subspace_fitting.py  # SSF, NSF, WSSF
│   ├── linear_prediction.py # Forward/Backward LP
│   ├── wideband.py       # TCT, ISM
│   ├── distributed.py    # Distributed source estimation
│   └── high_order.py     # 4th-order cumulant MUSIC
├── detection/
│   ├── __init__.py
│   └── source_number.py  # AIC, MDL, HQ, EDC, GDE
├── calibration/
│   ├── __init__.py
│   ├── gain_phase.py     # Gain/phase error calibration
│   └── position.py       # Position error calibration
└── utils/
    ├── __init__.py
    ├── smoothing.py      # Spatial smoothing
    └── math_utils.py     # dB conversion, projection matrices
```

## Algorithm Reference

Based on *"Spatial Spectrum Estimation Theory and Algorithms"* by Wang Yongliang:

| Chapter | Topic | Algorithms Implemented |
|---------|-------|----------------------|
| Ch. 2 | Basics | Array geometry, resolution, ambiguity |
| Ch. 3 | Linear Prediction | CBF, Forward/Backward LP |
| Ch. 4 | MUSIC | MUSIC, Root-MUSIC, Beamspace, MNM, smoothing |
| Ch. 5 | ML & Subspace Fitting | DML, SSF, NSF, WSSF, MODE |
| Ch. 6 | ESPRIT | LS-ESPRIT, TLS-ESPRIT, 2D-ESPRIT |
| Ch. 7 | Subspace Iteration | (Iterative methods) |
| Ch. 8 | Wideband | TCT, ISM |
| Ch. 10 | Distributed Sources | Distributed source MUSIC |
| Ch. 11 | Special Arrays | UCA, virtual array, mode space |
| Ch. 12 | High-Order Statistics | HO-MUSIC (4th-order cumulant) |
| Ch. 13 | Array Calibration | Gain/phase, position (active & self) |
| Ch. 14 | Multi-dimensional | 2D MUSIC (azimuth-elevation) |

## Citation

If you use this library in your research, please cite:

```bibtex
@book{wang2004spatial,
  title={Spatial Spectrum Estimation Theory and Algorithms},
  author={Wang, Yongliang},
  year={2004},
  publisher={Tsinghua University Press}
}

@software{spatial_spectrum_lib,
  title={spatial-spectrum: Python Library for Spatial Spectrum Estimation},
  year={2026},
  url={https://github.com/username/spatial-spectrum}
}
```

## License

MIT License. See LICENSE file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
