# Contributing Guide

Thank you for your interest in contributing to spatial-spectrum!

## Development Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/username/spatial-spectrum.git
cd spatial-spectrum
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install in development mode

```bash
pip install -e ".[dev]"
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Project Structure

```
spatial_spectrum/
├── __init__.py         # Package entry point, re-exports all public APIs
├── pyproject.toml      # Build configuration and dependencies
├── README.md           # User-facing documentation
├── API_REFERENCE.md    # Complete API reference
├── CONTRIBUTING.md     # This file
├── core/               # Foundational modules
│   ├── __init__.py
│   ├── array.py        # ULA, UCA, URA classes
│   ├── signals.py      # LFM, CW, data generation
│   └── covariance.py   # Covariance, eigendecomposition
├── algorithms/         # DOA estimation algorithms
│   ├── __init__.py
│   ├── beamforming.py
│   ├── music.py
│   ├── esprit.py
│   ├── ml.py
│   ├── subspace_fitting.py
│   ├── linear_prediction.py
│   ├── wideband.py
│   ├── distributed.py
│   └── high_order.py
├── detection/          # Source number estimation
│   ├── __init__.py
│   └── source_number.py
├── calibration/        # Array error calibration
│   ├── __init__.py
│   ├── gain_phase.py
│   └── position.py
└── utils/              # Helper functions
    ├── __init__.py
    ├── smoothing.py
    └── math_utils.py
```

## Coding Standards

- **Style**: Follow PEP 8. Use `black` for auto-formatting and `isort` for import sorting.
- **Type hints**: All public functions should have complete type annotations.
- **Docstrings**: Use NumPy-style docstrings with Parameters and Returns sections.
- **Naming**: Classes use PascalCase, functions/variables use snake_case.

## Adding a New Algorithm

1. Create the implementation file in the appropriate module directory.
2. Add the function to the module's `__init__.py`.
3. Add it to the main `spatial_spectrum/__init__.py` for top-level access.
4. Update `API_REFERENCE.md` with the new function signature.
5. Add tests in the `tests/` directory.
6. Update `README.md` if the algorithm belongs to a chapter.

## Algorithm Naming Convention

Use the standard names from the textbook:
- `music` - Standard MUSIC
- `root_music` - Root-MUSIC
- `ls_esprit` - LS-ESPRIT
- `tls_esprit` - TLS-ESPRIT

## Running Code Quality Checks

```bash
black spatial_spectrum/ tests/
isort spatial_spectrum/ tests/
mypy spatial_spectrum/
```

## Questions?

Open an issue on GitHub or contact the maintainers.
