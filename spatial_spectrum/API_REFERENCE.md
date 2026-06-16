# API Reference

Complete API documentation for the spatial_spectrum library.

---

## Core Module

### UniformLinearArray(num_elements, d=None, wavelength=None)

Uniform Linear Array.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| num_elements | int | - | Number of array elements |
| d | float | None | Element spacing |
| wavelength | float | None | Signal wavelength |

Methods: steering_vector(theta), steering_matrix(theta)
Attributes: num_elements, spacing, wavelength, positions

### UniformCircularArray(num_elements, radius=None, wavelength=None)

Uniform Circular Array. Methods: steering_vector(theta, phi=0.0), steering_matrix(theta, phi=0.0).

### UniformRectangularArray(nx, ny, dx=None, dy=None, wavelength=None)

Uniform Rectangular Array on x-y plane.

### Convenience Functions

- steering_vector_ula(theta, M, d=0.5, wavelength=1.0) -> np.ndarray
- steering_vector_uca(theta, phi, M, radius, wavelength=1.0) -> np.ndarray
- steering_vector_ura(theta, phi, nx, ny, dx=0.5, dy=0.5, wavelength=1.0) -> np.ndarray
- array_resolution(M, theta_scan, theta0, d, wavelength) -> np.ndarray
- array_ambiguity(theta, theta0, M, d, wavelength) -> np.ndarray

### Signal Generation

lfm_signal(f0, u, t, initial_phase=0.0) -> np.ndarray
  Generate complex LFM chirp signal: s(t) = exp(1j * 2*pi * (f0*t + 0.5*u*t^2))

cw_signal(f0, t, initial_phase=0.0) -> np.ndarray
  Generate continuous wave tone.

generate_array_data(steering_matrix, source_signals, snr_db=inf, noise_type=gaussian, random_seed=None) -> np.ndarray
  Generate array received data X = A*S + N.

### Covariance

- covariance_matrix(X) -> np.ndarray
- eigen_decomposition(R) -> Tuple[ndarray, ndarray]
- noise_subspace(eigenvectors, num_sources) -> np.ndarray
- signal_subspace(eigenvectors, num_sources) -> np.ndarray

---

## Algorithms Module

### Beamforming
- cbf(R, array, theta_scan) -> np.ndarray
- capon_mvdr(R, array, theta_scan, diagonal_loading=0.0) -> np.ndarray

### MUSIC Family
- music(R, array, theta_scan, num_sources, mode=noise, log_scale=False) -> np.ndarray
- root_music(R, array, num_sources, d, wavelength, method=sum_columns) -> np.ndarray
- beamspace_music(R, array, theta_scan, num_sources, beamforming_matrix) -> np.ndarray
- spatial_smoothing_music(R, array, theta_scan, num_sources, num_subarrays) -> np.ndarray
- mnm(R, array, theta_scan, num_sources, log_scale=False) -> np.ndarray

### ESPRIT Family
- ls_esprit(R, num_sources, d, wavelength) -> np.ndarray
- tls_esprit(R, num_sources, d, wavelength) -> np.ndarray
- two_d_esprit(R, nx, ny, num_sources, dx, dy, wavelength) -> Tuple[ndarray, ndarray]

### ML and Subspace Fitting
- maximum_likelihood(R, array, theta_scan, num_sources) -> np.ndarray
- ssf(R, array, theta_scan, num_sources) -> np.ndarray
- nsf(R, array, theta_scan, num_sources) -> np.ndarray
- wssf(R, array, theta_scan, num_sources) -> np.ndarray
- mode_algorithm(R, array, num_sources, d, wavelength) -> np.ndarray
- iqml(R, array, num_sources, d, wavelength, max_iter=10) -> np.ndarray

### Linear Prediction
- lp_forward(X, array, theta_scan, order=1) -> np.ndarray
- lp_backward(X, array, theta_scan, order=1) -> np.ndarray
- lp_forward_backward(X, array, theta_scan, order=1) -> np.ndarray

### Wideband
- tct_doa(X_segments, array, theta_scan, num_sources, focusing_freq_idx=None) -> np.ndarray
- ism_doa(X_segments, array, theta_scan, num_sources) -> np.ndarray

### Distributed Sources
- distributed_source_music(R, theta_scan, delta_scan, num_sources, d, wavelength, distribution=triangular, mode=eigenvalue) -> np.ndarray
- distributed_steering_vector(theta_center, delta, M, d, wavelength, distribution=triangular) -> np.ndarray

### High-Order Statistics
- ho_music(X, array, theta_scan, num_sources) -> np.ndarray
- fourth_order_cumulant_matrix(X) -> np.ndarray

---

## Detection Module

- detect_sources(eigenvalues, num_snapshots, method=mdl) -> int
- aic(eigenvalues, num_snapshots) -> np.ndarray
- mdl(eigenvalues, num_snapshots) -> np.ndarray
- hq(eigenvalues, num_snapshots) -> np.ndarray
- edc(eigenvalues, num_snapshots) -> np.ndarray
- gde_criterion(R, dl=0.5) -> Tuple[ndarray, ndarray]

---

## Calibration Module

### Gain/Phase
- active_gain_phase_calibration(X, array, known_doa, num_sources=1) -> np.ndarray
- self_calibration_gain_phase(X, array, num_sources, max_iter=50, tol=1e-12) -> Tuple[ndarray, ndarray]
- taylor_series_calibration(X, array, known_doas, num_sources, max_iter=10) -> np.ndarray

### Position
- active_position_calibration(X, array, known_doas, num_sources, nominal_positions, method=disjoint) -> np.ndarray
- self_calibration_position(X, array, num_sources, nominal_positions, max_iter=50) -> Tuple[ndarray, ndarray]

---

## Utils Module

- spatial_smoothing(R, num_subarrays, mode=forward_backward) -> np.ndarray
- forward_backward_averaging(R) -> np.ndarray
- db(x, reference=0.0) -> np.ndarray
- hermitian(A) -> np.ndarray
- projection_matrix(a) -> np.ndarray
- steering_matrix(array, theta, ...) -> np.ndarray
