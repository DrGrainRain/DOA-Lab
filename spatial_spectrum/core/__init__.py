"""Core: array geometry, steering vectors, signal generation, covariance estimation."""

from .array import (
    UniformLinearArray, UniformCircularArray, UniformRectangularArray,
    steering_vector_ula, steering_vector_uca, steering_vector_ura,
    array_resolution, array_ambiguity,
)
from .signals import lfm_signal, cw_signal, generate_array_data
from .covariance import covariance_matrix, eigen_decomposition, noise_subspace, signal_subspace

__all__ = [
    "UniformLinearArray","UniformCircularArray","UniformRectangularArray",
    "steering_vector_ula","steering_vector_uca","steering_vector_ura",
    "array_resolution","array_ambiguity",
    "lfm_signal","cw_signal","generate_array_data",
    "covariance_matrix","eigen_decomposition","noise_subspace","signal_subspace",
]
