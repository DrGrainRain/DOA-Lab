"""Array geometry classes and steering vector generation.

Implements: ULA, UCA, URA, and related utility functions.
References: Wang Yongliang, Spatial Spectrum Estimation Theory and Algorithms, Chapter 2.
"""

import numpy as np
from typing import Optional


class UniformLinearArray:
    """Uniform Linear Array (ULA).

    Parameters
    ----------
    num_elements : int
        Number of array elements.
    d : float, optional
        Element spacing. Defaults to half-wavelength.
    wavelength : float, optional
        Signal wavelength in meters.
    """

    def __init__(self, num_elements: int, d: Optional[float] = None, wavelength: Optional[float] = None):
        self.num_elements = num_elements
        self.wavelength = wavelength if wavelength is not None else 1.0
        self.spacing = d if d is not None else self.wavelength / 2.0
        self.positions = np.arange(num_elements) * self.spacing

    def steering_vector(self, theta: float) -> np.ndarray:
        """Compute steering vector for DOA theta (radians)."""
        k = np.arange(self.num_elements)
        return np.exp(1j * 2 * np.pi / self.wavelength * self.spacing * k * np.sin(theta))

    def steering_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Compute steering matrix for multiple DOA angles (radians)."""
        k = np.arange(self.num_elements)[:, np.newaxis]
        return np.exp(1j * 2 * np.pi / self.wavelength * self.spacing * k * np.sin(theta))


class UniformCircularArray:
    """Uniform Circular Array (UCA).

    Parameters
    ----------
    num_elements : int
        Number of elements on the circle.
    radius : float, optional
        Circle radius. Default: derived from half-wavelength equivalent spacing.
    wavelength : float, optional
    """

    def __init__(self, num_elements: int, radius: Optional[float] = None, wavelength: Optional[float] = None):
        self.num_elements = num_elements
        self.wavelength = wavelength if wavelength is not None else 1.0
        if radius is not None:
            self.radius = radius
        else:
            d_eq = self.wavelength / 2.0
            self.radius = d_eq / (2.0 * np.sin(np.pi / num_elements))
        self.element_angles = 2.0 * np.pi * np.arange(num_elements) / num_elements

    def steering_vector(self, theta: float, phi: float = 0.0) -> np.ndarray:
        """Compute steering vector for azimuth theta and elevation phi (radians)."""
        k = np.arange(self.num_elements)
        beta = 2.0 * np.pi / self.wavelength
        return np.exp(1j * beta * self.radius * np.cos(theta - 2.0 * np.pi * k / self.num_elements) * np.cos(phi))

    def steering_matrix(self, theta: np.ndarray, phi: float = 0.0) -> np.ndarray:
        """Compute steering matrix for multiple DOA angles."""
        k = np.arange(self.num_elements)[:, np.newaxis]
        beta = 2.0 * np.pi / self.wavelength
        return np.exp(1j * beta * self.radius * np.cos(theta - 2.0 * np.pi * k / self.num_elements) * np.cos(phi))


class UniformRectangularArray:
    """Uniform Rectangular Array (URA) on the x-y plane.

    Parameters
    ----------
    nx : int
        Number of elements along x-axis.
    ny : int
        Number of elements along y-axis.
    dx, dy : float, optional
        Element spacing along each axis.
    wavelength : float, optional
    """

    def __init__(self, nx: int, ny: int, dx: Optional[float] = None, dy: Optional[float] = None, wavelength: Optional[float] = None):
        self.nx = nx
        self.ny = ny
        self.num_elements = nx * ny
        self.wavelength = wavelength if wavelength is not None else 1.0
        self.dx = dx if dx is not None else self.wavelength / 2.0
        self.dy = dy if dy is not None else self.wavelength / 2.0
        xp = np.arange(nx) * self.dx
        yp = np.arange(ny) * self.dy
        xx, yy = np.meshgrid(xp, yp)
        self.positions = np.column_stack([xx.ravel(), yy.ravel()])

    def steering_vector(self, theta: float, phi: float = 0.0) -> np.ndarray:
        """Compute steering vector for azimuth theta and elevation phi (radians)."""
        k0 = 2.0 * np.pi / self.wavelength
        u = np.cos(theta) * np.cos(phi)
        v = np.sin(theta) * np.cos(phi)
        return np.exp(1j * k0 * (self.positions[:, 0] * u + self.positions[:, 1] * v))

    def steering_matrix(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Compute steering matrix for multiple DOA angles."""
        k0 = 2.0 * np.pi / self.wavelength
        u = np.cos(theta) * np.cos(phi)
        v = np.sin(theta) * np.cos(phi)
        return np.exp(1j * k0 * (np.outer(self.positions[:, 0], u) + np.outer(self.positions[:, 1], v)))


# Convenience functions

def steering_vector_ula(theta: float, M: int, d: float = 0.5, wavelength: float = 1.0) -> np.ndarray:
    """Quick ULA steering vector (theta in radians)."""
    k = np.arange(M)
    return np.exp(1j * 2.0 * np.pi * d / wavelength * k * np.sin(theta))


def steering_vector_uca(theta: float, phi: float, M: int, radius: float, wavelength: float = 1.0) -> np.ndarray:
    """Quick UCA steering vector."""
    k = np.arange(M)
    beta = 2.0 * np.pi / wavelength
    return np.exp(1j * beta * radius * np.cos(theta - 2.0 * np.pi * k / M) * np.cos(phi))


def steering_vector_ura(theta: float, phi: float, nx: int, ny: int, dx: float = 0.5, dy: float = 0.5, wavelength: float = 1.0) -> np.ndarray:
    """Quick URA steering vector."""
    k0 = 2.0 * np.pi / wavelength
    u = np.cos(theta) * np.cos(phi)
    v = np.sin(theta) * np.cos(phi)
    xp = np.arange(nx) * dx
    yp = np.arange(ny) * dy
    xx, yy = np.meshgrid(xp, yp)
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    return np.exp(1j * k0 * (pos[:, 0] * u + pos[:, 1] * v))


def array_resolution(M, theta_scan: np.ndarray, theta0: float, d: float = 0.5, wavelength: float = 1.0) -> np.ndarray:
    """Compute array resolution D(theta) = ||d a(theta)/d theta|| for ULA.

    Parameters
    ----------
    M : int or list of int
        Element count(s).
    theta_scan : np.ndarray
        Scan angles in radians.
    theta0 : float
        Reference angle (not used directly, kept for API compatibility).
    d : float
        Element spacing.
    wavelength : float

    Returns
    -------
    D : np.ndarray, shape (len(M), len(theta_scan))
    """
    if np.isscalar(M):
        M = [M]
    M = np.atleast_1d(M)
    D = np.zeros((len(M), len(theta_scan)))
    for jj, m in enumerate(M):
        a_idx = np.arange(m)
        for ii, th in enumerate(theta_scan):
            da = np.exp(1j * 2.0 * np.pi * d / wavelength * a_idx * np.sin(th)) * (
                1j * 2.0 * np.pi * d / wavelength * a_idx * np.cos(th)
            )
            D[jj, ii] = np.linalg.norm(da)
    return D


def array_ambiguity(theta: np.ndarray, theta0: float, M: int, d: float = 0.5, wavelength: float = 1.0) -> np.ndarray:
    """Compute array ambiguity function G(theta) = |a^H(theta) a(theta0)|."""
    a_idx = np.arange(M)
    a0 = np.exp(1j * 2.0 * np.pi * d / wavelength * a_idx * np.sin(theta0))
    G = np.zeros(len(theta))
    for ii, th in enumerate(theta):
        a_th = np.exp(1j * 2.0 * np.pi * d / wavelength * a_idx * np.sin(th))
        G[ii] = np.abs(np.conj(a_th) @ a0)
    return G / np.max(G)
