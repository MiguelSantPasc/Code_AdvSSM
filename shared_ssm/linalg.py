"""
Numerically stable SPD linear-algebra helpers for the shared SSM package.

Why this module exists:
1. The repository currently repeats the same PSD projection and matrix-inverse
   helpers in many scripts.
2. Filtering, attacks, and covariance adaptation all rely on the same SPD
   primitives, so they should live in one place.
3. Centralizing these helpers makes it easier to keep numerical conventions
   consistent across linear, nonlinear, and RL-based experiments.

The functions below operate on symmetric positive semidefinite or positive
definite matrices and favor solve-based implementations over explicit matrix
inversion whenever practical.
"""

from __future__ import annotations

import numpy as np


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of `matrix`."""
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def project_to_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project a square matrix onto the PSD cone.

    The implementation symmetrizes the input, clips negative eigenvalues, and
    reconstructs a PSD approximation.
    """
    matrix = symmetrize(np.asarray(matrix, dtype=float))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.maximum(eigenvalues, float(eps))
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T


def sqrtm_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return a PSD square root of `matrix`."""
    matrix = symmetrize(np.asarray(matrix, dtype=float))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.maximum(eigenvalues, float(eps))
    return eigenvectors @ np.diag(np.sqrt(clipped)) @ eigenvectors.T


def stabilized_cholesky(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Return a Cholesky factor of a stabilized SPD approximation of `matrix`.

    A small diagonal jitter is increased only when needed.
    """
    matrix = project_to_psd(np.asarray(matrix, dtype=float), eps=eps)
    eye = np.eye(matrix.shape[0], dtype=float)
    jitter = float(eps)

    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    raise np.linalg.LinAlgError("Failed to compute a stable Cholesky factor.")


def solve_spd(matrix: np.ndarray, rhs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Solve `matrix @ x = rhs` for SPD `matrix` using Cholesky factorization.

    The right-hand side can be a vector or a matrix.
    """
    chol = stabilized_cholesky(matrix, eps=eps)
    y_val = np.linalg.solve(chol, np.asarray(rhs, dtype=float))
    return np.linalg.solve(chol.T, y_val)


def spd_inverse(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return the inverse of an SPD matrix via solve calls."""
    matrix = np.asarray(matrix, dtype=float)
    identity = np.eye(matrix.shape[0], dtype=float)
    return project_to_psd(solve_spd(matrix, identity, eps=eps), eps=eps)


def quad_form_spd(matrix: np.ndarray, vector: np.ndarray, eps: float = 1e-10) -> float:
    """Return `vector.T @ matrix^{-1} @ vector` using a stable solve."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    solved = solve_spd(matrix, vector, eps=eps)
    return float(np.dot(vector, solved))


def gaussian_logpdf(
    x_val: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Evaluate the multivariate Gaussian log-density.

    The covariance is stabilized before factorization, and the computation uses
    Cholesky solves rather than an explicit inverse.
    """
    x_val = np.asarray(x_val, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(covariance, dtype=float), eps=eps)

    chol = stabilized_cholesky(covariance, eps=eps)
    diff = x_val - mean
    whitened = np.linalg.solve(chol, diff)
    quadratic_term = float(np.dot(whitened, whitened))
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    dim = x_val.size
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quadratic_term)
