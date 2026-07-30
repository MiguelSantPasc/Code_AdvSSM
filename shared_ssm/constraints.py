"""
Shared plausibility constraints for adversarial observations.

These helpers centralize the likelihood, log-likelihood, and Mahalanobis
regions that were previously implemented separately in experiment scripts.
They are intentionally small and array-oriented so linear, nonlinear, and RL
attacks can all reuse the same feasibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linalg import gaussian_logpdf
from .linalg import project_to_psd
from .linalg import quad_form_spd
from .linalg import sqrtm_psd


@dataclass(frozen=True)
class EllipsoidConstraint:
    """
    Mahalanobis ellipsoid for plausible adversarial observations.

    The feasible set is:
        (o - center)^T covariance^{-1} (o - center) <= epsilon.
    """

    center: np.ndarray
    covariance: np.ndarray
    epsilon: float

    def __post_init__(self) -> None:
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")

    def radius_squared(self, observation: np.ndarray) -> float:
        """Return the Mahalanobis radius squared of `observation`."""
        return mahalanobis_distance_squared(
            observation=observation,
            center=self.center,
            covariance=self.covariance,
        )

    def contains(self, observation: np.ndarray, tol: float = 1e-10) -> bool:
        """Return whether `observation` lies inside the ellipsoid."""
        return bool(self.radius_squared(observation) <= float(self.epsilon) + float(tol))

    def project(self, observation: np.ndarray) -> np.ndarray:
        """Project `observation` onto this ellipsoid in Euclidean distance."""
        return project_to_ellipsoid(
            observation=observation,
            center=self.center,
            covariance=self.covariance,
            epsilon=float(self.epsilon),
        )


def mahalanobis_distance_squared(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return `(observation - center)^T covariance^{-1} (observation - center)`."""
    observation = np.asarray(observation, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(covariance, dtype=float))
    return quad_form_spd(covariance, observation - center)


def gaussian_likelihood(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return the Gaussian likelihood of `observation` under the constraint law."""
    return float(np.exp(gaussian_log_likelihood(
        observation=observation,
        center=center,
        covariance=covariance,
    )))


def gaussian_log_likelihood(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return the Gaussian log-likelihood of `observation`."""
    return gaussian_logpdf(observation, center, covariance)


def project_to_ellipsoid(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    covariance: np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Return the Euclidean projection of `observation` onto a Mahalanobis ellipsoid.

    The projection solves:
        min_z ||z - observation||^2
        s.t.  (z - center)^T covariance^{-1} (z - center) <= epsilon.
    """
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive.")

    observation = np.asarray(observation, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(covariance, dtype=float))

    radius_sq = mahalanobis_distance_squared(
        observation=observation,
        center=center,
        covariance=covariance,
    )
    if radius_sq <= float(epsilon) + float(tol):
        return observation.copy()

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    rotated_diff = eigenvectors.T @ (observation - center)

    def radius_gap(lam: float) -> float:
        scaled = (eigenvalues * rotated_diff**2) / (eigenvalues + lam) ** 2
        return float(np.sum(scaled) - float(epsilon))

    lam_low = 0.0
    lam_high = 1.0
    while radius_gap(lam_high) > 0.0:
        lam_high *= 2.0
        if lam_high > 1e14:
            raise RuntimeError("Could not bracket lambda for ellipsoid projection.")

    for _ in range(int(max_iter)):
        lam_mid = 0.5 * (lam_low + lam_high)
        gap = radius_gap(lam_mid)
        if abs(gap) < float(tol):
            lam_low = lam_high = lam_mid
            break
        if gap > 0.0:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    lam_star = 0.5 * (lam_low + lam_high)
    projected_rotated = (eigenvalues / (eigenvalues + lam_star)) * rotated_diff
    return center + eigenvectors @ projected_rotated


def sample_from_ellipsoid_boundary(
    *,
    center: np.ndarray,
    covariance: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one random point on the ellipsoid boundary."""
    center = np.asarray(center, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(covariance, dtype=float))
    direction = rng.normal(size=center.size)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return center + np.sqrt(float(epsilon)) * (sqrtm_psd(covariance) @ direction)


def sample_from_ellipsoid(
    *,
    center: np.ndarray,
    covariance: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one random point inside the ellipsoid with uniform radial scaling."""
    center = np.asarray(center, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(covariance, dtype=float))
    direction = rng.normal(size=center.size)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    radial_scale = float(rng.random()) ** (1.0 / max(center.size, 1))
    return center + radial_scale * np.sqrt(float(epsilon)) * (sqrtm_psd(covariance) @ direction)
