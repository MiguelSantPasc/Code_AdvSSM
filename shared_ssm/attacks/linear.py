"""
Analytic linear attacks for the shared SSM package.

The main helper preserves the closed-form trust-region attack used by the
current AdvSSM scripts. It maximizes a quadratic effect induced by a linear map
from observation perturbations to the attacked state quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import AttackGeometry
from ..linalg import project_to_psd
from ..linalg import sqrtm_psd
from ..linalg import symmetrize


@dataclass(frozen=True)
class LinearAttackResult:
    """Result of one analytic linear attack."""

    adversarial_observation: np.ndarray
    objective_value: float
    geometry: AttackGeometry
    effect_matrix: np.ndarray
    target_observation: np.ndarray


def solve_linear_state_attack(
    *,
    geometry: AttackGeometry,
    clean_observation: np.ndarray,
    state_transform: np.ndarray | None = None,
) -> LinearAttackResult:
    """
    Maximize the state-posterior mean displacement inside the attack ellipsoid.

    The default objective is:
        ||K_t (o_t^* - o_t)||^2

    where `K_t` is the posterior affine map from attacked observation to state
    mean under either online or offline attack geometry. `state_transform`
    optionally replaces this with `||L K_t (o_t^* - o_t)||^2`.
    """
    clean_observation = np.asarray(clean_observation, dtype=float).reshape(-1)
    if state_transform is None:
        effect_matrix = geometry.posterior_gain
    else:
        state_transform = np.asarray(state_transform, dtype=float)
        effect_matrix = state_transform @ geometry.posterior_gain

    adversarial_observation, objective_value = solve_max_quadratic_over_ellipsoid(
        effect_matrix=effect_matrix,
        reference_observation=clean_observation,
        center=geometry.constraint.center,
        covariance=geometry.constraint.covariance,
        epsilon=geometry.constraint.epsilon,
    )
    return LinearAttackResult(
        adversarial_observation=adversarial_observation,
        objective_value=float(objective_value),
        geometry=geometry,
        effect_matrix=effect_matrix,
        target_observation=clean_observation,
    )


def solve_max_quadratic_over_ellipsoid(
    *,
    effect_matrix: np.ndarray,
    reference_observation: np.ndarray,
    center: np.ndarray,
    covariance: np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 250,
) -> tuple[np.ndarray, float]:
    """
    Solve the analytic KKT attack:

        max_o ||M (o - o_ref)||^2
        s.t.  (o - center)^T covariance^{-1} (o - center) <= epsilon.
    """
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive.")

    effect_matrix = np.asarray(effect_matrix, dtype=float)
    reference_observation = np.asarray(reference_observation, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(covariance, dtype=float))

    covariance_sqrt = sqrtm_psd(covariance)
    quadratic_matrix = project_to_psd(symmetrize(effect_matrix.T @ effect_matrix))
    center_offset = center - reference_observation
    A_mat = symmetrize(covariance_sqrt.T @ quadratic_matrix @ covariance_sqrt)
    b_vec = covariance_sqrt.T @ quadratic_matrix @ center_offset

    eigenvalues, eigenvectors = np.linalg.eigh(A_mat)
    lambda_floor = float(np.max(eigenvalues))
    rotated_b = eigenvectors.T @ b_vec

    if float(np.linalg.norm(b_vec)) < 1e-14:
        top_idx = int(np.argmax(eigenvalues))
        z_rotated = np.zeros_like(rotated_b)
        z_rotated[top_idx] = np.sqrt(float(epsilon))
        z_star = eigenvectors @ z_rotated
    else:
        def radius_gap(lam: float) -> float:
            z_rotated_local = -rotated_b / (eigenvalues - lam)
            return float(np.dot(z_rotated_local, z_rotated_local) - float(epsilon))

        lam_low = lambda_floor + 1e-12
        if radius_gap(lam_low) <= 0.0:
            lam_low = lambda_floor + 1e-16

        lam_high = lambda_floor + 1.0
        while radius_gap(lam_high) > 0.0:
            lam_high *= 2.0
            if lam_high > 1e14:
                raise RuntimeError("Failed to bracket lambda for KKT attack.")

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
        z_star = eigenvectors @ (-rotated_b / (eigenvalues - lam_star))
        z_norm = float(np.linalg.norm(z_star))
        if z_norm > 0.0:
            z_star = z_star * (np.sqrt(float(epsilon)) / z_norm)

    adversarial_observation = center + covariance_sqrt @ z_star
    objective_value = float(np.linalg.norm(effect_matrix @ (adversarial_observation - reference_observation)) ** 2)
    return adversarial_observation, objective_value
