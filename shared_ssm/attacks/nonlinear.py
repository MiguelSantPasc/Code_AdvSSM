"""
Projected-gradient attacks for nonlinear functions of the posterior state.

The attack optimizes an arbitrary Python objective through the affine posterior
mean induced by one attacked observation. Analytic gradients are accepted when
available; otherwise finite differences are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from ..constraints import project_to_ellipsoid
from ..geometry import AttackGeometry
from ..linalg import project_to_psd
from ..linalg import sqrtm_psd


ArrayFunction = Callable[[np.ndarray], np.ndarray | float]


@dataclass(frozen=True)
class NonlinearAttackConfig:
    """Configuration for projected-gradient nonlinear attacks."""

    target_value: np.ndarray | float
    step_size: float = 0.05
    num_steps: int = 500
    num_mc_samples: int = 128
    seed: int = 1234


@dataclass(frozen=True)
class NonlinearAttackResult:
    """Result of one nonlinear projected-gradient attack."""

    adversarial_observation: np.ndarray
    objective_value: float
    expected_value: np.ndarray
    observation_history: np.ndarray
    objective_history: np.ndarray
    expected_value_history: np.ndarray
    geometry: AttackGeometry


def solve_nonlinear_expectation_attack(
    *,
    geometry: AttackGeometry,
    clean_observation: np.ndarray,
    objective_function: ArrayFunction,
    objective_jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
    config: NonlinearAttackConfig,
) -> NonlinearAttackResult:
    """
    Minimize `||E[g(s_t) | o_t'] - target||^2` over the feasible ellipsoid.

    The posterior covariance does not depend on `o_t'`, so common random
    numbers keep the Monte Carlo objective smooth across PGD iterations.
    """
    if float(config.step_size) <= 0.0:
        raise ValueError("step_size must be positive.")
    if int(config.num_steps) <= 0:
        raise ValueError("num_steps must be positive.")
    if int(config.num_mc_samples) <= 0:
        raise ValueError("num_mc_samples must be positive.")

    rng = np.random.default_rng(int(config.seed))
    clean_observation = np.asarray(clean_observation, dtype=float).reshape(-1)
    target_value = _as_1d_output(config.target_value)
    y_current = geometry.constraint.project(clean_observation)

    posterior_covariance = project_to_psd(geometry.posterior_covariance)
    posterior_sqrt = sqrtm_psd(posterior_covariance)
    state_dim = geometry.state_mean_without_observation.size
    xi = rng.standard_normal(size=(int(config.num_mc_samples), state_dim))

    observation_history: list[np.ndarray] = []
    objective_history: list[float] = []
    expected_value_history: list[np.ndarray] = []

    expected_value = np.zeros_like(target_value)
    objective_value = np.inf

    for _ in range(int(config.num_steps)):
        expected_value, jacobian_mean = _estimate_expectation_and_jacobian(
            observation=y_current,
            geometry=geometry,
            objective_function=objective_function,
            objective_jacobian=objective_jacobian,
            posterior_sqrt=posterior_sqrt,
            xi=xi,
        )
        if expected_value.shape != target_value.shape:
            raise ValueError("objective output and target_value have different shapes.")

        diff = expected_value - target_value
        objective_value = float(np.dot(diff, diff))
        d_expected_d_observation = jacobian_mean @ geometry.posterior_gain
        gradient = 2.0 * (d_expected_d_observation.T @ diff)

        y_current = y_current - float(config.step_size) * gradient
        y_current = project_to_ellipsoid(
            observation=y_current,
            center=geometry.constraint.center,
            covariance=geometry.constraint.covariance,
            epsilon=geometry.constraint.epsilon,
        )

        observation_history.append(y_current.copy())
        objective_history.append(objective_value)
        expected_value_history.append(expected_value.copy())

    expected_value, _jacobian_mean = _estimate_expectation_and_jacobian(
        observation=y_current,
        geometry=geometry,
        objective_function=objective_function,
        objective_jacobian=objective_jacobian,
        posterior_sqrt=posterior_sqrt,
        xi=xi,
    )
    final_diff = expected_value - target_value
    objective_value = float(np.dot(final_diff, final_diff))

    return NonlinearAttackResult(
        adversarial_observation=y_current,
        objective_value=objective_value,
        expected_value=expected_value,
        observation_history=np.asarray(observation_history),
        objective_history=np.asarray(objective_history),
        expected_value_history=np.asarray(expected_value_history),
        geometry=geometry,
    )


def estimate_expectation(
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    function: ArrayFunction,
    num_mc_samples: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate `E[function(x)]` and `function(E[x])` under a Gaussian law."""
    rng = np.random.default_rng(int(seed))
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(covariance, dtype=float))
    samples = mean[None, :] + rng.standard_normal(size=(int(num_mc_samples), mean.size)) @ sqrtm_psd(covariance).T
    values = np.stack([_as_1d_output(function(sample)) for sample in samples], axis=0)
    return np.mean(values, axis=0), _as_1d_output(function(mean))


def finite_difference_jacobian(
    function: ArrayFunction,
    x_val: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Return a centered finite-difference Jacobian for scalar or vector outputs."""
    x_val = np.asarray(x_val, dtype=float).reshape(-1)
    base_output = _as_1d_output(function(x_val))
    jacobian = np.zeros((base_output.size, x_val.size), dtype=float)

    for dim_idx in range(x_val.size):
        step = np.zeros_like(x_val)
        step[dim_idx] = float(h)
        plus = _as_1d_output(function(x_val + step))
        minus = _as_1d_output(function(x_val - step))
        jacobian[:, dim_idx] = (plus - minus) / (2.0 * float(h))

    return jacobian


def _estimate_expectation_and_jacobian(
    *,
    observation: np.ndarray,
    geometry: AttackGeometry,
    objective_function: ArrayFunction,
    objective_jacobian: Callable[[np.ndarray], np.ndarray] | None,
    posterior_sqrt: np.ndarray,
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate both `E[g(s_t)]` and `E[J_g(s_t)]` with common samples."""
    posterior_mean = geometry.posterior_mean_for_observation(observation)
    samples = posterior_mean[None, :] + xi @ posterior_sqrt.T
    values = np.stack([_as_1d_output(objective_function(sample)) for sample in samples], axis=0)

    if objective_jacobian is None:
        jacobians = np.stack(
            [finite_difference_jacobian(objective_function, sample) for sample in samples],
            axis=0,
        )
    else:
        jacobians = np.stack(
            [_as_2d_jacobian(objective_jacobian(sample), n_state=posterior_mean.size) for sample in samples],
            axis=0,
        )

    return np.mean(values, axis=0), np.mean(jacobians, axis=0)


def _as_1d_output(value: np.ndarray | float) -> np.ndarray:
    """Normalize scalar or vector function outputs to shape `(d_out,)`."""
    return np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)


def _as_2d_jacobian(jacobian: np.ndarray, *, n_state: int) -> np.ndarray:
    """Normalize a gradient/Jacobian to shape `(d_out, d_s)`."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim == 1:
        if jacobian.shape[0] != int(n_state):
            raise ValueError("1D gradient has incompatible shape.")
        return jacobian[None, :]
    if jacobian.ndim == 2:
        if jacobian.shape[1] != int(n_state):
            raise ValueError("Jacobian has incompatible state dimension.")
        return jacobian
    raise ValueError("Jacobian must be a vector or a matrix.")
