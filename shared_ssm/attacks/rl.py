"""
Torch-based posterior attacks for RL objectives.

This module provides a small generic interface for attacks whose objective is
defined by a PyTorch callable over posterior state samples. The callable can
close over a policy, value function, goal, reward model, or any other RL
quantity needed by the experiment script.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..constraints import project_to_ellipsoid
from ..geometry import AttackGeometry
from ..linalg import project_to_psd
from ..linalg import sqrtm_psd


TorchAttackDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class TorchRLAttackConfig:
    """Configuration for Torch-based RL attacks."""

    direction: TorchAttackDirection = "minimize"
    step_size: float = 0.05
    num_steps: int = 80
    num_mc_samples: int = 64
    seed: int = 1234
    device: str = "cpu"


@dataclass(frozen=True)
class TorchRLAttackResult:
    """Result of one Torch-based RL posterior attack."""

    adversarial_observation: np.ndarray
    objective_value: float
    observation_history: np.ndarray
    objective_history: np.ndarray
    geometry: AttackGeometry


def solve_torch_rl_expectation_attack(
    *,
    geometry: AttackGeometry,
    clean_observation: np.ndarray,
    objective: Callable[["torch.Tensor"], "torch.Tensor"],
    config: TorchRLAttackConfig,
) -> TorchRLAttackResult:
    """
    Optimize a differentiable Torch objective over the attack ellipsoid.

    The objective receives posterior state samples with shape
    `(num_mc_samples, d_s)` and must return either a scalar tensor or one value
    per sample. The attack optimizes the mean objective value.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Torch RL attacks require torch to be installed.") from exc

    if config.direction not in ("minimize", "maximize"):
        raise ValueError("direction must be either 'minimize' or 'maximize'.")
    if float(config.step_size) <= 0.0:
        raise ValueError("step_size must be positive.")
    if int(config.num_steps) <= 0:
        raise ValueError("num_steps must be positive.")
    if int(config.num_mc_samples) <= 0:
        raise ValueError("num_mc_samples must be positive.")

    device = torch.device(config.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(config.seed))

    clean_observation = np.asarray(clean_observation, dtype=float).reshape(-1)
    current_observation = geometry.constraint.project(clean_observation)

    posterior_covariance = project_to_psd(geometry.posterior_covariance)
    posterior_sqrt = sqrtm_psd(posterior_covariance)
    xi = torch.randn(
        (int(config.num_mc_samples), geometry.state_mean_without_observation.size),
        generator=gen,
        device=device,
        dtype=torch.float32,
    )
    posterior_sqrt_t = torch.tensor(posterior_sqrt, dtype=torch.float32, device=device)
    state_mean_without_obs_t = torch.tensor(
        geometry.state_mean_without_observation,
        dtype=torch.float32,
        device=device,
    )
    observation_mean_without_obs_t = torch.tensor(
        geometry.observation_mean_without_observation,
        dtype=torch.float32,
        device=device,
    )
    posterior_gain_t = torch.tensor(geometry.posterior_gain, dtype=torch.float32, device=device)

    observation_history: list[np.ndarray] = []
    objective_history: list[float] = []
    objective_value = np.inf

    for _ in range(int(config.num_steps)):
        observation_t = torch.tensor(
            current_observation,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        posterior_mean_t = state_mean_without_obs_t + posterior_gain_t @ (
            observation_t - observation_mean_without_obs_t
        )
        samples_t = posterior_mean_t.unsqueeze(0) + xi @ posterior_sqrt_t.T
        objective_t = objective(samples_t)
        objective_mean_t = torch.mean(objective_t)
        objective_mean_t.backward()

        gradient = observation_t.grad.detach().cpu().numpy().astype(float)
        sign = 1.0 if config.direction == "minimize" else -1.0
        current_observation = current_observation - sign * float(config.step_size) * gradient
        current_observation = project_to_ellipsoid(
            observation=current_observation,
            center=geometry.constraint.center,
            covariance=geometry.constraint.covariance,
            epsilon=geometry.constraint.epsilon,
        )

        objective_value = float(objective_mean_t.detach().cpu().item())
        observation_history.append(current_observation.copy())
        objective_history.append(objective_value)

    final_observation_t = torch.tensor(current_observation, dtype=torch.float32, device=device)
    final_posterior_mean_t = state_mean_without_obs_t + posterior_gain_t @ (
        final_observation_t - observation_mean_without_obs_t
    )
    final_samples_t = final_posterior_mean_t.unsqueeze(0) + xi @ posterior_sqrt_t.T
    final_objective_t = torch.mean(objective(final_samples_t))
    objective_value = float(final_objective_t.detach().cpu().item())

    return TorchRLAttackResult(
        adversarial_observation=current_observation,
        objective_value=objective_value,
        observation_history=np.asarray(observation_history),
        objective_history=np.asarray(objective_history),
        geometry=geometry,
    )
