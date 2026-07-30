#!/usr/bin/env python3
"""
defense_benchmark.py

Reproducible RL defense benchmark for the wind-navigation experiment.

Why this module exists:
1. The existing RL scripts already cover clean, noisy, PGD, and random-contour
   baselines, but the requested benchmark needs a stricter experimental
   protocol: all methods must reuse the same seeds, goals, process noise,
   observation noise, attack decisions, and horizon whenever possible.
2. The benchmark also needs two families of defenses that should remain easy to
   compare and extend:
   - directional covariance adaptation using the PGD estimated-return
     direction,
   - WoLF-IMQ / WoLF-TMD robust filtering.
3. The WoLF hyperparameters must be tuned in a separate script before the final
   benchmark is evaluated.

Design choices implemented here:
1. The environment dynamics are replayed from pre-sampled exogenous variables,
   so every method sees comparable process and observation noise.
2. The attacker only uses predictive quantities that are available online. The
   real hidden state and the realized return are never used to optimize the
   attack or to activate the defense.
3. The defense modifies only the measurement update. The dynamics, attack
   budget, rewards, policy, and process/observation noise remain unchanged.
4. The code keeps the benchmark configuration in plain Python variables and
   command-line arguments rather than environment variables.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import argparse
import json
import os
import re
import sys
from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch


RL_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_MODULE_DIR)
if RL_MODULE_DIR not in sys.path:
    sys.path.insert(0, RL_MODULE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compare_wind_online_attack_rewards import coverage_to_mahalanobis_epsilon
from compare_wind_online_attack_rewards import estimated_return_pgd_attack_observation
from compare_wind_online_attack_rewards import random_observation_on_contour
from covaraicne_sweep import observation_std_from_ratio
from shared_ssm import AttackGeometry
from shared_ssm import EllipsoidConstraint
from shared_ssm import WoLFConfig
from shared_ssm import compute_adapted_observation_covariance
from shared_ssm import gaussian_logpdf
from shared_ssm import kalman_predict_step
from shared_ssm import kalman_update_step
from shared_ssm import project_to_psd
from shared_ssm import rank_one_covariance_update
from shared_ssm import run_wolf_measurement_update
from shared_ssm import solve_spd
from shared_ssm import spd_inverse
from shared_ssm.defenses.covariance_adaptation import log_mixture_posterior_weight
from wind_rl_setup import WindNavigationConfig
from wind_rl_setup import build_control_matrix
from wind_rl_setup import build_observation_covariance
from wind_rl_setup import build_observation_matrix
from wind_rl_setup import build_process_covariance
from wind_rl_setup import build_transition_matrix
from wind_rl_setup import default_device
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_data_dir
from wind_rl_setup import rl_figures_dir
from wind_rl_setup import rl_model_dir


@dataclass(frozen=True)
class BenchmarkMethod:
    """Describe one benchmarked attack/defense combination."""

    name: str
    attack_type: str
    filter_type: str
    use_covariance_adaptation: bool
    lambda_covariance: float
    defense_direction_source: str
    wolf_kind: str | None = None
    wolf_parameters: dict[str, float] | None = None


@dataclass(frozen=True)
class EpisodeRandomness:
    """All exogenous randomness needed to replay one comparable episode."""

    seed: int
    initial_hidden_state: np.ndarray
    goal_xy: np.ndarray
    delta_psis: np.ndarray
    process_noises: np.ndarray
    observation_noises: np.ndarray
    attack_uniforms: np.ndarray
    random_contour_seeds: np.ndarray


@dataclass(frozen=True)
class DefenseBenchmarkConfig:
    """Top-level benchmark configuration for the wind-defense study."""

    coverage: float
    attack_probability: float
    lambdas: tuple[float, ...]
    gamma_threshold: float
    goal_risk_scale: float
    omega_risk: float
    omega_observation: float
    strong_wolf_weight_threshold: float
    attack_step_size: float
    attack_num_steps: int
    attack_mc_samples: int
    transition_mc_samples: int
    attack_discount_gamma: float
    contour_relative_tolerance: float
    direction_eps: float
    base_seed: int
    n_tuning_episodes: int
    n_episodes: int
    filter_config: WindNavigationConfig
    simulation_config: WindNavigationConfig


@dataclass(frozen=True)
class StepDiagnostics:
    """Step-wise diagnostics recorded during one method rollout."""

    step_index: int
    attacked: bool
    attack_type: str
    filter_type: str
    gamma_t: float
    gamma_bar_t: float
    effective_inflation: float
    risk_expert: float
    observation_expert: float
    state_estimation_error: float
    goal_distance: float
    pgd_perturbation_norm: float
    random_contour_perturbation_norm: float
    pgd_direction_norm: float
    used_zero_pgd_direction: bool
    wolf_weight: float
    wolf_weight_squared: float
    wolf_heavily_discounted: bool


@dataclass(frozen=True)
class MethodEpisodeResult:
    """Full metrics of one method on one exogenous episode realization."""

    method: BenchmarkMethod
    episode_seed: int
    episode_return: float
    success: bool
    final_goal_distance: float
    mean_state_estimation_error: float
    steps: int
    diagnostics: list[StepDiagnostics]


def standard_error(values: np.ndarray) -> float:
    """Return the standard error, using zero for one-sample runs."""
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def set_plot_theme() -> None:
    """Apply the pastel plotting theme requested for RL figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10.0,
            "axes.labelsize": 10.8,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.68,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
            "savefig.facecolor": "white",
        }
    )


def deterministic_policy_action(
    *,
    policy: torch.nn.Module,
    state_estimate: np.ndarray,
    goal_xy: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return the policy's deterministic action for one estimated state."""
    policy_observation = np.concatenate([state_estimate, goal_xy]).astype(np.float32)
    obs_tensor = torch.as_tensor(policy_observation[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        action = policy.deterministic_action(obs_tensor).cpu().numpy()[0]
    return np.asarray(action, dtype=np.float32)


def build_filter_prior(config: WindNavigationConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a non-leaky prior for the filter at the start of the episode.

    The start position is known from the environment definition. The initial
    wind direction is unknown, but its magnitude is fixed. For a uniformly
    random angle the two wind components have zero mean and variance `m^2 / 2`.
    """
    start_xy = np.asarray(config.start_xy, dtype=float).reshape(2)
    position_variance = float(config.start_radius) ** 2 / 4.0
    wind_variance = float(config.initial_wind_magnitude) ** 2 / 2.0
    prior_mean = np.array([start_xy[0], start_xy[1], 0.0, 0.0], dtype=float)
    prior_covariance = np.diag(
        [
            max(position_variance, 1e-6),
            max(position_variance, 1e-6),
            max(wind_variance, 1e-6),
            max(wind_variance, 1e-6),
        ]
    ).astype(float)
    return prior_mean, project_to_psd(prior_covariance)


def build_initial_predict_result(
    *,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    observation_covariance: np.ndarray,
) -> Any:
    """Build the predictive observation law for the first benchmark step."""
    prior_mean = np.asarray(prior_mean, dtype=float).reshape(4)
    prior_covariance = project_to_psd(np.asarray(prior_covariance, dtype=float))
    observation_covariance = project_to_psd(np.asarray(observation_covariance, dtype=float))
    return type(
        "InitialPredictResult",
        (),
        {
            "state_mean": prior_mean.copy(),
            "state_covariance": prior_covariance.copy(),
            "observation_mean": prior_mean.copy(),
            "observation_covariance": project_to_psd(prior_covariance + observation_covariance),
        },
    )()


def build_attack_geometry_from_predict_result(
    *,
    predict_result: Any,
    coverage: float,
    observation_index: int,
) -> AttackGeometry:
    """Build the shared attack geometry directly from one predictive belief."""
    predictive_state_mean = np.asarray(predict_result.state_mean, dtype=float).reshape(4)
    predictive_state_covariance = project_to_psd(np.asarray(predict_result.state_covariance, dtype=float))
    predictive_observation_mean = np.asarray(predict_result.observation_mean, dtype=float).reshape(4)
    predictive_observation_covariance = project_to_psd(np.asarray(predict_result.observation_covariance, dtype=float))
    observation_matrix = np.eye(4, dtype=float)
    posterior_gain = solve_spd(
        predictive_observation_covariance,
        observation_matrix @ predictive_state_covariance.T,
    ).T
    posterior_covariance = project_to_psd(
        predictive_state_covariance - posterior_gain @ observation_matrix @ predictive_state_covariance
    )
    return AttackGeometry(
        mode="online",
        observation_index=int(observation_index),
        state_mean_without_observation=predictive_state_mean,
        state_covariance_without_observation=predictive_state_covariance,
        observation_mean_without_observation=predictive_observation_mean,
        observation_covariance_without_observation=predictive_observation_covariance,
        posterior_gain=posterior_gain,
        posterior_covariance=posterior_covariance,
        constraint=EllipsoidConstraint(
            center=predictive_observation_mean,
            covariance=predictive_observation_covariance,
            epsilon=float(coverage_to_mahalanobis_epsilon(float(coverage), obs_dim=4)),
        ),
        inference_without_observation=None,  # type: ignore[arg-type]
    )


def compute_goal_proximity_risk(
    *,
    predicted_state_mean: np.ndarray,
    goal_xy: np.ndarray,
    goal_risk_scale: float,
) -> float:
    """Return the goal-proximity risk expert requested for the defense."""
    if float(goal_risk_scale) <= 0.0:
        raise ValueError("goal_risk_scale must be positive.")
    predicted_state_mean = np.asarray(predicted_state_mean, dtype=float).reshape(4)
    goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
    distance_to_goal = float(np.linalg.norm(predicted_state_mean[:2] - goal_xy))
    return float(np.exp(-0.5 * (distance_to_goal / float(goal_risk_scale)) ** 2))


def compute_observation_plausibility_expert(
    *,
    predictive_observation_covariance: np.ndarray,
    attack_delta: np.ndarray,
) -> float:
    """Return the observation-space plausibility expert in `[0, 1]`."""
    predictive_observation_covariance = project_to_psd(
        np.asarray(predictive_observation_covariance, dtype=float)
    )
    attack_delta = np.asarray(attack_delta, dtype=float).reshape(-1)
    mahalanobis_squared = float(np.dot(attack_delta, solve_spd(predictive_observation_covariance, attack_delta)))
    return float(np.exp(-0.5 * mahalanobis_squared))


def normalize_direction(
    direction: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, float]:
    """Return the unit direction and its norm, handling near-zero vectors."""
    direction = np.asarray(direction, dtype=float).reshape(-1)
    norm_value = float(np.linalg.norm(direction))
    if norm_value < float(eps):
        return np.zeros_like(direction), float(norm_value)
    return direction / norm_value, float(norm_value)


def compute_defense_gamma(
    *,
    observation: np.ndarray,
    predictive_state_mean: np.ndarray,
    predictive_state_covariance: np.ndarray,
    predictive_observation_mean: np.ndarray,
    predictive_observation_covariance: np.ndarray,
    original_observation_covariance: np.ndarray,
    attack_target: np.ndarray,
    attack_direction: np.ndarray,
    lambda_covariance: float,
    gamma_threshold: float,
    goal_xy: np.ndarray,
    goal_risk_scale: float,
    omega_risk: float,
    omega_observation: float,
) -> tuple[float, float, float, float]:
    """
    Compute `gamma_t`, `gamma_bar_t`, and the two defense experts.

    The prior attack probability is the weighted combination of:
    1. the goal-proximity risk expert based on the predictive state,
    2. the observation plausibility expert based on the PGD perturbation.
    """
    if not np.isclose(float(omega_risk) + float(omega_observation), 1.0, atol=1e-9):
        raise ValueError("omega_risk and omega_observation must sum to one.")

    predictive_state_mean = np.asarray(predictive_state_mean, dtype=float).reshape(4)
    predictive_state_covariance = project_to_psd(np.asarray(predictive_state_covariance, dtype=float))
    predictive_observation_mean = np.asarray(predictive_observation_mean, dtype=float).reshape(4)
    predictive_observation_covariance = project_to_psd(np.asarray(predictive_observation_covariance, dtype=float))
    original_observation_covariance = project_to_psd(np.asarray(original_observation_covariance, dtype=float))
    observation = np.asarray(observation, dtype=float).reshape(4)
    attack_target = np.asarray(attack_target, dtype=float).reshape(4)
    attack_direction = np.asarray(attack_direction, dtype=float).reshape(4)

    attack_delta = attack_target - predictive_observation_mean
    risk_expert = compute_goal_proximity_risk(
        predicted_state_mean=predictive_state_mean,
        goal_xy=goal_xy,
        goal_risk_scale=float(goal_risk_scale),
    )
    observation_expert = compute_observation_plausibility_expert(
        predictive_observation_covariance=predictive_observation_covariance,
        attack_delta=attack_delta,
    )
    prior_probability = float(
        np.clip(
            float(omega_risk) * float(risk_expert) + float(omega_observation) * float(observation_expert),
            0.0,
            1.0,
        )
    )

    attack_predictive_covariance = rank_one_covariance_update(
        predictive_observation_covariance,
        float(lambda_covariance),
        attack_direction,
    )
    precision_poe = spd_inverse(attack_predictive_covariance) + spd_inverse(original_observation_covariance)
    covariance_poe = spd_inverse(precision_poe)
    rhs_poe = (
        solve_spd(attack_predictive_covariance, predictive_observation_mean)
        + solve_spd(original_observation_covariance, attack_target)
    )
    mean_poe = solve_spd(precision_poe, rhs_poe)

    log_clean = gaussian_logpdf(observation, predictive_observation_mean, predictive_observation_covariance)
    log_attack = gaussian_logpdf(observation, mean_poe, covariance_poe)
    gamma_t = log_mixture_posterior_weight(
        prior_probability=prior_probability,
        log_clean=log_clean,
        log_attack=log_attack,
    )
    _adapted_covariance, gamma_bar_t = compute_adapted_observation_covariance(
        observation_covariance=original_observation_covariance,
        lam=float(lambda_covariance),
        gamma=float(gamma_t),
        direction=attack_direction,
        gamma_threshold=float(gamma_threshold),
    )
    return float(gamma_t), float(gamma_bar_t), float(risk_expert), float(observation_expert)


def compute_mahalanobis_imq_posterior_gamma(
    *,
    observation: np.ndarray,
    predictive_state_mean: np.ndarray,
    predictive_observation_mean: np.ndarray,
    predictive_observation_covariance: np.ndarray,
    original_observation_covariance: np.ndarray,
    attack_direction: np.ndarray,
    lambda_covariance: float,
    gamma_threshold: float,
    coverage: float,
    goal_xy: np.ndarray,
    goal_risk_scale: float,
    omega_risk: float,
    omega_observation: float,
) -> tuple[float, float, float, float]:
    """
    Compute a WoLF-MD style `gamma_t` using an IMQ score on the innovation.

    The prior attack probability is kept in the same style as the benchmark
    covariance defense, but the posterior activation is replaced by a soft
    Mahalanobis anomaly score controlled by the ellipsoid epsilon.
    """
    if not np.isclose(float(omega_risk) + float(omega_observation), 1.0, atol=1e-9):
        raise ValueError("omega_risk and omega_observation must sum to one.")

    observation = np.asarray(observation, dtype=float).reshape(4)
    predictive_state_mean = np.asarray(predictive_state_mean, dtype=float).reshape(4)
    predictive_observation_mean = np.asarray(predictive_observation_mean, dtype=float).reshape(4)
    predictive_observation_covariance = project_to_psd(
        np.asarray(predictive_observation_covariance, dtype=float)
    )
    original_observation_covariance = project_to_psd(
        np.asarray(original_observation_covariance, dtype=float)
    )
    attack_direction = np.asarray(attack_direction, dtype=float).reshape(4)

    innovation = observation - predictive_observation_mean
    risk_expert = compute_goal_proximity_risk(
        predicted_state_mean=predictive_state_mean,
        goal_xy=goal_xy,
        goal_risk_scale=float(goal_risk_scale),
    )
    observation_expert = compute_observation_plausibility_expert(
        predictive_observation_covariance=predictive_observation_covariance,
        attack_delta=innovation,
    )
    prior_probability = float(
        np.clip(
            float(omega_risk) * float(risk_expert) + float(omega_observation) * float(observation_expert),
            0.0,
            1.0,
        )
    )

    epsilon = float(coverage_to_mahalanobis_epsilon(float(coverage), obs_dim=innovation.size))
    epsilon = max(epsilon, 1e-12)
    mahalanobis_squared = float(np.dot(innovation, solve_spd(predictive_observation_covariance, innovation)))
    anomaly_score = float(mahalanobis_squared / (epsilon + mahalanobis_squared))
    clean_score = float(1.0 - anomaly_score)
    posterior_numerator = float(prior_probability) * float(anomaly_score)
    posterior_denominator = posterior_numerator + float(1.0 - prior_probability) * float(clean_score)
    gamma_t = float(posterior_numerator / max(posterior_denominator, 1e-12))

    _adapted_covariance, gamma_bar_t = compute_adapted_observation_covariance(
        observation_covariance=original_observation_covariance,
        lam=float(lambda_covariance),
        gamma=float(gamma_t),
        direction=attack_direction,
        gamma_threshold=float(gamma_threshold),
    )
    return float(gamma_t), float(gamma_bar_t), float(risk_expert), float(observation_expert)


def simulate_wind_step(
    *,
    state: np.ndarray,
    action: np.ndarray,
    goal_xy: np.ndarray,
    config: WindNavigationConfig,
    delta_psi: float,
    process_noise: np.ndarray,
    step_index: int,
) -> dict[str, Any]:
    """Advance one wind-navigation state using one pre-sampled exogenous step."""
    state = np.asarray(state, dtype=np.float32).reshape(4)
    action = np.asarray(action, dtype=np.float32).reshape(2)
    goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(2)
    process_noise = np.asarray(process_noise, dtype=np.float32).reshape(4)

    bounded_action = np.clip(action, -float(config.action_limit), float(config.action_limit))
    wind = state[2:].copy()
    cos_psi = float(np.cos(float(delta_psi)))
    sin_psi = float(np.sin(float(delta_psi)))
    rotated_wind = np.zeros(2, dtype=np.float32)
    rotated_wind[0] = float(config.rho_w) * (cos_psi * wind[0] - sin_psi * wind[1])
    rotated_wind[1] = float(config.rho_w) * (sin_psi * wind[0] + cos_psi * wind[1])

    next_state = np.zeros(4, dtype=np.float32)
    next_state[:2] = state[:2] + wind + bounded_action + process_noise[:2]
    next_state[2:] = rotated_wind + process_noise[2:]

    goal_distance = float(np.linalg.norm(next_state[:2] - goal_xy))
    reached_goal = bool(goal_distance <= float(config.goal_radius))
    reached_horizon = bool(int(step_index) + 1 >= int(config.max_steps))
    timed_out = bool(reached_horizon and not reached_goal)
    done = bool(reached_goal or reached_horizon)

    normalized_distance = goal_distance / max(float(config.radius_max), 1e-6)
    reward = -normalized_distance
    if reached_goal:
        reward += float(config.goal_reward)
    if timed_out:
        reward += float(config.timeout_penalty)

    return {
        "next_state": next_state.astype(np.float32),
        "reward": float(reward),
        "done": bool(done),
        "reached_goal": bool(reached_goal),
        "timed_out": bool(timed_out),
        "goal_distance": float(goal_distance),
        "transition_matrix": build_transition_matrix(
            rho_w=float(config.rho_w),
            delta_psi=float(delta_psi),
        ).astype(np.float32),
    }


def generate_episode_randomness(
    *,
    config: WindNavigationConfig,
    seed: int,
) -> EpisodeRandomness:
    """Pre-sample one episode so all methods can replay the same exogenous data."""
    rng = np.random.default_rng(int(seed))
    start_xy = np.asarray(config.start_xy, dtype=np.float32).reshape(2)
    initial_hidden_state = np.zeros(4, dtype=np.float32)
    initial_hidden_state[:2] = start_xy
    wind_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    initial_hidden_state[2] = float(config.initial_wind_magnitude) * float(np.cos(wind_angle))
    initial_hidden_state[3] = float(config.initial_wind_magnitude) * float(np.sin(wind_angle))

    goal_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    goal_radius_sq = float(
        rng.uniform(
            float(config.goal_distance_min) ** 2,
            float(config.goal_distance_max) ** 2,
        )
    )
    goal_radius = float(np.sqrt(goal_radius_sq))
    goal_xy = np.array(
        [
            goal_radius * float(np.cos(goal_angle)),
            goal_radius * float(np.sin(goal_angle)),
        ],
        dtype=np.float32,
    )

    max_steps = int(config.max_steps)
    delta_psis = rng.normal(0.0, float(config.wind_turn_std), size=max_steps).astype(np.float32)
    process_covariance = build_process_covariance(config)
    process_noises = rng.multivariate_normal(
        mean=np.zeros(4, dtype=np.float32),
        cov=process_covariance,
        size=max_steps,
    ).astype(np.float32)
    observation_noises = rng.normal(
        0.0,
        float(config.observation_noise_std),
        size=(max_steps, 4),
    ).astype(np.float32)
    attack_uniforms = rng.uniform(0.0, 1.0, size=max_steps).astype(np.float32)
    random_contour_seeds = rng.integers(0, 2**31 - 1, size=max_steps, dtype=np.int64)

    return EpisodeRandomness(
        seed=int(seed),
        initial_hidden_state=initial_hidden_state,
        goal_xy=goal_xy,
        delta_psis=delta_psis,
        process_noises=process_noises,
        observation_noises=observation_noises,
        attack_uniforms=attack_uniforms,
        random_contour_seeds=random_contour_seeds,
    )


def build_default_benchmark_config() -> DefenseBenchmarkConfig:
    """
    Build the requested intermediate `1:1` case with `W = 2 * W_base`.

    The remaining environment parameters are inherited from the trained PPO
    checkpoint so existing RL experiments keep their default behavior.
    """
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    device = default_device()
    _policy, env_config, _net_config, train_config, _history = load_agent_checkpoint(model_path, device)

    base_process_position_std = 0.14
    base_process_wind_std = 0.10
    process_position_std = float(base_process_position_std * np.sqrt(2.0))
    process_wind_std = float(base_process_wind_std * np.sqrt(2.0))
    observation_noise_std = observation_std_from_ratio(
        covariance_ratio_value=1.0,
        process_position_std=float(process_position_std),
        process_wind_std=float(process_wind_std),
    )

    filter_config = replace(
        env_config,
        observation_noise_std=float(observation_noise_std),
        process_position_std=float(process_position_std),
        process_wind_std=float(process_wind_std),
    )
    simulation_config = replace(
        filter_config,
        observation_noise_std=0.0,
    )

    return DefenseBenchmarkConfig(
        coverage=0.95,
        attack_probability=0.10,
        # Random-contour attacks produce much smaller attacked-step gamma_t
        # values than PGD attacks, so we use a more permissive activation
        # threshold and a substantially larger lambda sweep by default.
        lambdas=(0.5, 2.0, 8.0),
        gamma_threshold=0.001,
        goal_risk_scale=float(filter_config.goal_distance_min),
        omega_risk=0.5,
        omega_observation=0.5,
        strong_wolf_weight_threshold=0.25,
        attack_step_size=0.05,
        attack_num_steps=120,
        attack_mc_samples=16,
        transition_mc_samples=8,
        attack_discount_gamma=float(train_config.gamma),
        contour_relative_tolerance=0.01,
        direction_eps=1e-10,
        base_seed=20260727,
        n_tuning_episodes=6,
        n_episodes=100,
        filter_config=filter_config,
        simulation_config=simulation_config,
    )


def build_benchmark_methods(
    *,
    lambdas: tuple[float, ...],
    wolf_params: dict[str, Any],
) -> list[BenchmarkMethod]:
    """Build the requested list of attack/defense variants."""
    methods = [
        BenchmarkMethod(
            name="Clean",
            attack_type="clean",
            filter_type="clean",
            use_covariance_adaptation=False,
            lambda_covariance=0.0,
            defense_direction_source="none",
        ),
        BenchmarkMethod(
            name="Noisy + KF",
            attack_type="none",
            filter_type="kf",
            use_covariance_adaptation=False,
            lambda_covariance=0.0,
            defense_direction_source="none",
        ),
        BenchmarkMethod(
            name="PGD estimated, no defense",
            attack_type="pgd_estimated",
            filter_type="kf",
            use_covariance_adaptation=False,
            lambda_covariance=0.0,
            defense_direction_source="none",
        ),
        BenchmarkMethod(
            name="Random contour, no defense",
            attack_type="random_contour",
            filter_type="kf",
            use_covariance_adaptation=False,
            lambda_covariance=0.0,
            defense_direction_source="none",
        ),
    ]

    for lambda_value in lambdas:
        methods.append(
            BenchmarkMethod(
                name=f"PGD estimated, lambda={lambda_value}",
                attack_type="pgd_estimated",
                filter_type="covariance_adaptation",
                use_covariance_adaptation=True,
                lambda_covariance=float(lambda_value),
                defense_direction_source="pgd_estimated",
            )
        )
        methods.append(
            BenchmarkMethod(
                name=f"Random contour + PGD direction, lambda={lambda_value}",
                attack_type="random_contour",
                filter_type="covariance_adaptation",
                use_covariance_adaptation=True,
                lambda_covariance=float(lambda_value),
                defense_direction_source="pgd_estimated",
            )
        )

    for wolf_key, label in (("wolf_imq", "WoLF-IMQ"), ("wolf_tmd", "WoLF-TMD")):
        selected = wolf_params[wolf_key]["selected_parameters"]
        wolf_kind = str(selected["kind"])
        methods.extend(
            [
                BenchmarkMethod(
                    name=f"Noisy + {label}",
                    attack_type="none",
                    filter_type=wolf_key,
                    use_covariance_adaptation=False,
                    lambda_covariance=0.0,
                    defense_direction_source="none",
                    wolf_kind=wolf_kind,
                    wolf_parameters={key: float(value) if isinstance(value, (int, float)) else value for key, value in selected.items()},
                ),
                BenchmarkMethod(
                    name=f"PGD estimated + {label}",
                    attack_type="pgd_estimated",
                    filter_type=wolf_key,
                    use_covariance_adaptation=False,
                    lambda_covariance=0.0,
                    defense_direction_source="none",
                    wolf_kind=wolf_kind,
                    wolf_parameters={key: float(value) if isinstance(value, (int, float)) else value for key, value in selected.items()},
                ),
                BenchmarkMethod(
                    name=f"Random contour + {label}",
                    attack_type="random_contour",
                    filter_type=wolf_key,
                    use_covariance_adaptation=False,
                    lambda_covariance=0.0,
                    defense_direction_source="none",
                    wolf_kind=wolf_kind,
                    wolf_parameters={key: float(value) if isinstance(value, (int, float)) else value for key, value in selected.items()},
                ),
            ]
        )
    return methods


def wolf_config_from_parameters(parameters: dict[str, Any]) -> WoLFConfig:
    """Convert a serialized parameter dictionary into a shared WoLF config."""
    return WoLFConfig(
        kind=str(parameters["kind"]),
        imq_soft_threshold=float(parameters.get("imq_soft_threshold", 1.0)),
        tmd_threshold=float(parameters.get("tmd_threshold", 3.0)),
        min_weight=float(parameters.get("min_weight", 1e-6)),
    )


def run_benchmark_episode(
    *,
    policy: torch.nn.Module,
    method: BenchmarkMethod,
    episode: EpisodeRandomness,
    config: DefenseBenchmarkConfig,
    device: torch.device,
) -> MethodEpisodeResult:
    """Run one attack/defense method on one pre-sampled episode realization."""
    observation_matrix = build_observation_matrix().astype(float)
    observation_control_matrix = np.zeros((4, 2), dtype=float)
    control_matrix = build_control_matrix().astype(float)
    observation_covariance = build_observation_covariance(config.filter_config).astype(float)
    process_covariance = build_process_covariance(config.filter_config).astype(float)
    zero_action = np.zeros(2, dtype=float)
    prior_mean, prior_covariance = build_filter_prior(config.filter_config)

    if method.filter_type == "clean":
        posterior_mean = np.asarray(episode.initial_hidden_state, dtype=float).copy()
        posterior_covariance = np.zeros((4, 4), dtype=float)
        predict_result = None
    else:
        posterior_mean = np.asarray(prior_mean, dtype=float).copy()
        posterior_covariance = np.asarray(prior_covariance, dtype=float).copy()
        predict_result = build_initial_predict_result(
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            observation_covariance=observation_covariance,
        )

    state = np.asarray(episode.initial_hidden_state, dtype=float).copy()
    goal_xy = np.asarray(episode.goal_xy, dtype=float).copy()
    total_reward = 0.0
    success = False
    final_goal_distance = float(np.linalg.norm(state[:2] - goal_xy))
    diagnostics: list[StepDiagnostics] = []

    for step_index in range(int(config.simulation_config.max_steps)):
        nominal_observation = (
            state + np.asarray(episode.observation_noises[step_index], dtype=float).reshape(4)
        )

        attacked = False
        observed_input = nominal_observation.copy()
        gamma_t = 0.0
        gamma_bar_t = 0.0
        effective_inflation = 0.0
        risk_expert = 0.0
        observation_expert = 0.0
        wolf_weight = 1.0
        wolf_weight_squared = 1.0
        wolf_heavily_discounted = False
        random_contour_perturbation_norm = 0.0
        pgd_perturbation_norm = 0.0
        used_zero_pgd_direction = False
        pgd_direction_norm = 0.0
        pgd_direction = np.zeros(4, dtype=float)
        pgd_attack_observation = nominal_observation.copy()
        estimated_state_nominal = np.asarray(state, dtype=float).copy()

        if method.filter_type != "clean":
            assert predict_result is not None
            geometry = build_attack_geometry_from_predict_result(
                predict_result=predict_result,
                coverage=float(config.coverage),
                observation_index=int(step_index),
            )
            estimated_state_nominal = geometry.posterior_mean_for_observation(nominal_observation)

            if method.attack_type in {"pgd_estimated", "random_contour"}:
                attacked = bool(float(episode.attack_uniforms[step_index]) < float(config.attack_probability))

            needs_pgd_reference = bool(
                attacked
                and (
                    method.attack_type == "pgd_estimated"
                    or method.defense_direction_source == "pgd_estimated"
                )
            )
            if needs_pgd_reference:
                pgd_attack_observation = estimated_return_pgd_attack_observation(
                    policy=policy,
                    geometry=geometry,
                    clean_observation=nominal_observation,
                    estimated_hidden_state=estimated_state_nominal,
                    goal_xy=goal_xy,
                    simulation_config=config.simulation_config,
                    current_step_index=int(step_index),
                    gamma=float(config.attack_discount_gamma),
                    attack_step_size=float(config.attack_step_size),
                    attack_num_steps=int(config.attack_num_steps),
                    attack_mc_samples=int(config.attack_mc_samples),
                    transition_mc_samples=int(config.transition_mc_samples),
                    seed=int(episode.seed) + 10_000 * (step_index + 1),
                    device=device,
                )
                pgd_delta = np.asarray(pgd_attack_observation, dtype=float) - np.asarray(
                    predict_result.observation_mean,
                    dtype=float,
                )
                pgd_direction, pgd_direction_norm = normalize_direction(
                    pgd_delta,
                    eps=float(config.direction_eps),
                )
                pgd_perturbation_norm = float(np.linalg.norm(pgd_attack_observation - nominal_observation))
                used_zero_pgd_direction = bool(pgd_direction_norm < float(config.direction_eps))

            if method.attack_type in {"pgd_estimated", "random_contour"}:
                if attacked and method.attack_type == "pgd_estimated":
                    observed_input = np.asarray(pgd_attack_observation, dtype=float)
                elif attacked and method.attack_type == "random_contour":
                    observed_input = np.asarray(
                        random_observation_on_contour(
                            geometry=geometry,
                            rng=np.random.default_rng(int(episode.random_contour_seeds[step_index])),
                        ),
                        dtype=float,
                    )
                    random_contour_perturbation_norm = float(np.linalg.norm(observed_input - nominal_observation))
                else:
                    observed_input = nominal_observation.copy()

            if method.filter_type == "kf":
                update_result = kalman_update_step(
                    pred_state_mean=np.asarray(predict_result.state_mean, dtype=float),
                    pred_state_covariance=np.asarray(predict_result.state_covariance, dtype=float),
                    observation=observed_input,
                    observation_matrix=observation_matrix,
                    observation_control_matrix=observation_control_matrix,
                    observation_covariance=observation_covariance,
                    action_prev=zero_action,
                )
                posterior_mean = np.asarray(update_result.state_mean, dtype=float)
                posterior_covariance = np.asarray(update_result.state_covariance, dtype=float)
            elif method.filter_type in {"covariance_adaptation", "covariance_adaptation_observation_imq"}:
                if attacked:
                    adaptation_direction = np.zeros(4, dtype=float)
                    adaptation_direction_norm = 0.0
                    if method.defense_direction_source == "pgd_estimated":
                        adaptation_direction = np.asarray(pgd_direction, dtype=float)
                        adaptation_direction_norm = float(pgd_direction_norm)
                    elif method.defense_direction_source == "observed_innovation":
                        adaptation_direction, adaptation_direction_norm = normalize_direction(
                            np.asarray(observed_input, dtype=float) - np.asarray(predict_result.observation_mean, dtype=float),
                            eps=float(config.direction_eps),
                        )
                    if adaptation_direction_norm >= float(config.direction_eps):
                        if method.filter_type == "covariance_adaptation":
                            gamma_t, gamma_bar_t, risk_expert, observation_expert = compute_defense_gamma(
                                observation=observed_input,
                                predictive_state_mean=np.asarray(predict_result.state_mean, dtype=float),
                                predictive_state_covariance=np.asarray(predict_result.state_covariance, dtype=float),
                                predictive_observation_mean=np.asarray(predict_result.observation_mean, dtype=float),
                                predictive_observation_covariance=np.asarray(
                                    predict_result.observation_covariance,
                                    dtype=float,
                                ),
                                original_observation_covariance=observation_covariance,
                                attack_target=np.asarray(pgd_attack_observation, dtype=float),
                                attack_direction=adaptation_direction,
                                lambda_covariance=float(method.lambda_covariance),
                                gamma_threshold=float(config.gamma_threshold),
                                goal_xy=goal_xy,
                                goal_risk_scale=float(config.goal_risk_scale),
                                omega_risk=float(config.omega_risk),
                                omega_observation=float(config.omega_observation),
                            )
                        else:
                            gamma_t, gamma_bar_t, risk_expert, observation_expert = compute_mahalanobis_imq_posterior_gamma(
                                observation=observed_input,
                                predictive_state_mean=np.asarray(predict_result.state_mean, dtype=float),
                                predictive_observation_mean=np.asarray(predict_result.observation_mean, dtype=float),
                                predictive_observation_covariance=np.asarray(
                                    predict_result.observation_covariance,
                                    dtype=float,
                                ),
                                original_observation_covariance=observation_covariance,
                                attack_direction=adaptation_direction,
                                lambda_covariance=float(method.lambda_covariance),
                                gamma_threshold=float(config.gamma_threshold),
                                coverage=float(config.coverage),
                                goal_xy=goal_xy,
                                goal_risk_scale=float(config.goal_risk_scale),
                                omega_risk=float(config.omega_risk),
                                omega_observation=float(config.omega_observation),
                            )
                    adapted_covariance, gamma_bar_t = compute_adapted_observation_covariance(
                        observation_covariance=observation_covariance,
                        lam=float(method.lambda_covariance),
                        gamma=float(gamma_t),
                        direction=adaptation_direction,
                        gamma_threshold=float(config.gamma_threshold),
                    )
                else:
                    adapted_covariance = np.asarray(observation_covariance, dtype=float).copy()
                    gamma_bar_t = 0.0
                update_result = kalman_update_step(
                    pred_state_mean=np.asarray(predict_result.state_mean, dtype=float),
                    pred_state_covariance=np.asarray(predict_result.state_covariance, dtype=float),
                    observation=observed_input,
                    observation_matrix=observation_matrix,
                    observation_control_matrix=observation_control_matrix,
                    observation_covariance=adapted_covariance,
                    action_prev=zero_action,
                )
                effective_inflation = float(method.lambda_covariance) * float(gamma_bar_t)
                posterior_mean = np.asarray(update_result.state_mean, dtype=float)
                posterior_covariance = np.asarray(update_result.state_covariance, dtype=float)
            else:
                assert method.wolf_parameters is not None
                if attacked:
                    wolf_result = run_wolf_measurement_update(
                        predict_result=predict_result,
                        observation=observed_input,
                        observation_matrix=observation_matrix,
                        observation_covariance=observation_covariance,
                        config=wolf_config_from_parameters(method.wolf_parameters),
                    )
                    posterior_mean = np.asarray(wolf_result.update.state_mean, dtype=float)
                    posterior_covariance = np.asarray(wolf_result.update.state_covariance, dtype=float)
                    wolf_weight = float(wolf_result.diagnostics.weight)
                    wolf_weight_squared = float(wolf_result.diagnostics.weight_squared)
                    wolf_heavily_discounted = bool(
                        wolf_weight_squared <= float(config.strong_wolf_weight_threshold)
                    )
                else:
                    update_result = kalman_update_step(
                        pred_state_mean=np.asarray(predict_result.state_mean, dtype=float),
                        pred_state_covariance=np.asarray(predict_result.state_covariance, dtype=float),
                        observation=observed_input,
                        observation_matrix=observation_matrix,
                        observation_control_matrix=observation_control_matrix,
                        observation_covariance=observation_covariance,
                        action_prev=zero_action,
                    )
                    posterior_mean = np.asarray(update_result.state_mean, dtype=float)
                    posterior_covariance = np.asarray(update_result.state_covariance, dtype=float)
        else:
            posterior_mean = np.asarray(state, dtype=float).copy()
            posterior_covariance = np.zeros((4, 4), dtype=float)

        state_estimation_error = float(np.linalg.norm(np.asarray(posterior_mean, dtype=float) - state))
        goal_distance = float(np.linalg.norm(state[:2] - goal_xy))
        action = deterministic_policy_action(
            policy=policy,
            state_estimate=np.asarray(posterior_mean, dtype=float),
            goal_xy=goal_xy,
            device=device,
        )

        step_result = simulate_wind_step(
            state=state,
            action=action,
            goal_xy=goal_xy,
            config=config.simulation_config,
            delta_psi=float(episode.delta_psis[step_index]),
            process_noise=np.asarray(episode.process_noises[step_index], dtype=float),
            step_index=int(step_index),
        )
        total_reward += float(step_result["reward"])
        success = bool(step_result["reached_goal"])
        final_goal_distance = float(step_result["goal_distance"])
        diagnostics.append(
            StepDiagnostics(
                step_index=int(step_index),
                attacked=bool(attacked),
                attack_type=method.attack_type,
                filter_type=method.filter_type,
                gamma_t=float(gamma_t),
                gamma_bar_t=float(gamma_bar_t),
                effective_inflation=float(effective_inflation),
                risk_expert=float(risk_expert),
                observation_expert=float(observation_expert),
                state_estimation_error=float(state_estimation_error),
                goal_distance=float(goal_distance),
                pgd_perturbation_norm=float(pgd_perturbation_norm),
                random_contour_perturbation_norm=float(random_contour_perturbation_norm),
                pgd_direction_norm=float(pgd_direction_norm),
                used_zero_pgd_direction=bool(used_zero_pgd_direction),
                wolf_weight=float(wolf_weight),
                wolf_weight_squared=float(wolf_weight_squared),
                wolf_heavily_discounted=bool(wolf_heavily_discounted),
            )
        )

        if bool(step_result["done"]):
            break

        state = np.asarray(step_result["next_state"], dtype=float)
        if method.filter_type != "clean":
            predict_result = kalman_predict_step(
                prev_state_mean=np.asarray(posterior_mean, dtype=float),
                prev_state_covariance=np.asarray(posterior_covariance, dtype=float),
                transition_matrix=np.asarray(step_result["transition_matrix"], dtype=float),
                control_matrix=control_matrix,
                process_covariance=process_covariance,
                observation_matrix=observation_matrix,
                observation_control_matrix=observation_control_matrix,
                observation_covariance=observation_covariance,
                action_prev=np.asarray(action, dtype=float),
            )

    mean_state_estimation_error = float(
        np.mean([diag.state_estimation_error for diag in diagnostics], dtype=float)
    )
    return MethodEpisodeResult(
        method=method,
        episode_seed=int(episode.seed),
        episode_return=float(total_reward),
        success=bool(success),
        final_goal_distance=float(final_goal_distance),
        mean_state_estimation_error=float(mean_state_estimation_error),
        steps=int(len(diagnostics)),
        diagnostics=diagnostics,
    )


def diagnostics_to_dataframe(results: list[MethodEpisodeResult]) -> pd.DataFrame:
    """Flatten all per-step diagnostics into one long-form dataframe."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for diag in result.diagnostics:
            rows.append(
                {
                    "method": result.method.name,
                    "episode_seed": int(result.episode_seed),
                    "step_index": int(diag.step_index),
                    "attacked": bool(diag.attacked),
                    "attack_type": diag.attack_type,
                    "filter_type": diag.filter_type,
                    "gamma_t": float(diag.gamma_t),
                    "gamma_bar_t": float(diag.gamma_bar_t),
                    "effective_inflation": float(diag.effective_inflation),
                    "risk_expert": float(diag.risk_expert),
                    "observation_expert": float(diag.observation_expert),
                    "state_estimation_error": float(diag.state_estimation_error),
                    "goal_distance": float(diag.goal_distance),
                    "pgd_perturbation_norm": float(diag.pgd_perturbation_norm),
                    "random_contour_perturbation_norm": float(diag.random_contour_perturbation_norm),
                    "pgd_direction_norm": float(diag.pgd_direction_norm),
                    "used_zero_pgd_direction": bool(diag.used_zero_pgd_direction),
                    "wolf_weight": float(diag.wolf_weight),
                    "wolf_weight_squared": float(diag.wolf_weight_squared),
                    "wolf_heavily_discounted": bool(diag.wolf_heavily_discounted),
                }
            )
    return pd.DataFrame(rows)


def aggregate_benchmark_results(results: list[MethodEpisodeResult]) -> pd.DataFrame:
    """Aggregate episode-level metrics into the requested summary table."""
    diagnostics_df = diagnostics_to_dataframe(results)
    episode_rows: list[dict[str, Any]] = []
    for result in results:
        episode_rows.append(
            {
                "method": result.method.name,
                "attack_type": result.method.attack_type,
                "filter_or_defense": result.method.filter_type,
                "lambda": float(result.method.lambda_covariance),
                "hyperparameters": json.dumps(result.method.wolf_parameters or {}, sort_keys=True),
                "episode_seed": int(result.episode_seed),
                "episode_return": float(result.episode_return),
                "success": float(result.success),
                "final_goal_distance": float(result.final_goal_distance),
                "mean_state_estimation_error": float(result.mean_state_estimation_error),
            }
        )
    episode_df = pd.DataFrame(episode_rows)
    summary_rows: list[dict[str, Any]] = []

    for method_name, group_df in episode_df.groupby("method", sort=False):
        method_diag_df = diagnostics_df.loc[diagnostics_df["method"] == method_name]
        lambda_value = float(group_df["lambda"].iloc[0])
        summary_rows.append(
            {
                "method": method_name,
                "filter_or_defense": str(group_df["filter_or_defense"].iloc[0]),
                "hyperparameters": str(group_df["hyperparameters"].iloc[0]),
                "attack_type": str(group_df["attack_type"].iloc[0]),
                "lambda": float(lambda_value),
                "mean_return": float(group_df["episode_return"].mean()),
                "std_return": float(group_df["episode_return"].std(ddof=1)) if len(group_df) > 1 else 0.0,
                "mean_state_estimation_error": float(group_df["mean_state_estimation_error"].mean()),
                "mean_final_goal_distance": float(group_df["final_goal_distance"].mean()),
                "success_rate": float(group_df["success"].mean()),
                "mean_gamma_t": float(method_diag_df["gamma_t"].mean()) if not method_diag_df.empty else 0.0,
                "gamma_ge_threshold_rate": float(
                    (method_diag_df["gamma_t"] >= 0.3).mean()
                ) if not method_diag_df.empty else 0.0,
                "mean_effective_inflation": float(method_diag_df["effective_inflation"].mean()) if not method_diag_df.empty else 0.0,
                "mean_wolf_weight": float(
                    method_diag_df.loc[:, "wolf_weight"].replace(1.0, np.nan).mean()
                ) if "wolf" in str(group_df["filter_or_defense"].iloc[0]) else np.nan,
                "wolf_strong_discount_rate": float(method_diag_df["wolf_heavily_discounted"].mean()) if "wolf" in str(group_df["filter_or_defense"].iloc[0]) else np.nan,
                "zero_pgd_direction_rate": float(method_diag_df["used_zero_pgd_direction"].mean()) if not method_diag_df.empty else 0.0,
                "num_runs": int(len(group_df)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def plot_benchmark_results(
    *,
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save the grouped return comparison for the PGD and random-contour families."""
    set_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6), sharey=True)
    axes = np.asarray(axes)

    def panel_methods(panel_kind: str) -> list[str]:
        """Return the display order for one attack family panel."""
        if panel_kind == "pgd":
            lambda_rows = summary_df.loc[
                summary_df["method"].str.startswith("PGD estimated, lambda="),
                ["method", "lambda"],
            ].sort_values("lambda")
            names = [
                "Clean",
                "PGD estimated, no defense",
                *lambda_rows["method"].tolist(),
            ]
            wolf_names = [
                "PGD estimated + WoLF-IMQ",
                "PGD estimated + WoLF-TMD",
            ]
        else:
            lambda_rows = summary_df.loc[
                summary_df["method"].str.startswith("Random contour + PGD direction, lambda="),
                ["method", "lambda"],
            ].sort_values("lambda")
            names = [
                "Clean",
                "Random contour, no defense",
                *lambda_rows["method"].tolist(),
            ]
            wolf_names = [
                "Random contour + WoLF-IMQ",
                "Random contour + WoLF-TMD",
            ]
        for wolf_name in wolf_names:
            if wolf_name in summary_df["method"].values:
                names.append(wolf_name)
        return names

    method_colors = {
        "Clean": "#8EC5B5",
        "Noisy + KF": "#A8D3C2",
        "PGD estimated, no defense": "#E6A57E",
        "Random contour, no defense": "#AFC7E8",
        "PGD estimated + WoLF-IMQ": "#D8C3EA",
        "PGD estimated + WoLF-TMD": "#BDA6DB",
        "Random contour + WoLF-IMQ": "#D8C3EA",
        "Random contour + WoLF-TMD": "#BDA6DB",
    }
    pgd_lambda_palette = ["#F6D6B8", "#F0C987", "#D9B977"]
    contour_lambda_palette = ["#D9E5F7", "#BFD3F1", "#96B6E3"]

    def simplified_label(method_name: str) -> str:
        """Return a short x-axis label for one method."""
        if method_name == "Clean":
            return "Clean"
        if method_name == "PGD estimated, no defense":
            return "+ KF (no defense)"
        if method_name == "Random contour, no defense":
            return "+ KF (no defense)"
        if method_name.startswith("PGD estimated, lambda="):
            return method_name.replace("PGD estimated, ", "DirCovAdapt, ")
        if method_name.startswith("Random contour + PGD direction, lambda="):
            return method_name.replace("Random contour + PGD direction, ", "DirCovAdapt, ")
        if "WoLF-IMQ" in method_name:
            return "WoLF-IMQ"
        if "WoLF-TMD" in method_name:
            return "WoLF-TMD"
        return method_name

    panel_specs = [
        ("pgd", "PGD estimated-return attack family"),
        ("random", "Random contour attack family"),
    ]
    plotted_mean_values: list[float] = []
    for ax, (panel_kind, panel_note) in zip(axes, panel_specs, strict=False):
        method_names = panel_methods(panel_kind)
        panel_df = summary_df.set_index("method").loc[method_names].reset_index()
        panel_mean_values = panel_df["mean_return"].to_numpy(dtype=float)
        plotted_mean_values.extend(panel_mean_values.tolist())
        colors: list[str] = []
        lambda_idx = 0
        for method_name in panel_df["method"].tolist():
            if method_name in method_colors:
                colors.append(method_colors[method_name])
            elif panel_kind == "pgd":
                colors.append(pgd_lambda_palette[lambda_idx % len(pgd_lambda_palette)])
                lambda_idx += 1
            else:
                colors.append(contour_lambda_palette[lambda_idx % len(contour_lambda_palette)])
                lambda_idx += 1

        x_positions = np.arange(len(panel_df), dtype=float)
        bars = ax.bar(
            x_positions,
            panel_mean_values,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [simplified_label(name) for name in panel_df["method"].tolist()],
            rotation=30,
            ha="right",
        )
        for bar, value in zip(bars, panel_mean_values, strict=False):
            ax.text(
                float(bar.get_x() + 0.5 * bar.get_width()),
                float(value) + 0.015 * max(abs(float(np.min(panel_mean_values))), abs(float(np.max(panel_mean_values))), 1.0),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#31424F",
            )
        ax.text(
            0.02,
            0.96,
            panel_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            color="#4E5A65",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "#D8E0E6",
            },
        )
        ax.set_xlabel("Method")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.24)
        ax.grid(False, axis="x")

    axes[0].set_ylabel("Mean accumulated reward")
    if plotted_mean_values:
        ymin = float(np.floor(np.min(plotted_mean_values)))
        ymax = float(np.ceil(np.max(plotted_mean_values)))
        if np.isclose(ymin, ymax):
            ymax = ymin + 1.0
        for ax in axes:
            ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_dynamics(
    *,
    diagnostics_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save the attacked-step diagnostic figure requested for the RL benchmark."""
    set_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    attacked_df = diagnostics_df.loc[diagnostics_df["attacked"] == True].copy()
    pgd_method_name = "PGD estimated, lambda=0.5"
    contour_method_name = "Random contour + PGD direction, lambda=0.5"
    pgd_display_label = r"Attack on $Q^\pi(m_T,a_T)$ + DirCovAdapt"
    contour_display_label = r"$\epsilon$ perturbation + DirCovAdapt"
    color_by_label = {
        pgd_display_label: "#E6A57E",
        contour_display_label: "#AFC7E8",
    }

    def epsilon_box_label_from_output_path(path: str) -> str | None:
        """Return the coverage percentage encoded indirectly by the epsilon tag."""
        match = re.search(r"eps([0-9]+p[0-9]+)", os.path.basename(path))
        if match is None:
            return None
        mahalanobis_radius = float(match.group(1).replace("p", "."))
        coverage_percent = 100.0 * float(chi2.cdf(mahalanobis_radius, df=4))
        return rf"$\epsilon = {coverage_percent:.0f}\%$"

    def metric_values(
        *,
        method_name: str,
        metric: str,
        positive_only: bool,
    ) -> np.ndarray:
        """Return the finite attacked-step values for one method and metric."""
        values = attacked_df.loc[attacked_df["method"] == method_name, metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if positive_only:
            values = values[values > 0.0]
        return values

    def draw_smoothed_density(
        *,
        ax: plt.Axes,
        first_values: np.ndarray,
        first_label: str,
        second_values: np.ndarray,
        second_label: str,
        xlabel: str,
        support_lower: float,
        support_upper: float | None,
    ) -> None:
        """Overlay two attacked-step smoothed empirical distributions."""
        if first_values.size == 0 and second_values.size == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#4E5A65",
            )
            ax.set_axis_off()
            return

        def silverman_bandwidth(values: np.ndarray) -> float:
            """Return a stable Gaussian-kernel bandwidth."""
            if values.size <= 1:
                return 0.1
            value_std = float(np.std(values, ddof=1))
            value_iqr = float(np.subtract(*np.percentile(values, [75.0, 25.0])))
            robust_scale = min(value_std, value_iqr / 1.34) if value_iqr > 0.0 else value_std
            if robust_scale <= 1e-12:
                robust_scale = max(abs(float(np.mean(values))), 1.0) * 0.05
            return max(0.9 * robust_scale * values.size ** (-0.2), 1e-3)

        def density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
            """Evaluate a boundary-corrected Gaussian-kernel density estimate."""
            if values.size == 0:
                return np.zeros_like(grid)
            bandwidth = silverman_bandwidth(values)
            kernel_sum = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2)
            kernel_sum += np.exp(-0.5 * ((grid[:, None] + values[None, :] - 2.0 * support_lower) / bandwidth) ** 2)
            if support_upper is not None:
                kernel_sum += np.exp(
                    -0.5 * ((grid[:, None] + values[None, :] - 2.0 * float(support_upper)) / bandwidth) ** 2
                )
            kernel_sum /= np.sqrt(2.0 * np.pi)
            return np.mean(kernel_sum, axis=1) / bandwidth

        combined = np.concatenate([values for values in (first_values, second_values) if values.size > 0])
        combined_max = float(np.max(combined))
        combined_std = float(np.std(combined, ddof=1)) if combined.size > 1 else 0.0
        if support_upper is not None:
            grid = np.linspace(float(support_lower), float(support_upper), 400)
        else:
            if np.isclose(float(support_lower), combined_max):
                span = 1e-3 if np.isclose(float(support_lower), 0.0) else 0.08 * abs(float(support_lower))
            else:
                span = max(0.12 * (combined_max - float(support_lower)), 0.35 * combined_std, 1e-3)
            grid = np.linspace(float(support_lower), combined_max + span, 400)

        for values, label in (
            (first_values, first_label),
            (second_values, second_label),
        ):
            if values.size == 0:
                continue
            density_values = density_curve(values, grid)
            ax.fill_between(
                grid,
                density_values,
                color=color_by_label[label],
                alpha=0.24,
                linewidth=0.0,
            )
            ax.plot(
                grid,
                density_values,
                color=color_by_label[label],
                linewidth=2.1,
                label=label,
            )

        ax.set_xlabel(xlabel, fontsize=12.8)
        ax.set_ylabel("Distribution")
        ax.tick_params(axis="x", labelsize=11.8)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=12.0)
        ax.set_ylim(bottom=0.0)
        ax.axhline(0.0, color="#5F6B76", linewidth=1.0, linestyle="--", alpha=0.9, zorder=0)
        ax.axvline(float(support_lower), color="#5F6B76", linewidth=0.9, linestyle="--", alpha=0.78, zorder=0)
        if support_upper is not None:
            ax.axvline(float(support_upper), color="#5F6B76", linewidth=0.9, linestyle="--", alpha=0.78, zorder=0)

    pgd_gamma_values = metric_values(
        method_name=pgd_method_name,
        metric="gamma_t",
        positive_only=False,
    )
    contour_gamma_values = metric_values(
        method_name=contour_method_name,
        metric="gamma_t",
        positive_only=False,
    )
    pgd_perturbation_values = metric_values(
        method_name=pgd_method_name,
        metric="pgd_perturbation_norm",
        positive_only=True,
    )
    contour_perturbation_values = metric_values(
        method_name=contour_method_name,
        metric="random_contour_perturbation_norm",
        positive_only=True,
    )

    draw_smoothed_density(
        ax=axes[0],
        first_values=pgd_gamma_values,
        first_label=pgd_display_label,
        second_values=contour_gamma_values,
        second_label=contour_display_label,
        xlabel=r"$\gamma_t$",
        support_lower=0.0,
        support_upper=1.0,
    )
    draw_smoothed_density(
        ax=axes[1],
        first_values=pgd_perturbation_values,
        first_label=pgd_display_label,
        second_values=contour_perturbation_values,
        second_label=contour_display_label,
        xlabel=r"Perturbation norm $\|o_t^{\mathrm{adv}} - \hat{o}_t\|_2$",
        support_lower=0.0,
        support_upper=None,
    )

    epsilon_label = epsilon_box_label_from_output_path(output_path)
    if epsilon_label is not None:
        axes[0].text(
            0.03,
            0.96,
            epsilon_label,
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=12.0,
            color="#31424F",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "alpha": 0.94,
                "edgecolor": "#BFC8D0",
            },
        )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.24, axis="y")
        ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def selection_score_from_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the WoLF tuning ranking requested in the sweep document."""
    attacked_df = summary_df.loc[summary_df["scenario"].isin(["pgd_estimated", "random_contour"])]
    grouped = attacked_df.groupby("config_label", sort=False).agg(
        attacked_mean_return=("mean_return", "mean"),
        attacked_mean_state_error=("mean_state_estimation_error", "mean"),
        attacked_std_return=("std_return", "mean"),
    )
    noisy_df = summary_df.loc[summary_df["scenario"] == "no_attack"].set_index("config_label")
    grouped["noisy_mean_return"] = noisy_df["mean_return"]
    grouped["noisy_mean_state_error"] = noisy_df["mean_state_estimation_error"]
    grouped = grouped.reset_index()
    grouped = grouped.sort_values(
        by=[
            "attacked_mean_return",
            "attacked_mean_state_error",
            "noisy_mean_return",
            "attacked_std_return",
        ],
        ascending=[False, True, False, True],
    )
    return grouped


def build_wolf_sweep_configurations() -> list[dict[str, Any]]:
    """Return a reasonable WoLF hyperparameter sweep grid."""
    configs: list[dict[str, Any]] = []
    for value in (0.35, 0.50, 0.65, 0.80, 1.00):
        configs.append(
            {
                "config_label": f"wolf_imq_tau_{str(value).replace('.', 'p')}",
                "kind": "imq",
                "imq_soft_threshold": float(value),
                "tmd_threshold": 3.0,
                "min_weight": 1e-6,
            }
        )
    for value in (1.5, 2.0, 2.5, 3.0, 3.5):
        configs.append(
            {
                "config_label": f"wolf_tmd_tau_{str(value).replace('.', 'p')}",
                "kind": "tmd",
                "imq_soft_threshold": 1.0,
                "tmd_threshold": float(value),
                "min_weight": 1e-6,
            }
        )
    return configs


def run_wolf_sweep(
    *,
    policy: torch.nn.Module,
    config: DefenseBenchmarkConfig,
    tuning_episodes: list[EpisodeRandomness],
    device: torch.device,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the WoLF sweep over noisy, PGD, and random-contour scenarios."""
    rows: list[dict[str, Any]] = []
    configurations = build_wolf_sweep_configurations()
    total_configurations = len(configurations)
    for config_index, parameters in enumerate(configurations, start=1):
        for scenario_index, scenario in enumerate(("no_attack", "pgd_estimated", "random_contour"), start=1):
            method = BenchmarkMethod(
                name=f"{parameters['config_label']}::{scenario}",
                attack_type="none" if scenario == "no_attack" else scenario,
                filter_type=f"wolf_{parameters['kind']}",
                use_covariance_adaptation=False,
                lambda_covariance=0.0,
                defense_direction_source="none",
                wolf_kind=str(parameters["kind"]),
                wolf_parameters=dict(parameters),
            )
            if progress_callback is not None:
                progress_callback(
                    "[wolf] "
                    f"configuration {config_index}/{total_configurations} | "
                    f"scenario {scenario_index}/3 | "
                    f"{parameters['config_label']} | {scenario}"
                )
            episode_results: list[MethodEpisodeResult] = []
            total_episodes = len(tuning_episodes)
            for episode_index, episode in enumerate(tuning_episodes, start=1):
                episode_results.append(
                    run_benchmark_episode(
                        policy=policy,
                        method=method,
                        episode=episode,
                        config=config,
                        device=device,
                    )
                )
                if progress_callback is not None and (
                    episode_index == 1
                    or episode_index == total_episodes
                    or episode_index % 10 == 0
                ):
                    progress_callback(
                        "[wolf] "
                        f"configuration {config_index}/{total_configurations} | "
                        f"{parameters['config_label']} | {scenario} | "
                        f"episode {episode_index}/{total_episodes}"
                    )
            diagnostics_df = diagnostics_to_dataframe(episode_results)
            episode_returns = np.asarray([result.episode_return for result in episode_results], dtype=float)
            rows.append(
                {
                    "config_label": str(parameters["config_label"]),
                    "kind": str(parameters["kind"]),
                    "scenario": scenario,
                    "imq_soft_threshold": float(parameters["imq_soft_threshold"]),
                    "tmd_threshold": float(parameters["tmd_threshold"]),
                    "mean_return": float(np.mean(episode_returns)),
                    "std_return": float(np.std(episode_returns, ddof=1)) if len(episode_returns) > 1 else 0.0,
                    "mean_state_estimation_error": float(
                        np.mean([result.mean_state_estimation_error for result in episode_results], dtype=float)
                    ),
                    "mean_final_goal_distance": float(
                        np.mean([result.final_goal_distance for result in episode_results], dtype=float)
                    ),
                    "success_rate": float(np.mean([result.success for result in episode_results], dtype=float)),
                    "mean_wolf_weight": float(diagnostics_df["wolf_weight"].mean()),
                    "wolf_strong_discount_rate": float(diagnostics_df["wolf_heavily_discounted"].mean()),
                    "num_runs": int(len(episode_results)),
                }
            )

    full_df = pd.DataFrame(rows)
    ranking_df = selection_score_from_summary(full_df)
    return full_df, ranking_df


def select_best_wolf_parameters(
    *,
    full_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Select the best IMQ and TMD configurations using the preset ranking."""
    selected: dict[str, dict[str, Any]] = {}
    for kind_key in ("imq", "tmd"):
        kind_full_df = full_df.loc[full_df["kind"] == kind_key]
        kind_ranking = selection_score_from_summary(kind_full_df)
        if kind_ranking.empty:
            raise RuntimeError(f"No WoLF sweep rows were generated for kind={kind_key}.")
        best_label = str(kind_ranking.iloc[0]["config_label"])
        best_row = kind_full_df.loc[kind_full_df["config_label"] == best_label].iloc[0]
        selected[f"wolf_{kind_key}"] = {
            "selected_parameters": {
                "kind": kind_key,
                "imq_soft_threshold": float(best_row["imq_soft_threshold"]),
                "tmd_threshold": float(best_row["tmd_threshold"]),
                "min_weight": 1e-6,
            }
        }
    return selected


def save_wolf_selection_artifacts(
    *,
    config: DefenseBenchmarkConfig,
    full_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    selected_parameters: dict[str, dict[str, Any]],
    output_prefix: str,
) -> dict[str, str]:
    """Save the standalone WoLF sweep tables, figure, and JSON selection file."""
    data_dir = rl_data_dir()
    figures_dir = rl_figures_dir()
    full_csv_path = os.path.join(data_dir, f"{output_prefix}_full.csv")
    ranking_csv_path = os.path.join(data_dir, f"{output_prefix}_ranking.csv")
    figure_path = os.path.join(figures_dir, f"{output_prefix}.png")
    json_path = os.path.join(data_dir, "best_wolf_params.json")

    full_df.to_csv(full_csv_path, index=False)
    ranking_df.to_csv(ranking_csv_path, index=False)

    set_plot_theme()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    pivot_df = full_df.pivot(index="config_label", columns="scenario", values="mean_return").fillna(np.nan)
    pivot_df = pivot_df.sort_index()
    x_positions = np.arange(len(pivot_df), dtype=float)
    width = 0.24
    scenario_order = ["no_attack", "pgd_estimated", "random_contour"]
    scenario_colors = {
        "no_attack": "#9DB7D5",
        "pgd_estimated": "#E6A57E",
        "random_contour": "#AFC7E8",
    }
    for idx, scenario in enumerate(scenario_order):
        values = pivot_df[scenario].to_numpy(dtype=float)
        ax.bar(
            x_positions + (idx - 1) * width,
            values,
            width=width,
            color=scenario_colors[scenario],
            edgecolor="white",
            linewidth=0.8,
            label=scenario.replace("_", " "),
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot_df.index.tolist(), rotation=35, ha="right")
    ax.set_ylabel("Mean accumulated reward")
    ax.set_xlabel("WoLF configuration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "wolf_imq": selected_parameters["wolf_imq"],
        "wolf_tmd": selected_parameters["wolf_tmd"],
        "selection_metric": "maximize attacked mean return; tie-break by lower state error, better noisy return, lower return variability",
        "tuning_seeds": [
            int(config.base_seed) + idx
            for idx in range(int(config.n_tuning_episodes))
        ],
        "evaluation_seeds": [
            int(config.base_seed) + int(config.n_tuning_episodes) + idx
            for idx in range(int(config.n_episodes))
        ],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "full_csv_path": full_csv_path,
        "ranking_csv_path": ranking_csv_path,
        "figure_path": figure_path,
        "json_path": json_path,
    }


def load_best_wolf_params() -> dict[str, Any]:
    """Load the persisted WoLF parameters selected by `wolf.py`."""
    json_path = os.path.join(rl_data_dir(), "best_wolf_params.json")
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def best_wolf_params_path() -> str:
    """Return the canonical path of the persisted WoLF parameter selection."""
    return os.path.join(rl_data_dir(), "best_wolf_params.json")


def build_episode_sets(config: DefenseBenchmarkConfig) -> tuple[list[EpisodeRandomness], list[EpisodeRandomness]]:
    """Create the tuning and evaluation episode lists from the configured counts."""
    tuning_episodes = [
        generate_episode_randomness(
            config=config.simulation_config,
            seed=int(config.base_seed) + idx,
        )
        for idx in range(int(config.n_tuning_episodes))
    ]
    evaluation_episodes = [
        generate_episode_randomness(
            config=config.simulation_config,
            seed=int(config.base_seed) + int(config.n_tuning_episodes) + idx,
        )
        for idx in range(int(config.n_episodes))
    ]
    return tuning_episodes, evaluation_episodes


def run_full_benchmark(
    *,
    config: DefenseBenchmarkConfig,
    output_prefix: str,
) -> dict[str, str]:
    """Run the full benchmark with the selected WoLF parameters and save outputs."""
    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, _env_config, _net_config, _train_config, _history = load_agent_checkpoint(model_path, device)
    policy.eval()

    _tuning_episodes, evaluation_episodes = build_episode_sets(config)
    wolf_params = load_best_wolf_params()
    methods = build_benchmark_methods(
        lambdas=tuple(float(value) for value in config.lambdas),
        wolf_params=wolf_params,
    )

    all_results: list[MethodEpisodeResult] = []
    for method in methods:
        for episode in evaluation_episodes:
            all_results.append(
                run_benchmark_episode(
                    policy=policy,
                    method=method,
                    episode=episode,
                    config=config,
                    device=device,
                )
            )

    summary_df = aggregate_benchmark_results(all_results)
    diagnostics_df = diagnostics_to_dataframe(all_results)
    episode_df = pd.DataFrame(
        [
            {
                "method": result.method.name,
                "episode_seed": int(result.episode_seed),
                "episode_return": float(result.episode_return),
                "success": float(result.success),
                "final_goal_distance": float(result.final_goal_distance),
                "mean_state_estimation_error": float(result.mean_state_estimation_error),
                "steps": int(result.steps),
            }
            for result in all_results
        ]
    )

    data_dir = rl_data_dir()
    figures_dir = rl_figures_dir()
    summary_csv_path = os.path.join(data_dir, f"{output_prefix}_summary.csv")
    episodes_csv_path = os.path.join(data_dir, f"{output_prefix}_episodes.csv")
    diagnostics_csv_path = os.path.join(data_dir, f"{output_prefix}_diagnostics.csv")
    results_npz_path = os.path.join(data_dir, f"{output_prefix}.npz")
    returns_figure_path = os.path.join(figures_dir, f"{output_prefix}_returns.png")
    diagnostics_figure_path = os.path.join(figures_dir, f"{output_prefix}_dynamics.png")

    summary_df.to_csv(summary_csv_path, index=False)
    episode_df.to_csv(episodes_csv_path, index=False)
    diagnostics_df.to_csv(diagnostics_csv_path, index=False)

    np.savez_compressed(
        results_npz_path,
        summary_columns=np.asarray(summary_df.columns.tolist(), dtype=object),
        summary_values=summary_df.to_numpy(dtype=object),
        episode_columns=np.asarray(episode_df.columns.tolist(), dtype=object),
        episode_values=episode_df.to_numpy(dtype=object),
        diagnostics_columns=np.asarray(diagnostics_df.columns.tolist(), dtype=object),
        diagnostics_values=diagnostics_df.to_numpy(dtype=object),
        lambdas=np.asarray(config.lambdas, dtype=float),
        gamma_threshold=np.asarray([config.gamma_threshold], dtype=float),
        goal_risk_scale=np.asarray([config.goal_risk_scale], dtype=float),
        attack_probability=np.asarray([config.attack_probability], dtype=float),
        coverage=np.asarray([config.coverage], dtype=float),
        n_tuning_episodes=np.asarray([config.n_tuning_episodes], dtype=int),
        n_episodes=np.asarray([config.n_episodes], dtype=int),
        process_position_std=np.asarray([config.filter_config.process_position_std], dtype=float),
        process_wind_std=np.asarray([config.filter_config.process_wind_std], dtype=float),
        observation_noise_std=np.asarray([config.filter_config.observation_noise_std], dtype=float),
    )

    plot_benchmark_results(summary_df=summary_df, output_path=returns_figure_path)
    plot_benchmark_dynamics(diagnostics_df=diagnostics_df, output_path=diagnostics_figure_path)
    return {
        "summary_csv_path": summary_csv_path,
        "episodes_csv_path": episodes_csv_path,
        "diagnostics_csv_path": diagnostics_csv_path,
        "results_npz_path": results_npz_path,
        "returns_figure_path": returns_figure_path,
        "diagnostics_figure_path": diagnostics_figure_path,
    }


def load_saved_benchmark_results(
    *,
    output_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved benchmark summary and diagnostics without rerunning experiments."""
    data_dir = rl_data_dir()
    summary_csv_path = os.path.join(data_dir, f"{output_prefix}_summary.csv")
    diagnostics_csv_path = os.path.join(data_dir, f"{output_prefix}_diagnostics.csv")
    if not os.path.exists(summary_csv_path):
        raise FileNotFoundError(f"Summary file not found: {summary_csv_path}")
    if not os.path.exists(diagnostics_csv_path):
        raise FileNotFoundError(f"Diagnostics file not found: {diagnostics_csv_path}")
    return pd.read_csv(summary_csv_path), pd.read_csv(diagnostics_csv_path)


def replot_saved_benchmark(
    *,
    output_prefix: str,
) -> dict[str, str]:
    """Regenerate the benchmark figures from the saved CSV files only."""
    summary_df, diagnostics_df = load_saved_benchmark_results(output_prefix=output_prefix)
    figures_dir = rl_figures_dir()
    returns_figure_path = os.path.join(figures_dir, f"{output_prefix}_returns.png")
    diagnostics_figure_path = os.path.join(figures_dir, f"{output_prefix}_dynamics.png")
    plot_benchmark_results(summary_df=summary_df, output_path=returns_figure_path)
    plot_benchmark_dynamics(diagnostics_df=diagnostics_df, output_path=diagnostics_figure_path)
    return {
        "returns_figure_path": returns_figure_path,
        "diagnostics_figure_path": diagnostics_figure_path,
    }


def run_wolf_selection(
    *,
    config: DefenseBenchmarkConfig,
    output_prefix: str,
) -> dict[str, str]:
    """Run the standalone WoLF tuning workflow and save its artifacts."""
    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, _env_config, _net_config, _train_config, _history = load_agent_checkpoint(model_path, device)
    policy.eval()

    tuning_episodes, _evaluation_episodes = build_episode_sets(config)
    full_df, ranking_df = run_wolf_sweep(
        policy=policy,
        config=config,
        tuning_episodes=tuning_episodes,
        device=device,
    )
    selected_parameters = select_best_wolf_parameters(full_df=full_df)
    return save_wolf_selection_artifacts(
        config=config,
        full_df=full_df,
        ranking_df=ranking_df,
        selected_parameters=selected_parameters,
        output_prefix=output_prefix,
    )


def ensure_best_wolf_params(
    *,
    config: DefenseBenchmarkConfig,
    force_rerun: bool,
    output_prefix: str,
) -> dict[str, str] | None:
    """
    Ensure the benchmark has a WoLF selection file available.

    In benchmark mode we automatically run the standalone WoLF tuning stage
    first when:
    1. the selection file does not exist, or
    2. the caller explicitly forces a fresh tuning pass.
    """
    json_path = best_wolf_params_path()
    if not force_rerun and os.path.exists(json_path):
        return None

    reason = "forced rerun" if force_rerun else "missing best_wolf_params.json"
    print(f"[benchmark] Running WoLF tuning first ({reason}).")
    return run_wolf_selection(
        config=config,
        output_prefix=output_prefix,
    )


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments shared by the benchmark entry points."""
    parser = argparse.ArgumentParser(description="Wind RL defense benchmark.")
    parser.add_argument("--mode", choices=("benchmark", "wolf", "plot"), default="benchmark")
    parser.add_argument("--n-tuning-episodes", "--tuning-seeds", dest="n_tuning_episodes", type=int, default=None)
    parser.add_argument("--n-episodes", "--evaluation-seeds", dest="n_episodes", type=int, default=None)
    parser.add_argument("--lambdas", type=float, nargs="+", default=None)
    parser.add_argument("--gamma-threshold", type=float, default=None)
    parser.add_argument("--attack-steps", type=int, default=None)
    parser.add_argument("--attack-mc-samples", type=int, default=None)
    parser.add_argument("--transition-mc-samples", type=int, default=None)
    parser.add_argument("--attack-step-size", type=float, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--attack-probability", type=float, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--force-wolf-tuning", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DefenseBenchmarkConfig:
    """Override the default benchmark config with command-line arguments."""
    config = build_default_benchmark_config()
    overrides: dict[str, Any] = {}
    if args.n_tuning_episodes is not None:
        overrides["n_tuning_episodes"] = int(args.n_tuning_episodes)
    if args.n_episodes is not None:
        overrides["n_episodes"] = int(args.n_episodes)
    if args.lambdas is not None:
        overrides["lambdas"] = tuple(float(value) for value in args.lambdas)
    if args.gamma_threshold is not None:
        overrides["gamma_threshold"] = float(args.gamma_threshold)
    if args.attack_steps is not None:
        overrides["attack_num_steps"] = int(args.attack_steps)
    if args.attack_mc_samples is not None:
        overrides["attack_mc_samples"] = int(args.attack_mc_samples)
    if args.transition_mc_samples is not None:
        overrides["transition_mc_samples"] = int(args.transition_mc_samples)
    if args.attack_step_size is not None:
        overrides["attack_step_size"] = float(args.attack_step_size)
    if args.coverage is not None:
        overrides["coverage"] = float(args.coverage)
    if args.attack_probability is not None:
        overrides["attack_probability"] = float(args.attack_probability)
    return replace(config, **overrides)


def main() -> None:
    """Run either the standalone WoLF sweep or the final defense benchmark."""
    args = parse_args()
    config = config_from_args(args)

    if args.mode == "wolf":
        output_prefix = args.output_prefix or "wolf_sweep"
        artifact_paths = run_wolf_selection(
            config=config,
            output_prefix=output_prefix,
        )
        for label, path in artifact_paths.items():
            print(f"{label}: {path}")
        return

    if args.mode == "plot":
        output_prefix = args.output_prefix or "wind_defense_benchmark"
        artifact_paths = replot_saved_benchmark(output_prefix=output_prefix)
        for label, path in artifact_paths.items():
            print(f"{label}: {path}")
        return

    output_prefix = args.output_prefix or "wind_defense_benchmark"
    wolf_output_prefix = f"{output_prefix}_wolf_tuning"
    wolf_artifact_paths = ensure_best_wolf_params(
        config=config,
        force_rerun=bool(args.force_wolf_tuning),
        output_prefix=wolf_output_prefix,
    )
    if wolf_artifact_paths is not None:
        for label, path in wolf_artifact_paths.items():
            print(f"wolf_{label}: {path}")
    artifact_paths = run_full_benchmark(
        config=config,
        output_prefix=output_prefix,
    )
    for label, path in artifact_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
