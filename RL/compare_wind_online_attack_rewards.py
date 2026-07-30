#!/usr/bin/env python3
"""
Compare wind-navigation returns under clean, noisy, PGD, and random attacks.

The script loads the trained PPO wind agent from `RL/model/`, runs a small
deterministic evaluation study, and saves a grouped bar plot in `RL/figures/`.

Compared settings:
1. Noiseless: the policy receives the true hidden state `s_t`.
2. Noisy + KF: noisy observations are filtered online; the policy receives the
   filtered mean.
3. PGD attack: with probability 0.10, the latest noisy observation is replaced
   by a worst-case observation found by the shared `shared_ssm` Torch attack.
4. Estimated-return PGD: with the same probability, the latest observation is
   attacked by minimizing a one-step Bellman objective evaluated at the noisy-KF
   current-state estimate, not at the simulator's hidden state.
5. Real-return PGD: an oracle diagnostic that uses the simulator's hidden state
   in the one-step Bellman objective.
6. Random heavy-tail: with the same probability, the latest observation is
   replaced by a Student-t perturbation without projection.
7. Uniform annulus: with the same probability, the latest observation is
   sampled uniformly between the 0.50 coverage ellipsoid and the current
   0.75/0.95 ellipsoid.
8. Random contour: with the same probability, the latest observation is
   replaced by a random point on the same ellipsoid contour.

The user-requested epsilon values are coverage probabilities, not squared
Mahalanobis radii. They are converted with `chi2.ppf(coverage, df=4)` because
the wind observation has dimension four.

Terminal diagnostics report the episode progress, the percentage of episodes
that reach the goal, and the critic values `V(s_t)` before and after every
attacked online-filter update.
"""

from __future__ import annotations

from dataclasses import replace
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scipy.stats import chi2
except ModuleNotFoundError:
    chi2 = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared_ssm import LinearGaussianStateSpaceModel
from shared_ssm import build_attack_geometry
from shared_ssm import run_kalman_inference
from shared_ssm.attacks import TorchRLAttackConfig
from shared_ssm.attacks import solve_torch_rl_expectation_attack
from shared_ssm.linalg import sqrtm_psd
from wind_rl_setup import WindNavigationBatch
from wind_rl_setup import WindNavigationConfig
from wind_rl_setup import build_control_matrix
from wind_rl_setup import build_observation_matrix
from wind_rl_setup import build_process_covariance
from wind_rl_setup import build_transition_matrix
from wind_rl_setup import default_device
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_data_dir
from wind_rl_setup import rl_figures_dir
from wind_rl_setup import rl_model_dir


SETTING_NAMES = (
    "noiseless",
    "noisy + KF",
    "PGD attack",
    "estimated-return PGD",
    "real-return PGD",
    "random heavy-tail",
    "uniform annulus",
    "random contour",
)


def coverage_to_mahalanobis_epsilon(coverage: float, obs_dim: int) -> float:
    """Convert an ellipsoid coverage probability into Mahalanobis radius squared."""
    coverage = float(coverage)
    if chi2 is not None:
        return float(chi2.ppf(coverage, df=int(obs_dim)))

    fallback_values = {
        (4, 0.50): 3.3566939800333224,
        (4, 0.60): 4.044626490649313,
        (4, 0.75): 5.385269057779394,
        (4, 0.95): 9.487729036781154,
    }
    key = (int(obs_dim), round(coverage, 2))
    if key not in fallback_values:
        raise ModuleNotFoundError("scipy is required for this coverage level.")
    return fallback_values[key]


def noisy_state_observation(
    *,
    state: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return `o_t = s_t + noise` for the 4D wind state."""
    state = np.asarray(state, dtype=float).reshape(4)
    if float(noise_std) <= 0.0:
        return state.copy()
    return state + rng.normal(0.0, float(noise_std), size=4)


def build_history_model(
    *,
    filter_config: WindNavigationConfig,
    initial_hidden_state: np.ndarray,
    transitions: list[np.ndarray],
) -> LinearGaussianStateSpaceModel:
    """Build the time-varying LGSSM for the available online history."""
    return LinearGaussianStateSpaceModel(
        A=np.asarray(transitions, dtype=float),
        B=build_control_matrix(),
        F=build_observation_matrix(),
        G=np.zeros((4, 2), dtype=float),
        W=build_process_covariance(filter_config),
        V=build_observation_covariance_from_std(float(filter_config.observation_noise_std)),
        m0=np.asarray(initial_hidden_state, dtype=float),
        P0=1e-4 * np.eye(4, dtype=float),
    )


def build_observation_covariance_from_std(observation_noise_std: float) -> np.ndarray:
    """Return the 4D diagonal observation covariance used by the online filter."""
    obs_var = float(observation_noise_std) ** 2
    return obs_var * np.eye(4, dtype=float)


def filtered_state_mean(
    *,
    filter_config: WindNavigationConfig,
    initial_hidden_state: np.ndarray,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    transitions: list[np.ndarray],
) -> np.ndarray:
    """Return the current online KF posterior mean from the history."""
    model = build_history_model(
        filter_config=filter_config,
        initial_hidden_state=initial_hidden_state,
        transitions=transitions,
    )
    inference = run_kalman_inference(
        model=model,
        observations=np.asarray(observations, dtype=float),
        actions=np.asarray(actions, dtype=float),
        mode="online",
    )
    return np.asarray(inference.filtered_state_means[-1], dtype=float)


def random_observation_on_contour(
    *,
    geometry: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a random observation on the same ellipsoid contour as the attack."""
    direction = rng.normal(size=geometry.constraint.center.size)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        direction = np.ones_like(direction)
        direction_norm = float(np.linalg.norm(direction))
    unit_direction = direction / direction_norm
    covariance_sqrt = sqrtm_psd(geometry.constraint.covariance)
    radius = np.sqrt(float(geometry.constraint.epsilon))
    return geometry.constraint.center + covariance_sqrt @ (radius * unit_direction)


def random_heavy_tail_observation(
    *,
    geometry: Any,
    rng: np.random.Generator,
    degrees_of_freedom: float,
) -> np.ndarray:
    """
    Sample an unprojected Student-t perturbation in predictive-observation space.

    The scale is chosen so the expected squared Mahalanobis radius matches the
    current ellipsoid budget when the Student-t variance exists, but individual
    draws may lie outside the ellipsoid because this baseline is intentionally
    not projected.
    """
    obs_dim = int(geometry.constraint.center.size)
    df = float(degrees_of_freedom)
    if df <= 0.0:
        raise ValueError("degrees_of_freedom must be positive.")

    gaussian = rng.normal(size=obs_dim)
    chi_square = max(float(rng.chisquare(df)), 1e-12)
    student_t = gaussian / np.sqrt(chi_square / df)
    if df > 2.0:
        scale = np.sqrt(float(geometry.constraint.epsilon) / (obs_dim * df / (df - 2.0)))
    else:
        scale = np.sqrt(float(geometry.constraint.epsilon) / max(obs_dim, 1))

    covariance_sqrt = sqrtm_psd(geometry.constraint.covariance)
    return geometry.constraint.center + covariance_sqrt @ (scale * student_t)


def random_observation_in_coverage_annulus(
    *,
    geometry: Any,
    min_coverage: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample uniformly in volume between two Mahalanobis ellipsoid levels."""
    obs_dim = int(geometry.constraint.center.size)
    lower_epsilon = coverage_to_mahalanobis_epsilon(float(min_coverage), obs_dim=obs_dim)
    upper_epsilon = float(geometry.constraint.epsilon)
    if lower_epsilon >= upper_epsilon:
        raise ValueError("min_coverage must be smaller than the evaluated coverage.")

    direction = rng.normal(size=obs_dim)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)

    # Uniform volume in d dimensions has radial CDF proportional to r^d.
    lower_power = float(lower_epsilon) ** (0.5 * obs_dim)
    upper_power = float(upper_epsilon) ** (0.5 * obs_dim)
    radius_power = rng.uniform(lower_power, upper_power)
    mahalanobis_sq = radius_power ** (2.0 / max(obs_dim, 1))

    covariance_sqrt = sqrtm_psd(geometry.constraint.covariance)
    return geometry.constraint.center + covariance_sqrt @ (np.sqrt(mahalanobis_sq) * direction)


def critic_value_for_state(
    *,
    policy: torch.nn.Module,
    state_mean: np.ndarray,
    goal_xy: np.ndarray,
    device: torch.device,
) -> float:
    """Return the critic value `V(s_t)` for one state estimate and goal."""
    policy_observation = np.concatenate([state_mean, goal_xy]).astype(np.float32)
    obs_tensor = torch.as_tensor(policy_observation[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        value = policy.value(obs_tensor).cpu().numpy()[0]
    return float(value)


def is_on_ellipsoid_contour(
    *,
    mahalanobis_sq: float,
    mahalanobis_epsilon: float,
    relative_tolerance: float,
) -> bool:
    """Return whether a Mahalanobis value is close to the ellipsoid boundary."""
    tolerance = float(relative_tolerance) * max(float(mahalanobis_epsilon), 1.0)
    return abs(float(mahalanobis_sq) - float(mahalanobis_epsilon)) <= tolerance


def is_inside_ellipsoid(
    *,
    mahalanobis_sq: float,
    mahalanobis_epsilon: float,
    relative_tolerance: float,
) -> bool:
    """Return whether a Mahalanobis value is inside the ellipsoid budget."""
    tolerance = float(relative_tolerance) * max(float(mahalanobis_epsilon), 1.0)
    return float(mahalanobis_sq) <= float(mahalanobis_epsilon) + tolerance


def pgd_attack_observation(
    *,
    policy: torch.nn.Module,
    geometry: Any,
    clean_observation: np.ndarray,
    goal_xy: np.ndarray,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Return the worst-case current observation found by the Torch PGD attack."""
    goal_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)

    def critic_objective(samples: torch.Tensor) -> torch.Tensor:
        goal_batch = goal_t.unsqueeze(0).expand(samples.shape[0], -1)
        policy_input = torch.cat([samples, goal_batch], dim=-1)
        return policy.value(policy_input)

    attack = solve_torch_rl_expectation_attack(
        geometry=geometry,
        clean_observation=clean_observation,
        objective=critic_objective,
        config=TorchRLAttackConfig(
            direction="minimize",
            step_size=float(attack_step_size),
            num_steps=int(attack_num_steps),
            num_mc_samples=int(attack_mc_samples),
            seed=int(seed),
            device=str(device),
        ),
    )
    return np.asarray(attack.adversarial_observation, dtype=float)


def bellman_return_pgd_attack_observation(
    *,
    policy: torch.nn.Module,
    geometry: Any,
    clean_observation: np.ndarray,
    base_hidden_state: np.ndarray,
    goal_xy: np.ndarray,
    simulation_config: WindNavigationConfig,
    current_step_index: int,
    gamma: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    transition_mc_samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """
    Attack a one-step Bellman objective induced by the perceived state.

    `base_hidden_state` is the current state used by the attacker to simulate
    consequences of the induced action. Passing the simulator hidden state gives
    an oracle attack; passing the noisy-KF posterior mean gives the realistic
    estimated-state attack.
    """
    goal_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)
    base_state_t = torch.as_tensor(base_hidden_state, dtype=torch.float32, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 911_731)
    num_posterior = int(attack_mc_samples)
    num_transition = int(transition_mc_samples)
    delta_psi = torch.randn(
        (num_posterior, num_transition),
        generator=gen,
        device=device,
        dtype=torch.float32,
    ) * float(simulation_config.wind_turn_std)
    process_std = torch.tensor(
        [
            float(simulation_config.process_position_std),
            float(simulation_config.process_position_std),
            float(simulation_config.process_wind_std),
            float(simulation_config.process_wind_std),
        ],
        dtype=torch.float32,
        device=device,
    )
    process_noise = torch.randn(
        (num_posterior, num_transition, 4),
        generator=gen,
        device=device,
        dtype=torch.float32,
    ) * process_std

    def bellman_objective(samples: torch.Tensor) -> torch.Tensor:
        sample_count = samples.shape[0]
        used_delta_psi = delta_psi[:sample_count]
        used_noise = process_noise[:sample_count]

        goal_batch = goal_t.unsqueeze(0).expand(sample_count, -1)
        perceived_policy_input = torch.cat([samples, goal_batch], dim=-1)
        action = policy.deterministic_action(perceived_policy_input)
        action = torch.clamp(
            action,
            min=-float(simulation_config.action_limit),
            max=float(simulation_config.action_limit),
        )

        action_expanded = action.unsqueeze(1).expand(-1, num_transition, -1)
        base_position = base_state_t[:2].view(1, 1, 2)
        base_wind = base_state_t[2:4].view(1, 1, 2)

        cos_psi = torch.cos(used_delta_psi)
        sin_psi = torch.sin(used_delta_psi)
        rotated_wind_x = float(simulation_config.rho_w) * (
            cos_psi * base_wind[..., 0] - sin_psi * base_wind[..., 1]
        )
        rotated_wind_y = float(simulation_config.rho_w) * (
            sin_psi * base_wind[..., 0] + cos_psi * base_wind[..., 1]
        )
        rotated_wind = torch.stack([rotated_wind_x, rotated_wind_y], dim=-1)

        next_state = torch.empty((sample_count, num_transition, 4), dtype=torch.float32, device=device)
        next_state[..., :2] = base_position + base_wind + action_expanded + used_noise[..., :2]
        next_state[..., 2:] = rotated_wind + used_noise[..., 2:]

        goal_error = next_state[..., :2] - goal_t.view(1, 1, 2)
        goal_distance = torch.linalg.norm(goal_error, dim=-1)
        distance_reward = -goal_distance / max(float(simulation_config.radius_max), 1e-6)

        reached_goal = goal_distance <= float(simulation_config.goal_radius)
        reached_horizon = (int(current_step_index) + 1) >= int(simulation_config.max_steps)
        timeout_only = torch.logical_and(
            torch.full_like(reached_goal, bool(reached_horizon), dtype=torch.bool),
            torch.logical_not(reached_goal),
        )
        reward = distance_reward
        reward = reward + reached_goal.to(torch.float32) * float(simulation_config.goal_reward)
        reward = reward + timeout_only.to(torch.float32) * float(simulation_config.timeout_penalty)

        flat_next_state = next_state.reshape(-1, 4)
        flat_goal = goal_t.view(1, 2).expand(flat_next_state.shape[0], -1)
        next_policy_input = torch.cat([flat_next_state, flat_goal], dim=-1)
        next_value = policy.value(next_policy_input).reshape(sample_count, num_transition)
        nonterminal = torch.logical_not(torch.logical_or(reached_goal, timeout_only)).to(torch.float32)
        one_step_return = reward + float(gamma) * nonterminal * next_value
        return one_step_return.mean(dim=1)

    attack = solve_torch_rl_expectation_attack(
        geometry=geometry,
        clean_observation=clean_observation,
        objective=bellman_objective,
        config=TorchRLAttackConfig(
            direction="minimize",
            step_size=float(attack_step_size),
            num_steps=int(attack_num_steps),
            num_mc_samples=int(attack_mc_samples),
            seed=int(seed),
            device=str(device),
        ),
    )
    return np.asarray(attack.adversarial_observation, dtype=float)


def real_return_pgd_attack_observation(
    *,
    policy: torch.nn.Module,
    geometry: Any,
    clean_observation: np.ndarray,
    actual_hidden_state: np.ndarray,
    goal_xy: np.ndarray,
    simulation_config: WindNavigationConfig,
    current_step_index: int,
    gamma: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    transition_mc_samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Return the oracle Bellman PGD attack that uses the simulator `s_t`."""
    return bellman_return_pgd_attack_observation(
        policy=policy,
        geometry=geometry,
        clean_observation=clean_observation,
        base_hidden_state=actual_hidden_state,
        goal_xy=goal_xy,
        simulation_config=simulation_config,
        current_step_index=int(current_step_index),
        gamma=float(gamma),
        attack_step_size=float(attack_step_size),
        attack_num_steps=int(attack_num_steps),
        attack_mc_samples=int(attack_mc_samples),
        transition_mc_samples=int(transition_mc_samples),
        seed=int(seed),
        device=device,
    )


def estimated_return_pgd_attack_observation(
    *,
    policy: torch.nn.Module,
    geometry: Any,
    clean_observation: np.ndarray,
    estimated_hidden_state: np.ndarray,
    goal_xy: np.ndarray,
    simulation_config: WindNavigationConfig,
    current_step_index: int,
    gamma: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    transition_mc_samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Return the Bellman PGD attack based only on the noisy-KF state estimate."""
    return bellman_return_pgd_attack_observation(
        policy=policy,
        geometry=geometry,
        clean_observation=clean_observation,
        base_hidden_state=estimated_hidden_state,
        goal_xy=goal_xy,
        simulation_config=simulation_config,
        current_step_index=int(current_step_index),
        gamma=float(gamma),
        attack_step_size=float(attack_step_size),
        attack_num_steps=int(attack_num_steps),
        attack_mc_samples=int(attack_mc_samples),
        transition_mc_samples=int(transition_mc_samples),
        seed=int(seed),
        device=device,
    )


def current_policy_state(
    *,
    setting: str,
    policy: torch.nn.Module,
    simulation_config: WindNavigationConfig,
    filter_config: WindNavigationConfig,
    initial_hidden_state: np.ndarray,
    actual_hidden_state: np.ndarray,
    goal_xy: np.ndarray,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    transitions: list[np.ndarray],
    current_noisy_observation: np.ndarray,
    current_step_index: int,
    mahalanobis_epsilon: float,
    gamma: float,
    attack_prob: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    real_attack_transition_samples: int,
    heavy_tail_degrees_of_freedom: float,
    uniform_annulus_min_coverage: float,
    attack_rng: np.random.Generator,
    random_rng: np.random.Generator,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, bool, dict[str, float] | None]:
    """Return the policy state, attack flag, and optional value diagnostics."""
    if not observations:
        return np.asarray(current_noisy_observation, dtype=float), False, None

    attack_settings = {
        "PGD attack",
        "estimated-return PGD",
        "real-return PGD",
        "random heavy-tail",
        "uniform annulus",
        "random contour",
    }
    do_attack = setting in attack_settings and attack_rng.random() < float(attack_prob)
    if not do_attack:
        state_mean = filtered_state_mean(
            filter_config=filter_config,
            initial_hidden_state=initial_hidden_state,
            observations=observations,
            actions=actions,
            transitions=transitions,
        )
        return state_mean, False, None

    model = build_history_model(
        filter_config=filter_config,
        initial_hidden_state=initial_hidden_state,
        transitions=transitions,
    )
    geometry = build_attack_geometry(
        model=model,
        observations=np.asarray(observations, dtype=float),
        actions=np.asarray(actions, dtype=float),
        observation_index=len(observations) - 1,
        epsilon=float(mahalanobis_epsilon),
        mode="online",
    )
    noisy_state_mean = geometry.posterior_mean_for_observation(current_noisy_observation)
    noisy_value = critic_value_for_state(
        policy=policy,
        state_mean=noisy_state_mean,
        goal_xy=goal_xy,
        device=device,
    )

    if setting == "PGD attack":
        attacked_observation = pgd_attack_observation(
            policy=policy,
            geometry=geometry,
            clean_observation=current_noisy_observation,
            goal_xy=goal_xy,
            attack_step_size=attack_step_size,
            attack_num_steps=attack_num_steps,
            attack_mc_samples=attack_mc_samples,
            seed=seed,
            device=device,
        )
    elif setting == "estimated-return PGD":
        attacked_observation = estimated_return_pgd_attack_observation(
            policy=policy,
            geometry=geometry,
            clean_observation=current_noisy_observation,
            estimated_hidden_state=noisy_state_mean,
            goal_xy=goal_xy,
            simulation_config=simulation_config,
            current_step_index=int(current_step_index),
            gamma=float(gamma),
            attack_step_size=attack_step_size,
            attack_num_steps=attack_num_steps,
            attack_mc_samples=attack_mc_samples,
            transition_mc_samples=real_attack_transition_samples,
            seed=seed,
            device=device,
        )
    elif setting == "real-return PGD":
        attacked_observation = real_return_pgd_attack_observation(
            policy=policy,
            geometry=geometry,
            clean_observation=current_noisy_observation,
            actual_hidden_state=actual_hidden_state,
            goal_xy=goal_xy,
            simulation_config=simulation_config,
            current_step_index=int(current_step_index),
            gamma=float(gamma),
            attack_step_size=attack_step_size,
            attack_num_steps=attack_num_steps,
            attack_mc_samples=attack_mc_samples,
            transition_mc_samples=real_attack_transition_samples,
            seed=seed,
            device=device,
        )
    elif setting == "random heavy-tail":
        attacked_observation = random_heavy_tail_observation(
            geometry=geometry,
            rng=random_rng,
            degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
        )
    elif setting == "uniform annulus":
        attacked_observation = random_observation_in_coverage_annulus(
            geometry=geometry,
            min_coverage=float(uniform_annulus_min_coverage),
            rng=random_rng,
        )
    else:
        attacked_observation = random_observation_on_contour(
            geometry=geometry,
            rng=random_rng,
        )

    # The attacked observation becomes part of the online history, so later
    # policy decisions are conditioned on the compromised filter belief.
    observations[-1] = np.asarray(attacked_observation, dtype=float)
    state_mean = geometry.posterior_mean_for_observation(attacked_observation)
    attacked_value = critic_value_for_state(
        policy=policy,
        state_mean=state_mean,
        goal_xy=goal_xy,
        device=device,
    )
    state_perturbation = np.asarray(state_mean, dtype=float) - np.asarray(noisy_state_mean, dtype=float)
    observation_perturbation = (
        np.asarray(attacked_observation, dtype=float) - np.asarray(current_noisy_observation, dtype=float)
    )
    diagnostics = {
        "value_noisy_kf": float(noisy_value),
        "value_attacked": float(attacked_value),
        "delta_value": float(attacked_value - noisy_value),
        "state_position_perturbation_norm": float(np.linalg.norm(state_perturbation[:2])),
        "state_wind_perturbation_norm": float(np.linalg.norm(state_perturbation[2:])),
        "observation_position_perturbation_norm": float(np.linalg.norm(observation_perturbation[:2])),
        "observation_wind_perturbation_norm": float(np.linalg.norm(observation_perturbation[2:])),
        "mahalanobis_sq": float(geometry.constraint.radius_squared(attacked_observation)),
    }
    return np.asarray(state_mean, dtype=float), True, diagnostics


def evaluate_episode(
    *,
    policy: torch.nn.Module,
    simulation_config: WindNavigationConfig,
    filter_config: WindNavigationConfig,
    setting: str,
    coverage: float,
    seed: int,
    observation_noise_std: float,
    gamma: float,
    attack_prob: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    real_attack_transition_samples: int,
    heavy_tail_degrees_of_freedom: float,
    uniform_annulus_min_coverage: float,
    contour_relative_tolerance: float,
    episode_number: int,
    n_episodes: int,
    verbose: bool,
    device: torch.device,
) -> dict[str, float | int | list[dict[str, float | int | str]]]:
    """Run one episode for one evaluation setting."""
    env = WindNavigationBatch(simulation_config, num_envs=1, seed=int(seed))
    _initial_observation = env.reset_all()
    initial_hidden_state = env.hidden_state[0].copy()
    goal_xy = env.goal_xy[0].copy()

    obs_rng = np.random.default_rng(int(seed) + 17_003)
    attack_rng = np.random.default_rng(int(seed) + 100_003)
    random_rng = np.random.default_rng(int(seed) + 201_337)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    transitions: list[np.ndarray] = []
    current_noisy_observation = noisy_state_observation(
        state=env.hidden_state[0],
        noise_std=observation_noise_std,
        rng=obs_rng,
    )

    mahalanobis_epsilon = coverage_to_mahalanobis_epsilon(float(coverage), obs_dim=4)
    accumulated_reward = 0.0
    num_attacks = 0
    attack_value_records: list[dict[str, float | int | str]] = []
    done = False
    reached_goal = False

    while not done:
        step_index = int(env.step_index[0])
        if setting == "noiseless":
            policy_state = env.hidden_state[0].astype(float).copy()
            attacked = False
            attack_diagnostics = None
        else:
            policy_state, attacked, attack_diagnostics = current_policy_state(
                setting=setting,
                policy=policy,
                simulation_config=simulation_config,
                filter_config=filter_config,
                initial_hidden_state=initial_hidden_state,
                actual_hidden_state=env.hidden_state[0].copy(),
                goal_xy=goal_xy,
                observations=observations,
                actions=actions,
                transitions=transitions,
                current_noisy_observation=current_noisy_observation,
                current_step_index=int(step_index),
                mahalanobis_epsilon=mahalanobis_epsilon,
                gamma=float(gamma),
                attack_prob=attack_prob,
                attack_step_size=attack_step_size,
                attack_num_steps=attack_num_steps,
                attack_mc_samples=attack_mc_samples,
                real_attack_transition_samples=int(real_attack_transition_samples),
                heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
                uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
                attack_rng=attack_rng,
                random_rng=random_rng,
                seed=int(seed) + len(actions) + 1_000,
                device=device,
            )

        num_attacks += int(attacked)
        if attacked and attack_diagnostics is not None:
            record = {
                "coverage": float(coverage),
                "setting": setting,
                "episode": int(episode_number),
                "step": int(step_index),
                "mahalanobis_epsilon": float(mahalanobis_epsilon),
                **attack_diagnostics,
            }
            record["contour_gap"] = float(record["mahalanobis_epsilon"] - record["mahalanobis_sq"])
            record["inside_ellipsoid"] = bool(
                is_inside_ellipsoid(
                    mahalanobis_sq=float(record["mahalanobis_sq"]),
                    mahalanobis_epsilon=float(record["mahalanobis_epsilon"]),
                    relative_tolerance=float(contour_relative_tolerance),
                )
            )
            record["on_contour"] = bool(
                is_on_ellipsoid_contour(
                    mahalanobis_sq=float(record["mahalanobis_sq"]),
                    mahalanobis_epsilon=float(record["mahalanobis_epsilon"]),
                    relative_tolerance=float(contour_relative_tolerance),
                )
            )
            attack_value_records.append(record)
            if verbose:
                print(
                    "    [attack] "
                    f"episode={episode_number:02d}/{n_episodes:02d} "
                    f"step={step_index:02d} "
                    f"setting={setting:20s} "
                    f"V_noisyKF={record['value_noisy_kf']:8.4f} "
                    f"V_attacked={record['value_attacked']:8.4f} "
                    f"delta={record['delta_value']:8.4f} "
                    f"|dp|={record['state_position_perturbation_norm']:6.3f} "
                    f"|dw|={record['state_wind_perturbation_norm']:6.3f} "
                    f"maha={record['mahalanobis_sq']:7.3f} "
                    f"gap={record['contour_gap']:7.3f} "
                    f"inside={str(record['inside_ellipsoid']):5s} "
                    f"contour={str(record['on_contour']):5s}"
                )

        policy_observation = np.concatenate([policy_state, goal_xy]).astype(np.float32)
        obs_tensor = torch.as_tensor(policy_observation[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            action = policy.deterministic_action(obs_tensor).cpu().numpy()[0]

        _next_obs, reward, done_mask, info = env.step(action[None, :])
        accumulated_reward += float(reward[0])
        reached_goal = bool(info["reached_goal"][0])

        if setting != "noiseless":
            current_noisy_observation = noisy_state_observation(
                state=env.hidden_state[0],
                noise_std=observation_noise_std,
                rng=obs_rng,
            )
            observations.append(current_noisy_observation.copy())
            actions.append(action.copy())
            transitions.append(
                build_transition_matrix(
                    rho_w=float(simulation_config.rho_w),
                    delta_psi=float(info["delta_psi"][0]),
                )
            )

        done = bool(done_mask[0])

    return {
        "return": float(accumulated_reward),
        "length": int(len(actions) if setting != "noiseless" else env.step_index[0]),
        "reached_goal": bool(reached_goal),
        "num_attacks": int(num_attacks),
        "attack_value_records": attack_value_records,
    }


def evaluate_setting(
    *,
    policy: torch.nn.Module,
    simulation_config: WindNavigationConfig,
    filter_config: WindNavigationConfig,
    setting: str,
    coverage: float,
    episode_seeds: np.ndarray,
    observation_noise_std: float,
    gamma: float,
    attack_prob: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    real_attack_transition_samples: int,
    heavy_tail_degrees_of_freedom: float,
    uniform_annulus_min_coverage: float,
    contour_relative_tolerance: float,
    verbose: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one setting for all episode seeds."""
    returns = []
    lengths = []
    successes = []
    num_attacks = []
    attack_value_records: list[dict[str, float | int | str]] = []

    n_episodes = int(len(episode_seeds))
    for episode_idx, seed in enumerate(episode_seeds, start=1):
        if verbose:
            print(
                f"  [{setting:20s}] episode {episode_idx:02d}/{n_episodes:02d} "
                f"seed={int(seed)}"
            )
        result = evaluate_episode(
            policy=policy,
            simulation_config=simulation_config,
            filter_config=filter_config,
            setting=setting,
            coverage=coverage,
            seed=int(seed),
            observation_noise_std=observation_noise_std,
            gamma=float(gamma),
            attack_prob=attack_prob,
            attack_step_size=attack_step_size,
            attack_num_steps=attack_num_steps,
            attack_mc_samples=attack_mc_samples,
            real_attack_transition_samples=int(real_attack_transition_samples),
            heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
            uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
            contour_relative_tolerance=float(contour_relative_tolerance),
            episode_number=int(episode_idx),
            n_episodes=int(n_episodes),
            verbose=verbose,
            device=device,
        )
        returns.append(float(result["return"]))
        lengths.append(int(result["length"]))
        successes.append(float(result["reached_goal"]))
        num_attacks.append(int(result["num_attacks"]))
        attack_value_records.extend(result["attack_value_records"])

    return {
        "returns": np.asarray(returns, dtype=float),
        "lengths": np.asarray(lengths, dtype=int),
        "successes": np.asarray(successes, dtype=float),
        "num_attacks": np.asarray(num_attacks, dtype=int),
        "attack_value_records": attack_value_records,
    }


def set_plot_theme() -> None:
    """Apply a compact pastel theme for the comparison bar plot."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10.2,
            "axes.labelsize": 11.0,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.70,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
            "savefig.facecolor": "white",
        }
    )


def plot_mean_return_bars(
    *,
    coverages: tuple[float, ...],
    mean_returns: np.ndarray,
    sem_returns: np.ndarray,
    outpath: str,
) -> None:
    """Save a grouped bar plot of mean accumulated returns."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(10.6, 5.4))

    colors = [
        "#8EC5B5",
        "#9DB7D5",
        "#E6A57E",
        "#F0C987",
        "#D8B07A",
        "#B8D88A",
        "#C6B18F",
        "#CFA7D8",
    ]
    x_positions = np.arange(len(coverages), dtype=float)
    width = min(0.16, 0.78 / max(len(SETTING_NAMES), 1))
    offsets = (np.arange(len(SETTING_NAMES)) - 0.5 * (len(SETTING_NAMES) - 1)) * width

    for setting_idx, (setting_name, color) in enumerate(zip(SETTING_NAMES, colors)):
        ax.bar(
            x_positions + offsets[setting_idx],
            mean_returns[:, setting_idx],
            width=width,
            yerr=sem_returns[:, setting_idx],
            capsize=3.0,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=setting_name,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([rf"$\epsilon={coverage:.2f}$" for coverage in coverages])
    ax.set_ylabel("Mean accumulated reward")
    ax.set_xlabel("Ellipsoid coverage constraint")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.24)
    ax.grid(False, axis="x")
    ax.legend(loc="best", ncols=2, frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def attack_records_to_arrays(records: list[dict[str, float | int | str]]) -> dict[str, np.ndarray]:
    """Convert attack value diagnostics into NPZ-friendly arrays."""
    return {
        "attack_record_coverage": np.asarray([record["coverage"] for record in records], dtype=float),
        "attack_record_setting": np.asarray([record["setting"] for record in records], dtype=object),
        "attack_record_episode": np.asarray([record["episode"] for record in records], dtype=int),
        "attack_record_step": np.asarray([record["step"] for record in records], dtype=int),
        "attack_record_mahalanobis_epsilon": np.asarray(
            [record["mahalanobis_epsilon"] for record in records],
            dtype=float,
        ),
        "attack_record_value_noisy_kf": np.asarray([record["value_noisy_kf"] for record in records], dtype=float),
        "attack_record_value_attacked": np.asarray([record["value_attacked"] for record in records], dtype=float),
        "attack_record_delta_value": np.asarray([record["delta_value"] for record in records], dtype=float),
        "attack_record_state_position_perturbation_norm": np.asarray(
            [record["state_position_perturbation_norm"] for record in records],
            dtype=float,
        ),
        "attack_record_state_wind_perturbation_norm": np.asarray(
            [record["state_wind_perturbation_norm"] for record in records],
            dtype=float,
        ),
        "attack_record_observation_position_perturbation_norm": np.asarray(
            [record["observation_position_perturbation_norm"] for record in records],
            dtype=float,
        ),
        "attack_record_observation_wind_perturbation_norm": np.asarray(
            [record["observation_wind_perturbation_norm"] for record in records],
            dtype=float,
        ),
        "attack_record_mahalanobis_sq": np.asarray([record["mahalanobis_sq"] for record in records], dtype=float),
        "attack_record_contour_gap": np.asarray([record["contour_gap"] for record in records], dtype=float),
        "attack_record_inside_ellipsoid": np.asarray([record["inside_ellipsoid"] for record in records], dtype=bool),
        "attack_record_on_contour": np.asarray([record["on_contour"] for record in records], dtype=bool),
    }


def attack_contour_summary(
    *,
    records: list[dict[str, float | int | str]],
    relative_tolerance: float,
) -> dict[str, float | int]:
    """Summarize how often attacks lie on the ellipsoid contour."""
    if not records:
        return {
            "count": 0,
            "on_contour_rate": float("nan"),
            "inside_rate": float("nan"),
            "mean_mahalanobis_sq": float("nan"),
            "mean_gap": float("nan"),
            "max_abs_gap": float("nan"),
        }

    mahalanobis_sq = np.asarray([record["mahalanobis_sq"] for record in records], dtype=float)
    mahalanobis_epsilon = np.asarray([record["mahalanobis_epsilon"] for record in records], dtype=float)
    gaps = mahalanobis_epsilon - mahalanobis_sq
    inside = np.asarray(
        [
            is_inside_ellipsoid(
                mahalanobis_sq=float(maha),
                mahalanobis_epsilon=float(epsilon),
                relative_tolerance=float(relative_tolerance),
            )
            for maha, epsilon in zip(mahalanobis_sq, mahalanobis_epsilon)
        ],
        dtype=bool,
    )
    on_contour = np.asarray(
        [
            is_on_ellipsoid_contour(
                mahalanobis_sq=float(maha),
                mahalanobis_epsilon=float(epsilon),
                relative_tolerance=float(relative_tolerance),
            )
            for maha, epsilon in zip(mahalanobis_sq, mahalanobis_epsilon)
        ],
        dtype=bool,
    )
    return {
        "count": int(len(records)),
        "on_contour_rate": float(np.mean(on_contour)),
        "inside_rate": float(np.mean(inside)),
        "mean_mahalanobis_sq": float(np.mean(mahalanobis_sq)),
        "mean_gap": float(np.mean(gaps)),
        "max_abs_gap": float(np.max(np.abs(gaps))),
    }


def main() -> None:
    """Run the 100-episode comparison and save the reward bar plot."""
    n_episodes = 250
    coverages = (0.95, 0.75)
    attack_prob = 0.10
    observation_noise_std = 0.75
    process_position_std = 0.14
    process_wind_std = 0.10
    contour_relative_tolerance = 0.01
    attack_step_size = 0.05
    attack_num_steps = 120
    attack_mc_samples = 16
    real_attack_transition_samples = 8
    heavy_tail_degrees_of_freedom = 3.0
    uniform_annulus_min_coverage = 0.50
    base_seed = 20260727

    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, env_config, _net_config, train_config, _history = load_agent_checkpoint(model_path, device)
    policy.eval()
    gamma = float(train_config.gamma)

    simulation_config = replace(
        env_config,
        observation_noise_std=0.0,
        process_position_std=float(process_position_std),
        process_wind_std=float(process_wind_std),
    )
    filter_config = replace(
        simulation_config,
        observation_noise_std=float(observation_noise_std),
    )
    episode_seeds = base_seed + np.arange(int(n_episodes), dtype=int)

    all_results: dict[float, dict[str, dict[str, np.ndarray]]] = {}
    attack_value_records: list[dict[str, float | int | str]] = []
    mean_returns = np.zeros((len(coverages), len(SETTING_NAMES)), dtype=float)
    sem_returns = np.zeros_like(mean_returns)
    success_rates = np.zeros_like(mean_returns)

    print(
        "Evaluation noise: "
        f"observation_std={observation_noise_std:.3f}, "
        f"process_position_std={process_position_std:.3f}, "
        f"process_wind_std={process_wind_std:.3f}"
    )
    print(
        "Attack settings: "
        f"attack_prob={attack_prob:.3f}, "
        f"pgd_steps={attack_num_steps}, "
        f"posterior_mc={attack_mc_samples}, "
        f"real_transition_mc={real_attack_transition_samples}, "
        f"heavy_tail_df={heavy_tail_degrees_of_freedom:.1f}, "
        f"annulus_min_coverage={uniform_annulus_min_coverage:.2f}, "
        f"gamma={gamma:.4f}"
    )

    for coverage_idx, coverage in enumerate(coverages):
        all_results[float(coverage)] = {}
        mahalanobis_epsilon = coverage_to_mahalanobis_epsilon(float(coverage), obs_dim=4)
        print(
            f"\ncoverage epsilon={coverage:.2f} "
            f"(Mahalanobis radius squared={mahalanobis_epsilon:.4f})"
        )
        for setting_idx, setting in enumerate(SETTING_NAMES):
            result = evaluate_setting(
                policy=policy,
                simulation_config=simulation_config,
                filter_config=filter_config,
                setting=setting,
                coverage=float(coverage),
                episode_seeds=episode_seeds,
                observation_noise_std=float(observation_noise_std),
                gamma=float(gamma),
                attack_prob=float(attack_prob),
                attack_step_size=float(attack_step_size),
                attack_num_steps=int(attack_num_steps),
                attack_mc_samples=int(attack_mc_samples),
                real_attack_transition_samples=int(real_attack_transition_samples),
                heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
                uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
                contour_relative_tolerance=float(contour_relative_tolerance),
                verbose=True,
                device=device,
            )
            all_results[float(coverage)][setting] = result
            attack_value_records.extend(result["attack_value_records"])
            returns = np.asarray(result["returns"], dtype=float)
            successes = np.asarray(result["successes"], dtype=float)
            mean_returns[coverage_idx, setting_idx] = float(np.mean(returns))
            sem_returns[coverage_idx, setting_idx] = float(np.std(returns, ddof=1) / np.sqrt(len(returns)))
            success_rates[coverage_idx, setting_idx] = float(np.mean(successes))
            print(
                f"  {setting:20s} "
                f"mean_return={mean_returns[coverage_idx, setting_idx]:8.3f} "
                f"success={100.0 * success_rates[coverage_idx, setting_idx]:6.2f}% "
                f"mean_attacks={np.mean(result['num_attacks']):5.2f}"
            )
            records = result["attack_value_records"]
            if records:
                noisy_values = np.asarray([record["value_noisy_kf"] for record in records], dtype=float)
                attacked_values = np.asarray([record["value_attacked"] for record in records], dtype=float)
                delta_values = np.asarray([record["delta_value"] for record in records], dtype=float)
                contour_summary = attack_contour_summary(
                    records=records,
                    relative_tolerance=float(contour_relative_tolerance),
                )
                print(
                    f"  {'attack V summary':15s} "
                    f"V_noisyKF={np.mean(noisy_values):8.4f} "
                    f"V_attacked={np.mean(attacked_values):8.4f} "
                    f"delta={np.mean(delta_values):8.4f}"
                )
                print(
                    f"  {'contour summary':15s} "
                    f"inside={100.0 * contour_summary['inside_rate']:6.2f}% "
                    f"on_contour={100.0 * contour_summary['on_contour_rate']:6.2f}% "
                    f"tol={100.0 * contour_relative_tolerance:4.1f}% "
                    f"mean_maha={contour_summary['mean_mahalanobis_sq']:7.3f} "
                    f"mean_gap={contour_summary['mean_gap']:7.3f} "
                    f"max_abs_gap={contour_summary['max_abs_gap']:7.3f}"
                )

    figure_path = os.path.join(rl_figures_dir(), "wind_online_attack_mean_rewards_bar.png")
    plot_mean_return_bars(
        coverages=coverages,
        mean_returns=mean_returns,
        sem_returns=sem_returns,
        outpath=figure_path,
    )

    data_path = os.path.join(rl_data_dir(), "wind_online_attack_mean_rewards_bar.npz")
    attack_record_arrays = attack_records_to_arrays(attack_value_records)
    np.savez_compressed(
        data_path,
        coverages=np.asarray(coverages, dtype=float),
        setting_names=np.asarray(SETTING_NAMES, dtype=object),
        mean_returns=mean_returns,
        sem_returns=sem_returns,
        n_episodes=np.asarray([n_episodes], dtype=int),
        attack_prob=np.asarray([attack_prob], dtype=float),
        attack_step_size=np.asarray([attack_step_size], dtype=float),
        attack_num_steps=np.asarray([attack_num_steps], dtype=int),
        attack_mc_samples=np.asarray([attack_mc_samples], dtype=int),
        real_attack_transition_samples=np.asarray([real_attack_transition_samples], dtype=int),
        heavy_tail_degrees_of_freedom=np.asarray([heavy_tail_degrees_of_freedom], dtype=float),
        uniform_annulus_min_coverage=np.asarray([uniform_annulus_min_coverage], dtype=float),
        gamma=np.asarray([gamma], dtype=float),
        observation_noise_std=np.asarray([observation_noise_std], dtype=float),
        process_position_std=np.asarray([process_position_std], dtype=float),
        process_wind_std=np.asarray([process_wind_std], dtype=float),
        contour_relative_tolerance=np.asarray([contour_relative_tolerance], dtype=float),
        success_rates=success_rates,
        **attack_record_arrays,
    )
    print(f"\nSaved figure to: {figure_path}")
    print(f"Saved data to:   {data_path}")


if __name__ == "__main__":
    main()
