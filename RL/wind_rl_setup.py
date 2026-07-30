#!/usr/bin/env python3
"""
wind_rl_setup.py

Custom RL building blocks for the adversarial-wind navigation benchmark.

Why this file exists:
1. The user asked for a fresh RL setup in `RL/` that does not follow the
   standard Gymnasium environment style.
2. The benchmark is defined by a four-dimensional linear-Gaussian SSM with
   time-varying wind, and we want one shared implementation for:
   - the latent-state dynamics,
   - the custom environment interface,
   - the policy-side observation transformation,
   - the actor-critic network,
   - PPO training utilities,
   - deterministic evaluation rollouts used by the figure scripts.
3. Keeping those pieces in one module lets the training and plotting scripts
   stay small while preserving a single mathematical definition of the task.

Mathematical setup implemented here:
1. Hidden state
      s_t = [p_{x,t}, p_{y,t}, d_{x,t}, d_{y,t}]^T
   contains position and wind.
2. Action
      a_t in [-1, 1]^2
   is the bounded displacement chosen by the policy.
3. Dynamics follow
      s_{t+1} = A_t s_t + B a_t + w_t
   with the time-varying lower-right wind block
      rho_w R(Delta psi_t).
4. Every episode starts from a fixed position and samples a fresh goal from
   an annulus around the origin. The policy therefore receives the hidden
   state together with the current goal, and it performs internally the
   transformation
      z_t = [(g_x - p_{x,t}) / R_max, (g_y - p_{y,t}) / R_max, d_{x,t}, d_{y,t}]
   before the neural-network layers.

Design note:
1. This is not a `gym.Env`.
2. The environment exposes a light custom API with `reset_all`, `reset_done`,
   and `step`.
3. PPO is implemented directly in PyTorch so the actor and critic are fully
   under our control.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import os
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

# Keep direct script execution working, e.g. `python RL/plot_wind_trajectories.py`.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared_ssm import LinearGaussianStateSpaceModel
from shared_ssm import build_attack_geometry
from shared_ssm.attacks import TorchRLAttackConfig
from shared_ssm.attacks import solve_torch_rl_expectation_attack


# ============================================================
# Filesystem helpers
# ============================================================
def ensure_dir(path: str) -> str:
    """Create `path` if it does not exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def rl_module_dir() -> str:
    """Return the absolute directory that contains the RL scripts."""
    return os.path.dirname(os.path.abspath(__file__))


def rl_model_dir() -> str:
    """Return the canonical directory where trained agents are stored."""
    return ensure_dir(os.path.join(rl_module_dir(), "model"))


def rl_figures_dir() -> str:
    """Return the canonical directory where RL figures are stored."""
    return ensure_dir(os.path.join(rl_module_dir(), "figures"))


def rl_data_dir() -> str:
    """Return the canonical directory where RL numerical results are stored."""
    return ensure_dir(os.path.join(rl_module_dir(), "data"))


# ============================================================
# Configuration containers
# ============================================================
@dataclass(frozen=True)
class WindNavigationConfig:
    """Configuration of the four-dimensional wind-navigation SSM."""

    start_xy: tuple[float, float]
    radius_max: float
    start_radius: float
    goal_distance_min: float
    goal_distance_max: float
    goal_radius: float
    max_steps: int
    rho_w: float
    wind_turn_std: float
    process_position_std: float
    process_wind_std: float
    initial_wind_magnitude: float
    action_limit: float
    step_cost: float
    progress_reward_weight: float
    near_goal_progress_bonus_weight: float
    near_goal_progress_power: float
    distance_reward_weight: float
    goal_reward: float
    timeout_penalty: float
    observation_noise_std: float


@dataclass(frozen=True)
class PolicyNetworkConfig:
    """Configuration of the actor-critic neural network."""

    hidden_size: int
    action_std: float


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Configuration of the direct PyTorch PPO trainer."""

    seed: int
    num_envs: int
    rollout_length: int
    total_updates: int
    ppo_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_loss_weight: float
    entropy_weight: float
    learning_rate: float
    max_grad_norm: float
    log_every: int


# ============================================================
# SSM utilities
# ============================================================
def build_transition_matrix(
    *,
    rho_w: float,
    delta_psi: float,
) -> np.ndarray:
    """
    Build the time-varying state transition matrix `A_t`.

    The upper-left block keeps the current position and adds the current wind,
    while the lower-right block rotates and contracts the wind vector.
    """
    cos_psi = float(np.cos(delta_psi))
    sin_psi = float(np.sin(delta_psi))
    return np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, rho_w * cos_psi, -rho_w * sin_psi],
            [0.0, 0.0, rho_w * sin_psi, rho_w * cos_psi],
        ],
        dtype=np.float32,
    )


def build_control_matrix() -> np.ndarray:
    """Return the fixed control matrix `B` for the bounded 2D action."""
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )


def build_observation_matrix() -> np.ndarray:
    """Return the observation matrix `F = I_4`."""
    return np.eye(4, dtype=np.float32)


def build_process_covariance(config: WindNavigationConfig) -> np.ndarray:
    """Return the diagonal process covariance matrix `Q`."""
    pos_var = float(config.process_position_std) ** 2
    wind_var = float(config.process_wind_std) ** 2
    return np.diag([pos_var, pos_var, wind_var, wind_var]).astype(np.float32)


def build_observation_covariance(config: WindNavigationConfig) -> np.ndarray:
    """Return the diagonal observation covariance matrix `R`."""
    obs_var = float(config.observation_noise_std) ** 2
    return np.diag([obs_var, obs_var, obs_var, obs_var]).astype(np.float32)


def default_device() -> torch.device:
    """Pick CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_global_seeds(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible training runs."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def normalize_env_config_dict(raw_config: dict[str, Any]) -> dict[str, Any]:
    """
    Upgrade older saved environment configurations to the current schema.

    Older checkpoints stored `initial_wind_std`. The current environment uses a
    fixed initial wind magnitude with random direction instead.
    """
    normalized = dict(raw_config)
    if "initial_wind_magnitude" not in normalized and "initial_wind_std" in normalized:
        normalized["initial_wind_magnitude"] = float(normalized["initial_wind_std"])
    normalized.pop("initial_wind_std", None)
    normalized.setdefault("near_goal_progress_bonus_weight", 0.0)
    normalized.setdefault("near_goal_progress_power", 1.0)
    return normalized


# ============================================================
# Custom vectorized environment
# ============================================================
class WindNavigationBatch:
    """
    Lightweight batch environment for the wind-navigation SSM.

    This class intentionally avoids the Gymnasium interface. It keeps a batch
    of hidden states, advances them with vectorized NumPy operations, and
    exposes only the pieces PPO needs.
    """

    def __init__(self, config: WindNavigationConfig, num_envs: int, seed: int):
        self.config = config
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(int(seed))
        self.hidden_state = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.goal_xy = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.step_index = np.zeros(self.num_envs, dtype=np.int32)
        self.control_matrix = build_control_matrix()
        self.observation_matrix = build_observation_matrix()
        self.process_cov = build_process_covariance(config)
        self.observation_cov = build_observation_covariance(config)
        self.reset_all()

    def _sample_start_positions(self, count: int) -> np.ndarray:
        """
        Sample start positions from a disc centered at `start_xy`.

        The repeated oversampling keeps the implementation simple while still
        generating a uniform disc sample in expectation.
        """
        center = np.asarray(self.config.start_xy, dtype=np.float32)
        accepted: list[np.ndarray] = []
        remaining = int(count)

        while remaining > 0:
            raw = self.rng.normal(size=(max(remaining * 3, 8), 2)).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            safe_norms = np.maximum(norms, 1e-8)
            directions = raw / safe_norms
            radii = np.sqrt(self.rng.uniform(0.0, 1.0, size=(raw.shape[0], 1))).astype(np.float32)
            points = center + float(self.config.start_radius) * directions * radii
            accepted.append(points[:remaining])
            remaining -= points[:remaining].shape[0]

        return np.vstack(accepted).astype(np.float32)

    def _sample_hidden_state(self, count: int) -> np.ndarray:
        """
        Sample a batch of initial hidden states.

        The initial wind has fixed magnitude and random direction so every
        episode starts with the same wind strength but not the same heading.
        """
        sampled = np.zeros((int(count), 4), dtype=np.float32)
        sampled[:, :2] = self._sample_start_positions(int(count))
        wind_angles = self.rng.uniform(0.0, 2.0 * np.pi, size=int(count)).astype(np.float32)
        sampled[:, 2] = float(self.config.initial_wind_magnitude) * np.cos(wind_angles)
        sampled[:, 3] = float(self.config.initial_wind_magnitude) * np.sin(wind_angles)
        return sampled

    def _sample_goals(self, count: int) -> np.ndarray:
        """
        Sample a batch of goals from the annulus `[r_min, r_max]`.

        The radial variable is sampled uniformly in area so goals do not cluster
        artificially near the inner radius.
        """
        count = int(count)
        goal_angles = self.rng.uniform(0.0, 2.0 * np.pi, size=count).astype(np.float32)
        goal_radius_sq = self.rng.uniform(
            float(self.config.goal_distance_min) ** 2,
            float(self.config.goal_distance_max) ** 2,
            size=count,
        ).astype(np.float32)
        goal_radii = np.sqrt(goal_radius_sq).astype(np.float32)

        goal_xy = np.zeros((count, 2), dtype=np.float32)
        goal_xy[:, 0] = goal_radii * np.cos(goal_angles)
        goal_xy[:, 1] = goal_radii * np.sin(goal_angles)
        return goal_xy

    def _make_observation(self, hidden_state: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        """
        Return the observation seen by the policy.

        Observation noise is kept configurable for completeness, but the
        training scripts set it to zero so the policy receives the hidden state
        exactly. The goal is concatenated without noise because it is task
        context rather than a latent variable to estimate.
        """
        hidden_state = np.asarray(hidden_state, dtype=np.float32)
        goal_xy = np.asarray(goal_xy, dtype=np.float32)
        if float(self.config.observation_noise_std) <= 0.0:
            return np.concatenate([hidden_state.copy(), goal_xy.copy()], axis=1).astype(np.float32)

        noise = self.rng.normal(
            loc=0.0,
            scale=float(self.config.observation_noise_std),
            size=hidden_state.shape,
        ).astype(np.float32)
        noisy_hidden = (hidden_state @ self.observation_matrix.T + noise).astype(np.float32)
        return np.concatenate([noisy_hidden, goal_xy.copy()], axis=1).astype(np.float32)

    def current_observation(self) -> np.ndarray:
        """Return the current state-goal observation for the whole batch."""
        return self._make_observation(self.hidden_state, self.goal_xy)

    def reset_all(self) -> np.ndarray:
        """Reset the full batch and return the corresponding observations."""
        self.hidden_state = self._sample_hidden_state(self.num_envs)
        self.goal_xy = self._sample_goals(self.num_envs)
        self.step_index = np.zeros(self.num_envs, dtype=np.int32)
        return self.current_observation()

    def reset_done(self, done_mask: np.ndarray) -> np.ndarray:
        """Reset only finished environments and return the fresh observations."""
        done_mask = np.asarray(done_mask, dtype=bool).reshape(self.num_envs)
        num_done = int(done_mask.sum())
        if num_done == 0:
            return self.current_observation()

        self.hidden_state[done_mask] = self._sample_hidden_state(num_done)
        self.goal_xy[done_mask] = self._sample_goals(num_done)
        self.step_index[done_mask] = 0
        return self.current_observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Advance the whole batch by one step.

        Returns:
        1. next observation,
        2. reward,
        3. done flag,
        4. diagnostics including success flags and goal distances.
        """
        action = np.asarray(action, dtype=np.float32).reshape(self.num_envs, 2)
        bounded_action = np.clip(action, -float(self.config.action_limit), float(self.config.action_limit))
        delta_psi = self.rng.normal(
            loc=0.0,
            scale=float(self.config.wind_turn_std),
            size=self.num_envs,
        ).astype(np.float32)
        wind = self.hidden_state[:, 2:].copy()

        cos_psi = np.cos(delta_psi).astype(np.float32)
        sin_psi = np.sin(delta_psi).astype(np.float32)
        rotated_wind = np.zeros_like(wind)
        rotated_wind[:, 0] = float(self.config.rho_w) * (cos_psi * wind[:, 0] - sin_psi * wind[:, 1])
        rotated_wind[:, 1] = float(self.config.rho_w) * (sin_psi * wind[:, 0] + cos_psi * wind[:, 1])

        process_noise = self.rng.multivariate_normal(
            mean=np.zeros(4, dtype=np.float32),
            cov=self.process_cov,
            size=self.num_envs,
        ).astype(np.float32)

        next_state = np.zeros_like(self.hidden_state)
        next_state[:, :2] = self.hidden_state[:, :2] + wind + bounded_action + process_noise[:, :2]
        next_state[:, 2:] = rotated_wind + process_noise[:, 2:]

        goal_distance = np.linalg.norm(next_state[:, :2] - self.goal_xy, axis=1)

        self.step_index = self.step_index + 1
        reached_goal = goal_distance <= float(self.config.goal_radius)
        reached_horizon = self.step_index >= int(self.config.max_steps)
        timeout_only = np.logical_and(reached_horizon, np.logical_not(reached_goal))
        done = np.logical_or(reached_goal, reached_horizon)

        normalized_distance = goal_distance / max(float(self.config.radius_max), 1e-6)
        distance_reward = -normalized_distance.astype(np.float32)
        progress_reward = np.zeros(self.num_envs, dtype=np.float32)
        progress_multiplier = np.ones(self.num_envs, dtype=np.float32)
        reward = distance_reward.copy()
        reward[reached_goal] = reward[reached_goal] + float(self.config.goal_reward)
        reward[timeout_only] = reward[timeout_only] + float(self.config.timeout_penalty)

        self.hidden_state = next_state.astype(np.float32)
        next_obs = self.current_observation()
        info = {
            "reached_goal": reached_goal.astype(bool),
            "timed_out": timeout_only.astype(bool),
            "goal_distance": goal_distance.astype(np.float32),
            "progress_reward": progress_reward.astype(np.float32),
            "progress_multiplier": progress_multiplier.astype(np.float32),
            "distance_reward": distance_reward.astype(np.float32),
            "delta_psi": delta_psi.astype(np.float32),
            "goal_xy": self.goal_xy.copy(),
        }
        return next_obs, reward, done.astype(bool), info


# ============================================================
# Policy transform and actor-critic network
# ============================================================
class GoalRelativeObservation(nn.Module):
    """
    Policy-side observation transform from hidden state to RL features.

    The environment returns the hidden state together with the current goal,
    but both actor and critic operate on the transformed vector `z_t`. This is
    exactly the user-requested design: the transformation lives inside the
    policy rather than in the environment.
    """

    def __init__(self, radius_max: float):
        super().__init__()
        radius_tensor = torch.tensor(float(radius_max), dtype=torch.float32)
        self.register_buffer("radius_max", radius_tensor)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Map `[p_x, p_y, d_x, d_y, g_x, g_y]` to the transformed policy input."""
        position = observation[..., :2]
        wind = observation[..., 2:4]
        goal_xy = observation[..., 4:6]
        relative_goal = (goal_xy - position) / self.radius_max
        return torch.cat([relative_goal, wind], dim=-1)


class SquashedGaussianActorCritic(nn.Module):
    """Actor-critic network with a fixed-variance squashed Gaussian actor."""

    def __init__(self, env_config: WindNavigationConfig, net_config: PolicyNetworkConfig):
        super().__init__()
        self.transform = GoalRelativeObservation(env_config.radius_max)
        self.action_std = float(net_config.action_std)

        self.shared_body = nn.Sequential(
            nn.Linear(4, int(net_config.hidden_size)),
            nn.Tanh(),
            nn.Linear(int(net_config.hidden_size), int(net_config.hidden_size)),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(int(net_config.hidden_size), 2)
        self.critic_head = nn.Linear(int(net_config.hidden_size), 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Use orthogonal initialization for stable PPO training."""
        for module in self.shared_body:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Transform the state-goal observation and pass it through the shared MLP."""
        transformed = self.transform(observation)
        return self.shared_body(transformed)

    def _base_distribution(self, observation: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Return the Gaussian in the unsquashed action space."""
        latent = self.encode(observation)
        mean = self.actor_head(latent)
        std = torch.full_like(mean, self.action_std)
        return Normal(mean, std), latent

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the scalar critic estimate `V(z_t)`."""
        latent = self.encode(observation)
        return self.critic_head(latent).squeeze(-1)

    def sample_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a squashed Gaussian action and return action, log-prob, value."""
        base_dist, latent = self._base_distribution(observation)
        unsquashed_action = base_dist.rsample()
        action = torch.tanh(unsquashed_action)
        log_prob = self._tanh_log_prob(base_dist, unsquashed_action, action)
        value = self.critic_head(latent).squeeze(-1)
        return action, log_prob, value

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the deterministic mean action used during evaluation."""
        base_dist, _latent = self._base_distribution(observation)
        return torch.tanh(base_dist.mean)

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities and values for already-sampled actions."""
        clipped_action = torch.clamp(action, -0.999999, 0.999999)
        unsquashed_action = torch.atanh(clipped_action)
        base_dist, latent = self._base_distribution(observation)
        log_prob = self._tanh_log_prob(base_dist, unsquashed_action, clipped_action)
        value = self.critic_head(latent).squeeze(-1)
        entropy = base_dist.entropy().sum(dim=-1)
        return log_prob, value, entropy

    @staticmethod
    def _tanh_log_prob(
        base_dist: Normal,
        unsquashed_action: torch.Tensor,
        squashed_action: torch.Tensor,
    ) -> torch.Tensor:
        """Return the corrected log-density after the `tanh` squash."""
        log_prob_u = base_dist.log_prob(unsquashed_action).sum(dim=-1)
        squash_correction = torch.log(1.0 - squashed_action.square() + 1e-6).sum(dim=-1)
        return log_prob_u - squash_correction


# ============================================================
# PPO rollout storage
# ============================================================
@dataclass
class RolloutBatch:
    """Container for one PPO rollout collected from the batch environment."""

    observation: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    value: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def make_empty_history() -> dict[str, list[float]]:
    """Return the standard training-history container used by PPO."""
    return {
        "mean_return": [],
        "mean_length": [],
        "success_rate": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }


def compute_gae(
    *,
    reward: torch.Tensor,
    done: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates and bootstrap returns."""
    rollout_length, num_envs = reward.shape
    advantages = torch.zeros((rollout_length, num_envs), device=reward.device, dtype=torch.float32)
    gae = torch.zeros(num_envs, device=reward.device, dtype=torch.float32)

    for step in reversed(range(rollout_length)):
        if step == rollout_length - 1:
            next_val = next_value
        else:
            next_val = value[step + 1]

        non_terminal = 1.0 - done[step]
        delta = reward[step] + float(gamma) * next_val * non_terminal - value[step]
        gae = delta + float(gamma) * float(gae_lambda) * non_terminal * gae
        advantages[step] = gae

    returns = advantages + value
    return returns, advantages


# ============================================================
# Training and evaluation helpers
# ============================================================
def collect_rollout(
    *,
    env: WindNavigationBatch,
    policy: SquashedGaussianActorCritic,
    rollout_length: int,
    device: torch.device,
    episode_return_tracker: np.ndarray,
    episode_length_tracker: np.ndarray,
    completed_returns: list[float],
    completed_lengths: list[int],
    completed_successes: list[float],
) -> tuple[RolloutBatch, torch.Tensor]:
    """
    Collect one PPO rollout and update episode-level logging lists.

    Finished episodes are reset immediately so a fixed-size batch stays active
    throughout the rollout.
    """
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    current_obs = torch.as_tensor(env.current_observation(), dtype=torch.float32, device=device)

    for _ in range(int(rollout_length)):
        with torch.no_grad():
            sampled_action, sampled_log_prob, sampled_value = policy.sample_action(current_obs)

        next_obs_np, reward_np, done_np, info = env.step(sampled_action.cpu().numpy())

        observations.append(current_obs)
        actions.append(sampled_action)
        log_probs.append(sampled_log_prob)
        rewards.append(torch.as_tensor(reward_np, dtype=torch.float32, device=device))
        dones.append(torch.as_tensor(done_np.astype(np.float32), dtype=torch.float32, device=device))
        values.append(sampled_value)

        episode_return_tracker += reward_np
        episode_length_tracker += 1

        if np.any(done_np):
            done_mask = np.asarray(done_np, dtype=bool)
            completed_returns.extend(episode_return_tracker[done_mask].tolist())
            completed_lengths.extend(episode_length_tracker[done_mask].astype(int).tolist())
            completed_successes.extend(info["reached_goal"][done_mask].astype(np.float32).tolist())

            episode_return_tracker[done_mask] = 0.0
            episode_length_tracker[done_mask] = 0
            reset_obs_np = env.reset_done(done_mask)
            next_obs_np[done_mask] = reset_obs_np[done_mask]

        current_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        next_value = policy.value(current_obs)

    reward_tensor = torch.stack(rewards)
    done_tensor = torch.stack(dones)
    value_tensor = torch.stack(values)
    return_tensor, advantage_tensor = compute_gae(
        reward=reward_tensor,
        done=done_tensor,
        value=value_tensor,
        next_value=next_value,
        gamma=0.995,
        gae_lambda=0.97,
    )

    rollout = RolloutBatch(
        observation=torch.stack(observations),
        action=torch.stack(actions),
        log_prob=torch.stack(log_probs),
        reward=reward_tensor,
        done=done_tensor,
        value=value_tensor,
        returns=return_tensor,
        advantages=advantage_tensor,
    )
    return rollout, current_obs


def ppo_update(
    *,
    policy: SquashedGaussianActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    train_config: PPOTrainingConfig,
) -> dict[str, float]:
    """Run the PPO minibatch update over one rollout."""
    rollout_length, num_envs = rollout.reward.shape
    batch_size = rollout_length * num_envs

    flat_observation = rollout.observation.reshape(batch_size, 6)
    flat_action = rollout.action.reshape(batch_size, 2)
    flat_log_prob = rollout.log_prob.reshape(batch_size)
    flat_return = rollout.returns.reshape(batch_size)
    flat_advantage = rollout.advantages.reshape(batch_size)

    flat_advantage = (flat_advantage - flat_advantage.mean()) / (flat_advantage.std(unbiased=False) + 1e-8)

    policy_loss_total = 0.0
    value_loss_total = 0.0
    entropy_total = 0.0
    num_updates = 0

    for _ in range(int(train_config.ppo_epochs)):
        permutation = torch.randperm(batch_size, device=flat_observation.device)
        for start in range(0, batch_size, int(train_config.minibatch_size)):
            stop = min(start + int(train_config.minibatch_size), batch_size)
            idx = permutation[start:stop]

            new_log_prob, new_value, entropy = policy.evaluate_actions(flat_observation[idx], flat_action[idx])
            ratio = torch.exp(new_log_prob - flat_log_prob[idx])

            unclipped_obj = ratio * flat_advantage[idx]
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - float(train_config.clip_ratio),
                1.0 + float(train_config.clip_ratio),
            )
            clipped_obj = clipped_ratio * flat_advantage[idx]
            policy_loss = -torch.min(unclipped_obj, clipped_obj).mean()
            value_loss = 0.5 * (flat_return[idx] - new_value).square().mean()
            entropy_bonus = entropy.mean()

            loss = (
                policy_loss
                + float(train_config.value_loss_weight) * value_loss
                - float(train_config.entropy_weight) * entropy_bonus
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), float(train_config.max_grad_norm))
            optimizer.step()

            policy_loss_total += float(policy_loss.item())
            value_loss_total += float(value_loss.item())
            entropy_total += float(entropy_bonus.item())
            num_updates += 1

    scale = max(num_updates, 1)
    return {
        "policy_loss": policy_loss_total / scale,
        "value_loss": value_loss_total / scale,
        "entropy": entropy_total / scale,
    }


def train_ppo_agent(
    *,
    env_config: WindNavigationConfig,
    net_config: PolicyNetworkConfig,
    train_config: PPOTrainingConfig,
    device: torch.device,
    initial_policy: SquashedGaussianActorCritic | None = None,
    initial_optimizer_state_dict: dict[str, Any] | None = None,
    initial_history: dict[str, list[float]] | None = None,
    completed_updates: int = 0,
) -> tuple[SquashedGaussianActorCritic, dict[str, list[float]], dict[str, Any], int]:
    """
    Train the wind-navigation policy with direct PyTorch PPO.

    The returned history is intentionally compact so the training script can
    print or save it without additional processing.
    """
    seed_offset = int(completed_updates) * 1000
    effective_seed = int(train_config.seed) + seed_offset
    set_global_seeds(effective_seed)

    env = WindNavigationBatch(env_config, int(train_config.num_envs), effective_seed)
    policy = initial_policy if initial_policy is not None else SquashedGaussianActorCritic(env_config, net_config).to(device)
    policy = policy.to(device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train_config.learning_rate))
    if initial_optimizer_state_dict is not None:
        optimizer.load_state_dict(initial_optimizer_state_dict)

    episode_return_tracker = np.zeros(int(train_config.num_envs), dtype=np.float32)
    episode_length_tracker = np.zeros(int(train_config.num_envs), dtype=np.int32)
    history = make_empty_history() if initial_history is None else {key: list(values) for key, values in initial_history.items()}

    for update_idx in range(1, int(train_config.total_updates) + 1):
        absolute_update_idx = int(completed_updates) + update_idx
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        completed_successes: list[float] = []

        rollout, _next_obs = collect_rollout(
            env=env,
            policy=policy,
            rollout_length=int(train_config.rollout_length),
            device=device,
            episode_return_tracker=episode_return_tracker,
            episode_length_tracker=episode_length_tracker,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
            completed_successes=completed_successes,
        )

        # Replace the default GAE hyperparameters with the user-configurable ones.
        with torch.no_grad():
            bootstrap_value = policy.value(
                torch.as_tensor(
                    env.current_observation(),
                    dtype=torch.float32,
                    device=device,
                )
            )

        rollout.returns, rollout.advantages = compute_gae(
            reward=rollout.reward,
            done=rollout.done,
            value=rollout.value,
            next_value=bootstrap_value,
            gamma=float(train_config.gamma),
            gae_lambda=float(train_config.gae_lambda),
        )
        rollout.returns = rollout.returns.detach()
        rollout.advantages = rollout.advantages.detach()

        losses = ppo_update(
            policy=policy,
            optimizer=optimizer,
            rollout=rollout,
            train_config=train_config,
        )

        mean_return = float(np.mean(completed_returns)) if completed_returns else float("nan")
        mean_length = float(np.mean(completed_lengths)) if completed_lengths else float("nan")
        success_rate = float(np.mean(completed_successes)) if completed_successes else float("nan")

        history["mean_return"].append(mean_return)
        history["mean_length"].append(mean_length)
        history["success_rate"].append(success_rate)
        history["policy_loss"].append(float(losses["policy_loss"]))
        history["value_loss"].append(float(losses["value_loss"]))
        history["entropy"].append(float(losses["entropy"]))

        if (
            update_idx == 1
            or update_idx % int(train_config.log_every) == 0
            or update_idx == int(train_config.total_updates)
        ):
            print(
                f"[update {absolute_update_idx:03d} | local {update_idx:03d}/{train_config.total_updates:03d}] "
                f"return={mean_return:7.3f} "
                f"length={mean_length:6.2f} "
                f"success={success_rate:5.2%} "
                f"pi_loss={losses['policy_loss']:8.4f} "
                f"v_loss={losses['value_loss']:8.4f}"
            )

    optimizer_state_dict = optimizer.state_dict()
    total_completed_updates = int(completed_updates) + int(train_config.total_updates)
    return policy, history, optimizer_state_dict, total_completed_updates


def evaluate_policy(
    *,
    policy: SquashedGaussianActorCritic,
    env_config: WindNavigationConfig,
    seed: int | None,
    device: torch.device,
    attack_prob: float = 0.0,
    attack_epsilon: float = 1.0,
    attack_step_size: float = 0.05,
    attack_num_steps: int = 20,
    attack_mc_samples: int = 16,
) -> dict[str, Any]:
    """
    Run one deterministic episode and return the full trajectory.

    The stored arrays are later used by the trajectory figure script.
    """
    if not (0.0 <= float(attack_prob) <= 1.0):
        raise ValueError("attack_prob must lie in [0, 1].")

    realized_seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    env = WindNavigationBatch(env_config, num_envs=1, seed=realized_seed)
    attack_rng = np.random.default_rng(realized_seed + 100_003)
    obs = env.reset_all()
    initial_hidden_state = env.hidden_state[0].copy()

    positions = [env.hidden_state[0, :2].copy()]
    winds = [env.hidden_state[0, 2:].copy()]
    actions = []
    rewards = []
    clean_observations = [obs[0].copy()]
    policy_observations = []
    attacked_observations = []
    attack_flags = []
    filter_observations: list[np.ndarray] = []
    filter_actions: list[np.ndarray] = []
    filter_transitions: list[np.ndarray] = []

    done = False
    reached_goal = False
    timed_out = False

    while not done:
        obs_for_policy = obs.copy()
        if (
            float(attack_prob) > 0.0
            and filter_observations
            and attack_rng.random() < float(attack_prob)
        ):
            attacked_state_obs = _build_wind_policy_attack_observation(
                policy=policy,
                env_config=env_config,
                clean_state_observation=obs[0, :4],
                goal_xy=env.goal_xy[0],
                initial_hidden_state=initial_hidden_state,
                filter_observations=filter_observations,
                filter_actions=filter_actions,
                filter_transitions=filter_transitions,
                attack_epsilon=attack_epsilon,
                attack_step_size=attack_step_size,
                attack_num_steps=attack_num_steps,
                attack_mc_samples=attack_mc_samples,
                seed=realized_seed + len(actions) + 1_000,
                device=device,
            )
            obs_for_policy[0, :4] = attacked_state_obs.astype(np.float32)
            attack_flags.append(True)
        else:
            attack_flags.append(False)

        obs_tensor = torch.as_tensor(obs_for_policy, dtype=torch.float32, device=device)
        policy_observations.append(obs_for_policy[0].copy())
        with torch.no_grad():
            action = policy.deterministic_action(obs_tensor).cpu().numpy()[0]

        next_obs, reward, done_mask, info = env.step(action[None, :])
        delta_psi = float(info["delta_psi"][0])
        filter_observations.append(next_obs[0, :4].copy())
        filter_actions.append(action.copy())
        filter_transitions.append(
            build_transition_matrix(
                rho_w=float(env_config.rho_w),
                delta_psi=delta_psi,
            )
        )
        obs = next_obs

        positions.append(env.hidden_state[0, :2].copy())
        winds.append(env.hidden_state[0, 2:].copy())
        actions.append(action.copy())
        rewards.append(float(reward[0]))
        clean_observations.append(obs[0].copy())
        attacked_observations.append(obs_for_policy[0, :4].copy())
        done = bool(done_mask[0])
        reached_goal = bool(info["reached_goal"][0])
        timed_out = bool(info["timed_out"][0])

    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "winds": np.asarray(winds, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "clean_observations": np.asarray(clean_observations, dtype=np.float32),
        "policy_observations": np.asarray(policy_observations, dtype=np.float32),
        "attacked_state_observations": np.asarray(attacked_observations, dtype=np.float32),
        "attack_flags": np.asarray(attack_flags, dtype=bool),
        "goal_xy": env.goal_xy[0].copy(),
        "return": float(np.sum(rewards)),
        "length": int(len(actions)),
        "reached_goal": reached_goal,
        "timed_out": timed_out,
        "seed": realized_seed,
    }


def _build_wind_policy_attack_observation(
    *,
    policy: SquashedGaussianActorCritic,
    env_config: WindNavigationConfig,
    clean_state_observation: np.ndarray,
    goal_xy: np.ndarray,
    initial_hidden_state: np.ndarray,
    filter_observations: list[np.ndarray],
    filter_actions: list[np.ndarray],
    filter_transitions: list[np.ndarray],
    attack_epsilon: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """
    Build one online adversarial state observation for the wind policy.

    The objective minimizes the policy critic over posterior state samples,
    which is the natural online RL analogue of the nonlinear `E[g(s_t)]`
    attacks.
    """
    observations = np.asarray(filter_observations, dtype=float)
    actions = np.asarray(filter_actions, dtype=float)
    transitions = np.asarray(filter_transitions, dtype=float)
    if observations.ndim != 2 or observations.shape[0] == 0:
        return np.asarray(clean_state_observation, dtype=np.float32)

    model = LinearGaussianStateSpaceModel(
        A=transitions,
        B=build_control_matrix(),
        F=build_observation_matrix(),
        G=np.zeros((4, 2), dtype=np.float32),
        W=build_process_covariance(env_config),
        V=build_observation_covariance(env_config),
        m0=np.asarray(initial_hidden_state, dtype=float),
        P0=1e-4 * np.eye(4, dtype=float),
    )
    observation_index = observations.shape[0] - 1
    geometry = build_attack_geometry(
        model=model,
        observations=observations,
        actions=actions,
        observation_index=observation_index,
        epsilon=float(attack_epsilon),
        mode="online",
    )
    goal_t = torch.tensor(np.asarray(goal_xy, dtype=np.float32), dtype=torch.float32, device=device)

    def critic_objective(samples: torch.Tensor) -> torch.Tensor:
        goal_batch = goal_t.unsqueeze(0).expand(samples.shape[0], -1)
        policy_input = torch.cat([samples, goal_batch], dim=-1)
        return policy.value(policy_input)

    attack = solve_torch_rl_expectation_attack(
        geometry=geometry,
        clean_observation=clean_state_observation,
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
    return np.asarray(attack.adversarial_observation, dtype=np.float32)


# ============================================================
# Persistence helpers
# ============================================================
def save_agent_checkpoint(
    *,
    path: str,
    policy: SquashedGaussianActorCritic,
    env_config: WindNavigationConfig,
    net_config: PolicyNetworkConfig,
    train_config: PPOTrainingConfig,
    history: dict[str, list[float]],
    optimizer_state_dict: dict[str, Any] | None = None,
    completed_updates: int = 0,
) -> None:
    """Save the trained actor-critic and all reproducibility metadata."""
    payload = {
        "policy_state_dict": policy.state_dict(),
        "env_config": asdict(env_config),
        "net_config": asdict(net_config),
        "train_config": asdict(train_config),
        "history": history,
        "optimizer_state_dict": optimizer_state_dict,
        "completed_updates": int(completed_updates),
    }
    torch.save(payload, path)


def load_agent_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[SquashedGaussianActorCritic, WindNavigationConfig, PolicyNetworkConfig, PPOTrainingConfig, dict[str, list[float]]]:
    """Load a saved actor-critic checkpoint from disk."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    env_config = WindNavigationConfig(**normalize_env_config_dict(checkpoint["env_config"]))
    net_config = PolicyNetworkConfig(**checkpoint["net_config"])
    train_config = PPOTrainingConfig(**checkpoint["train_config"])

    policy = SquashedGaussianActorCritic(env_config, net_config).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    history = checkpoint.get("history", make_empty_history())
    return policy, env_config, net_config, train_config, history


def load_training_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    """
    Load a training checkpoint with optimizer state and resume metadata.

    This helper is separate from `load_agent_checkpoint` so the plotting
    scripts can keep using the lightweight interface.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    env_config = WindNavigationConfig(**normalize_env_config_dict(checkpoint["env_config"]))
    net_config = PolicyNetworkConfig(**checkpoint["net_config"])
    train_config = PPOTrainingConfig(**checkpoint["train_config"])

    policy = SquashedGaussianActorCritic(env_config, net_config).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.train()

    return {
        "policy": policy,
        "env_config": env_config,
        "net_config": net_config,
        "train_config": train_config,
        "history": checkpoint.get("history", make_empty_history()),
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
        "completed_updates": int(checkpoint.get("completed_updates", 0)),
    }
