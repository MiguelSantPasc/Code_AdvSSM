#!/usr/bin/env python3
"""
Train and visualize the wind-driven 2D point-agent policy.

This file implements the RL environment as the same 4D linear-Gaussian
state-space model later reused by the attack and filtering benchmarks. The
latent state is

    s_t = [p_x,t, p_y,t, d_x,t, d_y,t]^T,

where the first two coordinates are the true position and the last two are the
current wind-displacement vector. The stochastic dynamics are

    s_{t+1} = A_t s_t + B a_t + w_t,
    o_t     = F s_t + v_t,

with F = I_4. The position is then converted into the policy input

    z_t = [ (goal - p_t) / goal_r_max, d_x,t, d_y,t ].

Using the same physical state in both the environment and the defended filter
keeps the RL benchmark aligned with the mathematical model described in the
project notes.
"""

# AdvRL_wind.py
# RL experiment: 2D point agent with WIND dynamics formalized as an SSM.
#
# SSM-inspired model (shared physical state):
#   s_t = [p_x, p_y, d_x, d_y]^T in R^4
#
#   s_{t+1} = A_t s_t + B a_t + w_t
#   o_t     = F s_t + v_t
#
# where:
#   A_t = [[1, 0, 1, 0],
#          [0, 1, 0, 1],
#          [0, 0, rho_w*cos(delta_psi_t), -rho_w*sin(delta_psi_t)],
#          [0, 0, rho_w*sin(delta_psi_t),  rho_w*cos(delta_psi_t)]]
#
#   B   = [[1, 0],
#          [0, 1],
#          [0, 0],
#          [0, 0]]
#
#   a_t = [a_x, a_y] in [-1,1]^2
#
# RL observation returned to policy (4D):
#   z_t = [ (goal - o_{p,t})/goal_r_max, o_{d_x,t}, o_{d_y,t} ]
#
# Rewards:
#   -1 per step until done
#   + terminal reward on success
#   + terminal penalty on timeout
#
# Saves model to:
#   RL/v2_wind_4dattack/outputs/saved_models/AdvRL_v3_shared4d_clean_policy.pt

from __future__ import annotations

import sys
import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


MODEL_FILENAME = "AdvRL_v3_shared4d_clean_policy.pt"
LEGACY_MODEL_FILENAMES = (
    "AdvRL_v2_policy.pt",
    "AdvRL_v2_nowind_policy.pt",
)


def default_policy_model_path() -> str:
    """Return the checkpoint path for the shared-SSM clean policy."""
    return os.path.join(_THIS_DIR, "outputs", "saved_models", MODEL_FILENAME)


def warm_start_policy_model_path() -> str | None:
    """
    Return the best available checkpoint to continue training from.

    Preference order:
    1. the current shared-SSM checkpoint if it already exists,
    2. otherwise the older wind-policy checkpoint,
    3. otherwise the older no-wind checkpoint.
    """
    candidate_paths = [default_policy_model_path()]
    candidate_paths.extend(
        os.path.join(_THIS_DIR, "outputs", "saved_models", filename)
        for filename in LEGACY_MODEL_FILENAMES
    )

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return None


# -----------------------------
# Environment (SSM-style with Wind)
# -----------------------------

@dataclass
class AdvRLEnvConfig:
    goal_r_min: float = 5.0
    goal_r_max: float = 15.0

    # Legacy scalar noise knobs kept so older scripts can still tweak the
    # environment without needing the full covariance parameterization.
    obs_noise_std: float = 0.00
    proc_noise_std: float = 0.00

    max_steps: int = 40
    goal_radius: float = 0.35
    seed: int = 0

    # Wind parameters of the shared 4D SSM.
    wind_epsilon: float = 0.85
    wind_volatility: float = 0.25
    wind_persistence: float = 0.92

    # Optional per-block process-noise overrides.
    position_process_std: float | None = None
    wind_process_std: float | None = None
    position_process_corr: float = 0.0
    wind_process_corr: float = 0.0
    process_cross_corr: float = 0.10

    # Observation-noise parameters for the 4D measurement `o_t = s_t + v_t`.
    obs_position_std: float | None = None
    obs_wind_std: float | None = None
    obs_position_corr: float = 0.20
    obs_wind_corr: float = 0.20
    obs_cross_corr: float = 0.08

    # Rewards (sparse)
    step_penalty: float = -1.0
    success_reward: float = 5.0
    timeout_penalty: float = -40.0


class AdvRL2DEnv:
    """
    Environment backed by the shared 4D physical SSM.

    Hidden state:
        s_t = [p_x,t, p_y,t, d_x,t, d_y,t]^T

    Dynamics:
        s_{t+1} = A_t s_t + B a_t + w_t

    Observation:
        o_t = F s_t + v_t,  with F = I_4

    Policy input:
        z_t = [(g_x - o_{p_x,t})/R_max,
               (g_y - o_{p_y,t})/R_max,
               o_{d_x,t},
               o_{d_y,t}]^T
    """

    def __init__(self, cfg: AdvRLEnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.F = np.eye(4, dtype=np.float32)
        self.B = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.reset()

    @staticmethod
    def _wrap_angle_pi(x: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _project_to_psd(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Return a numerically safe PSD approximation of `M`."""
        sym = 0.5 * (M + M.T)
        eigvals, eigvecs = np.linalg.eigh(np.asarray(sym, dtype=np.float64))
        eigvals = np.maximum(eigvals, eps)
        return (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(np.float32)

    @staticmethod
    def _corr_2x2(std_value: float, corr: float) -> np.ndarray:
        """Return a 2x2 covariance block with shared marginal std."""
        std_value = float(max(std_value, 0.0))
        corr = float(np.clip(corr, -0.95, 0.95))
        return np.array(
            [
                [std_value**2, corr * std_value**2],
                [corr * std_value**2, std_value**2],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _cross_block_2x2(std_left: float, std_right: float, corr: float) -> np.ndarray:
        """
        Return a dense 2x2 cross-covariance block.

        Using a dense block instead of a diagonal-only coupling means the
        position and wind coordinates are correlated in every direction of the
        4D state, as requested by the shared RL covariance-adaptation setup.
        """
        std_left = float(max(std_left, 0.0))
        std_right = float(max(std_right, 0.0))
        corr = float(np.clip(corr, -0.95, 0.95))
        return np.full(
            (2, 2),
            corr * std_left * std_right,
            dtype=np.float32,
        )

    def _position_process_std(self) -> float:
        value = self.cfg.position_process_std
        return float(self.cfg.proc_noise_std if value is None else value)

    def _wind_process_std(self) -> float:
        value = self.cfg.wind_process_std
        return float(self.cfg.proc_noise_std if value is None else value)

    def _obs_position_std(self) -> float:
        value = self.cfg.obs_position_std
        return float(self.cfg.obs_noise_std if value is None else value)

    def _obs_wind_std(self) -> float:
        value = self.cfg.obs_wind_std
        return float(self.cfg.obs_noise_std if value is None else value)

    def process_covariance(self) -> np.ndarray:
        """Return the 4D process covariance used by the environment SSM."""
        pos_std = self._position_process_std()
        wind_std = self._wind_process_std()
        pos_block = self._corr_2x2(
            pos_std,
            self.cfg.position_process_corr,
        )
        wind_block = self._corr_2x2(
            wind_std,
            self.cfg.wind_process_corr,
        )
        cross_block = self._cross_block_2x2(
            pos_std,
            wind_std,
            self.cfg.process_cross_corr,
        )
        cov = np.block(
            [
                [pos_block, cross_block],
                [cross_block.T, wind_block],
            ]
        )
        return self._project_to_psd(cov)

    def observation_covariance(self) -> np.ndarray:
        """Return the correlated 4D observation covariance used in `o_t = s_t + v_t`."""
        pos_std = self._obs_position_std()
        wind_std = self._obs_wind_std()
        pos_block = self._corr_2x2(pos_std, self.cfg.obs_position_corr)
        wind_block = self._corr_2x2(wind_std, self.cfg.obs_wind_corr)
        cross_block = self._cross_block_2x2(
            pos_std,
            wind_std,
            self.cfg.obs_cross_corr,
        )
        cov = np.block(
            [
                [pos_block, cross_block],
                [cross_block.T, wind_block],
            ]
        )
        return self._project_to_psd(cov)

    def build_transition_matrix(self, delta_psi: float) -> np.ndarray:
        """Return the shared physical transition matrix for one time step."""
        rho_w = float(self.cfg.wind_persistence)
        c = math.cos(delta_psi)
        s = math.sin(delta_psi)
        return np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, rho_w * c, -rho_w * s],
                [0.0, 0.0, rho_w * s, rho_w * c],
            ],
            dtype=np.float32,
        )

    def _wind_vec(self) -> np.ndarray:
        """Return the current true wind vector stored in the latent state."""
        return self.x_ssm[2:4].astype(np.float32).copy()

    def _sample_observation_noise(self) -> np.ndarray:
        """Sample the 4D observation noise using the configured covariance."""
        cov = self.observation_covariance().astype(np.float64)
        if float(np.max(np.abs(cov))) <= 1e-12:
            return np.zeros(4, dtype=np.float32)
        obs_rng = getattr(self, "rng_obs", self.rng)
        return obs_rng.multivariate_normal(
            mean=np.zeros(4, dtype=np.float64),
            cov=cov,
        ).astype(np.float32)

    def _sample_process_noise(self) -> np.ndarray:
        """Sample the 4D process noise using the configured covariance."""
        cov = self.process_covariance().astype(np.float64)
        if float(np.max(np.abs(cov))) <= 1e-12:
            return np.zeros(4, dtype=np.float32)
        return self.rng.multivariate_normal(
            mean=np.zeros(4, dtype=np.float64),
            cov=cov,
        ).astype(np.float32)

    def _measure_observation_state(self) -> np.ndarray:
        """Return the noisy 4D observation `o_t = s_t + v_t`."""
        y = (self.F @ self.x_ssm).astype(np.float32)
        return (y + self._sample_observation_noise()).astype(np.float32)

    def _measure_position(self) -> np.ndarray:
        """
        Compatibility helper returning only the measured position block.

        Older scripts in the repo override `_measure_position`; keeping this
        method avoids hard failures while the benchmark migrates to the shared
        4D observation model.
        """
        return self._measure_observation_state()[:2].astype(np.float32)

    def _build_rl_obs(self) -> np.ndarray:
        """
        Build the policy input from the noisy 4D observation state.
        """
        y_pos = self.y_meas[:2].astype(np.float32)
        y_wind = self.y_meas[2:4].astype(np.float32)
        delta = (self.goal - y_pos).astype(np.float32)

        denom = float(self.cfg.goal_r_max) if self.cfg.goal_r_max > 0 else 1.0
        delta = delta / denom

        return np.concatenate([delta, y_wind]).astype(np.float32)

    # Gym-like API
    def reset(self) -> np.ndarray:
        self.t = 0

        # Hidden 4D state s_0 = [0, 0, d_x, d_y].
        psi0 = float(self.rng.uniform(-math.pi, math.pi))
        self.psi = self._wrap_angle_pi(psi0)
        d0 = np.array(
            [
                float(self.cfg.wind_epsilon) * math.cos(self.psi),
                float(self.cfg.wind_epsilon) * math.sin(self.psi),
            ],
            dtype=np.float32,
        )
        self.x_ssm = np.array([0.0, 0.0, d0[0], d0[1]], dtype=np.float32)
        self.x = self.x_ssm[:2].copy()

        # Sample random goal
        r = float(self.rng.uniform(self.cfg.goal_r_min, self.cfg.goal_r_max))
        ang = float(self.rng.uniform(-math.pi, math.pi))
        self.goal = np.array([r * math.cos(ang), r * math.sin(ang)], dtype=np.float32)

        self.last_delta_psi = 0.0
        self.last_A_t = self.build_transition_matrix(self.last_delta_psi)
        self.y_meas = self._measure_observation_state()

        return self._build_rl_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.t += 1

        ax = float(action[0])
        ay = float(action[1])
        ax = max(-1.0, min(1.0, ax))
        ay = max(-1.0, min(1.0, ay))
        action_vec = np.array([ax, ay], dtype=np.float32)

        delta_psi_t = float(self.rng.normal(0.0, self.cfg.wind_volatility))
        A_t = self.build_transition_matrix(delta_psi_t)
        process_noise = self._sample_process_noise()
        wind_vec_this_step = self._wind_vec().copy()

        self.x_ssm = (
            A_t @ self.x_ssm
            + self.B @ action_vec
            + process_noise
        ).astype(np.float32)
        self.x = self.x_ssm[:2].copy()
        self.psi = self._wrap_angle_pi(float(math.atan2(self.x_ssm[3], self.x_ssm[2])))
        self.last_delta_psi = delta_psi_t
        self.last_A_t = A_t.copy()
        self.y_meas = self._measure_observation_state()

        dist = float(np.linalg.norm(self.goal - self.x))
        done_success = dist <= self.cfg.goal_radius
        done_timeout = self.t >= self.cfg.max_steps
        done = bool(done_success or done_timeout)

        if done_success:
            reward = float(self.cfg.success_reward)
        elif done_timeout:
            reward = float(self.cfg.timeout_penalty)
        else:
            reward = float(self.cfg.step_penalty)

        info = {
            "t": self.t,
            "dist": dist,
            "success": bool(done_success),
            "timeout": bool(done_timeout),
            "goal": self.goal.copy(),
            "x": self.x.copy(),
            "x_true": self.x.copy(),
            "x_ssm": self.x_ssm.copy(),
            "y_meas": self.y_meas.copy(),
            "wind_psi": float(self.psi),
            "wind_vec": wind_vec_this_step,
            "delta_psi_t": float(delta_psi_t),
            "A_t": A_t.copy(),
            "B": self.B.copy(),
            "F": self.F.copy(),
            "Q": self.process_covariance().copy(),
            "R": self.observation_covariance().copy(),
            "action_xy": np.array([ax, ay], dtype=np.float32),
            "action_norm": float(math.hypot(ax, ay)),
        }

        obs = self._build_rl_obs()
        return obs, reward, done, info


# -----------------------------
# Squashed Gaussian policy (2D) with FIXED std
# -----------------------------

def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class ActorCritic(nn.Module):
    """
    2D squashed Gaussian policy:
      y ~ Normal(mu, std_fixed)
      a = tanh(y) in [-1,1]^2

    std_fixed is constant and NOT learned.
    """

    def __init__(self, obs_dim: int = 4, hidden: int = 128, act_dim: int = 2, std_fixed: float = 0.35):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.std_fixed = float(std_fixed)

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        mu = self.mu(h)  # (B,2)
        v = self.v(h).squeeze(-1)  # (B,)
        std = torch.full_like(mu, self.std_fixed)  # (B,2)
        return mu, std, v

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """
        obs: (obs_dim,)
        returns:
          action: np.array (2,) in [-1,1]
          logp: float
          v: float
        """
        mu, std, v = self.forward(obs.unsqueeze(0))  # (1,2), (1,2), (1,)
        dist = torch.distributions.Normal(mu, std)

        y = dist.sample()  # (1,2)
        logp_y = dist.log_prob(y).sum(-1)  # (1,)

        a = torch.tanh(y)  # (1,2)
        log_det = torch.log(1 - a.pow(2) + 1e-8).sum(-1)  # (1,)
        logp = (logp_y - log_det).item()

        action = a.squeeze(0).cpu().numpy().astype(np.float32)  # (2,)
        return action, float(logp), float(v.item())

    @torch.no_grad()
    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _, _ = self.forward(obs.unsqueeze(0))
        return torch.tanh(mu).squeeze(0)  # (2,)

    def logp_and_value(self, obs: torch.Tensor, action: torch.Tensor):
        """
        obs: (B,4)
        action: (B,2) in [-1,1]
        """
        mu, std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)

        a = action
        y = atanh(a)  # inverse tanh
        logp_y = dist.log_prob(y).sum(-1)
        log_det = torch.log(1 - a.pow(2) + 1e-8).sum(-1)
        logp = logp_y - log_det

        entropy = dist.entropy().sum(-1)  # base normal entropy (ok as bonus)

        return logp, v, entropy


# -----------------------------
# PPO Utilities
# -----------------------------

@dataclass
class PPOConfig:
    total_steps: int = 250_000
    rollout_len: int = 2048
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    train_epochs: int = 10
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    device: str = "cpu"
    seed: int = 2026
    log_every: int = 20_000


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Training
# -----------------------------

def train(env: AdvRL2DEnv, ppo_cfg: PPOConfig, model_path: str | None = None) -> ActorCritic:
    set_seed(ppo_cfg.seed)
    device = torch.device(ppo_cfg.device)

    model = ActorCritic(obs_dim=4, hidden=128, act_dim=2, std_fixed=0.35).to(device)

    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            print("Could not load model (shape mismatch?), starting fresh.")

    opt = optim.Adam(model.parameters(), lr=ppo_cfg.lr)

    obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    steps_done = 0
    ep_returns: List[float] = []
    ep_success: List[float] = []
    current_ep_return = 0.0

    while steps_done < ppo_cfg.total_steps:
        obs_buf = np.zeros((ppo_cfg.rollout_len, 4), dtype=np.float32)
        act_buf = np.zeros((ppo_cfg.rollout_len, 2), dtype=np.float32)  # 2D action
        logp_buf = np.zeros((ppo_cfg.rollout_len,), dtype=np.float32)
        rew_buf = np.zeros((ppo_cfg.rollout_len,), dtype=np.float32)
        done_buf = np.zeros((ppo_cfg.rollout_len,), dtype=np.bool_)
        val_buf = np.zeros((ppo_cfg.rollout_len + 1,), dtype=np.float32)

        rollout_steps = 0
        for t in range(ppo_cfg.rollout_len):
            obs_buf[t] = obs_t.detach().cpu().numpy()

            action, logp, v = model.act(obs_t)
            val_buf[t] = v

            next_obs, reward, done, info = env.step(action)

            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = reward
            done_buf[t] = done

            current_ep_return += reward
            steps_done += 1
            rollout_steps += 1

            if done:
                ep_returns.append(current_ep_return)
                ep_success.append(1.0 if info.get("success", False) else 0.0)
                current_ep_return = 0.0
                next_obs = env.reset()

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            if steps_done >= ppo_cfg.total_steps:
                break

        with torch.no_grad():
            _, _, v_last = model.forward(obs_t.unsqueeze(0))
        val_buf[rollout_steps] = float(v_last.item())

        adv, ret = compute_gae(
            rew_buf[:rollout_steps],
            val_buf[:rollout_steps + 1],
            done_buf[:rollout_steps],
            ppo_cfg.gamma,
            ppo_cfg.lam,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_tensor = torch.tensor(obs_buf[:rollout_steps], dtype=torch.float32, device=device)
        act_tensor = torch.tensor(act_buf[:rollout_steps], dtype=torch.float32, device=device)
        logp_old = torch.tensor(logp_buf[:rollout_steps], dtype=torch.float32, device=device)
        adv_tensor = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(ret, dtype=torch.float32, device=device)

        idx = np.arange(obs_tensor.shape[0])
        for _ in range(ppo_cfg.train_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), ppo_cfg.minibatch_size):
                mb = idx[start:start + ppo_cfg.minibatch_size]

                logp, v, ent = model.logp_and_value(obs_tensor[mb], act_tensor[mb])
                ratio = torch.exp(logp - logp_old[mb])

                surr1 = ratio * adv_tensor[mb]
                surr2 = torch.clamp(ratio, 1 - ppo_cfg.clip, 1 + ppo_cfg.clip) * adv_tensor[mb]

                loss = (
                    -torch.min(surr1, surr2).mean()
                    + ppo_cfg.vf_coef * 0.5 * (ret_tensor[mb] - v).pow(2).mean()
                    - ppo_cfg.ent_coef * ent.mean()
                )

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                opt.step()

        if (steps_done // ppo_cfg.log_every) != ((steps_done - rollout_steps) // ppo_cfg.log_every):
            if ep_returns:
                last = ep_returns[-50:]
                succ = ep_success[-50:]
                print(f"[steps={steps_done}] avg_ret={np.mean(last):.2f} succ={np.mean(succ):.2f}")

    return model


# -----------------------------
# Visualization
# -----------------------------

@torch.no_grad()
def run_episode_collect(env: AdvRL2DEnv, model: ActorCritic, device: str = "cpu"):
    device_t = torch.device(device)
    obs = env.reset()

    traj = [env.x.copy()]
    obs_list = [obs.copy()]
    wind_list = []
    goal = env.goal.copy()
    success = False

    for _ in range(env.cfg.max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device_t)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)  # (2,)

        current_wind = env._wind_vec()
        wind_list.append(current_wind)

        obs, r, done, info = env.step(action)
        traj.append(env.x.copy())
        obs_list.append(obs.copy())

        if done:
            success = bool(info.get("success", False))
            break

    return np.array(traj), np.array(obs_list), np.array(wind_list), goal, success


def plot_trajectories_grid(
    env: AdvRL2DEnv,
    model: ActorCritic,
    episodes: int = 12,
    device: str = "cpu",
    results_dir: str = "."
):
    cols = int(math.ceil(math.sqrt(episodes)))
    rows = int(math.ceil(episodes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(episodes):
        ax = axes[i]
        traj, obs_list, winds, goal, success = run_episode_collect(env, model, device=device)

        ax.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2, linewidth=1, alpha=0.6)
        ax.scatter(traj[0, 0], traj[0, 1], marker="s", color="green")
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="X", color="red")
        ax.scatter(goal[0], goal[1], marker="*", color="gold", s=100)

        # Policy arrows: now directly action vectors
        Ux, Uy = [], []
        for ob in obs_list[:-1]:
            ob_t = torch.tensor(ob, dtype=torch.float32)
            a_mean = model.mean_action(ob_t).cpu().numpy()  # (2,)
            Ux.append(float(a_mean[0]))
            Uy.append(float(a_mean[1]))
        Ux = np.array(Ux, dtype=np.float32)
        Uy = np.array(Uy, dtype=np.float32)

        step = 2
        ax.quiver(
            traj[:-1:step, 0], traj[:-1:step, 1],
            Ux[::step], Uy[::step],
            color="blue", alpha=0.5, width=0.006
        )

        # Wind arrows
        if len(winds) > 0:
            ax.quiver(
                traj[:-1:step, 0], traj[:-1:step, 1],
                winds[::step, 0], winds[::step, 1],
                color="skyblue", alpha=0.4, width=0.005
            )

        circ = plt.Circle((goal[0], goal[1]), env.cfg.goal_radius, fill=False, color="gray", linestyle="--")
        ax.add_patch(circ)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Ep {i+1} | {'OK' if success else 'FAIL'}")

    for j in range(episodes, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "trajectories_grid_v2_wind_ssm.png"))
    plt.close(fig)


@torch.no_grad()
def plot_policy_on_one_trajectory(env: AdvRL2DEnv, model: ActorCritic, device: str = "cpu", results_dir: str = "."):
    traj, obs_list, winds, goal, success = run_episode_collect(env, model, device=device)

    # Policy vectors along trajectory
    Ux, Uy = [], []
    for ob in obs_list[:-1]:
        ob_t = torch.tensor(ob, dtype=torch.float32)
        a_mean = model.mean_action(ob_t).cpu().numpy()
        Ux.append(float(a_mean[0]))
        Uy.append(float(a_mean[1]))
    Ux = np.array(Ux, dtype=np.float32)
    Uy = np.array(Uy, dtype=np.float32)

    fig = plt.figure(figsize=(8, 8))
    plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=3, linewidth=1, label="trajectory", alpha=0.5)

    plt.quiver(
        traj[:-1, 0], traj[:-1, 1], Ux, Uy,
        angles="xy", scale_units="xy", scale=1.0, width=0.005,
        color="blue", label="policy action (ax,ay)"
    )

    plt.quiver(
        traj[:-1, 0], traj[:-1, 1], winds[:, 0], winds[:, 1],
        angles="xy", scale_units="xy", scale=1.0, width=0.003,
        color="skyblue", alpha=0.6, label="wind force"
    )

    plt.scatter(traj[0, 0], traj[0, 1], marker="s", s=100, label="start")
    plt.scatter(traj[-1, 0], traj[-1, 1], marker="X", s=100, label="end")
    plt.scatter(goal[0], goal[1], marker="*", s=200, color="gold", label="goal")

    circ = plt.Circle((goal[0], goal[1]), env.cfg.goal_radius, fill=False, color="red")
    plt.gca().add_patch(circ)
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"Policy vs Wind (SSM env) | Success: {success}")
    plt.savefig(os.path.join(results_dir, "policy_on_one_trajectory_v2_wind_ssm.png"))
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    # Train on clean policy observations so the controller learns directly from
    # the shared SSM state without measurement corruption.
    env_cfg = AdvRLEnvConfig(
        obs_noise_std=0.0,
        proc_noise_std=0.0,
        obs_position_std=0.0,
        obs_wind_std=0.0,
        position_process_std=0.0,
        wind_process_std=0.0,
        wind_epsilon=0.9,
        wind_volatility=0.25,
        wind_persistence=0.92,
        seed=2025,
        step_penalty=-1.0,
        success_reward=25.0,
        timeout_penalty=-25.0,
    )
    env = AdvRL2DEnv(env_cfg)

    ppo_cfg = PPOConfig(    
        total_steps=500_000,
        device="cpu",
        seed=2026,
    )

    save_path = default_policy_model_path()
    warm_start_path = warm_start_policy_model_path()

    print("Starting training of AdvRL_v3 (shared 4D SSM, clean policy observations, fixed std)...")
    if warm_start_path is not None and os.path.abspath(warm_start_path) != os.path.abspath(save_path):
        print(f"Warm-starting from checkpoint: {warm_start_path}")
    elif warm_start_path is not None:
        print(f"Continuing training from existing shared-SSM checkpoint: {warm_start_path}")
    else:
        print("No compatible checkpoint found; training starts from scratch.")

    save_dir = os.path.join(_THIS_DIR, "outputs", "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    model = train(env, ppo_cfg, model_path=warm_start_path)
    torch.save(model.state_dict(), save_path)
    print(f"Saved shared-SSM policy checkpoint to: {save_path}")

    figures_dir = os.path.join(_THIS_DIR, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print("Generating plots...")
    plot_trajectories_grid(env, model, episodes=12, device=ppo_cfg.device, results_dir=figures_dir)
    plot_policy_on_one_trajectory(env, model, device=ppo_cfg.device, results_dir=figures_dir)
    print(f"Done. Figures saved in {figures_dir}")


if __name__ == "__main__":
    main()
