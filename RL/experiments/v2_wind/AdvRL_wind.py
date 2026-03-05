#!/usr/bin/env python3
# AdvRL_v2.py
# RL experiment: 2D point agent with WIND dynamics formalized as an SSM.
#
# SSM-inspired model (augmented state):
#   x_t = [p_x, p_y, 1]^T in R^3
#
#   x_{t+1} = A_t x_t + B a_t_aug + w_t
#   y_t     = F x_t + v_t
#
# where:
#   A_t = [[1, 0, eps*cos(psi_t)],
#          [0, 1, eps*sin(psi_t)],
#          [0, 0, 1]]
#
#   B   = [[1, 0, 0],
#          [0, 1, 0],
#          [0, 0, 0]]
#
# NEW ACTION (2D):
#   a_t = [a_x, a_y] in [-1,1]^2
#   a_t_aug = [a_x, a_y, 0]^T
# -> Max step length is sqrt(2).
#
# RL observation returned to policy (4D):
#   z_t = [ (goal - y_t)/goal_r_max, wind_x_t, wind_y_t ]
#
# Rewards:
#   -1 per step (until done)
#   +40 on success
#   -40 on timeout
#
# No noise (default here): obs_noise_std=0, proc_noise_std=0
#
# Saves model to: RL/saved_models/AdvRL_v2_policy.pt

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
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# -----------------------------
# Environment (SSM-style with Wind)
# -----------------------------

@dataclass
class AdvRLEnvConfig:
    goal_r_min: float = 5.0
    goal_r_max: float = 15.0

    # SSM noises
    obs_noise_std: float = 0.00   # sigma_v (measurement noise)
    proc_noise_std: float = 0.00  # sigma_w (process noise on transition)

    max_steps: int = 40
    goal_radius: float = 1.0
    seed: int = 0

    # Wind parameters
    wind_epsilon: float = 0.85       # epsilon in A_t
    wind_volatility: float = 0.25    # random walk std for psi_t

    # Rewards (sparse)
    step_penalty: float = -1.0
    success_reward: float = 40.0
    timeout_penalty: float = -40.0


class AdvRL2DEnv:
    """
    SSM-based environment.

    Hidden augmented state:
        x_t = [p_x, p_y, 1]^T in R^3

    Transition:
        x_{t+1} = A_t x_t + B a_t_aug + w_t
        A_t = [[1, 0, eps*cos(psi_t)],
               [0, 1, eps*sin(psi_t)],
               [0, 0, 1]]
        B   = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0]]
        a_t_aug = [a_x, a_y, 0]^T   with a_x,a_y in [-1,1]

    Measurement:
        y_t = F x_t + v_t,   F = [[1,0,0],[0,1,0]]

    RL observation:
        z_t = [delta_x, delta_y, wind_x, wind_y] in R^4,
        where delta = (goal - y_meas) / goal_r_max (normalized).
    """

    def __init__(self, cfg: AdvRLEnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Measurement matrix F (2x3)
        self.F = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float32
        )

        # Control matrix B (3x3)
        self.B = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            dtype=np.float32
        )

        self.reset()

    @staticmethod
    def _wrap_angle_pi(x: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    def _A_t(self) -> np.ndarray:
        eps = float(self.cfg.wind_epsilon)
        c = math.cos(self.psi)
        s = math.sin(self.psi)
        return np.array(
            [[1.0, 0.0, eps * c],
             [0.0, 1.0, eps * s],
             [0.0, 0.0, 1.0]],
            dtype=np.float32
        )

    def _wind_vec(self) -> np.ndarray:
        return np.array(
            [self.cfg.wind_epsilon * math.cos(self.psi),
             self.cfg.wind_epsilon * math.sin(self.psi)],
            dtype=np.float32
        )

    def _measure_position(self) -> np.ndarray:
        """
        y_t = F x_t + v_t in R^2.
        If obs_noise_std=0 -> noise-free.
        """
        y = (self.F @ self.x_ssm).astype(np.float32)
        if self.cfg.obs_noise_std > 0:
            v = self.rng.normal(0.0, self.cfg.obs_noise_std, size=(2,)).astype(np.float32)
            y = y + v
        return y.astype(np.float32)

    def _build_rl_obs(self) -> np.ndarray:
        """
        RL obs z_t = [delta_x, delta_y, wind_x, wind_y],
        where delta = (goal - y_meas) / goal_r_max.
        """
        y = self.y_meas
        delta = (self.goal - y).astype(np.float32)

        denom = float(self.cfg.goal_r_max) if self.cfg.goal_r_max > 0 else 1.0
        delta = delta / denom  # IMPORTANT (you had this bugged before)

        wind_vec = self._wind_vec()
        return np.concatenate([delta, wind_vec]).astype(np.float32)

    # Gym-like API
    def reset(self) -> np.ndarray:
        self.t = 0

        # Hidden augmented state x_0 = [0,0,1]
        self.x_ssm = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.x = self.x_ssm[:2].copy()

        # Sample random goal
        r = float(self.rng.uniform(self.cfg.goal_r_min, self.cfg.goal_r_max))
        ang = float(self.rng.uniform(-math.pi, math.pi))
        self.goal = np.array([r * math.cos(ang), r * math.sin(ang)], dtype=np.float32)

        # Initial wind direction psi_0
        self.psi = float(self.rng.uniform(-math.pi, math.pi))
        self.psi = self._wrap_angle_pi(self.psi)

        # Initial measurement y_0
        self.y_meas = self._measure_position()

        return self._build_rl_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.t += 1

        # Action = 2D step vector in [-1,1]^2
        ax = float(action[0])
        ay = float(action[1])

        # Safety clamp
        ax = max(-1.0, min(1.0, ax))
        ay = max(-1.0, min(1.0, ay))

        # Augmented action in R^3
        a_aug = np.array([ax, ay, 0.0], dtype=np.float32)

        # Transition matrix using current psi_t
        A_t = self._A_t()
        wind_vec_this_step = self._wind_vec().copy()

        # Process noise (R^3); keep augmented coordinate exact
        if self.cfg.proc_noise_std > 0:
            w = self.rng.normal(0.0, self.cfg.proc_noise_std, size=(3,)).astype(np.float32)
            w[2] = 0.0
        else:
            w = np.zeros(3, dtype=np.float32)

        # SSM state update
        self.x_ssm = (A_t @ self.x_ssm + self.B @ a_aug + w).astype(np.float32)
        self.x_ssm[2] = 1.0  # keep augmented coordinate = 1

        # Update true 2D position
        self.x = self.x_ssm[:2].copy()

        # Evolve wind direction for NEXT step (random walk) + wrap
        self.psi += float(self.rng.normal(0.0, self.cfg.wind_volatility))
        self.psi = self._wrap_angle_pi(self.psi)

        # New measurement
        self.y_meas = self._measure_position()

        # Termination based on TRUE position
        dist = float(np.linalg.norm(self.goal - self.x))
        done_success = dist <= self.cfg.goal_radius
        done_timeout = self.t >= self.cfg.max_steps
        done = bool(done_success or done_timeout)

        # Reward
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
            "A_t": A_t.copy(),
            "B": self.B.copy(),
            "F": self.F.copy(),
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

        current_wind = np.array(
            [env.cfg.wind_epsilon * math.cos(env.psi),
             env.cfg.wind_epsilon * math.sin(env.psi)],
            dtype=np.float32
        )
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
    # No noise: deterministic measurement & transition
    env_cfg = AdvRLEnvConfig(
        obs_noise_std=0.0,
        proc_noise_std=0.0,
        wind_epsilon=0.9,
        wind_volatility=0.25,
        seed=2025,
        step_penalty=-1.0,
        success_reward=25.0,
        timeout_penalty=-25.0,
    )
    env = AdvRL2DEnv(env_cfg)

    ppo_cfg = PPOConfig(    
        total_steps=1_000_000,
        device="cpu",
        seed=2026,
    )

    save_dir = os.path.join(_PROJECT_ROOT, "RL", "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "AdvRL_v2_policy.pt")

    print("Starting training of AdvRL_v2 (SSM wind, 2D action, fixed std, sparse reward, no noise)...")
    model = train(env, ppo_cfg, model_path=save_path)
    torch.save(model.state_dict(), save_path)

    results_dir = os.path.join(_PROJECT_ROOT, "RL", "results")
    os.makedirs(results_dir, exist_ok=True)
    print("Generating plots...")
    plot_trajectories_grid(env, model, episodes=12, device=ppo_cfg.device, results_dir=results_dir)
    plot_policy_on_one_trajectory(env, model, device=ppo_cfg.device, results_dir=results_dir)
    print(f"Done. Results saved in {results_dir}")


if __name__ == "__main__":
    main()