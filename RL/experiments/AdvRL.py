# AdvRL.py
# RL experiment: 2D point agent starts at (0,0) and must reach a scenario-dependent goal (gx,gy).
# SSM dynamics with A = I:
#   x_t = x_{t-1} + u(a_{t-1}) + w_t
#
# Observation (relative goal vector, small noise):
#   z_t = [gx - px, gy - py] + v_t
#
# Action a = (theta, d): angle in [-pi, pi], distance in [0,1].
#
# Reward:
#   r_t = -1 each step until success
#   r_t = +10 (or whatever you set) when within goal_radius, and episode ends
#
# After training:
#   1) Plot trajectories in a single figure with one subplot per episode.
#   2) Plot the policy (mean action) along one trajectory as quiver arrows.
# Saves model to: RL/saved_models/AdvRL_policy.pt

from __future__ import annotations

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


# -----------------------------
# Environment (SSM-style)
# -----------------------------

@dataclass
class AdvRLEnvConfig:
    goal_r_min: float = 0.5
    goal_r_max: float = 6.0

    obs_noise_std: float = 0.005
    proc_noise_std: float = 0.00
    max_steps: int = 80
    goal_radius: float = 0.50
    seed: int = 0


class AdvRL2DEnv:
    """
    State x_t: 2D position (px, py)
    Goal g: 2D target (gx, gy) sampled per episode
    Observation z_t: [gx - px, gy - py] + noise   (2D)
    Action a_t: [theta]   theta in [-pi, pi]
    Control mapping u(a): d * [cos(theta), sin(theta)] with d fixed to 1.0 here
    """
    def __init__(self, cfg: AdvRLEnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.x = np.array([0.0, 0.0], dtype=np.float32)

        # Sample random goal each episode
        r = float(self.rng.uniform(self.cfg.goal_r_min, self.cfg.goal_r_max))
        ang = float(self.rng.uniform(-math.pi, math.pi))
        self.goal = np.array([r * math.cos(ang), r * math.sin(ang)], dtype=np.float32)

        return self._obs()

    def _obs(self) -> np.ndarray:
        delta = (self.goal - self.x).astype(np.float32)
        dist = float(np.linalg.norm(delta)) + 1e-8
        direction = (delta / dist).astype(np.float32)  # 2D unit vector: [cos, sin]
        # opcional ruido (si obs_noise_std > 0)
        if self.cfg.obs_noise_std > 0:
            direction = direction + self.rng.normal(0.0, self.cfg.obs_noise_std, size=(2,)).astype(np.float32)
        return direction.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action: np.array([theta]) where theta in [-pi, pi]
        """
        self.t += 1
        theta = float(action[0])
        d = 1.0  # Fixed distance

        # u(a) = d*[cos, sin]
        u = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32) * d

        # process noise w_t
        w = self.rng.normal(0.0, self.cfg.proc_noise_std, size=(2,)).astype(np.float32)

        # SSM dynamics with A=I: x_t = x_{t-1} + u(a_{t-1}) + w_t
        self.x = (self.x + u + w).astype(np.float32)

        dist = float(np.linalg.norm(self.goal - self.x))
        done_success = dist <= self.cfg.goal_radius
        done_timeout = self.t >= self.cfg.max_steps
        done = done_success or done_timeout

        # OJO: tú comentas +10, pero tu código original usaba 1000.0.
        # Aquí lo dejo igual que tu implementación original para no cambiar conducta:
        if done_success:
            reward = 10.0
        elif done_timeout:
            reward = -10.0
        else:
            reward = -1.0


        info = {
            "t": self.t,
            "dist": dist,
            "success": done_success,
            "timeout": done_timeout,
            "goal": self.goal.copy(),
            "x": self.x.copy(),
        }
        return self._obs(), float(reward), bool(done), info


# -----------------------------
# Squashed Gaussian policy
# -----------------------------

def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class ActorCritic(nn.Module):
    """
    Inputs: observation z_t = [gx - px, gy - py]  (2 dims)
    Outputs:
      - Stochastic policy over theta
        theta: sample y0 in R -> tanh -> [-1,1] -> scale to [-pi, pi]
      - Value function V(z_t)
    """
    def __init__(self, obs_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.2], dtype=torch.float32))  # latent std
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std, min=-1.5, max=1.0)  # std en [~0.22, ~2.7]
        std = torch.exp(log_std).unsqueeze(0).expand_as(mu)
        v = self.v(h).squeeze(-1)
        return mu, std, v

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """Sample a stochastic action from the policy."""
        mu, std, v = self.forward(obs.unsqueeze(0))
        dist = torch.distributions.Normal(mu, std)

        y = dist.sample()                # latent
        logp_y = dist.log_prob(y).sum(-1)

        y0 = y[..., 0]
        theta = math.pi * torch.tanh(y0)     # [-pi, pi]

        # change-of-variables correction
        log_det = torch.log(math.pi * (1 - torch.tanh(y0) ** 2) + 1e-8)
        logp = (logp_y - log_det).item()

        action = torch.stack([theta], dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
        return action, float(logp), float(v.item())

    @torch.no_grad()
    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic (mean) action given obs. Returns shape (1,) = [theta]."""
        mu, _, _ = self.forward(obs.unsqueeze(0))
        y0 = mu[..., 0]
        theta = math.pi * torch.tanh(y0)  # [-pi, pi]
        return torch.stack([theta], dim=-1).squeeze(0)

    def logp_and_value(self, obs: torch.Tensor, action: torch.Tensor):
        """Compute log pi(a|obs) (with change of variables) and value."""
        mu, std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)

        theta = action[:, 0]
        y0 = atanh(theta / math.pi)

        y = torch.stack([y0], dim=-1)
        logp_y = dist.log_prob(y).sum(-1)

        log_det = torch.log(math.pi * (1 - torch.tanh(y0) ** 2) + 1e-8)
        logp = logp_y - log_det

        entropy = dist.entropy().sum(-1)
        return logp, v, entropy


# -----------------------------
# PPO utilities
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
    ent_coef: float = 0.01
    train_epochs: int = 10
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    device: str = "cpu"
    seed: int = 0
    log_every: int = 10_000


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
# Training loop
# -----------------------------

def train(env: AdvRL2DEnv, ppo_cfg: PPOConfig, model_path: str | None = None) -> ActorCritic:
    set_seed(ppo_cfg.seed)
    device = torch.device(ppo_cfg.device)

    model = ActorCritic(obs_dim=2, hidden=128).to(device)
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded matching layers from pretrained model.")

    opt = optim.Adam(model.parameters(), lr=ppo_cfg.lr)

    obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    steps_done = 0
    ep_returns: List[float] = []
    ep_success: List[float] = []
    current_ep_return = 0.0

    while steps_done < ppo_cfg.total_steps:
        obs_buf = np.zeros((ppo_cfg.rollout_len, 2), dtype=np.float32)   # <--- 2 dims
        act_buf = np.zeros((ppo_cfg.rollout_len, 1), dtype=np.float32)
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

        # bootstrap
        with torch.no_grad():
            _, _, v_last = model.forward(obs_t.unsqueeze(0))
        val_buf[rollout_steps] = float(v_last.item())

        # GAE
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

        n = obs_tensor.shape[0]
        idx = np.arange(n)

        for _ in range(ppo_cfg.train_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, ppo_cfg.minibatch_size):
                mb = idx[start:start + ppo_cfg.minibatch_size]
                mb_obs = obs_tensor[mb]
                mb_act = act_tensor[mb]
                mb_adv = adv_tensor[mb]
                mb_ret = ret_tensor[mb]
                mb_logp_old = logp_old[mb]

                logp, v, ent = model.logp_and_value(mb_obs, mb_act)
                ratio = torch.exp(logp - mb_logp_old)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - ppo_cfg.clip, 1 + ppo_cfg.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (mb_ret - v).pow(2).mean()
                entropy_loss = -ent.mean()

                loss = policy_loss + ppo_cfg.vf_coef * value_loss + ppo_cfg.ent_coef * entropy_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                opt.step()

        # logging
        if (steps_done // ppo_cfg.log_every) != ((steps_done - rollout_steps) // ppo_cfg.log_every):
            if ep_returns:
                last = ep_returns[-50:] if len(ep_returns) >= 50 else ep_returns
                succ = ep_success[-50:] if len(ep_success) >= 50 else ep_success
                print(
                    f"[steps={steps_done}] "
                    f"avg_return({len(last)}eps)={np.mean(last):.2f} "
                    f"succ_rate={np.mean(succ):.2f}"
                )

    return model


# -----------------------------
# Visualization after training
# -----------------------------

@torch.no_grad()
def run_episode_collect(env: AdvRL2DEnv, model: ActorCritic, device: str = "cpu"):
    device_t = torch.device(device)
    obs = env.reset()

    traj = [env.x.copy()]      # true state positions
    obs_list = [obs.copy()]    # store obs (2D) to later query policy
    goal = env.goal.copy()
    success = False

    for _ in range(env.cfg.max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device_t)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)
        obs, r, done, info = env.step(action)
        traj.append(env.x.copy())
        obs_list.append(obs.copy())
        if done:
            success = bool(info.get("success", False))
            break

    traj = np.array(traj, dtype=np.float32)
    return traj, np.array(obs_list, dtype=np.float32), goal, success


def plot_trajectories_grid(env: AdvRL2DEnv, model: ActorCritic, episodes: int = 12, device: str = "cpu", results_dir: str = "."):
    cols = int(math.ceil(math.sqrt(episodes)))
    rows = int(math.ceil(episodes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    successes = 0
    for i in range(episodes):
        ax = axes[i]
        traj, _, goal, success = run_episode_collect(env, model, device=device)
        successes += int(success)

        ax.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2, linewidth=1)
        ax.scatter(traj[0, 0], traj[0, 1], marker="s")   # start
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="X") # end
        ax.scatter(goal[0], goal[1], marker="*")         # goal

        circ = plt.Circle((goal[0], goal[1]), env.cfg.goal_radius, fill=False)
        ax.add_patch(circ)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(f"Ep {i+1} | {'OK' if success else 'FAIL'}")

    for j in range(episodes, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Trajectories after training (success {successes}/{episodes})")
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, "trajectories_grid.png"))
    plt.close(fig)


@torch.no_grad()
def plot_policy_on_one_trajectory(env: AdvRL2DEnv, model: ActorCritic, device: str = "cpu", results_dir: str = "."):
    device_t = torch.device(device)

    traj, obs_list, goal, success = run_episode_collect(env, model, device=device)

    # policy mean-action at each observation (except last)
    thetas = []
    for ob in obs_list[:-1]:
        ob_t = torch.tensor(ob, dtype=torch.float32, device=device_t)
        a_mean = model.mean_action(ob_t).cpu().numpy()
        thetas.append(float(a_mean[0]))

    thetas = np.array(thetas, dtype=np.float32)
    ds = np.ones_like(thetas)  # Fixed distance

    # implied step vectors
    Ux = ds * np.cos(thetas)
    Uy = ds * np.sin(thetas)

    fig = plt.figure(figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2, linewidth=1, label="trajectory")
    plt.scatter(traj[0, 0], traj[0, 1], marker="s", label="start")
    plt.scatter(traj[-1, 0], traj[-1, 1], marker="X", label="end")
    plt.scatter(goal[0], goal[1], marker="*", label="goal")

    circ = plt.Circle((goal[0], goal[1]), env.cfg.goal_radius, fill=False)
    plt.gca().add_patch(circ)

    # arrows showing mean policy direction + magnitude along the trajectory
    plt.quiver(
        traj[:-1, 0], traj[:-1, 1],
        Ux, Uy,
        angles="xy", scale_units="xy", scale=1.0,
        width=0.003
    )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.title(f"Policy along one trajectory (mean action) | {'OK' if success else 'FAIL'}")
    plt.legend()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "policy_on_one_trajectory.png"))
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    env_cfg = AdvRLEnvConfig(
        goal_r_min=2.5,
        goal_r_max=15.0,
        obs_noise_std=0.00,
        proc_noise_std=0.00,
        max_steps=22,
        goal_radius=0.5,
        seed=0,
    )

    env = AdvRL2DEnv(env_cfg)

    ppo_cfg = PPOConfig(
        total_steps=100_000,
        rollout_len=2048,
        lr=3e-4,
        train_epochs=10,
        minibatch_size=256,
        device="cpu",  # set "cuda" if available
        seed=0,
        log_every=25_000,
        ent_coef=0.05,
    )

    save_dir = "RL/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "AdvRL_policy.pt")

    model = train(env, ppo_cfg, model_path=save_path)

    # Save policy weights to RL/saved_models
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to: {save_path}")

    # Visualizations
    results_dir = "RL/results"
    os.makedirs(results_dir, exist_ok=True)
    plot_trajectories_grid(env, model, episodes=12, device=ppo_cfg.device, results_dir=results_dir)
    plot_policy_on_one_trajectory(env, model, device=ppo_cfg.device, results_dir=results_dir)


if __name__ == "__main__":
    main()
