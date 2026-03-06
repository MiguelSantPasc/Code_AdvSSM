#!/usr/bin/env python3
# Eval_clean_vs_noisy.py
#
# HARD-CODED PATHS:
#   - AdvRL_wind.py    : ./AdvRL_wind.py  (same folder as this script)
#   - Model checkpoint : ../../saved_models/AdvRL_v2_policy.pt
#   - Output figures   : ../../results/
#
# What it does:
#   1) Plot 4 paired episodes (clean vs noisy obs) in one row:
#        - same wind process across both (noise RNG is separated)
#        - wind arrows + both trajectories
#   2) Plot accumulated episode returns clean vs noisy (same episode seeds)

from __future__ import annotations

import os
import sys
import math
from dataclasses import asdict

import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================
# HARD-CODED CONFIG
# =========================

DEVICE = "cpu"

# Noise ONLY on measured position (radar-like), not on wind
NOISE_STD = 0.50

# 4 episodes for the trajectory panel (paired clean/noisy with same seed)
SEEDS_4 = [111, 222, 333, 444]

# Returns plot settings
N_RETURNS_EPISODES = 500
RETURNS_SEED0 = 10_000

# -------------------------
# HARD-CODED PATHS (relative to this script)
# -------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ADV_FILE_PATH = os.path.abspath(os.path.join(_THIS_DIR, "AdvRL_wind.py"))
MODEL_PATH = os.path.abspath(os.path.join(_THIS_DIR, "../../saved_models/AdvRL_v2_policy.pt"))
RESULTS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "../../results"))


# =========================
# Env wrapper: separate RNG for obs noise only
# (so wind RNG stream is identical between clean & noisy)
# =========================

def make_env_with_separate_obs_rng(AdvRL2DEnv_base):
    class AdvRL2DEnv_SeparateObsRNG(AdvRL2DEnv_base):
        """
        Same env as AdvRL2DEnv, but observation noise uses rng_obs so it does NOT
        consume randomness from the env's main RNG (used for goal/wind evolution).
        """

        def __init__(self, cfg):
            self.rng_obs = np.random.default_rng(int(cfg.seed) + 12345)
            super().__init__(cfg)

        def _measure_position(self) -> np.ndarray:
            y = (self.F @ self.x_ssm).astype(np.float32)
            if self.cfg.obs_noise_std > 0:
                v = self.rng_obs.normal(0.0, self.cfg.obs_noise_std, size=(2,)).astype(np.float32)
                y = y + v
            return y.astype(np.float32)

    return AdvRL2DEnv_SeparateObsRNG


# =========================
# Rollout + plots
# =========================

@torch.no_grad()
def rollout_episode(env, model, device: str = "cpu"):
    """
    Deterministic eval using mean_action.
    Returns:
      traj: (T+1,2)
      winds: (T,2) wind vec used each step (computed before env.step)
      ep_return: float
      success: bool
      goal: (2,)
    """
    dev = torch.device(device)
    obs = env.reset()

    traj = [env.x.copy()]
    winds = []
    ep_return = 0.0
    success = False
    goal = env.goal.copy()

    for _ in range(env.cfg.max_steps):
        # wind applied THIS step (before env.step updates psi)
        w = np.array(
            [env.cfg.wind_epsilon * math.cos(env.psi),
             env.cfg.wind_epsilon * math.sin(env.psi)],
            dtype=np.float32,
        )
        winds.append(w)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev)
        a = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        obs, r, done, info = env.step(a)
        ep_return += float(r)
        traj.append(env.x.copy())

        if done:
            success = bool(info.get("success", False))
            break

    return (
        np.array(traj, dtype=np.float32),
        np.array(winds, dtype=np.float32),
        float(ep_return),
        bool(success),
        goal,
    )


def plot_4eps_row_clean_vs_noisy(
    AdvRLEnvConfig,
    EnvCleanClass,
    EnvNoisyClass,
    base_cfg,
    model,
    noise_std: float,
    seeds_4: list[int],
    device: str,
    results_dir: str,
):
    os.makedirs(results_dir, exist_ok=True)
    if len(seeds_4) != 4:
        raise ValueError("SEEDS_4 must have exactly 4 seeds")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for i, seed in enumerate(seeds_4):
        ax = axes[i]

        cfg_clean = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_noisy = AdvRLEnvConfig(**{**asdict(base_cfg)})

        cfg_clean.seed = int(seed)
        cfg_clean.obs_noise_std = 0.0

        cfg_noisy.seed = int(seed)
        cfg_noisy.obs_noise_std = float(noise_std)

        env_clean = EnvCleanClass(cfg_clean)
        env_noisy = EnvNoisyClass(cfg_noisy)

        traj_c, winds_c, ret_c, succ_c, goal_c = rollout_episode(env_clean, model, device=device)
        traj_n, winds_n, ret_n, succ_n, _goal_n = rollout_episode(env_noisy, model, device=device)

        goal = goal_c

        ax.plot(traj_c[:, 0], traj_c[:, 1], marker="o", markersize=2, linewidth=1, alpha=0.85, label="clean")
        ax.plot(traj_n[:, 0], traj_n[:, 1], marker="o", markersize=2, linewidth=1, alpha=0.85, label=f"noisy σ={noise_std}")

        ax.scatter(0.0, 0.0, marker="s", s=70, label="start")
        ax.scatter(goal[0], goal[1], marker="*", s=180, label="goal")

        step = 2
        if len(winds_c) > 0:
            ax.quiver(
                traj_c[:-1:step, 0], traj_c[:-1:step, 1],
                winds_c[::step, 0], winds_c[::step, 1],
                alpha=0.35, width=0.004
            )
        if len(winds_n) > 0:
            ax.quiver(
                traj_n[:-1:step, 0], traj_n[:-1:step, 1],
                winds_n[::step, 0], winds_n[::step, 1],
                alpha=0.20, width=0.004
            )

        circ = plt.Circle((goal[0], goal[1]), base_cfg.goal_radius, fill=False, linestyle="--", alpha=0.6)
        ax.add_patch(circ)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"Ep {i+1} (seed={seed})\n"
            f"clean: {'OK' if succ_c else 'FAIL'} ret={ret_c:.0f} | "
            f"noisy: {'OK' if succ_n else 'FAIL'} ret={ret_n:.0f}"
        )

        if i == 0:
            ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    outpath = os.path.join(results_dir, "eval_trajs_4eps_clean_vs_noisy.png")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


def plot_returns_clean_vs_noisy(
    AdvRLEnvConfig,
    EnvCleanClass,
    EnvNoisyClass,
    base_cfg,
    model,
    noise_std: float,
    n_episodes: int,
    seed0: int,
    device: str,
    results_dir: str,
):
    os.makedirs(results_dir, exist_ok=True)

    rets_clean = []
    rets_noisy = []

    for k in range(n_episodes):
        seed = int(seed0 + k)

        cfg_clean = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_noisy = AdvRLEnvConfig(**{**asdict(base_cfg)})

        cfg_clean.seed = seed
        cfg_clean.obs_noise_std = 0.0

        cfg_noisy.seed = seed
        cfg_noisy.obs_noise_std = float(noise_std)

        env_clean = EnvCleanClass(cfg_clean)
        env_noisy = EnvNoisyClass(cfg_noisy)

        _, _, ret_c, _, _ = rollout_episode(env_clean, model, device=device)
        _, _, ret_n, _, _ = rollout_episode(env_noisy, model, device=device)

        rets_clean.append(ret_c)
        rets_noisy.append(ret_n)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(rets_clean, linewidth=1.5, label="clean (σ=0)")
    plt.plot(rets_noisy, linewidth=1.5, label=f"noisy (σ={noise_std})")
    plt.grid(True, alpha=0.25)
    plt.xlabel("Episode")
    plt.ylabel("Accumulated reward (episode return)")
    plt.title("Episode returns: clean vs noisy position observation")
    plt.legend()

    outpath = os.path.join(results_dir, "eval_returns_clean_vs_noisy.png")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")

def plot_accumulated_reward_clean_vs_noisy(
    AdvRLEnvConfig,
    EnvCleanClass,
    EnvNoisyClass,
    base_cfg,
    model,
    noise_std: float,
    n_episodes: int,
    seed0: int,
    device: str,
    results_dir: str,
):
    """
    Plots ACCUMULATED reward across episodes (cumulative sum of episode returns),
    for clean vs noisy observation.

    x-axis: episode index
    y-axis: cumulative reward up to that episode (NOT per-episode return)

    Saves: eval_accumulated_reward_clean_vs_noisy.png
    """
    os.makedirs(results_dir, exist_ok=True)

    acc_clean = []
    acc_noisy = []
    total_c = 0.0
    total_n = 0.0

    for k in range(n_episodes):
        seed = int(seed0 + k)

        cfg_clean = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_noisy = AdvRLEnvConfig(**{**asdict(base_cfg)})

        cfg_clean.seed = seed
        cfg_clean.obs_noise_std = 0.0

        cfg_noisy.seed = seed
        cfg_noisy.obs_noise_std = float(noise_std)

        env_clean = EnvCleanClass(cfg_clean)
        env_noisy = EnvNoisyClass(cfg_noisy)

        # rollout_episode must exist in your file and return (traj, winds, ep_return, success, goal)
        _, _, ret_c, _, _ = rollout_episode(env_clean, model, device=device)
        _, _, ret_n, _, _ = rollout_episode(env_noisy, model, device=device)

        total_c += float(ret_c)
        total_n += float(ret_n)

        acc_clean.append(total_c)
        acc_noisy.append(total_n)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(acc_clean, linewidth=1.5, label="clean (σ=0) accumulated")
    plt.plot(acc_noisy, linewidth=1.5, label=f"noisy (σ={noise_std}) accumulated")
    plt.grid(True, alpha=0.25)
    plt.xlabel("Episode")
    plt.ylabel("Accumulated reward (cumulative sum)")
    plt.title("Accumulated reward across episodes: clean vs noisy observation")
    plt.legend()

    outpath = os.path.join(results_dir, "eval_accumulated_reward_clean_vs_noisy.png")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")

# =========================
# Main
# =========================

def main():
    # Hard checks for hardcoded paths
    if not os.path.exists(ADV_FILE_PATH):
        raise FileNotFoundError(f"AdvRL_wind.py not found at:\n  {ADV_FILE_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at:\n  {MODEL_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- NORMAL IMPORT (no importlib) ----
    # Ensure the script directory is importable
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)

    import AdvRL_wind as mod  # <-- normal import of your .py
    print(f"[import] Imported AdvRL_wind from: {ADV_FILE_PATH}")

    ActorCritic = mod.ActorCritic
    AdvRLEnvConfig = mod.AdvRLEnvConfig
    AdvRL2DEnv = mod.AdvRL2DEnv

    # Use same class for both, but noisy uses separate RNG for obs noise
    EnvClean = AdvRL2DEnv
    EnvNoisy = make_env_with_separate_obs_rng(AdvRL2DEnv)

    # Base env config: match your training main() (adjust if your AdvRL_wind.py differs)
    base_cfg = AdvRLEnvConfig(
        obs_noise_std=0.0,     # overridden per scenario
        proc_noise_std=0.0,
        wind_epsilon=0.9,
        wind_volatility=0.25,
        seed=2025,             # overridden per episode
        step_penalty=-1.0,
        success_reward=25.0,
        timeout_penalty=-25.0,
    )

    # Load trained model
    device = torch.device(DEVICE)
    model = ActorCritic(obs_dim=4, hidden=128, act_dim=2, std_fixed=0.35).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[load] Policy loaded from: {MODEL_PATH}")
    print(f"[cfg] DEVICE={DEVICE} | NOISE_STD={NOISE_STD}")
    print(f"[out] RESULTS_DIR={RESULTS_DIR}")

    plot_4eps_row_clean_vs_noisy(
        AdvRLEnvConfig=AdvRLEnvConfig,
        EnvCleanClass=EnvClean,
        EnvNoisyClass=EnvNoisy,
        base_cfg=base_cfg,
        model=model,
        noise_std=NOISE_STD,
        seeds_4=SEEDS_4,
        device=DEVICE,
        results_dir=RESULTS_DIR,
    )

    plot_accumulated_reward_clean_vs_noisy(
        AdvRLEnvConfig=AdvRLEnvConfig,
        EnvCleanClass=EnvClean,
        EnvNoisyClass=EnvNoisy,
        base_cfg=base_cfg,
        model=model,
        noise_std=NOISE_STD,
        n_episodes=N_RETURNS_EPISODES,
        seed0=RETURNS_SEED0,
        device=DEVICE,
        results_dir=RESULTS_DIR,
    )

    print("[done] Evaluation finished.")


if __name__ == "__main__":
    main()