#!/usr/bin/env python3
"""
Evaluate clean, noisy, and Kalman-filtered observations for the wind policy.

The filter estimates the augmented position state x_t = [p_x, p_y, 1]^T under:

    x_{t+1} = A_t x_t + B a_t + w_t
    y_t     = F x_t + v_t

with A_t determined by the known wind at the current step. The policy receives
z_t = [(goal - p_hat_t) / goal_r_max, wind_x, wind_y], where p_hat_t is the
filtered position estimate. This isolates whether KF smoothing of noisy
position measurements helps the trained policy recover clean performance.
"""

# Eval_clean_noisy_kf.py
#
# HARD-CODED PATHS:
#   - AdvRL_wind.py    : ./AdvRL_wind.py  (same folder as this script)
#   - Model checkpoint : ../../saved_models/AdvRL_v2_policy.pt
#   - Output figures   : ../../results/
#
# What it does:
#   1) Evaluate accumulated reward across episodes for:
#        - clean observation
#        - noisy observation
#        - noisy observation + Kalman filtering on position
#   2) Save a single plot with the 3 cumulative-reward curves
#
# Notes:
#   - Noise is only on measured position, not on wind.
#   - For KF mode, the policy receives:
#         z_t = [ (goal - p_hat_t)/goal_r_max , wind_x_t, wind_y_t ]
#     where p_hat_t is the filtered position estimate.
#   - The wind is assumed known correctly.
#
# Position SSM used by the KF (augmented formulation):
#   x_t = [p_x, p_y, 1]^T
#
#   x_{t+1} = A_t x_t + B a_t + w_t
#   y_t     = F x_t + v_t
#
# with
#   A_t = [[1, 0, wind_x_t],
#          [0, 1, wind_y_t],
#          [0, 0, 1       ]]
#
#   B   = [[1, 0],
#          [0, 1],
#          [0, 0]]
#
#   F   = [[1, 0, 0],
#          [0, 1, 0]]
#
# The tiny KF process variance floor is only for numerical stability when the
# environment process noise is zero.

from __future__ import annotations

import os
import sys
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

# Accumulated-reward plot settings
N_EPISODES = 500
SEED0 = 10_000

# KF numerical settings
KF_Q_FLOOR = 1e-6       # tiny floor so covariance does not collapse completely
KF_INIT_VAR_SCALE = 1.0 # initial position variance = scale * R

# -------------------------
# HARD-CODED PATHS (relative to this script)
# -------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../.."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from AdvSSM.io_utils import data_path_for_plot, load_npz, save_npz

ADV_FILE_PATH = os.path.abspath(os.path.join(_THIS_DIR, "AdvRL_wind.py"))
MODEL_PATH = os.path.abspath(os.path.join(_THIS_DIR, "../../saved_models/AdvRL_v2_policy.pt"))
RESULTS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "../../results"))


# =========================
# Env wrapper: separate RNG for obs noise only
# (so obs noise does NOT consume the main env RNG used by goal/wind)
# =========================

def make_env_with_separate_obs_rng(AdvRL2DEnv_base):
    class AdvRL2DEnv_SeparateObsRNG(AdvRL2DEnv_base):
        """
        Same env as AdvRL2DEnv, but observation noise uses rng_obs so it does NOT
        consume randomness from the env's main RNG.
        """

        def __init__(self, cfg):
            self.rng_obs = np.random.default_rng(int(cfg.seed) + 12345)
            super().__init__(cfg)

        def _measure_position(self) -> np.ndarray:
            y = (self.F @ self.x_ssm).astype(np.float32)
            if self.cfg.obs_noise_std > 0:
                v = self.rng_obs.normal(
                    0.0, self.cfg.obs_noise_std, size=(2,)
                ).astype(np.float32)
                y = y + v
            return y.astype(np.float32)

    return AdvRL2DEnv_SeparateObsRNG


# =========================
# Helpers
# =========================

def obs_to_measured_position(obs: np.ndarray, goal: np.ndarray, goal_r_max: float) -> np.ndarray:
    """
    Recover measured position y_t from the RL observation:
        z_t[:2] = (goal - y_t) / goal_r_max
    so
        y_t = goal - z_t[:2] * goal_r_max
    """
    delta = np.asarray(obs[:2], dtype=np.float32)
    y_meas = goal - delta * float(goal_r_max)
    return y_meas.astype(np.float32)


def build_policy_obs_from_position(
    pos_est: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
    wind_xy: np.ndarray,
) -> np.ndarray:
    """
    Build the 4D observation expected by the trained policy:
        [ (goal - pos_est)/goal_r_max , wind_x, wind_y ]
    """
    delta_est = (goal - pos_est) / float(goal_r_max)
    z = np.array(
        [delta_est[0], delta_est[1], wind_xy[0], wind_xy[1]],
        dtype=np.float32,
    )
    return z


def kf_predict(
    x_hat: np.ndarray,
    P: np.ndarray,
    action: np.ndarray,
    wind_xy: np.ndarray,
    q_pos_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    KF predict step for the augmented state x = [p_x, p_y, 1]^T.
    """
    A = np.array(
        [
            [1.0, 0.0, float(wind_xy[0])],
            [0.0, 1.0, float(wind_xy[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    B = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    Q = np.diag([q_pos_var, q_pos_var, 0.0]).astype(np.float32)

    x_pred = A @ x_hat + B @ action
    P_pred = A @ P @ A.T + Q
    return x_pred.astype(np.float32), P_pred.astype(np.float32)


def kf_update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    y_meas: np.ndarray,
    r_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    KF update step for:
        y_t = F x_t + v_t
    with
        F = [[1,0,0],
             [0,1,0]]
    """
    F = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    R = np.diag([r_var, r_var]).astype(np.float32)

    innov = y_meas - (F @ x_pred)
    S = F @ P_pred @ F.T + R
    K = P_pred @ F.T @ np.linalg.inv(S)

    x_hat = x_pred + K @ innov
    P = (np.eye(3, dtype=np.float32) - K @ F) @ P_pred

    # Keep the augmented constant state anchored at 1 numerically
    x_hat[2] = 1.0

    return x_hat.astype(np.float32), P.astype(np.float32)


# =========================
# Rollouts
# =========================

@torch.no_grad()
def rollout_episode_return_standard(env, model, device: str = "cpu") -> float:
    """
    Standard deterministic evaluation using the raw env observation.
    Returns only the episode return.
    """
    dev = torch.device(device)
    obs = env.reset()
    ep_return = 0.0

    for _ in range(env.cfg.max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        obs, reward, done, _info = env.step(action)
        ep_return += float(reward)

        if done:
            break

    return float(ep_return)


@torch.no_grad()
def rollout_episode_return_kf(env, model, device: str = "cpu") -> float:
    """
    Deterministic evaluation where the environment observation is noisy,
    but the policy acts on a Kalman-filtered estimate of position.

    The wind is assumed known correctly and is passed directly to the policy.
    """
    dev = torch.device(device)

    obs = env.reset()
    goal = env.goal.copy()
    goal_r_max = float(env.cfg.goal_r_max)

    # Measurement noise variance
    r_var = float(env.cfg.obs_noise_std ** 2)

    # Small process variance floor for numerical stability
    q_pos_var = max(float(env.cfg.proc_noise_std ** 2), KF_Q_FLOOR)

    # Initial state estimate from the first measurement
    y0 = obs_to_measured_position(obs, goal, goal_r_max)
    x_hat = np.array([y0[0], y0[1], 1.0], dtype=np.float32)
    P = np.diag(
        [
            KF_INIT_VAR_SCALE * max(r_var, 1e-8),
            KF_INIT_VAR_SCALE * max(r_var, 1e-8),
            1e-10,
        ]
    ).astype(np.float32)

    ep_return = 0.0

    for _ in range(env.cfg.max_steps):
        # Current wind is known correctly
        wind_xy = np.asarray(obs[2:4], dtype=np.float32)

        # Filtered position estimate used by the policy
        pos_hat = x_hat[:2].copy()
        obs_filt = build_policy_obs_from_position(pos_hat, goal, goal_r_max, wind_xy)

        obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        # Predict next hidden state using current known wind and chosen action
        x_pred, P_pred = kf_predict(
            x_hat=x_hat,
            P=P,
            action=action,
            wind_xy=wind_xy,
            q_pos_var=q_pos_var,
        )

        # Step the real env (which returns the next noisy observation)
        obs_next, reward, done, _info = env.step(action)
        ep_return += float(reward)

        if done:
            break

        # Update with next noisy position measurement
        y_next = obs_to_measured_position(obs_next, goal, goal_r_max)
        x_hat, P = kf_update(
            x_pred=x_pred,
            P_pred=P_pred,
            y_meas=y_next,
            r_var=max(r_var, 1e-8),
        )

        obs = obs_next

    return float(ep_return)


# =========================
# Plot
# =========================

def plot_accumulated_reward_clean_noisy_kf(
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
    Plot cumulative sum of episode returns across episodes for:
      - clean
      - noisy
      - noisy + KF
    """
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, "eval_accumulated_reward_clean_noisy_kf.png")
    data_path = data_path_for_plot(outpath)

    if os.path.exists(data_path):
        print(f"[cache] loading data: {data_path}")
        data = load_npz(data_path)
        acc_clean = np.asarray(data["acc_clean"], dtype=float)
        acc_noisy = np.asarray(data["acc_noisy"], dtype=float)
        acc_kf = np.asarray(data["acc_kf"], dtype=float)

        fig = plt.figure(figsize=(12, 5))
        plt.plot(acc_clean, linewidth=1.8, label="clean")
        plt.plot(acc_noisy, linewidth=1.8, label=f"noisy (sigma={noise_std})")
        plt.plot(acc_kf, linewidth=1.8, label=f"noisy + KF (sigma={noise_std})")

        plt.grid(True, alpha=0.25)
        plt.xlabel("Episode")
        plt.ylabel("Accumulated reward (cumulative sum)")
        plt.title("Accumulated reward across episodes: clean vs noisy vs noisy+KF")
        plt.legend()

        fig.savefig(outpath, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {outpath}")
        return

    acc_clean = []
    acc_noisy = []
    acc_kf = []

    total_clean = 0.0
    total_noisy = 0.0
    total_kf = 0.0

    for k in range(n_episodes):
        seed = int(seed0 + k)

        cfg_clean = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_noisy = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_kf = AdvRLEnvConfig(**{**asdict(base_cfg)})

        cfg_clean.seed = seed
        cfg_clean.obs_noise_std = 0.0

        cfg_noisy.seed = seed
        cfg_noisy.obs_noise_std = float(noise_std)

        cfg_kf.seed = seed
        cfg_kf.obs_noise_std = float(noise_std)

        env_clean = EnvCleanClass(cfg_clean)
        env_noisy = EnvNoisyClass(cfg_noisy)
        env_kf = EnvNoisyClass(cfg_kf)

        ret_clean = rollout_episode_return_standard(env_clean, model, device=device)
        ret_noisy = rollout_episode_return_standard(env_noisy, model, device=device)
        ret_kf = rollout_episode_return_kf(env_kf, model, device=device)

        total_clean += float(ret_clean)
        total_noisy += float(ret_noisy)
        total_kf += float(ret_kf)

        acc_clean.append(total_clean)
        acc_noisy.append(total_noisy)
        acc_kf.append(total_kf)

        if (k + 1) % 50 == 0 or (k + 1) == n_episodes:
            print(
                f"[{k+1:4d}/{n_episodes}] "
                f"clean={total_clean:.1f} | noisy={total_noisy:.1f} | kf={total_kf:.1f}"
            )

    fig = plt.figure(figsize=(12, 5))
    plt.plot(acc_clean, linewidth=1.8, label="clean")
    plt.plot(acc_noisy, linewidth=1.8, label=f"noisy (σ={noise_std})")
    plt.plot(acc_kf, linewidth=1.8, label=f"noisy + KF (σ={noise_std})")

    plt.grid(True, alpha=0.25)
    plt.xlabel("Episode")
    plt.ylabel("Accumulated reward (cumulative sum)")
    plt.title("Accumulated reward across episodes: clean vs noisy vs noisy+KF")
    plt.legend()

    save_npz(
        data_path,
        acc_clean=np.asarray(acc_clean, dtype=float),
        acc_noisy=np.asarray(acc_noisy, dtype=float),
        acc_kf=np.asarray(acc_kf, dtype=float),
        noise_std=np.asarray(noise_std, dtype=float),
        n_episodes=np.asarray(n_episodes, dtype=int),
        seed0=np.asarray(seed0, dtype=int),
    )
    print(f"[cache] saved data: {data_path}")
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


# =========================
# Main
# =========================

def main():
    if not os.path.exists(ADV_FILE_PATH):
        raise FileNotFoundError(f"AdvRL_wind.py not found at:\n  {ADV_FILE_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at:\n  {MODEL_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Normal import from the same folder
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)

    import AdvRL_wind as mod
    print(f"[import] Imported AdvRL_wind from: {ADV_FILE_PATH}")

    ActorCritic = mod.ActorCritic
    AdvRLEnvConfig = mod.AdvRLEnvConfig
    AdvRL2DEnv = mod.AdvRL2DEnv

    EnvClean = AdvRL2DEnv
    EnvNoisy = make_env_with_separate_obs_rng(AdvRL2DEnv)

    # Match your training/evaluation config here if needed
    base_cfg = AdvRLEnvConfig(
        obs_noise_std=0.0,     # overridden below
        proc_noise_std=0.0,
        wind_epsilon=0.9,
        wind_volatility=0.25,
        seed=2025,             # overridden per episode
        step_penalty=-1.0,
        success_reward=25.0,
        timeout_penalty=-25.0,
    )

    device = torch.device(DEVICE)
    model = ActorCritic(obs_dim=4, hidden=128, act_dim=2, std_fixed=0.35).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[load] Policy loaded from: {MODEL_PATH}")
    print(f"[cfg] DEVICE={DEVICE} | NOISE_STD={NOISE_STD}")
    print(f"[out] RESULTS_DIR={RESULTS_DIR}")

    plot_accumulated_reward_clean_noisy_kf(
        AdvRLEnvConfig=AdvRLEnvConfig,
        EnvCleanClass=EnvClean,
        EnvNoisyClass=EnvNoisy,
        base_cfg=base_cfg,
        model=model,
        noise_std=NOISE_STD,
        n_episodes=N_EPISODES,
        seed0=SEED0,
        device=DEVICE,
        results_dir=RESULTS_DIR,
    )

    print("[done] Evaluation finished.")


if __name__ == "__main__":
    main()
