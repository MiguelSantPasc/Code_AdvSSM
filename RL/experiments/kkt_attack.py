# RL/experiments/kkt_attack.py
# 3-agent rollout comparison with a non-compounding KKT observation attack.
#
# Agents (each runs its own episode/world, same loaded policy weights):
#   1) CLEAN: z_t = true_delta_t
#   2) NOISY: z_t = true_delta_t + N(0, sigma^2 I) and KF -> policy
#   3) ADV  : attacked observations z_adv with probability P_ATTACK, else z_real (noisy).
#        - shadow KF uses z_real always to compute attack ellipsoid (prevents compounding)
#        - attacked KF uses z_adv to choose actions
#        - objective uses TRUE delta at current state (oracle), ellipsoid from shadow predicted belief
#      Attack starts from the 2nd observation (t >= 1).
#
# Threat model / optimization (delta-space, H=I):
#   Predicted belief (shadow): mu = delta_hat_{t|t-1},  S = P_{t|t-1} + R,  K = P_{t|t-1} S^{-1}
#   Constraint (plausible): (z - mu)^T S^{-1} (z - mu) <= epsilon
#   Objective (oracle truth): maximize || K (z - delta_true) ||^2
#
# Outputs:
#   - RL/results/kkt_10_simulations_3agents.png
#   - RL/results/kkt_rewards_over_episodes.png

from __future__ import annotations

import os
import sys
from typing import Tuple, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DEVICE = "cpu"
SEED = 42

# 10-sim grid figure
N_SIMS = 10
GRID_ROWS, GRID_COLS = 2, 5

# Reward figure
EVAL_EPISODES = 200         # "several iterations"
RUNNING_MEAN_W = 20         # running mean window

# Noise settings
SIGMA_OBS = 1.0             # observation noise used for NOISY and for ADV "real sensor"
PROC_NOISE_STD = 0.05       # environment process noise

# Attack settings
EPSILON_KKT = 10.991         # Chi-square df=2, 95%
P_ATTACK = 0.50              # attack applied with probability p each step (from 2nd obs), else normal noise

# Environment
GOAL_R_MIN = 5.0
GOAL_R_MAX = 25.0
MAX_STEPS = 40
GOAL_RADIUS = 1.5

MODEL_PATH = os.path.join("RL", "saved_models", "AdvRL_policy.pt")
OUT_DIR = os.path.join("RL", "results")

# Numerical floors
R_VAR_FLOOR = 1e-10
Q_VAR_FLOOR = 1e-10


# ------------------------------------------------------------
# Project imports (robust)
# ------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from AdvRL import AdvRLEnvConfig, AdvRL2DEnv, ActorCritic  # type: ignore


# ------------------------------------------------------------
# Helpers: PSD + KKT solver (trust-region / ellipsoid)
# ------------------------------------------------------------
def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def project_to_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(M)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(w) @ V.T

def sqrtm_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(M)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(np.sqrt(w)) @ V.T

def solve_kkt_max_quadratic_over_ellipsoid(
    *,
    X: np.ndarray,          # (n_x, n_y)
    y_t: np.ndarray,        # (n_y,)
    mu: np.ndarray,         # (n_y,)
    Sigma: np.ndarray,      # (n_y, n_y)
    epsilon: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """
    Solve:
        maximize_y  || X (y - y_t) ||^2
        subject to  (y - mu)^T Sigma^{-1} (y - mu) <= epsilon

    Returns (y_star, obj_star).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    Sigma = project_to_psd(np.asarray(Sigma, dtype=float))
    S = sqrtm_psd(Sigma)
    X = np.asarray(X, dtype=float)
    y_t = np.asarray(y_t, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)

    M = X.T @ X
    d = (mu - y_t).reshape(-1)
    A = symmetrize(S.T @ M @ S)
    b = (S.T @ M @ d).reshape(-1)

    a, U = np.linalg.eigh(A)
    a_max = float(np.max(a))
    bp = U.T @ b

    if np.linalg.norm(b) < 1e-14:
        idx = int(np.argmax(a))
        z_star = np.zeros_like(b)
        z_star[idx] = np.sqrt(epsilon)
        z_star = U @ z_star
    else:
        def norm2_minus_eps(lam: float) -> float:
            zi = -bp / (a - lam)
            return float(np.dot(zi, zi) - epsilon)

        lam_low = a_max + 1e-12
        f_low = norm2_minus_eps(lam_low)
        if f_low <= 0:
            lam_low = a_max + 1e-16
            f_low = norm2_minus_eps(lam_low)

        lam_high = a_max + 1.0
        f_high = norm2_minus_eps(lam_high)
        while f_high > 0:
            lam_high *= 2.0
            f_high = norm2_minus_eps(lam_high)
            if lam_high > 1e12:
                raise RuntimeError("Failed to bracket lambda.")

        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_low + lam_high)
            f_mid = norm2_minus_eps(lam_mid)
            if abs(f_mid) < tol:
                lam_low = lam_high = lam_mid
                break
            if f_mid > 0:
                lam_low = lam_mid
            else:
                lam_high = lam_mid

        lam_star = 0.5 * (lam_low + lam_high)
        z_star = U @ (-bp / (a - lam_star))

        nz = np.linalg.norm(z_star)
        if nz > 0:
            z_star = z_star * (np.sqrt(epsilon) / nz)

    y_star = mu + S @ z_star
    obj_star = float(np.linalg.norm(X @ (y_star - y_t)) ** 2)
    return y_star, obj_star


# ------------------------------------------------------------
# KF in delta-space: delta_t = goal - x_t, H = I
# ------------------------------------------------------------
def unit_dir(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)

def kf_predict(delta_hat: np.ndarray, P: np.ndarray, u: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # delta_{t+1} = delta_t - u_t + w
    delta_hat = delta_hat - u
    P = P + Q
    return delta_hat, P

def kf_update(delta_hat: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # H = I
    S = P + R
    K = P @ np.linalg.inv(S)
    delta_hat_new = delta_hat + K @ (z - delta_hat)
    P_new = (np.eye(2, dtype=np.float32) - K) @ P
    return delta_hat_new, P_new


# ------------------------------------------------------------
# One rollout per agent (returns trajectory + episode return + success)
# ------------------------------------------------------------
@torch.no_grad()
def rollout_clean(model: ActorCritic, device: torch.device, env: AdvRL2DEnv, R: np.ndarray, Q: np.ndarray):
    traj_true = [env.x.copy()]
    done = False
    ep_ret = 0.0
    success = False

    true_delta = (env.goal - env.x).astype(np.float32)
    delta_hat = true_delta.copy()
    P = R.copy()

    for _ in range(env.cfg.max_steps):
        z = (env.goal - env.x).astype(np.float32)  # clean
        delta_hat, P = kf_update(delta_hat, P, z, R)

        obs_policy = unit_dir(delta_hat)
        obs_t = torch.tensor(obs_policy, dtype=torch.float32, device=device)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        theta = float(action[0])
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        _, r, done, info = env.step(action)
        ep_ret += float(r)
        success = bool(info.get("success", False))

        traj_true.append(env.x.copy())
        if done:
            break

        delta_hat, P = kf_predict(delta_hat, P, u, Q)

    return np.array(traj_true, dtype=np.float32), ep_ret, success


@torch.no_grad()
def rollout_noisy(model: ActorCritic, device: torch.device, env: AdvRL2DEnv, R: np.ndarray, Q: np.ndarray,
                  rng: np.random.Generator, sigma: float):
    traj_true = [env.x.copy()]
    done = False
    ep_ret = 0.0
    success = False

    true_delta = (env.goal - env.x).astype(np.float32)
    z = true_delta + rng.normal(0.0, sigma, size=(2,)).astype(np.float32)
    delta_hat = z.copy()
    P = R.copy()

    for _ in range(env.cfg.max_steps):
        true_delta = (env.goal - env.x).astype(np.float32)
        z = true_delta + rng.normal(0.0, sigma, size=(2,)).astype(np.float32)

        delta_hat, P = kf_update(delta_hat, P, z, R)

        obs_policy = unit_dir(delta_hat)
        obs_t = torch.tensor(obs_policy, dtype=torch.float32, device=device)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        theta = float(action[0])
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        _, r, done, info = env.step(action)
        ep_ret += float(r)
        success = bool(info.get("success", False))

        traj_true.append(env.x.copy())
        if done:
            break

        delta_hat, P = kf_predict(delta_hat, P, u, Q)

    return np.array(traj_true, dtype=np.float32), ep_ret, success


@torch.no_grad()
def rollout_adv(
    model: ActorCritic,
    device: torch.device,
    env: AdvRL2DEnv,
    R: np.ndarray,
    Q: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    epsilon: float,
    p_attack: float,
):
    """
    ADV agent:
      - shadow KF uses REAL noisy z_real always (prevents compounding)
      - attacked KF uses z_adv; with prob p_attack (from step>=1) do KKT, else z_adv=z_real
      - objective uses TRUE delta at current state (oracle), ellipsoid from shadow predicted belief
      - record:
          traj_true: env.x
          traj_belief: x_hat = goal - delta_hat_adv
    """
    traj_true = [env.x.copy()]
    traj_belief: List[np.ndarray] = []
    done = False
    ep_ret = 0.0
    success = False

    # Init both KFs with first real noisy measurement
    true_delta = (env.goal - env.x).astype(np.float32)
    z_real = true_delta + rng.normal(0.0, sigma, size=(2,)).astype(np.float32)

    delta_hat_shadow = z_real.copy()
    P_shadow = R.copy()

    delta_hat_adv = z_real.copy()
    P_adv = R.copy()

    for step_idx in range(env.cfg.max_steps):
        true_delta = (env.goal - env.x).astype(np.float32)

        # real sensor
        z_real = true_delta + rng.normal(0.0, sigma, size=(2,)).astype(np.float32)

        # shadow predicted -> ellipsoid params
        S_pred = P_shadow + R
        K_pred = P_shadow @ np.linalg.inv(S_pred)

        do_attack = (step_idx > 1) and (rng.random() < p_attack)

        if not do_attack:
            z_adv = z_real.copy()
        else:
            try:
                z_adv64, _ = solve_kkt_max_quadratic_over_ellipsoid(
                    X=K_pred,
                    y_t=true_delta,          # objective reference: TRUE delta (oracle)
                    mu=delta_hat_shadow,     # ellipsoid center: shadow predicted mean
                    Sigma=S_pred,            # ellipsoid shape: innovation covariance
                    epsilon=epsilon,
                )
                z_adv = z_adv64.astype(np.float32)
            except Exception:
                z_adv = z_real.copy()

        # update shadow with real sensor
        delta_hat_shadow, P_shadow = kf_update(delta_hat_shadow, P_shadow, z_real, R)

        # update attacked filter with attacked/noisy obs
        delta_hat_adv, P_adv = kf_update(delta_hat_adv, P_adv, z_adv, R)

        # believed position
        x_hat = (env.goal - delta_hat_adv).astype(np.float32)
        traj_belief.append(x_hat.copy())

        # policy uses attacked belief
        obs_policy = unit_dir(delta_hat_adv)
        obs_t = torch.tensor(obs_policy, dtype=torch.float32, device=device)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        theta = float(action[0])
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        _, r, done, info = env.step(action)
        ep_ret += float(r)
        success = bool(info.get("success", False))

        traj_true.append(env.x.copy())
        if done:
            break

        # predict both
        delta_hat_shadow, P_shadow = kf_predict(delta_hat_shadow, P_shadow, u, Q)
        delta_hat_adv, P_adv = kf_predict(delta_hat_adv, P_adv, u, Q)

    traj_true_arr = np.array(traj_true, dtype=np.float32)
    traj_belief_arr = np.array(traj_belief, dtype=np.float32)

    # pad belief to same length
    if traj_belief_arr.shape[0] < traj_true_arr.shape[0]:
        if traj_belief_arr.shape[0] > 0:
            last = traj_belief_arr[-1]
            pad = np.repeat(last[None, :], traj_true_arr.shape[0] - traj_belief_arr.shape[0], axis=0)
            traj_belief_arr = np.vstack([traj_belief_arr, pad])
        else:
            traj_belief_arr = np.repeat(env.x[None, :], traj_true_arr.shape[0], axis=0)

    return traj_true_arr, traj_belief_arr, ep_ret, success


# ------------------------------------------------------------
# Plotting: cream + stronger pastel palette
# ------------------------------------------------------------
def _set_rcparams():
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.labelsize": 11.0,
        "legend.fontsize": 9.8,
        "axes.linewidth": 1.0,
    })

def plot_10_simulations_3agents(model: ActorCritic, device: torch.device, sigma: float) -> None:
    _set_rcparams()

    cream = "#F7F1E3"
    cream2 = "#FBF7EE"
    ink = "#1F1F1F"
    grid = "#D9D2C3"

    # Stronger-but-pastel palette
    c_clean = "#2F2F2F"   # charcoal
    c_noisy = "#2F6FA8"   # stronger pastel blue
    c_adv = "#C85B4F"     # stronger pastel terracotta
    c_goal = "#D4A017"    # warm gold

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(20, 8), constrained_layout=True)
    fig.patch.set_facecolor(cream)
    axes = axes.flatten()

    # KF covariances
    r_var = max(float(sigma) ** 2, R_VAR_FLOOR)
    q_var = max(float(PROC_NOISE_STD) ** 2, Q_VAR_FLOOR)
    R = (r_var * np.eye(2)).astype(np.float32)
    Q = (q_var * np.eye(2)).astype(np.float32)

    lim = max(GOAL_R_MAX, MAX_STEPS) + 3.0

    for i in range(N_SIMS):
        ax = axes[i]
        ax.set_facecolor(cream2)
        ax.grid(True, color=grid, alpha=0.65, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BFB8A8")
        ax.spines["bottom"].set_color("#BFB8A8")
        ax.tick_params(colors=ink, labelsize=9)

        # IMPORTANT: limits first, then aspect "box" to avoid warnings
        ax.set_aspect("equal", adjustable="box")

        ep_seed = SEED + 1000 + i
        rng_noisy = np.random.default_rng(ep_seed + 22)
        rng_adv = np.random.default_rng(ep_seed + 33)

        cfg = AdvRLEnvConfig(
            goal_r_min=GOAL_R_MIN,
            goal_r_max=GOAL_R_MAX,
            obs_noise_std=sigma,
            proc_noise_std=PROC_NOISE_STD,
            max_steps=MAX_STEPS,
            goal_radius=GOAL_RADIUS,
            seed=ep_seed,
        )

        env_clean = AdvRL2DEnv(cfg)
        env_noisy = AdvRL2DEnv(cfg)
        env_adv = AdvRL2DEnv(cfg)

        env_clean.reset()
        goal = env_clean.goal.copy()

        env_noisy.reset()
        env_noisy.goal = goal.copy()

        env_adv.reset()
        env_adv.goal = goal.copy()

        traj_clean, _, succ_clean = rollout_clean(model, device, env_clean, R, Q)
        traj_noisy, _, succ_noisy = rollout_noisy(model, device, env_noisy, R, Q, rng_noisy, sigma)
        traj_adv_true, traj_adv_belief, _, succ_adv = rollout_adv(
            model, device, env_adv, R, Q, rng_adv, sigma, EPSILON_KKT, P_ATTACK
        )

        # trajectories
        ax.plot(traj_clean[:, 0], traj_clean[:, 1],
                color=c_clean, linewidth=1.8,
                label="Clean (true)" if i == 0 else None)

        ax.plot(traj_noisy[:, 0], traj_noisy[:, 1],
                color=c_noisy, linewidth=1.8,
                label="Noisy+KF (true)" if i == 0 else None)

        ax.plot(traj_adv_true[:, 0], traj_adv_true[:, 1],
                color=c_adv, linewidth=2.0, linestyle="--",
                label="Adv+KF (true)" if i == 0 else None)

        ax.plot(traj_adv_belief[:, 0], traj_adv_belief[:, 1],
                color=c_adv, linewidth=1.8, linestyle=":",
                label="Adv belief (pos est.)" if i == 0 else None)

        # start
        ax.scatter([0.0], [0.0], s=18, marker="s", color=ink, alpha=0.85)

        # goal + success circle
        ax.scatter([goal[0]], [goal[1]], s=85, marker="*", color=c_goal, zorder=6)
        ax.add_patch(plt.Circle((goal[0], goal[1]), GOAL_RADIUS,
                                color=c_goal, fill=False, linestyle="--",
                                linewidth=1.4, alpha=0.75))

        # title with outcomes
        ax.set_title(f"Sim {i+1} | C:{'OK' if succ_clean else 'X'}  N:{'OK' if succ_noisy else 'X'}  A:{'OK' if succ_adv else 'X'}",
                     color=ink)

        if i == 0:
            ax.legend(loc="upper left", frameon=False)

    for j in range(N_SIMS, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"3-agent comparison (σ={sigma}, ε={EPSILON_KKT}, p_attack={P_ATTACK}, steps≤{MAX_STEPS}, success r={GOAL_RADIUS})",
        fontsize=13, color=ink
    )

    os.makedirs(os.path.join(_PROJECT_ROOT, OUT_DIR), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, OUT_DIR, "kkt_10_simulations_3agents.png")
    plt.savefig(out_path, dpi=240, facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    plt.close(fig)


def _running_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    w = int(w)
    if x.size < w:
        return np.full_like(x, np.mean(x))
    kernel = np.ones(w) / w
    y = np.convolve(x, kernel, mode="valid")
    # pad to match length (align to center-ish; simplest: left pad)
    pad = np.full(w - 1, y[0])
    return np.concatenate([pad, y])


def plot_rewards_over_episodes(model: ActorCritic, device: torch.device, sigma: float) -> None:
    _set_rcparams()

    cream = "#F7F1E3"
    cream2 = "#FBF7EE"
    ink = "#1F1F1F"
    grid = "#D9D2C3"

    # Stronger-but-pastel palette
    c_clean = "#2F2F2F"
    c_noisy = "#2F6FA8"
    c_adv = "#C85B4F"

    # KF covariances
    r_var = max(float(sigma) ** 2, R_VAR_FLOOR)
    q_var = max(float(PROC_NOISE_STD) ** 2, Q_VAR_FLOOR)
    R = (r_var * np.eye(2)).astype(np.float32)
    Q = (q_var * np.eye(2)).astype(np.float32)

    rets_clean, rets_noisy, rets_adv = [], [], []
    succ_clean = succ_noisy = succ_adv = 0

    for ep in range(EVAL_EPISODES):
        ep_seed = SEED + 5000 + ep
        rng_noisy = np.random.default_rng(ep_seed + 11)
        rng_adv = np.random.default_rng(ep_seed + 22)

        cfg = AdvRLEnvConfig(
            goal_r_min=GOAL_R_MIN,
            goal_r_max=GOAL_R_MAX,
            obs_noise_std=sigma,
            proc_noise_std=PROC_NOISE_STD,
            max_steps=MAX_STEPS,
            goal_radius=GOAL_RADIUS,
            seed=ep_seed,
        )

        env_clean = AdvRL2DEnv(cfg)
        env_noisy = AdvRL2DEnv(cfg)
        env_adv = AdvRL2DEnv(cfg)

        env_clean.reset()
        goal = env_clean.goal.copy()

        env_noisy.reset()
        env_noisy.goal = goal.copy()

        env_adv.reset()
        env_adv.goal = goal.copy()

        _, ret_c, ok_c = rollout_clean(model, device, env_clean, R, Q)
        _, ret_n, ok_n = rollout_noisy(model, device, env_noisy, R, Q, rng_noisy, sigma)
        _, _, ret_a, ok_a = rollout_adv(model, device, env_adv, R, Q, rng_adv, sigma, EPSILON_KKT, P_ATTACK)

        rets_clean.append(ret_c)
        rets_noisy.append(ret_n)
        rets_adv.append(ret_a)

        succ_clean += int(ok_c)
        succ_noisy += int(ok_n)
        succ_adv += int(ok_a)

    rets_clean = np.array(rets_clean, dtype=float)
    rets_noisy = np.array(rets_noisy, dtype=float)
    rets_adv = np.array(rets_adv, dtype=float)

    acc_clean = np.cumsum(rets_clean)
    acc_noisy = np.cumsum(rets_noisy)
    acc_adv = np.cumsum(rets_adv)

    x = np.arange(1, EVAL_EPISODES + 1)

    fig, axes = plt.subplots(
        2, 1, figsize=(11.2, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
        constrained_layout=True
    )
    fig.patch.set_facecolor(cream)

    for ax in axes:
        ax.set_facecolor(cream2)
        ax.grid(True, which="major", color=grid, linewidth=0.9, alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BFB8A8")
        ax.spines["bottom"].set_color("#BFB8A8")
        ax.tick_params(colors=ink)

    # ---- Panel 1: Episode return
    ax0 = axes[0]
    ax0.plot(x, rets_clean, color=c_clean, linewidth=1.9, label="Clean")
    ax0.plot(x, rets_noisy, color=c_noisy, linewidth=1.9, label="Noisy+KF")
    ax0.plot(x, rets_adv, color=c_adv, linewidth=2.1, label="Adv+KF")
    ax0.set_ylabel("Episode return", color=ink)
    ax0.legend(frameon=False, ncol=3, loc="best")

    # ---- Panel 2: Accumulated return
    ax1 = axes[1]
    ax1.plot(x, acc_clean, color=c_clean, linewidth=2.3, label="Clean (accum.)")
    ax1.plot(x, acc_noisy, color=c_noisy, linewidth=2.3, label="Noisy+KF (accum.)")
    ax1.plot(x, acc_adv, color=c_adv, linewidth=2.5, label="Adv+KF (accum.)")
    ax1.set_xlabel("Episode", color=ink)
    ax1.set_ylabel("Accumulated return", color=ink)

    fig.suptitle(
        f"Reward over episodes (σ={sigma}, ε={EPSILON_KKT}, p_attack={P_ATTACK})\n"
        f"Success rate: Clean={succ_clean/EVAL_EPISODES:.2f}, "
        f"Noisy={succ_noisy/EVAL_EPISODES:.2f}, Adv={succ_adv/EVAL_EPISODES:.2f}",
        color=ink, fontsize=13
    )

    os.makedirs(os.path.join(_PROJECT_ROOT, OUT_DIR), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, OUT_DIR, "kkt_rewards_over_episodes.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=260, facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    plt.close(fig)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    device = torch.device(DEVICE)

    model_abs = os.path.join(_PROJECT_ROOT, MODEL_PATH)
    if not os.path.exists(model_abs):
        raise FileNotFoundError(f"Model not found at: {model_abs}")

    model = ActorCritic(obs_dim=2, hidden=128).to(device)
    state = torch.load(model_abs, map_location=device)
    model.load_state_dict(state)
    model.eval()

    plot_10_simulations_3agents(model, device, sigma=SIGMA_OBS)
    plot_rewards_over_episodes(model, device, sigma=SIGMA_OBS)


if __name__ == "__main__":
    main()
