#!/usr/bin/env python3
# Plot_three_paths_clean_noisyKF_attackKF.py
#
# Plots in ONE figure the 3 trajectories:
#   1) clean
#   2) noisy + KF
#   3) attack + KF
#
# For each method it shows:
#   - TRUE path              : solid line
#   - OBSERVED / ESTIMATED   : dashed line
#   - ACTION arrows          : quiver arrows at decision points
#
# Important choice:
#   - The FIRST observation is NEVER corrupted:
#       * no noise at step 0 for noisy+KF
#       * no attack at step 0 for attack+KF
#   so all perceived trajectories start at the true initial point, typically (0,0).
#
# Output:
#   ../../results/three_paths_clean_noisyKF_attackKF.png

from __future__ import annotations

from matplotlib.patches import FancyArrowPatch
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# HARD-CODED CONFIG
# ============================================================

DEVICE = "cpu"

# Single episode seed to plot
PLOT_SEED = 22

# Random-noise case
NOISE_STD = 0.5

# Adversarial attack geometry
ATTACK_STD = NOISE_STD
ATTACK_EPS = 5.991
ATTACK_PROB = 0.5

# KF model
KF_MEAS_STD = NOISE_STD
KF_PROC_STD = 0.000003

# PGD / MC
PGD_STEPS = 500
PGD_STEP_SIZE = 0.25
MC_SAMPLES = 518

# Plot styling
ARROW_EVERY = 1
TRUE_ALPHA = 0.95
OBS_ALPHA = 0.90

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ADV_FILE_PATH = os.path.abspath(os.path.join(_THIS_DIR, "AdvRL_wind.py"))
MODEL_PATH = os.path.abspath(os.path.join(_THIS_DIR, "../../saved_models/AdvRL_v2_policy.pt"))
RESULTS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "../../results"))


# ============================================================
# Linear algebra helpers
# ============================================================

def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def project_to_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=np.float64))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return (V @ np.diag(w) @ V.T).astype(np.float32)


def sqrtm_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=np.float64))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return (V @ np.diag(np.sqrt(w)) @ V.T).astype(np.float32)


# ============================================================
# Exact Euclidean projection onto ellipsoid
# ============================================================

def project_to_attack_region(
    y_candidate: np.ndarray,
    center: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Exact Euclidean projection onto:
        { y : (y-center)^T Sigma^{-1} (y-center) <= epsilon }
    """
    y_candidate = np.asarray(y_candidate, dtype=np.float64).reshape(-1)
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    Sigma = project_to_psd(Sigma).astype(np.float64)

    diff = y_candidate - center
    Sinv = np.linalg.inv(Sigma)
    maha = float(diff.T @ Sinv @ diff)

    if maha <= epsilon + tol:
        return y_candidate.astype(np.float32)

    s, U = np.linalg.eigh(Sigma)
    s = np.maximum(s, 1e-12)
    r = U.T @ diff

    def f(lam: float) -> float:
        return float(np.sum((s * r**2) / (s + lam) ** 2) - epsilon)

    lam_low = 0.0
    lam_high = 1.0
    while f(lam_high) > 0:
        lam_high *= 2.0
        if lam_high > 1e14:
            raise RuntimeError("Could not bracket lambda in ellipsoid projection.")

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_low + lam_high)
        val = f(lam_mid)
        if abs(val) < tol:
            lam_low = lam_high = lam_mid
            break
        if val > 0:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    lam_star = 0.5 * (lam_low + lam_high)
    z = (s / (s + lam_star)) * r
    y_proj = center + U @ z
    return y_proj.astype(np.float32)


# ============================================================
# Env wrapper: separate RNG for observation noise only
# ============================================================

def make_env_with_separate_obs_rng(AdvRL2DEnv_base):
    class AdvRL2DEnv_SeparateObsRNG(AdvRL2DEnv_base):
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


# ============================================================
# Observation helpers
# ============================================================

def obs_to_position(obs: np.ndarray, goal: np.ndarray, goal_r_max: float) -> np.ndarray:
    """
    Recover measured position y_t from policy observation:
        obs[:2] = (goal - y_t) / goal_r_max
    """
    delta = np.asarray(obs[:2], dtype=np.float32)
    pos = goal - delta * float(goal_r_max)
    return pos.astype(np.float32)


def build_policy_obs_from_position(
    pos: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
    wind_xy: np.ndarray,
) -> np.ndarray:
    """
    Build policy observation:
        [ (goal - pos)/goal_r_max , wind_x, wind_y ]
    """
    delta = (goal - pos) / float(goal_r_max)
    return np.array(
        [delta[0], delta[1], wind_xy[0], wind_xy[1]],
        dtype=np.float32,
    )


# ============================================================
# Critic helper
# ============================================================

def critic_values(model, obs_batch: torch.Tensor) -> torch.Tensor:
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.unsqueeze(0)

    mu, std, v = model(obs_batch)

    if v.ndim == 2 and v.shape[1] == 1:
        v = v.squeeze(-1)
    return v


# ============================================================
# KF
# ============================================================

def kf_update_position(
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2, dtype=np.float32)

    S = P_pred + R
    K = P_pred @ np.linalg.inv(S)

    innov = y_obs - m_pred
    m_post = m_pred + K @ innov
    P_post = (I - K) @ P_pred
    P_post = project_to_psd(P_post)

    return m_post.astype(np.float32), P_post.astype(np.float32), K.astype(np.float32)


def kf_predict_position(
    m_post: np.ndarray,
    P_post: np.ndarray,
    action: np.ndarray,
    wind_xy: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    drift = np.asarray(action, dtype=np.float32) + np.asarray(wind_xy, dtype=np.float32)
    m_next = m_post + drift
    P_next = P_post + Q
    P_next = project_to_psd(P_next)

    return m_next.astype(np.float32), P_next.astype(np.float32)


# ============================================================
# Expected critic value under posterior
# ============================================================

def expected_critic_value_mc(
    *,
    model,
    y_adv_torch: torch.Tensor,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    wind_xy: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
    xi_torch: torch.Tensor,
    device: str = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    dev = torch.device(device)

    m_pred_t = torch.tensor(m_pred, dtype=torch.float32, device=dev)
    P_pred_t = torch.tensor(P_pred, dtype=torch.float32, device=dev)
    R_t = torch.tensor(R, dtype=torch.float32, device=dev)
    goal_t = torch.tensor(goal, dtype=torch.float32, device=dev)
    wind_t = torch.tensor(wind_xy, dtype=torch.float32, device=dev)

    S_t = P_pred_t + R_t
    K_t = P_pred_t @ torch.linalg.inv(S_t)

    innov_t = y_adv_torch - m_pred_t
    m_post_t = m_pred_t + K_t @ innov_t

    I2 = torch.eye(2, dtype=torch.float32, device=dev)
    P_post_t = (I2 - K_t) @ P_pred_t

    P_post_np = P_post_t.detach().cpu().numpy().astype(np.float32)
    L_np = sqrtm_psd(P_post_np)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=dev)

    x_samples = m_post_t.unsqueeze(0) + xi_torch @ L_t.T

    delta = (goal_t.unsqueeze(0) - x_samples) / float(goal_r_max)
    wind_batch = wind_t.unsqueeze(0).repeat(x_samples.shape[0], 1)
    obs_batch = torch.cat([delta, wind_batch], dim=1)

    v_batch = critic_values(model, obs_batch)
    mu_V = v_batch.mean()

    return mu_V, m_post_t.detach().cpu().numpy().astype(np.float32), P_post_np


def pgd_attack_on_expected_value(
    *,
    model,
    y_nom: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    wind_xy: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
    attack_sigma: np.ndarray,
    attack_eps: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    rng_seed: int,
    device: str = "cpu",
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    dev = torch.device(device)

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng_seed))
    xi_torch = torch.randn((mc_samples, 2), generator=gen, device=dev, dtype=torch.float32)

    y_curr_np = y_nom.astype(np.float32).copy()

    best_y = y_curr_np.copy()
    best_obj = None
    best_m_post = None
    best_P_post = None

    for _ in range(pgd_steps):
        y_t = torch.tensor(y_curr_np, dtype=torch.float32, device=dev, requires_grad=True)

        mu_V_t, m_post_np, P_post_np = expected_critic_value_mc(
            model=model,
            y_adv_torch=y_t,
            m_pred=m_pred,
            P_pred=P_pred,
            R=R,
            wind_xy=wind_xy,
            goal=goal,
            goal_r_max=goal_r_max,
            xi_torch=xi_torch,
            device=device,
        )

        mu_V_t.backward()
        grad = y_t.grad.detach().cpu().numpy().astype(np.float32)

        y_next = y_curr_np - float(pgd_step_size) * grad
        y_next = project_to_attack_region(
            y_candidate=y_next,
            center=y_nom,
            Sigma=attack_sigma,
            epsilon=attack_eps,
        )

        obj_val = float(mu_V_t.detach().cpu().item())
        if best_obj is None or obj_val < best_obj:
            best_obj = obj_val
            best_y = y_curr_np.copy()
            best_m_post = m_post_np.copy()
            best_P_post = P_post_np.copy()

        y_curr_np = y_next

    y_t = torch.tensor(y_curr_np, dtype=torch.float32, device=dev, requires_grad=True)
    mu_V_t, m_post_np, P_post_np = expected_critic_value_mc(
        model=model,
        y_adv_torch=y_t,
        m_pred=m_pred,
        P_pred=P_pred,
        R=R,
        wind_xy=wind_xy,
        goal=goal,
        goal_r_max=goal_r_max,
        xi_torch=xi_torch,
        device=device,
    )
    obj_val = float(mu_V_t.detach().cpu().item())

    if best_obj is None or obj_val < best_obj:
        best_obj = obj_val
        best_y = y_curr_np.copy()
        best_m_post = m_post_np.copy()
        best_P_post = P_post_np.copy()

    return best_y, float(best_obj), best_m_post, best_P_post


# ============================================================
# Trajectory collectors
# ============================================================

@torch.no_grad()
def collect_clean_rollout(env, model, device: str = "cpu") -> dict:
    dev = torch.device(device)
    obs = env.reset()
    goal = env.goal.copy()

    true_pos = []
    seen_pos = []
    actions = []
    winds = []

    ep_return = 0.0
    success = False

    for _ in range(env.cfg.max_steps):
        true_pos.append(env.x.copy())

        y_seen = obs_to_position(obs, goal, float(env.cfg.goal_r_max))
        seen_pos.append(y_seen.copy())

        wind_xy = np.asarray(obs[2:4], dtype=np.float32)
        winds.append(wind_xy.copy())

        obs_t = torch.tensor(obs, dtype=torch.float32, device=dev)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)
        actions.append(action.copy())

        obs, reward, done, info = env.step(action)
        ep_return += float(reward)

        if done:
            success = bool(info.get("success", False))
            break

    true_pos.append(env.x.copy())

    return {
        "true_pos": np.asarray(true_pos, dtype=np.float32),
        "seen_pos": np.asarray(seen_pos, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "winds": np.asarray(winds, dtype=np.float32),
        "goal": goal.astype(np.float32),
        "return": float(ep_return),
        "success": bool(success),
        "label": "clean",
    }


def collect_noisy_kf_rollout(
    env,
    model,
    kf_meas_std: float,
    kf_proc_std: float,
    device: str = "cpu",
) -> dict:
    dev = torch.device(device)
    obs = env.reset()
    goal = env.goal.copy()
    goal_r_max = float(env.cfg.goal_r_max)

    R = (float(kf_meas_std) ** 2) * np.eye(2, dtype=np.float32)
    Q = (float(kf_proc_std) ** 2) * np.eye(2, dtype=np.float32)

    # Start exactly at the true initial state so dashed path begins at (0,0)
    y0_exact = env.x.copy().astype(np.float32)
    m_pred = y0_exact.copy()
    P_pred = R.copy()

    true_pos = []
    seen_pos = []
    actions = []
    winds = []

    ep_return = 0.0
    success = False
    step_idx = 0

    for _ in range(env.cfg.max_steps):
        true_pos.append(env.x.copy())

        wind_xy = np.asarray(obs[2:4], dtype=np.float32)
        winds.append(wind_xy.copy())

        if step_idx == 0:
            # No noise on first observation
            y_obs = env.x.copy().astype(np.float32)
        else:
            y_obs = obs_to_position(obs, goal, goal_r_max)

        m_post, P_post, _K = kf_update_position(
            m_pred=m_pred,
            P_pred=P_pred,
            y_obs=y_obs,
            R=R,
        )
        seen_pos.append(m_post.copy())

        obs_filt = build_policy_obs_from_position(
            pos=m_post,
            goal=goal,
            goal_r_max=goal_r_max,
            wind_xy=wind_xy,
        )

        with torch.no_grad():
            obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
            action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)
        actions.append(action.copy())

        m_pred, P_pred = kf_predict_position(
            m_post=m_post,
            P_post=P_post,
            action=action,
            wind_xy=wind_xy,
            Q=Q,
        )

        obs, reward, done, info = env.step(action)
        ep_return += float(reward)
        step_idx += 1

        if done:
            success = bool(info.get("success", False))
            break

    true_pos.append(env.x.copy())

    return {
        "true_pos": np.asarray(true_pos, dtype=np.float32),
        "seen_pos": np.asarray(seen_pos, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "winds": np.asarray(winds, dtype=np.float32),
        "goal": goal.astype(np.float32),
        "return": float(ep_return),
        "success": bool(success),
        "label": "noisy + KF",
    }


def collect_attack_kf_rollout(
    env,
    model,
    attack_std: float,
    attack_eps: float,
    attack_prob: float,
    kf_meas_std: float,
    kf_proc_std: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    seed_for_attack: int,
    device: str = "cpu",
) -> dict:
    dev = torch.device(device)
    obs = env.reset()
    goal = env.goal.copy()
    goal_r_max = float(env.cfg.goal_r_max)

    R = (float(kf_meas_std) ** 2) * np.eye(2, dtype=np.float32)
    Q = (float(kf_proc_std) ** 2) * np.eye(2, dtype=np.float32)
    Sigma_attack = (float(attack_std) ** 2) * np.eye(2, dtype=np.float32)

    # RNGs:
    # - attack gate decides whether to attack or not
    # - obs noise is used when the step is not attacked
    rng_attack_gate = np.random.default_rng(int(seed_for_attack) + 777777)
    rng_obs_noise = np.random.default_rng(int(seed_for_attack) + 888888)

    # Start exactly at the true initial state so dashed path begins at (0,0)
    y0_exact = env.x.copy().astype(np.float32)
    m_pred = y0_exact.copy()
    P_pred = R.copy()

    true_pos = []
    seen_pos = []
    actions = []
    winds = []
    attacked_meas = []
    was_attacked = []

    ep_return = 0.0
    success = False
    step_idx = 0

    for _ in range(env.cfg.max_steps):
        true_pos.append(env.x.copy())

        y_clean = obs_to_position(obs, goal, goal_r_max)
        wind_xy = np.asarray(obs[2:4], dtype=np.float32)
        winds.append(wind_xy.copy())

        if step_idx == 0:
            # No attack and no noise on first observation
            y_used = env.x.copy().astype(np.float32)
            m_post, P_post, _K = kf_update_position(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=y_used,
                R=R,
            )
            was_attacked.append(False)

        else:
            do_attack = bool(rng_attack_gate.random() < float(attack_prob))

            if do_attack:
                y_used, _obj_star, m_post, P_post = pgd_attack_on_expected_value(
                    model=model,
                    y_nom=y_clean,
                    m_pred=m_pred,
                    P_pred=P_pred,
                    R=R,
                    wind_xy=wind_xy,
                    goal=goal,
                    goal_r_max=goal_r_max,
                    attack_sigma=Sigma_attack,
                    attack_eps=attack_eps,
                    pgd_steps=pgd_steps,
                    pgd_step_size=pgd_step_size,
                    mc_samples=mc_samples,
                    rng_seed=int(seed_for_attack + 10_000 * step_idx),
                    device=device,
                )
                was_attacked.append(True)

            else:
                noise = rng_obs_noise.normal(
                    0.0, float(kf_meas_std), size=(2,)
                ).astype(np.float32)
                y_used = y_clean + noise

                m_post, P_post, _K = kf_update_position(
                    m_pred=m_pred,
                    P_pred=P_pred,
                    y_obs=y_used,
                    R=R,
                )
                was_attacked.append(False)

        attacked_meas.append(y_used.copy())
        seen_pos.append(m_post.copy())

        obs_filt = build_policy_obs_from_position(
            pos=m_post,
            goal=goal,
            goal_r_max=goal_r_max,
            wind_xy=wind_xy,
        )

        with torch.no_grad():
            obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
            action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)
        actions.append(action.copy())

        m_pred, P_pred = kf_predict_position(
            m_post=m_post,
            P_post=P_post,
            action=action,
            wind_xy=wind_xy,
            Q=Q,
        )

        obs, reward, done, info = env.step(action)
        ep_return += float(reward)
        step_idx += 1

        if done:
            success = bool(info.get("success", False))
            break

    true_pos.append(env.x.copy())

    return {
        "true_pos": np.asarray(true_pos, dtype=np.float32),
        "seen_pos": np.asarray(seen_pos, dtype=np.float32),
        "attacked_meas": np.asarray(attacked_meas, dtype=np.float32),
        "was_attacked": np.asarray(was_attacked, dtype=bool),
        "actions": np.asarray(actions, dtype=np.float32),
        "winds": np.asarray(winds, dtype=np.float32),
        "goal": goal.astype(np.float32),
        "return": float(ep_return),
        "success": bool(success),
        "label": f"attack + KF (p={attack_prob})",
    }

# ============================================================
# Plot
# ============================================================
def plot_three_rollouts_same_axes(
    clean_data: dict,
    noisy_kf_data: dict,
    attack_kf_data: dict,
    outpath: str,
    goal_radius: float,
    arrow_every: int = 2,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(wspace=0.08, hspace=-0.02)
    axes = axes.ravel()

    datasets = [clean_data, noisy_kf_data, attack_kf_data]
    panel_titles = [
        "All methods",
        "Clean",
        "Noisy + KF",
        "Attack + KF",
    ]

    # --------------------------------------------------------
    # Global limits so all panels share the same coordinate box
    # --------------------------------------------------------
    all_pts = []
    for d in datasets:
        all_pts.append(d["true_pos"])
        all_pts.append(d["seen_pos"])
        all_pts.append(d["goal"][None, :])

    P = np.vstack(all_pts)
    xmin, ymin = P.min(axis=0)
    xmax, ymax = P.max(axis=0)

    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    pad = 0.12 * max(dx, dy)

    xlim = (xmin - pad, xmax + pad)
    ylim = (ymin - pad, ymax + pad)

    # Colors
    def get_color(label: str) -> str:
        if label == "clean":
            return "tab:blue"
        elif label == "noisy + KF":
            return "tab:orange"
        else:
            return "tab:green"

    # --------------------------------------------------------
    # Small helper to draw one dataset on one axis
    # --------------------------------------------------------
    def draw_single_dataset(ax, d: dict, show_legend: bool = True):
        label = d["label"]
        c = get_color(label)

        true_pos = d["true_pos"]
        seen_pos = d["seen_pos"]
        actions = d["actions"]
        goal = d["goal"]
        start = true_pos[0]

        h_true, = ax.plot(
            true_pos[:, 0], true_pos[:, 1],
            linewidth=1.4,
            alpha=TRUE_ALPHA,
            color=c,
            label=f"{label} true path",
        )

        h_seen, = ax.plot(
            seen_pos[:, 0], seen_pos[:, 1],
            linewidth=1.1,
            linestyle="--",
            alpha=OBS_ALPHA,
            color=c,
            label=f"{label} seen state",
        )

        idx = np.arange(0, len(actions), max(1, arrow_every))
        ax.quiver(
            seen_pos[idx, 0],
            seen_pos[idx, 1],
            actions[idx, 0],
            actions[idx, 1],
            angles="xy",
            scale_units="xy",
            scale=2.0,
            width=0.002,
            alpha=0.80,
            color=c,
        )

        h_start = ax.scatter(
            start[0], start[1],
            s=85, marker="o", color="black",
            edgecolors="white", linewidths=1.2, label="start"
        )

        h_goal = ax.scatter(
            goal[0], goal[1],
            s=220, marker="*", color="gold",
            edgecolors="black", linewidths=1.0, label="goal"
        )

        circ = plt.Circle(
            (goal[0], goal[1]),
            goal_radius,
            fill=False,
            linestyle=":",
            linewidth=1.8,
            alpha=0.8,
            color="black",
        )
        ax.add_patch(circ)

        h_end = ax.scatter(
            true_pos[-1, 0], true_pos[-1, 1],
            s=75, marker="X", color=c,
            edgecolors="white", linewidths=1.0, label="end"
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if show_legend:
            wind_proxy = FancyArrowPatch(
                (0, 0), (1, 0),
                arrowstyle="->",
                mutation_scale=14,
                color=c,
                linewidth=1.5
            )

            ax.legend(
                handles=[h_true, h_seen, wind_proxy, h_start, h_goal, h_end],
                labels=[
                    f"{label} true path",
                    f"{label} seen state",
                    "wind",
                    "start",
                    "goal",
                    "end",
                ],
                fontsize=8,
                frameon=True,
                loc="best",
            )
    # --------------------------------------------------------
    # Panel 1: all together
    # --------------------------------------------------------
    ax = axes[0]

    for d in datasets:
        label = d["label"]
        c = get_color(label)

        true_pos = d["true_pos"]
        seen_pos = d["seen_pos"]
        actions = d["actions"]

        ax.plot(
            true_pos[:, 0], true_pos[:, 1],
            linewidth=1.2,
            alpha=TRUE_ALPHA,
            color=c,
            label=f"{label} true path",
        )

        ax.plot(
            seen_pos[:, 0], seen_pos[:, 1],
            linewidth=1.2,
            linestyle="--",
            alpha=OBS_ALPHA,
            color=c,
            label=f"{label} seen state",
        )

        ax.scatter(
            true_pos[-1, 0], true_pos[-1, 1],
            s=70, marker="X", color=c,
            edgecolors="white", linewidths=1.0
        )

    goal = clean_data["goal"]
    start = clean_data["true_pos"][0]

    ax.scatter(
        goal[0], goal[1],
        s=220, marker="*", color="gold",
        edgecolors="black", linewidths=1.0, label="goal"
    )
    circ = plt.Circle(
        (goal[0], goal[1]),
        goal_radius,
        fill=False,
        linestyle=":",
        linewidth=0.8,
        alpha=0.8,
        color="black",
    )
    ax.add_patch(circ)

    ax.scatter(
        start[0], start[1],
        s=90, marker="o", color="black",
        edgecolors="white", linewidths=1.2, label="start"
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, frameon=True, loc="best")
    ax.set_title(panel_titles[0])

    # --------------------------------------------------------
    # Panels 2, 3, 4: one method per panel
    # --------------------------------------------------------
    draw_single_dataset(axes[1], clean_data, show_legend=True)
    axes[1].set_title(panel_titles[1])

    draw_single_dataset(axes[2], noisy_kf_data, show_legend=True)
    axes[2].set_title(panel_titles[2])

    draw_single_dataset(axes[3], attack_kf_data, show_legend=True)
    axes[3].set_title(panel_titles[3])

    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"[saved] {outpath}")

# ============================================================
# Main
# ============================================================

def main():
    if not os.path.exists(ADV_FILE_PATH):
        raise FileNotFoundError(f"AdvRL_wind.py not found at:\n  {ADV_FILE_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at:\n  {MODEL_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)

    import AdvRL_wind as mod
    print(f"[import] Imported AdvRL_wind from: {ADV_FILE_PATH}")

    ActorCritic = mod.ActorCritic
    AdvRLEnvConfig = mod.AdvRLEnvConfig
    AdvRL2DEnv = mod.AdvRL2DEnv
    EnvNoisy = make_env_with_separate_obs_rng(AdvRL2DEnv)

    base_cfg = AdvRLEnvConfig(
        obs_noise_std=0.0,
        proc_noise_std=0.0,
        wind_epsilon=0.9,
        wind_volatility=0.25,
        seed=PLOT_SEED,
        step_penalty=-1.0,
        success_reward=25.0,
        timeout_penalty=-25.0,
    )

    device = torch.device(DEVICE)

    try:
        model = ActorCritic(obs_dim=4, hidden=128, act_dim=2, std_fixed=0.35).to(device)
    except TypeError:
        model = ActorCritic(obs_dim=4, hidden=128, act_dim=2).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cfg_clean = AdvRLEnvConfig(**{**asdict(base_cfg)})
    cfg_noisy = AdvRLEnvConfig(**{**asdict(base_cfg)})
    cfg_attack = AdvRLEnvConfig(**{**asdict(base_cfg)})

    cfg_clean.seed = PLOT_SEED
    cfg_clean.obs_noise_std = 0.0

    cfg_noisy.seed = PLOT_SEED
    cfg_noisy.obs_noise_std = float(NOISE_STD)

    cfg_attack.seed = PLOT_SEED
    cfg_attack.obs_noise_std = 0.0

    env_clean = AdvRL2DEnv(cfg_clean)
    env_noisy = EnvNoisy(cfg_noisy)
    env_attack = AdvRL2DEnv(cfg_attack)

    clean_data = collect_clean_rollout(
        env_clean,
        model,
        device=DEVICE,
    )

    noisy_kf_data = collect_noisy_kf_rollout(
        env_noisy,
        model,
        kf_meas_std=KF_MEAS_STD,
        kf_proc_std=KF_PROC_STD,
        device=DEVICE,
    )

    attack_kf_data = collect_attack_kf_rollout(
        env_attack,
        model,
        attack_std=ATTACK_STD,
        attack_eps=ATTACK_EPS,
        attack_prob=ATTACK_PROB,
        kf_meas_std=KF_MEAS_STD,
        kf_proc_std=KF_PROC_STD,
        pgd_steps=PGD_STEPS,
        pgd_step_size=PGD_STEP_SIZE,
        mc_samples=MC_SAMPLES,
        seed_for_attack=PLOT_SEED,
        device=DEVICE,
    )

    outpath = os.path.join(RESULTS_DIR, "three_paths_clean_noisyKF_attackKF.png")

    plot_three_rollouts_same_axes(
        clean_data=clean_data,
        noisy_kf_data=noisy_kf_data,
        attack_kf_data=attack_kf_data,
        outpath=outpath,
        goal_radius=float(cfg_clean.goal_radius),
        arrow_every=ARROW_EVERY,
    )

    print(
        f"[summary] clean ret={clean_data['return']:.1f} | "
        f"noisy+KF ret={noisy_kf_data['return']:.1f} | "
        f"attack+KF ret={attack_kf_data['return']:.1f}"
    )
    print("[done] Plot finished.")


if __name__ == "__main__":
    main()