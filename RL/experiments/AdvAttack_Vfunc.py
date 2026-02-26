#!/usr/bin/env python3
"""
RL/experiments/AdvAttack_VValue.py  (FIXED)

- Uses correct V(s) from your ActorCritic.forward(obs) = (mu, log_std, value)
  -> value = out[2]  (shape (N,))

- Attacks V(x_t) = V(obs(delta_hat_adv_post(z))) where:
    delta_hat_adv_post(z) is the KF posterior mean after updating with candidate z
  (NO lookahead, NO u(a) inside g)

- PGD + projection onto ellipsoid built from shadow prior:
    (z - mu)^T Sigma^{-1} (z - mu) <= epsilon
  with mu = delta_hat_shadow (prior mean), Sigma = P_shadow + R

Outputs:
  - RL/results/v_map_obs2d.png
  - RL/results/v_attack_4sims_belief_vs_real.png
  - RL/results/v_attack_rewards_over_episodes.png
"""

from __future__ import annotations

import os
import sys
from typing import Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DEVICE = "cpu"
SEED = 2020

# 4-sim grid figure
N_SIMS = 4
GRID_ROWS, GRID_COLS = 2, 2

# Reward figure
EVAL_EPISODES = 200

# Noise settings
SIGMA_OBS = 1.0
PROC_NOISE_STD = 0.05

# Attack settings
EPSILON_ATTACK = 15.991
P_ATTACK = 0.90

# PGD settings (attack V(s))
ATTACK_STEPS = 35
ATTACK_LR = 0.20
ATTACK_FD_H = 0.05

# If True: J=(V-M*)^2 ; else minimize V directly
ATTACK_USE_TARGET = False
ATTACK_M_STAR = -10.0

# Print gradient every k iters if non-zero
ATTACK_VERBOSE = False
ATTACK_PRINT_EVERY = 100
ATTACK_GRAD_TOL = 1e-8

# Value-map plot settings (obs-space)
VALUE_MAP_GRID_N = 240
VALUE_MAP_LIM = 1.0
VALUE_MAP_BATCH = 16384

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
# PSD / linear algebra helpers
# ------------------------------------------------------------
def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def project_to_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=float))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(w) @ V.T

def inv_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=float))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(1.0 / w) @ V.T


# ------------------------------------------------------------
# Projection to ellipsoid
# ------------------------------------------------------------
def project_to_attack_region(
    y_candidate: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Euclidean projection onto {y : (y-mu)^T Sigma^{-1} (y-mu) <= epsilon}.
    """
    y_candidate = np.asarray(y_candidate, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = project_to_psd(np.asarray(Sigma, dtype=float))

    Sinv = inv_psd(Sigma)
    d = y_candidate - mu
    maha = float(d.T @ Sinv @ d)
    if maha <= epsilon + tol:
        return y_candidate.copy()

    s, U = np.linalg.eigh(Sigma)
    s = np.maximum(s, 1e-12)
    r = U.T @ d

    def f(lam: float) -> float:
        return float(np.sum((s * r**2) / (s + lam)**2) - epsilon)

    lam_low, lam_high = 0.0, 1.0
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

    lam = 0.5 * (lam_low + lam_high)
    z = (s / (s + lam)) * r
    y_proj = mu + U @ z
    return y_proj


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
# Correct V(s) extraction for YOUR ActorCritic
# forward(obs) -> (mu, log_std, value)
# ------------------------------------------------------------
@torch.no_grad()
def V_from_obs(model: ActorCritic, obs_batch: torch.Tensor) -> torch.Tensor:
    """
    Returns V(obs) as (N,) tensor.
    Your model returns tuple len=3; value is out[2] with shape (N,).
    """
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.unsqueeze(0)

    out = model(obs_batch)
    if not (isinstance(out, (tuple, list)) and len(out) == 3):
        raise RuntimeError("Expected ActorCritic.forward(obs) to return a tuple/list of length 3.")

    v = out[2]
    if v.ndim == 2 and v.shape[1] == 1:
        v = v.squeeze(-1)
    return v


# ------------------------------------------------------------
# Objective for attack: V after attacked KF update with candidate z
# ------------------------------------------------------------
@torch.no_grad()
def V_given_z(
    *,
    z: np.ndarray,
    delta_hat_adv_pred: np.ndarray,
    P_adv_pred: np.ndarray,
    R: np.ndarray,
    model: ActorCritic,
    device: torch.device,
) -> float:
    """
    g(z) = V( obs(delta_hat_post(z)) )
    where delta_hat_post(z) = KF_update(delta_hat_adv_pred, P_adv_pred, z, R)[0]
    """
    z = np.asarray(z, dtype=np.float32).reshape(2,)
    delta_hat_new, _ = kf_update(delta_hat_adv_pred.copy(), P_adv_pred.copy(), z, R)
    obs = unit_dir(delta_hat_new)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    v = V_from_obs(model, obs_t)
    return float(v.squeeze().detach().cpu().numpy())


@torch.no_grad()
def fd_grad_objective_wrt_z(
    *,
    z: np.ndarray,
    ell_mu: np.ndarray,
    ell_Sigma: np.ndarray,
    epsilon: float,
    delta_hat_adv_pred: np.ndarray,
    P_adv_pred: np.ndarray,
    R: np.ndarray,
    model: ActorCritic,
    device: torch.device,
    use_target: bool,
    M_star: float,
    h: float,
) -> tuple[np.ndarray, float, float]:
    """
    Returns (grad, J(z), V(z)) with projection inside the feasible ellipsoid.

    If use_target:
        J(z) = (V(z) - M_star)^2
    else:
        J(z) = V(z)     (we minimize V directly)
    """
    z = np.asarray(z, dtype=float).reshape(2,)
    ell_mu = np.asarray(ell_mu, dtype=float).reshape(2,)
    ell_Sigma = np.asarray(ell_Sigma, dtype=float).reshape(2, 2)

    z0 = project_to_attack_region(z, ell_mu, ell_Sigma, epsilon)

    V0 = V_given_z(
        z=z0,
        delta_hat_adv_pred=delta_hat_adv_pred,
        P_adv_pred=P_adv_pred,
        R=R,
        model=model,
        device=device,
    )
    J0 = (V0 - float(M_star)) ** 2 if use_target else float(V0)

    grad = np.zeros(2, dtype=float)
    for j in range(2):
        ej = np.zeros(2, dtype=float)
        ej[j] = 1.0

        zp = project_to_attack_region(z0 + h * ej, ell_mu, ell_Sigma, epsilon)
        zm = project_to_attack_region(z0 - h * ej, ell_mu, ell_Sigma, epsilon)

        Vp = V_given_z(
            z=zp,
            delta_hat_adv_pred=delta_hat_adv_pred,
            P_adv_pred=P_adv_pred,
            R=R,
            model=model,
            device=device,
        )
        Vm = V_given_z(
            z=zm,
            delta_hat_adv_pred=delta_hat_adv_pred,
            P_adv_pred=P_adv_pred,
            R=R,
            model=model,
            device=device,
        )

        Jp = (Vp - float(M_star)) ** 2 if use_target else float(Vp)
        Jm = (Vm - float(M_star)) ** 2 if use_target else float(Vm)

        grad[j] = (Jp - Jm) / (2.0 * h)

    return grad, float(J0), float(V0)


@torch.no_grad()
def solve_pgd_attack_V_over_ellipsoid(
    *,
    z_init: np.ndarray,
    ell_mu: np.ndarray,
    ell_Sigma: np.ndarray,
    epsilon: float,
    delta_hat_adv_pred: np.ndarray,
    P_adv_pred: np.ndarray,
    R: np.ndarray,
    model: ActorCritic,
    device: torch.device,
    eta: float,
    n_steps: int,
    fd_h: float,
    use_target: bool,
    M_star: float,
    verbose: bool,
    print_every: int,
    grad_tol: float,
) -> tuple[np.ndarray, dict]:
    """
    PGD + projection to minimize V(z) (or (V-M*)^2) over the ellipsoid.
    """
    ell_mu = np.asarray(ell_mu, dtype=float).reshape(2,)
    ell_Sigma = project_to_psd(np.asarray(ell_Sigma, dtype=float))

    z = project_to_attack_region(np.asarray(z_init, dtype=float), ell_mu, ell_Sigma, epsilon)

    z_hist, J_hist, V_hist, g_hist = [], [], [], []

    for it in range(int(n_steps)):
        grad, J0, V0 = fd_grad_objective_wrt_z(
            z=z,
            ell_mu=ell_mu,
            ell_Sigma=ell_Sigma,
            epsilon=epsilon,
            delta_hat_adv_pred=delta_hat_adv_pred,
            P_adv_pred=P_adv_pred,
            R=R,
            model=model,
            device=device,
            use_target=use_target,
            M_star=M_star,
            h=fd_h,
        )

        gnorm = float(np.linalg.norm(grad))
        if verbose and (it % max(int(print_every), 1) == 0) and (gnorm > grad_tol):
            if use_target:
                print(f"[PGD it={it:03d}] V={V0:+.4f}  J=(V-M*)^2={J0:.3e}  z={z}  grad={grad}  ||g||={gnorm:.2e}")
            else:
                print(f"[PGD it={it:03d}] V={V0:+.4f}  z={z}  grad={grad}  ||g||={gnorm:.2e}")

        z = z - eta * grad
        z = project_to_attack_region(z, ell_mu, ell_Sigma, epsilon)

        z_hist.append(z.copy())
        J_hist.append(J0)
        V_hist.append(V0)
        g_hist.append(grad.copy())

    return z.astype(np.float32), {
        "z_hist": np.asarray(z_hist),
        "J_hist": np.asarray(J_hist),
        "V_hist": np.asarray(V_hist),
        "grad_hist": np.asarray(g_hist),
    }


# ------------------------------------------------------------
# Rollouts
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
        z = (env.goal - env.x).astype(np.float32)
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
def rollout_noisy(
    model: ActorCritic,
    device: torch.device,
    env: AdvRL2DEnv,
    R: np.ndarray,
    Q: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
):
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
    ADV agent (non-compounding):
      - shadow KF uses z_real always -> ellipsoid params (mu, Sigma)
      - attacked KF uses z_adv (PGD attack on V) else z_real
    """
    traj_true = [env.x.copy()]
    traj_shadow: List[np.ndarray] = []
    traj_adv: List[np.ndarray] = []
    attack_mask: List[int] = []

    done = False
    ep_ret = 0.0
    success = False

    true_delta = (env.goal - env.x).astype(np.float32)
    z_real = true_delta + rng.normal(0.0, sigma, size=(2,)).astype(np.float32)

    delta_hat_shadow = z_real.copy()
    P_shadow = R.copy()

    delta_hat_adv = z_real.copy()
    P_adv = R.copy()

    for step_idx in range(env.cfg.max_steps):
        true_delta = (env.goal - env.x).astype(np.float32)
        z_real = true_delta + rng.normal(0.0, sigma, size=(2,)).astype(np.float32)

        # ellipsoid from SHADOW PRIOR
        ell_mu = delta_hat_shadow
        ell_Sigma = P_shadow + R

        do_attack = (step_idx >= 1) and (rng.random() < p_attack)
        attack_mask.append(1 if do_attack else 0)

        if not do_attack:
            z_use = z_real.copy()
        else:
            try:
                z_adv, _info = solve_pgd_attack_V_over_ellipsoid(
                    z_init=z_real,
                    ell_mu=ell_mu,
                    ell_Sigma=ell_Sigma,
                    epsilon=epsilon,
                    delta_hat_adv_pred=delta_hat_adv,
                    P_adv_pred=P_adv,
                    R=R,
                    model=model,
                    device=device,
                    eta=ATTACK_LR,
                    n_steps=ATTACK_STEPS,
                    fd_h=ATTACK_FD_H,
                    use_target=ATTACK_USE_TARGET,
                    M_star=ATTACK_M_STAR,
                    verbose=ATTACK_VERBOSE,
                    print_every=ATTACK_PRINT_EVERY,
                    grad_tol=ATTACK_GRAD_TOL,
                )
                z_use = z_adv.astype(np.float32)
            except Exception:
                z_use = z_real.copy()

        # update shadow with real
        delta_hat_shadow, P_shadow = kf_update(delta_hat_shadow, P_shadow, z_real, R)
        # update attacked with attacked
        delta_hat_adv, P_adv = kf_update(delta_hat_adv, P_adv, z_use, R)

        traj_shadow.append((env.goal - delta_hat_shadow).astype(np.float32))
        traj_adv.append((env.goal - delta_hat_adv).astype(np.float32))

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

        delta_hat_shadow, P_shadow = kf_predict(delta_hat_shadow, P_shadow, u, Q)
        delta_hat_adv, P_adv = kf_predict(delta_hat_adv, P_adv, u, Q)

    traj_true_arr = np.array(traj_true, dtype=np.float32)
    traj_shadow_arr = np.array(traj_shadow, dtype=np.float32)
    traj_adv_arr = np.array(traj_adv, dtype=np.float32)
    attack_mask_arr = np.array(attack_mask, dtype=np.int32)

    # pad beliefs to match
    L = traj_true_arr.shape[0]

    def _pad(arr: np.ndarray, L: int) -> np.ndarray:
        if arr.shape[0] == L:
            return arr
        if arr.shape[0] == 0:
            return np.repeat(env.x[None, :], L, axis=0).astype(np.float32)
        last = arr[-1]
        pad = np.repeat(last[None, :], L - arr.shape[0], axis=0)
        return np.vstack([arr, pad]).astype(np.float32)

    traj_shadow_arr = _pad(traj_shadow_arr, L)
    traj_adv_arr = _pad(traj_adv_arr, L)

    if attack_mask_arr.shape[0] < L:
        pad = np.zeros(L - attack_mask_arr.shape[0], dtype=np.int32)
        attack_mask_arr = np.concatenate([attack_mask_arr, pad], axis=0)

    return traj_true_arr, traj_shadow_arr, traj_adv_arr, attack_mask_arr, ep_ret, success


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def _set_rcparams():
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.labelsize": 11.0,
        "legend.fontsize": 9.8,
        "axes.linewidth": 1.0,
    })


@torch.no_grad()
def plot_value_map_obs2d(model: ActorCritic, device: torch.device) -> None:
    """
    Plot V(obs) over obs-space in [-1,1]^2 (but remember your obs is usually unit direction).
    We normalize each point to unit_dir to match what the network actually sees.
    """
    _set_rcparams()

    n = int(VALUE_MAP_GRID_N)
    lim = float(VALUE_MAP_LIM)

    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, ys)
    obs = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)

    norms = np.linalg.norm(obs, axis=1, keepdims=True)
    obs_unit = np.where(norms > 1e-8, obs / norms, 0.0).astype(np.float32)

    Vvals = np.zeros((obs_unit.shape[0],), dtype=np.float32)
    bs = int(VALUE_MAP_BATCH)

    for i in range(0, obs_unit.shape[0], bs):
        batch = torch.tensor(obs_unit[i:i+bs], dtype=torch.float32, device=device)
        v = V_from_obs(model, batch).detach().cpu().numpy().reshape(-1)
        Vvals[i:i+bs] = v

    Vmap = Vvals.reshape(n, n)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(Vmap, origin="lower", extent=[-lim, lim, -lim, lim], aspect="equal", cmap="viridis")
    ax.set_title("V(obs) map (inputs normalized to unit direction)")
    ax.set_xlabel("obs[0]")
    ax.set_ylabel("obs[1]")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("V(obs)")

    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1.2, alpha=0.85)

    os.makedirs(os.path.join(_PROJECT_ROOT, OUT_DIR), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, OUT_DIR, "v_map_obs2d.png")
    fig.savefig(out_path, dpi=260, facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_4_simulations_beliefs(model: ActorCritic, device: torch.device, sigma: float) -> None:
    _set_rcparams()

    cream = "#F7F1E3"
    cream2 = "#FBF7EE"
    ink = "#1F1F1F"
    grid = "#D9D2C3"

    c_true = "#2F2F2F"
    c_shadow = "#2F6FA8"
    c_adv = "#C85B4F"
    c_goal = "#D4A017"
    c_marks = "#C85B4F"

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(14, 10), constrained_layout=True)
    fig.patch.set_facecolor(cream)
    axes = axes.flatten()

    r_var = max(float(sigma) ** 2, R_VAR_FLOOR)
    q_var = max(float(PROC_NOISE_STD) ** 2, Q_VAR_FLOOR)
    R = (r_var * np.eye(2)).astype(np.float32)
    Q = (q_var * np.eye(2)).astype(np.float32)

    for i in range(N_SIMS):
        ax = axes[i]
        ax.set_facecolor(cream2)
        ax.grid(True, color=grid, alpha=0.65, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#BFB8A8")
        ax.spines["bottom"].set_color("#BFB8A8")
        ax.tick_params(colors=ink, labelsize=9)
        ax.set_aspect("equal", adjustable="box")

        ep_seed = SEED + 1000 + i
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

        env = AdvRL2DEnv(cfg)
        env.reset()
        goal = env.goal.copy()

        traj_true, traj_shadow, traj_adv, atk_mask, ep_ret, succ = rollout_adv(
            model, device, env, R, Q, rng_adv, sigma, EPSILON_ATTACK, P_ATTACK
        )

        ax.plot(traj_true[:, 0], traj_true[:, 1],
                color=c_true, linewidth=2.3, label="Real position" if i == 0 else None)
        ax.plot(traj_shadow[:, 0], traj_shadow[:, 1],
                color=c_shadow, linewidth=2.0, linestyle=":",
                label="Belief (shadow)" if i == 0 else None)
        ax.plot(traj_adv[:, 0], traj_adv[:, 1],
                color=c_adv, linewidth=2.2, linestyle="--",
                label="Belief (attacked)" if i == 0 else None)

        idx = np.where(atk_mask[:traj_true.shape[0]] == 1)[0]
        if idx.size > 0:
            ax.scatter(traj_true[idx, 0], traj_true[idx, 1],
                       s=28, marker="x", color=c_marks, alpha=0.9,
                       label="Attack step" if i == 0 else None)

        ax.scatter([traj_true[0, 0]], [traj_true[0, 1]], s=28, marker="s", color=ink, alpha=0.9)
        ax.scatter([goal[0]], [goal[1]], s=110, marker="*", color=c_goal, zorder=6)
        ax.add_patch(plt.Circle((goal[0], goal[1]), GOAL_RADIUS,
                                color=c_goal, fill=False, linestyle="--",
                                linewidth=1.4, alpha=0.75))

        ax.set_title(
            f"Sim {i+1} | success={'OK' if succ else 'X'} | attacks={int(idx.size)} | return={ep_ret:.1f}",
            color=ink
        )
        if i == 0:
            ax.legend(loc="upper left", frameon=False)

    for j in range(N_SIMS, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"PGD attack on V(x_t) (σ={sigma}, ε={EPSILON_ATTACK}, p={P_ATTACK})",
        fontsize=13, color=ink
    )

    os.makedirs(os.path.join(_PROJECT_ROOT, OUT_DIR), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, OUT_DIR, "v_attack_4sims_belief_vs_real.png")
    plt.savefig(out_path, dpi=260, facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_rewards_over_episodes(model: ActorCritic, device: torch.device, sigma: float) -> None:
    _set_rcparams()

    cream = "#F7F1E3"
    cream2 = "#FBF7EE"
    ink = "#1F1F1F"
    grid = "#D9D2C3"

    c_clean = "#2F2F2F"
    c_noisy = "#2F6FA8"
    c_adv = "#C85B4F"

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
        _, _, _, _, ret_a, ok_a = rollout_adv(model, device, env_adv, R, Q, rng_adv, sigma, EPSILON_ATTACK, P_ATTACK)

        rets_clean.append(ret_c)
        rets_noisy.append(ret_n)
        rets_adv.append(ret_a)

        succ_clean += int(ok_c)
        succ_noisy += int(ok_n)
        succ_adv += int(ok_a)

    rets_clean = np.array(rets_clean, dtype=float)
    rets_noisy = np.array(rets_noisy, dtype=float)
    rets_adv = np.array(rets_adv, dtype=float)

    x = np.arange(1, EVAL_EPISODES + 1)

    fig, ax = plt.subplots(1, 1, figsize=(11.2, 6.2), constrained_layout=True)
    fig.patch.set_facecolor(cream)
    ax.set_facecolor(cream2)
    ax.grid(True, which="major", color=grid, linewidth=0.9, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BFB8A8")
    ax.spines["bottom"].set_color("#BFB8A8")
    ax.tick_params(colors=ink)

    ax.plot(x, rets_clean, color=c_clean, linewidth=1.9, label="Clean")
    ax.plot(x, rets_noisy, color=c_noisy, linewidth=1.9, label="Noisy+KF")
    ax.plot(x, rets_adv, color=c_adv, linewidth=2.1, label="Adv+KF (attack V)")

    ax.set_xlabel("Episode", color=ink)
    ax.set_ylabel("Episode return", color=ink)
    ax.legend(frameon=False, ncol=3, loc="best")

    ax.set_title(
        f"Rewards over episodes (attack V(x_t), σ={sigma}, ε={EPSILON_ATTACK}, p={P_ATTACK})\n"
        f"Success rate: Clean={succ_clean/EVAL_EPISODES:.2f}, "
        f"Noisy={succ_noisy/EVAL_EPISODES:.2f}, Adv={succ_adv/EVAL_EPISODES:.2f}",
        color=ink
    )

    os.makedirs(os.path.join(_PROJECT_ROOT, OUT_DIR), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, OUT_DIR, "v_attack_rewards_over_episodes.png")
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

    # 1) V map (no attack)
    plot_value_map_obs2d(model, device)

    # 2) 4 sims (belief vs real)
    plot_4_simulations_beliefs(model, device, sigma=SIGMA_OBS)

    # 3) rewards
    plot_rewards_over_episodes(model, device, sigma=SIGMA_OBS)


if __name__ == "__main__":
    main()