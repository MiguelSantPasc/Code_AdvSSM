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
from xml.parsers.expat import model

from networkx import sigma
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DEVICE = "cpu"
SEED = 2026

# 4-sim grid figure
N_SIMS = 4
GRID_ROWS, GRID_COLS = 2, 2

# Reward figure
EVAL_EPISODES = 200         # "several iterations"
RUNNING_MEAN_W = 20         # running mean window

# Noise settings
SIGMA_OBS = 1.0             # observation noise used for NOISY and for ADV "real sensor"
PROC_NOISE_STD = 0.05       # environment process noise

# Attack settings
EPSILON_KKT = 10.991         # Chi-square df=2, 95%
P_ATTACK = 0.50              # attack applied with probability p each step (from 2nd obs), else normal noise

# Reward-attack (PGD) settings
REWARD_ATTACK_STEPS = 40        # PGD iterations (empieza con 20-40)
REWARD_ATTACK_LR    = 0.20      # step size (ajusta 0.05..0.5)
REWARD_ATTACK_FD_H  = 0.05      # finite diff step in z-space (0.01..0.1)
REWARD_ATTACK_MC    = 1         # MC samples for E[r(z)] (1 es rápido; 3-5 más estable)

# Target for squared objective (muy bajo => fuerza reward hacia abajo)
REWARD_M_STAR       = -50.0     # ponlo acorde a la escala de tu reward
# Si prefieres minimizar reward directamente, lo dejamos como opción (ver abajo)
REWARD_USE_TARGET   = True

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

    Sinv = np.linalg.inv(Sigma)
    diff = y_candidate - mu
    maha = float(diff.T @ Sinv @ diff)
    if maha <= epsilon + tol:
        return y_candidate.copy()

    s, U = np.linalg.eigh(Sigma)
    s = np.maximum(s, 1e-12)
    r = U.T @ diff

    def f(lam: float) -> float:
        return float(np.sum((s * r**2) / (s + lam)**2) - epsilon)

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
    y_proj = mu + U @ z
    return y_proj


def _clone_env(env):
    return copy.deepcopy(env)

@torch.no_grad()
def reward_one_step_given_z(
    *,
    z: np.ndarray,
    delta_hat_adv: np.ndarray,
    P_adv: np.ndarray,
    R: np.ndarray,
    env: AdvRL2DEnv,
    model: ActorCritic,
    device: torch.device,
) -> float:
    z = np.asarray(z, dtype=np.float32).reshape(2,)

    # attacked KF update with candidate z
    delta_hat_new, _ = kf_update(delta_hat_adv.copy(), P_adv.copy(), z, R)

    # policy from updated belief
    obs_policy = unit_dir(delta_hat_new)
    obs_t = torch.tensor(obs_policy, dtype=torch.float32, device=device)
    action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

    # simulate one env step on a clone
    env_tmp = _clone_env(env)
    _, r, _, _ = env_tmp.step(action)
    return float(r)

@torch.no_grad()
def estimate_E_reward_given_z(
    *,
    z: np.ndarray,
    delta_hat_adv: np.ndarray,
    P_adv: np.ndarray,
    R: np.ndarray,
    env: AdvRL2DEnv,
    model: ActorCritic,
    device: torch.device,
    n_mc: int = 1,
) -> float:
    # Si deepcopy copia el RNG interno, MC>1 puede dar el mismo resultado.
    # Aún así, para empezar funciona perfecto con n_mc=1.
    rs = []
    for _ in range(int(n_mc)):
        rs.append(
            reward_one_step_given_z(
                z=z,
                delta_hat_adv=delta_hat_adv,
                P_adv=P_adv,
                R=R,
                env=env,
                model=model,
                device=device,
            )
        )
    return float(np.mean(rs))

@torch.no_grad()
def fd_grad_objective_wrt_z(
    *,
    z: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    delta_hat_adv: np.ndarray,
    P_adv: np.ndarray,
    R: np.ndarray,
    env: AdvRL2DEnv,
    model: ActorCritic,
    device: torch.device,
    M_star: float,
    use_target: bool,
    n_mc: int,
    h: float,
) -> tuple[np.ndarray, float]:
    """
    Returns: (grad (2,), J(z))
    J(z) = (E[r(z)] - M_star)^2   if use_target
         =  E[r(z)]              else  (minimize reward directly)
    All evaluations are projected into the feasible ellipsoid.
    """
    z = np.asarray(z, dtype=float).reshape(2,)
    mu = np.asarray(mu, dtype=float).reshape(2,)
    Sigma = np.asarray(Sigma, dtype=float).reshape(2, 2)

    # ensure feasible evaluation point
    z0 = project_to_attack_region(z, mu, Sigma, epsilon)

    Er0 = estimate_E_reward_given_z(
        z=z0.astype(np.float32),
        delta_hat_adv=delta_hat_adv,
        P_adv=P_adv,
        R=R,
        env=env,
        model=model,
        device=device,
        n_mc=n_mc,
    )

    if use_target:
        J0 = (Er0 - float(M_star)) ** 2
    else:
        J0 = Er0

    grad = np.zeros(2, dtype=float)
    for j in range(2):
        ej = np.zeros(2, dtype=float)
        ej[j] = 1.0

        zp = project_to_attack_region(z0 + h * ej, mu, Sigma, epsilon)
        zm = project_to_attack_region(z0 - h * ej, mu, Sigma, epsilon)

        Erp = estimate_E_reward_given_z(
            z=zp.astype(np.float32),
            delta_hat_adv=delta_hat_adv,
            P_adv=P_adv,
            R=R,
            env=env,
            model=model,
            device=device,
            n_mc=n_mc,
        )
        Erm = estimate_E_reward_given_z(
            z=zm.astype(np.float32),
            delta_hat_adv=delta_hat_adv,
            P_adv=P_adv,
            R=R,
            env=env,
            model=model,
            device=device,
            n_mc=n_mc,
        )

        if use_target:
            Jp = (Erp - float(M_star)) ** 2
            Jm = (Erm - float(M_star)) ** 2
        else:
            Jp = Erp
            Jm = Erm

        grad[j] = (Jp - Jm) / (2.0 * h)

    return grad, float(J0)


@torch.no_grad()
def solve_pgd_attack_reward_over_ellipsoid(
    *,
    z_init: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    delta_hat_adv: np.ndarray,
    P_adv: np.ndarray,
    R: np.ndarray,
    env: AdvRL2DEnv,
    model: ActorCritic,
    device: torch.device,
    M_star: float,
    use_target: bool,
    eta: float,
    n_steps: int,
    n_mc: int,
    fd_h: float,
    verbose: bool = False,
    print_every: int = 1,     # por defecto imprime todas; pon 5/10 si molesta
) -> tuple[np.ndarray, dict]:
    """
    PGD to minimize J(z) over ellipsoid:
      J(z) = (E[r(z)] - M_star)^2   if use_target
           =  E[r(z)]              else
    """
    mu = np.asarray(mu, dtype=float).reshape(2,)
    Sigma = project_to_psd(np.asarray(Sigma, dtype=float))

    z_curr = project_to_attack_region(np.asarray(z_init, dtype=float), mu, Sigma, epsilon)

    hist_z = []
    hist_J = []
    hist_Er = []
    hist_grad = []

    for it in range(int(n_steps)):
        grad, J0 = fd_grad_objective_wrt_z(
            z=z_curr,
            mu=mu,
            Sigma=Sigma,
            epsilon=epsilon,
            delta_hat_adv=delta_hat_adv,
            P_adv=P_adv,
            R=R,
            env=env,
            model=model,
            device=device,
            M_star=M_star,
            use_target=use_target,
            n_mc=n_mc,
            h=fd_h,
        )

        # también calcula E[r] para que lo veas
        Er0 = estimate_E_reward_given_z(
            z=z_curr.astype(np.float32),
            delta_hat_adv=delta_hat_adv,
            P_adv=P_adv,
            R=R,
            env=env,
            model=model,
            device=device,
            n_mc=n_mc,
        )

        gnorm = float(np.linalg.norm(grad))

        if verbose and (it % max(int(print_every), 1) == 0) and (gnorm > 1e-8):
            if use_target:
                print(
                    f"[PGD it={it:04d}] z={z_curr}  "
                    f"J=(E[r]-M*)^2={J0:.6e}  E[r]={Er0:+.4f}  "
                    f"grad={grad}  ||grad||={gnorm:.3e}"
                )
            else:
                print(
                    f"[PGD it={it:04d}] z={z_curr}  "
                    f"E[r]={Er0:+.4f}  "
                    f"grad={grad}  ||grad||={gnorm:.3e}"
                )

        # gradient step + projection
        z_curr = z_curr - eta * grad
        z_curr = project_to_attack_region(z_curr, mu, Sigma, epsilon)

        hist_z.append(z_curr.copy())
        hist_J.append(float(J0))
        hist_Er.append(float(Er0))
        hist_grad.append(grad.copy())

    info = {
        "z_hist": np.asarray(hist_z, dtype=float),
        "J_hist": np.asarray(hist_J, dtype=float),
        "Er_hist": np.asarray(hist_Er, dtype=float),
        "grad_hist": np.asarray(hist_grad, dtype=float),
    }
    return z_curr.astype(np.float32), info


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

import copy

def inv_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=float))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(1.0 / w) @ V.T


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
    (Same idea as in your LGSSM script.)
    """
    y_candidate = np.asarray(y_candidate, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = project_to_psd(np.asarray(Sigma, dtype=float))

    Sinv = inv_psd(Sigma)
    d = y_candidate - mu
    maha = float(d.T @ Sinv @ d)
    if maha <= epsilon + tol:
        return y_candidate.copy()

    # diagonalize Sigma
    s, U = np.linalg.eigh(Sigma)
    s = np.maximum(s, 1e-12)
    r = U.T @ d

    def f(lam: float) -> float:
        # sum_i (s_i r_i^2 / (s_i + lam)^2) - epsilon
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


def _reward_for_candidate_z(
    *,
    model: ActorCritic,
    device: torch.device,
    env: AdvRL2DEnv,
    z: np.ndarray,
    delta_hat_adv_pred: np.ndarray,
    P_adv_pred: np.ndarray,
    R: np.ndarray,
) -> float:
    """
    Defines g(z) := instantaneous reward if the policy acts based on the attacked KF update with obs z.
    Uses a deepcopy of env so evaluations are non-compounding and share the same RNG state.
    """
    z = np.asarray(z, dtype=np.float32).reshape(2,)

    # 1) clone env so evaluating g(z) doesn't advance the real episode
    env_tmp = copy.deepcopy(env)

    # 2) KF update (attacked filter candidate)
    delta_hat_new, P_new = kf_update(delta_hat_adv_pred.copy(), P_adv_pred.copy(), z, R)

    # 3) policy action from attacked belief
    obs_policy = unit_dir(delta_hat_new)
    obs_t = torch.tensor(obs_policy, dtype=torch.float32, device=device)
    action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

    # 4) one-step reward
    _, r, _, _ = env_tmp.step(action)
    return float(r)


def attack_observation_for_instant_reward_pgd(
    *,
    model: ActorCritic,
    device: torch.device,
    env: AdvRL2DEnv,
    z_init: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    delta_hat_adv_pred: np.ndarray,
    P_adv_pred: np.ndarray,
    R: np.ndarray,
    n_steps: int = 40,
    eta: float = 0.15,
    h: float = 1e-2,
    minimize_reward: bool = True,   # True => make agent worse
) -> np.ndarray:
    """
    Projected GD on z in ellipsoid to (min or max) the instantaneous reward.
    Gradient by central finite differences (2D).
    """
    mu = np.asarray(mu, dtype=float).reshape(2,)
    Sigma = project_to_psd(np.asarray(Sigma, dtype=float))

    z = project_to_attack_region(z_init, mu, Sigma, epsilon).astype(np.float32)

    for _ in range(n_steps):
        # finite-diff gradient in R^2
        grad = np.zeros(2, dtype=np.float32)
        for j in range(2):
            ej = np.zeros(2, dtype=np.float32)
            ej[j] = h

            r_p = _reward_for_candidate_z(
                model=model, device=device, env=env,
                z=(z + ej),
                delta_hat_adv_pred=delta_hat_adv_pred,
                P_adv_pred=P_adv_pred,
                R=R,
            )
            r_m = _reward_for_candidate_z(
                model=model, device=device, env=env,
                z=(z - ej),
                delta_hat_adv_pred=delta_hat_adv_pred,
                P_adv_pred=P_adv_pred,
                R=R,
            )
            grad[j] = (r_p - r_m) / (2.0 * h)

        # step: minimize or maximize reward
        if minimize_reward:
            z = z - eta * grad
        else:
            z = z + eta * grad

        z = project_to_attack_region(z, mu, Sigma, epsilon).astype(np.float32)

    return z

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
    ADV agent (non-compounding, PGD reward-attack):

      - shadow KF: uses REAL noisy z_real always (prevents compounding)
      - attacked KF: uses z_adv (if attacked), else z_real
      - ellipsoid computed from SHADOW *predicted* belief (prior): mu=delta_hat_shadow, Sigma=P_shadow + R
      - objective: g(z) = instantaneous reward after (KF_update with z -> policy -> env.step),
                   optimized by PGD + projection to ellipsoid.

    Returns:
      traj_true:          (L,2) real positions
      traj_belief_shadow: (L,2) belief positions if NOT attacked (shadow KF)
      traj_belief_adv:    (L,2) belief positions of attacked KF
      attack_mask:        (L,)  1 if attacked at step, else 0
      ep_ret: float
      success: bool
    """
    traj_true = [env.x.copy()]  # REAL position
    traj_belief_shadow: List[np.ndarray] = []
    traj_belief_adv: List[np.ndarray] = []
    attack_mask: List[int] = []

    done = False
    ep_ret = 0.0
    success = False

    # ---- Init both KFs with first real noisy measurement (t=0)
    true_delta = (env.goal - env.x).astype(np.float32)
    z_real = true_delta + rng.normal(0.0, sigma, size=(2,)).astype(np.float32)

    delta_hat_shadow = z_real.copy()
    P_shadow = R.copy()

    delta_hat_adv = z_real.copy()
    P_adv = R.copy()

    for step_idx in range(env.cfg.max_steps):
        # ---- oracle true delta at CURRENT state
        true_delta = (env.goal - env.x).astype(np.float32)

        # ---- real sensor measurement
        z_real = true_delta + rng.normal(0.0, sigma, size=(2,)).astype(np.float32)

        # ---- ellipsoid params from SHADOW predicted belief (prior)
        # H = I, so innovation covariance is S = P + R
        S_pred = P_shadow + R

        # Attack starts from 2nd observation => step_idx >= 1
        do_attack = (step_idx >= 1) and (rng.random() < p_attack)
        attack_mask.append(1 if do_attack else 0)

        if not do_attack:
            z_adv = z_real.copy()
        else:
            try:
                # PGD attack: minimize reward objective over ellipsoid
                z_adv, _info = solve_pgd_attack_reward_over_ellipsoid(
                    z_init=z_real,               # start from real sensor
                    mu=delta_hat_shadow,         # ellipsoid center (shadow prior mean)
                    Sigma=S_pred,                # ellipsoid shape (innovation cov)
                    epsilon=epsilon,
                    delta_hat_adv=delta_hat_adv, # attacked KF prior
                    P_adv=P_adv,
                    R=R,
                    env=env,                     # current env state (solver clones internally)
                    model=model,
                    device=device,
                    M_star=REWARD_M_STAR,
                    use_target=REWARD_USE_TARGET,
                    eta=REWARD_ATTACK_LR,
                    n_steps=REWARD_ATTACK_STEPS,
                    n_mc=REWARD_ATTACK_MC,
                    fd_h=REWARD_ATTACK_FD_H,
                    verbose=True,
                    print_every=5,
                )
            except Exception:
                z_adv = z_real.copy()

        # ---- Update shadow with real sensor
        delta_hat_shadow, P_shadow = kf_update(delta_hat_shadow, P_shadow, z_real, R)

        # ---- Update attacked KF with attacked/noisy obs
        delta_hat_adv, P_adv = kf_update(delta_hat_adv, P_adv, z_adv, R)

        # ---- Record beliefs in POSITION space
        x_hat_shadow = (env.goal - delta_hat_shadow).astype(np.float32)
        x_hat_adv = (env.goal - delta_hat_adv).astype(np.float32)
        traj_belief_shadow.append(x_hat_shadow.copy())
        traj_belief_adv.append(x_hat_adv.copy())

        # ---- Policy uses attacked belief
        obs_policy = unit_dir(delta_hat_adv)
        obs_t = torch.tensor(obs_policy, dtype=torch.float32, device=device)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        theta = float(action[0])
        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        # ---- Step env
        _, r, done, info = env.step(action)
        ep_ret += float(r)
        success = bool(info.get("success", False))

        traj_true.append(env.x.copy())

        if done:
            break

        # ---- Predict both filters (delta_{t+1} = delta_t - u_t + w)
        delta_hat_shadow, P_shadow = kf_predict(delta_hat_shadow, P_shadow, u, Q)
        delta_hat_adv, P_adv = kf_predict(delta_hat_adv, P_adv, u, Q)

    # ---- Convert to arrays
    traj_true_arr = np.array(traj_true, dtype=np.float32)
    traj_shadow_arr = np.array(traj_belief_shadow, dtype=np.float32)
    traj_adv_arr = np.array(traj_belief_adv, dtype=np.float32)
    attack_mask_arr = np.array(attack_mask, dtype=np.int32)

    # ---- Pad beliefs to same length as traj_true
    L = traj_true_arr.shape[0]

    def _pad_to_len(arr: np.ndarray, L: int) -> np.ndarray:
        if arr.shape[0] == L:
            return arr
        if arr.shape[0] == 0:
            return np.repeat(env.x[None, :], L, axis=0).astype(np.float32)
        last = arr[-1]
        pad = np.repeat(last[None, :], L - arr.shape[0], axis=0)
        return np.vstack([arr, pad]).astype(np.float32)

    traj_shadow_arr = _pad_to_len(traj_shadow_arr, L)
    traj_adv_arr = _pad_to_len(traj_adv_arr, L)

    # ---- Pad attack mask too
    if attack_mask_arr.shape[0] < L:
        pad = np.zeros(L - attack_mask_arr.shape[0], dtype=np.int32)
        attack_mask_arr = np.concatenate([attack_mask_arr, pad], axis=0)

    return traj_true_arr, traj_shadow_arr, traj_adv_arr, attack_mask_arr, ep_ret, success

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

def plot_4_simulations_beliefs(model: ActorCritic, device: torch.device, sigma: float) -> None:
    _set_rcparams()

    cream = "#F7F1E3"
    cream2 = "#FBF7EE"
    ink = "#1F1F1F"
    grid = "#D9D2C3"

    c_true   = "#2F2F2F"   # real position
    c_shadow = "#2F6FA8"   # non-attacked belief
    c_adv    = "#C85B4F"   # attacked belief
    c_goal   = "#D4A017"   # goal
    c_marks  = "#C85B4F"   # attack markers

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
            model, device, env, R, Q, rng_adv, sigma, EPSILON_KKT, P_ATTACK
        )

        # --- plot
        ax.plot(traj_true[:, 0], traj_true[:, 1],
                color=c_true, linewidth=2.2, label="Real position" if i == 0 else None)

        ax.plot(traj_shadow[:, 0], traj_shadow[:, 1],
                color=c_shadow, linewidth=2.0, linestyle=":",
                label="Belief (shadow, no-attack)" if i == 0 else None)

        ax.plot(traj_adv[:, 0], traj_adv[:, 1],
                color=c_adv, linewidth=2.2, linestyle="--",
                label="Belief (attacked)" if i == 0 else None)

        # mark attacked steps along REAL trajectory
        idx = np.where(atk_mask[:traj_true.shape[0]] == 1)[0]
        if idx.size > 0:
            ax.scatter(traj_true[idx, 0], traj_true[idx, 1],
                       s=28, marker="x", color=c_marks, alpha=0.9,
                       label="Attack step" if i == 0 else None)

        # start + goal
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
        f"Belief vs real position (σ={sigma}, ε={EPSILON_KKT}, p_attack={P_ATTACK})",
        fontsize=13, color=ink
    )

    os.makedirs(os.path.join(_PROJECT_ROOT, OUT_DIR), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, OUT_DIR, "kkt_4sims_belief_vs_real.png")
    plt.savefig(out_path, dpi=260, facecolor=fig.get_facecolor())
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
        _, _, _, _, ret_a, ok_a = rollout_adv(model, device, env_adv, R, Q, rng_adv, sigma, EPSILON_KKT, P_ATTACK)
        rets_clean.append(ret_c)
        rets_noisy.append(ret_n)
        rets_adv.append(ret_a)

        succ_clean += int(ok_c)
        succ_noisy += int(ok_n)
        succ_adv += int(ok_a)
        if (ep + 1) % 10 == 0 or ep == EVAL_EPISODES - 1:
            print(f"Episode {ep+1}/{EVAL_EPISODES} | Clean: {ret_c:.1f} ({'OK' if ok_c else 'X'}) | "
                  f"Noisy: {ret_n:.1f} ({'OK' if ok_n else 'X'}) | Adv: {ret_a:.1f} ({'OK' if ok_a else 'X'})")

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
    out_path = os.path.join(_PROJECT_ROOT, OUT_DIR, "kkt_rewards_over_episodes_gradrew.png")
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

    plot_4_simulations_beliefs(model, device, sigma=SIGMA_OBS)
    plot_rewards_over_episodes(model, device, sigma=SIGMA_OBS)


if __name__ == "__main__":
    main()
