#!/usr/bin/env python3
"""
Evaluate 4D observation attacks against the wind policy with a KF baseline.

The environment and trained policy are exactly the same as in ``v2_wind``:

    x_{t+1} = x_t + a_t + wind_t + q_t
    y_t     = x_t + r_t

The difference is the attack surface. Instead of perturbing only the measured
position, projected gradient descent attacks the full 4D policy observation

    z_t = [(goal - y_t) / goal_r_max, wind_x_t, wind_y_t]

inside a 4D ellipsoid around the nominal observation. The first two attacked
coordinates alter the KF position update; the last two alter the wind seen by
the policy and by the predictor used for the next-step belief.

This variant also adds a random baseline: whenever an attack is triggered, we
can replace the PGD adversarial point by a random point sampled from the same
4D ellipsoid defined by ``attack_std`` and ``attack_eps``. The script
therefore compares:

    1) clean
    2) noisy + KF
    3) PGD attack + KF
    4) random ellipsoid point + KF
"""

# Eval_clean_noisyKF_attackKF.py
#
# HARD-CODED PATHS:
#   - AdvRL_wind.py    : ./AdvRL_wind.py  (same folder as this script)
#   - Model checkpoint : ./outputs/saved_models/AdvRL_v2_policy.pt
#   - Output figures   : ./outputs/figures/
#
# Curves plotted:
#   1) clean
#   2) noisy + KF
#   3) attack 4D + KF
#   4) random ellipsoid + KF
#
# Interpretation:
#   - clean:
#       policy receives the clean env observation directly
#   - noisy + KF:
#       position observation is corrupted by Gaussian noise;
#       policy acts on the KF filtered position estimate
#   - attack + KF:
#       at each attacked step, a 4D observation z_t' is found with PGD
#       by minimizing the critic value under the posterior induced by its
#       attacked position component and the attacked wind seen by the policy
#   - random ellipsoid + KF:
#       at each attacked step, a 4D observation z_t' is sampled uniformly
#       from the same attack ellipsoid used by the adversary
#
# State model used by the KF attacker/filter:
#   x_t in R^2 = latent position
#   x_{t+1} = x_t + a_t + wind_t + q_t
#   y_t     = x_t + r_t
#
# Policy observation attacked by this script:
#   z_t = [ (goal - p)/goal_r_max , wind_x_t, wind_y_t ]

from __future__ import annotations

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

# Random-noise curve
NOISE_STD = 0.5

# Adversarial attack geometry
ATTACK_PROB = 0.15
ATTACK_STD = NOISE_STD
ATTACK_EPS = 2.488   # ~ chi-square 95% in 4D

# KF model
KF_MEAS_STD = NOISE_STD
KF_PROC_STD = 0.03

# PGD / MC for attack
PGD_STEPS = 65
PGD_STEP_SIZE = 0.25
MC_SAMPLES = 256

# Accumulated reward plot
N_EPISODES = 50
SEED0 = 1_000

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from AdvSSM.io_utils import data_path_for_plot, load_npz, save_npz

ADV_FILE_PATH = os.path.abspath(os.path.join(_THIS_DIR, "AdvRL_wind.py"))
MODEL_PATH = os.path.abspath(os.path.join(_THIS_DIR, "outputs", "saved_models", "AdvRL_v2_policy.pt"))
FIGURES_DIR = os.path.abspath(os.path.join(_THIS_DIR, "outputs", "figures"))


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


def sample_uniform_from_attack_region(
    center: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample uniformly from the boundary of:
        { y : (y-center)^T Sigma^{-1} (y-center) <= epsilon }

    The direction is sampled uniformly on the unit sphere and then mapped to
    the ellipsoid boundary with Mahalanobis radius exactly `sqrt(epsilon)`.
    """
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    Sigma = project_to_psd(Sigma).astype(np.float64)

    if epsilon <= 0.0:
        return center.astype(np.float32)

    dim = center.size
    direction = rng.normal(size=dim)
    norm = np.linalg.norm(direction)

    while norm <= 1e-12:
        direction = rng.normal(size=dim)
        norm = np.linalg.norm(direction)

    direction = direction / norm

    # Boundary-only epsilon perturbations use the full admissible Mahalanobis
    # radius instead of sampling a smaller interior radius.
    ball_sample = np.sqrt(float(epsilon)) * direction
    L = sqrtm_psd(Sigma).astype(np.float64)
    sample = center + L @ ball_sample
    return sample.astype(np.float32)


# ============================================================
# Env wrapper: separate RNG for observation noise only
# so wind / env randomness stays paired across evaluations
# ============================================================

def make_env_with_separate_obs_rng(AdvRL2DEnv_base):
    class AdvRL2DEnv_SeparateObsRNG(AdvRL2DEnv_base):
        def __init__(self, cfg):
            self.rng_obs = np.random.default_rng(int(cfg.seed) + 12345)
            super().__init__(cfg)

        def _measure_observation_state(self) -> np.ndarray:
            y = (self.F @ self.x_ssm).astype(np.float32)
            return (y + self._sample_observation_noise()).astype(np.float32)

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


def obs_to_wind(obs: np.ndarray) -> np.ndarray:
    """Recover the wind components seen by the policy from a 4D observation."""
    return np.asarray(obs[2:4], dtype=np.float32)


def split_policy_obs(
    obs: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4D policy observation into measured position and observed wind."""
    return (
        obs_to_position(obs, goal, goal_r_max),
        obs_to_wind(obs),
    )


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


def measurement_state_to_policy_obs(
    meas_state: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
) -> np.ndarray:
    """Map the 4D measured physical state into the 4D policy observation."""
    meas_state = np.asarray(meas_state, dtype=np.float32).reshape(4)
    return build_policy_obs_from_position(
        pos=meas_state[:2],
        goal=np.asarray(goal, dtype=np.float32).reshape(2),
        goal_r_max=goal_r_max,
        wind_xy=meas_state[2:4],
    )


def policy_obs_to_measurement_state(
    obs: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
) -> np.ndarray:
    """Invert the policy observation back into the 4D measured state."""
    pos, wind_xy = split_policy_obs(obs, goal, goal_r_max)
    return np.concatenate([pos, wind_xy]).astype(np.float32)


# ============================================================
# Critic helper
# ============================================================

def critic_values(model, obs_batch: torch.Tensor) -> torch.Tensor:
    """
    model(obs) -> (mu, std, v)
    Returns v with shape (N,)
    """
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.unsqueeze(0)

    mu, std, v = model(obs_batch)

    if v.ndim == 2 and v.shape[1] == 1:
        v = v.squeeze(-1)
    return v


# ============================================================
# KF for latent 2D position
# ============================================================

def kf_update_position(
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measurement model:
        y_t = x_t + r_t
    """
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
    """
    Dynamics model used by filter/attacker:
        x_{t+1} = x_t + action_t + wind_t + q_t
    """
    drift = np.asarray(action, dtype=np.float32) + np.asarray(wind_xy, dtype=np.float32)
    m_next = m_post + drift
    P_next = P_post + Q
    P_next = project_to_psd(P_next)

    return m_next.astype(np.float32), P_next.astype(np.float32)


def kf_update_state(
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one 4D KF update in the shared physical-state coordinates."""
    m_pred = np.asarray(m_pred, dtype=np.float32).reshape(4)
    P_pred = project_to_psd(np.asarray(P_pred, dtype=np.float32))
    y_obs = np.asarray(y_obs, dtype=np.float32).reshape(4)
    R = project_to_psd(np.asarray(R, dtype=np.float32))

    I4 = np.eye(4, dtype=np.float32)
    S = project_to_psd(P_pred + R)
    K = P_pred @ np.linalg.inv(S)
    innov = y_obs - m_pred
    m_post = m_pred + K @ innov
    joseph_left = I4 - K
    P_post = joseph_left @ P_pred @ joseph_left.T + K @ R @ K.T
    return m_post.astype(np.float32), project_to_psd(P_post), K.astype(np.float32)


def kf_predict_state(
    m_post: np.ndarray,
    P_post: np.ndarray,
    A_t: np.ndarray,
    B: np.ndarray,
    action: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one 4D KF prediction using the shared physical SSM."""
    m_post = np.asarray(m_post, dtype=np.float32).reshape(4)
    P_post = project_to_psd(np.asarray(P_post, dtype=np.float32))
    A_t = np.asarray(A_t, dtype=np.float32).reshape(4, 4)
    B = np.asarray(B, dtype=np.float32).reshape(4, 2)
    action = np.asarray(action, dtype=np.float32).reshape(2)
    Q = project_to_psd(np.asarray(Q, dtype=np.float32))

    m_next = A_t @ m_post + B @ action
    P_next = project_to_psd(A_t @ P_post @ A_t.T + Q)
    return m_next.astype(np.float32), P_next.astype(np.float32)


# ============================================================
# Expected critic value under posterior
#   mu_V(z') = E_{x ~ p(x | z'_pos, history)} [V([delta(x), z'_wind])]
# ============================================================

def expected_critic_value_mc(
    *,
    model,
    obs_adv_torch: torch.Tensor,        # (4,), requires_grad=True
    m_pred: np.ndarray,                 # (2,)
    P_pred: np.ndarray,                 # (2,2)
    R: np.ndarray,                      # (2,2)
    goal: np.ndarray,                   # (2,)
    goal_r_max: float,
    xi_torch: torch.Tensor,             # (MC,2), fixed common random numbers
    device: str = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Compute the critic expectation induced by a 4D attacked observation.

    The attacked observation contains both the measured position proxy and the
    wind reported to the policy. Only the attacked position enters the KF
    update; the attacked wind is appended to every sampled policy input.

    Returns:
      mu_V_torch: scalar torch tensor
      m_post_np:  posterior mean
      P_post_np:  posterior covariance
    """
    dev = torch.device(device)

    m_pred_t = torch.tensor(m_pred, dtype=torch.float32, device=dev)
    P_pred_t = torch.tensor(P_pred, dtype=torch.float32, device=dev)
    R_t = torch.tensor(R, dtype=torch.float32, device=dev)
    goal_t = torch.tensor(goal, dtype=torch.float32, device=dev)

    delta_adv_t = obs_adv_torch[:2]
    wind_adv_t = obs_adv_torch[2:4]
    y_adv_t = goal_t - float(goal_r_max) * delta_adv_t

    S_t = P_pred_t + R_t
    K_t = P_pred_t @ torch.linalg.inv(S_t)

    innov_t = y_adv_t - m_pred_t
    m_post_t = m_pred_t + K_t @ innov_t

    I2 = torch.eye(2, dtype=torch.float32, device=dev)
    P_post_t = (I2 - K_t) @ P_pred_t

    P_post_np = P_post_t.detach().cpu().numpy().astype(np.float32)
    L_np = sqrtm_psd(P_post_np)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=dev)

    x_samples = m_post_t.unsqueeze(0) + xi_torch @ L_t.T  # (MC,2)

    delta = (goal_t.unsqueeze(0) - x_samples) / float(goal_r_max)
    wind_batch = wind_adv_t.unsqueeze(0).repeat(x_samples.shape[0], 1)
    obs_batch = torch.cat([delta, wind_batch], dim=1)

    v_batch = critic_values(model, obs_batch)
    mu_V = v_batch.mean()

    return mu_V, m_post_t.detach().cpu().numpy().astype(np.float32), P_post_np


def expected_critic_value_mc_state4d(
    *,
    model,
    obs_adv_torch: torch.Tensor,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
    xi_torch: torch.Tensor,
    device: str = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Critic expectation induced by an attacked 4D policy observation.

    The posterior is computed in the shared physical state
        s_t = [p_x,t, p_y,t, d_x,t, d_y,t]^T,
    and the Monte Carlo samples are mapped back into the policy coordinates only
    right before evaluating the critic.
    """
    dev = torch.device(device)

    m_pred_t = torch.tensor(m_pred, dtype=torch.float32, device=dev)
    P_pred_t = torch.tensor(P_pred, dtype=torch.float32, device=dev)
    R_t = torch.tensor(R, dtype=torch.float32, device=dev)
    goal_t = torch.tensor(goal, dtype=torch.float32, device=dev)

    y_adv_t = torch.cat(
        [
            goal_t - float(goal_r_max) * obs_adv_torch[:2],
            obs_adv_torch[2:4],
        ]
    )

    S_t = P_pred_t + R_t
    K_t = P_pred_t @ torch.linalg.inv(S_t)
    innov_t = y_adv_t - m_pred_t
    m_post_t = m_pred_t + K_t @ innov_t

    I4 = torch.eye(4, dtype=torch.float32, device=dev)
    joseph_left = I4 - K_t
    P_post_t = joseph_left @ P_pred_t @ joseph_left.T + K_t @ R_t @ K_t.T

    P_post_np = project_to_psd(P_post_t.detach().cpu().numpy().astype(np.float32))
    L_np = sqrtm_psd(P_post_np)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=dev)

    state_samples = m_post_t.unsqueeze(0) + xi_torch @ L_t.T
    delta_batch = (goal_t.unsqueeze(0) - state_samples[:, :2]) / float(goal_r_max)
    obs_batch = torch.cat([delta_batch, state_samples[:, 2:4]], dim=1)
    v_batch = critic_values(model, obs_batch)
    mu_V = v_batch.mean()

    return mu_V, m_post_t.detach().cpu().numpy().astype(np.float32), P_post_np


def pgd_attack_on_expected_value(
    *,
    model,
    obs_nom: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
    attack_sigma: np.ndarray,
    attack_center: np.ndarray | None = None,
    attack_eps: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    rng_seed: int,
    device: str = "cpu",
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Solve approximately with PGD:
        min_{z' in ellipsoid(center, attack_sigma, attack_eps)}
            E_{x ~ p(x | z'_pos, history)} [V([delta(x), z'_wind])]

    If `attack_center` is not provided, the ellipsoid is centered at the
    nominal observation `obs_nom`, preserving the original script behavior.

    Returns:
      obs_star    attacked 4D policy observation
      obj_star    final estimated expectation
      m_post_star posterior mean at y_star
      P_post_star posterior covariance at y_star
    """
    dev = torch.device(device)

    # Common random numbers inside PGD
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng_seed))
    xi_torch = torch.randn((mc_samples, 2), generator=gen, device=dev, dtype=torch.float32)

    center = obs_nom.astype(np.float32).copy() if attack_center is None else np.asarray(attack_center, dtype=np.float32).copy()
    obs_curr_np = project_to_attack_region(
        y_candidate=obs_nom.astype(np.float32).copy(),
        center=center,
        Sigma=attack_sigma,
        epsilon=attack_eps,
    )

    best_obs = obs_curr_np.copy()
    best_obj = None
    best_m_post = None
    best_P_post = None

    for _ in range(pgd_steps):
        obs_t = torch.tensor(obs_curr_np, dtype=torch.float32, device=dev, requires_grad=True)

        mu_V_t, m_post_np, P_post_np = expected_critic_value_mc(
            model=model,
            obs_adv_torch=obs_t,
            m_pred=m_pred,
            P_pred=P_pred,
            R=R,
            goal=goal,
            goal_r_max=goal_r_max,
            xi_torch=xi_torch,
            device=device,
        )

        obj = mu_V_t
        obj.backward()

        grad = obs_t.grad.detach().cpu().numpy().astype(np.float32)
        obs_next = obs_curr_np - float(pgd_step_size) * grad

        obs_next = project_to_attack_region(
            y_candidate=obs_next,
            center=center,
            Sigma=attack_sigma,
            epsilon=attack_eps,
        )

        obj_val = float(obj.detach().cpu().item())
        if best_obj is None or obj_val < best_obj:
            best_obj = obj_val
            best_obs = obs_curr_np.copy()
            best_m_post = m_post_np.copy()
            best_P_post = P_post_np.copy()

        obs_curr_np = obs_next

    # final evaluation at last point
    obs_t = torch.tensor(obs_curr_np, dtype=torch.float32, device=dev, requires_grad=True)
    mu_V_t, m_post_np, P_post_np = expected_critic_value_mc(
        model=model,
        obs_adv_torch=obs_t,
        m_pred=m_pred,
        P_pred=P_pred,
        R=R,
        goal=goal,
        goal_r_max=goal_r_max,
        xi_torch=xi_torch,
        device=device,
    )
    obj_val = float(mu_V_t.detach().cpu().item())

    if best_obj is None or obj_val < best_obj:
        best_obj = obj_val
        best_obs = obs_curr_np.copy()
        best_m_post = m_post_np.copy()
        best_P_post = P_post_np.copy()

    return best_obs, float(best_obj), best_m_post, best_P_post


def pgd_attack_on_expected_value_state4d(
    *,
    model,
    obs_nom: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
    attack_sigma: np.ndarray,
    attack_center: np.ndarray | None = None,
    attack_eps: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    rng_seed: int,
    device: str = "cpu",
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    PGD attack that keeps the posterior model in the shared 4D physical state.
    """
    dev = torch.device(device)

    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng_seed))
    xi_torch = torch.randn((mc_samples, 4), generator=gen, device=dev, dtype=torch.float32)

    center = obs_nom.astype(np.float32).copy() if attack_center is None else np.asarray(attack_center, dtype=np.float32).copy()
    obs_curr_np = project_to_attack_region(
        y_candidate=obs_nom.astype(np.float32).copy(),
        center=center,
        Sigma=attack_sigma,
        epsilon=attack_eps,
    )

    best_obs = obs_curr_np.copy()
    best_obj = None
    best_m_post = None
    best_P_post = None

    for _ in range(pgd_steps):
        obs_t = torch.tensor(obs_curr_np, dtype=torch.float32, device=dev, requires_grad=True)

        mu_V_t, m_post_np, P_post_np = expected_critic_value_mc_state4d(
            model=model,
            obs_adv_torch=obs_t,
            m_pred=m_pred,
            P_pred=P_pred,
            R=R,
            goal=goal,
            goal_r_max=goal_r_max,
            xi_torch=xi_torch,
            device=device,
        )

        mu_V_t.backward()
        grad = obs_t.grad.detach().cpu().numpy().astype(np.float32)
        obs_next = obs_curr_np - float(pgd_step_size) * grad
        obs_next = project_to_attack_region(
            y_candidate=obs_next,
            center=center,
            Sigma=attack_sigma,
            epsilon=attack_eps,
        )

        obj_val = float(mu_V_t.detach().cpu().item())
        if best_obj is None or obj_val < best_obj:
            best_obj = obj_val
            best_obs = obs_curr_np.copy()
            best_m_post = m_post_np.copy()
            best_P_post = P_post_np.copy()

        obs_curr_np = obs_next

    obs_t = torch.tensor(obs_curr_np, dtype=torch.float32, device=dev, requires_grad=True)
    mu_V_t, m_post_np, P_post_np = expected_critic_value_mc_state4d(
        model=model,
        obs_adv_torch=obs_t,
        m_pred=m_pred,
        P_pred=P_pred,
        R=R,
        goal=goal,
        goal_r_max=goal_r_max,
        xi_torch=xi_torch,
        device=device,
    )
    obj_val = float(mu_V_t.detach().cpu().item())

    if best_obj is None or obj_val < best_obj:
        best_obj = obj_val
        best_obs = obs_curr_np.copy()
        best_m_post = m_post_np.copy()
        best_P_post = P_post_np.copy()

    return best_obs, float(best_obj), best_m_post, best_P_post


# ============================================================
# Rollouts
# ============================================================

@torch.no_grad()
def rollout_episode_return_clean(env, model, device: str = "cpu") -> float:
    """
    Standard deterministic evaluation using clean observation directly.
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


def rollout_episode_return_noisy_kf(
    env,
    model,
    kf_meas_std: float,
    kf_proc_std: float,
    device: str = "cpu",
) -> float:
    """
    Noisy observation + KF:
      - env returns noisy position observation
      - filter updates with that noisy observation
      - policy acts on posterior mean m_post
    """
    dev = torch.device(device)
    obs = env.reset()

    goal = env.goal.copy()
    goal_r_max = float(env.cfg.goal_r_max)

    R = (float(kf_meas_std) ** 2) * np.eye(2, dtype=np.float32)
    Q = (float(kf_proc_std) ** 2) * np.eye(2, dtype=np.float32)

    # initial prior from first observed measurement
    y0 = obs_to_position(obs, goal, goal_r_max)
    m_pred = y0.copy()
    P_pred = R.copy()

    ep_return = 0.0

    for _ in range(env.cfg.max_steps):
        y_obs, wind_xy = split_policy_obs(obs, goal, goal_r_max)

        m_post, P_post, _K = kf_update_position(
            m_pred=m_pred,
            P_pred=P_pred,
            y_obs=y_obs,
            R=R,
        )

        obs_filt = build_policy_obs_from_position(
            pos=m_post,
            goal=goal,
            goal_r_max=goal_r_max,
            wind_xy=wind_xy,
        )

        with torch.no_grad():
            obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
            action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        m_pred, P_pred = kf_predict_position(
            m_post=m_post,
            P_post=P_post,
            action=action,
            wind_xy=wind_xy,
            Q=Q,
        )

        obs, reward, done, _info = env.step(action)
        ep_return += float(reward)

        if done:
            break

    return float(ep_return)

def rollout_episode_return_attack_kf(
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
) -> float:
    """
    Attack + KF with probabilistic attacks:
      - env is clean
      - the first observation is NEVER attacked
      - first step acts from the initial nominal/KF state so the episode starts at (0,0)
      - from the second step onward, each observation is attacked with probability attack_prob
      - if not attacked, the KF updates with the nominal measurement
      - policy acts on the posterior mean m_post
    """
    dev = torch.device(device)
    obs = env.reset()

    goal = env.goal.copy()
    goal_r_max = float(env.cfg.goal_r_max)

    R = (float(kf_meas_std) ** 2) * np.eye(2, dtype=np.float32)
    Q = (float(kf_proc_std) ** 2) * np.eye(2, dtype=np.float32)
    Sigma_attack = (float(attack_std) ** 2) * np.eye(4, dtype=np.float32)

    # RNG for deciding whether to attack each step
    rng_attack_gate = np.random.default_rng(int(seed_for_attack) + 777777)
    rng_obs_noise = np.random.default_rng(int(seed_for_attack) + 888888)

    # ---------------------------------------------------------
    # Initial prior from first nominal observation (never attack)
    # ---------------------------------------------------------
    y0, wind_xy = split_policy_obs(obs, goal, goal_r_max)

    m_post, P_post, _K = kf_update_position(
        m_pred=y0.copy(),
        P_pred=R.copy(),
        y_obs=y0,
        R=R,
    )

    ep_return = 0.0
    step_idx = 0

    # ---------------------------------------------------------
    # FIRST ACTION: no attack on initial state
    # ---------------------------------------------------------
    obs_filt = build_policy_obs_from_position(
        pos=m_post,
        goal=goal,
        goal_r_max=goal_r_max,
        wind_xy=wind_xy,
    )

    with torch.no_grad():
        obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

    m_pred, P_pred = kf_predict_position(
        m_post=m_post,
        P_post=P_post,
        action=action,
        wind_xy=wind_xy,
        Q=Q,
    )

    obs, reward, done, _info = env.step(action)
    ep_return += float(reward)

    if done:
        return float(ep_return)

    step_idx = 1

    # ---------------------------------------------------------
    # From now on: attack each observation with probability p
    # ---------------------------------------------------------
    for _ in range(1, env.cfg.max_steps):
        y_clean, wind_xy_clean = split_policy_obs(obs, goal, goal_r_max)

        do_attack = bool(rng_attack_gate.random() < float(attack_prob))

        noise = rng_obs_noise.normal(0.0, float(kf_meas_std), size=(2,)).astype(np.float32)

        y_noisy = y_clean + noise
        obs_clean = build_policy_obs_from_position(
            pos=y_clean,
            goal=goal,
            goal_r_max=goal_r_max,
            wind_xy=wind_xy_clean,
        )


        if do_attack:
            obs_star, _obj_star, m_post, P_post = pgd_attack_on_expected_value(
                model=model,
                obs_nom=obs_clean,
                m_pred=m_pred,
                P_pred=P_pred,
                R=R,
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
            wind_xy_used = obs_to_wind(obs_star)
        else:
            m_post, P_post, _K = kf_update_position(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=y_noisy,
                R=R,
            )
            wind_xy_used = wind_xy_clean

        obs_filt = build_policy_obs_from_position(
            pos=m_post,
            goal=goal,
            goal_r_max=goal_r_max,
            wind_xy=wind_xy_used,
        )

        with torch.no_grad():
            obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
            action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        m_pred, P_pred = kf_predict_position(
            m_post=m_post,
            P_post=P_post,
            action=action,
            wind_xy=wind_xy_used,
            Q=Q,
        )

        obs, reward, done, _info = env.step(action)
        ep_return += float(reward)
        step_idx += 1

        if done:
            break

    return float(ep_return)


def rollout_episode_return_random_attack_kf(
    env,
    model,
    attack_std: float,
    attack_eps: float,
    attack_prob: float,
    kf_meas_std: float,
    kf_proc_std: float,
    seed_for_attack: int,
    device: str = "cpu",
) -> float:
    """
    Random-ellipsoid baseline + KF with probabilistic attacks:
      - env is clean
      - the first observation is NEVER attacked
      - first step acts from the initial nominal/KF state so the episode starts at (0,0)
      - from the second step onward, each observation is attacked with probability attack_prob
      - if an attack is triggered, sample a random 4D observation on the same ellipsoid boundary
      - if not attacked, the KF updates with the nominal noisy measurement
      - policy acts on the posterior mean m_post
    """
    dev = torch.device(device)
    obs = env.reset()

    goal = env.goal.copy()
    goal_r_max = float(env.cfg.goal_r_max)

    R = (float(kf_meas_std) ** 2) * np.eye(2, dtype=np.float32)
    Q = (float(kf_proc_std) ** 2) * np.eye(2, dtype=np.float32)
    Sigma_attack = (float(attack_std) ** 2) * np.eye(4, dtype=np.float32)

    # Reuse the same attack gate and noise structure as the PGD baseline.
    rng_attack_gate = np.random.default_rng(int(seed_for_attack) + 777777)
    rng_obs_noise = np.random.default_rng(int(seed_for_attack) + 888888)
    rng_attack_sample = np.random.default_rng(int(seed_for_attack) + 999999)

    y0, wind_xy = split_policy_obs(obs, goal, goal_r_max)

    m_post, P_post, _K = kf_update_position(
        m_pred=y0.copy(),
        P_pred=R.copy(),
        y_obs=y0,
        R=R,
    )

    ep_return = 0.0

    obs_filt = build_policy_obs_from_position(
        pos=m_post,
        goal=goal,
        goal_r_max=goal_r_max,
        wind_xy=wind_xy,
    )

    with torch.no_grad():
        obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

    m_pred, P_pred = kf_predict_position(
        m_post=m_post,
        P_post=P_post,
        action=action,
        wind_xy=wind_xy,
        Q=Q,
    )

    obs, reward, done, _info = env.step(action)
    ep_return += float(reward)

    if done:
        return float(ep_return)

    for _ in range(1, env.cfg.max_steps):
        y_clean, wind_xy_clean = split_policy_obs(obs, goal, goal_r_max)

        do_attack = bool(rng_attack_gate.random() < float(attack_prob))
        noise = rng_obs_noise.normal(0.0, float(kf_meas_std), size=(2,)).astype(np.float32)
        y_noisy = y_clean + noise

        obs_clean = build_policy_obs_from_position(
            pos=y_clean,
            goal=goal,
            goal_r_max=goal_r_max,
            wind_xy=wind_xy_clean,
        )

        if do_attack:
            obs_random = sample_uniform_from_attack_region(
                center=obs_clean,
                Sigma=Sigma_attack,
                epsilon=attack_eps,
                rng=rng_attack_sample,
            )
            y_random, wind_xy_used = split_policy_obs(obs_random, goal, goal_r_max)
            m_post, P_post, _K = kf_update_position(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=y_random,
                R=R,
            )
        else:
            m_post, P_post, _K = kf_update_position(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=y_noisy,
                R=R,
            )
            wind_xy_used = wind_xy_clean

        obs_filt = build_policy_obs_from_position(
            pos=m_post,
            goal=goal,
            goal_r_max=goal_r_max,
            wind_xy=wind_xy_used,
        )

        with torch.no_grad():
            obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
            action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        m_pred, P_pred = kf_predict_position(
            m_post=m_post,
            P_post=P_post,
            action=action,
            wind_xy=wind_xy_used,
            Q=Q,
        )

        obs, reward, done, _info = env.step(action)
        ep_return += float(reward)

        if done:
            break

    return float(ep_return)
# ============================================================
# Plot accumulated reward
# ============================================================

def plot_accumulated_reward_clean_noisykf_attackkf_randomkf(
    AdvRLEnvConfig,
    EnvCleanClass,
    EnvNoisyClass,
    base_cfg,
    model,
    noise_std: float,
    attack_std: float,
    attack_eps: float,
    attack_prob: float,
    kf_meas_std: float,
    kf_proc_std: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    n_episodes: int,
    seed0: int,
    device: str,
    results_dir: str,
):
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(
        results_dir,
        "eval_accumulated_reward_clean_noisyKF_attackKF_randomEllipseKF_4dobs.png",
    )
    data_path = data_path_for_plot(outpath)
    colors = {
        "clean": "#9ecae1",
        "noisy_kf": "#a8ddb5",
        "attack_kf": "#fbb4ae",
        "random_kf": "#decbe4",
    }

    if os.path.exists(data_path):
        print(f"[cache] loading data: {data_path}")
        data = load_npz(data_path)
        acc_clean = np.asarray(data["acc_clean"], dtype=float)
        acc_noisy_kf = np.asarray(data["acc_noisy_kf"], dtype=float)
        acc_attack_kf = np.asarray(data["acc_attack_kf"], dtype=float)
        acc_random_attack_kf = np.asarray(data["acc_random_attack_kf"], dtype=float)

        fig = plt.figure(figsize=(12, 5))
        plt.plot(acc_clean, linewidth=1.8, color=colors["clean"], label="clean")
        plt.plot(
            acc_noisy_kf,
            linewidth=1.8,
            color=colors["noisy_kf"],
            label="noisy + KF",
        )
        plt.plot(
            acc_attack_kf,
            linewidth=1.8,
            color=colors["attack_kf"],
            label="PGD attack + KF",
        )
        plt.plot(
            acc_random_attack_kf,
            linewidth=1.8,
            color=colors["random_kf"],
            label="random ellipse + KF",
        )

        plt.grid(True, alpha=0.25)
        plt.xlabel("Episode")
        plt.ylabel("Accumulated reward")
        plt.legend(loc="lower right")

        fig.savefig(outpath, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {outpath}")
        return

    acc_clean = []
    acc_noisy_kf = []
    acc_attack_kf = []
    acc_random_attack_kf = []

    total_clean = 0.0
    total_noisy_kf = 0.0
    total_attack_kf = 0.0
    total_random_attack_kf = 0.0

    for k in range(n_episodes):
        seed = int(seed0 + k)

        cfg_clean = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_noisy = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_attack = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_random = AdvRLEnvConfig(**{**asdict(base_cfg)})

        cfg_clean.seed = seed
        cfg_clean.obs_noise_std = 0.0

        cfg_noisy.seed = seed
        cfg_noisy.obs_noise_std = float(noise_std)

        cfg_attack.seed = seed
        cfg_attack.obs_noise_std = 0.0

        cfg_random.seed = seed
        cfg_random.obs_noise_std = 0.0

        env_clean = EnvCleanClass(cfg_clean)
        env_noisy = EnvNoisyClass(cfg_noisy)
        env_attack = EnvCleanClass(cfg_attack)
        env_random = EnvCleanClass(cfg_random)

        ret_clean = rollout_episode_return_clean(
            env_clean,
            model,
            device=device,
        )

        ret_noisy_kf = rollout_episode_return_noisy_kf(
            env_noisy,
            model,
            kf_meas_std=kf_meas_std,
            kf_proc_std=kf_proc_std,
            device=device,
        )

        ret_attack_kf = rollout_episode_return_attack_kf(
            env_attack,
            model,
            attack_std=attack_std,
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            kf_meas_std=kf_meas_std,
            kf_proc_std=kf_proc_std,
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            mc_samples=mc_samples,
            seed_for_attack=seed,
            device=device,
        )

        ret_random_attack_kf = rollout_episode_return_random_attack_kf(
            env_random,
            model,
            attack_std=attack_std,
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            kf_meas_std=kf_meas_std,
            kf_proc_std=kf_proc_std,
            seed_for_attack=seed,
            device=device,
        )

        total_clean += float(ret_clean)
        total_noisy_kf += float(ret_noisy_kf)
        total_attack_kf += float(ret_attack_kf)
        total_random_attack_kf += float(ret_random_attack_kf)

        acc_clean.append(total_clean)
        acc_noisy_kf.append(total_noisy_kf)
        acc_attack_kf.append(total_attack_kf)
        acc_random_attack_kf.append(total_random_attack_kf)

        if (k + 1) % 25 == 0 or (k + 1) == n_episodes:
            print(
                f"[{k+1:4d}/{n_episodes}] "
                f"clean={total_clean:.1f} | "
                f"noisy+KF={total_noisy_kf:.1f} | "
                f"attack+KF={total_attack_kf:.1f} | "
                f"random+KF={total_random_attack_kf:.1f}"
            )

    fig = plt.figure(figsize=(12, 5))
    plt.plot(acc_clean, linewidth=1.8, color=colors["clean"], label="clean")
    plt.plot(
        acc_noisy_kf,
        linewidth=1.8,
        color=colors["noisy_kf"],
        label=f"noisy + KF (σ={noise_std})",
    )
    plt.plot(
        acc_attack_kf,
        linewidth=1.8,
        color=colors["attack_kf"],
        label=(
            r"attack 4D + KF "
            f"(p={attack_prob}, σ={attack_std}, ε={attack_eps}, MC={mc_samples}, steps={pgd_steps})"
        ),
    )
    plt.plot(
        acc_random_attack_kf,
        linewidth=1.8,
        color=colors["random_kf"],
        label="random ellipse + KF",
    )

    plt.grid(True, alpha=0.25)
    plt.xlabel("Episode")
    plt.ylabel("Accumulated reward")
    plt.legend(loc="lower right")

    save_npz(
        data_path,
        acc_clean=np.asarray(acc_clean, dtype=float),
        acc_noisy_kf=np.asarray(acc_noisy_kf, dtype=float),
        acc_attack_kf=np.asarray(acc_attack_kf, dtype=float),
        acc_random_attack_kf=np.asarray(acc_random_attack_kf, dtype=float),
        noise_std=np.asarray(noise_std, dtype=float),
        attack_std=np.asarray(attack_std, dtype=float),
        attack_eps=np.asarray(attack_eps, dtype=float),
        attack_prob=np.asarray(attack_prob, dtype=float),
        kf_meas_std=np.asarray(kf_meas_std, dtype=float),
        kf_proc_std=np.asarray(kf_proc_std, dtype=float),
        pgd_steps=np.asarray(pgd_steps, dtype=int),
        pgd_step_size=np.asarray(pgd_step_size, dtype=float),
        mc_samples=np.asarray(mc_samples, dtype=int),
        n_episodes=np.asarray(n_episodes, dtype=int),
        seed0=np.asarray(seed0, dtype=int),
    )
    print(f"[cache] saved data: {data_path}")
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
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

    os.makedirs(FIGURES_DIR, exist_ok=True)

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
        seed=2025,
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

    print(f"[load] Policy loaded from: {MODEL_PATH}")
    print(f"[cfg] ATTACK_PROB={ATTACK_PROB}")
    print("[cfg] ATTACK_SPACE=4D observation (position + wind)")
    print(f"[cfg] DEVICE={DEVICE}")
    print(f"[cfg] NOISE_STD={NOISE_STD}")
    print(f"[cfg] ATTACK_STD={ATTACK_STD} | ATTACK_EPS={ATTACK_EPS}")
    print(f"[cfg] KF_MEAS_STD={KF_MEAS_STD} | KF_PROC_STD={KF_PROC_STD}")
    print(f"[cfg] PGD_STEPS={PGD_STEPS} | PGD_STEP_SIZE={PGD_STEP_SIZE} | MC_SAMPLES={MC_SAMPLES}")
    print("[cfg] RANDOM_BASELINE=same 4D ellipsoid boundary, uniform direction when attack is triggered")
    print(f"[out] FIGURES_DIR={FIGURES_DIR}")

    plot_accumulated_reward_clean_noisykf_attackkf_randomkf(
        AdvRLEnvConfig=AdvRLEnvConfig,
        EnvCleanClass=AdvRL2DEnv,
        EnvNoisyClass=EnvNoisy,
        base_cfg=base_cfg,
        model=model,
        noise_std=NOISE_STD,
        attack_std=ATTACK_STD,
        attack_eps=ATTACK_EPS,
        attack_prob=ATTACK_PROB,
        kf_meas_std=KF_MEAS_STD,
        kf_proc_std=KF_PROC_STD,
        pgd_steps=PGD_STEPS,
        pgd_step_size=PGD_STEP_SIZE,
        mc_samples=MC_SAMPLES,
        n_episodes=N_EPISODES,
        seed0=SEED0,
        device=DEVICE,
        results_dir=FIGURES_DIR,
    )

    print("[done] Evaluation finished.")


if __name__ == "__main__":
    main()
