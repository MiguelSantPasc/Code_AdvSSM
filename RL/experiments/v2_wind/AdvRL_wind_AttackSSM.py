#!/usr/bin/env python3
# Eval_clean_noisyKF_attackKF.py
#
# HARD-CODED PATHS:
#   - AdvRL_wind.py    : ./AdvRL_wind.py  (same folder as this script)
#   - Model checkpoint : ../../saved_models/AdvRL_v2_policy.pt
#   - Output figures   : ../../results/
#
# Curves plotted:
#   1) clean
#   2) noisy + KF
#   3) attack + KF
#
# Interpretation:
#   - clean:
#       policy receives the clean env observation directly
#   - noisy + KF:
#       position observation is corrupted by Gaussian noise;
#       policy acts on the KF filtered position estimate
#   - attack + KF:
#       at each step, an adversarial observation y_t' is found with PGD
#       by minimizing E[V(x_t) | y_t', history], and then the policy acts
#       on the KF filtered position estimate based on that attacked measurement
#
# State model used by the KF attacker/filter:
#   x_t in R^2 = latent position
#   x_{t+1} = x_t + a_t + wind_t + q_t
#   y_t     = x_t + r_t
#
# Policy observation built from any position p:
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
ATTACK_EPS = 5.991   # ~ chi-square 95% in 2D

# KF modelt
KF_MEAS_STD = NOISE_STD
KF_PROC_STD = 0.00003

# PGD / MC for attack
PGD_STEPS = 65  
PGD_STEP_SIZE = 0.25
MC_SAMPLES = 256

# Accumulated reward plot
N_EPISODES = 2000
SEED0 = 1_000

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
# so wind / env randomness stays paired across evaluations
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


# ============================================================
# Expected critic value under posterior
#   mu_V(y') = E_{x ~ p(x | y', history)} [V(x)]
# ============================================================

def expected_critic_value_mc(
    *,
    model,
    y_adv_torch: torch.Tensor,          # (2,), requires_grad=True
    m_pred: np.ndarray,                 # (2,)
    P_pred: np.ndarray,                 # (2,2)
    R: np.ndarray,                      # (2,2)
    wind_xy: np.ndarray,                # (2,)
    goal: np.ndarray,                   # (2,)
    goal_r_max: float,
    xi_torch: torch.Tensor,             # (MC,2), fixed common random numbers
    device: str = "cpu",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Compute:
        mu_V(y') = E_x[V(x_t)]  with x_t ~ posterior given y'
    using MC.

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

    x_samples = m_post_t.unsqueeze(0) + xi_torch @ L_t.T  # (MC,2)

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
    """
    Solve approximately with PGD:
        min_{y' in ellipsoid(y_nom, attack_sigma, attack_eps)}
            E_{x ~ p(x | y', history)} [V(x)]

    Returns:
      y_star      attacked observation
      obj_star    final estimated expectation
      m_post_star posterior mean at y_star
      P_post_star posterior covariance at y_star
    """
    dev = torch.device(device)

    # Common random numbers inside PGD
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

        obj = mu_V_t
        obj.backward()

        grad = y_t.grad.detach().cpu().numpy().astype(np.float32)
        y_next = y_curr_np - float(pgd_step_size) * grad

        y_next = project_to_attack_region(
            y_candidate=y_next,
            center=y_nom,
            Sigma=attack_sigma,
            epsilon=attack_eps,
        )

        obj_val = float(obj.detach().cpu().item())
        if best_obj is None or obj_val < best_obj:
            best_obj = obj_val
            best_y = y_curr_np.copy()
            best_m_post = m_post_np.copy()
            best_P_post = P_post_np.copy()

        y_curr_np = y_next

    # final evaluation at last point
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
        y_obs = obs_to_position(obs, goal, goal_r_max)
        wind_xy = np.asarray(obs[2:4], dtype=np.float32)

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
    Sigma_attack = (float(attack_std) ** 2) * np.eye(2, dtype=np.float32)

    # RNG for deciding whether to attack each step
    rng_attack_gate = np.random.default_rng(int(seed_for_attack) + 777777)
    rng_obs_noise = np.random.default_rng(int(seed_for_attack) + 888888)

    # ---------------------------------------------------------
    # Initial prior from first nominal observation (never attack)
    # ---------------------------------------------------------
    y0 = obs_to_position(obs, goal, goal_r_max)
    wind_xy = np.asarray(obs[2:4], dtype=np.float32)

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
        y_clean  = obs_to_position(obs, goal, goal_r_max)
        wind_xy = np.asarray(obs[2:4], dtype=np.float32)

        do_attack = bool(rng_attack_gate.random() < float(attack_prob))

        noise = rng_obs_noise.normal(0.0, float(kf_meas_std), size=(2,)).astype(np.float32)

        y_noisy = y_clean + noise


        if do_attack:
            _y_star, _obj_star, m_post, P_post = pgd_attack_on_expected_value(
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
        else:
            m_post, P_post, _K = kf_update_position(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=y_noisy,
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
        step_idx += 1

        if done:
            break

    return float(ep_return)
# ============================================================
# Plot accumulated reward
# ============================================================

def plot_accumulated_reward_clean_noisykf_attackkf(
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

    acc_clean = []
    acc_noisy_kf = []
    acc_attack_kf = []

    total_clean = 0.0
    total_noisy_kf = 0.0
    total_attack_kf = 0.0

    for k in range(n_episodes):
        seed = int(seed0 + k)

        cfg_clean = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_noisy = AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_attack = AdvRLEnvConfig(**{**asdict(base_cfg)})

        cfg_clean.seed = seed
        cfg_clean.obs_noise_std = 0.0

        cfg_noisy.seed = seed
        cfg_noisy.obs_noise_std = float(noise_std)

        cfg_attack.seed = seed
        cfg_attack.obs_noise_std = 0.0

        env_clean = EnvCleanClass(cfg_clean)
        env_noisy = EnvNoisyClass(cfg_noisy)
        env_attack = EnvCleanClass(cfg_attack)

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

        total_clean += float(ret_clean)
        total_noisy_kf += float(ret_noisy_kf)
        total_attack_kf += float(ret_attack_kf)

        acc_clean.append(total_clean)
        acc_noisy_kf.append(total_noisy_kf)
        acc_attack_kf.append(total_attack_kf)

        if (k + 1) % 25 == 0 or (k + 1) == n_episodes:
            print(
                f"[{k+1:4d}/{n_episodes}] "
                f"clean={total_clean:.1f} | "
                f"noisy+KF={total_noisy_kf:.1f} | "
                f"attack+KF={total_attack_kf:.1f}"
            )

    fig = plt.figure(figsize=(12, 5))
    plt.plot(acc_clean, linewidth=1.8, label="clean")
    plt.plot(
        acc_noisy_kf,
        linewidth=1.8,
        label=f"noisy + KF (σ={noise_std})",
    )
    plt.plot(
        acc_attack_kf,
        linewidth=1.8,
        label=(
            r"attack + KF "
            f"(p={attack_prob}, σ={attack_std}, ε={attack_eps}, MC={mc_samples}, steps={pgd_steps})"
        ),
    )

    plt.grid(True, alpha=0.25)
    plt.xlabel("Episode")
    plt.ylabel("Accumulated reward (cumulative sum)")
    plt.title("Accumulated reward: clean vs noisy+KF vs attack+KF")
    plt.legend()

    outpath = os.path.join(results_dir, "eval_accumulated_reward_clean_noisyKF_attackKF.png")
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
    print(f"[cfg] DEVICE={DEVICE}")
    print(f"[cfg] NOISE_STD={NOISE_STD}")
    print(f"[cfg] ATTACK_STD={ATTACK_STD} | ATTACK_EPS={ATTACK_EPS}")
    print(f"[cfg] KF_MEAS_STD={KF_MEAS_STD} | KF_PROC_STD={KF_PROC_STD}")
    print(f"[cfg] PGD_STEPS={PGD_STEPS} | PGD_STEP_SIZE={PGD_STEP_SIZE} | MC_SAMPLES={MC_SAMPLES}")
    print(f"[out] RESULTS_DIR={RESULTS_DIR}")

    plot_accumulated_reward_clean_noisykf_attackkf(
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
        results_dir=RESULTS_DIR,
    )

    print("[done] Evaluation finished.")


if __name__ == "__main__":
    main()