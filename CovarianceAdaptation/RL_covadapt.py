#!/usr/bin/env python3
"""
rl_wind_covadapt_reward.py

Accumulated-reward comparison for the 4D wind RL attack with covariance
adaptation.

This script reuses the exact environment, model checkpoint, attack geometry,
and random-ellipse baseline from
`RL/v2_wind_4dattack/AdvRL_wind_AttackSSM_randomEllipse.py`. The new part is
an online covariance-adaptation defense applied directly in the 4D policy
observation space whenever a 4D attacked observation is injected.

What is compared:
1. clean
2. noisy + KF
3. PGD attack + KF
4. random ellipse + KF
5. PGD attack + cov-adapt with `lambda_t = c * lambda_max(S_t)` for each chosen `c`
6. random ellipse + cov-adapt with `lambda_t = c * lambda_max(S_t)` for each chosen `c`

Important modeling choice:
- The attack modifies the full 4D policy observation.
- The covariance adaptation also acts on the full 4D policy observation
  `[delta_x, delta_y, wind_x, wind_y]`, so both the filtered policy input and
  the next-step predictor use defended position and defended wind components.
- The PGD attacker is kept consistent with the original 4D attack script: it
  still optimizes a full 4D adversarial observation while using the induced
  posterior over position inside its value objective.

Plotting conventions followed here:
- no figure title,
- short legends inside the axes,
- pastel colors by default,
- comments explain the nontrivial filtering and defense steps.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
RL_4D_DIR = os.path.join(REPO_ROOT, "RL", "v2_wind_4dattack")

for import_path in (REPO_ROOT, RL_4D_DIR):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

try:
    import AdvRL_wind as rl_mod
    import AdvRL_wind_AttackSSM_randomEllipse as attack_mod
    from AdvSSM.io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from CovarianceAdaptation.covariance_adaptation_utils import (
        compute_contamination_prior,
        gaussian_logpdf,
        log_mix_posterior_weight,
        rank_one_covariance_update,
        safe_unit_direction,
        set_plot_theme,
        solve_spd,
        spd_inverse,
        style_axis,
    )
except ModuleNotFoundError:
    import AdvRL_wind as rl_mod
    import AdvRL_wind_AttackSSM_randomEllipse as attack_mod
    from io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from covariance_adaptation_utils import (
        compute_contamination_prior,
        gaussian_logpdf,
        log_mix_posterior_weight,
        rank_one_covariance_update,
        safe_unit_direction,
        set_plot_theme,
        solve_spd,
        spd_inverse,
        style_axis,
    )


def lambda_max_from_covariance(cov: np.ndarray) -> float:
    """Return the largest eigenvalue of a symmetric covariance matrix."""
    eigvals = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    return float(max(np.max(eigvals), 1e-12))


def build_policy_obs_filter_covariances(
    *,
    goal_r_max: float,
    kf_meas_std: float,
    kf_proc_std: float,
    wind_process_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 4D observation-space covariances for `[delta_x, delta_y, wind_x, wind_y]`.

    The environment noise is specified in physical position units, whereas the
    policy sees normalized relative-position coordinates
    `delta = (goal - y_t) / goal_r_max`. We therefore scale the position
    variances by `goal_r_max` before combining them with the wind variances.
    """
    if goal_r_max <= 0.0:
        raise ValueError("goal_r_max must be positive.")

    delta_meas_std = float(kf_meas_std) / float(goal_r_max)
    delta_proc_std = float(kf_proc_std) / float(goal_r_max)

    R_policy = np.diag(
        [
            delta_meas_std**2,
            delta_meas_std**2,
            0.0,
            0.0,
        ]
    ).astype(np.float32)
    Q_policy = np.diag(
        [
            delta_proc_std**2,
            delta_proc_std**2,
            float(wind_process_std) ** 2,
            float(wind_process_std) ** 2,
        ]
    ).astype(np.float32)
    R_position = (float(kf_meas_std) ** 2) * np.eye(2, dtype=np.float32)
    return (
        attack_mod.project_to_psd(R_policy),
        attack_mod.project_to_psd(Q_policy),
        attack_mod.project_to_psd(R_position),
    )


def policy_obs_state_to_position_belief(
    *,
    m_policy: np.ndarray,
    P_policy: np.ndarray,
    goal: np.ndarray,
    goal_r_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4D policy-observation belief into the 2D position belief used by PGD.

    The attack optimizer from the original wind script expects a Gaussian prior
    on the measured position. Since the defended filter now lives in policy
    observation coordinates, we map its delta block back into position space.
    """
    m_policy = np.asarray(m_policy, dtype=float).reshape(4)
    P_policy = attack_mod.project_to_psd(np.asarray(P_policy, dtype=float))
    goal = np.asarray(goal, dtype=float).reshape(2)

    delta_mean = m_policy[:2]
    pos_mean = goal - float(goal_r_max) * delta_mean
    pos_cov = attack_mod.project_to_psd((float(goal_r_max) ** 2) * P_policy[:2, :2])
    return pos_mean.astype(np.float32), pos_cov.astype(np.float32)


def kf_update_policy_observation(
    *,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one nominal KF update in 4D policy-observation coordinates.

    The latent defended state is the policy observation itself:
        z_t = [delta_x, delta_y, wind_x, wind_y].
    The measurement model is the identity because the policy observation is the
    directly attacked quantity.
    """
    m_pred = np.asarray(m_pred, dtype=float).reshape(4)
    P_pred = attack_mod.project_to_psd(np.asarray(P_pred, dtype=float))
    y_obs = np.asarray(y_obs, dtype=float).reshape(4)
    R = attack_mod.project_to_psd(np.asarray(R, dtype=float))

    I4 = np.eye(4, dtype=float)
    S = attack_mod.project_to_psd(P_pred + R)
    K = solve_spd(S, P_pred.T).T
    innov = y_obs - m_pred
    m_post = m_pred + K @ innov

    # Joseph form keeps the observation-space covariance PSD after updates.
    joseph_left = I4 - K
    P_post = attack_mod.project_to_psd(
        joseph_left @ P_pred @ joseph_left.T + K @ R @ K.T
    )
    return m_post.astype(np.float32), P_post.astype(np.float32), K.astype(np.float32)


def predict_policy_observation(
    *,
    m_post: np.ndarray,
    P_post: np.ndarray,
    action: np.ndarray,
    goal_r_max: float,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the next 4D policy observation from the filtered current one.

    With `delta_t = (goal - position_t) / goal_r_max` and
    `position_{t+1} = position_t + action_t + wind_t + q_t`, the normalized
    relative-position coordinates evolve linearly as

        delta_{t+1} = delta_t - (action_t + wind_t) / goal_r_max.

    We pair that with a random-walk model for the observed wind components.
    """
    if goal_r_max <= 0.0:
        raise ValueError("goal_r_max must be positive.")

    m_post = np.asarray(m_post, dtype=float).reshape(4)
    P_post = attack_mod.project_to_psd(np.asarray(P_post, dtype=float))
    action = np.asarray(action, dtype=float).reshape(2)
    Q = attack_mod.project_to_psd(np.asarray(Q, dtype=float))

    inv_goal = 1.0 / float(goal_r_max)
    A = np.array(
        [
            [1.0, 0.0, -inv_goal, 0.0],
            [0.0, 1.0, 0.0, -inv_goal],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [-inv_goal, 0.0],
            [0.0, -inv_goal],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    m_next = A @ m_post + B @ action
    P_next = attack_mod.project_to_psd(A @ P_post @ A.T + Q)
    return m_next.astype(np.float32), P_next.astype(np.float32)


def covariance_adapted_kf_update_policy_observation(
    *,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
    adv_target: np.ndarray | None,
    c_scale: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    direction_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | np.ndarray]]:
    """
    Run one 4D policy-observation KF update with online covariance adaptation.

    The hidden defended quantity is the 4D policy observation itself:
        z_t = [delta_x, delta_y, wind_x, wind_y].
    The measurement model is therefore
        z_t^{obs} = z_t + r_t,
    so `H = I` and the predictive observation law is:
        z_t^{obs} | history ~ N(m_pred, P_pred + R).

    The defense uses the attacked 4D target when available. The current
    `lambda_t` is scaled online as:
        lambda_t = c_scale * lambda_max(S_t),
    where `S_t = P_pred + R` is the predictive observation covariance.
    """
    m_pred = np.asarray(m_pred, dtype=float).reshape(4)
    P_pred = attack_mod.project_to_psd(np.asarray(P_pred, dtype=float))
    y_obs = np.asarray(y_obs, dtype=float).reshape(4)
    R = attack_mod.project_to_psd(np.asarray(R, dtype=float))

    I4 = np.eye(4, dtype=float)
    y_hat = m_pred.copy()
    S_nom = attack_mod.project_to_psd(P_pred + R)
    innov = y_obs - y_hat

    V_tilde = R.copy()
    S_tilde = S_nom.copy()
    K_tilde = solve_spd(S_tilde, P_pred.T).T
    pi_t = 0.0
    gamma_t = 0.0
    bar_gamma_t = 0.0
    lambda_t = 0.0
    u_dir = np.zeros(4, dtype=float)

    if adv_target is not None:
        adv_target = np.asarray(adv_target, dtype=float).reshape(4)
        delta_adv = adv_target - y_hat
        u_dir, delta_norm = safe_unit_direction(delta_adv, eps=direction_eps)

        if delta_norm >= direction_eps:
            lambda_t = float(c_scale) * lambda_max_from_covariance(S_nom)
            pi_t, _, _ = compute_contamination_prior(
                delta_adv=delta_adv,
                S_t=S_nom,
                P_pred_t=P_pred,
                H_t=I4,
                omega_h=omega_h,
                omega_o=omega_o,
            )

            # The PoE branch models a contaminated observation law centered
            # between the nominal predictor and the adversarial target.
            S_adv = rank_one_covariance_update(S_nom, lambda_t, u_dir)
            precision_poe = spd_inverse(S_adv) + spd_inverse(R)
            Sigma_poe = spd_inverse(precision_poe)
            rhs_poe = solve_spd(S_adv, y_hat) + solve_spd(R, adv_target)
            mu_poe = solve_spd(precision_poe, rhs_poe)

            log_p0 = gaussian_logpdf(y_obs, y_hat, S_nom)
            log_p1 = gaussian_logpdf(y_obs, mu_poe, Sigma_poe)
            gamma_t = log_mix_posterior_weight(pi_t, log_p0, log_p1)
            bar_gamma_t = gamma_t if gamma_t >= delta_threshold else 0.0

            V_tilde = rank_one_covariance_update(R, lambda_t, u_dir, weight=bar_gamma_t)
            S_tilde = rank_one_covariance_update(S_nom, lambda_t, u_dir, weight=bar_gamma_t)
            K_tilde = solve_spd(S_tilde, P_pred.T).T

    m_post = m_pred + K_tilde @ innov

    # Joseph form keeps the adapted covariance numerically well behaved.
    joseph_left = I4 - K_tilde
    P_post = attack_mod.project_to_psd(
        joseph_left @ P_pred @ joseph_left.T + K_tilde @ V_tilde @ K_tilde.T
    )

    diagnostics = {
        "pi_t": float(pi_t),
        "gamma_t": float(gamma_t),
        "bar_gamma_t": float(bar_gamma_t),
        "lambda_t": float(lambda_t),
        "u_t": u_dir.astype(np.float32),
    }
    return m_post.astype(np.float32), P_post.astype(np.float32), diagnostics


def rollout_episode_return_attack_kf_covadapt(
    env,
    model,
    *,
    attack_std: float,
    attack_eps: float,
    attack_prob: float,
    kf_meas_std: float,
    kf_proc_std: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    seed_for_attack: int,
    c_scale: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    device: str = "cpu",
) -> float:
    """
    Roll out one episode under PGD attacks with covariance adaptation.

    The first observation is kept nominal exactly as in the original attack
    script so the comparison remains paired step by step. The defended filter
    now tracks the full 4D policy observation `[delta_x, delta_y, wind_x,
    wind_y]` instead of filtering only position.
    """
    dev = torch.device(device)
    obs = env.reset()

    goal = env.goal.copy()
    goal_r_max = float(env.cfg.goal_r_max)
    wind_process_std = float(env.cfg.wind_epsilon) * float(env.cfg.wind_volatility)
    R_policy, Q_policy, R_position = build_policy_obs_filter_covariances(
        goal_r_max=goal_r_max,
        kf_meas_std=kf_meas_std,
        kf_proc_std=kf_proc_std,
        wind_process_std=wind_process_std,
    )
    Sigma_attack = (float(attack_std) ** 2) * np.eye(4, dtype=np.float32)

    rng_attack_gate = np.random.default_rng(int(seed_for_attack) + 777777)
    rng_obs_noise = np.random.default_rng(int(seed_for_attack) + 888888)

    obs_state = np.asarray(obs, dtype=np.float32)
    m_post, P_post, _ = kf_update_policy_observation(
        m_pred=obs_state.copy(),
        P_pred=R_policy.copy(),
        y_obs=obs_state,
        R=R_policy,
    )

    ep_return = 0.0
    step_idx = 0

    obs_filt = m_post.copy()

    with torch.no_grad():
        obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

    m_pred, P_pred = predict_policy_observation(
        m_post=m_post,
        P_post=P_post,
        action=action,
        goal_r_max=goal_r_max,
        Q=Q_policy,
    )

    obs, reward, done, _info = env.step(action)
    ep_return += float(reward)
    if done:
        return float(ep_return)

    step_idx = 1

    for _ in range(1, env.cfg.max_steps):
        obs_clean = np.asarray(obs, dtype=np.float32)
        do_attack = bool(rng_attack_gate.random() < float(attack_prob))
        noise = rng_obs_noise.normal(0.0, float(kf_meas_std), size=(2,)).astype(np.float32)
        obs_noisy = obs_clean.copy()
        obs_noisy[:2] = obs_clean[:2] - noise / float(goal_r_max)

        if do_attack:
            m_pred_pos, P_pred_pos = policy_obs_state_to_position_belief(
                m_policy=m_pred,
                P_policy=P_pred,
                goal=goal,
                goal_r_max=goal_r_max,
            )
            obs_star, _obj_star, _m_post_attack, _P_post_attack = attack_mod.pgd_attack_on_expected_value(
                model=model,
                obs_nom=obs_clean,
                m_pred=m_pred_pos,
                P_pred=P_pred_pos,
                R=R_position,
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
            obs_attack = np.asarray(obs_star, dtype=np.float32)
            m_post, P_post, _diag = covariance_adapted_kf_update_policy_observation(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=obs_attack,
                R=R_policy,
                adv_target=obs_attack,
                c_scale=c_scale,
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
            )
        else:
            m_post, P_post, _ = kf_update_policy_observation(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=obs_noisy,
                R=R_policy,
            )

        obs_filt = m_post.copy()

        with torch.no_grad():
            obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
            action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        m_pred, P_pred = predict_policy_observation(
            m_post=m_post,
            P_post=P_post,
            action=action,
            goal_r_max=goal_r_max,
            Q=Q_policy,
        )

        obs, reward, done, _info = env.step(action)
        ep_return += float(reward)
        step_idx += 1

        if done:
            break

    return float(ep_return)


def rollout_episode_return_random_attack_kf_covadapt(
    env,
    model,
    *,
    attack_std: float,
    attack_eps: float,
    attack_prob: float,
    kf_meas_std: float,
    kf_proc_std: float,
    seed_for_attack: int,
    c_scale: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    device: str = "cpu",
) -> float:
    """
    Roll out one episode under random-ellipse attacks with covariance adaptation.

    The defended filter operates on the full 4D policy observation so the
    random attacked wind coordinates are defended together with the attacked
    relative-position coordinates.
    """
    dev = torch.device(device)
    obs = env.reset()

    goal = env.goal.copy()
    goal_r_max = float(env.cfg.goal_r_max)
    wind_process_std = float(env.cfg.wind_epsilon) * float(env.cfg.wind_volatility)
    R_policy, Q_policy, _R_position = build_policy_obs_filter_covariances(
        goal_r_max=goal_r_max,
        kf_meas_std=kf_meas_std,
        kf_proc_std=kf_proc_std,
        wind_process_std=wind_process_std,
    )
    Sigma_attack = (float(attack_std) ** 2) * np.eye(4, dtype=np.float32)

    rng_attack_gate = np.random.default_rng(int(seed_for_attack) + 777777)
    rng_obs_noise = np.random.default_rng(int(seed_for_attack) + 888888)
    rng_attack_sample = np.random.default_rng(int(seed_for_attack) + 999999)

    obs_state = np.asarray(obs, dtype=np.float32)
    m_post, P_post, _ = kf_update_policy_observation(
        m_pred=obs_state.copy(),
        P_pred=R_policy.copy(),
        y_obs=obs_state,
        R=R_policy,
    )

    ep_return = 0.0

    obs_filt = m_post.copy()

    with torch.no_grad():
        obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
        action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

    m_pred, P_pred = predict_policy_observation(
        m_post=m_post,
        P_post=P_post,
        action=action,
        goal_r_max=goal_r_max,
        Q=Q_policy,
    )

    obs, reward, done, _info = env.step(action)
    ep_return += float(reward)
    if done:
        return float(ep_return)

    for _ in range(1, env.cfg.max_steps):
        obs_clean = np.asarray(obs, dtype=np.float32)
        do_attack = bool(rng_attack_gate.random() < float(attack_prob))
        noise = rng_obs_noise.normal(0.0, float(kf_meas_std), size=(2,)).astype(np.float32)
        obs_noisy = obs_clean.copy()
        obs_noisy[:2] = obs_clean[:2] - noise / float(goal_r_max)

        if do_attack:
            obs_random = attack_mod.sample_uniform_from_attack_region(
                center=obs_clean,
                Sigma=Sigma_attack,
                epsilon=attack_eps,
                rng=rng_attack_sample,
            )
            obs_attack = np.asarray(obs_random, dtype=np.float32)
            m_post, P_post, _diag = covariance_adapted_kf_update_policy_observation(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=obs_attack,
                R=R_policy,
                adv_target=obs_attack,
                c_scale=c_scale,
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
            )
        else:
            m_post, P_post, _ = kf_update_policy_observation(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=obs_noisy,
                R=R_policy,
            )

        obs_filt = m_post.copy()

        with torch.no_grad():
            obs_t = torch.tensor(obs_filt, dtype=torch.float32, device=dev)
            action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)

        m_pred, P_pred = predict_policy_observation(
            m_post=m_post,
            P_post=P_post,
            action=action,
            goal_r_max=goal_r_max,
            Q=Q_policy,
        )

        obs, reward, done, _info = env.step(action)
        ep_return += float(reward)

        if done:
            break

    return float(ep_return)


def load_policy(device: torch.device, model_path: str) -> torch.nn.Module:
    """Load the same trained policy used by the 4D wind attack scripts."""
    try:
        model = rl_mod.ActorCritic(obs_dim=4, hidden=128, act_dim=2, std_fixed=0.35).to(device)
    except TypeError:
        model = rl_mod.ActorCritic(obs_dim=4, hidden=128, act_dim=2).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_base_cfg() -> rl_mod.AdvRLEnvConfig:
    """Return the exact environment configuration used in the random-ellipse script."""
    return rl_mod.AdvRLEnvConfig(
        obs_noise_std=0.0,
        proc_noise_std=0.0,
        wind_epsilon=0.9,
        wind_volatility=0.25,
        seed=2025,
        step_penalty=-1.0,
        success_reward=25.0,
        timeout_penalty=-25.0,
    )


def compute_accumulated_reward_data(
    *,
    n_episodes: int,
    seed0: int,
    model_path: str,
    noise_std: float,
    attack_std: float,
    attack_eps: float,
    attack_prob: float,
    kf_meas_std: float,
    kf_proc_std: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    c_scales: tuple[float, ...],
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    device: str,
) -> dict[str, np.ndarray | float | int]:
    """
    Compute accumulated rewards for the original RL baselines plus cov-adapt.

    Every curve uses the same episode seeds as the original script so the
    comparison stays tightly paired across clean, noisy, attacked, and defended
    rollouts.
    """
    if not np.isclose(omega_h + omega_o, 1.0, atol=1e-9):
        raise ValueError("omega_h and omega_o must sum to 1.")
    if not (0.0 <= delta_threshold <= 1.0):
        raise ValueError("delta_threshold must lie in [0, 1].")

    device_t = torch.device(device)
    model = load_policy(device_t, model_path)
    env_noisy_class = attack_mod.make_env_with_separate_obs_rng(rl_mod.AdvRL2DEnv)
    base_cfg = build_base_cfg()

    totals: dict[str, float] = {
        "clean": 0.0,
        "noisy_kf": 0.0,
        "attack_kf": 0.0,
        "random_kf": 0.0,
    }
    for c_scale in c_scales:
        totals[f"attack_cov_{c_scale:g}"] = 0.0
        totals[f"random_cov_{c_scale:g}"] = 0.0

    series: dict[str, list[float]] = {key: [] for key in totals}

    for episode_idx in range(n_episodes):
        seed = int(seed0 + episode_idx)

        cfg_clean = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_noisy = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_attack = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_random = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})

        cfg_clean.seed = seed
        cfg_clean.obs_noise_std = 0.0
        cfg_noisy.seed = seed
        cfg_noisy.obs_noise_std = float(noise_std)
        cfg_attack.seed = seed
        cfg_attack.obs_noise_std = 0.0
        cfg_random.seed = seed
        cfg_random.obs_noise_std = 0.0

        env_clean = rl_mod.AdvRL2DEnv(cfg_clean)
        env_noisy = env_noisy_class(cfg_noisy)
        env_attack = rl_mod.AdvRL2DEnv(cfg_attack)
        env_random = rl_mod.AdvRL2DEnv(cfg_random)

        ret_clean = attack_mod.rollout_episode_return_clean(env_clean, model, device=device)
        ret_noisy = attack_mod.rollout_episode_return_noisy_kf(
            env_noisy,
            model,
            kf_meas_std=kf_meas_std,
            kf_proc_std=kf_proc_std,
            device=device,
        )
        ret_attack = attack_mod.rollout_episode_return_attack_kf(
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
        ret_random = attack_mod.rollout_episode_return_random_attack_kf(
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

        totals["clean"] += float(ret_clean)
        totals["noisy_kf"] += float(ret_noisy)
        totals["attack_kf"] += float(ret_attack)
        totals["random_kf"] += float(ret_random)

        series["clean"].append(totals["clean"])
        series["noisy_kf"].append(totals["noisy_kf"])
        series["attack_kf"].append(totals["attack_kf"])
        series["random_kf"].append(totals["random_kf"])

        for c_scale in c_scales:
            cfg_attack_cov = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})
            cfg_random_cov = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})
            cfg_attack_cov.seed = seed
            cfg_attack_cov.obs_noise_std = 0.0
            cfg_random_cov.seed = seed
            cfg_random_cov.obs_noise_std = 0.0

            env_attack_cov = rl_mod.AdvRL2DEnv(cfg_attack_cov)
            env_random_cov = rl_mod.AdvRL2DEnv(cfg_random_cov)

            ret_attack_cov = rollout_episode_return_attack_kf_covadapt(
                env_attack_cov,
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
                c_scale=float(c_scale),
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
                device=device,
            )
            ret_random_cov = rollout_episode_return_random_attack_kf_covadapt(
                env_random_cov,
                model,
                attack_std=attack_std,
                attack_eps=attack_eps,
                attack_prob=attack_prob,
                kf_meas_std=kf_meas_std,
                kf_proc_std=kf_proc_std,
                seed_for_attack=seed,
                c_scale=float(c_scale),
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
                device=device,
            )

            attack_key = f"attack_cov_{c_scale:g}"
            random_key = f"random_cov_{c_scale:g}"
            totals[attack_key] += float(ret_attack_cov)
            totals[random_key] += float(ret_random_cov)
            series[attack_key].append(totals[attack_key])
            series[random_key].append(totals[random_key])

        print(
            f"[{episode_idx + 1:4d}/{n_episodes}] "
            f"clean={totals['clean']:.1f} | "
            f"noisy+KF={totals['noisy_kf']:.1f} | "
            f"PGD+KF={totals['attack_kf']:.1f} | "
            f"random+KF={totals['random_kf']:.1f}"
        )

    data: dict[str, np.ndarray | float | int] = {
        "n_episodes": int(n_episodes),
        "seed0": int(seed0),
        "noise_std": float(noise_std),
        "attack_std": float(attack_std),
        "attack_eps": float(attack_eps),
        "attack_prob": float(attack_prob),
        "kf_meas_std": float(kf_meas_std),
        "kf_proc_std": float(kf_proc_std),
        "pgd_steps": int(pgd_steps),
        "pgd_step_size": float(pgd_step_size),
        "mc_samples": int(mc_samples),
        "omega_h": float(omega_h),
        "omega_o": float(omega_o),
        "delta_threshold": float(delta_threshold),
        "c_scales": np.asarray(c_scales, dtype=float),
        "acc_clean": np.asarray(series["clean"], dtype=float),
        "acc_noisy_kf": np.asarray(series["noisy_kf"], dtype=float),
        "acc_attack_kf": np.asarray(series["attack_kf"], dtype=float),
        "acc_random_kf": np.asarray(series["random_kf"], dtype=float),
    }
    for c_scale in c_scales:
        data[f"acc_attack_cov_{c_scale:g}"] = np.asarray(series[f"attack_cov_{c_scale:g}"], dtype=float)
        data[f"acc_random_cov_{c_scale:g}"] = np.asarray(series[f"random_cov_{c_scale:g}"], dtype=float)
    return data


def plot_accumulated_reward_comparison(
    *,
    data: dict[str, np.ndarray | float | int],
    outpath: str,
) -> None:
    """
    Plot the final normalized accumulated rewards as three coordinated bar panels.

    The layout is:
    1. baseline references: clean, noisy + KF, PGD attack + KF, random ellipse + KF,
    2. PGD-focused comparison: clean, noisy + KF, PGD attack + KF, and PGD cov-adapt bars,
    3. random-focused comparison: clean, noisy + KF, random ellipse + KF, and random cov-adapt bars.

    Each bar shows the final accumulated reward divided by the total number of
    episodes so the comparison is standardized by the episode budget and can be
    read as mean reward per episode.
    """
    set_plot_theme()

    c_scales = tuple(float(value) for value in np.asarray(data["c_scales"], dtype=float))
    n_episodes = int(data["n_episodes"])
    acc_clean = np.asarray(data["acc_clean"], dtype=float)
    acc_noisy_kf = np.asarray(data["acc_noisy_kf"], dtype=float)
    acc_attack_kf = np.asarray(data["acc_attack_kf"], dtype=float)
    acc_random_kf = np.asarray(data["acc_random_kf"], dtype=float)

    colors = {
        "clean": "#6FAF8F",
        "noisy_kf": "#A7D37A",
        "attack_kf": "#E9B188",
        "random_kf": "#8FAEDF",
    }
    # Warm palette for PGD-based curves and cool palette for random-ellipse curves.
    attack_cov_colors = ["#F3D0B7", "#E7B28A", "#D89269"]
    random_cov_colors = ["#CAD9F2", "#ADC4E9", "#8EACDB"]

    def normalized_final_reward(series: np.ndarray) -> float:
        """Return the final accumulated reward standardized by episode count."""
        series = np.asarray(series, dtype=float)
        if series.size == 0:
            return 0.0
        return float(series[-1] / float(n_episodes))

    baseline_labels = [
        "Noise-Free",
        "Noise + KF",
        "Attacked + KF",
        r"$\epsilon$-perturbation + KF",
    ]
    baseline_values = [
        normalized_final_reward(acc_clean),
        normalized_final_reward(acc_noisy_kf),
        normalized_final_reward(acc_attack_kf),
        normalized_final_reward(acc_random_kf),
    ]
    baseline_colors = [
        colors["clean"],
        colors["noisy_kf"],
        colors["attack_kf"],
        colors["random_kf"],
    ]

    attack_labels = [
        "Noise-Free",
        "Noise + KF",
        "Attacked + KF",
    ]
    attack_labels.extend(
        [
            rf"Attacked + cov-adapt" + "\n" + rf"($\lambda={c_scale:g}\lambda_{{\max}}$)"
            for c_scale in c_scales
        ]
    )
    attack_values = [
        normalized_final_reward(acc_clean),
        normalized_final_reward(acc_noisy_kf),
        normalized_final_reward(acc_attack_kf),
    ]
    attack_values.extend(
        normalized_final_reward(np.asarray(data[f"acc_attack_cov_{c_scale:g}"], dtype=float))
        for c_scale in c_scales
    )
    attack_colors = [
        colors["clean"],
        colors["noisy_kf"],
        colors["attack_kf"],
        *[attack_cov_colors[idx % len(attack_cov_colors)] for idx, _ in enumerate(c_scales)],
    ]

    random_labels = [
        "Noise-Free",
        "Noise + KF",
        r"$\epsilon$-perturbation + KF",
    ]
    random_labels.extend(
        [
            rf"$\epsilon$-perturbation + cov-adapt"
            + "\n"
            + rf"($\lambda={c_scale:g}\lambda_{{\max}}$)"
            for c_scale in c_scales
        ]
    )
    random_values = [
        normalized_final_reward(acc_clean),
        normalized_final_reward(acc_noisy_kf),
        normalized_final_reward(acc_random_kf),
    ]
    random_values.extend(
        normalized_final_reward(np.asarray(data[f"acc_random_cov_{c_scale:g}"], dtype=float))
        for c_scale in c_scales
    )
    random_colors = [
        colors["clean"],
        colors["noisy_kf"],
        colors["random_kf"],
        *[random_cov_colors[idx % len(random_cov_colors)] for idx, _ in enumerate(c_scales)],
    ]

    def darken_hex(hex_color: str, factor: float = 0.88) -> tuple[float, float, float]:
        """Return a darker RGB edge color derived from a hex face color."""
        raw = hex_color.lstrip("#")
        rgb = tuple(int(raw[idx:idx + 2], 16) / 255.0 for idx in (0, 2, 4))
        return tuple(max(0.0, min(1.0, factor * channel)) for channel in rgb)

    all_values = np.asarray(baseline_values + attack_values + random_values, dtype=float)
    value_abs_max = float(np.max(np.abs(all_values))) if all_values.size > 0 else 1.0
    value_abs_max = max(value_abs_max, 1.0)
    y_pad = 0.18 * value_abs_max
    y_min = float(np.min(all_values)) - y_pad
    y_max = float(np.max(all_values)) + y_pad
    if y_min > -0.28 * value_abs_max:
        y_min = -0.28 * value_abs_max

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.8, 5.9),
        sharey=True,
        constrained_layout=True,
    )
    ax_baseline, ax_attack, ax_random = axes

    for ax in axes:
        style_axis(ax)
        ax.axhline(0.0, color="#4E4E4E", linewidth=1.15, alpha=0.92, zorder=1)
        ax.set_ylim(y_min, y_max)
        ax.set_axisbelow(True)

    def draw_bar_panel(
        *,
        ax: plt.Axes,
        labels: list[str],
        values: list[float],
        bar_colors: list[str],
        legend_width: str = "normal",
    ) -> None:
        """
        Draw one normalized final-reward bar panel with a zero reference line.

        The value labels are shown directly on the bars because the whole point
        of this figure is the final standardized performance rather than the
        transient path.
        """
        x = np.arange(len(labels), dtype=float)
        bars = ax.bar(
            x,
            np.asarray(values, dtype=float),
            width=0.72,
            color=bar_colors,
            edgecolor=[darken_hex(color) for color in bar_colors],
            linewidth=1.0,
            alpha=0.97,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

        for bar, value in zip(bars, values):
            offset = 0.028 * (y_max - y_min)
            text_y = value + offset if value >= 0.0 else value - offset
            ax.text(
                float(bar.get_x() + 0.5 * bar.get_width()),
                float(text_y),
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0.0 else "top",
                fontsize=9.2,
                color="#2F2F2F",
            )

        legend_handles = [
            Patch(
                facecolor=color,
                edgecolor=darken_hex(color),
                linewidth=1.0,
            )
            for color in bar_colors
        ]
        if legend_width == "wide":
            borderpad = 1.22
            handlelength = 5.0
            handletextpad = 1.55
            labelspacing = 0.82
        elif legend_width == "medium":
            borderpad = 0.61
            handlelength = 2.5
            handletextpad = 0.84
            labelspacing = 0.53
        else:
            borderpad = 0.52
            handlelength = 2.2
            handletextpad = 0.75
            labelspacing = 0.50

        ax.legend(
            legend_handles,
            labels,
            loc="lower left",
            frameon=True,
            framealpha=0.97,
            borderpad=borderpad,
            fontsize=9.9,
            handlelength=handlelength,
            handletextpad=handletextpad,
            labelspacing=labelspacing,
        )

    draw_bar_panel(
        ax=ax_baseline,
        labels=baseline_labels,
        values=baseline_values,
        bar_colors=baseline_colors,
    )
    draw_bar_panel(
        ax=ax_attack,
        labels=attack_labels,
        values=attack_values,
        bar_colors=attack_colors,
        legend_width="medium",
    )
    draw_bar_panel(
        ax=ax_random,
        labels=random_labels,
        values=random_values,
        bar_colors=random_colors,
        legend_width="medium",
    )

    ax_baseline.set_ylabel("Mean accumulated reward")

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    """
    Generate the RL accumulated-reward comparison with covariance adaptation.

    Change the constants below directly when you want a lighter validation run
    or a fuller experiment. This keeps the script easy to tweak without relying
    on environment variables.
    """
    model_path = os.path.abspath(os.path.join(RL_4D_DIR, "outputs", "saved_models", "AdvRL_v2_policy.pt"))
    device = "cpu"
    noise_std = 0.5
    attack_prob = 0.15
    attack_std = noise_std
    attack_eps = 0.75
    kf_meas_std = noise_std
    kf_proc_std = 0.03
    pgd_steps = 65
    pgd_step_size = 0.25
    mc_samples = 256
    n_episodes = 500
    seed0 = 1_000

    c_scales = (0.5, 1.0)
    omega_h = 0.50
    omega_o = 0.50
    delta_threshold = 0.20
    force_cache = False

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at:\n  {model_path}")

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    c_tag = "-".join(f"{value:g}" for value in c_scales).replace(".", "p")
    outpath = os.path.join(
        out_dir,
        (
            "comparison_RL_v2_wind_4dattack_"
            f"N{n_episodes}_p{attack_prob}_eps{str(attack_eps).replace('.', 'p')}_c{c_tag}.png"
        ),
    )
    data_path = data_path_for_plot(outpath)

    def compute_data() -> dict[str, np.ndarray | float | int]:
        return compute_accumulated_reward_data(
            n_episodes=n_episodes,
            seed0=seed0,
            model_path=model_path,
            noise_std=noise_std,
            attack_std=attack_std,
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            kf_meas_std=kf_meas_std,
            kf_proc_std=kf_proc_std,
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            mc_samples=mc_samples,
            c_scales=c_scales,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            device=device,
        )

    data = cached_npz(data_path, compute_data, force=force_cache)
    plot_accumulated_reward_comparison(data=data, outpath=outpath)
    print(f"Saved figure to: {outpath}")


if __name__ == "__main__":
    main()
