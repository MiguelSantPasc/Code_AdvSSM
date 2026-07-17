#!/usr/bin/env python3
"""
RL_covadapt_compare_epsilons_wolf.py

Compare two attack-radius settings for the 4D RL covariance-adaptation
experiment while also including the WoLF robust-filter baselines.

What this script does:
1. Reuse the exact environment, policy checkpoint, attack geometry, and KF
   baseline already used by `CovarianceAdaptation/RL_covadapt.py`.
2. Compare two attack ellipsoid radii, by default `epsilon = 0.75` and
   `epsilon = 0.95`, in a single figure.
3. Keep the same three-panel structure as
   `CovarianceAdaptation/RL_covadapt_compare_epsilons.py`:
   - baseline methods,
   - attacked-case methods,
   - boundary epsilon-perturbation methods.
4. Extend the attack and boundary epsilon-perturbation panels with two additional
   robust filters inspired by Duran-Martin et al.:
   - WoLF-IMQ,
   - WoLF-TMD.

Important modeling details:
1. The attacked quantity is still the full 4D policy observation
   `[delta_x, delta_y, wind_x, wind_y]`.
2. The WoLF update is applied in the same 4D observation-space filter used by
   the current covariance-adaptation defense, so all methods see the same
   attacked coordinates.
3. The wind coordinates in this RL setup are observed with zero measurement
   noise, which makes the nominal measurement covariance singular. Because of
   that, the TMD gate is evaluated with the predictive observation covariance
   `S_t = P_pred + R` instead of `R` alone, so the Mahalanobis distance is
   well defined in the full 4D space.

Design choices:
1. Method identity is encoded by color and remains stable across panels.
2. Epsilon values are encoded by hatch, just as in the current two-epsilon
   comparison script.
3. The horizontal zero line is kept because negative mean reward is meaningful
   in this environment.
4. The configuration stays in plain Python variables inside `main()` so the
   script is easy to tweak without relying on environment variables.
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

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from AdvSSM.io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from CovarianceAdaptation.RL_covadapt import (
        RL_4D_DIR,
        attack_mod,
        build_base_cfg,
        build_policy_obs_filter_covariances,
        kf_update_policy_observation,
        load_policy,
        policy_obs_state_to_position_belief,
        predict_policy_observation,
        rl_mod,
        rollout_episode_return_attack_kf_covadapt,
        rollout_episode_return_random_attack_kf_covadapt,
        set_plot_theme,
        style_axis,
    )
    from CovarianceAdaptation.covariance_adaptation_utils import solve_spd
except ModuleNotFoundError:
    from io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from RL_covadapt import (
        RL_4D_DIR,
        attack_mod,
        build_base_cfg,
        build_policy_obs_filter_covariances,
        kf_update_policy_observation,
        load_policy,
        policy_obs_state_to_position_belief,
        predict_policy_observation,
        rl_mod,
        rollout_episode_return_attack_kf_covadapt,
        rollout_episode_return_random_attack_kf_covadapt,
        set_plot_theme,
        style_axis,
    )
    from covariance_adaptation_utils import solve_spd


def normalized_final_reward(series: np.ndarray, *, n_episodes: int) -> float:
    """Return the final accumulated reward standardized by episode count."""
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return 0.0
    return float(series[-1] / float(n_episodes))


def darken_hex(hex_color: str, factor: float = 0.88) -> tuple[float, float, float]:
    """Return a slightly darker RGB color for edges and legend patches."""
    raw = hex_color.lstrip("#")
    rgb = tuple(int(raw[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))
    return tuple(max(0.0, min(1.0, factor * channel)) for channel in rgb)


def wolf_imq_weight_squared(
    innovation: np.ndarray,
    *,
    soft_threshold: float,
    min_weight: float = 1e-6,
) -> float:
    """
    Return the WoLF-IMQ observation weight `w_t^2`.

    The WoLF website writes the IMQ weight as
        W = (1 + ||e||^2 / c^2)^(-1/2).
    The Gaussian update depends on `W^2`, so we work directly with
        w_t^2 = c^2 / (c^2 + ||e||^2),
    matching the authors' public reference code.
    """
    if soft_threshold <= 0.0:
        raise ValueError("soft_threshold must be positive.")

    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    threshold_sq = float(soft_threshold) ** 2
    weight_sq = threshold_sq / (threshold_sq + float(np.dot(innovation, innovation)))
    return float(max(weight_sq, min_weight))


def wolf_tmd_weight_squared(
    innovation: np.ndarray,
    *,
    innovation_covariance: np.ndarray,
    threshold: float,
    min_weight: float = 1e-6,
) -> float:
    """
    Return the WoLF-TMD observation weight `w_t^2`.

    In the authors' linear-SSM reference code the TMD gate is a hard decision:
    the observation is either accepted (`w_t^2 = 1`) or nearly rejected
    (`w_t^2 ~= 0`). In this RL script we evaluate the Mahalanobis gate with the
    predictive observation covariance `S_t = P_pred + R` because the 4D policy
    observation uses zero measurement noise on wind coordinates, making `R`
    singular.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")

    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    innovation_covariance = attack_mod.project_to_psd(np.asarray(innovation_covariance, dtype=float))
    mahal_sq = float(np.dot(innovation, solve_spd(innovation_covariance, innovation)))
    mahal = float(np.sqrt(max(mahal_sq, 0.0)))
    weight_sq = 1.0 if mahal < float(threshold) else min_weight
    return float(weight_sq)


def wolf_kf_update_policy_observation(
    *,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
    wolf_kind: str,
    imq_soft_threshold: float,
    tmd_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Run one 4D policy-observation KF update with a WoLF weighting rule.

    The latent defended state is the policy observation itself, so the
    measurement model is the identity. The WoLF step is implemented by scaling
    the effective observation covariance with the inverse weight, exactly as in
    the reference implementations for linear SSMs.
    """
    m_pred = np.asarray(m_pred, dtype=float).reshape(4)
    P_pred = attack_mod.project_to_psd(np.asarray(P_pred, dtype=float))
    y_obs = np.asarray(y_obs, dtype=float).reshape(4)
    R = attack_mod.project_to_psd(np.asarray(R, dtype=float))

    innovation = y_obs - m_pred
    innovation_covariance = attack_mod.project_to_psd(P_pred + R)

    if wolf_kind == "imq":
        weight_sq = wolf_imq_weight_squared(
            innovation,
            soft_threshold=imq_soft_threshold,
        )
    elif wolf_kind == "tmd":
        weight_sq = wolf_tmd_weight_squared(
            innovation,
            innovation_covariance=innovation_covariance,
            threshold=tmd_threshold,
        )
    else:
        raise ValueError(f"Unsupported wolf_kind: {wolf_kind}")

    # Scaling `R` by `1 / w_t^2` reproduces the closed-form WoLF Gaussian update.
    effective_R = attack_mod.project_to_psd(R / float(weight_sq))
    S_eff = attack_mod.project_to_psd(P_pred + effective_R)
    K_eff = solve_spd(S_eff, P_pred.T).T
    m_post = m_pred + K_eff @ innovation

    # Joseph form preserves PSD numerically after the weighted update.
    I4 = np.eye(4, dtype=float)
    joseph_left = I4 - K_eff
    P_post = attack_mod.project_to_psd(
        joseph_left @ P_pred @ joseph_left.T + K_eff @ effective_R @ K_eff.T
    )

    diagnostics = {
        "weight_sq": float(weight_sq),
        "innovation_norm": float(np.linalg.norm(innovation)),
    }
    return m_post.astype(np.float32), P_post.astype(np.float32), diagnostics


def rollout_episode_return_attack_kf_wolf(
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
    wolf_kind: str,
    imq_soft_threshold: float,
    tmd_threshold: float,
    device: str = "cpu",
) -> float:
    """
    Roll out one episode under PGD attacks with a WoLF defense.

    The episode protocol is kept identical to the current covariance-adaptation
    code so the comparison remains paired step by step:
    - same initial nominal step,
    - same 4D PGD attack,
    - same observation and attack RNG conventions,
    - same policy evaluation.
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
            m_post, P_post, _diag = wolf_kf_update_policy_observation(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=obs_attack,
                R=R_policy,
                wolf_kind=wolf_kind,
                imq_soft_threshold=imq_soft_threshold,
                tmd_threshold=tmd_threshold,
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


def rollout_episode_return_random_attack_kf_wolf(
    env,
    model,
    *,
    attack_std: float,
    attack_eps: float,
    attack_prob: float,
    kf_meas_std: float,
    kf_proc_std: float,
    seed_for_attack: int,
    wolf_kind: str,
    imq_soft_threshold: float,
    tmd_threshold: float,
    device: str = "cpu",
) -> float:
    """
    Roll out one episode under random-boundary ellipsoid attacks with a WoLF defense.

    This mirrors the covariance-adaptation random-attack rollout so the only
    difference is the filter update rule used after an attacked observation is
    injected.
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
            m_post, P_post, _diag = wolf_kf_update_policy_observation(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=obs_attack,
                R=R_policy,
                wolf_kind=wolf_kind,
                imq_soft_threshold=imq_soft_threshold,
                tmd_threshold=tmd_threshold,
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


def compute_accumulated_reward_data_with_wolf(
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
    wolf_imq_soft_threshold: float,
    wolf_tmd_threshold: float,
    device: str,
) -> dict[str, np.ndarray | float | int]:
    """
    Compute accumulated rewards for KF, cov-adapt, WoLF-IMQ, and WoLF-TMD.

    All methods reuse the same episode seeds as the existing RL comparison
    scripts so the attack realizations and environment trajectories stay paired
    across every method in the figure.
    """
    if not np.isclose(omega_h + omega_o, 1.0, atol=1e-9):
        raise ValueError("omega_h and omega_o must sum to 1.")
    if not (0.0 <= delta_threshold <= 1.0):
        raise ValueError("delta_threshold must lie in [0, 1].")
    if wolf_imq_soft_threshold <= 0.0:
        raise ValueError("wolf_imq_soft_threshold must be positive.")
    if wolf_tmd_threshold <= 0.0:
        raise ValueError("wolf_tmd_threshold must be positive.")

    device_t = torch.device(device)
    model = load_policy(device_t, model_path)
    env_noisy_class = attack_mod.make_env_with_separate_obs_rng(rl_mod.AdvRL2DEnv)
    base_cfg = build_base_cfg()

    totals: dict[str, float] = {
        "clean": 0.0,
        "noisy_kf": 0.0,
        "attack_kf": 0.0,
        "random_kf": 0.0,
        "attack_wolf_imq": 0.0,
        "attack_wolf_tmd": 0.0,
        "random_wolf_imq": 0.0,
        "random_wolf_tmd": 0.0,
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
        cfg_attack_wolf_imq = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_attack_wolf_tmd = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_random_wolf_imq = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})
        cfg_random_wolf_tmd = rl_mod.AdvRLEnvConfig(**{**asdict(base_cfg)})

        cfg_clean.seed = seed
        cfg_clean.obs_noise_std = 0.0
        cfg_noisy.seed = seed
        cfg_noisy.obs_noise_std = float(noise_std)
        cfg_attack.seed = seed
        cfg_attack.obs_noise_std = 0.0
        cfg_random.seed = seed
        cfg_random.obs_noise_std = 0.0
        cfg_attack_wolf_imq.seed = seed
        cfg_attack_wolf_imq.obs_noise_std = 0.0
        cfg_attack_wolf_tmd.seed = seed
        cfg_attack_wolf_tmd.obs_noise_std = 0.0
        cfg_random_wolf_imq.seed = seed
        cfg_random_wolf_imq.obs_noise_std = 0.0
        cfg_random_wolf_tmd.seed = seed
        cfg_random_wolf_tmd.obs_noise_std = 0.0

        env_clean = rl_mod.AdvRL2DEnv(cfg_clean)
        env_noisy = env_noisy_class(cfg_noisy)
        env_attack = rl_mod.AdvRL2DEnv(cfg_attack)
        env_random = rl_mod.AdvRL2DEnv(cfg_random)
        env_attack_wolf_imq = rl_mod.AdvRL2DEnv(cfg_attack_wolf_imq)
        env_attack_wolf_tmd = rl_mod.AdvRL2DEnv(cfg_attack_wolf_tmd)
        env_random_wolf_imq = rl_mod.AdvRL2DEnv(cfg_random_wolf_imq)
        env_random_wolf_tmd = rl_mod.AdvRL2DEnv(cfg_random_wolf_tmd)

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
        ret_attack_wolf_imq = rollout_episode_return_attack_kf_wolf(
            env_attack_wolf_imq,
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
            wolf_kind="imq",
            imq_soft_threshold=wolf_imq_soft_threshold,
            tmd_threshold=wolf_tmd_threshold,
            device=device,
        )
        ret_attack_wolf_tmd = rollout_episode_return_attack_kf_wolf(
            env_attack_wolf_tmd,
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
            wolf_kind="tmd",
            imq_soft_threshold=wolf_imq_soft_threshold,
            tmd_threshold=wolf_tmd_threshold,
            device=device,
        )
        ret_random_wolf_imq = rollout_episode_return_random_attack_kf_wolf(
            env_random_wolf_imq,
            model,
            attack_std=attack_std,
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            kf_meas_std=kf_meas_std,
            kf_proc_std=kf_proc_std,
            seed_for_attack=seed,
            wolf_kind="imq",
            imq_soft_threshold=wolf_imq_soft_threshold,
            tmd_threshold=wolf_tmd_threshold,
            device=device,
        )
        ret_random_wolf_tmd = rollout_episode_return_random_attack_kf_wolf(
            env_random_wolf_tmd,
            model,
            attack_std=attack_std,
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            kf_meas_std=kf_meas_std,
            kf_proc_std=kf_proc_std,
            seed_for_attack=seed,
            wolf_kind="tmd",
            imq_soft_threshold=wolf_imq_soft_threshold,
            tmd_threshold=wolf_tmd_threshold,
            device=device,
        )

        totals["clean"] += float(ret_clean)
        totals["noisy_kf"] += float(ret_noisy)
        totals["attack_kf"] += float(ret_attack)
        totals["random_kf"] += float(ret_random)
        totals["attack_wolf_imq"] += float(ret_attack_wolf_imq)
        totals["attack_wolf_tmd"] += float(ret_attack_wolf_tmd)
        totals["random_wolf_imq"] += float(ret_random_wolf_imq)
        totals["random_wolf_tmd"] += float(ret_random_wolf_tmd)

        series["clean"].append(totals["clean"])
        series["noisy_kf"].append(totals["noisy_kf"])
        series["attack_kf"].append(totals["attack_kf"])
        series["random_kf"].append(totals["random_kf"])
        series["attack_wolf_imq"].append(totals["attack_wolf_imq"])
        series["attack_wolf_tmd"].append(totals["attack_wolf_tmd"])
        series["random_wolf_imq"].append(totals["random_wolf_imq"])
        series["random_wolf_tmd"].append(totals["random_wolf_tmd"])

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
            f"rand+KF={totals['random_kf']:.1f} | "
            f"PGD+IMQ={totals['attack_wolf_imq']:.1f} | "
            f"PGD+TMD={totals['attack_wolf_tmd']:.1f}"
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
        "wolf_imq_soft_threshold": float(wolf_imq_soft_threshold),
        "wolf_tmd_threshold": float(wolf_tmd_threshold),
        "c_scales": np.asarray(c_scales, dtype=float),
        "acc_clean": np.asarray(series["clean"], dtype=float),
        "acc_noisy_kf": np.asarray(series["noisy_kf"], dtype=float),
        "acc_attack_kf": np.asarray(series["attack_kf"], dtype=float),
        "acc_random_kf": np.asarray(series["random_kf"], dtype=float),
        "acc_attack_wolf_imq": np.asarray(series["attack_wolf_imq"], dtype=float),
        "acc_attack_wolf_tmd": np.asarray(series["attack_wolf_tmd"], dtype=float),
        "acc_random_wolf_imq": np.asarray(series["random_wolf_imq"], dtype=float),
        "acc_random_wolf_tmd": np.asarray(series["random_wolf_tmd"], dtype=float),
    }
    for c_scale in c_scales:
        data[f"acc_attack_cov_{c_scale:g}"] = np.asarray(series[f"attack_cov_{c_scale:g}"], dtype=float)
        data[f"acc_random_cov_{c_scale:g}"] = np.asarray(series[f"random_cov_{c_scale:g}"], dtype=float)
    return data


def build_panel_specifications(
    c_scales: tuple[float, ...],
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """Return the label/key mapping for the three grouped-bar panels."""
    baseline_spec = [
        ("Noise-Free", "acc_clean"),
        ("Noise + KF", "acc_noisy_kf"),
        ("Attack + KF", "acc_attack_kf"),
        (r"Boundary $\epsilon$-perturbation + KF", "acc_random_kf"),
    ]

    attack_spec = [
        ("Noise-Free", "acc_clean"),
        ("Noise + KF", "acc_noisy_kf"),
        ("Attack + KF", "acc_attack_kf"),
    ]
    attack_spec.extend(
        [
            (
                rf"Attack + cov-adapt ($\lambda={c_scale:g}\lambda_{{\max}}$)",
                f"acc_attack_cov_{c_scale:g}",
            )
            for c_scale in c_scales
        ]
    )
    attack_spec.extend(
        [
            ("Attack + WoLF-IMQ", "acc_attack_wolf_imq"),
            ("Attack + WoLF-TMD", "acc_attack_wolf_tmd"),
        ]
    )

    random_spec = [
        ("Noise-Free", "acc_clean"),
        ("Noise + KF", "acc_noisy_kf"),
        (r"Boundary $\epsilon$-perturbation + KF", "acc_random_kf"),
    ]
    random_spec.extend(
        [
            (
                rf"Boundary $\epsilon$-perturbation + cov-adapt ($\lambda={c_scale:g}\lambda_{{\max}}$)",
                f"acc_random_cov_{c_scale:g}",
            )
            for c_scale in c_scales
        ]
    )
    random_spec.extend(
        [
            (r"Boundary $\epsilon$-perturbation + WoLF-IMQ", "acc_random_wolf_imq"),
            (r"Boundary $\epsilon$-perturbation + WoLF-TMD", "acc_random_wolf_tmd"),
        ]
    )
    return baseline_spec, attack_spec, random_spec


def plot_accumulated_reward_comparison_two_epsilons_with_wolf(
    *,
    data_by_epsilon: dict[float, dict[str, np.ndarray | float | int]],
    epsilon_values: tuple[float, ...],
    outpath: str,
) -> None:
    """
    Plot the three-panel bar comparison for two attack-radius values.

    Method identity comes from bar color and epsilon identity comes from hatch.
    The clean and noisy references are epsilon-invariant, so they are drawn
    once per group center rather than duplicated.
    """
    set_plot_theme()

    reference_data = data_by_epsilon[float(epsilon_values[0])]
    c_scales = tuple(float(value) for value in np.asarray(reference_data["c_scales"], dtype=float))
    baseline_spec, attack_spec, random_spec = build_panel_specifications(c_scales)

    method_colors = {
        "acc_clean": "#6FAF8F",
        "acc_noisy_kf": "#A7D37A",
        "acc_attack_kf": "#E9B188",
        "acc_random_kf": "#8FAEDF",
        # WoLF colors are separated by panel family so they do not visually
        # compete with the covariance-adaptation bars.
        "acc_attack_wolf_imq": "#B69BD9",
        "acc_attack_wolf_tmd": "#8C72BF",
        "acc_random_wolf_imq": "#B69BD9",
        "acc_random_wolf_tmd": "#8C72BF",
    }
    attack_cov_colors = {
        f"acc_attack_cov_{c_scale:g}": color
        for c_scale, color in zip(c_scales, ["#E4B2AA", "#C96F5C", "#A54E42"], strict=False)
    }
    random_cov_colors = {
        f"acc_random_cov_{c_scale:g}": color
        for c_scale, color in zip(c_scales, ["#B8C9EE", "#6F92D8", "#4D73BD"], strict=False)
    }
    method_colors.update(attack_cov_colors)
    method_colors.update(random_cov_colors)

    hatch_by_epsilon = {
        float(epsilon_values[0]): "",
        float(epsilon_values[1]): "//////" if len(epsilon_values) > 1 else "",
    }
    epsilon_invariant_keys = {"acc_clean", "acc_noisy_kf"}

    def panel_values(spec: list[tuple[str, str]]) -> dict[float, list[float]]:
        """Return normalized final values for every epsilon in one panel."""
        values_by_epsilon: dict[float, list[float]] = {}
        for epsilon in epsilon_values:
            panel_data = data_by_epsilon[float(epsilon)]
            panel_n_episodes = int(panel_data["n_episodes"])
            values_by_epsilon[float(epsilon)] = [
                normalized_final_reward(np.asarray(panel_data[key], dtype=float), n_episodes=panel_n_episodes)
                for _label, key in spec
            ]
        return values_by_epsilon

    baseline_values = panel_values(baseline_spec)
    attack_values = panel_values(attack_spec)
    random_values = panel_values(random_spec)

    all_values = []
    for values_by_epsilon in (baseline_values, attack_values, random_values):
        for epsilon in epsilon_values:
            all_values.extend(values_by_epsilon[float(epsilon)])
    all_values_np = np.asarray(all_values, dtype=float)
    value_abs_max = float(np.max(np.abs(all_values_np))) if all_values_np.size > 0 else 1.0
    value_abs_max = max(value_abs_max, 1.0)
    y_pad = 0.18 * value_abs_max
    y_min = float(np.min(all_values_np)) - y_pad
    y_max = float(np.max(all_values_np)) + y_pad
    # Keep extra room below the bars so the middle legend does not hide the
    # most negative values in the shared y-axis layout.
    y_min = min(y_min, -25.0)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22.4, 6.5),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.82, 1.18, 1.18]},
    )
    ax_baseline, ax_attack, ax_random = axes

    def draw_grouped_bar_panel(
        *,
        ax: plt.Axes,
        spec: list[tuple[str, str]],
        values_by_epsilon: dict[float, list[float]],
        legend_width: str = "normal",
        legend_loc: str = "lower left",
    ) -> None:
        """
        Draw one grouped-bar panel with method and epsilon legends.

        The x-axis text is hidden because the method legend is a cleaner map
        once each method is split into epsilon-specific bars.
        """
        labels = [label for label, _key in spec]
        keys = [key for _label, key in spec]
        x = np.arange(len(labels), dtype=float)
        if x.size >= 2:
            x[1:] += 0.28
            x[1] -= 0.48

        n_eps = len(epsilon_values)
        group_width = 0.76
        bar_width = group_width / float(max(1, n_eps))
        offsets = (
            np.linspace(-0.5 * group_width + 0.5 * bar_width, 0.5 * group_width - 0.5 * bar_width, n_eps)
            if n_eps > 1
            else np.array([0.0], dtype=float)
        )
        key_by_index = {idx: key for idx, (_label, key) in enumerate(spec)}

        single_indices = [idx for idx, key in key_by_index.items() if key in epsilon_invariant_keys]
        if single_indices:
            single_positions = x[single_indices]
            single_values = np.asarray(values_by_epsilon[float(epsilon_values[0])], dtype=float)[single_indices]
            single_colors = [method_colors[key_by_index[idx]] for idx in single_indices]
            single_bars = ax.bar(
                single_positions,
                single_values,
                width=0.66 * group_width,
                color=single_colors,
                edgecolor=[darken_hex(color) for color in single_colors],
                linewidth=1.0,
                alpha=0.97,
                zorder=3,
            )

            for bar, value in zip(single_bars, single_values):
                offset = 0.028 * (y_max - y_min)
                text_y = value + offset if value >= 0.0 else value - offset
                ax.text(
                    float(bar.get_x() + 0.5 * bar.get_width()),
                    float(text_y),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0.0 else "top",
                    fontsize=8.6,
                    color="#2F2F2F",
                )

        for eps_idx, epsilon in enumerate(epsilon_values):
            varying_indices = [idx for idx, key in key_by_index.items() if key not in epsilon_invariant_keys]
            if not varying_indices:
                continue

            bar_positions = x[varying_indices] + offsets[eps_idx]
            panel_values_eps = np.asarray(values_by_epsilon[float(epsilon)], dtype=float)[varying_indices]
            panel_colors = [method_colors[key_by_index[idx]] for idx in varying_indices]
            bars = ax.bar(
                bar_positions,
                panel_values_eps,
                width=0.92 * bar_width,
                color=panel_colors,
                edgecolor=[darken_hex(color) for color in panel_colors],
                linewidth=1.0,
                hatch=hatch_by_epsilon[float(epsilon)],
                alpha=0.97,
                zorder=3,
            )

            for bar, value in zip(bars, panel_values_eps):
                offset = 0.028 * (y_max - y_min)
                text_y = value + offset if value >= 0.0 else value - offset
                ax.text(
                    float(bar.get_x() + 0.5 * bar.get_width()),
                    float(text_y),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0.0 else "top",
                    fontsize=8.6,
                    color="#2F2F2F",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.axhline(0.0, color="#4E4E4E", linewidth=1.15, alpha=0.92, zorder=1)
        ax.set_ylim(y_min, y_max)
        ax.set_axisbelow(True)

        method_handles = [
            Patch(
                facecolor=method_colors[key],
                edgecolor=darken_hex(method_colors[key]),
                linewidth=1.0,
            )
            for key in keys
        ]

        if legend_width == "wide":
            borderpad = 0.80
            handlelength = 3.1
            handletextpad = 0.98
            labelspacing = 0.63
        elif legend_width == "medium":
            borderpad = 0.64
            handlelength = 2.7
            handletextpad = 0.88
            labelspacing = 0.55
        else:
            borderpad = 0.54
            handlelength = 2.3
            handletextpad = 0.78
            labelspacing = 0.50

        method_legend = ax.legend(
            method_handles,
            labels,
            loc=legend_loc,
            frameon=True,
            framealpha=0.97,
            borderpad=borderpad,
            fontsize=9.3,
            handlelength=handlelength,
            handletextpad=handletextpad,
            labelspacing=labelspacing,
        )
        ax.add_artist(method_legend)

        epsilon_handles = [
            Patch(
                facecolor="#FFFFFF",
                edgecolor="#666666",
                linewidth=1.0,
                hatch=hatch_by_epsilon[float(epsilon)],
            )
            for epsilon in epsilon_values
        ]
        epsilon_labels = [rf"$\epsilon={epsilon:g}$" for epsilon in epsilon_values]
        ax.legend(
            epsilon_handles,
            epsilon_labels,
            loc="upper right",
            frameon=True,
            framealpha=0.97,
            borderpad=0.40,
            fontsize=9.1,
            handlelength=2.0,
            handletextpad=0.70,
            labelspacing=0.42,
        )

    for ax in axes:
        style_axis(ax)

    draw_grouped_bar_panel(
        ax=ax_baseline,
        spec=baseline_spec,
        values_by_epsilon=baseline_values,
    )
    draw_grouped_bar_panel(
        ax=ax_attack,
        spec=attack_spec,
        values_by_epsilon=attack_values,
        legend_width="wide",
        legend_loc="lower right",
    )
    draw_grouped_bar_panel(
        ax=ax_random,
        spec=random_spec,
        values_by_epsilon=random_values,
        legend_width="wide",
    )

    ax_baseline.set_ylabel("Mean accumulated reward")

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    """
    Generate the two-epsilon RL comparison including WoLF-IMQ and WoLF-TMD.

    Change the defaults below directly when you want a lighter validation run
    or a larger experiment. As in the rest of this folder, the configuration
    is kept in plain Python variables instead of environment variables.
    """
    model_path = os.path.abspath(os.path.join(RL_4D_DIR, "outputs", "saved_models", "AdvRL_v2_policy.pt"))
    device = "cpu"
    noise_std = 0.5
    attack_prob = 0.15
    attack_std = noise_std
    attack_eps_values = (0.75, 0.95)
    kf_meas_std = noise_std
    kf_proc_std = 0.03
    pgd_steps = 65
    pgd_step_size = 0.25
    mc_samples = 256
    n_episodes = 100
    seed0 = 1_000

    c_scales = (0.5, 1.0, 2.0)
    omega_h = 0.50
    omega_o = 0.50
    delta_threshold = 0.20
    # Slightly more aggressive defaults so WoLF downweights suspicious
    # observations earlier than in the initial comparison setup.
    wolf_imq_soft_threshold = 0.55
    wolf_tmd_threshold = 2.5
    force_cache = False

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at:\n  {model_path}")

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    c_tag = "-".join(f"{value:g}" for value in c_scales).replace(".", "p")
    eps_tag = "-".join(str(value).replace(".", "p") for value in attack_eps_values)
    imq_tag = str(wolf_imq_soft_threshold).replace(".", "p")
    tmd_tag = str(wolf_tmd_threshold).replace(".", "p")

    outpath = os.path.join(
        out_dir,
        (
            "comparison_RL_v2_wind_4dattack_two_eps_wolf_"
            f"N{n_episodes}_p{attack_prob}_eps{eps_tag}_c{c_tag}_"
            f"imq{imq_tag}_tmd{tmd_tag}.png"
        ),
    )

    data_by_epsilon: dict[float, dict[str, np.ndarray | float | int]] = {}
    for attack_eps in attack_eps_values:
        single_eps_outpath = os.path.join(
            out_dir,
            (
                "comparison_RL_v2_wind_4dattack_wolf_"
                f"N{n_episodes}_p{attack_prob}_eps{str(attack_eps).replace('.', 'p')}_"
                f"c{c_tag}_imq{imq_tag}_tmd{tmd_tag}.png"
            ),
        )
        data_path = data_path_for_plot(single_eps_outpath)

        def compute_data_for_epsilon(attack_eps_value: float = float(attack_eps)) -> dict[str, np.ndarray | float | int]:
            return compute_accumulated_reward_data_with_wolf(
                n_episodes=n_episodes,
                seed0=seed0,
                model_path=model_path,
                noise_std=noise_std,
                attack_std=attack_std,
                attack_eps=attack_eps_value,
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
                wolf_imq_soft_threshold=wolf_imq_soft_threshold,
                wolf_tmd_threshold=wolf_tmd_threshold,
                device=device,
            )

        data_by_epsilon[float(attack_eps)] = cached_npz(
            data_path,
            compute_data_for_epsilon,
            force=force_cache,
        )

    plot_accumulated_reward_comparison_two_epsilons_with_wolf(
        data_by_epsilon=data_by_epsilon,
        epsilon_values=tuple(float(value) for value in attack_eps_values),
        outpath=outpath,
    )
    print(f"Saved figure to: {outpath}")


if __name__ == "__main__":
    main()
