#!/usr/bin/env python3
"""
summarize_cartpole_attack_geometry.py

Summarize how the CartPole attacks behave relative to the EKF attack
constraint, without printing one full block per attacked step.

What this script reports:
1. For three epsilon values and for two attack modes (`pgd` and `random`), it
   aggregates attack statistics over a short batch of CartPole rollouts.
2. It prints how often the attacked observation ends up closer to the predicted
   observation center `o_pred = H m_pred` than the nominal noisy observation.
   In this benchmark `H = I`, so `o_pred = m_pred`.
3. It prints how often the final attacked observation:
   - stays strictly inside the ellipsoid,
   - lands numerically on the boundary,
   - violates the constraint.

Important geometric convention:
    E_t = { y : (y - m_pred)^T (P_pred + R)^(-1) (y - m_pred) <= epsilon }.

This script now reuses the globally calibrated fixed-discount CartPole
predictor from the Gymnasium benchmark so the inspection numbers match the
shared filtering geometry.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import cartpole_covadapt_compare_epsilons_wolf as cartpole_mod


@dataclass
class AttackSummary:
    """Accumulate summary statistics for one `(mode, epsilon)` pair."""

    count: int = 0
    closer_than_nominal: int = 0
    strict_inside: int = 0
    on_boundary: int = 0
    violates: int = 0
    eps_real_sum: float = 0.0
    eps_adv_sum: float = 0.0
    critic_sum: float = 0.0


def quadratic_radius_sq(
    *,
    vector: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """
    Return `v^T Sigma^{-1} v` using the repository SPD solver.
    """
    vector = np.asarray(vector, dtype=float).reshape(-1)
    Sigma = cartpole_mod.project_to_psd(np.asarray(Sigma, dtype=float))
    return float(np.dot(vector, cartpole_mod.solve_spd(Sigma, vector)))


def achieved_attack_epsilon(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """
    Return the ellipsoidal epsilon reached by one observation.
    """
    delta = np.asarray(observation, dtype=float) - np.asarray(center, dtype=float)
    return quadratic_radius_sq(vector=delta, Sigma=Sigma)


def build_correlated_filter_covariances() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the CartPole filter noise settings used by this inspection script.

    The observation covariance still follows the same correlated 4D structure
    as before. The process-noise block is kept only as a legacy reference for
    building comparable covariance matrices, while the actual predictor now
    uses the shared fixed discount from the main Gymnasium benchmark.
    """
    kf_meas_std = np.array([0.080, 0.180, 0.040, 0.180], dtype=np.float32)
    kf_proc_std = np.array([0.320, 0.720, 0.160, 0.720], dtype=np.float32)

    kf_meas_corr = np.array(
        [
            [1.00, 0.22, 0.08, 0.00],
            [0.22, 1.00, 0.18, 0.28],
            [0.08, 0.18, 1.00, 0.24],
            [0.00, 0.28, 0.24, 1.00],
        ],
        dtype=np.float32,
    )
    kf_proc_corr = np.array(
        [
            [1.00, 0.40, 0.16, 0.00],
            [0.40, 1.00, 0.30, 0.42],
            [0.16, 0.30, 1.00, 0.34],
            [0.00, 0.42, 0.34, 1.00],
        ],
        dtype=np.float32,
    )
    return kf_meas_std, kf_proc_std, kf_meas_corr, kf_proc_corr


def print_summary_table(
    *,
    epsilon_values: tuple[float, ...],
    summaries: dict[tuple[str, float], AttackSummary],
    tol: float,
) -> None:
    """
    Print one compact summary row per attack mode and epsilon.
    """
    print("CartPole attack inspection summary")
    print(f"strict_tolerance = {tol:.1e}")
    print("metric_closer = pct[o_adv is closer to o_pred=m_pred than o_real, in Mahalanobis radius]")
    print("-" * 122)
    print(
        "mode     epsilon   n    pct_closer   pct_strict_inside   pct_on_boundary   "
        "pct_violates   mean_eps_real   mean_eps_adv   mean_critic"
    )
    print("-" * 122)

    for mode in ("pgd", "random"):
        for epsilon in epsilon_values:
            summary = summaries[(mode, float(epsilon))]
            n = max(summary.count, 1)
            print(
                f"{mode:<8} {epsilon:>7.2f} "
                f"{summary.count:>4d} "
                f"{100.0 * summary.closer_than_nominal / n:>11.2f}% "
                f"{100.0 * summary.strict_inside / n:>18.2f}% "
                f"{100.0 * summary.on_boundary / n:>17.2f}% "
                f"{100.0 * summary.violates / n:>12.2f}% "
                f"{summary.eps_real_sum / n:>15.4f} "
                f"{summary.eps_adv_sum / n:>14.4f} "
                f"{summary.critic_sum / n:>12.4f}"
            )


def inspect_attack_examples() -> None:
    """
    Aggregate attack diagnostics over a short batch of CartPole rollouts.
    """
    seed0 = 7
    n_episodes = 3
    max_steps_per_episode = 15
    epsilon_values = (0.35, 0.95, 1.80)
    pgd_steps = 20
    pgd_step_size = 0.20
    mc_samples = 128
    strict_tol = 1e-2
    discount_delta = cartpole_mod.DEFAULT_GYMNASIUM_DISCOUNT_DELTA

    kf_meas_std, kf_proc_std, kf_meas_corr, kf_proc_corr = build_correlated_filter_covariances()
    R, legacy_Q = cartpole_mod.build_filter_covariances(
        meas_std=kf_meas_std,
        proc_std=kf_proc_std,
        meas_corr=kf_meas_corr,
        proc_corr=kf_proc_corr,
    )

    model_path = cartpole_mod.ensure_downloaded_cartpole_checkpoint()
    device = torch.device("cpu")
    model = cartpole_mod.load_cartpole_policy(model_path, device)
    ssm = cartpole_mod.build_cartpole_linear_ssm()

    summaries: dict[tuple[str, float], AttackSummary] = {
        (mode, float(epsilon)): AttackSummary()
        for mode in ("pgd", "random")
        for epsilon in epsilon_values
    }

    for episode_idx in range(int(n_episodes)):
        seed = int(seed0 + episode_idx)
        env = gym.make("CartPole-v1")
        obs, _info = env.reset(seed=seed)

        rng_nominal_noise = np.random.default_rng(seed + 707_002)
        init_noise = rng_nominal_noise.normal(0.0, kf_meas_std, size=(4,)).astype(np.float32)
        obs_init_noisy = (np.asarray(obs, dtype=np.float32) + init_noise).astype(np.float32)

        m_post, P_post = cartpole_mod.kf_update_state(
            m_pred=np.asarray(obs, dtype=np.float32),
            P_pred=cartpole_mod.project_to_psd(R.copy()),
            y_obs=obs_init_noisy,
            R=R,
        )
        action = cartpole_mod.select_action(model, m_post)
        force = cartpole_mod.action_to_force(action, ssm.force_mag)
        m_pred, P_pred = cartpole_mod.kf_predict_state(
            m_post=m_post,
            P_post=P_post,
            force=force,
            Q=None,
            discount_delta=discount_delta,
            ssm=ssm,
        )

        obs, reward, terminated, truncated, _info = env.step(action)
        _ = reward
        if terminated or truncated:
            env.close()
            continue

        step_idx = 1
        while step_idx <= int(max_steps_per_episode):
            y_clean = np.asarray(obs, dtype=np.float32)
            nominal_noise = rng_nominal_noise.normal(0.0, kf_meas_std, size=(4,)).astype(np.float32)
            o_real = (y_clean + nominal_noise).astype(np.float32)
            attack_center = np.asarray(m_pred, dtype=np.float32)
            attack_sigma = cartpole_mod.project_to_psd(np.asarray(P_pred, dtype=float) + np.asarray(R, dtype=float))
            eps_real = achieved_attack_epsilon(
                observation=o_real,
                center=attack_center,
                Sigma=attack_sigma,
            )

            for eps_idx, epsilon in enumerate(epsilon_values):
                o_adv_pgd, obj_star, _m_post_attack, _P_post_attack = cartpole_mod.pgd_attack_on_expected_value(
                    model=model,
                    obs_nom=o_real,
                    m_pred=attack_center,
                    P_pred=P_pred,
                    R=R,
                    attack_center=attack_center,
                    attack_sigma=attack_sigma,
                    attack_eps=float(epsilon),
                    pgd_steps=pgd_steps,
                    pgd_step_size=pgd_step_size,
                    mc_samples=mc_samples,
                    rng_seed=seed + 10_000 * step_idx + eps_idx,
                    device="cpu",
                )
                eps_adv_pgd = achieved_attack_epsilon(
                    observation=o_adv_pgd,
                    center=attack_center,
                    Sigma=attack_sigma,
                )
                summary_pgd = summaries[("pgd", float(epsilon))]
                summary_pgd.count += 1
                summary_pgd.closer_than_nominal += int(eps_adv_pgd < eps_real)
                summary_pgd.strict_inside += int(eps_adv_pgd < float(epsilon) - strict_tol)
                summary_pgd.on_boundary += int(abs(eps_adv_pgd - float(epsilon)) <= strict_tol)
                summary_pgd.violates += int(eps_adv_pgd > float(epsilon) + strict_tol)
                summary_pgd.eps_real_sum += float(eps_real)
                summary_pgd.eps_adv_sum += float(eps_adv_pgd)
                summary_pgd.critic_sum += float(obj_star)

                rng_boundary = np.random.default_rng(seed + 20_000 * step_idx + eps_idx)
                o_adv_random = cartpole_mod.sample_random_attack_in_ellipsoid(
                    center=attack_center,
                    Sigma=attack_sigma,
                    epsilon=float(epsilon),
                    rng=rng_boundary,
                )
                m_post_random, _ = cartpole_mod.kf_update_state(
                    m_pred=attack_center,
                    P_pred=P_pred,
                    y_obs=o_adv_random,
                    R=R,
                )
                random_critic = float(
                    cartpole_mod.critic_state_values(
                        model,
                        torch.tensor(m_post_random, dtype=torch.float32).unsqueeze(0),
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                eps_adv_random = achieved_attack_epsilon(
                    observation=o_adv_random,
                    center=attack_center,
                    Sigma=attack_sigma,
                )
                summary_random = summaries[("random", float(epsilon))]
                summary_random.count += 1
                summary_random.closer_than_nominal += int(eps_adv_random < eps_real)
                summary_random.strict_inside += int(eps_adv_random < float(epsilon) - strict_tol)
                summary_random.on_boundary += int(abs(eps_adv_random - float(epsilon)) <= strict_tol)
                summary_random.violates += int(eps_adv_random > float(epsilon) + strict_tol)
                summary_random.eps_real_sum += float(eps_real)
                summary_random.eps_adv_sum += float(eps_adv_random)
                summary_random.critic_sum += float(random_critic)

            m_post_clean, P_post_clean = cartpole_mod.kf_update_state(
                m_pred=attack_center,
                P_pred=P_pred,
                y_obs=o_real,
                R=R,
            )
            action = cartpole_mod.select_action(model, m_post_clean)
            force = cartpole_mod.action_to_force(action, ssm.force_mag)
            m_pred, P_pred = cartpole_mod.kf_predict_state(
                m_post=m_post_clean,
                P_post=P_post_clean,
                force=force,
                Q=None,
                discount_delta=discount_delta,
                ssm=ssm,
            )
            obs, reward, terminated, truncated, _info = env.step(action)
            _ = reward
            step_idx += 1

            if terminated or truncated:
                break

        env.close()

    print("CartPole attack inspection")
    print(f"seed0 = {seed0}")
    print(f"n_episodes = {n_episodes}")
    print(f"max_steps_per_episode = {max_steps_per_episode}")
    print(f"pgd_steps = {pgd_steps}")
    print(f"pgd_step_size = {pgd_step_size}")
    print(f"mc_samples = {mc_samples}")
    print(f"discount_delta = {discount_delta:.2f}")
    print(f"epsilon_values = {epsilon_values}")
    print(f"meas_std = {np.asarray(kf_meas_std, dtype=float)}")
    print(f"legacy_proc_std = {np.asarray(kf_proc_std, dtype=float)}")
    print("R =")
    print(np.asarray(R, dtype=float))
    print("legacy_Q_reference =")
    print(np.asarray(legacy_Q, dtype=float))
    print_summary_table(
        epsilon_values=epsilon_values,
        summaries=summaries,
        tol=strict_tol,
    )


def main() -> None:
    """
    Entry point for the CartPole attack inspection script.
    """
    inspect_attack_examples()


if __name__ == "__main__":
    main()
