#!/usr/bin/env python3
"""
sweep_cartpole_transition_q.py

Sweep several additive transition-noise covariance scales around the shared
CartPole discounted predictor and report how the attack diagnostics change.

Why this script exists:
1. In the current CartPole setup, a recurring question is whether the filter
   is too confident in the transition model, which makes `o_real` fall too far
   from the predicted center `o_pred = H m_pred`.
2. After calibrating one fixed discount for the shared Gymnasium predictor, the
   next question is whether adding an extra transition covariance floor still
   helps or just makes the filter too conservative.
3. This script reuses the same attack construction as the CartPole benchmark
   and prints compact tables for several additive `Q` scales.

What it reports for each `Q` scale:
1. `mean_eps_real`, the average Mahalanobis radius of the nominal noisy
   observation with respect to the attack geometry.
2. `pct_real_inside`, the percentage of nominal noisy observations already
   inside the attack ellipsoid for each epsilon.
3. For both `pgd` and `random` attacks:
   - `pct_closer`: how often `o_adv` is closer to `o_pred = m_pred` than
     `o_real`,
   - `mean_eps_adv`: the achieved adversarial radius,
   - `mean_critic`: the local value proxy after the attack.

The goal is not to benchmark reward here, but to understand whether an
additional transition covariance floor on top of the calibrated discount
predictor makes the attack geometry more compatible with the noisy
observations seen by the filter.
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
class SweepSummary:
    """Accumulate diagnostics for one `(mode, epsilon)` pair inside one Q scale."""

    count: int = 0
    closer_than_nominal: int = 0
    eps_adv_sum: float = 0.0
    critic_sum: float = 0.0


def quadratic_radius_sq(
    *,
    vector: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """Return `v^T Sigma^{-1} v` using the repository SPD helper."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    Sigma = cartpole_mod.project_to_psd(np.asarray(Sigma, dtype=float))
    return float(np.dot(vector, cartpole_mod.solve_spd(Sigma, vector)))


def achieved_attack_epsilon(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """Return the attack radius of one observation relative to the attack center."""
    delta = np.asarray(observation, dtype=float) - np.asarray(center, dtype=float)
    return quadratic_radius_sq(vector=delta, Sigma=Sigma)


def base_noise_settings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the base correlated observation/process-noise settings.

    The process standard deviations are the base values that will later be
    multiplied by several `Q` scales.
    """
    meas_std = np.array([0.080, 0.180, 0.040, 0.180], dtype=np.float32)
    proc_std = np.array([0.090, 0.200, 0.045, 0.200], dtype=np.float32)

    meas_corr = np.array(
        [
            [1.00, 0.22, 0.08, 0.00],
            [0.22, 1.00, 0.18, 0.28],
            [0.08, 0.18, 1.00, 0.24],
            [0.00, 0.28, 0.24, 1.00],
        ],
        dtype=np.float32,
    )
    proc_corr = np.array(
        [
            [1.00, 0.40, 0.16, 0.00],
            [0.40, 1.00, 0.30, 0.42],
            [0.16, 0.30, 1.00, 0.34],
            [0.00, 0.42, 0.34, 1.00],
        ],
        dtype=np.float32,
    )
    return meas_std, proc_std, meas_corr, proc_corr


def print_scale_report(
    *,
    q_scale: float,
    proc_std: np.ndarray,
    mean_eps_real: float,
    pct_inside_by_epsilon: dict[float, float],
    summaries: dict[tuple[str, float], SweepSummary],
    epsilon_values: tuple[float, ...],
) -> None:
    """Print one compact report block for a single Q scale."""
    print("=" * 122)
    print(f"q_scale = {q_scale:.2f}")
    print(f"proc_std = {np.asarray(proc_std, dtype=float)}")
    print(f"mean_eps_real = {mean_eps_real:.4f}")
    print(
        "pct_real_inside = "
        + ", ".join(f"eps={eps:.2f}:{pct_inside_by_epsilon[float(eps)]:.2f}%" for eps in epsilon_values)
    )
    print("-" * 122)
    print("mode     epsilon   n    pct_closer   mean_eps_adv   mean_critic")
    print("-" * 122)

    for mode in ("pgd", "random"):
        for epsilon in epsilon_values:
            summary = summaries[(mode, float(epsilon))]
            n = max(summary.count, 1)
            print(
                f"{mode:<8} {epsilon:>7.2f} "
                f"{summary.count:>4d} "
                f"{100.0 * summary.closer_than_nominal / n:>11.2f}% "
                f"{summary.eps_adv_sum / n:>14.4f} "
                f"{summary.critic_sum / n:>12.4f}"
            )


def run_q_sweep() -> None:
    """
    Sweep several transition-covariance scales and summarize attack geometry.
    """
    seed0 = 7
    n_episodes = 3
    max_steps_per_episode = 12
    epsilon_values = (0.35, 0.95, 1.80)
    q_scale_values = (0.50, 1.00, 1.50, 2.00, 3.00, 3.50)
    pgd_steps = 20
    pgd_step_size = 0.20
    mc_samples = 64
    discount_delta = cartpole_mod.DEFAULT_GYMNASIUM_DISCOUNT_DELTA

    meas_std, proc_std_base, meas_corr, proc_corr = base_noise_settings()
    model_path = cartpole_mod.ensure_downloaded_cartpole_checkpoint()
    device = torch.device("cpu")
    model = cartpole_mod.load_cartpole_policy(model_path, device)
    ssm = cartpole_mod.build_cartpole_linear_ssm()

    print("CartPole transition-Q sweep")
    print(f"seed0 = {seed0}")
    print(f"n_episodes = {n_episodes}")
    print(f"max_steps_per_episode = {max_steps_per_episode}")
    print(f"pgd_steps = {pgd_steps}")
    print(f"pgd_step_size = {pgd_step_size}")
    print(f"mc_samples = {mc_samples}")
    print(f"discount_delta = {discount_delta:.2f}")
    print(f"epsilon_values = {epsilon_values}")
    print(f"q_scale_values = {q_scale_values}")
    print(f"meas_std = {np.asarray(meas_std, dtype=float)}")
    print(f"proc_std_base = {np.asarray(proc_std_base, dtype=float)}")

    for q_scale in q_scale_values:
        proc_std = (float(q_scale) * np.asarray(proc_std_base, dtype=float)).astype(np.float32)
        R, Q = cartpole_mod.build_filter_covariances(
            meas_std=meas_std,
            proc_std=proc_std,
            meas_corr=meas_corr,
            proc_corr=proc_corr,
        )

        summaries: dict[tuple[str, float], SweepSummary] = {
            (mode, float(epsilon)): SweepSummary()
            for mode in ("pgd", "random")
            for epsilon in epsilon_values
        }
        eps_real_values: list[float] = []
        inside_counts = {float(epsilon): 0 for epsilon in epsilon_values}
        total_points = 0

        for episode_idx in range(int(n_episodes)):
            seed = int(seed0 + episode_idx)
            env = gym.make("CartPole-v1")
            obs, _info = env.reset(seed=seed)

            rng_nominal_noise = np.random.default_rng(seed + 707_002)
            init_noise = rng_nominal_noise.normal(0.0, meas_std, size=(4,)).astype(np.float32)
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
                Q=Q,
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
                nominal_noise = rng_nominal_noise.normal(0.0, meas_std, size=(4,)).astype(np.float32)
                o_real = (y_clean + nominal_noise).astype(np.float32)
                attack_center = np.asarray(m_pred, dtype=np.float32)
                attack_sigma = cartpole_mod.project_to_psd(np.asarray(P_pred, dtype=float) + np.asarray(R, dtype=float))
                eps_real = achieved_attack_epsilon(
                    observation=o_real,
                    center=attack_center,
                    Sigma=attack_sigma,
                )

                eps_real_values.append(float(eps_real))
                total_points += 1
                for epsilon in epsilon_values:
                    inside_counts[float(epsilon)] += int(eps_real <= float(epsilon))

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
                    Q=Q,
                    discount_delta=discount_delta,
                    ssm=ssm,
                )
                obs, reward, terminated, truncated, _info = env.step(action)
                _ = reward
                step_idx += 1

                if terminated or truncated:
                    break

            env.close()

        mean_eps_real = float(np.mean(np.asarray(eps_real_values, dtype=float))) if eps_real_values else 0.0
        pct_inside_by_epsilon = {
            float(epsilon): 100.0 * float(inside_counts[float(epsilon)]) / max(total_points, 1)
            for epsilon in epsilon_values
        }
        print_scale_report(
            q_scale=float(q_scale),
            proc_std=proc_std,
            mean_eps_real=mean_eps_real,
            pct_inside_by_epsilon=pct_inside_by_epsilon,
            summaries=summaries,
            epsilon_values=epsilon_values,
        )


def main() -> None:
    """Entry point for the CartPole transition-covariance sweep."""
    run_q_sweep()


if __name__ == "__main__":
    main()
