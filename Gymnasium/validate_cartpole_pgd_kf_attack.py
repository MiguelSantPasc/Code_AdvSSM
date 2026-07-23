#!/usr/bin/env python3
"""
validate_cartpole_pgd_kf_attack.py

Validate only the CartPole PGD attack under the plain KF defense.

Why this script exists:
1. Sometimes we do not want the full CartPole benchmark or the comparison
   across defenses. We only want to sanity-check the PGD attack itself.
2. The two attack diagnostics that matter most here are:
   - how often the PGD iterate stays near the ellipsoid boundary,
   - how often the attacked observation is actually used instead of falling
     back to the nominal noisy observation.
3. This script isolates exactly that check for the two 4D CartPole radii used
   in the benchmark:
      - `epsilon = 5.39`  (approximately 75% coverage),
      - `epsilon = 9.49`  (approximately 95% coverage).

What this script reports:
1. `cases`: number of PGD attack attempts that were actually measured.
2. `pct_boundary`: percentage of those attacks whose slack
      `epsilon - epsilon_adv`
   is at most `boundary_tol`.
3. `attacked_used_pct`: percentage of those attacks where the final attacked
   observation was kept instead of being replaced by the nominal noisy
   observation because the attack was weaker.

Implementation note:
1. The script reuses the exact attack and KF rollout code from
   `cartpole_covadapt_compare_epsilons_wolf.py`.
2. This keeps the validation aligned with the main CartPole setup while
   avoiding the extra WoLF and covariance-adaptation runs.
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import cartpole_covadapt_compare_epsilons_wolf as cartpole_mod

try:
    from AdvSSM.io_utils import data_dir_for, save_npz
except ModuleNotFoundError:
    from io_utils import data_dir_for, save_npz


def summarize_kf_attack_stats(stats: dict[str, int | float], *, boundary_tol: float) -> dict[str, float | int]:
    """Convert one raw PGD diagnostic accumulator into the final KF summary."""
    cases = int(stats["pgd_diagnostic_cases"])
    case_denom = max(cases, 1)
    attack_attempts = max(int(stats["attack_attempts"]), 1)
    attacked_used = float(stats["attack_attempts"]) - float(stats["fallback_to_real"])
    return {
        "cases": cases,
        "mean_gap": float(stats["pgd_gap_sum"]) / float(case_denom),
        "pct_boundary": 100.0 * float(stats["pgd_on_boundary"]) / float(case_denom),
        "attacked_used_pct": 100.0 * attacked_used / float(attack_attempts),
        "boundary_tol": float(boundary_tol),
    }


def print_kf_attack_report(
    *,
    summaries_by_epsilon: dict[float, dict[str, float | int]],
) -> None:
    """Print the standalone KF-only PGD validation table."""
    print("CartPole PGD attack validation under KF")
    print("  cases = number of PGD attack attempts measured.")
    print("  epsilon   coverage   cases   pct_boundary   attacked_used_pct   mean_gap")
    for attack_eps, summary in summaries_by_epsilon.items():
        coverage_label = "0.75" if np.isclose(float(attack_eps), 5.39) else "0.95"
        print(
            f"  {float(attack_eps):>7.2f}   {coverage_label:>8s}   "
            f"{int(summary['cases']):>5d}   {float(summary['pct_boundary']):>11.2f}%   "
            f"{float(summary['attacked_used_pct']):>16.2f}%   {float(summary['mean_gap']):>8.4f}"
        )


def run_kf_attack_validation_for_epsilon(
    *,
    attack_eps: float,
    n_episodes: int,
    seed0: int,
    attack_prob: float,
    model_path: str,
    obs_noise_std: np.ndarray,
    discount_delta: float,
    kf_meas_std: np.ndarray,
    kf_proc_std: np.ndarray,
    kf_meas_corr: np.ndarray,
    kf_proc_corr: np.ndarray,
    pgd_steps: int,
    pgd_step_size: float,
    pgd_boundary_tol: float,
    mc_samples: int,
    device: str,
) -> dict[str, float | int]:
    """Run the standalone KF-only PGD validation for one attack epsilon."""
    model = cartpole_mod.load_cartpole_policy(model_path, torch.device(device))
    ssm = cartpole_mod.build_cartpole_linear_ssm()
    R, _legacy_Q = cartpole_mod.build_filter_covariances(
        meas_std=kf_meas_std,
        proc_std=kf_proc_std,
        meas_corr=kf_meas_corr,
        proc_corr=kf_proc_corr,
    )

    stats = cartpole_mod.make_pgd_boundary_stats()

    for episode_idx in range(int(n_episodes)):
        seed = int(seed0 + episode_idx)
        env = gym.make("CartPole-v1")
        try:
            _ret_attack, episode_stats = cartpole_mod.rollout_episode_return_attacked(
                env,
                model,
                seed=seed,
                ssm=ssm,
                R=R,
                discount_delta=discount_delta,
                obs_noise_std=np.asarray(obs_noise_std, dtype=float),
                attack_eps=float(attack_eps),
                attack_prob=float(attack_prob),
                attack_mode="pgd",
                defense="kf",
                pgd_steps=int(pgd_steps),
                pgd_step_size=float(pgd_step_size),
                mc_samples=int(mc_samples),
                c_scale=0.0,
                omega_h=0.50,
                omega_o=0.50,
                delta_threshold=0.20,
                wolf_kind="imq",
                wolf_imq_soft_threshold=0.20,
                wolf_tmd_threshold=2.8,
                pgd_boundary_tol=float(pgd_boundary_tol),
                device=device,
            )
        finally:
            env.close()

        cartpole_mod.merge_pgd_boundary_stats(stats, episode_stats)
        print(f"[eps={attack_eps:.2f}] run {episode_idx + 1:4d}/{n_episodes}")

    return summarize_kf_attack_stats(stats, boundary_tol=pgd_boundary_tol)


def save_kf_attack_summary(
    *,
    outpath: str,
    attack_eps_values: tuple[float, ...],
    n_episodes: int,
    seed0: int,
    attack_prob: float,
    pgd_steps: int,
    pgd_step_size: float,
    pgd_boundary_tol: float,
    mc_samples: int,
    summaries_by_epsilon: dict[float, dict[str, float | int]],
) -> None:
    """Save the KF-only attack validation summary as a compressed NPZ file."""
    payload: dict[str, object] = {
        "attack_eps_values": np.asarray(attack_eps_values, dtype=float),
        "n_episodes": int(n_episodes),
        "seed0": int(seed0),
        "attack_prob": float(attack_prob),
        "pgd_steps": int(pgd_steps),
        "pgd_step_size": float(pgd_step_size),
        "pgd_boundary_tol": float(pgd_boundary_tol),
        "mc_samples": int(mc_samples),
    }

    for attack_eps, summary in summaries_by_epsilon.items():
        eps_tag = str(float(attack_eps)).replace(".", "p")
        payload[f"cases_eps{eps_tag}"] = int(summary["cases"])
        payload[f"mean_gap_eps{eps_tag}"] = float(summary["mean_gap"])
        payload[f"pct_boundary_eps{eps_tag}"] = float(summary["pct_boundary"])
        payload[f"attacked_used_pct_eps{eps_tag}"] = float(summary["attacked_used_pct"])

    save_npz(outpath, **payload)
    print(f"Saved KF attack summary to: {outpath}")


def main() -> None:
    """Entry point for the standalone KF-only CartPole PGD validation."""
    # Attack radii used in the main CartPole benchmark.
    attack_eps_values = (9.49,)

    # Use a larger batch than the main reward benchmark to stabilize the two
    # PGD percentages we care about.    
    n_episodes = 15
    seed0 = 100
    attack_prob = 0.20

    # Shared observation and predictor settings.
    # Slightly stronger observation noise than the main benchmark so the
    # standalone KF attack validation is a bit more demanding.
    obs_noise_std = np.array([0.10, 0.22, 0.05, 0.22], dtype=float)
    discount_delta = cartpole_mod.DEFAULT_GYMNASIUM_DISCOUNT_DELTA
    kf_meas_std = obs_noise_std.copy()
    kf_proc_std = np.array([0.320, 0.720, 0.160, 0.720], dtype=float)
    kf_meas_corr = np.array(
        [
            [1.00, 0.18, 0.06, 0.00],
            [0.18, 1.00, 0.14, 0.22],
            [0.06, 0.14, 1.00, 0.18],
            [0.00, 0.22, 0.18, 1.00],
        ],
        dtype=float,
    )
    kf_proc_corr = np.array(
        [
            [1.00, 0.24, 0.08, 0.00],
            [0.24, 1.00, 0.16, 0.26],
            [0.08, 0.16, 1.00, 0.22],
            [0.00, 0.26, 0.22, 1.00],
        ],
        dtype=float,
    )

    # PGD settings aligned with the main CartPole benchmark.
    pgd_steps = 20
    pgd_step_size = 0.34
    pgd_boundary_tol = 0.1
    mc_samples = 64
    device = "cpu"

    print(f"n_episodes = {n_episodes}")
    print(f"seed0 = {seed0}")
    print(f"attack_prob = {attack_prob}")
    print(f"pgd_steps = {pgd_steps}")
    print(f"pgd_step_size = {pgd_step_size}")
    print(f"pgd_boundary_tol = {pgd_boundary_tol}")
    print(f"mc_samples = {mc_samples}")

    model_path = cartpole_mod.ensure_downloaded_cartpole_checkpoint()
    summaries_by_epsilon: dict[float, dict[str, float | int]] = {}

    for attack_eps in attack_eps_values:
        summaries_by_epsilon[float(attack_eps)] = run_kf_attack_validation_for_epsilon(
            attack_eps=float(attack_eps),
            n_episodes=n_episodes,
            seed0=seed0,
            attack_prob=attack_prob,
            model_path=model_path,
            obs_noise_std=obs_noise_std,
            discount_delta=discount_delta,
            kf_meas_std=kf_meas_std,
            kf_proc_std=kf_proc_std,
            kf_meas_corr=kf_meas_corr,
            kf_proc_corr=kf_proc_corr,
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            pgd_boundary_tol=pgd_boundary_tol,
            mc_samples=mc_samples,
            device=device,
        )

    print()
    print_kf_attack_report(summaries_by_epsilon=summaries_by_epsilon)

    out_dir = data_dir_for(CURRENT_DIR)
    outpath = os.path.join(
        out_dir,
        (
            "validate_cartpole_pgd_kf_attack_"
            f"N{n_episodes}_eps{'-'.join(str(value).replace('.', 'p') for value in attack_eps_values)}.npz"
        ),
    )
    save_kf_attack_summary(
        outpath=outpath,
        attack_eps_values=attack_eps_values,
        n_episodes=n_episodes,
        seed0=seed0,
        attack_prob=attack_prob,
        pgd_steps=pgd_steps,
        pgd_step_size=pgd_step_size,
        pgd_boundary_tol=pgd_boundary_tol,
        mc_samples=mc_samples,
        summaries_by_epsilon=summaries_by_epsilon,
    )


if __name__ == "__main__":
    main()
