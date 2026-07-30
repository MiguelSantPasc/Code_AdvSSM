#!/usr/bin/env python3
"""
sweep_cartpole_wolf_only.py

Sweep only the WoLF defense on the current CartPole attack scenario.

Why this script exists:
1. The full CartPole comparison now includes several defenses, but tuning WoLF
   does not require rerunning the whole benchmark every time.
2. This script isolates the two WoLF curves that matter for that tuning pass:
   - `Attack + WoLF`, where the adversary uses the PGD attack,
   - `Boundary epsilon-perturbation + WoLF`, meaning the benchmark's random
     epsilon-perturbation baseline under the same WoLF filter.
3. We sweep the actual WoLF hyperparameters:
   - IMQ: `imq_soft_threshold`,
   - TMD: `tmd_threshold`.

What this script reports:
1. The mean episode return for `Attack + WoLF`.
2. The mean episode return for the benchmark's random epsilon-perturbation
   baseline under WoLF.
3. For the attacked PGD branch, the same PGD convergence diagnostics already
   used elsewhere:
   - `pct_boundary`,
   - `attacked_used_pct`.
4. The mean WoLF weight and mean innovation norm across attacked updates, so it
   is easier to see whether one setting is aggressively downweighting the
   observation stream.

Important implementation convention:
1. The script reuses the exact CartPole model, predictor, attack construction,
   and WoLF update equations from `cartpole_covadapt_compare_epsilons_wolf.py`.
2. It also uses the current no-fallback scenario and the recalibrated discount
   factor, so results stay aligned with the latest CartPole experiment.
3. All tuning grids are plain Python variables inside `main()`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import cartpole_covadapt_compare_epsilons_wolf as cartpole_mod

from shared_ssm.artifacts import data_dir_for
from shared_ssm.artifacts import save_npz


@dataclass(frozen=True)
class WolfSweepConfig:
    """Describe one WoLF hyperparameter setting to evaluate."""

    label: str
    wolf_kind: str
    imq_soft_threshold: float
    tmd_threshold: float


@dataclass
class WolfModeSummary:
    """Accumulate rollout diagnostics for one WoLF setting and one attack mode."""

    return_sum: float = 0.0
    episode_count: int = 0
    attack_attempts: int = 0
    attacked_used: int = 0
    boundary_cases: int = 0
    boundary_hits: int = 0
    gap_sum: float = 0.0
    wolf_update_count: int = 0
    weight_sq_sum: float = 0.0
    innovation_norm_sum: float = 0.0


def init_wolf_summary() -> WolfModeSummary:
    """Create one empty WoLF sweep accumulator."""
    return WolfModeSummary()


def summarize_wolf_summary(summary: WolfModeSummary) -> dict[str, float | int]:
    """Convert one raw WoLF accumulator into report-ready scalar metrics."""
    episode_denom = max(int(summary.episode_count), 1)
    boundary_denom = max(int(summary.boundary_cases), 1)
    attack_denom = max(int(summary.attack_attempts), 1)
    wolf_update_denom = max(int(summary.wolf_update_count), 1)
    return {
        "episodes": int(summary.episode_count),
        "mean_return": float(summary.return_sum) / float(episode_denom),
        "attack_attempts": int(summary.attack_attempts),
        "pct_boundary": 100.0 * float(summary.boundary_hits) / float(boundary_denom),
        "attacked_used_pct": 100.0 * float(summary.attacked_used) / float(attack_denom),
        "mean_gap": float(summary.gap_sum) / float(boundary_denom),
        "mean_weight_sq": float(summary.weight_sq_sum) / float(wolf_update_denom),
        "mean_innovation_norm": float(summary.innovation_norm_sum) / float(wolf_update_denom),
    }


def rollout_episode_return_wolf_sweep(
    env: gym.Env,
    model: cartpole_mod.DQN,
    *,
    seed: int,
    ssm: cartpole_mod.CartPoleLinearSSM,
    R: np.ndarray,
    discount_delta: float,
    obs_noise_std: np.ndarray,
    attack_eps: float,
    attack_prob: float,
    attack_mode: str,
    wolf_kind: str,
    imq_soft_threshold: float,
    tmd_threshold: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    pgd_boundary_tol: float,
    device: str,
) -> tuple[float, WolfModeSummary]:
    """
    Run one CartPole episode under WoLF and collect attack-only diagnostics.

    This mirrors the current benchmark logic, but it keeps only the WoLF branch
    and records the effective WoLF observation weights on attacked steps.
    """
    obs, _info = env.reset(seed=int(seed))
    summary = init_wolf_summary()

    rng_attack_gate = np.random.default_rng(int(seed) + 707_001)
    rng_nominal_noise = np.random.default_rng(int(seed) + 707_002)
    rng_boundary = np.random.default_rng(int(seed) + 707_003)
    init_noise = rng_nominal_noise.normal(0.0, obs_noise_std, size=(4,)).astype(np.float32)
    obs_init_noisy = (np.asarray(obs, dtype=np.float32) + init_noise).astype(np.float32)

    m_post, P_post = cartpole_mod.kf_update_state(
        m_pred=np.asarray(obs, dtype=np.float32),
        P_pred=cartpole_mod.project_to_psd(R.copy()),
        y_obs=obs_init_noisy,
        R=R,
    )

    ep_return = 0.0
    step_idx = 0

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
    ep_return += float(reward)
    if terminated or truncated:
        summary.return_sum += float(ep_return)
        summary.episode_count += 1
        return float(ep_return), summary

    step_idx = 1

    while True:
        y_clean = np.asarray(obs, dtype=np.float32)
        do_attack = bool(rng_attack_gate.random() < float(attack_prob))
        nominal_noise = rng_nominal_noise.normal(0.0, obs_noise_std, size=(4,)).astype(np.float32)
        y_noisy = (y_clean + nominal_noise).astype(np.float32)
        attack_center = np.asarray(m_pred, dtype=np.float32)
        attack_sigma = cartpole_mod.project_to_psd(np.asarray(P_pred, dtype=float) + np.asarray(R, dtype=float))

        if do_attack:
            summary.attack_attempts += 1
            if attack_mode == "pgd":
                y_used, _obj_star, _m_post_attack, _P_post_attack = cartpole_mod.pgd_attack_on_expected_value(
                    model=model,
                    obs_nom=y_noisy,
                    m_pred=m_pred,
                    P_pred=P_pred,
                    R=R,
                    attack_center=attack_center,
                    attack_sigma=attack_sigma,
                    attack_eps=attack_eps,
                    pgd_steps=pgd_steps,
                    pgd_step_size=pgd_step_size,
                    mc_samples=mc_samples,
                    rng_seed=int(seed) + 10_000 * step_idx,
                    device=device,
                )
                eps_attack = cartpole_mod.mahalanobis_radius_sq(
                    observation=y_used,
                    center=attack_center,
                    covariance=attack_sigma,
                )
                gap = max(float(attack_eps) - float(eps_attack), 0.0)
                summary.boundary_cases += 1
                summary.gap_sum += float(gap)
                summary.boundary_hits += int(gap <= float(pgd_boundary_tol))
            elif attack_mode == "random":
                y_used = cartpole_mod.sample_random_attack_in_ellipsoid(
                    center=attack_center,
                    Sigma=attack_sigma,
                    epsilon=attack_eps,
                    rng=rng_boundary,
                )
            else:
                raise ValueError(f"Unsupported attack_mode: {attack_mode}")

            # The current CartPole scenario disables the fallback-to-real rule.
            summary.attacked_used += 1
            y_filter = y_used
        else:
            y_filter = y_noisy

        m_post, P_post, wolf_diag = cartpole_mod.wolf_kf_update_state(
            m_pred=m_pred,
            P_pred=P_pred,
            y_obs=y_filter,
            R=R,
            wolf_kind=wolf_kind,
            imq_soft_threshold=imq_soft_threshold,
            tmd_threshold=tmd_threshold,
        )

        if do_attack:
            summary.wolf_update_count += 1
            summary.weight_sq_sum += float(wolf_diag["weight_sq"])
            summary.innovation_norm_sum += float(wolf_diag["innovation_norm"])

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
        ep_return += float(reward)
        step_idx += 1

        if terminated or truncated:
            break

    summary.return_sum += float(ep_return)
    summary.episode_count += 1
    return float(ep_return), summary


def merge_wolf_summary(accumulator: WolfModeSummary, update: WolfModeSummary) -> None:
    """Add one episode-level WoLF summary into a running configuration summary."""
    accumulator.return_sum += float(update.return_sum)
    accumulator.episode_count += int(update.episode_count)
    accumulator.attack_attempts += int(update.attack_attempts)
    accumulator.attacked_used += int(update.attacked_used)
    accumulator.boundary_cases += int(update.boundary_cases)
    accumulator.boundary_hits += int(update.boundary_hits)
    accumulator.gap_sum += float(update.gap_sum)
    accumulator.wolf_update_count += int(update.wolf_update_count)
    accumulator.weight_sq_sum += float(update.weight_sq_sum)
    accumulator.innovation_norm_sum += float(update.innovation_norm_sum)


def print_wolf_sweep_report(
    *,
    attack_eps: float,
    rows: list[dict[str, float | int | str]],
) -> None:
    """Print a compact WoLF tuning table for one epsilon and one attack mode."""
    print(f"[eps={attack_eps:.2f}] WoLF-only sweep")
    print("  config                  attack_ret   eps_ret   mean_weight_sq")
    for row in rows:
        print(
            f"  {str(row['config']):<22} "
            f"{float(row['attack_mean_return']):>10.3f}   "
            f"{float(row['random_mean_return']):>7.3f}   "
            f"{float(row['mean_weight_sq']):>14.4f}"
        )
    print()


def save_wolf_sweep_summary(
    *,
    outpath: str,
    scenario_tag: str,
    attack_eps_values: tuple[float, ...],
    n_episodes: int,
    seed0: int,
    attack_prob: float,
    pgd_steps: int,
    pgd_step_size: float,
    pgd_boundary_tol: float,
    mc_samples: int,
    rows_by_epsilon: dict[float, list[dict[str, float | int | str]]],
) -> None:
    """Save the WoLF-only sweep as a compressed NPZ file."""
    payload: dict[str, object] = {
        "scenario_tag": np.asarray(scenario_tag),
        "attack_eps_values": np.asarray(attack_eps_values, dtype=float),
        "n_episodes": int(n_episodes),
        "seed0": int(seed0),
        "attack_prob": float(attack_prob),
        "pgd_steps": int(pgd_steps),
        "pgd_step_size": float(pgd_step_size),
        "pgd_boundary_tol": float(pgd_boundary_tol),
        "mc_samples": int(mc_samples),
    }

    for attack_eps, rows in rows_by_epsilon.items():
        eps_tag = str(float(attack_eps)).replace(".", "p")
        payload[f"config_labels_eps{eps_tag}"] = np.asarray([str(row["config"]) for row in rows])
        payload[f"kinds_eps{eps_tag}"] = np.asarray([str(row["kind"]) for row in rows])
        payload[f"mode_attack_mean_return_eps{eps_tag}"] = np.asarray(
            [float(row["attack_mean_return"]) for row in rows],
            dtype=float,
        )
        payload[f"mode_random_mean_return_eps{eps_tag}"] = np.asarray(
            [float(row["random_mean_return"]) for row in rows],
            dtype=float,
        )
        payload[f"pgd_pct_boundary_eps{eps_tag}"] = np.asarray(
            [float(row["pct_boundary"]) for row in rows],
            dtype=float,
        )
        payload[f"pgd_attacked_used_pct_eps{eps_tag}"] = np.asarray(
            [float(row["attacked_used_pct"]) for row in rows],
            dtype=float,
        )
        payload[f"pgd_mean_gap_eps{eps_tag}"] = np.asarray(
            [float(row["mean_gap"]) for row in rows],
            dtype=float,
        )
        payload[f"attack_mean_weight_sq_eps{eps_tag}"] = np.asarray(
            [float(row["mean_weight_sq"]) for row in rows],
            dtype=float,
        )
        payload[f"attack_mean_innovation_norm_eps{eps_tag}"] = np.asarray(
            [float(row["mean_innovation_norm"]) for row in rows],
            dtype=float,
        )

    save_npz(outpath, **payload)
    print(f"Saved WoLF sweep summary to: {outpath}")


def main() -> None:
    """Entry point for the standalone WoLF-only CartPole sweep."""
    scenario_tag = "obs010-022-005-022_nofallback"
    device = "cpu"
    model_path = cartpole_mod.ensure_downloaded_cartpole_checkpoint()

    # Reuse the current CartPole attack scenario so the WoLF sweep stays
    # directly comparable to the latest benchmark outputs.
    obs_noise_std = np.array([0.10, 0.22, 0.05, 0.22], dtype=float)
    attack_prob = 0.20
    attack_eps_values = (5.39, 9.49)
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

    # Keep the PGD optimizer exactly aligned with the current benchmark.
    pgd_steps = 20
    pgd_step_size = 0.34
    mc_samples = 64
    pgd_boundary_tol = 0.1

    # Use a slightly larger batch than the coarse sweep so the ranking is less
    # noisy, while still staying much cheaper than the full benchmark.
    n_episodes = 12
    seed0 = 100

    # Fine sweep around the best regions found by the first coarse scan:
    # IMQ looked strongest near 0.45, while TMD looked strongest near 2.8.
    imq_soft_threshold_values = (0.35, 0.40, 0.45, 0.50, 0.60)
    tmd_threshold_values = (2.6, 2.7, 2.8, 2.9, 3.0)

    print(f"scenario_tag = {scenario_tag}")
    print(f"n_episodes = {n_episodes}")
    print(f"seed0 = {seed0}")
    print(f"attack_prob = {attack_prob}")
    print(f"pgd_steps = {pgd_steps}")
    print(f"pgd_step_size = {pgd_step_size}")
    print(f"discount_delta = {discount_delta}")

    sweep_configs: list[WolfSweepConfig] = []
    sweep_configs.extend(
        [
            WolfSweepConfig(
                label=f"imq_tau{str(value).replace('.', 'p')}",
                wolf_kind="imq",
                imq_soft_threshold=float(value),
                tmd_threshold=2.8,
            )
            for value in imq_soft_threshold_values
        ]
    )
    sweep_configs.extend(
        [
            WolfSweepConfig(
                label=f"tmd_tau{str(value).replace('.', 'p')}",
                wolf_kind="tmd",
                imq_soft_threshold=0.20,
                tmd_threshold=float(value),
            )
            for value in tmd_threshold_values
        ]
    )

    model = cartpole_mod.load_cartpole_policy(model_path, torch.device(device))
    ssm = cartpole_mod.build_cartpole_linear_ssm()
    R, _legacy_Q = cartpole_mod.build_filter_covariances(
        meas_std=kf_meas_std,
        proc_std=kf_proc_std,
        meas_corr=kf_meas_corr,
        proc_corr=kf_proc_corr,
    )

    rows_by_epsilon: dict[float, list[dict[str, float | int | str]]] = {}

    for attack_eps in attack_eps_values:
        attack_rows: list[dict[str, float | int | str]] = []

        for config in sweep_configs:
            attack_summary = init_wolf_summary()
            random_summary = init_wolf_summary()

            for episode_idx in range(int(n_episodes)):
                seed = int(seed0 + episode_idx)
                env_attack = gym.make("CartPole-v1")
                env_random = gym.make("CartPole-v1")
                try:
                    _ret_attack, attack_update = rollout_episode_return_wolf_sweep(
                        env_attack,
                        model,
                        seed=seed,
                        ssm=ssm,
                        R=R,
                        discount_delta=discount_delta,
                        obs_noise_std=obs_noise_std,
                        attack_eps=float(attack_eps),
                        attack_prob=attack_prob,
                        attack_mode="pgd",
                        wolf_kind=config.wolf_kind,
                        imq_soft_threshold=config.imq_soft_threshold,
                        tmd_threshold=config.tmd_threshold,
                        pgd_steps=pgd_steps,
                        pgd_step_size=pgd_step_size,
                        mc_samples=mc_samples,
                        pgd_boundary_tol=pgd_boundary_tol,
                        device=device,
                    )
                    _ret_random, random_update = rollout_episode_return_wolf_sweep(
                        env_random,
                        model,
                        seed=seed,
                        ssm=ssm,
                        R=R,
                        discount_delta=discount_delta,
                        obs_noise_std=obs_noise_std,
                        attack_eps=float(attack_eps),
                        attack_prob=attack_prob,
                        attack_mode="random",
                        wolf_kind=config.wolf_kind,
                        imq_soft_threshold=config.imq_soft_threshold,
                        tmd_threshold=config.tmd_threshold,
                        pgd_steps=pgd_steps,
                        pgd_step_size=pgd_step_size,
                        mc_samples=mc_samples,
                        pgd_boundary_tol=pgd_boundary_tol,
                        device=device,
                    )
                finally:
                    env_attack.close()
                    env_random.close()

                merge_wolf_summary(attack_summary, attack_update)
                merge_wolf_summary(random_summary, random_update)

            attack_metrics = summarize_wolf_summary(attack_summary)
            random_metrics = summarize_wolf_summary(random_summary)
            attack_rows.append(
                {
                    "config": config.label,
                    "kind": config.wolf_kind,
                    "attack_mean_return": float(attack_metrics["mean_return"]),
                    "random_mean_return": float(random_metrics["mean_return"]),
                    "mean_return": float(attack_metrics["mean_return"]),
                    "pct_boundary": float(attack_metrics["pct_boundary"]),
                    "attacked_used_pct": float(attack_metrics["attacked_used_pct"]),
                    "mean_gap": float(attack_metrics["mean_gap"]),
                    "mean_weight_sq": float(attack_metrics["mean_weight_sq"]),
                    "mean_innovation_norm": float(attack_metrics["mean_innovation_norm"]),
                }
            )

            print(
                f"[eps={attack_eps:.2f}] finished {config.label} "
                f"| attack_mean_return={float(attack_metrics['mean_return']):.3f} "
                f"| random_mean_return={float(random_metrics['mean_return']):.3f}"
            )

        rows_by_epsilon[float(attack_eps)] = attack_rows
        print_wolf_sweep_report(
            attack_eps=float(attack_eps),
            rows=attack_rows,
        )

    out_dir = data_dir_for(CURRENT_DIR)
    outpath = os.path.join(
        out_dir,
        (
            "sweep_cartpole_wolf_only_"
            f"{scenario_tag}_delta{str(discount_delta).replace('.', 'p')}_"
            f"N{n_episodes}_eps{'-'.join(str(value).replace('.', 'p') for value in attack_eps_values)}.npz"
        ),
    )
    save_wolf_sweep_summary(
        outpath=outpath,
        scenario_tag=scenario_tag,
        attack_eps_values=attack_eps_values,
        n_episodes=n_episodes,
        seed0=seed0,
        attack_prob=attack_prob,
        pgd_steps=pgd_steps,
        pgd_step_size=pgd_step_size,
        pgd_boundary_tol=pgd_boundary_tol,
        mc_samples=mc_samples,
        rows_by_epsilon=rows_by_epsilon,
    )


if __name__ == "__main__":
    main()
