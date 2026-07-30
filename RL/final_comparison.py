#!/usr/bin/env python3
"""
final_comparison.py

Run the final compact RL attack comparison.

Why this script exists:
1. The broad comparison script contains several exploratory baselines.
2. The final experiment requested here only compares:
   - noiseless state feedback,
   - noisy observations filtered online with KF,
   - the estimated-return PGD attack that starts from the noisy-KF state
     estimate,
   - the real-return PGD attack,
   - the random contour baseline.
3. Numerical results are saved in `RL/data/`, while `RL/figures/` contains only
   rendered figures.

The epsilon values are coverage probabilities. The third value is set to 0.95
instead of duplicating 0.75, because repeated x-axis groups would be redundant.
"""

from __future__ import annotations

from dataclasses import replace
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RL_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_MODULE_DIR)
if RL_MODULE_DIR not in sys.path:
    sys.path.insert(0, RL_MODULE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compare_wind_online_attack_rewards import attack_contour_summary
from compare_wind_online_attack_rewards import attack_records_to_arrays
from compare_wind_online_attack_rewards import coverage_to_mahalanobis_epsilon
from compare_wind_online_attack_rewards import evaluate_setting
from wind_rl_setup import default_device
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_data_dir
from wind_rl_setup import rl_figures_dir
from wind_rl_setup import rl_model_dir


FINAL_SETTING_NAMES = (
    "noiseless",
    "noisy + KF",
    "estimated-return PGD",
    "real-return PGD",
    "random contour",
)


def standard_error(values: np.ndarray) -> float:
    """Return the standard error, using zero for one-sample smoke tests."""
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def set_plot_theme() -> None:
    """Apply the compact pastel plotting style used in the RL figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10.4,
            "axes.labelsize": 11.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.70,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
            "savefig.facecolor": "white",
        }
    )


def plot_final_comparison_bars(
    *,
    coverages: tuple[float, ...],
    mean_returns: np.ndarray,
    sem_returns: np.ndarray,
    outpath: str,
) -> None:
    """Save the final grouped bar plot of mean accumulated rewards."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    colors = ["#8EC5B5", "#9DB7D5", "#F0C987", "#E6A57E", "#CFA7D8"]
    x_positions = np.arange(len(coverages), dtype=float)
    width = 0.15
    offsets = (np.arange(len(FINAL_SETTING_NAMES)) - 0.5 * (len(FINAL_SETTING_NAMES) - 1)) * width

    for setting_idx, (setting_name, color) in enumerate(zip(FINAL_SETTING_NAMES, colors)):
        ax.bar(
            x_positions + offsets[setting_idx],
            mean_returns[:, setting_idx],
            width=width,
            yerr=sem_returns[:, setting_idx],
            capsize=3.0,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=setting_name,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([rf"$\epsilon={coverage:.2f}$" for coverage in coverages])
    ax.set_ylabel("Mean accumulated reward")
    ax.set_xlabel("Ellipsoid coverage constraint")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.24)
    ax.grid(False, axis="x")
    ax.legend(loc="best", frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def run_final_comparison(
    *,
    n_episodes: int,
    coverages: tuple[float, ...],
    attack_prob: float,
    observation_noise_std: float,
    process_position_std: float,
    process_wind_std: float,
    contour_relative_tolerance: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    real_attack_transition_samples: int,
    heavy_tail_degrees_of_freedom: float,
    uniform_annulus_min_coverage: float,
    base_seed: int,
    verbose: bool,
    output_stem: str,
) -> dict[str, Any]:
    """Run the final comparison and save its figure and numerical data."""
    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, env_config, _net_config, train_config, _history = load_agent_checkpoint(model_path, device)
    policy.eval()
    gamma = float(train_config.gamma)

    simulation_config = replace(
        env_config,
        observation_noise_std=0.0,
        process_position_std=float(process_position_std),
        process_wind_std=float(process_wind_std),
    )
    filter_config = replace(
        simulation_config,
        observation_noise_std=float(observation_noise_std),
    )
    episode_seeds = int(base_seed) + np.arange(int(n_episodes), dtype=int)

    mean_returns = np.zeros((len(coverages), len(FINAL_SETTING_NAMES)), dtype=float)
    sem_returns = np.zeros_like(mean_returns)
    success_rates = np.zeros_like(mean_returns)
    mean_lengths = np.zeros_like(mean_returns)
    mean_attacks = np.zeros_like(mean_returns)
    attack_value_records: list[dict[str, float | int | str]] = []

    print(
        "Final comparison: "
        f"n_episodes={n_episodes}, "
        f"attack_prob={attack_prob:.3f}, "
        f"pgd_steps={attack_num_steps}, "
        f"posterior_mc={attack_mc_samples}, "
        f"real_transition_mc={real_attack_transition_samples}"
    )
    print(
        "Evaluation noise: "
        f"observation_std={observation_noise_std:.3f}, "
        f"process_position_std={process_position_std:.3f}, "
        f"process_wind_std={process_wind_std:.3f}"
    )

    for coverage_idx, coverage in enumerate(coverages):
        mahalanobis_epsilon = coverage_to_mahalanobis_epsilon(float(coverage), obs_dim=4)
        print(
            f"\ncoverage epsilon={coverage:.2f} "
            f"(Mahalanobis radius squared={mahalanobis_epsilon:.4f})"
        )
        for setting_idx, setting in enumerate(FINAL_SETTING_NAMES):
            result = evaluate_setting(
                policy=policy,
                simulation_config=simulation_config,
                filter_config=filter_config,
                setting=setting,
                coverage=float(coverage),
                episode_seeds=episode_seeds,
                observation_noise_std=float(observation_noise_std),
                gamma=float(gamma),
                attack_prob=float(attack_prob),
                attack_step_size=float(attack_step_size),
                attack_num_steps=int(attack_num_steps),
                attack_mc_samples=int(attack_mc_samples),
                real_attack_transition_samples=int(real_attack_transition_samples),
                heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
                uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
                contour_relative_tolerance=float(contour_relative_tolerance),
                verbose=bool(verbose),
                device=device,
            )
            attack_value_records.extend(result["attack_value_records"])

            returns = np.asarray(result["returns"], dtype=float)
            successes = np.asarray(result["successes"], dtype=float)
            lengths = np.asarray(result["lengths"], dtype=float)
            attacks = np.asarray(result["num_attacks"], dtype=float)

            mean_returns[coverage_idx, setting_idx] = float(np.mean(returns))
            sem_returns[coverage_idx, setting_idx] = standard_error(returns)
            success_rates[coverage_idx, setting_idx] = float(np.mean(successes))
            mean_lengths[coverage_idx, setting_idx] = float(np.mean(lengths))
            mean_attacks[coverage_idx, setting_idx] = float(np.mean(attacks))

            print(
                f"  {setting:20s} "
                f"mean_return={mean_returns[coverage_idx, setting_idx]:8.3f} "
                f"success={100.0 * success_rates[coverage_idx, setting_idx]:6.2f}% "
                f"mean_length={mean_lengths[coverage_idx, setting_idx]:6.2f} "
                f"mean_attacks={mean_attacks[coverage_idx, setting_idx]:5.2f}"
            )

            records = result["attack_value_records"]
            if records:
                contour_summary = attack_contour_summary(
                    records=records,
                    relative_tolerance=float(contour_relative_tolerance),
                )
                delta_values = np.asarray([record["delta_value"] for record in records], dtype=float)
                print(
                    f"  {'attack summary':20s} "
                    f"inside={100.0 * contour_summary['inside_rate']:6.2f}% "
                    f"on_contour={100.0 * contour_summary['on_contour_rate']:6.2f}% "
                    f"mean_deltaV={np.mean(delta_values):8.4f}"
                )

    figure_path = os.path.join(rl_figures_dir(), f"{output_stem}.png")
    plot_final_comparison_bars(
        coverages=coverages,
        mean_returns=mean_returns,
        sem_returns=sem_returns,
        outpath=figure_path,
    )

    data_path = os.path.join(rl_data_dir(), f"{output_stem}.npz")
    np.savez_compressed(
        data_path,
        coverages=np.asarray(coverages, dtype=float),
        setting_names=np.asarray(FINAL_SETTING_NAMES, dtype=object),
        mean_returns=mean_returns,
        sem_returns=sem_returns,
        success_rates=success_rates,
        mean_lengths=mean_lengths,
        mean_attacks=mean_attacks,
        n_episodes=np.asarray([n_episodes], dtype=int),
        attack_prob=np.asarray([attack_prob], dtype=float),
        attack_step_size=np.asarray([attack_step_size], dtype=float),
        attack_num_steps=np.asarray([attack_num_steps], dtype=int),
        attack_mc_samples=np.asarray([attack_mc_samples], dtype=int),
        real_attack_transition_samples=np.asarray([real_attack_transition_samples], dtype=int),
        gamma=np.asarray([gamma], dtype=float),
        observation_noise_std=np.asarray([observation_noise_std], dtype=float),
        process_position_std=np.asarray([process_position_std], dtype=float),
        process_wind_std=np.asarray([process_wind_std], dtype=float),
        contour_relative_tolerance=np.asarray([contour_relative_tolerance], dtype=float),
        **attack_records_to_arrays(attack_value_records),
    )

    print(f"\nSaved figure to: {figure_path}")
    print(f"Saved data to:   {data_path}")
    return {
        "figure_path": figure_path,
        "data_path": data_path,
        "mean_returns": mean_returns,
        "success_rates": success_rates,
    }


def main() -> None:
    """Configure and run the final 100-episode comparison."""
    n_episodes = 100
    coverages = (0.95, )
    attack_prob = 0.10
    observation_noise_std = 0.75
    process_position_std = 0.14
    process_wind_std = 0.10
    contour_relative_tolerance = 0.01
    attack_step_size = 0.05
    attack_num_steps = 120
    attack_mc_samples = 16
    real_attack_transition_samples = 8
    heavy_tail_degrees_of_freedom = 3.0
    uniform_annulus_min_coverage = 0.50
    base_seed = 20260727
    verbose = True
    output_stem = "final_comparison_mean_rewards_bar"

    run_final_comparison(
        n_episodes=int(n_episodes),
        coverages=coverages,
        attack_prob=float(attack_prob),
        observation_noise_std=float(observation_noise_std),
        process_position_std=float(process_position_std),
        process_wind_std=float(process_wind_std),
        contour_relative_tolerance=float(contour_relative_tolerance),
        attack_step_size=float(attack_step_size),
        attack_num_steps=int(attack_num_steps),
        attack_mc_samples=int(attack_mc_samples),
        real_attack_transition_samples=int(real_attack_transition_samples),
        heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
        uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
        base_seed=int(base_seed),
        verbose=bool(verbose),
        output_stem=output_stem,
    )


if __name__ == "__main__":
    main()
