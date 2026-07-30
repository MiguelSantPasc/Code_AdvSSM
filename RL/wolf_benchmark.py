#!/usr/bin/env python3
"""
wolf_benchmark.py

Standalone WoLF hyperparameter benchmark for the wind-defense experiments.

Why this module exists:
1. WoLF tuning should be runnable on its own, with its own visible
   configuration, instead of being hidden behind another benchmark script.
2. The generated artifacts must be case-specific so multiple epsilon/covariance
   runs can coexist without overwriting one another.
3. The selected parameters are saved in one JSON file that the final defense
   benchmark can read later, but this script does not call the other benchmark
   automatically.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_core import DefenseBenchmarkConfig
from benchmark_core import build_episode_sets
from benchmark_core import run_wolf_sweep
from benchmark_core import select_best_wolf_parameters
from benchmark_core import set_plot_theme
from compare_wind_online_attack_rewards import coverage_to_mahalanobis_epsilon
from covaraicne_sweep import observation_std_from_ratio
from wind_rl_setup import default_device
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_data_dir
from wind_rl_setup import rl_figures_dir
from wind_rl_setup import rl_model_dir


# Standalone WoLF benchmark configuration.
BENCHMARK_COVERAGE = 0.95
TRANSITION_COVARIANCE_MULTIPLIER = 2.0
OBSERVATION_TO_TRANSITION_RATIO = 1.0
ATTACK_PROBABILITY = 0.10
BASE_SEED = 20260727
N_TUNING_EPISODES = 6
N_EPISODES = 100
ATTACK_STEP_SIZE = 0.05
ATTACK_NUM_STEPS = 120
ATTACK_MC_SAMPLES = 16
TRANSITION_MC_SAMPLES = 8
GAMMA_THRESHOLD = 0.001
GOAL_RISK_SCALE_SOURCE = "goal_distance_min"
OMEGA_RISK = 0.5
OMEGA_OBSERVATION = 0.5
STRONG_WOLF_WEIGHT_THRESHOLD = 0.25
CONTOUR_RELATIVE_TOLERANCE = 0.01
DIRECTION_EPS = 1e-10
BASE_PROCESS_POSITION_STD = 0.14
BASE_PROCESS_WIND_STD = 0.10


def float_tag(value: float, *, digits: int = 3) -> str:
    """Return a filename-safe tag for one float."""
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def benchmark_case_tag(*, coverage: float) -> str:
    """Return the case tag appended to output files for this script."""
    epsilon_value = float(coverage_to_mahalanobis_epsilon(float(coverage), obs_dim=4))
    epsilon_tag = float_tag(epsilon_value, digits=3)
    ratio_tag = float_tag(OBSERVATION_TO_TRANSITION_RATIO, digits=2)
    transition_tag = float_tag(TRANSITION_COVARIANCE_MULTIPLIER, digits=2)
    return f"eps{epsilon_tag}_VtoW{ratio_tag}_{transition_tag}Wbase"


def default_output_prefix(*, coverage: float, n_tuning_episodes: int, n_episodes: int) -> str:
    """Return the default output prefix for this standalone benchmark."""
    return (
        f"wolf_benchmark_{benchmark_case_tag(coverage=coverage)}"
        f"_ntune{int(n_tuning_episodes)}_n{int(n_episodes)}"
    )


def best_wolf_json_path(*, coverage: float, n_tuning_episodes: int, n_episodes: int) -> str:
    """Return the case-specific JSON path with the selected WoLF parameters."""
    filename = (
        f"best_wolf_params_{benchmark_case_tag(coverage=coverage)}"
        f"_ntune{int(n_tuning_episodes)}_n{int(n_episodes)}.json"
    )
    return os.path.join(rl_data_dir(), filename)


def build_wolf_benchmark_config() -> DefenseBenchmarkConfig:
    """Build the standalone WoLF tuning configuration with local script parameters."""
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    device = default_device()
    _policy, env_config, _net_config, train_config, _history = load_agent_checkpoint(model_path, device)
    process_std_multiplier = float(np.sqrt(TRANSITION_COVARIANCE_MULTIPLIER))
    process_position_std = float(BASE_PROCESS_POSITION_STD * process_std_multiplier)
    process_wind_std = float(BASE_PROCESS_WIND_STD * process_std_multiplier)
    observation_noise_std = observation_std_from_ratio(
        covariance_ratio_value=OBSERVATION_TO_TRANSITION_RATIO,
        process_position_std=process_position_std,
        process_wind_std=process_wind_std,
    )
    filter_config = replace(
        env_config,
        observation_noise_std=observation_noise_std,
        process_position_std=process_position_std,
        process_wind_std=process_wind_std,
    )
    simulation_config = replace(
        filter_config,
        observation_noise_std=0.0,
    )
    return DefenseBenchmarkConfig(
        coverage=float(BENCHMARK_COVERAGE),
        attack_probability=float(ATTACK_PROBABILITY),
        lambdas=(0.5, 2.0, 8.0),
        gamma_threshold=float(GAMMA_THRESHOLD),
        goal_risk_scale=float(getattr(filter_config, GOAL_RISK_SCALE_SOURCE)),
        omega_risk=float(OMEGA_RISK),
        omega_observation=float(OMEGA_OBSERVATION),
        strong_wolf_weight_threshold=float(STRONG_WOLF_WEIGHT_THRESHOLD),
        attack_step_size=float(ATTACK_STEP_SIZE),
        attack_num_steps=int(ATTACK_NUM_STEPS),
        attack_mc_samples=int(ATTACK_MC_SAMPLES),
        transition_mc_samples=int(TRANSITION_MC_SAMPLES),
        attack_discount_gamma=float(train_config.gamma),
        contour_relative_tolerance=float(CONTOUR_RELATIVE_TOLERANCE),
        direction_eps=float(DIRECTION_EPS),
        base_seed=int(BASE_SEED),
        n_tuning_episodes=int(N_TUNING_EPISODES),
        n_episodes=int(N_EPISODES),
        filter_config=filter_config,
        simulation_config=simulation_config,
    )


def plot_wolf_benchmark_results(*, full_df: pd.DataFrame, output_path: str) -> None:
    """Save the WoLF tuning comparison figure."""
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    pivot_df = full_df.pivot(index="config_label", columns="scenario", values="mean_return").fillna(np.nan)
    pivot_df = pivot_df.sort_index()
    x_positions = np.arange(len(pivot_df), dtype=float)
    width = 0.24
    scenario_order = ["no_attack", "pgd_estimated", "random_contour"]
    scenario_colors = {
        "no_attack": "#9DB7D5",
        "pgd_estimated": "#E6A57E",
        "random_contour": "#AFC7E8",
    }
    for idx, scenario in enumerate(scenario_order):
        values = pivot_df[scenario].to_numpy(dtype=float)
        ax.bar(
            x_positions + (idx - 1) * width,
            values,
            width=width,
            color=scenario_colors[scenario],
            edgecolor="white",
            linewidth=0.8,
            label=scenario.replace("_", " "),
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot_df.index.tolist(), rotation=35, ha="right")
    ax.set_ylabel("Mean accumulated reward")
    ax.set_xlabel("WoLF configuration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_wolf_benchmark_artifacts(
    *,
    config: DefenseBenchmarkConfig,
    full_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    selected_parameters: dict[str, dict[str, Any]],
    output_prefix: str,
) -> dict[str, str]:
    """Save the WoLF tuning tables, figure, and case-specific JSON selection."""
    data_dir = rl_data_dir()
    figures_dir = rl_figures_dir()
    full_csv_path = os.path.join(data_dir, f"{output_prefix}_full.csv")
    ranking_csv_path = os.path.join(data_dir, f"{output_prefix}_ranking.csv")
    figure_path = os.path.join(figures_dir, f"{output_prefix}.png")
    json_path = best_wolf_json_path(
        coverage=float(config.coverage),
        n_tuning_episodes=int(config.n_tuning_episodes),
        n_episodes=int(config.n_episodes),
    )

    full_df.to_csv(full_csv_path, index=False)
    ranking_df.to_csv(ranking_csv_path, index=False)
    plot_wolf_benchmark_results(full_df=full_df, output_path=figure_path)

    payload = {
        "wolf_imq": selected_parameters["wolf_imq"],
        "wolf_tmd": selected_parameters["wolf_tmd"],
        "selection_metric": "maximize attacked mean return; tie-break by lower state error, better noisy return, lower return variability",
        "coverage": float(config.coverage),
        "epsilon": float(coverage_to_mahalanobis_epsilon(float(config.coverage), obs_dim=4)),
        "transition_covariance_multiplier": float(TRANSITION_COVARIANCE_MULTIPLIER),
        "observation_to_transition_ratio": float(OBSERVATION_TO_TRANSITION_RATIO),
        "n_tuning_episodes": int(config.n_tuning_episodes),
        "n_episodes": int(config.n_episodes),
        "tuning_seeds": [
            int(config.base_seed) + idx
            for idx in range(int(config.n_tuning_episodes))
        ],
        "evaluation_seeds": [
            int(config.base_seed) + int(config.n_tuning_episodes) + idx
            for idx in range(int(config.n_episodes))
        ],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "full_csv_path": full_csv_path,
        "ranking_csv_path": ranking_csv_path,
        "figure_path": figure_path,
        "json_path": json_path,
    }


def run_wolf_benchmark(*, config: DefenseBenchmarkConfig, output_prefix: str) -> dict[str, str]:
    """Run the standalone WoLF benchmark and save its artifacts."""
    print(
        "[wolf_benchmark] "
        f"starting run | output={output_prefix} | "
        f"n_tuning_episodes={int(config.n_tuning_episodes)} | "
        f"n_episodes={int(config.n_episodes)}"
    )
    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, _env_config, _net_config, _train_config, _history = load_agent_checkpoint(model_path, device)
    policy.eval()
    tuning_episodes, _evaluation_episodes = build_episode_sets(config)
    full_df, ranking_df = run_wolf_sweep(
        policy=policy,
        config=config,
        tuning_episodes=tuning_episodes,
        device=device,
        progress_callback=print,
    )
    selected_parameters = select_best_wolf_parameters(full_df=full_df)
    artifact_paths = save_wolf_benchmark_artifacts(
        config=config,
        full_df=full_df,
        ranking_df=ranking_df,
        selected_parameters=selected_parameters,
        output_prefix=output_prefix,
    )
    print(f"[wolf_benchmark] saved full sweep: {artifact_paths['full_csv_path']}")
    print(f"[wolf_benchmark] saved ranking: {artifact_paths['ranking_csv_path']}")
    print(f"[wolf_benchmark] saved figure: {artifact_paths['figure_path']}")
    print(f"[wolf_benchmark] saved json: {artifact_paths['json_path']}")
    return artifact_paths


def parse_args() -> argparse.Namespace:
    """Parse the standalone WoLF benchmark arguments."""
    parser = argparse.ArgumentParser(description="Standalone WoLF RL benchmark.")
    parser.add_argument("--n-tuning-episodes", "--tuning-seeds", dest="n_tuning_episodes", type=int, default=None)
    parser.add_argument("--n-episodes", "--evaluation-seeds", dest="n_episodes", type=int, default=None)
    parser.add_argument("--attack-steps", type=int, default=None)
    parser.add_argument("--attack-mc-samples", type=int, default=None)
    parser.add_argument("--transition-mc-samples", type=int, default=None)
    parser.add_argument("--attack-step-size", type=float, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--attack-probability", type=float, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DefenseBenchmarkConfig:
    """Override the local WoLF configuration with command-line arguments."""
    config = build_wolf_benchmark_config()
    overrides: dict[str, Any] = {}
    if args.n_tuning_episodes is not None:
        overrides["n_tuning_episodes"] = int(args.n_tuning_episodes)
    if args.n_episodes is not None:
        overrides["n_episodes"] = int(args.n_episodes)
    if args.attack_steps is not None:
        overrides["attack_num_steps"] = int(args.attack_steps)
    if args.attack_mc_samples is not None:
        overrides["attack_mc_samples"] = int(args.attack_mc_samples)
    if args.transition_mc_samples is not None:
        overrides["transition_mc_samples"] = int(args.transition_mc_samples)
    if args.attack_step_size is not None:
        overrides["attack_step_size"] = float(args.attack_step_size)
    if args.coverage is not None:
        overrides["coverage"] = float(args.coverage)
    if args.attack_probability is not None:
        overrides["attack_probability"] = float(args.attack_probability)
    return replace(config, **overrides)


def main() -> None:
    """Run the standalone WoLF benchmark."""
    args = parse_args()
    config = config_from_args(args)
    output_prefix = args.output_prefix or default_output_prefix(
        coverage=float(config.coverage),
        n_tuning_episodes=int(config.n_tuning_episodes),
        n_episodes=int(config.n_episodes),
    )
    artifact_paths = run_wolf_benchmark(
        config=config,
        output_prefix=output_prefix,
    )
    for label, path in artifact_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
