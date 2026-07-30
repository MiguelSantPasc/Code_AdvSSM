#!/usr/bin/env python3
"""
random_contour_benchmark.py

Standalone lambda sweep for the random-contour defense benchmark.

Why this module exists:
1. The full `defense_benchmark.py` script mixes several attack families, WoLF
   baselines, plotting modes, and persistence details that are useful for the
   final paper benchmark but cumbersome when we only want to tune the
   covariance-adaptation strength for random-contour attacks.
2. This script isolates the random-contour case in a lightweight entry point:
   it reuses the same RL environment, attack generation, and filtering logic as
   the shared benchmark, but only evaluates the baselines and the requested
   lambda sweep for this one attack family.
3. The script stays standalone in its configuration and outputs, but it also
   loads the case-specific WoLF JSON so the random-contour defenses can be
   compared against the tuned WoLF baselines under the same benchmark case.

Default experiment setup:
1. Attack family: random contour.
2. No gamma activation threshold: the standalone sweep uses `gamma_threshold = 0.0`.
3. Defenses compared in one script:
   - covariance adaptation with PGD direction and `lambda = 1.0`,
   - covariance adaptation with observed-innovation direction and a
     WoLF-MD / IMQ-style `gamma_t`,
   - WoLF-IMQ and WoLF-TMD loaded from the best-configuration JSON of the same
     benchmark case.
4. Outputs: summary CSV, per-episode CSV, diagnostics CSV, compressed NPZ, and
   a compact return-comparison figure in the usual `RL/data` and `RL/figures`
   folders.
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

from benchmark_core import BenchmarkMethod
from benchmark_core import DefenseBenchmarkConfig
from benchmark_core import aggregate_benchmark_results
from benchmark_core import build_episode_sets
from benchmark_core import diagnostics_to_dataframe
from benchmark_core import run_benchmark_episode
from benchmark_core import set_plot_theme
from compare_wind_online_attack_rewards import coverage_to_mahalanobis_epsilon
from covaraicne_sweep import observation_std_from_ratio
from wind_rl_setup import default_device
from wind_rl_setup import rl_model_dir
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_data_dir
from wind_rl_setup import rl_figures_dir


# Standalone random-contour benchmark configuration.
BENCHMARK_COVERAGE = 0.95
TRANSITION_COVARIANCE_MULTIPLIER = 3.0
OBSERVATION_TO_TRANSITION_RATIO = 1.0
ATTACK_PROBABILITY = 0.10
LAMBDA_VALUE = 1.0
GAMMA_THRESHOLD = 0.0
BASE_SEED = 20260727
N_EPISODES = 100
WOLF_JSON_N_TUNING_EPISODES = 6
WOLF_JSON_N_EPISODES = 100
ATTACK_STEP_SIZE = 0.05
ATTACK_NUM_STEPS = 120
ATTACK_MC_SAMPLES = 16
TRANSITION_MC_SAMPLES = 8
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


def default_output_prefix(*, coverage: float, n_episodes: int) -> str:
    """Return the default output prefix for this standalone benchmark."""
    return f"random_contour_benchmark_{benchmark_case_tag(coverage=coverage)}_n{int(n_episodes)}"


def best_wolf_json_path(*, coverage: float) -> str:
    """Return the case-specific JSON path with the selected WoLF parameters."""
    filename = (
        f"best_wolf_params_{benchmark_case_tag(coverage=coverage)}"
        f"_ntune{int(WOLF_JSON_N_TUNING_EPISODES)}_n{int(WOLF_JSON_N_EPISODES)}.json"
    )
    return os.path.join(rl_data_dir(), filename)


def load_best_wolf_params(*, coverage: float) -> dict[str, Any]:
    """Load the selected WoLF parameters for the same epsilon/covariance case."""
    json_path = best_wolf_json_path(coverage=coverage)
    legacy_json_path = os.path.join(rl_data_dir(), "best_wolf_params.json")
    if os.path.exists(json_path):
        selected_json_path = json_path
    elif os.path.exists(legacy_json_path):
        selected_json_path = legacy_json_path
        print(
            "[random_contour_benchmark] "
            f"using legacy WoLF JSON: {legacy_json_path}"
        )
    else:
        raise FileNotFoundError(
            "Missing WoLF parameter JSON for this random-contour case. "
            f"Expected: {json_path} or legacy fallback {legacy_json_path}. "
            "Run RL\\wolf_benchmark.py first."
        )
    with open(selected_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_random_contour_config() -> Any:
    """
    Build the standalone random-contour tuning configuration.

    The script has no separate tuning split, so `n_tuning_episodes` is set to
    zero and every configured seed is used directly for evaluation.
    """
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
        lambdas=(float(LAMBDA_VALUE),),
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
        n_tuning_episodes=0,
        n_episodes=int(N_EPISODES),
        filter_config=filter_config,
        simulation_config=simulation_config,
    )


def build_random_contour_methods(*, lambda_value: float, wolf_params: dict[str, Any]) -> list[BenchmarkMethod]:
    """Return the defended random-contour comparison used in this standalone script."""
    methods = [
        BenchmarkMethod(
            name="Clean",
            attack_type="clean",
            filter_type="clean",
            use_covariance_adaptation=False,
            lambda_covariance=0.0,
            defense_direction_source="none",
        ),
        BenchmarkMethod(
            name="Noisy + KF",
            attack_type="none",
            filter_type="kf",
            use_covariance_adaptation=False,
            lambda_covariance=0.0,
            defense_direction_source="none",
        ),
        BenchmarkMethod(
            name="Random contour, no defense",
            attack_type="random_contour",
            filter_type="kf",
            use_covariance_adaptation=False,
            lambda_covariance=0.0,
            defense_direction_source="none",
        ),
        BenchmarkMethod(
            name=f"Random contour + PGD direction, lambda={float(lambda_value)}",
            attack_type="random_contour",
            filter_type="covariance_adaptation",
            use_covariance_adaptation=True,
            lambda_covariance=float(lambda_value),
            defense_direction_source="pgd_estimated",
        ),
        BenchmarkMethod(
            name=f"Random contour + observed direction + WoLF-MD, lambda={float(lambda_value)}",
            attack_type="random_contour",
            filter_type="covariance_adaptation_observation_imq",
            use_covariance_adaptation=True,
            lambda_covariance=float(lambda_value),
            defense_direction_source="observed_innovation",
        ),
    ]
    for wolf_key, label in (("wolf_imq", "WoLF-IMQ"), ("wolf_tmd", "WoLF-TMD")):
        selected = wolf_params[wolf_key]["selected_parameters"]
        methods.append(
            BenchmarkMethod(
                name=f"Random contour + {label}",
                attack_type="random_contour",
                filter_type=wolf_key,
                use_covariance_adaptation=False,
                lambda_covariance=0.0,
                defense_direction_source="none",
                wolf_kind=str(selected["kind"]),
                wolf_parameters={
                    key: float(value) if isinstance(value, (int, float)) else value
                    for key, value in selected.items()
                },
            )
        )
    return methods


def ordered_method_names(*, lambda_value: float) -> list[str]:
    """Return the plotting and reporting order for the random-contour comparison."""
    return [
        "Clean",
        "Noisy + KF",
        "Random contour, no defense",
        f"Random contour + PGD direction, lambda={float(lambda_value)}",
        f"Random contour + observed direction + WoLF-MD, lambda={float(lambda_value)}",
        "Random contour + WoLF-IMQ",
        "Random contour + WoLF-TMD",
    ]


def plot_random_contour_returns(
    *,
    summary_df: pd.DataFrame,
    lambda_value: float,
    output_path: str,
) -> None:
    """Save the compact return comparison used to compare random-contour defenses."""
    ordered_labels = ordered_method_names(lambda_value=lambda_value)
    subset_df = summary_df.loc[summary_df["method"].isin(ordered_labels)].copy()
    subset_df["method"] = pd.Categorical(subset_df["method"], categories=ordered_labels, ordered=True)
    subset_df = subset_df.sort_values("method")

    short_labels = {
        "Clean": "Clean",
        "Noisy + KF": "Noisy + KF",
        "Random contour, no defense": "No defense",
        f"Random contour + PGD direction, lambda={float(lambda_value)}": "PGD dir",
        f"Random contour + observed direction + WoLF-MD, lambda={float(lambda_value)}": "Obs dir + WoLF-MD",
        "Random contour + WoLF-IMQ": "WoLF-IMQ",
        "Random contour + WoLF-TMD": "WoLF-TMD",
    }
    color_by_method = {
        "Clean": "#D7E3F1",
        "Noisy + KF": "#BFD3EA",
        "Random contour, no defense": "#AFC7E8",
        f"Random contour + PGD direction, lambda={float(lambda_value)}": "#F4D6A0",
        f"Random contour + observed direction + WoLF-MD, lambda={float(lambda_value)}": "#E6A57E",
        "Random contour + WoLF-IMQ": "#D8C3EA",
        "Random contour + WoLF-TMD": "#BDA6DB",
    }

    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x_positions = np.arange(len(subset_df), dtype=float)
    bar_values = subset_df["mean_return"].to_numpy(dtype=float)
    bars = ax.bar(
        x_positions,
        bar_values,
        width=0.72,
        color=[color_by_method[str(method_name)] for method_name in subset_df["method"].tolist()],
        edgecolor="white",
        linewidth=0.9,
    )
    value_span = float(np.max(bar_values) - np.min(bar_values)) if bar_values.size > 0 else 0.0
    text_offset = 0.02 * max(1.0, abs(float(np.max(bar_values))) if bar_values.size > 0 else 1.0, value_span)
    for bar, value in zip(bars, bar_values, strict=False):
        label_y = float(bar.get_height()) + text_offset
        ax.text(
            float(bar.get_x() + bar.get_width() / 2.0),
            label_y,
            f"{float(value):.2f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#31424F",
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [short_labels[str(method_name)] for method_name in subset_df["method"].tolist()],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Mean accumulated reward")
    ax.set_xlabel("Random contour defense")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.24)
    ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_saved_random_contour_results(*, output_prefix: str) -> pd.DataFrame:
    """Load the saved random-contour summary table without rerunning episodes."""
    summary_csv_path = os.path.join(rl_data_dir(), f"{output_prefix}_summary.csv")
    if not os.path.exists(summary_csv_path):
        raise FileNotFoundError(f"Summary file not found: {summary_csv_path}")
    return pd.read_csv(summary_csv_path)


def run_random_contour_benchmark(*, config: Any, output_prefix: str) -> dict[str, str]:
    """Run the standalone random-contour lambda benchmark and save its artifacts."""
    wolf_params = load_best_wolf_params(coverage=float(config.coverage))
    lambda_value = float(config.lambdas[0])
    methods = build_random_contour_methods(
        lambda_value=lambda_value,
        wolf_params=wolf_params,
    )
    print(
        "[random_contour_benchmark] "
        f"starting run | output={output_prefix} | "
        f"n_episodes={int(config.n_episodes)} | "
        f"n_methods={len(methods)}"
    )
    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, _env_config, _net_config, _train_config, _history = load_agent_checkpoint(model_path, device)
    policy.eval()

    _tuning_episodes, evaluation_episodes = build_episode_sets(config)

    all_results: list[Any] = []
    total_methods = len(methods)
    total_episodes = len(evaluation_episodes)
    for method_index, method in enumerate(methods, start=1):
        print(
            "[random_contour_benchmark] "
            f"method {method_index}/{total_methods} | {method.name}"
        )
        for episode_index, episode in enumerate(evaluation_episodes, start=1):
            all_results.append(
                run_benchmark_episode(
                    policy=policy,
                    method=method,
                    episode=episode,
                    config=config,
                    device=device,
                )
            )
            if episode_index == 1 or episode_index == total_episodes or episode_index % 10 == 0:
                print(
                    "[random_contour_benchmark] "
                    f"{method.name} | episode {episode_index}/{total_episodes}"
                )

    summary_df = aggregate_benchmark_results(all_results)
    diagnostics_df = diagnostics_to_dataframe(all_results)
    episode_df = pd.DataFrame(
        [
            {
                "method": result.method.name,
                "episode_seed": int(result.episode_seed),
                "episode_return": float(result.episode_return),
                "success": float(result.success),
                "final_goal_distance": float(result.final_goal_distance),
                "mean_state_estimation_error": float(result.mean_state_estimation_error),
                "steps": int(result.steps),
            }
            for result in all_results
        ]
    )

    data_dir = rl_data_dir()
    figures_dir = rl_figures_dir()
    summary_csv_path = os.path.join(data_dir, f"{output_prefix}_summary.csv")
    episodes_csv_path = os.path.join(data_dir, f"{output_prefix}_episodes.csv")
    diagnostics_csv_path = os.path.join(data_dir, f"{output_prefix}_diagnostics.csv")
    results_npz_path = os.path.join(data_dir, f"{output_prefix}.npz")
    returns_figure_path = os.path.join(figures_dir, f"{output_prefix}_returns.png")

    summary_df.to_csv(summary_csv_path, index=False)
    episode_df.to_csv(episodes_csv_path, index=False)
    diagnostics_df.to_csv(diagnostics_csv_path, index=False)
    np.savez_compressed(
        results_npz_path,
        summary_columns=np.asarray(summary_df.columns.tolist(), dtype=object),
        summary_values=summary_df.to_numpy(dtype=object),
        episode_columns=np.asarray(episode_df.columns.tolist(), dtype=object),
        episode_values=episode_df.to_numpy(dtype=object),
        diagnostics_columns=np.asarray(diagnostics_df.columns.tolist(), dtype=object),
        diagnostics_values=diagnostics_df.to_numpy(dtype=object),
        lambda_value=np.asarray([lambda_value], dtype=float),
        gamma_threshold=np.asarray([config.gamma_threshold], dtype=float),
        attack_probability=np.asarray([config.attack_probability], dtype=float),
        coverage=np.asarray([config.coverage], dtype=float),
        n_episodes=np.asarray([config.n_episodes], dtype=int),
        process_position_std=np.asarray([config.filter_config.process_position_std], dtype=float),
        process_wind_std=np.asarray([config.filter_config.process_wind_std], dtype=float),
        observation_noise_std=np.asarray([config.filter_config.observation_noise_std], dtype=float),
    )
    max_gamma = float(diagnostics_df["gamma_t"].max()) if not diagnostics_df.empty else 0.0
    max_gamma_row = diagnostics_df.loc[diagnostics_df["gamma_t"].idxmax()] if not diagnostics_df.empty else None
    plot_random_contour_returns(
        summary_df=summary_df,
        lambda_value=lambda_value,
        output_path=returns_figure_path,
    )
    print(f"max_gamma_t: {max_gamma:.6f}")
    if max_gamma_row is not None:
        print(
            "max_gamma_t_details: "
            f"method={max_gamma_row['method']}, "
            f"episode_seed={int(max_gamma_row['episode_seed'])}, "
            f"step_index={int(max_gamma_row['step_index'])}"
        )
    print(f"[random_contour_benchmark] saved summary: {summary_csv_path}")
    print(f"[random_contour_benchmark] saved diagnostics: {diagnostics_csv_path}")
    print(f"[random_contour_benchmark] saved figure: {returns_figure_path}")
    return {
        "summary_csv_path": summary_csv_path,
        "episodes_csv_path": episodes_csv_path,
        "diagnostics_csv_path": diagnostics_csv_path,
        "results_npz_path": results_npz_path,
        "returns_figure_path": returns_figure_path,
    }


def replot_saved_random_contour_benchmark(*, config: Any, output_prefix: str) -> dict[str, str]:
    """Regenerate the random-contour return figure from the saved summary CSV."""
    summary_df = load_saved_random_contour_results(output_prefix=output_prefix)
    returns_figure_path = os.path.join(rl_figures_dir(), f"{output_prefix}_returns.png")
    plot_random_contour_returns(
        summary_df=summary_df,
        lambda_value=float(config.lambdas[0]),
        output_path=returns_figure_path,
    )
    return {"returns_figure_path": returns_figure_path}


def parse_args() -> argparse.Namespace:
    """Parse the standalone random-contour benchmark arguments."""
    parser = argparse.ArgumentParser(description="Standalone random-contour RL benchmark.")
    parser.add_argument("--mode", choices=("benchmark", "plot"), default="benchmark")
    parser.add_argument("--n-episodes", "--evaluation-seeds", dest="n_episodes", type=int, default=None)
    parser.add_argument("--attack-steps", type=int, default=None)
    parser.add_argument("--attack-mc-samples", type=int, default=None)
    parser.add_argument("--transition-mc-samples", type=int, default=None)
    parser.add_argument("--attack-step-size", type=float, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--attack-probability", type=float, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Any:
    """Apply the command-line overrides to the standalone random-contour config."""
    config = build_random_contour_config()
    overrides: dict[str, Any] = {}
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
    """Run or replot the standalone random-contour lambda sweep."""
    args = parse_args()
    config = config_from_args(args)
    output_prefix = args.output_prefix or default_output_prefix(
        coverage=float(config.coverage),
        n_episodes=int(config.n_episodes),
    )

    if args.mode == "plot":
        artifact_paths = replot_saved_random_contour_benchmark(
            config=config,
            output_prefix=output_prefix,
        )
    else:
        artifact_paths = run_random_contour_benchmark(
            config=config,
            output_prefix=output_prefix,
        )

    for label, path in artifact_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
