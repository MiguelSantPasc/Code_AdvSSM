#!/usr/bin/env python3
"""
covaraicne_sweep.py

Run the final RL attack benchmark for several observation/process covariance
ratios.

Why this script exists:
1. The approximation error of the online KF state estimate depends strongly on
   the relative size of the observation covariance `V_t` and transition noise
   covariance `W_t`.
2. This script repeats the compact final benchmark for three requested ratios:
      V:W = 1:4, 1:1, and 4:1.
3. Each ratio is evaluated for three real/environment transition-noise levels:
      W_t, 2 W_t, and 4 W_t.
   Larger rows multiply the transition covariance, so the process standard
   deviations used by the simulator are multiplied by the square root of that
   factor. The online filter is matched to the same process covariance, as in a
   white-box setting.
4. Because the wind process covariance is diagonal but not isotropic, the ratio
   is defined against the mean diagonal transition variance:
      mean_diag(W_t) = mean([sigma_p^2, sigma_p^2, sigma_w^2, sigma_w^2]).
   The observation covariance remains isotropic:
      V_t = sigma_o^2 I,
      sigma_o^2 = ratio * mean_diag(W_t).
5. Numerical results are saved in `RL/data/`; the figure is saved in
   `RL/figures/`.
6. Output filenames include the evaluated coverage, e.g.
      covaraicne_sweep_eps0p95.npz
   If that data file already exists, the script loads it instead of rerunning
   the benchmark.

The file name intentionally follows the user-requested spelling:
`covaraicne_sweep.py`.
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
from final_comparison import FINAL_SETTING_NAMES
from final_comparison import standard_error
from wind_rl_setup import default_device
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_data_dir
from wind_rl_setup import rl_figures_dir
from wind_rl_setup import rl_model_dir


def transition_reference_variance(
    *,
    process_position_std: float,
    process_wind_std: float,
) -> float:
    """Return the mean diagonal variance of the transition covariance `W_t`."""
    position_variance = float(process_position_std) ** 2
    wind_variance = float(process_wind_std) ** 2
    return float(np.mean([position_variance, position_variance, wind_variance, wind_variance]))


def observation_std_from_ratio(
    *,
    covariance_ratio_value: float,
    process_position_std: float,
    process_wind_std: float,
) -> float:
    """Return `sigma_o` such that `V_t / mean_diag(W_t) = covariance_ratio_value`."""
    reference_variance = transition_reference_variance(
        process_position_std=float(process_position_std),
        process_wind_std=float(process_wind_std),
    )
    observation_variance = float(covariance_ratio_value) * reference_variance
    return float(np.sqrt(max(observation_variance, 0.0)))


def scaled_process_stds(
    *,
    base_process_position_std: float,
    base_process_wind_std: float,
    transition_covariance_multiplier: float,
) -> tuple[float, float]:
    """Scale process standard deviations so the transition covariance is multiplied."""
    std_multiplier = np.sqrt(float(transition_covariance_multiplier))
    return (
        float(base_process_position_std) * float(std_multiplier),
        float(base_process_wind_std) * float(std_multiplier),
    )


def coverage_value_tag(coverage: float) -> str:
    """Return a filename-safe tag such as `0p95` for one coverage value."""
    return f"{float(coverage):.2f}".replace(".", "p")


def output_stem_with_coverages(*, base_output_stem: str, coverages: tuple[float, ...]) -> str:
    """Append the evaluated epsilon/coverage values to the result filename stem."""
    coverage_part = "_".join(coverage_value_tag(float(coverage)) for coverage in coverages)
    return f"{base_output_stem}_eps{coverage_part}"


def set_plot_theme() -> None:
    """Apply a compact pastel plotting style for the covariance sweep figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9.6,
            "axes.labelsize": 10.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.68,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
            "savefig.facecolor": "white",
        }
    )


def transition_multiplier_label(transition_covariance_multiplier: float) -> str:
    """Return a compact label such as `W_base`, `2W_base`, or `4W_base`."""
    multiplier_value = float(transition_covariance_multiplier)
    if np.isclose(multiplier_value, 1.0):
        return r"$W_{\mathrm{base}}$"
    multiplier_text = f"{int(multiplier_value)}" if multiplier_value.is_integer() else f"{multiplier_value:g}"
    return rf"${multiplier_text}W_{{\mathrm{{base}}}}$"


def coverage_box_label(coverages: tuple[float, ...]) -> str:
    """Return the compact epsilon label shown in the top-right figure box."""
    if len(coverages) == 1:
        return rf"$\epsilon = {coverages[0]:.2f}$"
    coverage_text = ", ".join(f"{coverage:.2f}" for coverage in coverages)
    return rf"$\epsilon \in \{{{coverage_text}\}}$"


def covariance_sweep_legend_labels() -> tuple[str, ...]:
    """Return the user-facing legend labels for the covariance sweep figure."""
    return (
        "Noiseless",
        "Noisy + KF",
        r"Attack on $Q^\pi(m_T,a_T)$ + KF",
        r"Attack on $Q^\pi(s_T^0,a_T)$ + KF",
        r"$\epsilon$ perturbation + KF",
    )


def covariance_sweep_extreme_case_specs() -> tuple[tuple[float, str], ...]:
    """Return the selected `W` / `V:W` cases for the multi-panel compact plot."""
    return (
        (1.0, "4:1"),
        (2.0, "1:1"),
        (4.0, "1:4"),
    )


def covariance_sweep_intermediate_case_specs() -> tuple[tuple[float, str], ...]:
    """Return the intermediate `W` / `V:W` case requested for the compact plot."""
    return ((2.0, "1:1"),)


def plot_covariance_sweep(
    *,
    coverages: tuple[float, ...],
    transition_noise_labels: tuple[str, ...],
    transition_covariance_multipliers: np.ndarray,
    transition_reference_variances: np.ndarray,
    ratio_labels: tuple[str, ...],
    ratio_values: np.ndarray,
    observation_noise_stds: np.ndarray,
    mean_returns: np.ndarray,
    sem_returns: np.ndarray,
    outpath: str,
) -> None:
    """Save a benchmark grid over transition noise and `V:W` ratios."""
    set_plot_theme()
    fig, axes = plt.subplots(
        len(transition_noise_labels),
        len(ratio_labels),
        figsize=(5.05 * len(ratio_labels), 4.6 * len(transition_noise_labels)),
        sharey=True,
    )
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(len(transition_noise_labels), len(ratio_labels))

    colors = ["#8EC5B5", "#9DB7D5", "#F0C987", "#E6A57E", "#CFA7D8"]
    legend_labels = covariance_sweep_legend_labels()
    x_positions = np.arange(len(coverages), dtype=float)
    width = min(0.058, 0.38 / max(len(FINAL_SETTING_NAMES), 1))
    offsets = (np.arange(len(FINAL_SETTING_NAMES)) - 0.5 * (len(FINAL_SETTING_NAMES) - 1)) * (1.18 * width)
    legend_handles: list[Any] = []

    for transition_idx, _transition_label in enumerate(transition_noise_labels):
        for ratio_idx, ratio_label in enumerate(ratio_labels):
            ax = axes[transition_idx, ratio_idx]
            for setting_idx, (_setting_name, color) in enumerate(zip(FINAL_SETTING_NAMES, colors)):
                bar_container = ax.bar(
                    x_positions + offsets[setting_idx],
                    mean_returns[transition_idx, ratio_idx, :, setting_idx],
                    width=width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.45,
                    label=legend_labels[setting_idx],
                )
                if transition_idx == 0 and ratio_idx == 0:
                    legend_handles.append(bar_container[0])

            ax.set_xticks(x_positions)
            if len(coverages) == 1:
                ax.set_xticklabels([""])
            else:
                ax.set_xticklabels([rf"$\epsilon={coverage:.2f}$" for coverage in coverages])
            ax.set_xlabel(
                "  ".join(
                    [
                        rf"$W = {transition_multiplier_label(transition_covariance_multipliers[transition_idx])[1:-1]}$",
                        rf"$V\!:\!W = {ratio_label}$",
                    ]
                ),
                labelpad=8.0,
                fontsize=12.2,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.24)
            ax.grid(False, axis="x")
            ax.axhline(0.0, color="black", linewidth=0.65, alpha=0.75, zorder=0)

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean accumulated reward")
    if legend_handles:
        # Keep one shared legend for the full figure instead of repeating it in every panel.
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.60, 0.985),
            ncols=5,
            fontsize=12.6,
            frameon=True,
            framealpha=0.92,
        )
    fig.text(
        0.035,
        0.972,
        coverage_box_label(coverages),
        ha="left",
        va="top",
        fontsize=12.6,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "alpha": 0.96,
            "edgecolor": "#BFC8D0",
        },
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_covariance_sweep_selected_cases(
    *,
    coverages: tuple[float, ...],
    transition_covariance_multipliers: np.ndarray,
    ratio_labels: tuple[str, ...],
    mean_returns: np.ndarray,
    case_specs: tuple[tuple[float, str], ...],
    outpath: str,
) -> None:
    """Save a compact figure for one or more selected covariance cases."""
    if len(coverages) != 1:
        return

    set_plot_theme()
    legend_labels = covariance_sweep_legend_labels()
    colors = ["#8EC5B5", "#9DB7D5", "#F0C987", "#E6A57E", "#CFA7D8"]
    width = 0.055
    offsets = (np.arange(len(FINAL_SETTING_NAMES)) - 0.5 * (len(FINAL_SETTING_NAMES) - 1)) * (1.22 * width)

    num_cases = len(case_specs)
    fig_width = 7.4 if num_cases == 1 else 12.6
    legend_width = 0.82 if num_cases == 1 else 0.68
    fig = plt.figure(figsize=(fig_width, 4.4))
    width_ratios = [1.0] * num_cases + [legend_width]
    grid = fig.add_gridspec(1, num_cases + 1, width_ratios=width_ratios, wspace=0.10)
    data_axes = [fig.add_subplot(grid[0, axis_idx]) for axis_idx in range(num_cases)]
    for axis_idx in range(1, num_cases):
        data_axes[axis_idx].sharey(data_axes[0])
    legend_ax = fig.add_subplot(grid[0, num_cases])
    legend_ax.axis("off")

    for axis_idx, (ax, (target_multiplier, target_ratio_label)) in enumerate(zip(data_axes, case_specs)):
        transition_idx = int(np.where(np.isclose(transition_covariance_multipliers, float(target_multiplier)))[0][0])
        ratio_idx = ratio_labels.index(target_ratio_label)
        for setting_idx, color in enumerate(colors):
            ax.bar(
                offsets[setting_idx],
                mean_returns[transition_idx, ratio_idx, 0, setting_idx],
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                label=legend_labels[setting_idx],
            )

        ax.set_xlim(-0.22, 0.22)
        ax.set_xticks([0.0])
        ax.set_xticklabels([""])
        ax.set_xlabel(
            "  ".join(
                [
                    rf"$W = {transition_multiplier_label(target_multiplier)[1:-1]}$",
                    rf"$V\!:\!W = {target_ratio_label}$",
                ]
            ),
            labelpad=8.0,
            fontsize=12.2,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.24)
        ax.grid(False, axis="x")
        ax.axhline(0.0, color="black", linewidth=0.65, alpha=0.75, zorder=0)
        if axis_idx > 0:
            plt.setp(ax.get_yticklabels(), visible=False)

    data_axes[0].set_ylabel("Mean accumulated reward")

    # Use a dedicated legend column so the legend sits at the far right
    # without overlapping the data axes.
    legend_ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.45)
            for color in colors
        ],
        labels=legend_labels,
        loc="center",
        fontsize=12.4,
        frameon=True,
        framealpha=0.92,
    )
    legend_ax.text(
        0.5,
        0.89,
        coverage_box_label(coverages),
        ha="center",
        va="top",
        fontsize=12.4,
        transform=legend_ax.transAxes,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "alpha": 0.96,
            "edgecolor": "#BFC8D0",
        },
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.20, wspace=0.10)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_covariance_sweep_extremes(
    *,
    coverages: tuple[float, ...],
    transition_covariance_multipliers: np.ndarray,
    ratio_labels: tuple[str, ...],
    mean_returns: np.ndarray,
    outpath: str,
) -> None:
    """Save a compact figure with the requested selected covariance cases."""
    plot_covariance_sweep_selected_cases(
        coverages=coverages,
        transition_covariance_multipliers=transition_covariance_multipliers,
        ratio_labels=ratio_labels,
        mean_returns=mean_returns,
        case_specs=covariance_sweep_extreme_case_specs(),
        outpath=outpath,
    )


def plot_covariance_sweep_intermediate(
    *,
    coverages: tuple[float, ...],
    transition_covariance_multipliers: np.ndarray,
    ratio_labels: tuple[str, ...],
    mean_returns: np.ndarray,
    outpath: str,
) -> None:
    """Save a compact figure with the requested intermediate covariance case."""
    plot_covariance_sweep_selected_cases(
        coverages=coverages,
        transition_covariance_multipliers=transition_covariance_multipliers,
        ratio_labels=ratio_labels,
        mean_returns=mean_returns,
        case_specs=covariance_sweep_intermediate_case_specs(),
        outpath=outpath,
    )


def covariance_sweep_attack_records_to_arrays(
    records: list[dict[str, float | int | str]],
) -> dict[str, np.ndarray]:
    """Convert attack diagnostics plus covariance-ratio metadata into arrays."""
    arrays = attack_records_to_arrays(records)
    arrays.update(
        {
            "attack_record_covariance_ratio_label": np.asarray(
                [record["covariance_ratio_label"] for record in records],
                dtype=object,
            ),
            "attack_record_covariance_ratio_value": np.asarray(
                [record["covariance_ratio_value"] for record in records],
                dtype=float,
            ),
            "attack_record_observation_noise_std": np.asarray(
                [record["observation_noise_std"] for record in records],
                dtype=float,
            ),
            "attack_record_transition_noise_label": np.asarray(
                [record["transition_noise_label"] for record in records],
                dtype=object,
            ),
            "attack_record_transition_covariance_multiplier": np.asarray(
                [record["transition_covariance_multiplier"] for record in records],
                dtype=float,
            ),
            "attack_record_transition_reference_variance": np.asarray(
                [record["transition_reference_variance"] for record in records],
                dtype=float,
            ),
            "attack_record_process_position_std": np.asarray(
                [record["process_position_std"] for record in records],
                dtype=float,
            ),
            "attack_record_process_wind_std": np.asarray(
                [record["process_wind_std"] for record in records],
                dtype=float,
            ),
        }
    )
    return arrays


def load_cached_covariance_sweep(
    *,
    data_path: str,
    figure_path: str,
    extremes_figure_path: str | None = None,
    intermediate_figure_path: str | None = None,
) -> dict[str, Any]:
    """Load a cached covariance sweep and ensure its figure exists."""
    with np.load(data_path, allow_pickle=True) as cached:
        coverages = tuple(float(value) for value in cached["coverages"])
        transition_noise_labels = tuple(str(value) for value in cached["transition_noise_labels"])
        covariance_ratio_labels = tuple(str(value) for value in cached["covariance_ratio_labels"])
        transition_covariance_multipliers = np.asarray(
            cached["transition_covariance_multipliers"],
            dtype=float,
        )
        transition_reference_variances = np.asarray(
            cached["transition_reference_variances"],
            dtype=float,
        )
        covariance_ratio_values = np.asarray(cached["covariance_ratio_values"], dtype=float)
        observation_noise_stds = np.asarray(cached["observation_noise_stds"], dtype=float)
        mean_returns = np.asarray(cached["mean_returns"], dtype=float)
        sem_returns = np.asarray(cached["sem_returns"], dtype=float)
        success_rates = np.asarray(cached["success_rates"], dtype=float)

    print(f"Using cached covariance sweep data: {data_path}")
    plot_covariance_sweep(
        coverages=coverages,
        transition_noise_labels=transition_noise_labels,
        transition_covariance_multipliers=transition_covariance_multipliers,
        transition_reference_variances=transition_reference_variances,
        ratio_labels=covariance_ratio_labels,
        ratio_values=covariance_ratio_values,
        observation_noise_stds=observation_noise_stds,
        mean_returns=mean_returns,
        sem_returns=sem_returns,
        outpath=figure_path,
    )
    print(f"Regenerated figure from cache:      {figure_path}")
    if extremes_figure_path is not None:
        plot_covariance_sweep_extremes(
            coverages=coverages,
            transition_covariance_multipliers=transition_covariance_multipliers,
            ratio_labels=covariance_ratio_labels,
            mean_returns=mean_returns,
            outpath=extremes_figure_path,
        )
        print(f"Regenerated extreme-case figure:   {extremes_figure_path}")
    if intermediate_figure_path is not None:
        plot_covariance_sweep_intermediate(
            coverages=coverages,
            transition_covariance_multipliers=transition_covariance_multipliers,
            ratio_labels=covariance_ratio_labels,
            mean_returns=mean_returns,
            outpath=intermediate_figure_path,
        )
        print(f"Regenerated intermediate figure:  {intermediate_figure_path}")

    return {
        "figure_path": figure_path,
        "extremes_figure_path": extremes_figure_path,
        "intermediate_figure_path": intermediate_figure_path,
        "data_path": data_path,
        "mean_returns": mean_returns,
        "success_rates": success_rates,
        "observation_noise_stds": observation_noise_stds,
        "transition_reference_variances": transition_reference_variances,
        "cached": True,
    }


def run_covariance_sweep(
    *,
    n_episodes: int,
    coverages: tuple[float, ...],
    covariance_ratio_specs: tuple[tuple[str, float], ...],
    transition_noise_specs: tuple[tuple[str, float], ...],
    attack_prob: float,
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
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    """Run the final benchmark for each transition-noise level and ratio."""
    figure_path = os.path.join(rl_figures_dir(), f"{output_stem}.png")
    extremes_figure_path = os.path.join(rl_figures_dir(), f"{output_stem}_extremes.png")
    intermediate_figure_path = os.path.join(rl_figures_dir(), f"{output_stem}_intermediate.png")
    data_path = os.path.join(rl_data_dir(), f"{output_stem}.npz")
    if os.path.exists(data_path):
        return load_cached_covariance_sweep(
            data_path=data_path,
            figure_path=figure_path,
            extremes_figure_path=extremes_figure_path,
            intermediate_figure_path=intermediate_figure_path,
        )

    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, env_config, _net_config, train_config, _history = load_agent_checkpoint(model_path, device)
    policy.eval()
    gamma = float(train_config.gamma)

    episode_seeds = int(base_seed) + np.arange(int(n_episodes), dtype=int)

    transition_noise_labels = tuple(label for label, _multiplier in transition_noise_specs)
    transition_covariance_multipliers = np.asarray(
        [multiplier for _label, multiplier in transition_noise_specs],
        dtype=float,
    )
    process_position_stds = np.zeros(len(transition_noise_specs), dtype=float)
    process_wind_stds = np.zeros(len(transition_noise_specs), dtype=float)
    transition_reference_variances = np.zeros(len(transition_noise_specs), dtype=float)
    ratio_labels = tuple(label for label, _ratio in covariance_ratio_specs)
    ratio_values = np.asarray([ratio for _label, ratio in covariance_ratio_specs], dtype=float)
    observation_noise_stds = np.zeros((len(transition_noise_specs), len(covariance_ratio_specs)), dtype=float)

    for transition_idx, (_label, covariance_multiplier) in enumerate(transition_noise_specs):
        scaled_position_std, scaled_wind_std = scaled_process_stds(
            base_process_position_std=float(process_position_std),
            base_process_wind_std=float(process_wind_std),
            transition_covariance_multiplier=float(covariance_multiplier),
        )
        process_position_stds[transition_idx] = scaled_position_std
        process_wind_stds[transition_idx] = scaled_wind_std
        transition_reference_variances[transition_idx] = transition_reference_variance(
            process_position_std=float(scaled_position_std),
            process_wind_std=float(scaled_wind_std),
        )
        for ratio_idx, (_ratio_label, ratio_value) in enumerate(covariance_ratio_specs):
            observation_noise_stds[transition_idx, ratio_idx] = observation_std_from_ratio(
                covariance_ratio_value=float(ratio_value),
                process_position_std=float(scaled_position_std),
                process_wind_std=float(scaled_wind_std),
            )

    mean_returns = np.zeros(
        (
            len(transition_noise_specs),
            len(covariance_ratio_specs),
            len(coverages),
            len(FINAL_SETTING_NAMES),
        ),
        dtype=float,
    )
    sem_returns = np.zeros_like(mean_returns)
    success_rates = np.zeros_like(mean_returns)
    mean_lengths = np.zeros_like(mean_returns)
    mean_attacks = np.zeros_like(mean_returns)
    attack_value_records: list[dict[str, float | int | str]] = []

    print(
        "Covariance sweep benchmark: "
        f"n_episodes={n_episodes}, "
        f"attack_prob={attack_prob:.3f}, "
        f"pgd_steps={attack_num_steps}, "
        f"posterior_mc={attack_mc_samples}, "
        f"transition_mc={real_attack_transition_samples}"
    )
    print(
        "Base transition noise: "
        f"process_position_std={process_position_std:.3f}, "
        f"process_wind_std={process_wind_std:.3f}"
    )

    for transition_idx, ((transition_label, covariance_multiplier), scaled_position_std, scaled_wind_std) in enumerate(
        zip(transition_noise_specs, process_position_stds, process_wind_stds)
    ):
        simulation_config = replace(
            env_config,
            max_steps=int(max_steps_override) if max_steps_override is not None else int(env_config.max_steps),
            observation_noise_std=0.0,
            process_position_std=float(scaled_position_std),
            process_wind_std=float(scaled_wind_std),
        )
        print(
            f"\nReal transition row: {transition_label} "
            f"(W/W0={float(covariance_multiplier):.2f}, "
            f"process_position_std={float(scaled_position_std):.4f}, "
            f"process_wind_std={float(scaled_wind_std):.4f}, "
            f"mean_diag_W={transition_reference_variances[transition_idx]:.6f})"
        )

        for ratio_idx, (ratio_label, ratio_value) in enumerate(covariance_ratio_specs):
            observation_noise_std = float(observation_noise_stds[transition_idx, ratio_idx])
            filter_config = replace(
                simulation_config,
                observation_noise_std=float(observation_noise_std),
            )
            print(
                f"  V:W={ratio_label} "
                f"(V/mean_diag_W={float(ratio_value):.2f}, "
                f"observation_std={float(observation_noise_std):.4f})"
            )

            for coverage_idx, coverage in enumerate(coverages):
                mahalanobis_epsilon = coverage_to_mahalanobis_epsilon(float(coverage), obs_dim=4)
                print(
                    f"    coverage epsilon={coverage:.2f} "
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

                    records = result["attack_value_records"]
                    for record in records:
                        record["covariance_ratio_label"] = ratio_label
                        record["covariance_ratio_value"] = float(ratio_value)
                        record["observation_noise_std"] = float(observation_noise_std)
                        record["transition_noise_label"] = transition_label
                        record["transition_covariance_multiplier"] = float(covariance_multiplier)
                        record["transition_reference_variance"] = float(
                            transition_reference_variances[transition_idx]
                        )
                        record["process_position_std"] = float(scaled_position_std)
                        record["process_wind_std"] = float(scaled_wind_std)
                    attack_value_records.extend(records)

                    returns = np.asarray(result["returns"], dtype=float)
                    successes = np.asarray(result["successes"], dtype=float)
                    lengths = np.asarray(result["lengths"], dtype=float)
                    attacks = np.asarray(result["num_attacks"], dtype=float)

                    mean_returns[transition_idx, ratio_idx, coverage_idx, setting_idx] = float(np.mean(returns))
                    sem_returns[transition_idx, ratio_idx, coverage_idx, setting_idx] = standard_error(returns)
                    success_rates[transition_idx, ratio_idx, coverage_idx, setting_idx] = float(np.mean(successes))
                    mean_lengths[transition_idx, ratio_idx, coverage_idx, setting_idx] = float(np.mean(lengths))
                    mean_attacks[transition_idx, ratio_idx, coverage_idx, setting_idx] = float(np.mean(attacks))

                    print(
                        f"      {setting:20s} "
                        f"mean_return={mean_returns[transition_idx, ratio_idx, coverage_idx, setting_idx]:8.3f} "
                        f"success={100.0 * success_rates[transition_idx, ratio_idx, coverage_idx, setting_idx]:6.2f}% "
                        f"mean_length={mean_lengths[transition_idx, ratio_idx, coverage_idx, setting_idx]:6.2f} "
                        f"mean_attacks={mean_attacks[transition_idx, ratio_idx, coverage_idx, setting_idx]:5.2f}"
                    )

                    if records:
                        contour_summary = attack_contour_summary(
                            records=records,
                            relative_tolerance=float(contour_relative_tolerance),
                        )
                        delta_values = np.asarray([record["delta_value"] for record in records], dtype=float)
                        position_perturbations = np.asarray(
                            [record["state_position_perturbation_norm"] for record in records],
                            dtype=float,
                        )
                        wind_perturbations = np.asarray(
                            [record["state_wind_perturbation_norm"] for record in records],
                            dtype=float,
                        )
                        print(
                            f"      {'attack summary':20s} "
                            f"inside={100.0 * contour_summary['inside_rate']:6.2f}% "
                            f"on_contour={100.0 * contour_summary['on_contour_rate']:6.2f}% "
                            f"mean_deltaV={np.mean(delta_values):8.4f} "
                            f"mean|dp|={np.mean(position_perturbations):6.3f} "
                            f"mean|dw|={np.mean(wind_perturbations):6.3f}"
                        )

    plot_covariance_sweep(
        coverages=coverages,
        transition_noise_labels=transition_noise_labels,
        transition_covariance_multipliers=transition_covariance_multipliers,
        transition_reference_variances=transition_reference_variances,
        ratio_labels=ratio_labels,
        ratio_values=ratio_values,
        observation_noise_stds=observation_noise_stds,
        mean_returns=mean_returns,
        sem_returns=sem_returns,
        outpath=figure_path,
    )
    plot_covariance_sweep_extremes(
        coverages=coverages,
        transition_covariance_multipliers=transition_covariance_multipliers,
        ratio_labels=ratio_labels,
        mean_returns=mean_returns,
        outpath=extremes_figure_path,
    )
    plot_covariance_sweep_intermediate(
        coverages=coverages,
        transition_covariance_multipliers=transition_covariance_multipliers,
        ratio_labels=ratio_labels,
        mean_returns=mean_returns,
        outpath=intermediate_figure_path,
    )

    np.savez_compressed(
        data_path,
        coverages=np.asarray(coverages, dtype=float),
        covariance_ratio_labels=np.asarray(ratio_labels, dtype=object),
        covariance_ratio_values=ratio_values,
        observation_noise_stds=observation_noise_stds,
        transition_noise_labels=np.asarray(transition_noise_labels, dtype=object),
        transition_covariance_multipliers=transition_covariance_multipliers,
        transition_reference_variances=transition_reference_variances,
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
        heavy_tail_degrees_of_freedom=np.asarray([heavy_tail_degrees_of_freedom], dtype=float),
        uniform_annulus_min_coverage=np.asarray([uniform_annulus_min_coverage], dtype=float),
        gamma=np.asarray([gamma], dtype=float),
        base_process_position_std=np.asarray([process_position_std], dtype=float),
        base_process_wind_std=np.asarray([process_wind_std], dtype=float),
        process_position_stds=process_position_stds,
        process_wind_stds=process_wind_stds,
        contour_relative_tolerance=np.asarray([contour_relative_tolerance], dtype=float),
        **covariance_sweep_attack_records_to_arrays(attack_value_records),
    )

    print(f"\nSaved covariance sweep figure to: {figure_path}")
    print(f"Saved extreme-case figure to:    {extremes_figure_path}")
    print(f"Saved intermediate figure to:    {intermediate_figure_path}")
    print(f"Saved covariance sweep data to:   {data_path}")
    return {
        "figure_path": figure_path,
        "extremes_figure_path": extremes_figure_path,
        "intermediate_figure_path": intermediate_figure_path,
        "data_path": data_path,
        "mean_returns": mean_returns,
        "success_rates": success_rates,
        "observation_noise_stds": observation_noise_stds,
        "transition_reference_variances": transition_reference_variances,
    }


def main() -> None:
    """Configure and run the three-ratio covariance sweep benchmark."""
    n_episodes = 100
    coverages = (0.75,)
    covariance_ratio_specs = (
        ("1:4", 0.25),
        ("1:1", 1.0),
        ("4:1", 4.0),
    )
    transition_noise_specs = (
        ("base W", 1.0),
        ("2x W", 2.0),
        ("4x W", 4.0),
    )
    attack_prob = 0.10
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
    base_output_stem = "covaraicne_sweep"
    output_stem = output_stem_with_coverages(
        base_output_stem=base_output_stem,
        coverages=coverages,
    )

    run_covariance_sweep(
        n_episodes=int(n_episodes),
        coverages=coverages,
        covariance_ratio_specs=covariance_ratio_specs,
        transition_noise_specs=transition_noise_specs,
        attack_prob=float(attack_prob),
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
