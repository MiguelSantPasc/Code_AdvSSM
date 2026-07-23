#!/usr/bin/env python3
"""
RL_covadapt_compare_epsilons.py

Compare two attack-probability settings for the RL covariance-adaptation experiment
inside one figure.

What this script does:
1. Reuse the same 4D RL attack / defense pipeline already implemented in
   `CovarianceAdaptation/RL_covadapt.py`.
2. Evaluate the same bar-style summaries for two different attack
   probabilities, by default `epsilon = 0.75` and `epsilon = 0.95`.
3. Plot the same three panels as the single-epsilon script:
   - baseline methods,
   - attacked-case methods,
   - epsilon-perturbation methods.
4. Show the final accumulated reward standardized by the number of episodes,
   i.e. the mean accumulated reward per episode.

Design choices:
- The method colors remain the main identifier so each family stays visually
  stable across panels.
- The epsilon values are encoded through bar hatches and a dedicated epsilon
  legend, which keeps the method legend readable even when two epsilon values
  are compared at once.
- The horizontal zero line is kept because negative mean reward is meaningful
  in this environment and should be visually explicit.
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from AdvSSM.io_utils import cached_npz, data_path_for_plot, figures_dir_for, load_npz
    from CovarianceAdaptation.RL_covadapt import (
        RL_4D_DIR,
        compute_accumulated_reward_data,
        rl_mod,
        set_plot_theme,
        style_axis,
    )
except ModuleNotFoundError:
    from io_utils import cached_npz, data_path_for_plot, figures_dir_for, load_npz
    from RL_covadapt import (
        RL_4D_DIR,
        compute_accumulated_reward_data,
        rl_mod,
        set_plot_theme,
        style_axis,
    )


def normalized_final_reward(series: np.ndarray, *, n_episodes: int) -> float:
    """
    Return the final accumulated reward standardized by the episode count.

    The underlying experiment stores cumulative reward along episodes. For this
    comparison we only need the final value, divided by the number of episodes,
    so the bars can be interpreted as mean reward per episode.
    """
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return 0.0
    return float(series[-1] / float(n_episodes))


def darken_hex(hex_color: str, factor: float = 0.88) -> tuple[float, float, float]:
    """Return a slightly darker RGB color for bar edges and legend patches."""
    raw = hex_color.lstrip("#")
    rgb = tuple(int(raw[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))
    return tuple(max(0.0, min(1.0, factor * channel)) for channel in rgb)


def find_existing_single_epsilon_cache(
    *,
    data_dir: str,
    attack_prob: float,
    attack_eps: float,
    c_tag: str,
    preferred_n_episodes: int,
) -> str | None:
    """
    Return the best existing single-epsilon cache path if one is already saved.

    Preference order:
    1. exact episode count match,
    2. otherwise the largest available episode count.

    This lets the comparison script reuse earlier runs even when the current
    requested `N` differs, which is safe because the plotted metric is later
    normalized by each cache's own stored episode count.
    """
    eps_tag = str(attack_eps).replace(".", "p")
    prob_tag = f"{attack_prob:g}"
    pattern = re.compile(
        rf"^comparison_RL_v2_wind_4dattack_N(?P<n>\d+)_p{re.escape(prob_tag)}_eps{re.escape(eps_tag)}_c{re.escape(c_tag)}\.npz$"
    )

    candidates: list[tuple[int, str]] = []
    if not os.path.isdir(data_dir):
        return None

    for filename in os.listdir(data_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        candidates.append((int(match.group("n")), os.path.join(data_dir, filename)))

    if not candidates:
        return None

    exact_matches = [path for n_value, path in candidates if n_value == int(preferred_n_episodes)]
    if exact_matches:
        return exact_matches[0]

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_panel_specifications(
    c_scales: tuple[float, ...],
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """
    Build the label/key specifications for the three grouped-bar panels.

    Keeping this mapping centralized makes it easy to guarantee that both
    epsilon runs are compared with the same ordering and labels.
    """
    baseline_spec = [
        ("Noise-Free", "acc_clean"),
        ("Noise + KF", "acc_noisy_kf"),
        ("Attack + KF", "acc_attack_kf"),
        ("Random boundary attack + KF", "acc_random_kf"),
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

    random_spec = [
        ("Noise-Free", "acc_clean"),
        ("Noise + KF", "acc_noisy_kf"),
        (r"Boundary $\epsilon$-perturbation + KF", "acc_random_kf"),
    ]
    random_spec.extend(
        [
            (
                rf"Random boundary attack + cov-adapt ($\lambda={c_scale:g}\lambda_{{\max}}$)",
                f"acc_random_cov_{c_scale:g}",
            )
            for c_scale in c_scales
        ]
    )
    return baseline_spec, attack_spec, random_spec


def plot_accumulated_reward_comparison_two_epsilons(
    *,
    data_by_epsilon: dict[float, dict[str, np.ndarray | float | int]],
    epsilon_values: tuple[float, ...],
    outpath: str,
) -> None:
    """
    Plot the three-panel bar comparison for two attack-radius values.

    Each method appears once per panel, but with two grouped bars, one for each
    epsilon value. Method identity comes from the bar color, while epsilon is
    encoded by hatch and the small epsilon legend.
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
    }
    attack_cov_colors = {
        f"acc_attack_cov_{c_scale:g}": color
        for c_scale, color in zip(c_scales, ["#F3D0B7", "#E7B28A", "#D89269"], strict=False)
    }
    random_cov_colors = {
        f"acc_random_cov_{c_scale:g}": color
        for c_scale, color in zip(c_scales, ["#CAD9F2", "#ADC4E9", "#8EACDB"], strict=False)
    }
    method_colors.update(attack_cov_colors)
    method_colors.update(random_cov_colors)

    hatch_by_epsilon = {
        float(epsilon_values[0]): "",
        float(epsilon_values[1]): "//////" if len(epsilon_values) > 1 else "",
    }
    epsilon_invariant_keys = {"acc_clean", "acc_noisy_kf"}

    def panel_values(spec: list[tuple[str, str]]) -> dict[float, list[float]]:
        """Return the normalized final values for every epsilon in one panel."""
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
    if y_min > -0.28 * value_abs_max:
        y_min = -0.28 * value_abs_max

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.4, 6.1),
        sharey=True,
        constrained_layout=True,
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

        We intentionally hide the x-axis words because the method legend is a
        cleaner mapping for the bar colors, especially once every method is
        split into two epsilon-specific bars.
        """
        labels = [label for label, _key in spec]
        keys = [key for _label, key in spec]
        x = np.arange(len(labels), dtype=float)
        if x.size >= 2:
            # Pull the first two baseline-reference columns closer together
            # because they are the paired non-adversarial references.
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

        # Methods that do not depend on epsilon are drawn once at the group
        # center so the figure does not suggest a non-existent epsilon effect.
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
                    fontsize=8.7,
                    color="#2F2F2F",
                )

        for eps_idx, epsilon in enumerate(epsilon_values):
            varying_indices = [idx for idx, key in key_by_index.items() if key not in epsilon_invariant_keys]
            if not varying_indices:
                continue

            bar_positions = x[varying_indices] + offsets[eps_idx]
            panel_values_eps = np.asarray(values_by_epsilon[float(epsilon)], dtype=float)
            panel_values_eps = panel_values_eps[varying_indices]
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
                    fontsize=8.7,
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
            borderpad = 0.90
            handlelength = 3.4
            handletextpad = 1.10
            labelspacing = 0.70
        elif legend_width == "medium":
            borderpad = 0.66
            handlelength = 2.8
            handletextpad = 0.92
            labelspacing = 0.58
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
            fontsize=9.6,
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
            fontsize=9.2,
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
        legend_width="medium",
        legend_loc="lower right",
    )
    draw_grouped_bar_panel(
        ax=ax_random,
        spec=random_spec,
        values_by_epsilon=random_values,
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
    Generate the two-epsilon RL covariance-adaptation comparison figure.

    Change the defaults below directly when you want a lighter validation run
    or a larger experiment. As in the rest of this folder, the configuration
    is kept in plain Python variables instead of environment variables.
    """
    model_path = rl_mod.default_policy_model_path()
    device = "cpu"
    noise_std = 0.6
    attack_region_radius = 2.488
    attack_prob_values = (0.75, 0.95)
    kf_meas_std = noise_std
    kf_proc_std = 0.03
    pgd_steps = 120
    pgd_step_size = 0.35
    mc_samples = 256
    n_episodes = 100
    seed0 = 1_000

    c_scales = (0.5, 1.0, 2.0)
    omega_h = 0.50
    omega_o = 0.50
    delta_threshold = 0.20
    # Keep cache reuse enabled so repeated runs load the saved epsilon-specific
    # experiment outputs instead of recomputing them.
    force_cache = True

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at:\n  {model_path}")

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    c_tag = "-".join(f"{value:g}" for value in c_scales).replace(".", "p")
    eps_tag = "-".join(str(value).replace(".", "p") for value in attack_prob_values)
    radius_tag = str(attack_region_radius).replace(".", "p")
    outpath = os.path.join(
        out_dir,
        (
            "comparison_RL_v2_wind_4dattack_two_eps_"
            f"N{n_episodes}_eps{eps_tag}_rad{radius_tag}_c{c_tag}.png"
        ),
    )

    data_by_epsilon: dict[float, dict[str, np.ndarray | float | int]] = {}
    data_dir = os.path.dirname(data_path_for_plot(outpath))
    for attack_prob in attack_prob_values:
        existing_cache_path = find_existing_single_epsilon_cache(
            data_dir=data_dir,
            attack_prob=float(attack_prob),
            attack_eps=float(attack_region_radius),
            c_tag=c_tag,
            preferred_n_episodes=n_episodes,
        )
        if existing_cache_path is not None and not force_cache:
            print(f"[cache] loading compatible data: {existing_cache_path}")
            data_by_epsilon[float(attack_prob)] = load_npz(existing_cache_path)
            continue

        # Fall back to the exact single-epsilon cache name when no compatible
        # cache already exists.
        single_eps_outpath = os.path.join(
            out_dir,
            (
                "comparison_RL_v2_wind_4dattack_"
                f"N{n_episodes}_eps{str(attack_prob).replace('.', 'p')}_rad{radius_tag}_c{c_tag}.png"
            ),
        )
        data_path = data_path_for_plot(single_eps_outpath)

        def compute_data_for_epsilon(attack_prob_value: float = float(attack_prob)) -> dict[str, np.ndarray | float | int]:
            return compute_accumulated_reward_data(
                n_episodes=n_episodes,
                seed0=seed0,
                model_path=model_path,
                noise_std=noise_std,
                attack_eps=attack_region_radius,
                attack_prob=attack_prob_value,
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

        data_by_epsilon[float(attack_prob)] = cached_npz(
            data_path,
            compute_data_for_epsilon,
            force=force_cache,
        )

    plot_accumulated_reward_comparison_two_epsilons(
        data_by_epsilon=data_by_epsilon,
        epsilon_values=tuple(float(value) for value in attack_prob_values),
        outpath=outpath,
    )
    print(f"Saved figure to: {outpath}")


if __name__ == "__main__":
    main()
