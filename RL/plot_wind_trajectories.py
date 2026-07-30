#!/usr/bin/env python3
"""
plot_wind_trajectories.py

Plot six deterministic wind-navigation trajectories in a `2 x 3` layout.

Why this script exists:
1. The user asked for a dedicated figure with six trajectories arranged on a
   `2 x 3` grid.
2. Each subplot should show how the trained policy steers the agent from its
   initial position toward the goal under a different wind realization.
3. The figure is saved in `RL/figures/` and reuses the PPO checkpoint stored
   in `RL/model/`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from wind_rl_setup import default_device
from wind_rl_setup import evaluate_policy
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_figures_dir
from wind_rl_setup import rl_model_dir


def set_plot_theme() -> None:
    """Apply the shared plotting style used by the RL figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9.8,
            "axes.labelsize": 10.7,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.68,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    """Apply axis styling while keeping legends inside the subplot."""
    ax.set_facecolor("#FCFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.22, linewidth=0.68)
    ax.grid(True, which="minor", alpha=0.09, linewidth=0.42)
    ax.set_axisbelow(True)


def main() -> None:
    """Load the saved policy and plot six deterministic evaluation episodes."""
    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, env_config, _net_config, _train_config, _history = load_agent_checkpoint(model_path, device)

    set_plot_theme()

    # Online posterior attack used for the wind-policy robustness figure.
    attack_prob = 0.1
    attack_epsilon = 1.0
    attack_step_size = 0.05
    attack_num_steps = 20
    attack_mc_samples = 16

    trajectories = [
        evaluate_policy(
            policy=policy,
            env_config=env_config,
            seed=None,
            device=device,
            attack_prob=attack_prob,
            attack_epsilon=attack_epsilon,
            attack_step_size=attack_step_size,
            attack_num_steps=attack_num_steps,
            attack_mc_samples=attack_mc_samples,
        )
        for _ in range(6)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), sharex=True, sharey=True)
    axes_flat = axes.reshape(-1)

    path_color = "#7BC8A4"
    wind_color = "#6C91BF"
    action_color = "#D66A4E"
    start_color = "#F4A261"
    goal_color = "#E76F51"
    attack_color = "#C77DA4"

    for panel_idx, (ax, trajectory) in enumerate(zip(axes_flat, trajectories), start=1):
        style_axis(ax)

        positions = trajectory["positions"]
        winds = trajectory["winds"]
        actions = trajectory["actions"]
        goal_xy = trajectory["goal_xy"]
        wind_skip = max(1, len(winds) // 8)
        action_skip = max(1, len(actions) // 8)

        boundary_circle = plt.Circle(
            (0.0, 0.0),
            env_config.radius_max,
            facecolor="none",
            edgecolor="#D9E3EB",
            linewidth=1.0,
        )
        goal_circle = plt.Circle(
            goal_xy,
            env_config.goal_radius,
            facecolor="none",
            edgecolor=goal_color,
            linewidth=1.2,
            linestyle="--",
        )
        ax.add_patch(boundary_circle)
        ax.add_patch(goal_circle)

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color=path_color,
            linewidth=2.0,
            alpha=0.95,
            label="trayectoria",
        )
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            s=42,
            color=start_color,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
            label="inicio",
        )
        ax.scatter(
            goal_xy[0],
            goal_xy[1],
            s=44,
            color=goal_color,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
            label="goal",
        )

        ax.quiver(
            positions[:-1:wind_skip, 0],
            positions[:-1:wind_skip, 1],
            winds[:-1:wind_skip, 0],
            winds[:-1:wind_skip, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            color=wind_color,
            alpha=0.72,
            label="viento",
        )
        ax.quiver(
            positions[:-1:action_skip, 0],
            positions[:-1:action_skip, 1],
            actions[::action_skip, 0],
            actions[::action_skip, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            color=action_color,
            alpha=0.78,
            label="accion",
        )

        attack_flags = np.asarray(trajectory["attack_flags"], dtype=bool)
        attack_positions = positions[:-1][attack_flags]
        if attack_positions.size > 0:
            ax.scatter(
                attack_positions[:, 0],
                attack_positions[:, 1],
                s=30,
                marker="x",
                color=attack_color,
                linewidths=1.3,
                zorder=4,
                label="ataque",
            )

        outcome_text = "goal" if trajectory["reached_goal"] else "timeout"
        ax.text(
            0.03,
            0.97,
            f"run {panel_idx} | {outcome_text}\n"
            f"T={trajectory['length']} | G={trajectory['return']:.1f} | A={int(attack_flags.sum())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color="#31424F",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.88, "edgecolor": "#D8E0E6"},
        )

        ax.set_xlim(-env_config.radius_max, env_config.radius_max)
        ax.set_ylim(-env_config.radius_max, env_config.radius_max)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", frameon=True, framealpha=0.90)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$p_{x,t}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p_{y,t}$")

    fig.tight_layout()
    attack_tag = str(attack_prob).replace(".", "p")
    figure_path = os.path.join(rl_figures_dir(), f"wind_navigation_trajectories_2x3_attack_p{attack_tag}.png")
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory figure to: {figure_path}")


if __name__ == "__main__":
    main()
