#!/usr/bin/env python3
"""
plot_wind_value_function.py

Render the critic value function over the navigation plane with a viridis map.

Why this script exists:
1. The user asked for a dedicated script that plots the learned RL value
   function after PPO training.
2. The critic depends on the transformed policy input, but the script should
   be easy to read from the hidden-state point of view, so we evaluate the
   critic on a position grid while fixing the wind slice.
3. The output is saved in `RL/figures/` and uses a viridis palette as
   requested.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from wind_rl_setup import default_device
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_figures_dir
from wind_rl_setup import rl_model_dir


def set_plot_theme() -> None:
    """Apply a clean plot style consistent with the rest of the repository."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.70,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    """Apply the shared axis styling without adding subplot titles."""
    ax.set_facecolor("#FBFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.22, linewidth=0.70)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.45)
    ax.set_axisbelow(True)


def main() -> None:
    """Load the saved critic and plot one position-space slice of `V`."""
    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, env_config, _net_config, _train_config, _history = load_agent_checkpoint(model_path, device)

    set_plot_theme()
    rng = np.random.default_rng()

    # Draw one random wind direction and one random goal from the same annulus
    # used by the environment so each plot call shows a different scenario.
    wind_angle_for_plot = rng.uniform(0.0, 2.0 * np.pi)
    wind_dx = float(env_config.initial_wind_magnitude * np.cos(wind_angle_for_plot))
    wind_dy = float(env_config.initial_wind_magnitude * np.sin(wind_angle_for_plot))
    goal_radius_for_plot = float(
        np.sqrt(
            rng.uniform(
                float(env_config.goal_distance_min) ** 2,
                float(env_config.goal_distance_max) ** 2,
            )
        )
    )
    goal_angle_for_plot = float(rng.uniform(0.0, 2.0 * np.pi))
    goal_x = float(goal_radius_for_plot * np.cos(goal_angle_for_plot))
    goal_y = float(goal_radius_for_plot * np.sin(goal_angle_for_plot))

    grid_size = 181
    x_grid = np.linspace(-env_config.radius_max, env_config.radius_max, grid_size, dtype=np.float32)
    y_grid = np.linspace(-env_config.radius_max, env_config.radius_max, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    observation = np.zeros((grid_size * grid_size, 6), dtype=np.float32)
    observation[:, 0] = grid_x.reshape(-1)
    observation[:, 1] = grid_y.reshape(-1)
    observation[:, 2] = wind_dx
    observation[:, 3] = wind_dy
    observation[:, 4] = goal_x
    observation[:, 5] = goal_y

    with torch.no_grad():
        values = policy.value(torch.as_tensor(observation, dtype=torch.float32, device=device))
        value_grid = values.cpu().numpy().reshape(grid_size, grid_size)

    domain_mask = grid_x**2 + grid_y**2 <= env_config.radius_max**2
    value_grid = np.where(domain_mask, value_grid, np.nan)

    fig, ax = plt.subplots(figsize=(7.1, 6.2))
    style_axis(ax)

    image = ax.imshow(
        value_grid,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        cmap="viridis",
        interpolation="bilinear",
        aspect="equal",
    )

    goal_circle = plt.Circle(
        (goal_x, goal_y),
        env_config.goal_radius,
        facecolor="none",
        edgecolor="white",
        linewidth=1.6,
        linestyle="--",
        label="goal",
    )
    goal_annulus_inner = plt.Circle(
        (0.0, 0.0),
        env_config.goal_distance_min,
        facecolor="none",
        edgecolor="#BFD3DE",
        linewidth=1.0,
        linestyle=":",
        label="goal annulus",
    )
    goal_annulus_outer = plt.Circle(
        (0.0, 0.0),
        env_config.goal_distance_max,
        facecolor="none",
        edgecolor="#BFD3DE",
        linewidth=1.0,
        linestyle=":",
    )
    boundary_circle = plt.Circle(
        (0.0, 0.0),
        env_config.radius_max,
        facecolor="none",
        edgecolor="#D8E2EA",
        linewidth=1.1,
        label=r"$R_{\max}$",
    )

    ax.add_patch(boundary_circle)
    ax.add_patch(goal_annulus_inner)
    ax.add_patch(goal_annulus_outer)
    ax.add_patch(goal_circle)
    ax.scatter(
        [goal_x],
        [goal_y],
        s=60,
        c=["#FDE725"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        [env_config.start_xy[0]],
        [env_config.start_xy[1]],
        s=58,
        c=["#F4A261"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        label="start",
    )

    quiver_x = np.linspace(-4.8, 4.8, 6, dtype=np.float32)
    quiver_y = np.linspace(-4.8, 4.8, 6, dtype=np.float32)
    quiver_grid_x, quiver_grid_y = np.meshgrid(quiver_x, quiver_y)
    quiver_mask = quiver_grid_x**2 + quiver_grid_y**2 <= (env_config.radius_max - 0.4) ** 2
    ax.quiver(
        quiver_grid_x[quiver_mask],
        quiver_grid_y[quiver_mask],
        np.full(np.count_nonzero(quiver_mask), wind_dx, dtype=np.float32),
        np.full(np.count_nonzero(quiver_mask), wind_dy, dtype=np.float32),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
        color="#D7F3FF",
        alpha=0.75,
        zorder=2,
    )

    ax.annotate(
        "",
        xy=(env_config.start_xy[0] + 5.0 * wind_dx, env_config.start_xy[1] + 5.0 * wind_dy),
        xytext=(env_config.start_xy[0], env_config.start_xy[1]),
        arrowprops={"arrowstyle": "-|>", "lw": 2.1, "color": "#D7F3FF"},
        zorder=4,
    )
    ax.text(
        env_config.start_xy[0] + 5.2 * wind_dx,
        env_config.start_xy[1] + 5.2 * wind_dy,
        "wind",
        color="#F6FBFF",
        fontsize=8.9,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#2F4858", "alpha": 0.82, "edgecolor": "none"},
        zorder=5,
    )

    ax.set_xlabel(r"$p_{x,t}$")
    ax.set_ylabel(r"$p_{y,t}$")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label(r"$V(s_t)$")

    ax.text(
        0.98,
        0.04,
        rf"$g=({goal_x:.2f},{goal_y:.2f})$"
        + "\n"
        + rf"$\|d_t\|={env_config.initial_wind_magnitude:.2f},\ d_{{x,t}}={wind_dx:.2f},\ d_{{y,t}}={wind_dy:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#2F4858", "alpha": 0.82, "edgecolor": "none"},
    )

    fig.tight_layout()
    figure_path = os.path.join(rl_figures_dir(), "wind_value_function_viridis.png")
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved value-function figure to: {figure_path}")


if __name__ == "__main__":
    main()
