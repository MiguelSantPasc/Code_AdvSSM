#!/usr/bin/env python3
"""
visualize_cartpole_value_slices.py

Visualize the learned value landscape of the pretrained CartPole DQN used by
the Gymnasium benchmark.

What this script shows:
1. The DQN does not output a true standalone `V(s)` head. Instead it outputs
   action values `Q(s, 0)` and `Q(s, 1)`.
2. For debugging the adversarial benchmark, the most natural value proxy is

       V(s) = max_a Q(s, a),

   because the deterministic policy also acts greedily with respect to those
   two action values.
3. It is often even more informative to inspect the action gap

       gap(s) = Q(s, 1) - Q(s, 0),

   because the sign of that gap tells us which action the greedy policy picks,
   and a very small gap means the policy is locally fragile.

State convention used by Gymnasium `CartPole-v1`:
    s_t = [x_t, xdot_t, theta_t, thetadot_t]

Figures produced:
1. A 2x3 heatmap figure for `V(s)` over the `(x, theta)` plane while fixing
   several `(xdot, thetadot)` slices.
2. A matching 2x3 heatmap figure for the action gap `Q(right) - Q(left)`.
3. A 1D line-cut figure along the stable center line `x = 0, xdot = 0,
   thetadot = 0`, varying only `theta`, which is useful for checking whether
   the value and the action preference change smoothly near the upright region.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from AdvSSM.io_utils import cached_npz, data_path_for_plot, figures_dir_for
from CovarianceAdaptation.covariance_adaptation_utils import set_plot_theme, style_axis
import cartpole_covadapt_compare_epsilons_wolf as cartpole_mod


@torch.no_grad()
def dqn_q_values(model, obs_batch: torch.Tensor) -> torch.Tensor:
    """
    Return the two DQN action values for a batch of CartPole states.

    Output shape is `(N, 2)` with columns:
    - column 0: `Q(s, left)`
    - column 1: `Q(s, right)`
    """
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.unsqueeze(0)
    return model.q_net(obs_batch)


def build_slice_specifications() -> list[tuple[str, float, float]]:
    """
    Return a small family of `(xdot, thetadot)` slices for the 2D heatmaps.

    These values are intentionally moderate: large enough to reveal structure,
    but still close enough to the regime where the trained CartPole policy is
    expected to spend meaningful probability mass.
    """
    return [
        ("xdot=0.00, thetadot=0.00", 0.00, 0.00),
        ("xdot=0.50, thetadot=0.00", 0.50, 0.00),
        ("xdot=-0.50, thetadot=0.00", -0.50, 0.00),
        ("xdot=0.00, thetadot=0.50", 0.00, 0.50),
        ("xdot=0.00, thetadot=-0.50", 0.00, -0.50),
        ("xdot=0.50, thetadot=0.50", 0.50, 0.50),
    ]


@torch.no_grad()
def evaluate_q_slice(
    *,
    model,
    x_lim: float,
    theta_lim: float,
    grid_n: int,
    xdot_fixed: float,
    thetadot_fixed: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate `Q(s, left)` and `Q(s, right)` on an `(x, theta)` grid slice.
    """
    xs = np.linspace(-x_lim, x_lim, int(grid_n), dtype=np.float32)
    thetas = np.linspace(-theta_lim, theta_lim, int(grid_n), dtype=np.float32)
    X, THETA = np.meshgrid(xs, thetas)

    states = np.stack(
        [
            X.reshape(-1),
            np.full(X.size, float(xdot_fixed), dtype=np.float32),
            THETA.reshape(-1),
            np.full(X.size, float(thetadot_fixed), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    q_left = np.zeros((states.shape[0],), dtype=np.float32)
    q_right = np.zeros((states.shape[0],), dtype=np.float32)

    for start in range(0, states.shape[0], int(batch_size)):
        stop = start + int(batch_size)
        batch = torch.tensor(states[start:stop], dtype=torch.float32, device=device)
        q_batch = dqn_q_values(model, batch).detach().cpu().numpy()
        q_left[start:stop] = q_batch[:, 0]
        q_right[start:stop] = q_batch[:, 1]

    return (
        X,
        THETA,
        q_left.reshape(int(grid_n), int(grid_n)),
        q_right.reshape(int(grid_n), int(grid_n)),
    )


def plot_heatmap_panels(
    *,
    x_grid: np.ndarray,
    theta_grid: np.ndarray,
    panel_values: list[np.ndarray],
    panel_labels: list[str],
    colorbar_label: str,
    cmap: str,
    outpath: str,
) -> None:
    """
    Save one 2x3 heatmap figure with a shared color scale across all slices.
    """
    set_plot_theme()

    stacked = np.stack(panel_values, axis=0)
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8), constrained_layout=True, sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, values, label in zip(axes_flat, panel_values, panel_labels, strict=False):
        style_axis(ax)
        image = ax.imshow(
            values,
            origin="lower",
            extent=[
                float(x_grid.min()),
                float(x_grid.max()),
                float(theta_grid.min()),
                float(theta_grid.max()),
            ],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        contours = ax.contour(
            x_grid,
            theta_grid,
            values,
            levels=8,
            colors="white",
            linewidths=0.60,
            alpha=0.72,
        )
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.1f")
        ax.axvline(0.0, color="#2F2F2F", linewidth=0.9, alpha=0.55)
        ax.axhline(0.0, color="#2F2F2F", linewidth=0.9, alpha=0.55)
        ax.text(
            0.03,
            0.97,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            color="#243B53",
            bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=2.6),
        )

    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    axes[1, 2].set_xlabel("x")
    axes[0, 0].set_ylabel("theta")
    axes[1, 0].set_ylabel("theta")

    colorbar = fig.colorbar(image, ax=axes_flat.tolist(), shrink=0.95, pad=0.015)
    colorbar.set_label(colorbar_label)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


def plot_theta_line_cut(
    *,
    theta_axis: np.ndarray,
    value_line: np.ndarray,
    gap_line: np.ndarray,
    q_left_line: np.ndarray,
    q_right_line: np.ndarray,
    outpath: str,
) -> None:
    """
    Save a compact line-cut figure through the most interpretable central slice.
    """
    set_plot_theme()

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.2), constrained_layout=True, sharex=True)
    ax_top, ax_bottom = axes

    style_axis(ax_top)
    ax_top.plot(theta_axis, value_line, color="#6F92D8", linewidth=2.2, label=r"$V(s)=\max_a Q(s,a)$")
    ax_top.plot(theta_axis, q_left_line, color="#D88C9A", linewidth=1.8, alpha=0.92, label=r"$Q(s,\mathrm{left})$")
    ax_top.plot(theta_axis, q_right_line, color="#8AB17D", linewidth=1.8, alpha=0.92, label=r"$Q(s,\mathrm{right})$")
    ax_top.axvline(0.0, color="#2F2F2F", linewidth=0.9, alpha=0.55)
    ax_top.set_ylabel("value")
    ax_top.legend(loc="best", frameon=True, framealpha=0.94)

    style_axis(ax_bottom)
    ax_bottom.plot(theta_axis, gap_line, color="#C96F5C", linewidth=2.1, label=r"$Q(s,\mathrm{right})-Q(s,\mathrm{left})$")
    ax_bottom.axhline(0.0, color="#2F2F2F", linewidth=0.9, alpha=0.55)
    ax_bottom.axvline(0.0, color="#2F2F2F", linewidth=0.9, alpha=0.55)
    ax_bottom.set_xlabel(r"$\theta$ with $x=0$, $\dot{x}=0$, $\dot{\theta}=0$")
    ax_bottom.set_ylabel("action gap")
    ax_bottom.legend(loc="best", frameon=True, framealpha=0.94)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    """
    Build and save CartPole value-function slices from the pretrained DQN.
    """
    model_path = cartpole_mod.ensure_downloaded_cartpole_checkpoint()
    device = torch.device("cpu")
    x_lim = 2.4
    theta_lim = 0.25
    grid_n = 220
    batch_size = 16_384
    force_cache = False

    model = cartpole_mod.load_cartpole_policy(model_path, device)
    slice_specs = build_slice_specifications()

    figures_dir = figures_dir_for(CURRENT_DIR)
    value_outpath = os.path.join(figures_dir, "cartpole_value_slices_x_theta.png")
    gap_outpath = os.path.join(figures_dir, "cartpole_action_gap_slices_x_theta.png")
    line_outpath = os.path.join(figures_dir, "cartpole_theta_line_cut.png")
    data_path = data_path_for_plot(value_outpath)

    def compute_data() -> dict[str, np.ndarray]:
        value_panels: list[np.ndarray] = []
        gap_panels: list[np.ndarray] = []
        q_left_panels: list[np.ndarray] = []
        q_right_panels: list[np.ndarray] = []
        panel_labels: list[str] = []
        x_grid = None
        theta_grid = None

        for label, xdot_fixed, thetadot_fixed in slice_specs:
            X, THETA, q_left, q_right = evaluate_q_slice(
                model=model,
                x_lim=x_lim,
                theta_lim=theta_lim,
                grid_n=grid_n,
                xdot_fixed=xdot_fixed,
                thetadot_fixed=thetadot_fixed,
                batch_size=batch_size,
                device=device,
            )
            x_grid = X
            theta_grid = THETA
            q_left_panels.append(q_left.astype(np.float32))
            q_right_panels.append(q_right.astype(np.float32))
            value_panels.append(np.maximum(q_left, q_right).astype(np.float32))
            gap_panels.append((q_right - q_left).astype(np.float32))
            panel_labels.append(label)

            print(
                f"[cartpole-slice] {label} | "
                f"V_min={float(np.min(np.maximum(q_left, q_right))):.3f} | "
                f"V_max={float(np.max(np.maximum(q_left, q_right))):.3f} | "
                f"gap_min={float(np.min(q_right - q_left)):.3f} | "
                f"gap_max={float(np.max(q_right - q_left)):.3f}"
            )

        return {
            "x_grid": np.asarray(x_grid, dtype=np.float32),
            "theta_grid": np.asarray(theta_grid, dtype=np.float32),
            "value_panels": np.stack(value_panels, axis=0).astype(np.float32),
            "gap_panels": np.stack(gap_panels, axis=0).astype(np.float32),
            "q_left_panels": np.stack(q_left_panels, axis=0).astype(np.float32),
            "q_right_panels": np.stack(q_right_panels, axis=0).astype(np.float32),
            "panel_labels": np.asarray(panel_labels, dtype=object),
        }

    data = cached_npz(data_path, compute_data, force=force_cache)
    x_grid = np.asarray(data["x_grid"], dtype=float)
    theta_grid = np.asarray(data["theta_grid"], dtype=float)
    value_panels = [np.asarray(values, dtype=float) for values in np.asarray(data["value_panels"], dtype=float)]
    gap_panels = [np.asarray(values, dtype=float) for values in np.asarray(data["gap_panels"], dtype=float)]
    q_left_panels = [np.asarray(values, dtype=float) for values in np.asarray(data["q_left_panels"], dtype=float)]
    q_right_panels = [np.asarray(values, dtype=float) for values in np.asarray(data["q_right_panels"], dtype=float)]
    panel_labels = [str(label) for label in np.asarray(data["panel_labels"], dtype=object).tolist()]

    plot_heatmap_panels(
        x_grid=x_grid,
        theta_grid=theta_grid,
        panel_values=value_panels,
        panel_labels=panel_labels,
        colorbar_label=r"$V(s)=\max_a Q(s,a)$",
        cmap="YlGnBu",
        outpath=value_outpath,
    )
    plot_heatmap_panels(
        x_grid=x_grid,
        theta_grid=theta_grid,
        panel_values=gap_panels,
        panel_labels=panel_labels,
        colorbar_label=r"$Q(s,\mathrm{right})-Q(s,\mathrm{left})$",
        cmap="RdBu_r",
        outpath=gap_outpath,
    )

    center_panel_idx = 0
    center_x_col = int(np.argmin(np.abs(np.asarray(x_grid[0], dtype=float))))
    theta_axis = np.asarray(theta_grid[:, center_x_col], dtype=float)
    value_line = np.asarray(value_panels[center_panel_idx][:, center_x_col], dtype=float)
    gap_line = np.asarray(gap_panels[center_panel_idx][:, center_x_col], dtype=float)
    q_left_line = np.asarray(q_left_panels[center_panel_idx][:, center_x_col], dtype=float)
    q_right_line = np.asarray(q_right_panels[center_panel_idx][:, center_x_col], dtype=float)

    plot_theta_line_cut(
        theta_axis=theta_axis,
        value_line=value_line,
        gap_line=gap_line,
        q_left_line=q_left_line,
        q_right_line=q_right_line,
        outpath=line_outpath,
    )

    print(f"Saved value slices to: {value_outpath}")
    print(f"Saved action-gap slices to: {gap_outpath}")
    print(f"Saved theta line cut to: {line_outpath}")


if __name__ == "__main__":
    main()
