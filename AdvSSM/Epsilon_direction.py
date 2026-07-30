#!/usr/bin/env python3
"""
Observation-space geometry for multiple epsilon values in a linear Gaussian
state-space model.

Notation used throughout this file:
    s_t      : hidden state
    o_t      : observation
    \hat{o}_t: predicted / leave-one-out observation used as the reference point

The figure shows, for several epsilon values:
- constraint ellipses centred at \hat{o}_t
- objective level-set ellipses built from directions relative to \hat{o}_t
- tangent points o^*(epsilon)
- a direction arrow that starts at the predicted observation, not at o_t
"""

from __future__ import annotations

import os
import sys
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared_ssm.artifacts import cached_npz
from shared_ssm.artifacts import data_path_for_plot
from shared_ssm.artifacts import figures_dir_for


# ============================================================
# Geometry helpers
# ============================================================
def _ellipse_points_from_quad(
    center: np.ndarray,
    shape_inv: np.ndarray,
    level: float,
    n: int = 400,
) -> np.ndarray:
    center = np.asarray(center, dtype=float).reshape(2,)
    M = project_to_psd(symmetrize(np.asarray(shape_inv, dtype=float).reshape(2, 2)))

    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-14)
    radii = np.sqrt(level / w)

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    pts = (V @ (radii[:, None] * circle)).T + center[None, :]
    return pts


def _set_plot_theme() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
    })


def _style_axis(ax) -> None:
    ax.set_facecolor("#FCFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.50)
    ax.spines["bottom"].set_alpha(0.50)
    ax.grid(True, alpha=0.18)


def _points_limits(points: list[np.ndarray], pad_frac: float = 0.12):
    arrs = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        if p.size > 0:
            arrs.append(p[:, :2])

    P = np.vstack(arrs)
    xmin, ymin = np.min(P[:, 0]), np.min(P[:, 1])
    xmax, ymax = np.max(P[:, 0]), np.max(P[:, 1])

    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    d = max(dx, dy)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    pad = pad_frac * d + 1e-6

    return (cx - 0.5 * d - pad, cx + 0.5 * d + pad), (cy - 0.5 * d - pad, cy + 0.5 * d + pad)


# ============================================================
# Plot
# ============================================================
def plot_multi_epsilon_geometry_with_tangency(
    *,
    o_t: np.ndarray,
    o_hat_t: np.ndarray,
    Sigma_t: np.ndarray,
    X_t: np.ndarray,
    epsilons: list[float],
    outpath: str,
    t: int,
) -> None:
    """
    Plot the observation-space geometry using o_t for the realised observation
    and \hat{o}_t for the predicted observation.
    """
    if o_t.shape != (2,) or o_hat_t.shape != (2,):
        raise ValueError("This plot expects n_y=2.")
    if Sigma_t.shape != (2, 2):
        raise ValueError("Sigma_t must be (2,2).")
    if X_t.shape[1] != 2:
        raise ValueError("X_t must be (n_x, 2).")

    _set_plot_theme()
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    _style_axis(ax)

    Sigma_t = project_to_psd(Sigma_t)
    Sigma_inv = inv_psd(Sigma_t)

    # objective quadratic form
    M = project_to_psd(symmetrize(X_t.T @ X_t), eps=1e-10)

    # soft palette
    soft_colors = [
        "#7C8DA6",  # muted blue-gray
        "#9A8C98",  # mauve gray
        "#8FAE9D",  # muted green
        "#B39B7D",  # muted sand
        "#8C7C74",  # warm gray-brown
        "#6F8F8D",  # desaturated teal
    ]

    all_points = [o_t, o_hat_t]

    # base points (not added to automatic legend)
    ax.scatter(
        [o_hat_t[0]], [o_hat_t[1]],
        s=80, marker="o", color="#6C6F7D",
        edgecolor="black", linewidths=0.45, zorder=8
    )
    ax.scatter(
        [o_t[0]], [o_t[1]],
        s=95, marker="x", color="#222222",
        linewidths=2.0, zorder=9
    )

    ax.annotate(
        r"$\hat{o}_t$",
        xy=o_hat_t,
        xytext=(7, 8),
        textcoords="offset points",
        color="#4A4A4A",
    )
    ax.annotate(
        r"$o_t$",
        xy=o_t,
        xytext=(7, -14),
        textcoords="offset points",
        color="#2A2A2A",
    )

    # store handles for epsilon legend
    epsilon_handles = []

    # draw each epsilon
    for i, eps in enumerate(epsilons):
        color = soft_colors[i % len(soft_colors)]

        o_star, obj_star = solve_kkt_max_quadratic_over_ellipsoid(
            X=X_t, o_t=o_t, o_hat_t=o_hat_t, Sigma=Sigma_t, epsilon=eps
        )

        # Constraint ellipse centred at the predicted observation \hat{o}_t.
        pts_constraint = _ellipse_points_from_quad(o_hat_t, Sigma_inv, eps)
        all_points.extend([pts_constraint, o_star])

        ax.plot(
            pts_constraint[:, 0], pts_constraint[:, 1],
            color=color, linewidth=2.0, alpha=0.95, zorder=1
        )

        # Objective level-set centred at the realised observation o_t.
        pts_obj = _ellipse_points_from_quad(o_t, M, max(obj_star, 1e-12))
        all_points.append(pts_obj)

        ax.plot(
            pts_obj[:, 0], pts_obj[:, 1],
            color=color, linewidth=1.5, linestyle="--", alpha=0.90, zorder=2
        )

        # Tangent point on the epsilon boundary.
        ax.scatter(
            [o_star[0]], [o_star[1]],
            s=68, color=color, edgecolor="black", linewidths=0.45, zorder=10
        )

        # Direction now starts at the predicted observation \hat{o}_t.
        ax.annotate(
            "",
            xy=o_star,
            xytext=o_hat_t,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.9,
                color=color,
                mutation_scale=14,
                alpha=0.95,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=6,
        )

        # Dotted underlay to make the direction easier to spot.
        ax.plot(
            [o_hat_t[0], o_star[0]],
            [o_hat_t[1], o_star[1]],
            linestyle=":",
            linewidth=1.2,
            color=color,
            alpha=0.75,
            zorder=5,
        )

        # Tangent-point label for the current epsilon.
        ax.annotate(
            fr"$o^\star_{{{i+1}}}$",
            xy=o_star,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

        constr_val = float((o_star - o_hat_t).T @ Sigma_inv @ (o_star - o_hat_t))
        delta_adv = o_star - o_hat_t
        print(
            f"epsilon={eps:.6f} | constraint={constr_val:.6f} | "
            f"obj={obj_star:.6f} | delta_from_prediction={delta_adv}"
        )

        epsilon_handles.append(
            Line2D([0], [0], color=color, lw=2.2, label=fr"$\epsilon={eps:.3f}$")
        )

    xlim, ylim = _points_limits(all_points, pad_frac=0.14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("o1")
    ax.set_ylabel("o2")

    # Legend 1: meaning of each visual element
    style_handles = [
        Line2D([0], [0], color="#7A7A7A", lw=2.0, linestyle="-",
               label="Constraint ellipse"),
        Line2D([0], [0], color="#7A7A7A", lw=1.5, linestyle="--",
               label="Optimization function level-set"),
        Line2D([0], [0], color="#7A7A7A", lw=0, linestyle="None",
               marker="o", markersize=6,
               markerfacecolor="#6C6F7D", markeredgecolor="black",
               label=r"$\hat{o}_t$"),
        Line2D([0], [0], color="#222222", lw=0, linestyle="None",
               marker="x", markersize=8, markeredgewidth=2.0,
               label=r"$o_t$"),
        Line2D([0], [0], color="#7A7A7A", lw=0, linestyle="None",
               marker="o", markersize=6,
               markerfacecolor="#BBBBBB", markeredgecolor="black",
               label=r"$o^{\text{adv}}$"),
        Line2D([0], [0], color="#7A7A7A", lw=1.8, linestyle="-",
               label=r"Adversarial direction $\hat{o}_t \rightarrow o^{\text{adv}}$"),
    ]

    legend_style = ax.legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        framealpha=0.96,
        fontsize=15.3,
        title_fontsize=9,
        borderpad=0.35,
        labelspacing=0.28,
        handlelength=2.0,
        handletextpad=0.55,
    )

    # Legend 2: epsilon/color map
    legend_eps = ax.legend(
        handles=epsilon_handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        frameon=True,
        framealpha=0.96,
        fontsize=15.3,
        title=r"$\epsilon$ values",
        title_fontsize=9,
        borderpad=0.35,
        labelspacing=0.22,
        handlelength=1.8,
        handletextpad=0.45,
        ncol=1,
    )

    ax.add_artist(legend_style)

    fig.tight_layout()

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, facecolor="white", dpi=300)
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    n_x = n_y = n_u = 2
    T = 10
    seed = 2026
    t = 4
    epsilons = [1.0, 2.0, 3.84, 5.991, 9.210]  # chi-square-style levels for 2D
    force_recompute = False

    A0 = np.array([[0.65, 0.40],
                   [-0.15, 0.70]], dtype=float)

    B0 = np.array([[1.65, 1.40],
                   [-0.15, 0.70]], dtype=float)

    H0 = np.array([[0.65, 0.40],
                   [-0.15, 0.370]], dtype=float)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.3 * np.array([[1.6, -1.40],
                         [0.95, 0.70]], dtype=float)

    R0 = 0.2 * np.array([[0.65, 0.80],
                         [-0.15, 0.70]], dtype=float)

    Q0 = project_to_psd(Q0)
    R0 = project_to_psd(R0)

    dA = np.zeros_like(A0)
    dB = np.zeros_like(B0)
    dH = np.zeros_like(H0)
    dD = np.zeros_like(D0)
    dQ = np.zeros_like(Q0)
    dR = np.zeros_like(R0)

    x0 = np.array([0.5, 0.5], dtype=float)
    m0 = x0.copy()
    P0 = 0.05 * np.eye(n_x)

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    outpath = os.path.join(out_dir, f"multi_epsilon_tangent_geometry_t{t}_T{T}_seed{seed}.png")
    data_path = data_path_for_plot(outpath)

    def compute_plot_data() -> dict[str, np.ndarray]:
        x, y, u, mats = simulate_lgssm_nd(
            A0=A0, B0=B0, H0=H0, D0=D0,
            T=T, seed=seed, x0=x0,
            Q0=Q0, R0=R0,
            dA=dA, dB=dB, dH=dH, dD=dD, dQ=dQ, dR=dR,
            u_low=-0.5, u_high=0.5,
        )

        X_t, mu_t, Sigma_t = loo_values_nd(
            t=t,
            y=y, u=u,
            A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
            Q_t=mats["Q_t"], R_t=mats["R_t"],
            P0=P0, m0=m0,
        )
        return {
            "y_t": y[t].copy(),
            "mu_t": mu_t,
            "Sigma_t": Sigma_t,
            "X_t": X_t,
            "epsilons": np.asarray(epsilons, dtype=float),
        }

    data = cached_npz(data_path, compute_plot_data, force=force_recompute)

    plot_multi_epsilon_geometry_with_tangency(
        o_t=data["y_t"],
        o_hat_t=data["mu_t"],
        Sigma_t=data["Sigma_t"],
        X_t=data["X_t"],
        epsilons=data["epsilons"].astype(float).tolist(),
        outpath=outpath,
        t=t,
    )

    print(f"\nSaved figure to: {outpath}")


import os as _os
import sys as _sys

# Make `shared_ssm` importable when this legacy script is run directly.
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from shared_ssm.linalg import project_to_psd
from shared_ssm.linalg import spd_inverse as inv_psd
from shared_ssm.linalg import sqrtm_psd
from shared_ssm.linalg import symmetrize
from shared_ssm.legacy import loo_values_nd_previous_observation as loo_values_nd
from shared_ssm.legacy import simulate_lgssm_nd_previous_observation as simulate_lgssm_nd
from shared_ssm.legacy import solve_kkt_max_quadratic_over_ellipsoid


if __name__ == "__main__":
    main()
