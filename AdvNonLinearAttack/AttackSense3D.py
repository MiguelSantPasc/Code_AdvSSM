#!/usr/bin/env python3
"""
attack_direction_density_lgssm_3d.py

Runs N independent simulations of a 3D LGSSM, attacks the last time step t=T
via a leave-one-out white-box point attack on E[g(x_T) | y_T', y_-T], and
builds a 3D density-colored scatter of the attack displacements in observation space.

Main output:
- A 3D density plot of attack displacements:
      delta_i = y_T'_i - y_T_i

Notes:
- We attack the last observation y_T, which changes the posterior on x_T.
- The feasible attack region is the ellipsoid:
      (y_T' - mu_T)^T Sigma_T^{-1} (y_T' - mu_T) <= epsilon
  where p(y_T | y_-T) = N(mu_T, Sigma_T).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Global configuration
# ============================================================
N_RUNS = 500
T = 5
ATTACK_T = T

# 95% chi-square threshold for 3D is about 7.8147
EPSILON = 7.814727903251179

# Attack target for scalar g(x) in [0,1]
M_STAR = np.array([1.00], dtype=float)

# Optimization settings
ETA = 1.5
N_STEPS = 800
N_MC_OPT = 128
N_MC_EST = 2000

# Output
FIGURES_DIRNAME = os.path.join("outputs", "figures")
DATA_DIRNAME = os.path.join("outputs", "data")


# ============================================================
# Risk model g(x) in 3D
# ============================================================
def sigmoid(z: float | np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def g_scalar(x_vec: np.ndarray) -> float:
    """
    3D risk / alarm probability.

    Designed so that the (x2, x3) plane dominates clearly:
    - weak dependence on x1
    - strong quadratic and interaction terms on x2, x3
    """
    x_vec = np.asarray(x_vec, dtype=float)
    x1, x2, x3 = x_vec[0], x_vec[1], x_vec[2]

    beta0 = -3.2

    # x1 has small influence
    beta1 = 0.15

    # x2/x3 plane dominates
    beta1 = 0.05
    beta2 = 1.60
    beta3 = 1.60
    beta23 = 3.50
    beta22 = 3.00
    beta33 = 3.00

    # tiny couplings with x1
    beta12 = 0.10
    beta13 = 0.08

    z = (
        beta0
        + beta1 * x1
        + beta2 * x2
        + beta3 * x3
        + beta23 * x2 * x3
        + beta22 * x2**2
        + beta33 * x3**2
        + beta12 * x1 * x2
        + beta13 * x1 * x3
    )
    return float(sigmoid(z))


def g_scalar_grad(x_vec: np.ndarray) -> np.ndarray:
    """
    Gradient of scalar g wrt x = (x1, x2, x3).
    Returns shape (3,)
    """
    x_vec = np.asarray(x_vec, dtype=float)
    x1, x2, x3 = x_vec[0], x_vec[1], x_vec[2]

    beta0 = -3.2
    beta1 = 0.05
    beta2 = 1.60
    beta3 = 1.60
    beta23 = 3.50
    beta22 = 3.00
    beta33 = 3.00
    # tiny couplings with x1
    beta12 = 0.10
    beta13 = 0.08

    z = (
        beta0
        + beta1 * x1
        + beta2 * x2
        + beta3 * x3
        + beta23 * x2 * x3
        + beta22 * x2**2
        + beta33 * x3**2
        + beta12 * x1 * x2
        + beta13 * x1 * x3
    )
    s = float(sigmoid(z))
    common = s * (1.0 - s)

    dz_dx1 = beta1 + beta12 * x2 + beta13 * x3
    dz_dx2 = beta2 + beta23 * x3 + 2.0 * beta22 * x2 + beta12 * x1
    dz_dx3 = beta3 + beta23 * x2 + 2.0 * beta33 * x3 + beta13 * x1

    return np.array(
        [
            common * dz_dx1,
            common * dz_dx2,
            common * dz_dx3,
        ],
        dtype=float,
    )


# ============================================================
# Experiment setup
# ============================================================
def get_system_parameters() -> dict[str, np.ndarray]:
    A0 = np.array(
        [
            [0.92, 0.10, 0.03],
            [0.02, 0.90, 0.12],
            [0.01, 0.08, 0.88],
        ],
        dtype=float,
    )

    B0 = np.array(
        [
            [-0.22, 0.10, 0.06],
            [-0.10, 0.20, 0.12],
            [0.04, -0.08, 0.18],
        ],
        dtype=float,
    )

    H0 = np.array(
        [
            [1.10, 0.20, 0.10],
            [0.15, 1.05, 0.25],
            [0.08, 0.30, 0.95],
        ],
        dtype=float,
    )

    D0 = np.array(
        [
            [-0.04, 0.10, 0.05],
            [-0.08, 0.14, 0.10],
            [0.02, -0.03, 0.12],
        ],
        dtype=float,
    )

    Q0 = np.array(
        [
            [0.018, 0.004, 0.002],
            [0.004, 0.020, 0.006],
            [0.002, 0.006, 0.019],
        ],
        dtype=float,
    )

    R0 = np.array(
        [
            [0.040, 0.008, 0.004],
            [0.008, 0.042, 0.010],
            [0.004, 0.010, 0.038],
        ],
        dtype=float,
    )

    x0 = np.array([0.15, 0.35, 0.30], dtype=float)
    m0 = x0.copy()

    P0 = np.array(
        [
            [0.035, 0.006, 0.003],
            [0.006, 0.038, 0.008],
            [0.003, 0.008, 0.036],
        ],
        dtype=float,
    )

    return {
        "A0": A0,
        "B0": B0,
        "H0": H0,
        "D0": D0,
        "Q0": project_to_psd(Q0),
        "R0": project_to_psd(R0),
        "x0": x0,
        "m0": m0,
        "P0": project_to_psd(P0),
        "dA": np.zeros_like(A0),
        "dB": np.zeros_like(B0),
        "dH": np.zeros_like(H0),
        "dD": np.zeros_like(D0),
        "dQ": np.zeros_like(Q0),
        "dR": np.zeros_like(R0),
    }


# ============================================================
# Single run
# ============================================================
def run_one_experiment(run_id: int) -> dict[str, np.ndarray | float]:
    pars = get_system_parameters()

    sim_seed = 2025 + run_id
    attack_seed = 6000 + run_id

    x, y, u, mats = simulate_lgssm_nd(
        A0=pars["A0"],
        B0=pars["B0"],
        H0=pars["H0"],
        D0=pars["D0"],
        T=T,
        seed=sim_seed,
        x0=pars["x0"],
        Q0=pars["Q0"],
        R0=pars["R0"],
        dA=pars["dA"],
        dB=pars["dB"],
        dH=pars["dH"],
        dD=pars["dD"],
        dQ=pars["dQ"],
        dR=pars["dR"],
        u_low=-0.5,
        u_high=0.5,
    )

    y_star, attack_hist = white_box_point_attack_nd(
        t=ATTACK_T,
        y=y,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        P0=pars["P0"],
        m0=pars["m0"],
        epsilon=EPSILON,
        M_star=M_STAR,
        g=g_scalar,
        g_grad=g_scalar_grad,
        eta=ETA,
        n_steps=N_STEPS,
        n_mc=N_MC_OPT,
        seed=attack_seed,
    )

    y_T = y[ATTACK_T].copy()
    delta_y = y_star - y_T
    delta_norm = float(np.linalg.norm(delta_y))

    if delta_norm > 1e-12:
        direction = delta_y / delta_norm
    else:
        direction = np.zeros_like(delta_y)

    m_filt_b, P_filt_b, m_pred_b, P_pred_b = kalman_filter_nd(
        y=y,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )
    m_smooth_b, P_smooth_b = rts_smoother_nd(
        m_filt=m_filt_b,
        P_filt=P_filt_b,
        m_pred=m_pred_b,
        P_pred=P_pred_b,
        A_t=mats["A_t"],
    )

    y_adv = y.copy()
    y_adv[ATTACK_T] = y_star

    m_filt_a, P_filt_a, m_pred_a, P_pred_a = kalman_filter_nd(
        y=y_adv,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )
    m_smooth_a, P_smooth_a = rts_smoother_nd(
        m_filt=m_filt_a,
        P_filt=P_filt_a,
        m_pred=m_pred_a,
        P_pred=P_pred_a,
        A_t=mats["A_t"],
    )

    mu_g_base, _ = estimate_E_g(
        m=m_smooth_b[ATTACK_T],
        P=P_smooth_b[ATTACK_T],
        g=g_scalar,
        n_mc=N_MC_EST,
        seed=77,
    )
    mu_g_adv, _ = estimate_E_g(
        m=m_smooth_a[ATTACK_T],
        P=P_smooth_a[ATTACK_T],
        g=g_scalar,
        n_mc=N_MC_EST,
        seed=77,
    )

    return {
        "run_id": run_id,
        "y_T": y_T,
        "y_star": y_star,
        "delta_y": delta_y,
        "delta_norm": delta_norm,
        "direction": direction,
        "final_obj": float(attack_hist["obj_hist"][-1]),
        "risk_base": float(mu_g_base[0]),
        "risk_adv": float(mu_g_adv[0]),
    }


def plot_delta_density_pairwise(
    deltas: np.ndarray,
    outpath: str,
    gridsize: int = 220,
    scatter_size: float = 18.0,
) -> None:
    """
    Pairwise 2D KDE density plots of 3D attack displacements:
        delta_y = y_T' - y_T

    Creates a 1x3 figure with panels:
        (Δo1, Δo2), (Δo1, Δo3), (Δo2, Δo3)

    deltas: shape (N, 3)
    """
    deltas = np.asarray(deltas, dtype=float)

    if deltas.ndim != 2 or deltas.shape[1] != 3:
        raise ValueError("deltas must have shape (N, 3)")

    pairs = [
        (0, 1, r"$\Delta o_1$", r"$\Delta o_2$"),
        (0, 2, r"$\Delta o_1$", r"$\Delta o_3$"),
        (1, 2, r"$\Delta o_2$", r"$\Delta o_3$"),
    ]

    # common symmetric axis limit across all coordinates
    max_abs = max(np.max(np.abs(deltas[:, 0])),
                  np.max(np.abs(deltas[:, 1])),
                  np.max(np.abs(deltas[:, 2])),
                  1e-3)
    lim = 1.10 * max_abs

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)

    last_im = None

    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        x = deltas[:, i]
        y = deltas[:, j]

        values = np.vstack([x, y])
        kde = gaussian_kde(values)

        xi = np.linspace(-lim, lim, gridsize)
        yi = np.linspace(-lim, lim, gridsize)
        XX, YY = np.meshgrid(xi, yi)

        grid_coords = np.vstack([XX.ravel(), YY.ravel()])
        Z = kde(grid_coords).reshape(XX.shape)

        im = ax.imshow(
            Z,
            origin="lower",
            extent=[-lim, lim, -lim, lim],
            cmap="cividis",
            aspect="equal",
        )
        last_im = im

        # optional point overlay
        ax.scatter(x, y, s=scatter_size, alpha=0.35, edgecolors="none")

        # mark origin
        ax.scatter(0.0, 0.0, marker="x", s=80, linewidths=2.0, color="white")
        ax.axhline(0.0, linewidth=0.8, alpha=0.35, color="white")
        ax.axvline(0.0, linewidth=0.8, alpha=0.35, color="white")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.12)

    axes[0].set_title(r"Density of $(\Delta o_1, \Delta o_2)$")
    axes[1].set_title(r"Density of $(\Delta o_1, \Delta o_3)$")
    axes[2].set_title(r"Density of $(\Delta o_2, \Delta o_3)$")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.90, pad=0.02)
    cbar.set_label("density")

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


def plot_delta_density_3d(
    deltas: np.ndarray,
    outpath: str,
    scatter_size: float = 30.0,
) -> None:
    """
    Single 3D scatter of attack displacements with 2D KDE density maps
    projected onto the coordinate planes.

    The 3D axes/box are hidden. Only the density projections and their
    plane axes are shown.

    Projections:
        (Δo1, Δo2) density map on z = -lim
        (Δo1, Δo3) density map on y = -lim
        (Δo2, Δo3) density map on x = -lim

    deltas: shape (N, 3)
    """
    deltas = np.asarray(deltas, dtype=float)

    if deltas.ndim != 2 or deltas.shape[1] != 3:
        raise ValueError("deltas must have shape (N, 3)")

    x0 = deltas[:, 0]
    y0 = deltas[:, 1]
    z0 = deltas[:, 2]

    # ============================================================
    # 3D density for coloring the point cloud
    # ============================================================
    values_3d = np.vstack([x0, y0, z0])
    density0 = gaussian_kde(values_3d)(values_3d)

    order = np.argsort(density0)
    x = x0[order]
    y = y0[order]
    z = z0[order]
    density = density0[order]

    max_abs = max(
        np.max(np.abs(x0)),
        np.max(np.abs(y0)),
        np.max(np.abs(z0)),
        1e-3,
    )
    lim = 1.10 * max_abs

    density_lo, density_hi = np.percentile(density, [5.0, 95.0])
    if density_hi <= density_lo:
        density_lo = float(np.min(density))
        density_hi = float(np.max(density))
    if density_hi <= density_lo:
        density_hi = density_lo + 1e-12

    density_norm = Normalize(vmin=density_lo, vmax=density_hi, clip=True)

    # ============================================================
    # Helper: projected 2D KDE on a coordinate pair
    # ============================================================
    def build_pairwise_kde(
        a: np.ndarray,
        b: np.ndarray,
        grid_size: int = 160,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ai = np.linspace(-lim, lim, grid_size)
        bi = np.linspace(-lim, lim, grid_size)

        AA, BB = np.meshgrid(ai, bi)

        values_2d = np.vstack([a, b])
        kde = gaussian_kde(values_2d)

        DD = kde(np.vstack([AA.ravel(), BB.ravel()])).reshape(AA.shape)

        return AA, BB, DD

    # These are real 2D projected density maps.
    X_xy, Y_xy, D_xy = build_pairwise_kde(x0, y0)  # density of (x, y)
    X_xz, Z_xz, D_xz = build_pairwise_kde(x0, z0)  # density of (x, z)
    Y_yz, Z_yz, D_yz = build_pairwise_kde(y0, z0)  # density of (y, z)

    # Common normalization for the three projected density maps.
    proj_density_lo = min(
        np.percentile(D_xy, 5.0),
        np.percentile(D_xz, 5.0),
        np.percentile(D_yz, 5.0),
    )
    proj_density_hi = max(
        np.percentile(D_xy, 98.0),
        np.percentile(D_xz, 98.0),
        np.percentile(D_yz, 98.0),
    )

    if proj_density_hi <= proj_density_lo:
        proj_density_hi = proj_density_lo + 1e-12

    proj_norm = Normalize(vmin=proj_density_lo, vmax=proj_density_hi, clip=True)
    cmap = plt.get_cmap("cividis")

    # ============================================================
    # Figure
    # ============================================================
    fig = plt.figure(figsize=(10.2, 8.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # ============================================================
    # Projected density maps as real colored planes
    # ============================================================

    # ------------------------------------------------------------
    # Projection 1: density of (Δo1, Δo2) on plane z = -lim
    # ------------------------------------------------------------
    Z_plane_xy = -lim * np.ones_like(X_xy)

    ax.plot_surface(
        X_xy,
        Y_xy,
        Z_plane_xy,
        rstride=1,
        cstride=1,
        facecolors=cmap(proj_norm(D_xy)),
        shade=False,
        alpha=0.58,
        linewidth=0,
        antialiased=False,
    )

    # ------------------------------------------------------------
    # Projection 2: density of (Δo1, Δo3) on plane y = -lim
    # ------------------------------------------------------------
    Y_plane_xz = -lim * np.ones_like(X_xz)

    ax.plot_surface(
        X_xz,
        Y_plane_xz,
        Z_xz,
        rstride=1,
        cstride=1,
        facecolors=cmap(proj_norm(D_xz)),
        shade=False,
        alpha=0.58,
        linewidth=0,
        antialiased=False,
    )

    # ------------------------------------------------------------
    # Projection 3: density of (Δo2, Δo3) on plane x = -lim
    # ------------------------------------------------------------
    X_plane_yz = -lim * np.ones_like(Y_yz)

    ax.plot_surface(
        X_plane_yz,
        Y_yz,
        Z_yz,
        rstride=1,
        cstride=1,
        facecolors=cmap(proj_norm(D_yz)),
        shade=False,
        alpha=0.58,
        linewidth=0,
        antialiased=False,
    )


    # ============================================================
    # Axes only on the projection planes
    # ============================================================

    axis_lw = 1.35
    axis_alpha = 0.98

    # Axes on xy plane: z = -lim
    ax.plot(
        [-lim, lim],
        [0.0, 0.0],
        [-lim, -lim],
        color="black",
        linewidth=axis_lw,
        alpha=axis_alpha,
    )
    ax.plot(
        [0.0, 0.0],
        [-lim, lim],
        [-lim, -lim],
        color="black",
        linewidth=axis_lw,
        alpha=axis_alpha,
    )

    # Axes on xz plane: y = -lim
    ax.plot(
        [-lim, lim],
        [-lim, -lim],
        [0.0, 0.0],
        color="black",
        linewidth=axis_lw,
        alpha=axis_alpha,
    )
    ax.plot(
        [0.0, 0.0],
        [-lim, -lim],
        [-lim, lim],
        color="black",
        linewidth=axis_lw,
        alpha=axis_alpha,
    )

    # Axes on yz plane: x = -lim
    ax.plot(
        [-lim, -lim],
        [-lim, lim],
        [0.0, 0.0],
        color="black",
        linewidth=axis_lw,
        alpha=axis_alpha,
    )
    ax.plot(
        [-lim, -lim],
        [0.0, 0.0],
        [-lim, lim],
        color="black",
        linewidth=axis_lw,
        alpha=axis_alpha,
    )

    # ============================================================
    # Labels on projection-plane axes
    # ============================================================

    label_fs = 11

    # xy projection labels
    ax.text(
        lim,
        0.0,
        -lim - 0.25,
        r"$\Delta o_1$",
        fontsize=label_fs,
        ha="left",
        va="center",
    )
    ax.text(
        0.0,
        lim,
        -lim - 0.25,
        r"$\Delta o_2$",
        fontsize=label_fs,
        ha="center",
        va="bottom",
    )

    # xz projection labels
    ax.text(
        lim,
        -lim - 0.25,
        0.0,
        r"$\Delta o_1$",
        fontsize=label_fs,
        ha="left",
        va="center",
    )
    ax.text(
        0.0,
        -lim - 0.25,
        lim,
        r"$\Delta o_3$",
        fontsize=label_fs,
        ha="center",
        va="bottom",
    )

    # yz projection labels
    ax.text(
        -lim - 0.25,
        lim ,
        0.0,
        r"$\Delta o_2$",
        fontsize=label_fs,
        ha="left",
        va="center",
    )
    ax.text(
        -lim - 0.25,
        0.0,
        lim,
        r"$\Delta o_3$",
        fontsize=label_fs,
        ha="center",
        va="bottom",
    )

    # ============================================================
    # Main 3D displacement cloud
    # ============================================================

    scatter = ax.scatter(
        x,
        y,
        z,
        c=density,
        cmap="cividis",
        norm=density_norm,
        s=scatter_size,
        alpha=0.92,
        edgecolors="black",
        linewidths=0.25,
        marker="o",
        depthshade=True,
    )

    # Origin marker
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="x",
        s=95,
        linewidths=2.0,
        color="crimson",
        depthshade=False,
    )

    # ============================================================
    # Limits and camera
    # ============================================================

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.view_init(elev=24, azim=36)

    # ============================================================
    # Hide default 3D axes, ticks, grid and panes
    # ============================================================

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))

    ax.set_axis_off()

    # Colorbar for the 3D point cloud
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.06)
    cbar.set_label("KDE density of attack perturbations")

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig) 

# ============================================================
# Main
# ============================================================
def main() -> None:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(module_dir, FIGURES_DIRNAME)
    data_dir = os.path.join(module_dir, DATA_DIRNAME)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(
        data_dir,
        f"delta_y_points_3d_N{N_RUNS}_T{T}_t{ATTACK_T}.npz",
    )

    density_path = os.path.join(
        figures_dir,
        f"delta_y_density_3d_N{N_RUNS}_T{T}_t{ATTACK_T}.png",
    )

    if os.path.exists(data_path):
        print("\nData already exists. Skipping simulation...")

        data = np.load(data_path)
        deltas = data["deltas"]

        plot_delta_density_3d(
            deltas=deltas,
            outpath=density_path,
        )

        print(f"3D density plot regenerated from existing data: {density_path}")
        return

    all_results: list[dict[str, np.ndarray | float]] = []

    print("\nRunning repeated attacks...")
    print(f"N_RUNS   = {N_RUNS}")
    print(f"T        = {T}")
    print(f"attack t = {ATTACK_T}")
    print(f"epsilon  = {EPSILON:.6f}")
    print(f"M_STAR   = {M_STAR}\n")

    for run_id in range(N_RUNS):
        res = run_one_experiment(run_id)
        all_results.append(res)

        if (run_id + 1) % 10 == 0 or run_id == 0:
            print(
                f"[{run_id + 1:3d}/{N_RUNS}] "
                f"delta_y = {res['delta_y']} | "
                f"||delta|| = {res['delta_norm']:.4f} | "
                f"risk: {res['risk_base']:.3f} -> {res['risk_adv']:.3f} | "
                f"final obj = {res['final_obj']:.4e}"
            )

    deltas = np.stack(
        [np.asarray(r["delta_y"], dtype=float) for r in all_results],
        axis=0,
    )

    plot_delta_density_3d(
        deltas=deltas,
        outpath=density_path,
    )

    np.savez(
        data_path,
        deltas=deltas,
    )

    print(f"\nSaved 3D density plot to: {density_path}")
    print(f"Saved raw delta points to: {data_path}")


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
from shared_ssm.legacy import estimate_E_g
from shared_ssm.legacy import finite_diff_jacobian_g
from shared_ssm.legacy import kalman_filter_nd_current_observation as kalman_filter_nd
from shared_ssm.legacy import leave_one_out_attack_stats_nd_current_observation as leave_one_out_attack_stats_nd
from shared_ssm.legacy import project_to_attack_region
from shared_ssm.legacy import rts_smoother_nd
from shared_ssm.legacy import simulate_lgssm_nd_current_observation as simulate_lgssm_nd
from shared_ssm.legacy import white_box_point_attack_nd


if __name__ == "__main__":
    main()
