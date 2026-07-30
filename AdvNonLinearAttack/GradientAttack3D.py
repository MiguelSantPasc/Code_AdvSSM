#!/usr/bin/env python3
"""
GradientAttack3D.py

White-box point attack on a 3D LGSSM with a 3D observation attack region.

This script mirrors the existing 2D `GradientAttack.py`, but it reuses the
3D dynamics and risk model from `AttackSense3D.py`. The figure is organized
as four panels:

(A) x1 over time (base vs adversarial)
(B) x2 over time (base vs adversarial)
(C) x3 over time (base vs adversarial)
(D) A single 3D gradient-descent panel in observation space

The 3D state-space panel is intentionally omitted so the layout stays compact.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _path in (_THIS_DIR, _PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from AttackSense3D import get_system_parameters
from AttackSense3D import g_scalar
from AttackSense3D import g_scalar_grad
from shared_ssm.artifacts import data_path_for_plot
from shared_ssm.artifacts import load_npz
from shared_ssm.artifacts import save_npz
from shared_ssm.legacy import estimate_E_g
from shared_ssm.legacy import kalman_filter_nd_current_observation as kalman_filter_nd
from shared_ssm.legacy import rts_smoother_nd
from shared_ssm.legacy import simulate_lgssm_nd_current_observation as simulate_lgssm_nd
from shared_ssm.legacy import white_box_point_attack_nd
from shared_ssm.linalg import project_to_psd
from shared_ssm.linalg import spd_inverse as inv_psd
from shared_ssm.linalg import symmetrize


# ============================================================
# Default experiment configuration
# ============================================================
DEFAULT_T = 5
DEFAULT_ATTACK_T = DEFAULT_T
DEFAULT_SEED = 2025
DEFAULT_EPSILON = 7.814727903251179  # 95% chi-square threshold in 3D
DEFAULT_M_STAR = np.array([1.0], dtype=float)
# Slightly larger PGD step than the initial version so the optimizer moves
# more decisively while still relying on the feasible-set projection.
DEFAULT_ETA = 0.12
DEFAULT_N_STEPS = 800
DEFAULT_N_MC_OPT = 128
DEFAULT_N_MC_EST = 2000
OUTPUT_DIRNAME = os.path.join("outputs", "figures")


# ============================================================
# Plot helpers
# ============================================================
def _ellipse_points_from_quad(
    center: np.ndarray,
    shape_inv: np.ndarray,
    level: float,
    n: int = 360,
) -> np.ndarray:
    """
    Return a dense 2D ellipse sampled from:

        (y - center)^T shape_inv (y - center) = level
    """
    center = np.asarray(center, dtype=float).reshape(2,)
    M = project_to_psd(symmetrize(np.asarray(shape_inv, dtype=float).reshape(2, 2)))

    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-14)
    radii = np.sqrt(level / eigvals)

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    return (eigvecs @ (radii[:, None] * circle)).T + center[None, :]


def _set_plot_theme() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10.2,
            "axes.titlesize": 12.6,
            "axes.labelsize": 10.8,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.1,
            "ytick.labelsize": 9.1,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.grid": True,
        }
    )


def _style_axis(ax, *, facecolor: str = "#FBFBFD") -> None:
    ax.set_facecolor(facecolor)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.55)
    ax.spines["bottom"].set_alpha(0.55)
    ax.grid(True, alpha=0.20)


def _points_limits(
    points: list[np.ndarray],
    pad_frac: float = 0.10,
) -> tuple[tuple[float, float], tuple[float, float]]:
    arrs: list[np.ndarray] = []
    for point_set in points:
        pts = np.asarray(point_set, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.size > 0:
            arrs.append(pts[:, :2])

    if not arrs:
        return (-1.0, 1.0), (-1.0, 1.0)

    stacked = np.vstack(arrs)
    xmin, ymin = np.min(stacked[:, 0]), np.min(stacked[:, 1])
    xmax, ymax = np.max(stacked[:, 0]), np.max(stacked[:, 1])

    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    diameter = max(dx, dy)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    pad = pad_frac * diameter + 1e-6

    return (
        cx - 0.5 * diameter - pad,
        cx + 0.5 * diameter + pad,
    ), (
        cy - 0.5 * diameter - pad,
        cy + 0.5 * diameter + pad,
    )


def _projected_pairwise_ellipse(
    center: np.ndarray,
    Sigma_t: np.ndarray,
    epsilon: float,
    dims: tuple[int, int],
) -> np.ndarray:
    """
    Build the projected feasible ellipse for a coordinate pair.

    For a Gaussian ellipsoid defined by Sigma_t in 3D, the orthogonal
    projection onto a coordinate pair is represented by the corresponding
    marginal covariance block.
    """
    Sigma_pair = project_to_psd(Sigma_t[np.ix_(dims, dims)])
    Sigma_pair_inv = inv_psd(Sigma_pair)
    center_pair = np.asarray(center, dtype=float)[list(dims)]
    return _ellipse_points_from_quad(center_pair, Sigma_pair_inv, epsilon)


def _ellipsoid_wireframe_points(
    center: np.ndarray,
    shape_inv: np.ndarray,
    level: float,
    n_u: int = 44,
    n_v: int = 22,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return a wireframe parameterization of:

        (y - center)^T shape_inv (y - center) = level

    for the 3D observation-space attack region.
    """
    center = np.asarray(center, dtype=float).reshape(3,)
    M = project_to_psd(symmetrize(np.asarray(shape_inv, dtype=float).reshape(3, 3)))

    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-14)
    radii = np.sqrt(level / eigvals)

    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    sphere = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=0,
    )
    ellipsoid = eigvecs @ (radii[:, None] * sphere.reshape(3, -1))
    ellipsoid = ellipsoid.reshape(3, n_u, n_v) + center[:, None, None]
    return ellipsoid[0], ellipsoid[1], ellipsoid[2]


def plot_attack_figure_four_panels(
    *,
    t: int,
    y_t: np.ndarray,
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    y_star: np.ndarray,
    y_path: np.ndarray,
    epsilon: float,
    x_true: np.ndarray,
    m_smooth_base: np.ndarray,
    P_smooth_base: np.ndarray,
    m_smooth_adv: np.ndarray,
    P_smooth_adv: np.ndarray,
    obj_hist: np.ndarray,
    outpath: str,
) -> None:
    """
    Plot the 3D attack summary as four main panels:
    three time-series panels and one 3D observation-space gradient-descent
    panel.
    """
    if y_t.shape != (3,) or mu_t.shape != (3,) or y_star.shape != (3,):
        raise ValueError("This plot expects n_y = 3.")
    if Sigma_t.shape != (3, 3):
        raise ValueError("Sigma_t must have shape (3, 3).")
    if x_true.shape[1] < 3:
        raise ValueError("Need at least 3 latent state dimensions.")

    _set_plot_theme()

    c_base = "#2F6FA8"
    c_adv = "#C85B4F"
    c_true = "#2F2F2F"
    c_mu = "#6A5ACD"
    c_feasible = "#6A5ACD"
    c_star = "#D4A017"
    c_attack_band = "#E9D8A6"
    c_path = "#2CA58D"

    horizon = x_true.shape[0] - 1
    time_idx = np.arange(horizon + 1)
    conf_multiplier = 1.96

    fig = plt.figure(figsize=(17.5, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.18,
    )

    ax_ts1 = fig.add_subplot(gs[0, 0])
    ax_ts2 = fig.add_subplot(gs[0, 1])
    ax_ts3 = fig.add_subplot(gs[1, 0])
    ax_geom_panel = fig.add_subplot(gs[1, 1], projection="3d")

    for ax in [ax_ts1, ax_ts2, ax_ts3]:
        _style_axis(ax)
    ax_geom_panel.set_facecolor("#FBFBFD")

    def _plot_state_time(ax, idx: int, panel_title: str) -> None:
        true_line = x_true[:, idx]
        base_mean = m_smooth_base[:, idx]
        base_sd = np.sqrt(np.maximum(P_smooth_base[:, idx, idx], 0.0))
        adv_mean = m_smooth_adv[:, idx]
        adv_sd = np.sqrt(np.maximum(P_smooth_adv[:, idx, idx], 0.0))

        ax.axvspan(t - 0.35, t + 0.35, color=c_attack_band, alpha=0.28, zorder=0)
        ax.axvline(t, color="#8D6E63", linewidth=1.0, alpha=0.45)

        ax.fill_between(
            time_idx,
            base_mean - conf_multiplier * base_sd,
            base_mean + conf_multiplier * base_sd,
            color=c_base,
            alpha=0.18,
            label="Base 95% CI",
            zorder=1,
        )
        ax.plot(time_idx, base_mean, color=c_base, linewidth=1.9, label="Base RTS mean", zorder=3)

        ax.fill_between(
            time_idx,
            adv_mean - conf_multiplier * adv_sd,
            adv_mean + conf_multiplier * adv_sd,
            color=c_adv,
            alpha=0.13,
            label="Adv 95% CI",
            zorder=1,
        )
        ax.plot(
            time_idx,
            adv_mean,
            color=c_adv,
            linewidth=1.9,
            linestyle="--",
            label="Adv RTS mean",
            zorder=3,
        )

        ax.plot(
            time_idx,
            true_line,
            color=c_true,
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            alpha=0.9,
            label=f"True x[{idx}]",
            zorder=2,
        )

        ax.scatter([t], [true_line[t]], color=c_true, s=28, zorder=5)
        ax.scatter([t], [base_mean[t]], color=c_base, s=30, zorder=5)
        ax.scatter([t], [adv_mean[t]], color=c_adv, s=34, marker="D", zorder=5)

        ax.set_title(panel_title, loc="left", fontweight="semibold")
        ax.set_xlabel("time k")
        ax.set_ylabel(f"x[{idx}]")
        ax.margins(x=0.02)
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.94,
            ncol=2,
            columnspacing=0.9,
            handlelength=1.8,
            borderpad=0.45,
        )

    _plot_state_time(ax_ts1, idx=0, panel_title="(A) Attack impact on x1 over time")
    _plot_state_time(ax_ts2, idx=1, panel_title="(B) Attack impact on x2 over time")
    _plot_state_time(ax_ts3, idx=2, panel_title="(C) Attack impact on x3 over time")

    Sigma_t = project_to_psd(Sigma_t)
    Sigma_t_inv = inv_psd(Sigma_t)
    ell_x, ell_y, ell_z = _ellipsoid_wireframe_points(mu_t, Sigma_t_inv, epsilon)

    ax_geom_panel.plot_wireframe(
        ell_x,
        ell_y,
        ell_z,
        rstride=2,
        cstride=2,
        color=c_feasible,
        linewidth=0.9,
        alpha=0.35,
    )
    ax_geom_panel.plot(
        [],
        [],
        [],
        color=c_feasible,
        linewidth=1.6,
        alpha=0.9,
        label="Feasible attack region",
    )

    if y_path.shape[0] > 0:
        ax_geom_panel.plot(
            y_path[:, 0],
            y_path[:, 1],
            y_path[:, 2],
            color=c_path,
            linewidth=1.8,
            marker="o",
            markersize=3.1,
            alpha=0.92,
            label="Optimization path",
        )

    ax_geom_panel.scatter(
        [mu_t[0]],
        [mu_t[1]],
        [mu_t[2]],
        s=60,
        marker="o",
        color=c_mu,
        label=r"$\mu_t$",
        depthshade=False,
    )
    ax_geom_panel.scatter(
        [y_t[0]],
        [y_t[1]],
        [y_t[2]],
        s=75,
        marker="x",
        linewidths=2.0,
        color="#111111",
        label=r"$y_t$",
        depthshade=False,
    )
    ax_geom_panel.scatter(
        [y_star[0]],
        [y_star[1]],
        [y_star[2]],
        s=140,
        marker="*",
        color=c_star,
        edgecolor="black",
        linewidths=0.4,
        label=r"$y_t^\prime$",
        depthshade=False,
    )
    ax_geom_panel.plot(
        [y_t[0], y_star[0]],
        [y_t[1], y_star[1]],
        [y_t[2], y_star[2]],
        color=c_adv,
        linewidth=1.4,
        alpha=0.86,
        linestyle="-.",
    )

    x_parts = [ell_x.ravel(), np.array([mu_t[0], y_t[0], y_star[0]])]
    y_parts = [ell_y.ravel(), np.array([mu_t[1], y_t[1], y_star[1]])]
    z_parts = [ell_z.ravel(), np.array([mu_t[2], y_t[2], y_star[2]])]
    if y_path.size > 0:
        x_parts.append(y_path[:, 0])
        y_parts.append(y_path[:, 1])
        z_parts.append(y_path[:, 2])

    x_all = np.concatenate(x_parts)
    y_all = np.concatenate(y_parts)
    z_all = np.concatenate(z_parts)

    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    zmin, zmax = float(np.min(z_all)), float(np.max(z_all))
    span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-6)
    pad = 0.14 * span
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    zmid = 0.5 * (zmin + zmax)

    ax_geom_panel.set_xlim(xmid - 0.5 * span - pad, xmid + 0.5 * span + pad)
    ax_geom_panel.set_ylim(ymid - 0.5 * span - pad, ymid + 0.5 * span + pad)
    ax_geom_panel.set_zlim(zmid - 0.5 * span - pad, zmid + 0.5 * span + pad)
    ax_geom_panel.set_box_aspect((1.0, 1.0, 1.0))
    ax_geom_panel.view_init(elev=24, azim=38)
    ax_geom_panel.set_title("(D) 3D gradient-descent path", loc="left", fontweight="semibold")
    ax_geom_panel.set_xlabel("o[0]")
    ax_geom_panel.set_ylabel("o[1]")
    ax_geom_panel.set_zlabel("o[2]")
    ax_geom_panel.xaxis.pane.set_facecolor((0.98, 0.98, 1.0, 0.18))
    ax_geom_panel.yaxis.pane.set_facecolor((0.98, 0.98, 1.0, 0.18))
    ax_geom_panel.zaxis.pane.set_facecolor((0.98, 0.98, 1.0, 0.18))
    ax_geom_panel.grid(True, alpha=0.18)
    ax_geom_panel.legend(loc="upper left", frameon=True, framealpha=0.94)

    if obj_hist.size > 0:
        ax_geom_panel.text2D(
            0.03,
            0.04,
            f"final objective = {obj_hist[-1]:.4e}\nsteps = {obj_hist.size}",
            transform=ax_geom_panel.transAxes,
            fontsize=9.0,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.92,
                edgecolor="#BBBBBB",
            ),
        )

    fig.align_ylabels([ax_ts1, ax_ts3])

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, facecolor="white", dpi=300)
    plt.close(fig)


# ============================================================
# Single-run experiment
# ============================================================
def run_attack_experiment(
    *,
    use_analytic_grad: bool = True,
    T: int = DEFAULT_T,
    t: int = DEFAULT_ATTACK_T,
    seed: int = DEFAULT_SEED,
    epsilon: float = DEFAULT_EPSILON,
    M_star: np.ndarray = DEFAULT_M_STAR,
    eta: float = DEFAULT_ETA,
    n_steps: int = DEFAULT_N_STEPS,
    n_mc_opt: int = DEFAULT_N_MC_OPT,
    n_mc_est: int = DEFAULT_N_MC_EST,
    force_recompute: bool = False,
    out_filename: str | None = None,
) -> str:
    """
    Run one 3D attack experiment, cache the data, and save the figure.
    """
    pars = get_system_parameters()

    if out_filename is None:
        if use_analytic_grad:
            out_filename = f"attack_on_g_3d_four_panels_t{t}_T{T}_seed{seed}.png"
        else:
            out_filename = f"attack_on_g_nograd_3d_four_panels_t{t}_T{T}_seed{seed}.png"

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    outpath = os.path.join(out_dir, out_filename)
    data_path = data_path_for_plot(outpath)

    if os.path.exists(data_path) and not force_recompute:
        print(f"[cache] loading data: {data_path}")
        data = load_npz(data_path)
        plot_attack_figure_four_panels(
            t=int(data["t"]),
            y_t=data["y_t"],
            mu_t=data["mu_t"],
            Sigma_t=data["Sigma_t"],
            y_star=data["y_star"],
            y_path=data["y_path"],
            epsilon=float(data["epsilon"]),
            x_true=data["x_true"],
            m_smooth_base=data["m_smooth_base"],
            P_smooth_base=data["P_smooth_base"],
            m_smooth_adv=data["m_smooth_adv"],
            P_smooth_adv=data["P_smooth_adv"],
            obj_hist=data["obj_hist"],
            outpath=outpath,
        )
        print(f"Saved figure to: {outpath}")
        return outpath

    x, y, u, mats = simulate_lgssm_nd(
        A0=pars["A0"],
        B0=pars["B0"],
        H0=pars["H0"],
        D0=pars["D0"],
        T=T,
        seed=seed,
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
        t=t,
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
        epsilon=epsilon,
        M_star=M_star,
        g=g_scalar,
        g_grad=g_scalar_grad if use_analytic_grad else None,
        eta=eta,
        n_steps=n_steps,
        n_mc=n_mc_opt,
        seed=seed + 1,
    )

    mu_t = attack_hist["mu_t"]
    Sigma_t = attack_hist["Sigma_t"]
    y_t = y[t].copy()

    Sigma_t_inv = inv_psd(Sigma_t)
    constr_val = float((y_star - mu_t).T @ Sigma_t_inv @ (y_star - mu_t))
    print(f"\n[t={t}] feasibility value = {constr_val:.6f} (should be <= epsilon={epsilon})")
    print(f"[t={t}] y_t             = {y_t}")
    print(f"[t={t}] mu_t            = {mu_t}")
    print(f"[t={t}] y_star          = {y_star}")
    print(f"[t={t}] final objective = {attack_hist['obj_hist'][-1]:.6f}")

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
    y_adv[t] = y_star

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

    mu_g_base_t, _ = estimate_E_g(
        m=m_smooth_b[t],
        P=P_smooth_b[t],
        g=g_scalar,
        n_mc=n_mc_est,
        seed=7,
    )
    mu_g_adv_t, _ = estimate_E_g(
        m=m_smooth_a[t],
        P=P_smooth_a[t],
        g=g_scalar,
        n_mc=n_mc_est,
        seed=7,
    )

    print("\n=== g(x_t) comparison at attacked time t ===")
    print(f"t = {t}")
    print(f"E[g(x_t) | y]        = {mu_g_base_t}")
    print(f"E[g(x_t) | y_attack] = {mu_g_adv_t}")

    save_npz(
        data_path,
        t=t,
        T=T,
        seed=seed,
        epsilon=epsilon,
        y_t=y_t,
        mu_t=mu_t,
        Sigma_t=Sigma_t,
        y_star=y_star,
        y_path=attack_hist["y_hist"],
        obj_hist=attack_hist["obj_hist"],
        x_true=x,
        m_smooth_base=m_smooth_b,
        P_smooth_base=P_smooth_b,
        m_smooth_adv=m_smooth_a,
        P_smooth_adv=P_smooth_a,
        mu_g_base_t=mu_g_base_t,
        mu_g_adv_t=mu_g_adv_t,
    )
    print(f"[cache] saved data: {data_path}")

    plot_attack_figure_four_panels(
        t=t,
        y_t=y_t,
        mu_t=mu_t,
        Sigma_t=Sigma_t,
        y_star=y_star,
        y_path=attack_hist["y_hist"],
        epsilon=epsilon,
        x_true=x,
        m_smooth_base=m_smooth_b,
        P_smooth_base=P_smooth_b,
        m_smooth_adv=m_smooth_a,
        P_smooth_adv=P_smooth_a,
        obj_hist=attack_hist["obj_hist"],
        outpath=outpath,
    )
    print(f"\nSaved figure to: {outpath}")
    return outpath


def main() -> None:
    run_attack_experiment(use_analytic_grad=True)


if __name__ == "__main__":
    main()
