#!/usr/bin/env python3
"""
KKTOpt.py

End-to-end ND LGSSM + leave-one-out (p(y_t | y_-t)) + "image-formula" X_t
+ KKT adversarial optimization at a fixed time t (default t=5)
+ Figure with 4 panels, read left-to-right and top-to-bottom:
  (1) State component s1 over time with CI (base vs adversarial)
  (2) Observation-space geometry (ellipses + mu_t, y_t, y*)
  (3) State component s2 over time with CI (base vs adversarial)
  (4) State-space trajectory (s1 vs s2) without CI

Notes:
- This script expects n_y = 2 for the ellipse panel.
- Q_t and R_t are projected to PSD for numerical stability (since your Q0/R0 were not symmetric).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared_ssm.artifacts import cached_npz
from shared_ssm.artifacts import data_path_for_plot
from shared_ssm.artifacts import figures_dir_for


# ============================================================
# Ellipse drawing helpers (2D)
# ============================================================
def _ellipse_points_from_quad(
    center: np.ndarray,
    shape_inv: np.ndarray,   # M in (y-c)^T M (y-c) = level
    level: float,
    n: int = 360,
) -> np.ndarray:
    center = np.asarray(center, dtype=float).reshape(2,)
    M = project_to_psd(symmetrize(np.asarray(shape_inv, dtype=float).reshape(2, 2)))

    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-14)
    radii = np.sqrt(level / w)

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2,n)
    pts = (V @ (radii[:, None] * circle)).T + center[None, :]
    return pts


def _fill_ellipse(ax, pts: np.ndarray, *, alpha: float, label: str | None = None) -> None:
    ax.fill(pts[:, 0], pts[:, 1], alpha=alpha, label=label)


# ============================================================
# Final 4-panel figure
# ============================================================
def _set_plot_theme() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "axes.grid": True,
    })


def _style_axis(ax, *, facecolor: str = "#FBFBFD") -> None:
    ax.set_facecolor(facecolor)
    # Draw a full black frame so each panel is visually boxed.
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, alpha=0.20)


def _points_limits(points: list[np.ndarray], pad_frac: float = 0.10) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute nice x/y limits from a list of (N,2) point arrays or (2,) vectors.
    """
    arrs = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        if p.size > 0:
            arrs.append(p[:, :2])
    if not arrs:
        return (-1.0, 1.0), (-1.0, 1.0)

    P = np.vstack(arrs)
    xmin, ymin = np.min(P[:, 0]), np.min(P[:, 1])
    xmax, ymax = np.max(P[:, 0]), np.max(P[:, 1])

    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)

    # force a bit more square-ish limits for prettier geometry/trajectory panels
    d = max(dx, dy)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    pad = pad_frac * d + 1e-6

    return (cx - 0.5 * d - pad, cx + 0.5 * d + pad), (cy - 0.5 * d - pad, cy + 0.5 * d + pad)


def plot_attack_figure_four_panels(
    *,
    t: int,
    y_t: np.ndarray,
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    X_t: np.ndarray,
    y_star: np.ndarray,
    obj_star: float,
    epsilon: float,
    x_true: np.ndarray,
    m_smooth_base: np.ndarray,
    P_smooth_base: np.ndarray,
    m_smooth_adv: np.ndarray,
    P_smooth_adv: np.ndarray,
    outpath: str,
) -> None:
    """
    Layout (2x2), read left-to-right and then top-to-bottom:
      Top row:
        (A) s1 vs time (with confidence interval)
        (B) Attack geometry (ellipses in observation space)
      Bottom row:
        (C) s2 vs time (with confidence interval)
        (D) State-space trajectory (s1 vs s2)
    """
    if y_t.shape != (2,) or mu_t.shape != (2,) or y_star.shape != (2,):
        raise ValueError("This plot expects n_y=2 (y_t/mu_t/y_star must be shape (2,)).")
    if Sigma_t.shape != (2, 2):
        raise ValueError("Sigma_t must be (2,2).")
    if X_t.shape[1] != 2:
        raise ValueError("X_t must be (n_x,2).")
    if x_true.shape[1] < 2:
        raise ValueError("Need at least 2 state dims to show x1 vs x2.")

    # Si mantienes los helpers que te pasé antes, se usarán aquí.
    try:
        _set_plot_theme()
    except NameError:
        plt.rcParams.update({
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.7,
        })

    def _style_axis_local(ax):
        ax.set_facecolor("#FBFBFD")
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(1.0)
        ax.grid(True, alpha=0.20)

    def _points_limits_local(points: list[np.ndarray], pad_frac: float = 0.10):
        arrs = []
        for p in points:
            p = np.asarray(p, dtype=float)
            if p.ndim == 1:
                p = p[None, :]
            if p.size > 0:
                arrs.append(p[:, :2])
        if not arrs:
            return (-1.0, 1.0), (-1.0, 1.0)
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

    # Pastel palette aligned with the requested reference figure.
    c_base = "#5B84B1"
    c_adv = "#D9826B"
    c_true = "#3A3A3A"
    c_constraint = "#7C8F69"
    c_objective = "#9C7B62"
    c_mu = "#35B24A"
    c_y = "#E53935"
    c_star = "#8E63CE"
    c_attack_line = "#8CC7FF"

    T = x_true.shape[0] - 1
    tt = np.arange(T + 1)
    z = 1.96

    # Figura más grande para que respiren leyendas + ejes
    fig = plt.figure(figsize=(13.6, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.04, hspace=0.035,
    )

    # Read the panels from left to right and then top to bottom:
    # s1 impact, observation-space ellipses, s2 impact, and state path.
    ax_ts1 = fig.add_subplot(gs[0, 0])   # s1 vs time
    ax_geom = fig.add_subplot(gs[0, 1])  # geometry ellipses
    ax_ts2 = fig.add_subplot(gs[1, 0])   # s2 vs time
    ax_traj = fig.add_subplot(gs[1, 1])  # state-space s1 vs s2

    for ax in [ax_ts1, ax_ts2, ax_geom, ax_traj]:
        try:
            _style_axis(ax)
        except NameError:
            _style_axis_local(ax)
        ax.set_box_aspect(0.62)

    # =========================================================
    # TIME-SERIES PANELS: s1 at top-left and s2 at bottom-left
    # =========================================================
    def _plot_state_time(ax, idx: int, state_label: str) -> None:
        x_line = x_true[:, idx]
        m_base = m_smooth_base[:, idx]
        sd_base = np.sqrt(np.maximum(P_smooth_base[:, idx, idx], 0.0))

        m_adv = m_smooth_adv[:, idx]
        sd_adv = np.sqrt(np.maximum(P_smooth_adv[:, idx, idx], 0.0))

        # Mark the attacked time with a thin guide like in the reference.
        ax.axvline(t, color=c_attack_line, linewidth=1.0, alpha=0.95, zorder=0)

        # Base
        ax.fill_between(
            tt, m_base - z * sd_base, m_base + z * sd_base,
            color=c_base, alpha=0.18, zorder=1
        )
        ax.plot(tt, m_base, color=c_base, linewidth=1.4, linestyle="--", label=r"Estimated $s_t$", zorder=3)

        # Adversarial
        ax.fill_between(
            tt, m_adv - z * sd_adv, m_adv + z * sd_adv,
            color=c_adv, alpha=0.13, zorder=1
        )
        ax.plot(tt, m_adv, color=c_adv, linewidth=1.4, linestyle="--", label=r"Attacked $s_t$", zorder=3)

        # True state
        ax.plot(
            tt, x_line, color=c_true, marker="o", markersize=2.6,
            linewidth=1.0, alpha=0.9, label=r"Actual $s_t$", zorder=2
        )

        # Marcadores en t
        ax.scatter([t], [x_line[t]], color=c_true, s=28, zorder=5)
        ax.scatter([t], [m_base[t]], color=c_base, s=30, zorder=5)
        ax.scatter([t], [m_adv[t]], color=c_adv, s=34, marker="D", zorder=5)
        ax.set_title(fr"${state_label}$ impact", fontweight="normal", pad=3.0)
        ax.set_xlabel("time t")
        ax.set_ylabel(fr"${state_label}$")
        ax.margins(x=0.02)

        # Leyenda compacta dentro (sin montarse)
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.94,
            handlelength=1.6,
            borderpad=0.45,
        )

    _plot_state_time(ax_ts1, idx=0, state_label="s_1")
    _plot_state_time(ax_ts2, idx=1, state_label="s_2")

    # =========================================================
    # TOP-RIGHT: geometry (ellipses)
    # =========================================================
    Sigma_t = project_to_psd(Sigma_t)
    Sigma_inv = inv_psd(Sigma_t)

    pts_constraint = _ellipse_points_from_quad(mu_t, Sigma_inv, epsilon)

    M = project_to_psd(symmetrize(X_t.T @ X_t))
    M_plot = project_to_psd(M, eps=1e-8)  # regularización solo para dibujar
    pts_obj = _ellipse_points_from_quad(y_t, M_plot, max(obj_star, 1e-10))

    ax_geom.fill(
        pts_constraint[:, 0], pts_constraint[:, 1],
        color=c_constraint, alpha=0.10
    )
    constraint_line, = ax_geom.plot(
        pts_constraint[:, 0], pts_constraint[:, 1],
        color=c_constraint, linewidth=1.6, alpha=0.95, label="Constraint"
    )

    ax_geom.fill(
        pts_obj[:, 0], pts_obj[:, 1],
        color=c_objective, alpha=0.08
    )
    objective_line, = ax_geom.plot(
        pts_obj[:, 0], pts_obj[:, 1],
        color=c_objective, linewidth=1.6, linestyle="--", alpha=0.95, label="Objective"
    )

    mu_handle = ax_geom.scatter(
        [mu_t[0]], [mu_t[1]], s=60, marker="x", linewidths=1.6, color=c_mu,
        label=r"$\hat{o}_{-t}$", zorder=5
    )
    y_handle = ax_geom.scatter(
        [y_t[0]], [y_t[1]], s=60, marker="x", linewidths=1.6, color=c_y,
        label=r"$o_t$", zorder=6
    )
    star_handle = ax_geom.scatter(
        [y_star[0]], [y_star[1]], s=90, marker="x", linewidths=1.8, color=c_star,
        label=r"$o_t^{\mathrm{adv}}$", zorder=7
    )

    try:
        (xlim_g, ylim_g) = _points_limits([pts_constraint, pts_obj, y_t, mu_t, y_star], pad_frac=0.12)
    except NameError:
        (xlim_g, ylim_g) = _points_limits_local([pts_constraint, pts_obj, y_t, mu_t, y_star], pad_frac=0.12)

    ax_geom.set_xlim(*xlim_g)
    ax_geom.set_ylim(*ylim_g)

    ax_geom.set_title(fr"Attack geometry at t={t} ($\epsilon$=90%)", fontweight="normal", pad=3.0)
    ax_geom.set_xlabel(r"$o_1$")
    ax_geom.set_ylabel(r"$o_2$")
    legend_regions = ax_geom.legend(
        handles=[constraint_line, objective_line],
        loc="upper left",
        frameon=True,
        framealpha=0.94,
    )
    ax_geom.add_artist(legend_regions)
    ax_geom.legend(
        handles=[mu_handle, y_handle, star_handle],
        loc="upper right",
        frameon=True,
        framealpha=0.94,
    )

    # =========================================================
    # BOTTOM-RIGHT: trajectory in state-space (x1 vs x2)
    # =========================================================
    x_true_xy = x_true[:, :2]
    base_xy = m_smooth_base[:, :2]
    adv_xy = m_smooth_adv[:, :2]

    ax_traj.plot(
        x_true_xy[:, 0], x_true_xy[:, 1],
        color=c_true, linewidth=1.2, marker="o", markersize=2.4,
        alpha=0.85, label=r"Actual $s_t$"
    )
    ax_traj.plot(
        base_xy[:, 0], base_xy[:, 1],
        color=c_base, linewidth=1.5, linestyle="--", marker="o", markersize=2.4,
        label=r"Estimated $s_t$"
    )
    ax_traj.plot(
        adv_xy[:, 0], adv_xy[:, 1],
        color=c_adv, linewidth=1.5, linestyle="--", marker="o", markersize=2.4,
        label=r"Attacked $s_t$"
    )

    start_handle = ax_traj.scatter(
        [x_true_xy[0, 0]],
        [x_true_xy[0, 1]],
        marker="s",
        s=68,
        color="#2E7D32",
        edgecolor="black",
        linewidths=0.4,
        zorder=8,
        label="Start",
    )
    finished_handle = ax_traj.scatter(
        [x_true_xy[-1, 0]],
        [x_true_xy[-1, 1]],
        marker="X",
        s=78,
        color="#B71C1C",
        edgecolor="black",
        linewidths=0.4,
        zorder=8,
        label="Finished",
    )

    # Highlight the attacked time with explicit black/blue/orange points.
    ax_traj.scatter(
        [x_true_xy[t, 0]], [x_true_xy[t, 1]],
        s=54, marker="o", color=c_true, edgecolor="white", linewidths=0.5, zorder=9
    )
    ax_traj.scatter(
        [base_xy[t, 0]], [base_xy[t, 1]],
        s=54, marker="o", color=c_base, edgecolor="white", linewidths=0.5, zorder=9
    )
    ax_traj.scatter(
        [adv_xy[t, 0]], [adv_xy[t, 1]],
        s=54, marker="o", color=c_adv, edgecolor="white", linewidths=0.5, zorder=9
    )

    ax_traj.plot([base_xy[t, 0], adv_xy[t, 0]], [base_xy[t, 1], adv_xy[t, 1]],
                 color=c_adv, linestyle=":", linewidth=1.2, alpha=0.9, zorder=6)

    try:
        (xlim_tr, ylim_tr) = _points_limits([x_true_xy, base_xy, adv_xy], pad_frac=0.10)
    except NameError:
        (xlim_tr, ylim_tr) = _points_limits_local([x_true_xy, base_xy, adv_xy], pad_frac=0.10)

    ax_traj.set_xlim(*xlim_tr)
    ax_traj.set_ylim(*ylim_tr)

    ax_traj.set_title(r"Hidden State-space($s_1$ vs $s_2$)", fontweight="normal", pad=3.0)
    ax_traj.set_xlabel(r"$s_1$")
    ax_traj.set_ylabel(r"$s_2$")
    ax_traj.legend(
        handles=[ax_traj.lines[0], ax_traj.lines[1], ax_traj.lines[2], start_handle, finished_handle],
        loc="upper right",
        frameon=True,
        framealpha=0.94,
    )

    # Título global
    fig.suptitle(
        f"KKT adversarial observation attack at t={t} | ε={epsilon:.3f} | objective={obj_star:.4f}",
        fontsize=14,
        fontweight="semibold",
        y=0.995,
    )

    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)

    fig.align_ylabels([ax_ts1, ax_ts2, ax_geom, ax_traj])

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # PNG ONLY (como me pediste)
    fig.savefig(outpath, facecolor="white", dpi=300)
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    # ---- your setup
    n_x = n_y = n_u = 2
    T = 10
    candidate_seeds = [2058]
    t = 5
    epsilon_prob = 0.90
    epsilon_percent = int(round(100.0 * epsilon_prob))
    matrix_tag = "swap12"
    # Probability mass used for the 2D observation-space chi-square constraint.
    epsilon = -2.0 * np.log(1.0 - epsilon_prob)
    force_recompute = False

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))

    # your matrices
    # Swap coordinates 1 <-> 2 by permuting rows/columns in the 2D system.
    A0 = np.array([[0.70, -0.15],
                   [0.40, 0.65]], dtype=float)

    B0 = np.array([[0.70, -0.15],
                   [1.40, 1.65]], dtype=float)

    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = np.array([[0.2100, -0.2325],
                   [-0.2325, 0.4800]], dtype=float)

    R0 = np.array([[0.7140, 0.0525],
                   [0.0525, 0.2730]], dtype=float)

    # project covariances to PSD (recommended)
    Q0 = project_to_psd(Q0)
    R0 = project_to_psd(R0)

    # drift (optional)
    dA = np.zeros_like(A0)
    dB = np.zeros_like(B0)
    dH = np.zeros_like(H0)
    dD = np.zeros_like(D0)
    dQ = np.zeros_like(Q0)
    dR = np.zeros_like(R0)

    x0 = np.array([0.5, 0.5], dtype=float)
    m0 = x0.copy()
    P0 = 0.05 * np.eye(n_x)

    def run_seed(seed: int) -> str:
        outpath = os.path.join(out_dir, f"attack_four_panels_t{t}_T{T}_seed{seed}.png")
        cache_plot_path = outpath.replace(".png", f"_{matrix_tag}_eps{epsilon_percent}.png")
        data_path = data_path_for_plot(cache_plot_path)

        def compute_plot_data() -> dict[str, np.ndarray | float | int]:
            # Simulate and solve the attack for one candidate seed.
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
            y_t = y[t].copy()

            y_star, obj_star = solve_kkt_max_quadratic_over_ellipsoid(
                X=X_t, y_t=y_t, mu=mu_t, Sigma=Sigma_t, epsilon=epsilon
            )

            m_filt_b, P_filt_b, m_pred_b, P_pred_b = kalman_filter_nd(
                y=y, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=m0, P0=P0
            )
            m_smooth_b, P_smooth_b = rts_smoother_nd(
                m_filt=m_filt_b, P_filt=P_filt_b,
                m_pred=m_pred_b, P_pred=P_pred_b,
                A_t=mats["A_t"]
            )

            y_adv = y.copy()
            y_adv[t] = y_star

            m_filt_a, P_filt_a, m_pred_a, P_pred_a = kalman_filter_nd(
                y=y_adv, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=m0, P0=P0
            )
            m_smooth_a, P_smooth_a = rts_smoother_nd(
                m_filt=m_filt_a, P_filt=P_filt_a,
                m_pred=m_pred_a, P_pred=P_pred_a,
                A_t=mats["A_t"]
            )

            Sinv = inv_psd(Sigma_t)
            constr_val = float((y_star - mu_t).T @ Sinv @ (y_star - mu_t))

            return {
                "t": t,
                "T": T,
                "seed": seed,
                "epsilon": epsilon,
                "x_true": x,
                "y_t": y_t,
                "mu_t": mu_t,
                "Sigma_t": Sigma_t,
                "X_t": X_t,
                "y_star": y_star,
                "obj_star": obj_star,
                "constr_val": constr_val,
                "m_smooth_base": m_smooth_b,
                "P_smooth_base": P_smooth_b,
                "m_smooth_adv": m_smooth_a,
                "P_smooth_adv": P_smooth_a,
            }

        data = cached_npz(data_path, compute_plot_data, force=force_recompute)

        y_t = data["y_t"]
        mu_t = data["mu_t"]
        Sigma_t = data["Sigma_t"]
        X_t = data["X_t"]
        y_star = data["y_star"]
        obj_star = float(data["obj_star"])
        constr_val = float(data["constr_val"])

        print(f"\n[seed={seed}, t={t}] constraint value = {constr_val:.6f} (should be <= epsilon={epsilon})")
        print(f"[seed={seed}, t={t}] objective value  = {obj_star:.6f}")
        print(f"[seed={seed}, t={t}] o_t              = {y_t}")
        print(f"[seed={seed}, t={t}] o_-t             = {mu_t}")
        print(f"[seed={seed}, t={t}] o_t^{{adv}}      = {y_star}")

        plot_attack_figure_four_panels(
            t=t,
            y_t=y_t, mu_t=mu_t, Sigma_t=Sigma_t, X_t=X_t,
            y_star=y_star, obj_star=obj_star, epsilon=epsilon,
            x_true=data["x_true"],
            m_smooth_base=data["m_smooth_base"], P_smooth_base=data["P_smooth_base"],
            m_smooth_adv=data["m_smooth_adv"], P_smooth_adv=data["P_smooth_adv"],
            outpath=outpath,
        )
        print(f"Saved figure to: {outpath}")
        return outpath

    saved_paths = [run_seed(seed) for seed in candidate_seeds]

    print("\nGenerated candidate figures:")
    for path in saved_paths:
        print(path)


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
from shared_ssm.legacy import kalman_filter_nd_previous_observation as kalman_filter_nd
from shared_ssm.legacy import loo_values_nd_previous_observation as loo_values_nd
from shared_ssm.legacy import rts_smoother_nd
from shared_ssm.legacy import simulate_lgssm_nd_previous_observation as simulate_lgssm_nd
from shared_ssm.legacy import solve_kkt_max_quadratic_over_ellipsoid


if __name__ == "__main__":
    main()
