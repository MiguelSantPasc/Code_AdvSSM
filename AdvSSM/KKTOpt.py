#!/usr/bin/env python3
"""
KKTOpt.py

End-to-end ND LGSSM + leave-one-out (p(y_t | y_-t)) + "image-formula" X_t
+ KKT adversarial optimization at a fixed time t (default t=5)
+ Beautiful figure with 4 panels:

LEFT COLUMN:
  (1) Observation-space geometry (ellipses + mu_t, y_t, y*)
  (2) State component x1 (component 0) over time with CI (base vs adversarial)
  (3) State component x2 (component 1) over time with CI (base vs adversarial)

RIGHT COLUMN:
  (4) State-space trajectory (x1 vs x2) WITHOUT CI (true vs base vs adversarial)

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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.55)
    ax.spines["bottom"].set_alpha(0.55)
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
    NUEVO layout (2x2):
      Top row:
        (A) x1 vs time (con CI)
        (B) x2 vs time (con CI)
      Bottom row:
        (C) Geometría del ataque (elipses en y-space)
        (D) Trayectoria en espacio de estados (x1 vs x2)
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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.55)
        ax.spines["bottom"].set_alpha(0.55)
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

    # ---------- palette ----------
    c_base = "#2F6FA8"   # azul
    c_adv = "#C85B4F"    # terracota
    c_true = "#2F2F2F"   # charcoal
    c_geom1 = "#6A5ACD"  # constraint
    c_geom2 = "#2CA58D"  # objective level-set
    c_mu = "#6A5ACD"
    c_y = "#111111"
    c_star = "#D4A017"
    c_attack_band = "#E9D8A6"

    T = x_true.shape[0] - 1
    tt = np.arange(T + 1)
    z = 1.96

    # Figura más grande para que respiren leyendas + ejes
    fig = plt.figure(figsize=(17.0, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.08],
        wspace=0.18, hspace=0.16,
    )

    # ARRIBA: series temporales
    ax_ts1 = fig.add_subplot(gs[0, 0])   # x1 vs t
    ax_ts2 = fig.add_subplot(gs[0, 1])   # x2 vs t

    # ABAJO: elipses + trayectoria
    ax_geom = fig.add_subplot(gs[1, 0])  # geometry ellipses
    ax_traj = fig.add_subplot(gs[1, 1])  # state-space x1 vs x2

    for ax in [ax_ts1, ax_ts2, ax_geom, ax_traj]:
        try:
            _style_axis(ax)
        except NameError:
            _style_axis_local(ax)

    # =========================================================
    # TOP ROW: time-series panels (x1 and x2)
    # =========================================================
    def _plot_state_time(ax, idx: int, panel_title: str) -> None:
        x_line = x_true[:, idx]
        m_base = m_smooth_base[:, idx]
        sd_base = np.sqrt(np.maximum(P_smooth_base[:, idx, idx], 0.0))

        m_adv = m_smooth_adv[:, idx]
        sd_adv = np.sqrt(np.maximum(P_smooth_adv[:, idx, idx], 0.0))

        # Banda vertical del instante atacado
        ax.axvspan(t - 0.35, t + 0.35, color=c_attack_band, alpha=0.28, zorder=0)
        ax.axvline(t, color="#8D6E63", linewidth=1.0, alpha=0.45)

        # Base
        ax.fill_between(
            tt, m_base - z * sd_base, m_base + z * sd_base,
            color=c_base, alpha=0.18, label="Base 95% CI", zorder=1
        )
        ax.plot(tt, m_base, color=c_base, linewidth=1.9, label="Base RTS mean", zorder=3)

        # Adversarial
        ax.fill_between(
            tt, m_adv - z * sd_adv, m_adv + z * sd_adv,
            color=c_adv, alpha=0.13, label="Adv 95% CI", zorder=1
        )
        ax.plot(tt, m_adv, color=c_adv, linewidth=1.9, linestyle="--", label="Adv RTS mean", zorder=3)

        # True state
        ax.plot(
            tt, x_line, color=c_true, marker="o", markersize=2.6,
            linewidth=1.1, alpha=0.9, label=f"True x[{idx}]", zorder=2
        )

        # Marcadores en t
        ax.scatter([t], [x_line[t]], color=c_true, s=28, zorder=5)
        ax.scatter([t], [m_base[t]], color=c_base, s=30, zorder=5)
        ax.scatter([t], [m_adv[t]], color=c_adv, s=34, marker="D", zorder=5)

        ax.set_title(panel_title, loc="left", fontweight="semibold")
        ax.set_xlabel("time t")
        ax.set_ylabel(f"x[{idx}]")
        ax.margins(x=0.02)

        # Leyenda compacta dentro (sin montarse)
        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.94,
            ncol=2,
            columnspacing=0.9,
            handlelength=1.8,
            borderpad=0.45,
        )

    _plot_state_time(ax_ts1, idx=0, panel_title="(A) Attack impact on x1 over time")
    _plot_state_time(ax_ts2, idx=1, panel_title="(B) Attack impact on x2 over time")

    # =========================================================
    # BOTTOM-LEFT: geometry (ellipses)
    # =========================================================
    Sigma_t = project_to_psd(Sigma_t)
    Sigma_inv = inv_psd(Sigma_t)

    pts_constraint = _ellipse_points_from_quad(mu_t, Sigma_inv, epsilon)

    M = project_to_psd(symmetrize(X_t.T @ X_t))
    M_plot = project_to_psd(M, eps=1e-8)  # regularización solo para dibujar
    pts_obj = _ellipse_points_from_quad(y_t, M_plot, max(obj_star, 1e-10))

    ax_geom.fill(
        pts_constraint[:, 0], pts_constraint[:, 1],
        color=c_geom1, alpha=0.14, label="Constraint region"
    )
    ax_geom.plot(
        pts_constraint[:, 0], pts_constraint[:, 1],
        color=c_geom1, linewidth=2.0, alpha=0.95
    )

    ax_geom.fill(
        pts_obj[:, 0], pts_obj[:, 1],
        color=c_geom2, alpha=0.10, label="Objective level-set"
    )
    ax_geom.plot(
        pts_obj[:, 0], pts_obj[:, 1],
        color=c_geom2, linewidth=1.8, linestyle="--", alpha=0.95
    )

    ax_geom.scatter([mu_t[0]], [mu_t[1]], s=55, marker="o", color=c_mu, label=r"$o_{-t}$", zorder=5)
    ax_geom.scatter([y_t[0]], [y_t[1]], s=65, marker="x", linewidths=2.0, color=c_y, label=r"$o_t$", zorder=6)
    ax_geom.scatter([y_star[0]], [y_star[1]], s=120, marker="*", color=c_star, edgecolor="black",
                    linewidths=0.4, label=r"$o^{adv}_t$", zorder=7)

    ax_geom.plot([y_t[0], y_star[0]], [y_t[1], y_star[1]],
                 color=c_adv, linewidth=1.4, alpha=0.85, linestyle="-.", zorder=4)
    
    ax_geom.annotate(
        "",
        xy=(y_star[0], y_star[1]),      # destino: o_t^{adv}
        xytext=(y_t[0], y_t[1]),        # origen: o_t
        arrowprops=dict(
            arrowstyle="->",
            color=c_adv,
            lw=1.8,
            alpha=0.95,
            linestyle="-.",
            shrinkA=6,
            shrinkB=8,
            mutation_scale=14,
        ),
        zorder=4
    )

    try:
        (xlim_g, ylim_g) = _points_limits([pts_constraint, pts_obj, y_t, mu_t, y_star], pad_frac=0.12)
    except NameError:
        (xlim_g, ylim_g) = _points_limits_local([pts_constraint, pts_obj, y_t, mu_t, y_star], pad_frac=0.12)

    ax_geom.set_xlim(*xlim_g)
    ax_geom.set_ylim(*ylim_g)

    ax_geom.set_title(f"(C) Attack geometry at t={t}", loc="left", fontweight="semibold")
    ax_geom.set_xlabel(r"$o_t^x$")
    ax_geom.set_ylabel(r"$o_t^y$", labelpad=-75)
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.legend(loc="upper left", frameon=True, framealpha=0.94)

    # =========================================================
    # BOTTOM-RIGHT: trajectory in state-space (x1 vs x2)
    # =========================================================
    x_true_xy = x_true[:, :2]
    base_xy = m_smooth_base[:, :2]
    adv_xy = m_smooth_adv[:, :2]

    ax_traj.plot(
        x_true_xy[:, 0], x_true_xy[:, 1],
        color=c_true, linewidth=1.2, marker="o", markersize=2.4,
        alpha=0.85, label="True path"
    )
    ax_traj.plot(
        base_xy[:, 0], base_xy[:, 1],
        color=c_base, linewidth=2.0, label="Base RTS path"
    )
    ax_traj.plot(
        adv_xy[:, 0], adv_xy[:, 1],
        color=c_adv, linewidth=2.0, linestyle="--", label="Adv RTS path"
    )

    # Start/end
    ax_traj.scatter([x_true_xy[0, 0]], [x_true_xy[0, 1]],
                    marker="s", s=70, color="#2E7D32", edgecolor="black", linewidths=0.4, zorder=7, label="Start")
    ax_traj.scatter([x_true_xy[-1, 0]], [x_true_xy[-1, 1]],
                    marker="X", s=85, color="#B71C1C", edgecolor="black", linewidths=0.4, zorder=7, label="End")

    # Highlight attack time
    ax_traj.scatter([base_xy[t, 0]], [base_xy[t, 1]],
                    s=65, marker="o", color=c_base, edgecolor="white", linewidths=0.7, zorder=8, label="Base at t")
    ax_traj.scatter([adv_xy[t, 0]], [adv_xy[t, 1]],
                    s=95, marker="*", color=c_star, edgecolor="black", linewidths=0.5, zorder=9, label="Adv at t")

    ax_traj.plot([base_xy[t, 0], adv_xy[t, 0]], [base_xy[t, 1], adv_xy[t, 1]],
                 color=c_adv, linestyle=":", linewidth=1.2, alpha=0.9, zorder=6)

    try:
        (xlim_tr, ylim_tr) = _points_limits([x_true_xy, base_xy, adv_xy], pad_frac=0.10)
    except NameError:
        (xlim_tr, ylim_tr) = _points_limits_local([x_true_xy, base_xy, adv_xy], pad_frac=0.10)

    ax_traj.set_xlim(*xlim_tr)
    ax_traj.set_ylim(*ylim_tr)

    ax_traj.set_title("(D) State-space trajectory (x1 vs x2)", loc="left", fontweight="semibold")
    ax_traj.set_xlabel("x1 (component 0)")
    ax_traj.set_ylabel("x2 (component 1)")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.legend(loc="upper right", frameon=True, framealpha=0.94)

    # Título global
    fig.suptitle(
        f"KKT adversarial observation attack at t={t} | ε={epsilon:.3f} | objective={obj_star:.4f}",
        fontsize=14,
        fontweight="semibold",
        y=0.995,
    )

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
    T = 12
    seed = 2026
    t = T
    epsilon = 5.991  # typical 95% chi-square in 2D constraint
    force_recompute = False

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    outpath = os.path.join(out_dir, f"attack_four_panels_t{t}_T{T}_seed{seed}.png")
    data_path = data_path_for_plot(outpath)

    # your matrices
    A0 = np.array([[0.65, 0.40],
                   [-0.15, 0.70]], dtype=float)

    B0 = np.array([[1.65, 1.40],
                   [-0.15, 0.70]], dtype=float)

    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.3 * np.array([[1.6, -0.40],
                         [-1.15, 0.70]], dtype=float)

    R0 = 0.42 * np.array([[0.65, 0.40],
                         [-0.15, 1.70]], dtype=float)

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

    def compute_plot_data() -> dict[str, np.ndarray | float | int]:
        # ---- simulate
        x, y, u, mats = simulate_lgssm_nd(
            A0=A0, B0=B0, H0=H0, D0=D0,
            T=T, seed=seed, x0=x0,
            Q0=Q0, R0=R0,
            dA=dA, dB=dB, dH=dH, dD=dD, dQ=dQ, dR=dR,
            u_low=-0.5, u_high=0.5,
        )

        # ---- compute X_t, mu_t, Sigma_t at time t
        X_t, mu_t, Sigma_t = loo_values_nd(
            t=t,
            y=y, u=u,
            A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
            Q_t=mats["Q_t"], R_t=mats["R_t"],
            P0=P0, m0=m0,
        )
        y_t = y[t].copy()

        # ---- KKT solve
        y_star, obj_star = solve_kkt_max_quadratic_over_ellipsoid(
            X=X_t, y_t=y_t, mu=mu_t, Sigma=Sigma_t, epsilon=epsilon
        )

        # ---- RTS smoother on baseline y
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

        # ---- RTS smoother on adversarial y': replace only y[t]
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

    t = int(data["t"])
    epsilon = float(data["epsilon"])
    y_t = data["y_t"]
    mu_t = data["mu_t"]
    Sigma_t = data["Sigma_t"]
    X_t = data["X_t"]
    y_star = data["y_star"]
    obj_star = float(data["obj_star"])
    constr_val = float(data["constr_val"])

    print(f"\n[t={t}] constraint value = {constr_val:.6f} (should be <= epsilon={epsilon})")
    print(f"[t={t}] objective value  = {obj_star:.6f}")
    print(f"[t={t}] o_t              = {y_t}")
    print(f"[t={t}] o_-t             = {mu_t}")
    print(f"[t={t}] o_t^{{adv}}      = {y_star}")

    plot_attack_figure_four_panels(
        t=t,
        y_t=y_t, mu_t=mu_t, Sigma_t=Sigma_t, X_t=X_t,
        y_star=y_star, obj_star=obj_star, epsilon=epsilon,
        x_true=data["x_true"],
        m_smooth_base=data["m_smooth_base"], P_smooth_base=data["P_smooth_base"],
        m_smooth_adv=data["m_smooth_adv"], P_smooth_adv=data["P_smooth_adv"],
        outpath=outpath,
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
from shared_ssm.legacy import kalman_filter_nd_previous_observation as kalman_filter_nd
from shared_ssm.legacy import loo_values_nd_previous_observation as loo_values_nd
from shared_ssm.legacy import rts_smoother_nd
from shared_ssm.legacy import simulate_lgssm_nd_previous_observation as simulate_lgssm_nd
from shared_ssm.legacy import solve_kkt_max_quadratic_over_ellipsoid


if __name__ == "__main__":
    main()
