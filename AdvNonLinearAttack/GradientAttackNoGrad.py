#!/usr/bin/env python3
"""
GradientAttackNoGrad.py

End-to-end ND LGSSM + leave-one-out attack region + white-box gradient attack
on E[g(x_t) | y_t', y_-t] with projection onto the feasible attack region.

Key corrections / design choices:
- Controls are indexed consistently with u[k].
- We use u shape = (T+1, n_u), so:
    y_k     = H_k x_k + D_k u_k + v_k
    x_{k+1} = A_k x_k + B_k u_k + w_{k+1}
- The leave-one-out distribution p(x_t | y_-t) is obtained cleanly by
  running KF + RTS while skipping the measurement update at time t.
- The attack optimizes y_t' inside the ellipsoid:
      (y_t' - mu_t)^T Sigma_t^{-1} (y_t' - mu_t) <= epsilon
  where p(y_t | y_-t) = N(mu_t, Sigma_t).
- This version intentionally sets g_grad=None in the main experiment, so the
  attack uses finite-difference Jacobians instead of an analytic gradient.

Panels:
(A) x1 over time (base vs adversarial)
(B) x2 over time (base vs adversarial)
(C) Feasible attack ellipse + optimization path in observation space
(D) State-space trajectory (x1 vs x2)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from shared_ssm.artifacts import data_path_for_plot
from shared_ssm.artifacts import load_npz
from shared_ssm.artifacts import save_npz


# ============================================================
# Plot helpers
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
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    pts = (V @ (radii[:, None] * circle)).T + center[None, :]
    return pts


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
    if y_t.shape != (2,) or mu_t.shape != (2,) or y_star.shape != (2,):
        raise ValueError("This plot expects n_y = 2.")
    if Sigma_t.shape != (2, 2):
        raise ValueError("Sigma_t must be (2, 2).")
    if x_true.shape[1] < 2:
        raise ValueError("Need at least 2 latent state dimensions.")

    _set_plot_theme()

    c_base = "#2F6FA8"
    c_adv = "#C85B4F"
    c_true = "#2F2F2F"
    c_mu = "#6A5ACD"
    c_feasible = "#6A5ACD"
    c_star = "#D4A017"
    c_attack_band = "#E9D8A6"
    c_path = "#2CA58D"

    T = x_true.shape[0] - 1
    tt = np.arange(T + 1)
    z = 1.96

    fig = plt.figure(figsize=(17.0, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.08],
        wspace=0.18, hspace=0.16,
    )

    ax_ts1 = fig.add_subplot(gs[0, 0])
    ax_ts2 = fig.add_subplot(gs[0, 1])
    ax_geom = fig.add_subplot(gs[1, 0])
    ax_traj = fig.add_subplot(gs[1, 1])

    for ax in [ax_ts1, ax_ts2, ax_geom, ax_traj]:
        _style_axis(ax)

    # --------------------------------------------------------
    # Top row: x1 and x2 over time
    # --------------------------------------------------------
    def _plot_state_time(ax, idx: int, panel_title: str) -> None:
        x_line = x_true[:, idx]
        m_base = m_smooth_base[:, idx]
        sd_base = np.sqrt(np.maximum(P_smooth_base[:, idx, idx], 0.0))

        m_adv = m_smooth_adv[:, idx]
        sd_adv = np.sqrt(np.maximum(P_smooth_adv[:, idx, idx], 0.0))

        ax.axvspan(t - 0.35, t + 0.35, color=c_attack_band, alpha=0.28, zorder=0)
        ax.axvline(t, color="#8D6E63", linewidth=1.0, alpha=0.45)

        ax.fill_between(
            tt, m_base - z * sd_base, m_base + z * sd_base,
            color=c_base, alpha=0.18, label="Base 95% CI", zorder=1
        )
        ax.plot(tt, m_base, color=c_base, linewidth=1.9, label="Base RTS mean", zorder=3)

        ax.fill_between(
            tt, m_adv - z * sd_adv, m_adv + z * sd_adv,
            color=c_adv, alpha=0.13, label="Adv 95% CI", zorder=1
        )
        ax.plot(tt, m_adv, color=c_adv, linewidth=1.9, linestyle="--", label="Adv RTS mean", zorder=3)

        ax.plot(
            tt, x_line, color=c_true, marker="o", markersize=2.6,
            linewidth=1.1, alpha=0.9, label=f"True x[{idx}]", zorder=2
        )

        ax.scatter([t], [x_line[t]], color=c_true, s=28, zorder=5)
        ax.scatter([t], [m_base[t]], color=c_base, s=30, zorder=5)
        ax.scatter([t], [m_adv[t]], color=c_adv, s=34, marker="D", zorder=5)

        ax.set_title(panel_title, loc="left", fontweight="semibold")
        ax.set_xlabel("time k")
        ax.set_ylabel(f"x[{idx}]")
        ax.margins(x=0.02)
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

    # --------------------------------------------------------
    # Bottom-left: feasible attack region + optimization path
    # --------------------------------------------------------
    Sigma_t = project_to_psd(Sigma_t)
    Sigma_inv = inv_psd(Sigma_t)
    pts_constraint = _ellipse_points_from_quad(mu_t, Sigma_inv, epsilon)

    ax_geom.fill(
        pts_constraint[:, 0], pts_constraint[:, 1],
        color=c_feasible, alpha=0.14, label="Feasible attack region"
    )
    ax_geom.plot(
        pts_constraint[:, 0], pts_constraint[:, 1],
        color=c_feasible, linewidth=2.0, alpha=0.95
    )

    if y_path.shape[0] > 0:
        ax_geom.plot(
            y_path[:, 0], y_path[:, 1],
            color=c_path, linewidth=1.7, marker="o",
            markersize=3.3, alpha=0.9, label="Optimization path"
        )

    ax_geom.scatter([mu_t[0]], [mu_t[1]], s=55, marker="o", color=c_mu, label=r"$\mu_t$", zorder=5)
    ax_geom.scatter([y_t[0]], [y_t[1]], s=65, marker="x", linewidths=2.0, color="#111111", label=r"$y_t$", zorder=6)
    ax_geom.scatter([y_star[0]], [y_star[1]], s=120, marker="*", color=c_star, edgecolor="black",
                    linewidths=0.4, label=r"$y_t^\prime$", zorder=7)

    ax_geom.plot([y_t[0], y_star[0]], [y_t[1], y_star[1]],
                 color=c_adv, linewidth=1.4, alpha=0.85, linestyle="-.", zorder=4)

    xlim_g, ylim_g = _points_limits([pts_constraint, y_t, mu_t, y_star, y_path], pad_frac=0.12)
    ax_geom.set_xlim(*xlim_g)
    ax_geom.set_ylim(*ylim_g)

    ax_geom.set_title(f"(C) Feasible attack region at t={t}", loc="left", fontweight="semibold")
    ax_geom.set_xlabel(r"$o_t^x$")
    ax_geom.set_ylabel(r"$o_t^y$",labelpad=-75)
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.legend(loc="upper left", frameon=True, framealpha=0.94)

    # add inset-like text with final objective
    if obj_hist.size > 0:
        ax_geom.text(
            0.02, 0.03,
            f"steps = {obj_hist.size}",
            transform=ax_geom.transAxes,
            fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="#BBBBBB")
        )

    # --------------------------------------------------------
    # Bottom-right: state-space trajectory
    # --------------------------------------------------------
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

    ax_traj.scatter([x_true_xy[0, 0]], [x_true_xy[0, 1]],
                    marker="s", s=70, color="#2E7D32", edgecolor="black", linewidths=0.4, zorder=7, label="Start")
    ax_traj.scatter([x_true_xy[-1, 0]], [x_true_xy[-1, 1]],
                    marker="X", s=85, color="#B71C1C", edgecolor="black", linewidths=0.4, zorder=7, label="End")

    ax_traj.scatter([base_xy[t, 0]], [base_xy[t, 1]],
                    s=65, marker="o", color=c_base, edgecolor="white", linewidths=0.7, zorder=8, label="Base at t")
    ax_traj.scatter([adv_xy[t, 0]], [adv_xy[t, 1]],
                    s=95, marker="*", color=c_star, edgecolor="black", linewidths=0.5, zorder=9, label="Adv at t")

    ax_traj.plot([base_xy[t, 0], adv_xy[t, 0]], [base_xy[t, 1], adv_xy[t, 1]],
                 color=c_adv, linestyle=":", linewidth=1.2, alpha=0.9, zorder=6)

    xlim_tr, ylim_tr = _points_limits([x_true_xy, base_xy, adv_xy], pad_frac=0.10)
    ax_traj.set_xlim(*xlim_tr)
    ax_traj.set_ylim(*ylim_tr)

    ax_traj.set_title("(D) State-space trajectory (x1 vs x2)", loc="left", fontweight="semibold")
    ax_traj.set_xlabel("x1 (component 0)")
    ax_traj.set_ylabel("x2 (component 1)")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.legend(loc="lower right", frameon=True, framealpha=0.94)

    fig.suptitle(
        f"White-box point attack on E[g(x_t) | y_t', y_-t] at t={t} | ε={epsilon:.3f}",
        fontsize=14,
        fontweight="semibold",
        y=0.995,
    )

    fig.align_ylabels([ax_ts1, ax_ts2, ax_geom, ax_traj])

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, facecolor="white", dpi=300)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------
    n_x = n_y = n_u = 2
    T = 12
    seed = 2024

    # --------------------------------------------------------
    # Latent state:
    # x[0] = degradation / severity level
    # x[1] = deterioration rate / worsening trend
    #
    # Inputs:
    # u[0] = maintenance intensity  (positive = more maintenance)
    # u[1] = operational load       (positive = more stress/load)
    # --------------------------------------------------------

    A0 = np.array([
        [0.93, 0.22],
        [0.03, 0.86],
    ], dtype=float)

    B0 = np.array([
        [-0.28, 0.18],
        [-0.20, 0.24],
    ], dtype=float)

    # In your code H0 plays the role of C
    H0 = np.array([
        [1.10, 0.35],
        [0.55, 0.95],
    ], dtype=float)

    D0 = np.array([
        [-0.04, 0.20],
        [-0.10, 0.16],
    ], dtype=float)

    Q0 = np.array([
        [0.05, 0.004],
        [0.0048, 0.024],
    ], dtype=float)

    R0 = np.array([
        [0.015, 0.02],
        [0.02, 0.0420],
    ], dtype=float)

    Q0 = project_to_psd(Q0)
    R0 = project_to_psd(R0)

    x0 = np.array([0.35, 0.10], dtype=float)
    m0 = x0.copy()

    P0 = np.array([
        [0.040, 0.010],
        [0.010, 0.030],
    ], dtype=float)
    P0 = project_to_psd(P0)

    t = T
    epsilon = 5.991  # 95% chi-square threshold in 2D
    force_recompute = False

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"attack_on_g_nograd_four_panels_t{t}_T{T}_seed{seed}.png")
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
            m_smooth_base=data["m_smooth_base"], P_smooth_base=data["P_smooth_base"],
            m_smooth_adv=data["m_smooth_adv"], P_smooth_adv=data["P_smooth_adv"],
            obj_hist=data["obj_hist"],
            outpath=outpath,
        )
        print(f"\nSaved figure to: {outpath}")
        return

    dA = np.zeros_like(A0)
    dB = np.zeros_like(B0)
    dH = np.zeros_like(H0)
    dD = np.zeros_like(D0)
    dQ = np.zeros_like(Q0)
    dR = np.zeros_like(R0)

    # --------------------------------------------------------
    # Simulate
    # --------------------------------------------------------
    x, y, u, mats = simulate_lgssm_nd(
        A0=A0, B0=B0, H0=H0, D0=D0,
        T=T, seed=seed, x0=x0,
        Q0=Q0, R0=R0,
        dA=dA, dB=dB, dH=dH, dD=dD, dQ=dQ, dR=dR,
        u_low=-0.5, u_high=0.5,
    )

    # --------------------------------------------------------
    # Attack setup
    # --------------------------------------------------------
    def sigmoid(z: float | np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return 1.0 / (1.0 + np.exp(-z))


    def g_scalar(x_vec: np.ndarray) -> float:
        """
        Risk / alarm probability based on latent severity and trend.
        x_vec[0] = severity level
        x_vec[1] = worsening trend
        """
        x_vec = np.asarray(x_vec, dtype=float)
        x1, x2 = x_vec[0], x_vec[1]

        beta0 = -1.8
        beta1 =  1.4
        beta2 =  0.9
        beta3 =  0.8
        beta4 =  0.6

        z = beta0 + beta1 * x1 + beta2 * x2 + beta3 * x1 * x2 + beta4 * x1**2
        return float(sigmoid(z))


    def g_scalar_grad(x_vec: np.ndarray) -> np.ndarray:
        """
        Gradient of the scalar risk probability wrt x.
        Returns shape (n_x,)
        """
        x_vec = np.asarray(x_vec, dtype=float)
        x1, x2 = x_vec[0], x_vec[1]

        beta0 = -1.8
        beta1 =  1.4
        beta2 =  0.9
        beta3 =  0.8
        beta4 =  0.6

        z = beta0 + beta1 * x1 + beta2 * x2 + beta3 * x1 * x2 + beta4 * x1**2
        s = float(sigmoid(z))

        dz_dx1 = beta1 + beta3 * x2 + 2.0 * beta4 * x1
        dz_dx2 = beta2 + beta3 * x1

        # d sigma(z) / dz = sigma(z) * (1 - sigma(z))
        common = s * (1.0 - s)

        return np.array([
            common * dz_dx1,
            common * dz_dx2,
        ], dtype=float)

    M_star = np.array([0.0],dtype=float)  # target risk level (e.g. want to minimize risk)

    # --------------------------------------------------------
    # Attack optimization
    # --------------------------------------------------------
    y_star, attack_hist = white_box_point_attack_nd(
        t=t,
        y=y, u=u,
        A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
        Q_t=mats["Q_t"], R_t=mats["R_t"],
        P0=P0, m0=m0,
        epsilon=epsilon,
        M_star=M_star,
        g=g_scalar,
        g_grad=None,   # put None if you want finite differences
        eta=0.15,
        n_steps=500,
        n_mc=400,
        seed=2026,
    )

    mu_t = attack_hist["mu_t"]
    Sigma_t = attack_hist["Sigma_t"]
    y_t = y[t].copy()

    Sinv = inv_psd(Sigma_t)
    constr_val = float((y_star - mu_t).T @ Sinv @ (y_star - mu_t))

    print(f"\n[t={t}] feasibility value = {constr_val:.6f} (should be <= epsilon={epsilon})")
    print(f"[t={t}] y_t               = {y_t}")
    print(f"[t={t}] mu_t              = {mu_t}")
    print(f"[t={t}] y_star            = {y_star}")
    print(f"[t={t}] final objective   = {attack_hist['obj_hist'][-1]:.6f}")

    # --------------------------------------------------------
    # Baseline smoothing on original y
    # --------------------------------------------------------
    m_filt_b, P_filt_b, m_pred_b, P_pred_b = kalman_filter_nd(
        y=y, u=u,
        A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
        Q_t=mats["Q_t"], R_t=mats["R_t"],
        m0=m0, P0=P0,
    )
    m_smooth_b, P_smooth_b = rts_smoother_nd(
        m_filt=m_filt_b, P_filt=P_filt_b,
        m_pred=m_pred_b, P_pred=P_pred_b,
        A_t=mats["A_t"],
    )

    # --------------------------------------------------------
    # Adversarial smoothing: replace only y[t] by y_star
    # --------------------------------------------------------
    y_adv = y.copy()
    y_adv[t] = y_star

    m_filt_a, P_filt_a, m_pred_a, P_pred_a = kalman_filter_nd(
        y=y_adv, u=u,
        A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
        Q_t=mats["Q_t"], R_t=mats["R_t"],
        m0=m0, P0=P0,
    )
    m_smooth_a, P_smooth_a = rts_smoother_nd(
        m_filt=m_filt_a, P_filt=P_filt_a,
        m_pred=m_pred_a, P_pred=P_pred_a,
        A_t=mats["A_t"],
    )


    # ---- Compare E[g(x_t)] (baseline vs attacked) at the attacked time t
    mu_g_base_t, g_mean_base_t = estimate_E_g(
        m=m_smooth_b[t],
        P=P_smooth_b[t],
        g=g_scalar,
        n_mc=400,
        seed=7,
    )

    mu_g_adv_t, g_mean_adv_t = estimate_E_g(
        m=m_smooth_a[t],
        P=P_smooth_a[t],
        g=g_scalar,
        n_mc=400,
        seed=7,   # same seed => fair comparison
    )

    print("\n=== g(x_t) comparison at attacked time t ===")
    print(f"t = {t}")
    print(f"E[g(x_t) | y]        = {mu_g_base_t}")
    print(f"E[g(x_t) | y_attack] = {mu_g_adv_t}")

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------
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
        m_smooth_base=m_smooth_b, P_smooth_base=P_smooth_b,
        m_smooth_adv=m_smooth_a, P_smooth_adv=P_smooth_a,
        obj_hist=attack_hist["obj_hist"],
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
