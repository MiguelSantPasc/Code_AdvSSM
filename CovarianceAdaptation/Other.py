#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Import path setup
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from AdvSSM.KKTOpt import (
    simulate_lgssm_nd,
    kalman_filter_nd,
    loo_values_nd,
    solve_kkt_max_quadratic_over_ellipsoid,
    project_to_psd,
    inv_psd,
    _ellipse_points_from_quad,
    symmetrize,
)


# ============================================================
# Helpers
# ============================================================
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def predictive_obs_law_from_filter(
    *,
    t_attack: int,
    m_pred: np.ndarray,   # (T+1, n_x)
    P_pred: np.ndarray,   # (T+1, n_x, n_x)
    H_t: np.ndarray,
    D_t: np.ndarray,
    R_t: np.ndarray,
    u_controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes p(o_t_attack | o_1:t_attack-1) from KF outputs.
    """
    T = m_pred.shape[0] - 1

    def u_at(k: int) -> np.ndarray:
        return u_controls[k] if k < T else u_controls[T - 1]

    mu = H_t[t_attack] @ m_pred[t_attack] + D_t[t_attack] @ u_at(t_attack)
    Sigma = H_t[t_attack] @ P_pred[t_attack] @ H_t[t_attack].T + project_to_psd(R_t[t_attack])

    return mu, project_to_psd(Sigma)


def plot_tplus1_attack_comparison(
    *,
    t_mod: int,
    t_attack: int,
    epsilon: float,
    y_attack: np.ndarray,
    mu_base: np.ndarray,
    Sigma_base: np.ndarray,
    mu_mod: np.ndarray,
    Sigma_mod: np.ndarray,
    X_base: np.ndarray,
    X_mod: np.ndarray,
    y_star_base: np.ndarray,
    y_star_mod: np.ndarray,
    obj_base: float,
    obj_mod: float,
    u_dir: np.ndarray,
    outpath: str,
) -> None:
    """
    Compares attack at time t_attack = t_mod + 1:
      - baseline predictive plausible region
      - modified predictive plausible region
      - baseline objective level set
      - modified objective level set
      - true observation
      - both adversarial optima
      - direction u used to modify V_t_mod
    """
    Sigma_base = project_to_psd(Sigma_base)
    Sigma_mod = project_to_psd(Sigma_mod)

    Sigma_base_inv = inv_psd(Sigma_base)
    Sigma_mod_inv = inv_psd(Sigma_mod)

    pts_base = _ellipse_points_from_quad(mu_base, Sigma_base_inv, epsilon)
    pts_mod = _ellipse_points_from_quad(mu_mod, Sigma_mod_inv, epsilon)

    M_base = project_to_psd(symmetrize(X_base.T @ X_base), eps=1e-10)
    M_mod = project_to_psd(symmetrize(X_mod.T @ X_mod), eps=1e-10)

    pts_obj_base = _ellipse_points_from_quad(y_attack, M_base, max(obj_base, 1e-10))
    pts_obj_mod = _ellipse_points_from_quad(y_attack, M_mod, max(obj_mod, 1e-10))

    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Plausibility ellipses
    ax.fill(pts_base[:, 0], pts_base[:, 1], color="#6A5ACD", alpha=0.10,
            label=rf"Plausible region at t={t_attack} (baseline)")
    ax.plot(pts_base[:, 0], pts_base[:, 1], color="#6A5ACD", linewidth=2.0)

    ax.fill(pts_mod[:, 0], pts_mod[:, 1], color="#2CA58D", alpha=0.10,
            label=rf"Plausible region at t={t_attack} (after changing $V_{{{t_mod}}}$)")
    ax.plot(pts_mod[:, 0], pts_mod[:, 1], color="#2CA58D", linewidth=2.0, linestyle="--")

    # Objective ellipses
    ax.fill(pts_obj_base[:, 0], pts_obj_base[:, 1], color="#F4A261", alpha=0.08,
            label="Objective level-set (baseline)")
    ax.plot(pts_obj_base[:, 0], pts_obj_base[:, 1], color="#F4A261", linewidth=1.8)

    ax.fill(pts_obj_mod[:, 0], pts_obj_mod[:, 1], color="#E76F51", alpha=0.08,
            label="Objective level-set (modified)")
    ax.plot(pts_obj_mod[:, 0], pts_obj_mod[:, 1], color="#E76F51", linewidth=1.8, linestyle=":")

    # Centers
    ax.scatter(mu_base[0], mu_base[1], s=75, color="#6A5ACD", label=rf"$\mu_{{{t_attack}|{t_mod}}}^{{base}}$", zorder=5)
    ax.scatter(mu_mod[0], mu_mod[1], s=75, color="#2CA58D", label=rf"$\mu_{{{t_attack}|{t_mod}}}^{{mod}}$", zorder=5)

    # Actual observation
    ax.scatter(y_attack[0], y_attack[1], s=85, color="black", marker="x", linewidths=2.2,
               label=rf"$o_{{{t_attack}}}$", zorder=6)

    # Adversarial points
    ax.scatter(
        y_star_base[0], y_star_base[1],
        s=180, color="#D4A017", marker="*", edgecolor="black", linewidths=0.5,
        label=rf"$o^{{adv}}_{{{t_attack}}}$ baseline", zorder=7
    )
    ax.scatter(
        y_star_mod[0], y_star_mod[1],
        s=150, color="#C85B4F", marker="D", edgecolor="black", linewidths=0.5,
        label=rf"$o^{{adv}}_{{{t_attack}}}$ after changing $V_{{{t_mod}}}$", zorder=7
    )

    # Arrows from actual observation
    ax.annotate(
        "",
        xy=(y_star_base[0], y_star_base[1]),
        xytext=(y_attack[0], y_attack[1]),
        arrowprops=dict(arrowstyle="->", color="#D4A017", lw=1.8, mutation_scale=16),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(y_star_mod[0], y_star_mod[1]),
        xytext=(y_attack[0], y_attack[1]),
        arrowprops=dict(arrowstyle="->", color="#C85B4F", lw=1.8, linestyle="--", mutation_scale=16),
        zorder=4,
    )

    # Direction used to modify V_t
    arrow_scale = max(
        np.linalg.norm(y_star_base - mu_base),
        np.linalg.norm(y_star_mod - mu_mod),
        1.0,
    ) * 0.8

    ax.annotate(
        "",
        xy=(mu_base[0] + arrow_scale * u_dir[0], mu_base[1] + arrow_scale * u_dir[1]),
        xytext=(mu_base[0], mu_base[1]),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.2, mutation_scale=16),
        zorder=5,
    )
    ax.text(
        mu_base[0] + 1.05 * arrow_scale * u_dir[0],
        mu_base[1] + 1.05 * arrow_scale * u_dir[1],
        rf"$u$ used to modify $V_{{{t_mod}}}$",
        fontsize=11,
        color="#1f77b4",
        fontweight="bold",
    )

    # Limits
    all_pts = np.vstack([
        pts_base, pts_mod,
        pts_obj_base, pts_obj_mod,
        mu_base.reshape(1, 2),
        mu_mod.reshape(1, 2),
        y_attack.reshape(1, 2),
        y_star_base.reshape(1, 2),
        y_star_mod.reshape(1, 2),
    ])

    xmin, ymin = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
    xmax, ymax = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])

    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    d = max(dx, dy)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    pad = 0.15 * d

    ax.set_xlim(cx - 0.5 * d - pad, cx + 0.5 * d + pad)
    ax.set_ylim(cy - 0.5 * d - pad, cy + 0.5 * d + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(r"$o^x$")
    ax.set_ylabel(r"$o^y$")
    ax.set_title(
        f"Effect on attack at t+1 of changing covariance at t\n"
        f"modified time t={t_mod}, compared attack at t+1={t_attack}",
        fontsize=13,
    )
    ax.legend(loc="best", framealpha=0.95)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


# ============================================================
# Main experiment
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------
    n_x = n_y = n_u = 2
    T = 6         # must include time 13
    seed = 2025
    t_mod = 5        # change covariance here
    t_attack = 6      # compare attack here
    epsilon = 1.991
    lam = 150.5

    if t_attack != t_mod + 1:
        raise ValueError("This script is set up for t_attack = t_mod + 1.")

    out_dir = os.path.join(CURRENT_DIR, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(
        out_dir,
        f"attack_at_tplus1_after_change_t{t_mod}_to_{t_attack}_seed{seed}.png"
    )

    # --------------------------------------------------------
    # MODEL MATRICES
    # --------------------------------------------------------
    A0 = np.array([[0.65, 0.40],
                   [-0.55, 0.70]], dtype=float)

    B0 = np.array([[1.65, 1.40],
                   [-0.15, 0.70]], dtype=float)

    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.3 * np.array([[1.6, -0.40],
                         [-1.15, 0.70]], dtype=float)

    R0 = 0.42 * np.array([[0.65, 0.040],
                          [-0.15, 1.70]], dtype=float)

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

    # --------------------------------------------------------
    # SIMULATE BASELINE WORLD
    # --------------------------------------------------------
    x, y, u_controls, mats = simulate_lgssm_nd(
        A0=A0, B0=B0, H0=H0, D0=D0,
        T=T, seed=seed, x0=x0,
        Q0=Q0, R0=R0,
        dA=dA, dB=dB, dH=dH, dD=dD, dQ=dQ, dR=dR,
        u_low=-0.5, u_high=0.5,
    )

    # --------------------------------------------------------
    # STEP 1: baseline dangerous direction at t_mod
    # use baseline non-attack geometry at t_mod
    # --------------------------------------------------------
    X_tmod_base, mu_tmod_base, Sigma_tmod_base = loo_values_nd(
        t=t_mod,
        y=y, u=u_controls,
        A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
        Q_t=mats["Q_t"], R_t=mats["R_t"],
        P0=P0, m0=m0,
    )

    y_tmod = y[t_mod].copy()

    y_star_tmod_base, obj_tmod_base = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_tmod_base,
        y_t=y_tmod,
        mu=mu_tmod_base,
        Sigma=Sigma_tmod_base,
        epsilon=epsilon,
    )

    # Direction used to modify V_t in baseline setting
    u_dir = normalize(y_star_tmod_base - mu_tmod_base)

    # --------------------------------------------------------
    # STEP 2: baseline KF
    # --------------------------------------------------------
    m_filt_base, P_filt_base, m_pred_base, P_pred_base = kalman_filter_nd(
        y=y,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=m0,
        P0=P0,
    )

    # Explicit predictive law at t+1 from baseline KF
    mu_pred_base, Sigma_pred_base = predictive_obs_law_from_filter(
        t_attack=t_attack,
        m_pred=m_pred_base,
        P_pred=P_pred_base,
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        R_t=mats["R_t"],
        u_controls=u_controls,
    )

    # Also get X at t+1 under baseline world
    X_tnext_base, _, _ = loo_values_nd(
        t=t_attack,
        y=y, u=u_controls,
        A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
        Q_t=mats["Q_t"], R_t=mats["R_t"],
        P0=P0, m0=m0,
    )

    y_tnext = y[t_attack].copy()

    y_star_tnext_base, obj_tnext_base = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_tnext_base,
        y_t=y_tnext,
        mu=mu_pred_base,
        Sigma=Sigma_pred_base,
        epsilon=epsilon,
    )

    # --------------------------------------------------------
    # STEP 3: modify ONLY V_tmod, rerun KF completely
    # IMPORTANT: this recomputes m_{t|t} with the new covariance
    # and only then propagates to m_{t+1|t}
    # --------------------------------------------------------
    mats_mod = {k: np.copy(v) for k, v in mats.items()}
    mats_mod["R_t"][t_mod] = project_to_psd(
        mats_mod["R_t"][t_mod] + lam * np.outer(u_dir, u_dir)
    )

    m_filt_mod, P_filt_mod, m_pred_mod, P_pred_mod = kalman_filter_nd(
        y=y,
        u=u_controls,
        A_t=mats_mod["A_t"],
        B_t=mats_mod["B_t"],
        H_t=mats_mod["H_t"],
        D_t=mats_mod["D_t"],
        Q_t=mats_mod["Q_t"],
        R_t=mats_mod["R_t"],
        m0=m0,
        P0=P0,
    )

    # Explicit predictive law at t+1 from modified KF
    mu_pred_mod, Sigma_pred_mod = predictive_obs_law_from_filter(
        t_attack=t_attack,
        m_pred=m_pred_mod,
        P_pred=P_pred_mod,
        H_t=mats_mod["H_t"],
        D_t=mats_mod["D_t"],
        R_t=mats_mod["R_t"],
        u_controls=u_controls,
    )

    # X at t+1 under modified world
    X_tnext_mod, _, _ = loo_values_nd(
        t=t_attack,
        y=y, u=u_controls,
        A_t=mats_mod["A_t"], B_t=mats_mod["B_t"], H_t=mats_mod["H_t"], D_t=mats_mod["D_t"],
        Q_t=mats_mod["Q_t"], R_t=mats_mod["R_t"],
        P0=P0, m0=m0,
    )

    y_star_tnext_mod, obj_tnext_mod = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_tnext_mod,
        y_t=y_tnext,
        mu=mu_pred_mod,
        Sigma=Sigma_pred_mod,
        epsilon=epsilon,
    )

    # --------------------------------------------------------
    # PRINT KEY CHECKS
    # --------------------------------------------------------
    print("\n================ EXPERIMENT =================")
    print(f"T = {T}, seed = {seed}")
    print(f"Modified covariance at t = {t_mod}")
    print(f"Compared attack at t+1 = {t_attack}")
    print(f"lambda = {lam:.6f}")
    print("--------------------------------------------")
    print(f"Direction u used at time {t_mod}: {u_dir}")
    print("--------------------------------------------")
    print("CHECK PROPAGATION THROUGH FILTER:")
    print(f"m_{{t|t}} baseline   = {m_filt_base[t_mod]}")
    print(f"m_{{t|t}} modified   = {m_filt_mod[t_mod]}")
    print(f"||difference||       = {np.linalg.norm(m_filt_mod[t_mod] - m_filt_base[t_mod]):.6f}")
    print("--------------------------------------------")
    print(f"m_{{t+1|t}} baseline = {m_pred_base[t_attack]}")
    print(f"m_{{t+1|t}} modified = {m_pred_mod[t_attack]}")
    print(f"||difference||       = {np.linalg.norm(m_pred_mod[t_attack] - m_pred_base[t_attack]):.6f}")
    print("--------------------------------------------")
    print("Baseline predictive law at t+1:")
    print(f"mu_pred_base     = {mu_pred_base}")
    print(f"y_star_base      = {y_star_tnext_base}")
    print(f"obj_base         = {obj_tnext_base:.6f}")
    print("--------------------------------------------")
    print("Modified predictive law at t+1:")
    print(f"mu_pred_mod      = {mu_pred_mod}")
    print(f"y_star_mod       = {y_star_tnext_mod}")
    print(f"obj_mod          = {obj_tnext_mod:.6f}")
    print("--------------------------------------------")
    print(f"||mu_pred_mod - mu_pred_base||   = {np.linalg.norm(mu_pred_mod - mu_pred_base):.6f}")
    print(f"||y_star_mod - y_star_base||     = {np.linalg.norm(y_star_tnext_mod - y_star_tnext_base):.6f}")
    print("============================================\n")

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------
    plot_tplus1_attack_comparison(
        t_mod=t_mod,
        t_attack=t_attack,
        epsilon=epsilon,
        y_attack=y_tnext,
        mu_base=mu_pred_base,
        Sigma_base=Sigma_pred_base,
        mu_mod=mu_pred_mod,
        Sigma_mod=Sigma_pred_mod,
        X_base=X_tnext_base,
        X_mod=X_tnext_mod,
        y_star_base=y_star_tnext_base,
        y_star_mod=y_star_tnext_mod,
        obj_base=obj_tnext_base,
        obj_mod=obj_tnext_mod,
        u_dir=u_dir,
        outpath=outpath,
    )

    print(f"Saved figure to: {outpath}")


if __name__ == "__main__":
    main()
