#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensitivity of the KKT observation attack with respect to epsilon.

For each run, the script simulates an ND linear Gaussian SSM,

    x_{k+1} = A_k x_k + B_k u_k + w_{k+1},    w_{k+1} ~ N(0, Q_k)
    y_k     = H_k x_k + D_k u_k + v_k,        v_k     ~ N(0, R_k),

then attacks one observation y_t inside the leave-one-out ellipsoid

    (y_t* - mu_{t|-t})^T Sigma_{t|-t}^{-1} (y_t* - mu_{t|-t}) <= epsilon,

where p(y_t | y_{-t}) = N(mu_{t|-t}, Sigma_{t|-t}). The KKT solution maximizes
the quadratic state perturbation ||X_t (y_t* - y_t)||^2 under that constraint.

The plotted response is the Euclidean distance at the attacked time between
the true state and either the baseline RTS smoother or the adversarial RTS
smoother. Repeating this for many seeds gives a bootstrap 95% confidence
interval for the mean sensitivity curve.
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

# -----------------------------
# Utilities: PSD symmetrize + sqrt
# -----------------------------


# -----------------------------
# Simulator with drift (yours)
# -----------------------------


# -----------------------------
# LOO + X_t (yours, compact)
# -----------------------------


# -----------------------------
# KKT solver (yours)
# -----------------------------


# -----------------------------
# Kalman + RTS (yours)
# -----------------------------


# -----------------------------
# Bootstrap CI for the mean
# -----------------------------
def bootstrap_ci_of_mean(
    x: np.ndarray,
    n_boot: int = 600,
    alpha: float = 0.05,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x shape: (n_samples,) or (n_samples, n_eps)
    returns: mean, lo, hi (bootstrap percentile CI for mean)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = x.shape[0]

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)  # (n_boot,) or (n_boot, n_eps)

    mean = x.mean(axis=0)
    lo = np.quantile(boot_means, alpha / 2, axis=0)
    hi = np.quantile(boot_means, 1 - alpha / 2, axis=0)
    return mean, lo, hi


# -----------------------------
# Sensitivity experiment runner
# -----------------------------
def run_sensitivity_two_views(
    *,
    T: int,
    t_selected: list[int],
    eps_grid: np.ndarray,
    eps_fixed: float,
    n_seeds: int = 70,
    seed0: int = 2026,
    n_boot: int = 600,
    seed_boot_left: int = 999,
    seed_boot_right: int = 1234,
) -> dict:
    """
    Devuelve resultados para:
      (A) ratio vs eps en t_selected
      (B) ratio vs t (todos los t=0..T) para eps_fixed
    """

    t_selected = [int(t) for t in t_selected]
    for t in t_selected:
        if not (0 <= t <= T):
            raise ValueError(f"t_selected contiene t={t} fuera de rango [0, T] con T={T}")

    eps_grid = np.asarray(eps_grid, dtype=float)
    eps_grid = eps_grid[eps_grid > 0]
    if eps_grid.size == 0:
        raise ValueError("eps_grid debe tener epsilons > 0")

    eps_fixed = float(eps_fixed)
    if eps_fixed <= 0:
        raise ValueError("eps_fixed debe ser > 0")

    n_sel = len(t_selected)
    n_eps = eps_grid.size

    # -----------------------
    # MODELO (igual que tu ejemplo)
    # -----------------------
    n_x = n_y = n_u = 2

    A0 = np.array([[0.65, 0.40],
                   [-0.15, 0.70]], dtype=float)

    B0 = np.array([[1.65, 0.40],
                   [-0.15, 0.70]], dtype=float)

    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.03 * np.array([[1.6, -0.40],
                          [0.15, 0.70]], dtype=float)

    R0 = 0.02 * np.array([[0.65, 0.40],
                          [-0.15, 1.70]], dtype=float)

    Q0 = project_to_psd(Q0)
    R0 = project_to_psd(R0)

    dA = np.zeros_like(A0)
    dB = np.zeros_like(B0)
    dH = np.zeros_like(H0)
    dD = np.zeros_like(D0)
    dQ = np.zeros_like(Q0)
    dR = np.zeros_like(R0)

    x0 = np.array([0.5, 0.5])
    m0 = x0.copy()
    P0 = 0.05 * np.eye(n_x)

    # -----------------------
    # ARRAYS: por semilla
    # -----------------------
    # Vista A (t_selected x eps_grid)
    d_base_sel = np.zeros((n_seeds, n_sel), dtype=float)
    d_adv0_sel = np.zeros((n_seeds, n_sel), dtype=float)
    d_adv_sel  = np.zeros((n_seeds, n_sel, n_eps), dtype=float)

    # Vista B (todos los t) para eps_fixed
    d_base_all = np.zeros((n_seeds, T + 1), dtype=float)
    d_adv0_all = np.zeros((n_seeds, T + 1), dtype=float)
    d_adv_fix  = np.zeros((n_seeds, T + 1), dtype=float)

    for i in range(n_seeds):
        seed = seed0 + i

        # --- simula
        x, y, u, mats = simulate_lgssm_nd(
            A0=A0, B0=B0, H0=H0, D0=D0,
            T=T, seed=seed, x0=x0,
            Q0=Q0, R0=R0,
            dA=dA, dB=dB, dH=dH, dD=dD, dQ=dQ, dR=dR,
            u_low=-0.5, u_high=0.5,
        )

        # --- baseline RTS (una vez)
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

        # baseline error para todos los t
        for t in range(T + 1):
            d_base_all[i, t] = float(np.linalg.norm(m_smooth_b[t] - x[t]))
        # baseline para t_selected
        for k, t in enumerate(t_selected):
            d_base_sel[i, k] = d_base_all[i, t]

        # -----------------------
        # Vista A: para t_selected y eps_grid
        # -----------------------
        for k, t in enumerate(t_selected):
            X_t, mu_t, Sigma_t = loo_values_nd(
                t=t, y=y, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                P0=P0, m0=m0,
            )
            y_t = y[t].copy()

            # d_adv(0): eps=0 => y[t] = mu_t
            y_adv0 = y.copy()
            y_adv0[t] = mu_t
            m_filt0, P_filt0, m_pred0, P_pred0 = kalman_filter_nd(
                y=y_adv0, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=m0, P0=P0
            )
            m_smooth0, P_smooth0 = rts_smoother_nd(
                m_filt=m_filt0, P_filt=P_filt0,
                m_pred=m_pred0, P_pred=P_pred0,
                A_t=mats["A_t"]
            )
            d_adv0_sel[i, k] = float(np.linalg.norm(m_smooth0[t] - x[t]))

            # barrido eps
            for j, eps in enumerate(eps_grid):
                y_star, _ = solve_kkt_max_quadratic_over_ellipsoid(
                    X=X_t, y_t=y_t, mu=mu_t, Sigma=Sigma_t, epsilon=float(eps)
                )
                y_adv = y.copy()
                y_adv[t] = y_star

                m_filtd, P_filtd, m_predd, P_predd = kalman_filter_nd(
                    y=y_adv, u=u,
                    A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                    Q_t=mats["Q_t"], R_t=mats["R_t"],
                    m0=m0, P0=P0
                )
                m_smoothd, P_smoothd = rts_smoother_nd(
                    m_filt=m_filtd, P_filt=P_filtd,
                    m_pred=m_predd, P_pred=P_predd,
                    A_t=mats["A_t"]
                )
                d_adv_sel[i, k, j] = float(np.linalg.norm(m_smoothd[t] - x[t]))

        # -----------------------
        # Vista B: para todos los t con eps_fixed
        # -----------------------
        for t in range(T + 1):
            X_t, mu_t, Sigma_t = loo_values_nd(
                t=t, y=y, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                P0=P0, m0=m0,
            )
            y_t = y[t].copy()

            # d_adv0(t): y[t]=mu_t
            y_adv0 = y.copy()
            y_adv0[t] = mu_t
            m_filt0, P_filt0, m_pred0, P_pred0 = kalman_filter_nd(
                y=y_adv0, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=m0, P0=P0
            )
            m_smooth0, P_smooth0 = rts_smoother_nd(
                m_filt=m_filt0, P_filt=P_filt0,
                m_pred=m_pred0, P_pred=P_pred0,
                A_t=mats["A_t"]
            )
            d_adv0_all[i, t] = float(np.linalg.norm(m_smooth0[t] - x[t]))

            # d_adv_fix(t): y[t]=y*(eps_fixed)
            y_star, _ = solve_kkt_max_quadratic_over_ellipsoid(
                X=X_t, y_t=y_t, mu=mu_t, Sigma=Sigma_t, epsilon=eps_fixed
            )
            y_adv = y.copy()
            y_adv[t] = y_star

            m_filtd, P_filtd, m_predd, P_predd = kalman_filter_nd(
                y=y_adv, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=m0, P0=P0
            )
            m_smoothd, P_smoothd = rts_smoother_nd(
                m_filt=m_filtd, P_filt=P_filtd,
                m_pred=m_predd, P_pred=P_predd,
                A_t=mats["A_t"]
            )
            d_adv_fix[i, t] = float(np.linalg.norm(m_smoothd[t] - x[t]))

    # -----------------------
    # Ratios
    # -----------------------
    # Vista A: (S, n_sel, n_eps)
    denA = np.where(np.abs(d_base_sel[:, :, None]) < 1e-12, 1e-12, d_base_sel[:, :, None])
    ratio_sel = (d_adv_sel - d_adv0_sel[:, :, None]) / denA

    meanA, loA, hiA = bootstrap_ci_of_mean(ratio_sel, n_boot=n_boot, seed=seed_boot_left)

    # Vista B: (S, T+1)
    denB = np.where(np.abs(d_base_all) < 1e-12, 1e-12, d_base_all)
    ratio_t = (d_adv_fix - d_adv0_all) / denB

    meanB, loB, hiB = bootstrap_ci_of_mean(ratio_t, n_boot=n_boot, seed=seed_boot_right)

    return dict(
        T=T,
        eps_grid=eps_grid,
        eps_fixed=eps_fixed,
        t_selected=np.array(t_selected, dtype=int),
        meanA=meanA, loA=loA, hiA=hiA,        # (n_sel, n_eps)
        t_all=np.arange(T + 1, dtype=int),
        meanB=meanB, loB=loB, hiB=hiB,        # (T+1,)
    )


def plot_two_panel(
    res2: dict,
    *,
    use_logx_left: bool = True,
    eps_ref: float | None = 5.991,
    savepath: str | None = None,
    show: bool = True,
) -> str:
    if savepath is None:
        figures_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
        savepath = os.path.join(figures_dir, "two_panel_ratio.png")
    """
    Figura con 2 paneles:
      Izq: ratio vs eps para t_selected
      Der: ratio vs t para eps_fixed
    """

    eps = np.asarray(res2["eps_grid"], dtype=float)
    t_sel = np.asarray(res2["t_selected"], dtype=int)
    meanA = np.asarray(res2["meanA"], dtype=float)
    loA   = np.asarray(res2["loA"], dtype=float)
    hiA   = np.asarray(res2["hiA"], dtype=float)

    t_all = np.asarray(res2["t_all"], dtype=int)
    meanB = np.asarray(res2["meanB"], dtype=float)
    loB   = np.asarray(res2["loB"], dtype=float)
    hiB   = np.asarray(res2["hiB"], dtype=float)

    # ordenar eps
    order = np.argsort(eps)
    eps = eps[order]
    meanA = meanA[:, order]
    loA   = loA[:, order]
    hiA   = hiA[:, order]

    if use_logx_left:
        mask = eps > 0
        eps = eps[mask]
        meanA = meanA[:, mask]
        loA   = loA[:, mask]
        hiA   = hiA[:, mask]

    plt.rcParams.update({
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.2, 6.2))

    # ---- Left: ratio vs eps for selected t
    for k, t in enumerate(t_sel):
        (line,) = axL.plot(eps, meanA[k], linewidth=2.0, label=fr"$t={t}$")
        axL.fill_between(eps, loA[k], hiA[k], alpha=0.18, color=line.get_color())

    axL.axhline(0.0, linestyle=":", linewidth=1.1, alpha=0.9)
    if eps_ref is not None:
        axL.axvline(float(eps_ref), linestyle="--", linewidth=1.0, alpha=0.35)

    if use_logx_left:
        axL.set_xscale("log")

    axL.set_xlabel("epsilon (ε)")
    axL.set_ylabel(r"$(d_{adv}(\epsilon)-d_{adv}(0))/d_{base}$")
    axL.set_title("Ratio vs ε (95% bootstrap CI)")
    axL.grid(True, which="major", alpha=0.28)
    axL.grid(True, which="minor", alpha=0.12)
    axL.legend(frameon=True, loc="best")

    # ---- Right: ratio vs time t for fixed eps
    (lineR,) = axR.plot(t_all, meanB, linewidth=2.2, marker="o", markersize=4.5,
                        label=fr"$\epsilon={res2['eps_fixed']}$")
    axR.fill_between(t_all, loB, hiB, alpha=0.18, color=lineR.get_color())

    axR.axhline(0.0, linestyle=":", linewidth=1.1, alpha=0.9)
    axR.set_xlabel("time index t")
    axR.set_ylabel(r"$(d_{adv}(\epsilon_{fix})-d_{adv}(0))/d_{base}$")
    axR.set_title("Ratio vs time (ε fixed, 95% bootstrap CI)")
    axR.grid(True, alpha=0.25)
    axR.legend(frameon=True, loc="best")

    fig.suptitle("Adversarial impact sensitivity: ε-sweep and time profile", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return savepath


def main():
    T = 50
    t_selected = [1, int(T/2), T]
    eps_grid = np.r_[np.linspace(0.1, 2.0, 10, endpoint=False),
                    np.linspace(2.0, 12.0, 16)]
    eps_fixed = 5.991
    n_seeds = 70
    seed0 = 2026
    force_recompute = False

    savepath = os.path.join(
        figures_dir_for(os.path.dirname(os.path.abspath(__file__))),
        "two_panel_ratio_eps_and_time.png",
    )
    data_path = data_path_for_plot(savepath)

    def compute_sensitivity_data() -> dict:
        return run_sensitivity_two_views(
            T=T,
            t_selected=t_selected,
            eps_grid=eps_grid,
            eps_fixed=eps_fixed,
            n_seeds=n_seeds,
            seed0=seed0,
        )

    res2 = cached_npz(data_path, compute_sensitivity_data, force=force_recompute)

    out = plot_two_panel(
        res2,
        use_logx_left=False,
        eps_ref=5.991,
        savepath=savepath,
        show=False,   # pon False si solo quieres guardar
    )
    print("Saved:", out)

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
