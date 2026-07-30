#!/usr/bin/env python3
"""
attack_direction_heatmap_lgssm.py

Runs N independent simulations of an ND LGSSM, attacks the last time step t=T
via a leave-one-out white-box point attack on E[g(x_T) | y_T', y_-T], and
builds a viridis heatmap of the 2D attack directions in observation space.

Main output:
- A heatmap over the unit disk of normalized attack directions:
      d_i = (y_T'_i - y_T_i) / ||y_T'_i - y_T_i||

Notes:
- We attack the last observation y_T, which changes the posterior on x_T.
- The feasible attack region is the ellipsoid:
      (y_T' - mu_T)^T Sigma_T^{-1} (y_T' - mu_T) <= epsilon
  where p(y_T | y_-T) = N(mu_T, Sigma_T).
- The heatmap is built from the normalized attack directions in R^2.
"""

from __future__ import annotations

import os
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Global configuration
# ============================================================
N_RUNS = 250
T = 5
ATTACK_T = T

# 95% chi-square threshold for 2D is about 5.991
EPSILON = 5.991464547107979

# Attack target for scalar g(x) in [0,1]
# Set near 1.0 to push toward dangerous/high-risk directions.
# Set near 0.0 if you want the opposite.
M_STAR = np.array([1.00], dtype=float)

# Optimization settings
ETA = 0.05
N_STEPS = 800
N_MC_OPT = 128
N_MC_EST = 2000

# Heatmap settings
HEATMAP_BINS = 121
FIGURES_DIRNAME = os.path.join("outputs", "figures")
DATA_DIRNAME = os.path.join("outputs", "data")


# ============================================================
# Risk model g(x)
# ============================================================
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

    beta0 = -2.8
    beta1 = 1.4
    beta2 = 0.1
    beta3 = 0.8
    beta4 = 2.6

    z = beta0 + beta1 * x1 + beta2 * x2 + beta3 * x1 * x2 + beta4 * x1**2
    return float(sigmoid(z))


def g_scalar_grad(x_vec: np.ndarray) -> np.ndarray:
    """
    Gradient of scalar g wrt x.
    Returns shape (n_x,)
    """
    x_vec = np.asarray(x_vec, dtype=float)
    x1, x2 = x_vec[0], x_vec[1]

    beta0 = -2.8
    beta1 = 1.4
    beta2 = 0.1
    beta3 = 0.8
    beta4 = 2.6

    z = beta0 + beta1 * x1 + beta2 * x2 + beta3 * x1 * x2 + beta4 * x1**2
    s = float(sigmoid(z))

    dz_dx1 = beta1 + beta3 * x2 + 2.0 * beta4 * x1
    dz_dx2 = beta2 + beta3 * x1

    common = s * (1.0 - s)

    return np.array([
        common * dz_dx1,
        common * dz_dx2,
    ], dtype=float)


# ============================================================
# Experiment setup
# ============================================================
def get_system_parameters() -> dict[str, np.ndarray]:
    A0 = np.array([
        [0.93, 0.22],
        [0.03, 0.86],
    ], dtype=float)

    B0 = np.array([
        [-0.28, 0.18],
        [-0.20, 0.24],
    ], dtype=float)

    H0 = np.array([
        [1.10, 0.35],
        [0.55, 0.95],
    ], dtype=float)

    D0 = np.array([
        [-0.04, 0.20],
        [-0.10, 0.16],
    ], dtype=float)

    Q0 = np.array([
        [0.020, 0.008],
        [0.008, 0.015],
    ], dtype=float)

    R0 = np.array([
        [0.045, 0.012],
        [0.012, 0.040],
    ], dtype=float)

    x0 = np.array([0.35, 0.10], dtype=float)
    m0 = x0.copy()

    P0 = np.array([
        [0.040, 0.010],
        [0.010, 0.030],
    ], dtype=float)

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

    # Optional posterior comparison at t=T
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


# ============================================================
# Density plot of attack displacements
# ============================================================
def plot_delta_density(
    deltas: np.ndarray,
    outpath: str,
    gridsize: int = 200,
) -> None:
    """
    Density estimation (KDE) of attack displacements:
        delta_y = y_T' - y_T

    deltas: shape (N, 2)
    """
    deltas = np.asarray(deltas, dtype=float)

    if deltas.ndim != 2 or deltas.shape[1] != 2:
        raise ValueError("deltas must have shape (N, 2)")

    x = deltas[:, 0]
    y = deltas[:, 1]

    # KDE
    values = np.vstack([x, y])
    kde = gaussian_kde(values)

    # --------------------------------------------------------
    # SAME AXIS SCALE (important)
    # --------------------------------------------------------
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    lim_min = min(x_min, y_min)
    lim_max = max(x_max, y_max)

    span = lim_max - lim_min
    pad = 0.1 * span if span > 1e-12 else 1e-2

    xmin = lim_min - pad
    xmax = lim_max + pad

    # square domain
    xi = np.linspace(xmin, xmax, gridsize)
    yi = np.linspace(xmin, xmax, gridsize)
    XX, YY = np.meshgrid(xi, yi)

    # KDE evaluation
    grid_coords = np.vstack([XX.ravel(), YY.ravel()])
    Z = kde(grid_coords).reshape(XX.shape)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[xmin, xmax, xmin, xmax],
        cmap="viridis",
        alpha=0.8,
        aspect="equal",
    )

    #  Mark origin
    ax.scatter(0.0, 0.0, marker="x", s=80, linewidths=2)
    ax.text(0.0, 0.0, "  (0,0)", fontsize=9, va="bottom")

    # y = x line
    ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1.5)

    ax.set_title(r"Density of attack displacements $\Delta y$")
    ax.set_xlabel(r"$\Delta y_1$")
    ax.set_ylabel(r"$\Delta y_2$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.grid(alpha=0.15)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("density")

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(module_dir, FIGURES_DIRNAME)
    data_dir = os.path.join(module_dir, DATA_DIRNAME)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(
        data_dir,
        f"delta_y_points_N{N_RUNS}_T{T}_t{ATTACK_T}.npz",
    )

    density_path = os.path.join(
        figures_dir,
        f"delta_y_density_N{N_RUNS}_T{T}_t{ATTACK_T}.png",
    )

    # --------------------------------------------------------
    # SKIP computation if data already exists
    # --------------------------------------------------------
    if os.path.exists(data_path):
        print("\nData already exists. Skipping simulation...")

        data = np.load(data_path)
        deltas = data["deltas"]

        plot_delta_density(
            deltas=deltas,
            outpath=density_path,
        )

        print(f"Density plot regenerated from existing data: {density_path}")
        return

    # --------------------------------------------------------
    # RUN experiments
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # COLLECT results
    # --------------------------------------------------------
    deltas = np.stack(
        [np.asarray(r["delta_y"], dtype=float) for r in all_results],
        axis=0,
    )

    # --------------------------------------------------------
    # PLOT density
    # --------------------------------------------------------
    plot_delta_density(
        deltas=deltas,
        outpath=density_path,
    )

    # --------------------------------------------------------
    # SAVE data
    # --------------------------------------------------------
    np.savez(
        data_path,
        deltas=deltas,
    )

    print(f"\nSaved density plot to: {density_path}")
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
