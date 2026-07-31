#!/usr/bin/env python3
"""
Monte Carlo study of how the attacked time changes the KKT attack impact.

Each run samples a random 2D linear Gaussian state-space model,

    x_{k+1} = A_k x_k + B_k u_k + w_{k+1},    w_{k+1} ~ N(0, Q_k)
    y_k     = H_k x_k + D_k u_k + v_k,        v_k     ~ N(0, R_k),

then attacks exactly one observation y_t for t = 1, ..., T. The feasible set
is the leave-one-out predictive ellipsoid p(y_t | y_{-t}) and the KKT
objective is ||X_t (y_t* - y_t)||^2.

For each attacked time, the script records:
- the local smoothing error at the attacked time,
- the global smoothing error over the full trajectory.

The final plot is distribution-aware: each attacked time is summarized with a
mean comparison curve for the local effect and another for the global effect.
Both are shown on the same panel with separate vertical axes so their temporal
trends can be compared without one scale visually flattening the other.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared_ssm.artifacts import data_dir_for
from shared_ssm.artifacts import figures_dir_for


# ============================================================
# Random SSM generation for Monte Carlo
# ============================================================
def _spectral_radius(M: np.ndarray) -> float:
    vals = np.linalg.eigvals(M)
    return float(np.max(np.abs(vals)))


def sample_random_ssm_run_params(
    rng: np.random.Generator,
    *,
    n_x: int = 2,
    n_y: int = 2,
    n_u: int = 2,
    std: float = 8.0,  # variance 4
) -> dict[str, np.ndarray]:
    """
    Sample one random SSM run with entries ~ N(0,4) for A0,B0,H0,D0.
    Q0,R0 are also sampled entrywise ~ N(0,4) and projected to PSD.
    """
    A0 = rng.normal(0.0, std, size=(n_x, n_x))
    B0 = rng.normal(0.0, std, size=(n_x, n_u))
    H0 = rng.normal(0.0, std, size=(n_y, n_x))
    D0 = rng.normal(0.0, std, size=(n_y, n_u))

    # Mild stabilization (numerical robustness)
    rho = _spectral_radius(A0)
    if rho > 0.98:
        A0 = A0 * (0.95 / rho)

    Q0_raw = rng.normal(0.0, std, size=(n_x, n_x))
    R0_raw = rng.normal(0.0, std, size=(n_y, n_y))
    Q0 = project_to_psd(Q0_raw) + 0.05 * np.eye(n_x)
    R0 = project_to_psd(R0_raw) + 0.05 * np.eye(n_y)

    x0 = rng.normal(0.0, 1.0, size=(n_x,))
    m0 = x0.copy()
    P0 = 0.10 * np.eye(n_x)

    return {
        "A0": A0, "B0": B0, "H0": H0, "D0": D0,
        "Q0": Q0, "R0": R0,
        "x0": x0, "m0": m0, "P0": P0,
    }


# ============================================================
# Monte Carlo evaluation
# ============================================================
def evaluate_attack_effects_single_run(
    *,
    run_seed: int,
    T: int,
    t_values: np.ndarray,
    epsilon: float,
    var_entries: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For one random SSM:
      - simulate once
      - for each attacked time t in t_values, replace only y[t] with KKT y*
      - compute local and global effects
    """
    rng = np.random.default_rng(run_seed)
    std = float(np.sqrt(var_entries))  # sqrt(4)=2

    params = sample_random_ssm_run_params(rng, n_x=2, n_y=2, n_u=2, std=std)

    zeros = {
        "dA": np.zeros_like(params["A0"]),
        "dB": np.zeros_like(params["B0"]),
        "dH": np.zeros_like(params["H0"]),
        "dD": np.zeros_like(params["D0"]),
        "dQ": np.zeros_like(params["Q0"]),
        "dR": np.zeros_like(params["R0"]),
    }

    x_true, y, u, mats = simulate_lgssm_nd(
        A0=params["A0"], B0=params["B0"], H0=params["H0"], D0=params["D0"],
        T=T, seed=run_seed, x0=params["x0"], Q0=params["Q0"], R0=params["R0"],
        dA=zeros["dA"], dB=zeros["dB"], dH=zeros["dH"], dD=zeros["dD"], dQ=zeros["dQ"], dR=zeros["dR"],
        u_low=-0.5, u_high=0.5,
    )

    local_effects = np.full(len(t_values), np.nan, dtype=float)
    global_effects = np.full(len(t_values), np.nan, dtype=float)

    for idx, t in enumerate(t_values):
        try:
            X_t, mu_t, Sigma_t = loo_values_nd(
                t=int(t), y=y, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"],
                H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                P0=params["P0"], m0=params["m0"],
            )

            y_star, _ = solve_kkt_max_quadratic_over_ellipsoid(
                X=X_t, y_t=y[t], mu=mu_t, Sigma=Sigma_t, epsilon=epsilon
            )

            y_adv = y.copy()
            y_adv[t] = y_star  # attack only one time

            m_filt_a, P_filt_a, m_pred_a, P_pred_a = kalman_filter_nd(
                y=y_adv, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"],
                H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=params["m0"], P0=params["P0"],
            )
            m_smooth_a, _ = rts_smoother_nd(
                m_filt=m_filt_a, P_filt=P_filt_a,
                m_pred=m_pred_a, P_pred=P_pred_a,
                A_t=mats["A_t"],
            )

            # Local effect at attacked time
            local_effects[idx] = float(np.sum(np.abs(x_true[t] - m_smooth_a[t])))

            # Global effect over all times and hidden dims
            global_effects[idx] = float(np.sum(np.abs(x_true - m_smooth_a)))

        except Exception as e:
            print(f"[WARN run_seed={run_seed} t={int(t)}] {type(e).__name__}: {e}")
            continue

    return local_effects, global_effects


def run_monte_carlo_attack_study(
    *,
    N_runs: int = 20,
    T: int = 10,
    epsilon: float = 5.991,
    base_seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_values  : (10,) with t=1..10
      local_mat : (N_runs, 10)
      global_mat: (N_runs, 10)
    """
    t_values = np.arange(1, T+1, dtype=int)  # do not attack t=0

    if T < int(t_values[-1]):
        raise ValueError(f"T={T} must be >= 10")

    local_mat = np.full((N_runs, len(t_values)), np.nan, dtype=float)
    global_mat = np.full((N_runs, len(t_values)), np.nan, dtype=float)

    for r in range(N_runs):
        run_seed = base_seed + 1000 * r
        print(f"[MC] run {r+1}/{N_runs} (seed={run_seed})")

        local_eff, global_eff = evaluate_attack_effects_single_run(
            run_seed=run_seed,
            T=T,
            t_values=t_values,
            epsilon=epsilon,
            var_entries=8.0,
        )
        local_mat[r] = local_eff
        global_mat[r] = global_eff

    return t_values, local_mat, global_mat


# ============================================================
# Plot helpers
# ============================================================
def _set_plot_theme() -> None:
    """Apply the muted plotting theme used across the AdvSSM figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linewidth": 0.75,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _style_axis(ax) -> None:
    """Apply a soft background, darker spines, and denser dashed grid lines."""
    ax.set_facecolor("#FCFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_alpha(0.9)
    ax.spines["bottom"].set_alpha(0.9)
    ax.minorticks_on()
    ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.28, linewidth=0.75)
    ax.grid(True, which="minor", axis="both", linestyle="--", alpha=0.14, linewidth=0.55)
    ax.set_axisbelow(True)


def _mean_line_plot(
    ax,
    data_mat: np.ndarray,
    t_values: np.ndarray,
    title: str,
    ylabel: str,
    mean_color: str,
    adaptive_ylim: bool = True,
) -> None:
    means = np.array([
        np.nanmean(data_mat[:, i]) if np.any(np.isfinite(data_mat[:, i])) else np.nan
        for i in range(data_mat.shape[1])
    ], dtype=float)


    # Línea de la media
    ax.plot(
        t_values,
        means,
        color=mean_color,
        marker="o",
        markersize=6,
        linewidth=2.4,
        zorder=3,
        label="Mean",
    )

    if adaptive_ylim:
        finite_lower = means[np.isfinite(means)]
        finite_upper = means[np.isfinite(means)]

        if finite_lower.size > 0 and finite_upper.size > 0:
            y_min = float(np.min(finite_lower))
            y_max = float(np.max(finite_upper))
            span = max(y_max - y_min, 1e-8)
            pad = 0.12 * span

            if span < 1e-6:
                pad = 0.01 * max(abs(y_min), abs(y_max), 1.0)

            ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_title(title, loc="left", pad=10)
    ax.set_ylabel(ylabel)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
    )

def plot_attack_effect_boxplots(
    *,
    t_values: np.ndarray,
    local_mat: np.ndarray,
    global_mat: np.ndarray,
    outpath: str,
    epsilon: float,
) -> None:
    """
    Plot the mean local and global attack effects in two stacked panels.
    Each panel shows one mean curve so the labels can stay clean and the
    panel title can occupy the old "(A)/(B)" position.
    """
    _set_plot_theme()

    c_local_line = "#5E738F"
    c_global_line = "#6E9181"
    title_fontsize = 15.5
    ylabel_fontsize = 18.0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15.5, 9.5),
        sharex=True,
        constrained_layout=True,
    )

    for ax in axes:
        _style_axis(ax)

    local_means = np.array([
        np.nanmean(local_mat[:, i]) if np.any(np.isfinite(local_mat[:, i])) else np.nan
        for i in range(local_mat.shape[1])
    ], dtype=float)
    global_means = np.array([
        np.nanmean(global_mat[:, i]) if np.any(np.isfinite(global_mat[:, i])) else np.nan
        for i in range(global_mat.shape[1])
    ], dtype=float)

    axes[0].plot(
        t_values,
        local_means,
        color=c_local_line,
        marker="o",
        markersize=5.5,
        linewidth=2.3,
        zorder=3,
    )

    axes[1].plot(
        t_values,
        global_means,
        color=c_global_line,
        marker="o",
        markersize=5.5,
        linewidth=2.3,
        zorder=3,
    )

    finite_local = local_means[np.isfinite(local_means)]
    if finite_local.size > 0:
        y_min = float(np.min(finite_local))
        y_max = float(np.max(finite_local))
        span = max(y_max - y_min, 1e-8)
        pad = 0.12 * span

        if span < 1e-6:
            pad = 0.01 * max(abs(y_min), abs(y_max), 1.0)

        axes[0].set_ylim(y_min - pad, y_max + pad)

    finite_global = global_means[np.isfinite(global_means)]
    if finite_global.size > 0:
        y_min = float(np.min(finite_global))
        y_max = float(np.max(finite_global))
        span = max(y_max - y_min, 1e-8)
        pad = 0.12 * span

        if span < 1e-6:
            pad = 0.01 * max(abs(y_min), abs(y_max), 1.0)

        axes[1].set_ylim(y_min - pad, y_max + pad)

    # Keep the titles where the old panel tags lived and enlarge the formulas.
    axes[0].set_title(
        "Mean local effect of attack",
        loc="left",
        pad=10,
        fontweight="bold",
        fontsize=title_fontsize,
    )
    axes[1].set_title(
        "Mean global effect of attack",
        loc="left",
        pad=10,
        fontweight="bold",
        fontsize=title_fontsize,
    )

    axes[0].set_ylabel(
        r"$\mathbb{E}\!\left[\sum_j |x_t^{(j)}-\hat{x}_{t,\mathrm{adv}}^{(j)}|\right]$",
        color="black",
        fontsize=ylabel_fontsize,
    )
    axes[1].set_ylabel(
        r"$\mathbb{E}\!\left[\sum_{k=0}^{T}\sum_j |x_k^{(j)}-\hat{x}_{k,\mathrm{adv}}^{(j)}|\right]$",
        color="black",
        fontsize=ylabel_fontsize,
    )
    axes[1].set_xlabel("Attacked time step $t$")

    for ax in axes:
        ax.set_xticks(t_values)
        ax.set_xlim(float(t_values[0]) - 0.75, float(t_values[-1]) + 0.75)
        ax.tick_params(axis="y", colors="black", direction="in", pad=8)

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    N_runs = 5000
    T = 12
    epsilon = 5.991
    base_seed = 2022
    force_recompute = False

    module_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = figures_dir_for(module_dir)
    data_dir = data_dir_for(module_dir)

    cache_path = os.path.join(
        data_dir,
        f"mc_attack_effects_data_N{N_runs}_T{T}_eps{epsilon:.3f}_seed{base_seed}.npz"
    )
    fig_path = os.path.join(
        figures_dir,
        f"mc_attack_effects_boxplots_N{N_runs}_T{T}_eps{epsilon:.3f}.png"
    )

    if os.path.exists(cache_path) and not force_recompute:
        print(f"[INFO] Cache found. Loading results from: {cache_path}")
        data = np.load(cache_path)
        t_values = data["t_values"]
        local_mat = data["local_mat"]
        global_mat = data["global_mat"]
    else:
        print("[INFO] Running Monte Carlo study...")
        t_values, local_mat, global_mat = run_monte_carlo_attack_study(
            N_runs=N_runs,
            T=T,
            epsilon=epsilon,
            base_seed=base_seed,
        )

        np.savez_compressed(
            cache_path,
            t_values=t_values,
            local_mat=local_mat,
            global_mat=global_mat,
            N_runs=N_runs,
            T=T,
            epsilon=epsilon,
            base_seed=base_seed,
        )
        print(f"[INFO] Saved cache to: {cache_path}")

    print("\n=== Means across runs by attacked t ===")
    local_means = np.array([np.nanmean(local_mat[:, i]) for i in range(local_mat.shape[1])])
    global_means = np.array([np.nanmean(global_mat[:, i]) for i in range(global_mat.shape[1])])
    for t, lm, gm in zip(t_values, local_means, global_means):
        print(f"t={int(t):2d} | local_mean={lm:.6f} | global_mean={gm:.6f}")

    plot_attack_effect_boxplots(
        t_values=t_values,
        local_mat=local_mat,
        global_mat=global_mat,
        outpath=fig_path,
        epsilon=epsilon,
    )

    print(f"\nSaved PNG figure to: {fig_path}")

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
