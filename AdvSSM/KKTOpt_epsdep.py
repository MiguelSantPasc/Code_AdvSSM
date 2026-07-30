#!/usr/bin/env python3
"""
Monte Carlo study of KKT attack strength as epsilon changes.

Each run samples a random 2D linear Gaussian SSM,

    x_{k+1} = A_k x_k + B_k u_k + w_{k+1},    w_{k+1} ~ N(0, Q_k)
    y_k     = H_k x_k + D_k u_k + v_k,        v_k     ~ N(0, R_k),

then attacks one selected observation y_t for several ellipsoid radii:

    (y_t* - mu_{t|-t})^T Sigma_{t|-t}^{-1} (y_t* - mu_{t|-t}) <= epsilon.

The KKT optimizer maximizes ||X_t (y_t* - y_t)||^2 inside the ellipsoid. The
script summarizes how local and global RTS smoothing errors change across
epsilon values, using cached arrays when available.
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
    std: float = 2.0,  # N(0, 4) <=> std = 2
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
# Monte Carlo evaluation for MULTIPLE epsilons
# ============================================================
def evaluate_attack_effects_single_run_multi_epsilon(
    *,
    run_seed: int,
    T: int,
    t_values: np.ndarray,
    epsilons: list[float] | np.ndarray,
    entry_variance: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For one random SSM:
      - simulate once
      - for each attacked time t in t_values:
          * compute loo_values once
          * for each epsilon, attack only y[t]
      - compute local and global effects

    Returns
    -------
    local_effects  : (n_eps, n_t)
    global_effects : (n_eps, n_t)
    """
    rng = np.random.default_rng(run_seed)
    std = float(np.sqrt(entry_variance))  # variance 4 => std 2

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
        dA=zeros["dA"], dB=zeros["dB"], dH=zeros["dH"], dD=zeros["dD"],
        dQ=zeros["dQ"], dR=zeros["dR"],
        u_low=-0.5, u_high=0.5,
    )

    epsilons = np.asarray(epsilons, dtype=float)
    n_eps = len(epsilons)
    n_t = len(t_values)

    local_effects = np.full((n_eps, n_t), np.nan, dtype=float)
    global_effects = np.full((n_eps, n_t), np.nan, dtype=float)

    for tidx, t in enumerate(t_values):
        try:
            # This does not depend on epsilon -> compute once per t
            X_t, mu_t, Sigma_t = loo_values_nd(
                t=int(t), y=y, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"],
                H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                P0=params["P0"], m0=params["m0"],
            )

            for eidx, eps in enumerate(epsilons):
                y_star, _ = solve_kkt_max_quadratic_over_ellipsoid(
                    X=X_t,
                    y_t=y[t],
                    mu=mu_t,
                    Sigma=Sigma_t,
                    epsilon=float(eps),
                )

                y_adv = y.copy()
                y_adv[t] = y_star  # attack only one time t

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
                local_effects[eidx, tidx] = float(
                    np.sum(np.abs(x_true[t] - m_smooth_a[t]))
                )

                # Global effect over all times and hidden dims
                global_effects[eidx, tidx] = float(
                    np.sum(np.abs(x_true - m_smooth_a))
                )

        except Exception as e:
            print(f"[WARN run_seed={run_seed} t={int(t)}] {type(e).__name__}: {e}")
            continue

    return local_effects, global_effects


def run_monte_carlo_attack_study_multi_epsilon(
    *,
    N_runs: int = 20,
    T: int = 10,
    epsilons: list[float] | np.ndarray = (0.5, 1.0, 2.0, 5.991, 9.21),
    base_seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    t_values    : (T,) with t=1..T
    local_cube  : (n_eps, N_runs, T)
    global_cube : (n_eps, N_runs, T)
    """
    t_values = np.arange(1, T + 1, dtype=int)  # all attacked times: 1..T
    epsilons = np.asarray(epsilons, dtype=float)

    n_eps = len(epsilons)
    n_t = len(t_values)

    local_cube = np.full((n_eps, N_runs, n_t), np.nan, dtype=float)
    global_cube = np.full((n_eps, N_runs, n_t), np.nan, dtype=float)

    for r in range(N_runs):
        run_seed = base_seed + 1000 * r
        print(f"[MC multi-eps] run {r+1}/{N_runs} (seed={run_seed})")

        local_eff, global_eff = evaluate_attack_effects_single_run_multi_epsilon(
            run_seed=run_seed,
            T=T,
            t_values=t_values,
            epsilons=epsilons,
            entry_variance=4.0,
        )

        local_cube[:, r, :] = local_eff
        global_cube[:, r, :] = global_eff

    return t_values, local_cube, global_cube


# ============================================================
# Plot means only (multi-epsilon)
# ============================================================
def _set_plot_theme() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.7,
    })


def _style_axis(ax) -> None:
    ax.set_facecolor("#FBFBFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.55)
    ax.spines["bottom"].set_alpha(0.55)
    ax.grid(True, alpha=0.20)


def plot_attack_effect_means_multi_epsilon(
    *,
    t_values: np.ndarray,
    local_cube: np.ndarray,     # (n_eps, N_runs, n_t)
    global_cube: np.ndarray,    # (n_eps, N_runs, n_t)
    epsilons: list[float] | np.ndarray,
    outpath: str,
) -> None:
    """
    One figure, two panels:
      - top: local means vs attacked time t
      - bottom: global means vs attacked time t
    Each epsilon is one dashed curve.
    """
    _set_plot_theme()
    epsilons = np.asarray(epsilons, dtype=float)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15.5, 9.5),
        sharex=True,
        constrained_layout=True,
    )

    for ax in axes:
        _style_axis(ax)

    # Mean across Monte Carlo runs
    local_means = np.nanmean(local_cube, axis=1)    # (n_eps, n_t)
    global_means = np.nanmean(global_cube, axis=1)  # (n_eps, n_t)

    # Top: local
    for eidx, eps in enumerate(epsilons):
        axes[0].plot(
            t_values,
            local_means[eidx],
            linestyle="--",
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=fr"$\epsilon={eps:g}$",
        )

    axes[0].set_title(
        "(A) Media Monte Carlo del efecto local según el instante atacado",
        loc="left",
        fontweight="semibold",
    )
    axes[0].set_ylabel(
        r"$\mathbb{E}\!\left[\sum_j |x_t^{(j)}-\hat{x}_{t,\mathrm{adv}}^{(j)}|\right]$"
    )
    axes[0].legend(loc="best", frameon=True, framealpha=0.95)

    # Bottom: global
    for eidx, eps in enumerate(epsilons):
        axes[1].plot(
            t_values,
            global_means[eidx],
            linestyle="--",
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=fr"$\epsilon={eps:g}$",
        )

    axes[1].set_title(
        "(B) Media Monte Carlo del efecto global según el instante atacado",
        loc="left",
        fontweight="semibold",
    )
    axes[1].set_ylabel(
        r"$\mathbb{E}\!\left[\sum_{k=0}^{T}\sum_j |x_k^{(j)}-\hat{x}_{k,\mathrm{adv}}^{(j)}|\right]$"
    )
    axes[1].set_xlabel("Instante atacado t")
    axes[1].legend(loc="best", frameon=True, framealpha=0.95)

    axes[1].set_xticks(t_values)
    axes[1].set_xlim(float(t_values[0]) - 0.4, float(t_values[-1]) + 0.4)

    fig.suptitle(
        f"Monte Carlo KKT attack study | medias por t | N_runs={local_cube.shape[1]} | multi-$\\epsilon$",
        fontsize=13.5,
        fontweight="semibold",
        y=0.995,
    )

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, facecolor="white", dpi=300)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    N_runs = 1000
    T = 12
    epsilons = [0.5, 1.0, 2.0, 5.991, 9.21, 12.0,20.0, 30.0]  # 0.5,1,2: small; 5.991: chi2(2,0.95); 9.21: chi2(2,0.99); 12: chi2(2,0.999); 20,30: large
    base_seed = 2026
    force_recompute = False

    module_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = figures_dir_for(module_dir)
    data_dir = data_dir_for(module_dir)
    outpath = os.path.join(
        figures_dir,
        f"mc_attack_effects_means_multi_eps_N{N_runs}_T{T}.png"
    )
    cache_path = os.path.join(
        data_dir,
        f"mc_attack_effects_means_multi_eps_N{N_runs}_T{T}_seed{base_seed}.npz"
    )

    def compute_mc_data() -> dict[str, np.ndarray | int]:
        t_values, local_cube, global_cube = run_monte_carlo_attack_study_multi_epsilon(
            N_runs=N_runs,
            T=T,
            epsilons=epsilons,
            base_seed=base_seed,
        )
        return {
            "t_values": t_values,
            "local_cube": local_cube,
            "global_cube": global_cube,
            "epsilons": np.asarray(epsilons, dtype=float),
            "N_runs": N_runs,
            "T": T,
            "base_seed": base_seed,
        }

    data = cached_npz(cache_path, compute_mc_data, force=force_recompute)
    t_values = data["t_values"]
    local_cube = data["local_cube"]
    global_cube = data["global_cube"]
    epsilons = data["epsilons"].astype(float).tolist()

    # Optional terminal summary
    local_means = np.nanmean(local_cube, axis=1)    # (n_eps, n_t)
    global_means = np.nanmean(global_cube, axis=1)  # (n_eps, n_t)

    print("\n=== Means across runs by epsilon and attacked t ===")
    for eidx, eps in enumerate(epsilons):
        print(f"\n--- epsilon = {eps} ---")
        for t, lm, gm in zip(t_values, local_means[eidx], global_means[eidx]):
            print(f"t={int(t):2d} | local_mean={lm:.6f} | global_mean={gm:.6f}")

    plot_attack_effect_means_multi_epsilon(
        t_values=t_values,
        local_cube=local_cube,
        global_cube=global_cube,
        epsilons=epsilons,
        outpath=outpath,
    )

    print(f"\nSaved PNG figure to: {outpath}")


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
