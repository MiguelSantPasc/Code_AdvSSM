#!/usr/bin/env python3
"""
comparison_lambdas.py

Lambda-sweep comparison for the online covariance-adaptation defense.

This script is the multi-lambda companion to
`CovarianceAdaptation/comparison.py`. It keeps the same KF-only adversarial
setup, but replaces the single defended trajectory by a sweep over several
multipliers of `lambda_max`, where `lambda_max` is the largest eigenvalue of
the attacked-time observation covariance `V_t` that is being modified.

The output figure contains four compact panels:
1. First hidden-state dimension `s_t^(1)` over time for:
   - the true state,
   - the clean non-attacked KF baseline,
   - the attacked KF,
   - defended KF trajectories for several `lambda` values.
2. Second hidden-state dimension `s_t^(2)` with the same trajectory
   comparison.
3. Monte Carlo mean local effect versus `lambda = c lambda_max`, with
   horizontal clean/attack references and a shaded spread band.
4. The same mean-curve comparison for the global effect.

To make the trajectory lines easy to inspect, this figure intentionally omits
all uncertainty bands and confidence intervals.

Important modeling choices enforced here:
- the defended estimator is KF-only, with no smoothing anywhere,
- the covariance adaptation is applied while filtering the attacked
  observation sequence `o_t^{adv}`.
"""

from __future__ import annotations

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from AdvSSM.KKTOpt import kalman_filter_nd, simulate_lgssm_nd
    from AdvSSM.KKTOpt_tdependent import sample_random_ssm_run_params
    from AdvSSM.io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from CovarianceAdaptation.comparison import (
        build_kf_attack,
        build_reference_setup,
        kalman_filter_with_online_covariance_adaptation,
        set_plot_theme,
        style_axis,
    )
except ModuleNotFoundError:
    from KKTOpt import kalman_filter_nd, simulate_lgssm_nd
    from KKTOpt_tdependent import sample_random_ssm_run_params
    from io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from comparison import (
        build_kf_attack,
        build_reference_setup,
        kalman_filter_with_online_covariance_adaptation,
        set_plot_theme,
        style_axis,
    )


def parse_lambda_scales(raw_values: str) -> np.ndarray:
    """
    Parse a comma-separated scale list into a sorted nonnegative array.

    Each scale `c` later becomes `lambda = c lambda_max`, where `lambda_max`
    is computed from the attacked-time observation covariance that is being
    adapted.
    """
    pieces = [piece.strip() for piece in raw_values.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("At least one lambda-scale value is required.")

    lambda_scales = np.array([float(piece) for piece in pieces], dtype=float)
    if np.any(lambda_scales < 0.0):
        raise ValueError("All lambda-scale values must be nonnegative.")

    lambda_scales = np.unique(lambda_scales)
    return np.sort(lambda_scales)


def lambda_max_from_observation_covariance(R_tk: np.ndarray) -> float:
    """
    Return `lambda_max` for the observation covariance modified by the defense.

    The defense modifies the attacked-time observation covariance `V_t`, so the
    natural normalization uses the largest eigenvalue of that covariance.
    """
    eigvals = np.linalg.eigvalsh(np.asarray(R_tk, dtype=float))
    return float(np.max(eigvals))


def build_lambda_colormap(
    lambda_scales: np.ndarray,
) -> tuple[mcolors.Colormap, mcolors.Normalize, np.ndarray]:
    """
    Build the viridis color mapping shared by lambda curves and colorbar.

    The same normalization is reused everywhere so each trajectory color
    matches the tick positions shown in the standalone lambda color scale.
    """
    lambda_scales = np.asarray(lambda_scales, dtype=float)
    positive_scales = lambda_scales[lambda_scales > 0.0]
    if positive_scales.size == 0:
        positive_scales = np.array([1.0], dtype=float)

    if positive_scales.size == 1:
        center = float(positive_scales[0])
        vmin = max(center / 1.5, 1e-12)
        vmax = max(center * 1.5, vmin * 1.001)
        norm: mcolors.Normalize = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.LogNorm(vmin=float(np.min(positive_scales)), vmax=float(np.max(positive_scales)))
    return plt.get_cmap("viridis"), norm, positive_scales


def lambda_colors(lambda_scales: np.ndarray) -> list[str]:
    """
    Return a viridis sweep for the lambda trajectories.

    Larger `lambda` values receive later viridis hues so the color progression
    visually follows the strength of the covariance-adaptation penalty.
    """
    lambda_scales = np.asarray(lambda_scales, dtype=float)
    if lambda_scales.size == 0:
        return []

    cmap, norm, positive_scales = build_lambda_colormap(lambda_scales)
    if positive_scales.size == 1 and lambda_scales.size == 1:
        return [mcolors.to_hex(cmap(0.58)[:3])]
    return [mcolors.to_hex(cmap(float(norm(max(scale, positive_scales[0]))))[:3]) for scale in lambda_scales]


def effect_box_colors(n_lambda_values: int) -> list[str]:
    """
    Return pastel colors for clean, attack, and the lambda-sweep boxplots.

    The first two colors are fixed so the clean and attacked references stay
    recognizable across figures, while the lambda boxes follow a softer sweep.
    """
    lambda_palette = [
        "#C9DEF1",
        "#D0E6B8",
        "#F4D1A7",
        "#DCC7EE",
        "#F6C1C9",
        "#BFE6E0",
        "#E0DAA6",
        "#C9D4F0",
        "#D7EAC4",
        "#F7D8BC",
        "#D8CCE8",
    ]
    colors = ["#9BCBE7", "#F0A79D"]
    colors.extend(lambda_palette[idx % len(lambda_palette)] for idx in range(n_lambda_values))
    return colors


def tukey_inliers(values: np.ndarray, whisker_scale: float = 1.5) -> np.ndarray:
    """
    Return the Tukey inliers used for boxplot-style outlier suppression.

    This helper is only used for visualization so the plotted boxplots remain
    readable when a few Monte Carlo runs are extreme.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 3:
        return values

    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = q3 - q1
    lower = q1 - whisker_scale * iqr
    upper = q3 + whisker_scale * iqr
    inliers = values[(values >= lower) & (values <= upper)]
    return inliers if inliers.size > 0 else values


def evaluate_reference_lambda_sweep(
    *,
    T: int,
    attack_t: int,
    seed: int,
    epsilon: float,
    lambda_scales: np.ndarray,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
) -> dict[str, np.ndarray | float | int]:
    """
    Simulate one deterministic reference trajectory and sweep several
    `lambda = c lambda_max` values.

    The attack is built only once from the clean trajectory so the defended
    trajectories are directly comparable across lambda values.
    """
    setup = build_reference_setup()
    m0 = np.asarray(setup["x0"], dtype=float).copy()
    P0 = np.asarray(setup["P0"], dtype=float).copy()

    x_true, y_clean, u_controls, mats = simulate_lgssm_nd(
        A0=np.asarray(setup["A0"], dtype=float),
        B0=np.asarray(setup["B0"], dtype=float),
        H0=np.asarray(setup["H0"], dtype=float),
        D0=np.asarray(setup["D0"], dtype=float),
        T=T,
        seed=seed,
        x0=np.asarray(setup["x0"], dtype=float),
        Q0=np.asarray(setup["Q0"], dtype=float),
        R0=np.asarray(setup["R0"], dtype=float),
        dA=np.asarray(setup["dA"], dtype=float),
        dB=np.asarray(setup["dB"], dtype=float),
        dH=np.asarray(setup["dH"], dtype=float),
        dD=np.asarray(setup["dD"], dtype=float),
        dQ=np.asarray(setup["dQ"], dtype=float),
        dR=np.asarray(setup["dR"], dtype=float),
        u_low=-0.5,
        u_high=0.5,
    )

    attack_data = build_kf_attack(
        y_clean=y_clean,
        u_controls=u_controls,
        mats=mats,
        attack_t=attack_t,
        m0=m0,
        P0=P0,
        epsilon=epsilon,
    )
    y_adv = np.asarray(attack_data["y_adv"], dtype=float)
    attack_targets = {attack_t: np.asarray(attack_data["adv_target"], dtype=float)}
    lambda_max = lambda_max_from_observation_covariance(mats["R_t"][attack_t])
    lambda_values = lambda_scales * lambda_max

    clean_m, _, _, _ = kalman_filter_nd(
        y=y_clean,
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
    attack_m, _, _, _ = kalman_filter_nd(
        y=y_adv,
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

    adapt_means = np.zeros((lambda_values.size, T + 1, x_true.shape[1]), dtype=float)
    pi_values = np.zeros(lambda_values.size, dtype=float)
    gamma_values = np.zeros(lambda_values.size, dtype=float)
    bar_gamma_values = np.zeros(lambda_values.size, dtype=float)

    for idx, lam in enumerate(lambda_values):
        adapt_m, _, _, _, diagnostics = kalman_filter_with_online_covariance_adaptation(
            y=y_adv,
            u=u_controls,
            A_t=mats["A_t"],
            B_t=mats["B_t"],
            H_t=mats["H_t"],
            D_t=mats["D_t"],
            Q_t=mats["Q_t"],
            R_t=mats["R_t"],
            m0=m0,
            P0=P0,
            attack_targets=attack_targets,
            lam=float(lam),
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
        )
        adapt_means[idx] = adapt_m
        pi_values[idx] = float(diagnostics["pi_t"][attack_t])
        gamma_values[idx] = float(diagnostics["gamma_t"][attack_t])
        bar_gamma_values[idx] = float(diagnostics["bar_gamma_t"][attack_t])

    return {
        "T": T,
        "attack_t": attack_t,
        "seed": seed,
        "epsilon": epsilon,
        "lambda_scales": np.asarray(lambda_scales, dtype=float),
        "lambda_max": float(lambda_max),
        "lambda_values": np.asarray(lambda_values, dtype=float),
        "omega_h": omega_h,
        "omega_o": omega_o,
        "delta_threshold": delta_threshold,
        "x_true": x_true,
        "clean_m": clean_m,
        "attack_m": attack_m,
        "adapt_means": adapt_means,
        "pi_values": pi_values,
        "gamma_values": gamma_values,
        "bar_gamma_values": bar_gamma_values,
    }


def evaluate_single_monte_carlo_run_lambda_sweep(
    *,
    run_seed: int,
    T: int,
    attack_t: int,
    epsilon: float,
    lambda_scales: np.ndarray,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    var_entries: float = 8.0,
) -> dict[str, np.ndarray | float]:
    """
    Evaluate one random SSM for the full lambda sweep.

    The attack is constructed once for the run, then the defended estimate is
    recomputed for each `lambda = c lambda_max` so the effect curves stay
    comparable across random systems with different covariance scales.
    """
    rng = np.random.default_rng(run_seed)
    std = float(np.sqrt(var_entries))
    params = sample_random_ssm_run_params(rng, n_x=2, n_y=2, n_u=2, std=std)

    zero_drifts = {
        "dA": np.zeros_like(params["A0"]),
        "dB": np.zeros_like(params["B0"]),
        "dH": np.zeros_like(params["H0"]),
        "dD": np.zeros_like(params["D0"]),
        "dQ": np.zeros_like(params["Q0"]),
        "dR": np.zeros_like(params["R0"]),
    }

    x_true, y_clean, u_controls, mats = simulate_lgssm_nd(
        A0=params["A0"],
        B0=params["B0"],
        H0=params["H0"],
        D0=params["D0"],
        T=T,
        seed=run_seed,
        x0=params["x0"],
        Q0=params["Q0"],
        R0=params["R0"],
        dA=zero_drifts["dA"],
        dB=zero_drifts["dB"],
        dH=zero_drifts["dH"],
        dD=zero_drifts["dD"],
        dQ=zero_drifts["dQ"],
        dR=zero_drifts["dR"],
        u_low=-0.5,
        u_high=0.5,
    )

    attack_data = build_kf_attack(
        y_clean=y_clean,
        u_controls=u_controls,
        mats=mats,
        attack_t=attack_t,
        m0=params["m0"],
        P0=params["P0"],
        epsilon=epsilon,
    )
    y_adv = np.asarray(attack_data["y_adv"], dtype=float)
    attack_targets = {attack_t: np.asarray(attack_data["adv_target"], dtype=float)}
    lambda_max = lambda_max_from_observation_covariance(mats["R_t"][attack_t])
    lambda_values = lambda_scales * lambda_max

    clean_m, _, _, _ = kalman_filter_nd(
        y=y_clean,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=params["m0"],
        P0=params["P0"],
    )
    attack_m, _, _, _ = kalman_filter_nd(
        y=y_adv,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=params["m0"],
        P0=params["P0"],
    )

    clean_local = float(np.sum(np.abs(x_true[attack_t] - clean_m[attack_t])))
    clean_global = float(np.sum(np.abs(x_true - clean_m)))
    attack_local = float(np.sum(np.abs(x_true[attack_t] - attack_m[attack_t])))
    attack_global = float(np.sum(np.abs(x_true - attack_m)))

    clean_local_by_lambda = np.zeros(lambda_scales.size, dtype=float)
    clean_global_by_lambda = np.zeros(lambda_scales.size, dtype=float)
    local_by_lambda = np.zeros(lambda_scales.size, dtype=float)
    global_by_lambda = np.zeros(lambda_scales.size, dtype=float)

    for idx, lam in enumerate(lambda_values):
        clean_adapt_m, _, _, _, _ = kalman_filter_with_online_covariance_adaptation(
            y=y_clean,
            u=u_controls,
            A_t=mats["A_t"],
            B_t=mats["B_t"],
            H_t=mats["H_t"],
            D_t=mats["D_t"],
            Q_t=mats["Q_t"],
            R_t=mats["R_t"],
            m0=params["m0"],
            P0=params["P0"],
            attack_targets=attack_targets,
            lam=float(lam),
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
        )
        adapt_m, _, _, _, _ = kalman_filter_with_online_covariance_adaptation(
            y=y_adv,
            u=u_controls,
            A_t=mats["A_t"],
            B_t=mats["B_t"],
            H_t=mats["H_t"],
            D_t=mats["D_t"],
            Q_t=mats["Q_t"],
            R_t=mats["R_t"],
            m0=params["m0"],
            P0=params["P0"],
            attack_targets=attack_targets,
            lam=float(lam),
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
        )
        clean_local_by_lambda[idx] = float(np.sum(np.abs(x_true[attack_t] - clean_adapt_m[attack_t])))
        clean_global_by_lambda[idx] = float(np.sum(np.abs(x_true - clean_adapt_m)))
        local_by_lambda[idx] = float(np.sum(np.abs(x_true[attack_t] - adapt_m[attack_t])))
        global_by_lambda[idx] = float(np.sum(np.abs(x_true - adapt_m)))

    return {
        "clean_local": clean_local,
        "clean_global": clean_global,
        "attack_local": attack_local,
        "attack_global": attack_global,
        "lambda_max": float(lambda_max),
        "clean_local_by_lambda": clean_local_by_lambda,
        "clean_global_by_lambda": clean_global_by_lambda,
        "local_by_lambda": local_by_lambda,
        "global_by_lambda": global_by_lambda,
    }


def run_monte_carlo_lambda_sweep(
    *,
    N_runs: int,
    T: int,
    attack_t: int,
    epsilon: float,
    lambda_scales: np.ndarray,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    base_seed: int,
) -> dict[str, np.ndarray | float | int]:
    """
    Run a compact Monte Carlo sweep over the requested lambda scales.

    The output stores full per-run arrays so later plots can switch between
    mean, median, or spread summaries without rerunning the experiment.
    """
    clean_local = np.full(N_runs, np.nan, dtype=float)
    clean_global = np.full(N_runs, np.nan, dtype=float)
    attack_local = np.full(N_runs, np.nan, dtype=float)
    attack_global = np.full(N_runs, np.nan, dtype=float)
    clean_local_by_lambda = np.full((lambda_scales.size, N_runs), np.nan, dtype=float)
    clean_global_by_lambda = np.full((lambda_scales.size, N_runs), np.nan, dtype=float)
    local_by_lambda = np.full((lambda_scales.size, N_runs), np.nan, dtype=float)
    global_by_lambda = np.full((lambda_scales.size, N_runs), np.nan, dtype=float)
    lambda_max_by_run = np.full(N_runs, np.nan, dtype=float)

    for run_idx in range(N_runs):
        run_seed = base_seed + 1000 * run_idx
        print(f"[MC-lambda] run {run_idx + 1}/{N_runs} (seed={run_seed})")
        try:
            effects = evaluate_single_monte_carlo_run_lambda_sweep(
                run_seed=run_seed,
                T=T,
                attack_t=attack_t,
                epsilon=epsilon,
                lambda_scales=lambda_scales,
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
            )
            clean_local[run_idx] = float(effects["clean_local"])
            clean_global[run_idx] = float(effects["clean_global"])
            attack_local[run_idx] = float(effects["attack_local"])
            attack_global[run_idx] = float(effects["attack_global"])
            lambda_max_by_run[run_idx] = float(effects["lambda_max"])
            clean_local_by_lambda[:, run_idx] = np.asarray(effects["clean_local_by_lambda"], dtype=float)
            clean_global_by_lambda[:, run_idx] = np.asarray(effects["clean_global_by_lambda"], dtype=float)
            local_by_lambda[:, run_idx] = np.asarray(effects["local_by_lambda"], dtype=float)
            global_by_lambda[:, run_idx] = np.asarray(effects["global_by_lambda"], dtype=float)
        except Exception as exc:
            print(f"[WARN seed={run_seed}] {type(exc).__name__}: {exc}")

    return {
        "N_runs": N_runs,
        "T": T,
        "attack_t": attack_t,
        "epsilon": epsilon,
        "lambda_scales": np.asarray(lambda_scales, dtype=float),
        "lambda_max_by_run": lambda_max_by_run,
        "omega_h": omega_h,
        "omega_o": omega_o,
        "delta_threshold": delta_threshold,
        "base_seed": base_seed,
        "clean_local": clean_local,
        "clean_global": clean_global,
        "attack_local": attack_local,
        "attack_global": attack_global,
        "clean_local_by_lambda": clean_local_by_lambda,
        "clean_global_by_lambda": clean_global_by_lambda,
        "local_by_lambda": local_by_lambda,
        "global_by_lambda": global_by_lambda,
    }


def summarize_effect_series(values: np.ndarray) -> tuple[float, float, float]:
    """
    Return mean and an interquartile band after suppressing extreme outliers.

    The plotting summary uses Tukey inliers so the reference lines and shaded
    envelopes remain readable even when a few Monte Carlo runs are extreme.
    """
    inliers = tukey_inliers(values)
    return (
        float(np.mean(inliers)),
        float(np.percentile(inliers, 25.0)),
        float(np.percentile(inliers, 75.0)),
    )


def draw_effect_curve_panel(
    *,
    ax: plt.Axes,
    lambda_scales: np.ndarray,
    clean_values: np.ndarray,
    attack_values: np.ndarray,
    clean_lambda_values_by_scale: np.ndarray,
    lambda_values_by_scale: np.ndarray,
    ylabel: str,
) -> None:
    """
    Draw one compact effect panel as a mean curve across lambda values.

    The defended curves are drawn from Monte Carlo means, while the shaded
    bands show the interquartile range after suppressing a few extreme
    outliers. Clean and attacked KF are shown as dashed horizontal references.
    """
    style_axis(ax)

    positive_mask = np.asarray(lambda_scales, dtype=float) > 0.0
    positive_lambda_scales = np.asarray(lambda_scales, dtype=float)[positive_mask]
    positive_clean_lambda_values_by_scale = np.asarray(clean_lambda_values_by_scale, dtype=float)[positive_mask]
    positive_lambda_values_by_scale = np.asarray(lambda_values_by_scale, dtype=float)[positive_mask]

    if positive_lambda_scales.size == 0:
        raise ValueError("At least one positive lambda scale is required for log-scale effect plots.")

    clean_mean, clean_q25, clean_q75 = summarize_effect_series(clean_values)
    attack_mean, attack_q25, attack_q75 = summarize_effect_series(attack_values)

    clean_adapt_mean = np.zeros(positive_lambda_scales.size, dtype=float)
    clean_adapt_q25 = np.zeros(positive_lambda_scales.size, dtype=float)
    clean_adapt_q75 = np.zeros(positive_lambda_scales.size, dtype=float)
    defended_mean = np.zeros(positive_lambda_scales.size, dtype=float)
    defended_q25 = np.zeros(positive_lambda_scales.size, dtype=float)
    defended_q75 = np.zeros(positive_lambda_scales.size, dtype=float)
    for lam_idx in range(positive_lambda_scales.size):
        clean_adapt_mean[lam_idx], clean_adapt_q25[lam_idx], clean_adapt_q75[lam_idx] = summarize_effect_series(
            positive_clean_lambda_values_by_scale[lam_idx]
        )
        defended_mean[lam_idx], defended_q25[lam_idx], defended_q75[lam_idx] = summarize_effect_series(
            positive_lambda_values_by_scale[lam_idx]
        )

    ax.fill_between(
        positive_lambda_scales,
        clean_adapt_q25,
        clean_adapt_q75,
        color="#CFE3F2",
        alpha=0.0,
        zorder=1,
    )

    ax.fill_between(
        positive_lambda_scales,
        defended_q25,
        defended_q75,
        color="#B9D9A9",
        alpha=0.24,
        zorder=1,
    )
    ax.plot(
        positive_lambda_scales,
        defended_mean,
        color="#7FB26A",
        linewidth=2.1,
        marker="o",
        markersize=4.5,
        label="KF + cov-adapt",
        zorder=3,
    )

    ax.axhline(
        clean_mean,
        color="#5C97BF",
        linewidth=1.9,
        linestyle="--",
        label="Clean KF (sin cov-adapt)",
        zorder=4,
    )
    ax.axhline(
        attack_mean,
        color="#F0A79D",
        linewidth=1.8,
        linestyle="--",
        label="Attacked KF",
        zorder=4,
    )

    effect_values = np.concatenate(
        [
            np.array(
                [
                    clean_q25,
                    clean_q75,
                    attack_q25,
                    attack_q75,
                    float(np.min(clean_adapt_q25)),
                    float(np.max(clean_adapt_q75)),
                ],
                dtype=float,
            ),
            defended_q25,
            defended_q75,
        ]
    )
    effect_min = float(np.min(effect_values))
    effect_max = float(np.max(effect_values))
    effect_pad = 0.08 * max(effect_max - effect_min, 1e-8)
    ax.set_ylim(effect_min - effect_pad, effect_max + effect_pad)

    ax.set_xscale("log")
    ax.set_xlim(
        float(np.min(positive_lambda_scales)) * 0.92,
        float(np.max(positive_lambda_scales)) * 1.08,
    )
    ax.xaxis.set_major_locator(mticker.FixedLocator(positive_lambda_scales))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{scale:g}" for scale in positive_lambda_scales]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.minorticks_off()
    ax.set_xlabel(r"$c$ in $\lambda = c\,\lambda_{\max}$")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, borderpad=0.35)


def draw_state_dimension_panel(
    *,
    ax: plt.Axes,
    dim_idx: int,
    x_true: np.ndarray,
    clean_m: np.ndarray,
    attack_m: np.ndarray,
    adapt_means: np.ndarray,
    lambda_scales: np.ndarray,
    lambda_line_colors: list[str],
    attack_t: int,
    show_lambda_note: bool,
    lambda_max_reference: float,
) -> None:
    """
    Draw one hidden-state trajectory panel without uncertainty bands.

    Both state dimensions share the same visual treatment so the only changes
    across the two panels are the selected component and the axis label.
    """
    tt = np.arange(x_true.shape[0])
    true_color = "#6A6F77"
    clean_color = "#9BCBE7"
    attack_color = "#F0A79D"
    attack_marker_color = "#D8BE74"

    style_axis(ax)
    ax.axvspan(attack_t - 0.32, attack_t + 0.32, color="#F6E7B4", alpha=0.38, zorder=0)
    ax.axvline(attack_t, color=attack_marker_color, linewidth=1.08, zorder=1)

    ax.plot(
        tt,
        x_true[:, dim_idx],
        color=true_color,
        linewidth=1.4,
        marker="o",
        markersize=2.6,
        label=rf"True $s_t^{{({dim_idx + 1})}}$",
        zorder=5,
    )
    ax.plot(
        tt,
        clean_m[:, dim_idx],
        color=clean_color,
        linewidth=1.9,
        label="Clean KF",
        zorder=3,
    )
    ax.plot(
        tt,
        attack_m[:, dim_idx],
        color=attack_color,
        linewidth=1.9,
        linestyle="--",
        label="Attacked KF",
        zorder=4,
    )

    for lam_idx, _ in enumerate(lambda_scales):
        ax.plot(
            tt,
            adapt_means[lam_idx, :, dim_idx],
            color=lambda_line_colors[lam_idx],
            linewidth=1.8,
            alpha=0.72,
            zorder=2,
        )

    state_curves = [x_true[:, dim_idx], clean_m[:, dim_idx], attack_m[:, dim_idx]]
    state_curves.extend([adapt_means[lam_idx, :, dim_idx] for lam_idx in range(lambda_scales.size)])
    state_min = float(np.min([np.min(curve) for curve in state_curves]))
    state_max = float(np.max([np.max(curve) for curve in state_curves]))
    state_pad = 0.06 * max(state_max - state_min, 1e-8)
    ax.set_ylim(state_min - state_pad, state_max + state_pad)

    ax.set_xlim(-0.15, x_true.shape[0] - 0.85)
    ax.set_xlabel("time t")
    ax.set_ylabel(rf"$s_t^{{({dim_idx + 1})}}$")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, borderpad=0.35)

    if show_lambda_note:
        ax.text(
            0.985,
            0.02,
            rf"$\lambda_{{\max}}={lambda_max_reference:.3g}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.4,
            color="#4A4A4A",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="#D6D6D6",
                alpha=0.92,
            ),
        )


def add_lambda_colorbar(
    *,
    fig: plt.Figure,
    axes: list[plt.Axes],
    lambda_scales: np.ndarray,
) -> None:
    """
    Add a dedicated viridis color scale for the lambda trajectories.

    The state-panel legends stay focused on the baseline trajectories, while
    this colorbar communicates how the defended curves map to the lambda sweep.
    """
    lambda_scales = np.asarray(lambda_scales, dtype=float)
    _, norm, positive_scales = build_lambda_colormap(lambda_scales)
    if positive_scales.size == 0:
        return

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("viridis"))
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        fraction=0.040,
        pad=0.020,
    )
    colorbar.set_label(r"$c$ in $\lambda = c\,\lambda_{\max}$")
    colorbar.set_ticks(positive_scales)
    colorbar.set_ticklabels([f"{scale:g}" for scale in positive_scales])


def plot_lambda_sweep_figure(
    *,
    x_true: np.ndarray,
    clean_m: np.ndarray,
    attack_m: np.ndarray,
    adapt_means: np.ndarray,
    lambda_scales: np.ndarray,
    lambda_max_reference: float,
    clean_local: np.ndarray,
    clean_global: np.ndarray,
    attack_local: np.ndarray,
    attack_global: np.ndarray,
    clean_local_by_lambda: np.ndarray,
    clean_global_by_lambda: np.ndarray,
    local_by_lambda: np.ndarray,
    global_by_lambda: np.ndarray,
    attack_t: int,
    outpath: str,
) -> None:
    """
    Plot the requested lambda-sweep figure without uncertainty intervals.

    The first two panels show the two hidden-state dimensions, while the last
    two panels show defended-effect mean curves with dashed clean/attack
    references and shaded Monte Carlo spread.
    """
    set_plot_theme()

    lambda_line_colors = lambda_colors(lambda_scales)

    fig = plt.figure(figsize=(20.2, 4.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.7, 1.7, 1.45, 1.45], wspace=0.07)
    ax_state_1 = fig.add_subplot(grid[0, 0])
    ax_state_2 = fig.add_subplot(grid[0, 1], sharex=ax_state_1)
    ax_local = fig.add_subplot(grid[0, 2])
    ax_global = fig.add_subplot(grid[0, 3])

    draw_state_dimension_panel(
        ax=ax_state_1,
        dim_idx=0,
        x_true=x_true,
        clean_m=clean_m,
        attack_m=attack_m,
        adapt_means=adapt_means,
        lambda_scales=lambda_scales,
        lambda_line_colors=lambda_line_colors,
        attack_t=attack_t,
        show_lambda_note=False,
        lambda_max_reference=lambda_max_reference,
    )
    draw_state_dimension_panel(
        ax=ax_state_2,
        dim_idx=1,
        x_true=x_true,
        clean_m=clean_m,
        attack_m=attack_m,
        adapt_means=adapt_means,
        lambda_scales=lambda_scales,
        lambda_line_colors=lambda_line_colors,
        attack_t=attack_t,
        show_lambda_note=True,
        lambda_max_reference=lambda_max_reference,
    )
    add_lambda_colorbar(
        fig=fig,
        axes=[ax_state_1, ax_state_2],
        lambda_scales=lambda_scales,
    )

    draw_effect_curve_panel(
        ax=ax_local,
        lambda_scales=lambda_scales,
        clean_values=clean_local,
        attack_values=attack_local,
        clean_lambda_values_by_scale=clean_local_by_lambda,
        lambda_values_by_scale=local_by_lambda,
        ylabel="Local effect",
    )
    draw_effect_curve_panel(
        ax=ax_global,
        lambda_scales=lambda_scales,
        clean_values=clean_global,
        attack_values=attack_global,
        clean_lambda_values_by_scale=clean_global_by_lambda,
        lambda_values_by_scale=global_by_lambda,
        ylabel="Global effect",
    )

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(outpath, facecolor="white", dpi=300)
    plt.close(fig)


def main() -> None:
    """
    Generate the lambda-sweep covariance-adaptation figure and caches.

    The defaults are still lightweight enough for experimentation, and the
    lambda scales can be overridden through `COVADAPT_LAMBDA_SCALES`.
    """
    T = int(os.environ.get("COVADAPT_T", "10"))
    attack_t = int(os.environ.get("COVADAPT_ATTACK_T", "5"))
    seed = int(os.environ.get("COVADAPT_SEED", "202"))
    epsilon = float(os.environ.get("COVADAPT_EPS", "5.991"))
    raw_lambda_scales = os.environ.get(
        "COVADAPT_LAMBDA_SCALES",
        "0.2,0.5,1.0,5.0,10.0,50.0,100.0",
    )
    omega_h = float(os.environ.get("COVADAPT_OMEGA_H", "0.50"))
    omega_o = float(os.environ.get("COVADAPT_OMEGA_O", "0.50"))
    delta_threshold = float(os.environ.get("COVADAPT_DELTA", "0.20"))
    N_runs = int(os.environ.get("COVADAPT_MC_RUNS", "1000"))
    mc_seed = int(os.environ.get("COVADAPT_MC_BASE_SEED", "20"))
    force_reference = os.environ.get("COVADAPT_FORCE_REFERENCE", "0") == "1"
    force_mc = os.environ.get("COVADAPT_FORCE_MC", "0") == "1"

    lambda_scales = parse_lambda_scales(raw_lambda_scales)

    if not (0 <= attack_t <= T):
        raise ValueError("attack_t must satisfy 0 <= attack_t <= T")
    if not np.isclose(omega_h + omega_o, 1.0, atol=1e-9):
        raise ValueError("omega_h and omega_o must sum to 1.")
    if not (0.0 <= delta_threshold <= 1.0):
        raise ValueError("delta_threshold must lie in [0, 1].")

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    lambda_tag = "-".join(f"{value:g}" for value in lambda_scales).replace(".", "p")
    outpath = os.path.join(
        out_dir,
        f"comparison_covadapt_lambdas_t{attack_t}_T{T}_seed{seed}_N{N_runs}_{lambda_tag}.png",
    )
    reference_data_path = data_path_for_plot(outpath.replace(".png", "_reference.png"))
    mc_data_path = data_path_for_plot(outpath.replace(".png", "_mc.png"))

    def compute_reference_data() -> dict[str, np.ndarray | float | int]:
        return evaluate_reference_lambda_sweep(
            T=T,
            attack_t=attack_t,
            seed=seed,
            epsilon=epsilon,
            lambda_scales=lambda_scales,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
        )

    def compute_mc_data() -> dict[str, np.ndarray | float | int]:
        return run_monte_carlo_lambda_sweep(
            N_runs=N_runs,
            T=T,
            attack_t=attack_t,
            epsilon=epsilon,
            lambda_scales=lambda_scales,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            base_seed=mc_seed,
        )

    reference_data = cached_npz(reference_data_path, compute_reference_data, force=force_reference)
    mc_data = cached_npz(mc_data_path, compute_mc_data, force=force_mc)
    required_mc_keys = {"clean_local_by_lambda", "clean_global_by_lambda"}
    if not required_mc_keys.issubset(mc_data):
        print("[cache] existing Monte Carlo cache is missing clean cov-adapt curves; recomputing.")
        mc_data = cached_npz(mc_data_path, compute_mc_data, force=True)

    plot_lambda_sweep_figure(
        x_true=np.asarray(reference_data["x_true"], dtype=float),
        clean_m=np.asarray(reference_data["clean_m"], dtype=float),
        attack_m=np.asarray(reference_data["attack_m"], dtype=float),
        adapt_means=np.asarray(reference_data["adapt_means"], dtype=float),
        lambda_scales=np.asarray(reference_data["lambda_scales"], dtype=float),
        lambda_max_reference=float(reference_data["lambda_max"]),
        clean_local=np.asarray(mc_data["clean_local"], dtype=float),
        clean_global=np.asarray(mc_data["clean_global"], dtype=float),
        attack_local=np.asarray(mc_data["attack_local"], dtype=float),
        attack_global=np.asarray(mc_data["attack_global"], dtype=float),
        clean_local_by_lambda=np.asarray(mc_data["clean_local_by_lambda"], dtype=float),
        clean_global_by_lambda=np.asarray(mc_data["clean_global_by_lambda"], dtype=float),
        local_by_lambda=np.asarray(mc_data["local_by_lambda"], dtype=float),
        global_by_lambda=np.asarray(mc_data["global_by_lambda"], dtype=float),
        attack_t=attack_t,
        outpath=outpath,
    )
    print(f"Saved figure to: {outpath}")


if __name__ == "__main__":
    main()
