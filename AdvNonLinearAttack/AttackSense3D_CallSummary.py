#!/usr/bin/env python3
"""
Summarize clean-versus-attacked call probabilities in the 3D AttackSense setup.

This script reuses the 3D nonlinear-sensing experiment from
`AttackSense3D.py`, but changes the final visualization completely.

Experiment design:
    - Horizon is fixed to T = 5.
    - The attacked observation is always the last one, o_T.
    - We repeat the simulation for many Monte Carlo runs.
    - For each run we compute:
          E[g(s_T) | o_0:T]                    (clean)
          E[g(s_T) | o_0:T-1, o_T^*(alpha)]    (attacked)
      for several ellipsoid coverage levels alpha.

Important convention used here:
    The requested values are interpreted as coverage levels of the 3D
    chi-square ellipsoid. Each one is converted into the corresponding attack
    threshold epsilon via:

        epsilon(alpha) = chi2_ppf(alpha; df = 3).

Outputs:
    - One figure with two panels in a single row.
      Left panel:
          clean and attacked call probabilities across Monte Carlo runs,
          sorted by the clean probability.
      Right panel:
          the percentage False/Total versus ellipsoid coverage, annotated with
          the total number of calls so we can see how often they happen.

    - One cached `.npz` file containing the clean and attacked probability
      arrays used to redraw the figure without rerunning the attacks.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import chi2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared_ssm.artifacts import data_path_for_plot
from shared_ssm.artifacts import load_npz
from shared_ssm.artifacts import save_npz

try:
    from AttackSense3D import (
        g_scalar,
        g_scalar_grad,
        get_system_parameters,
    )
except ModuleNotFoundError:
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)

    from AttackSense3D import (
        g_scalar,
        g_scalar_grad,
        get_system_parameters,
    )

from shared_ssm.legacy import estimate_E_g
from shared_ssm.legacy import kalman_filter_nd_current_observation as kalman_filter_nd
from shared_ssm.legacy import rts_smoother_nd
from shared_ssm.legacy import simulate_lgssm_nd_current_observation as simulate_lgssm_nd
from shared_ssm.legacy import white_box_point_attack_nd


# ============================================================
# Experiment configuration
# ============================================================
HORIZON_T = 5
ATTACK_T = HORIZON_T
N_RUNS = 500

# The user-requested values are interpreted as ellipsoid coverage levels.
PROBABILITY_COVERAGE_LEVELS = np.array([0.20, 0.50, 0.75, 0.90], dtype=float)
CALL_RATIO_COVERAGE_LEVELS = np.array([0.05,0.1, 0.20, 0.3,0.4, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98,0.995], dtype=float)

# Calls are probabilities above this threshold.
CALL_THRESHOLD = 0.90

# Attack target and optimizer settings match the original 3D experiment.
M_STAR = np.array([1.00], dtype=float)
ETA = 1.5
N_STEPS = 500
N_MC_OPT = 96
N_MC_EST = 1200

FIGURES_DIRNAME = os.path.join("outputs", "figures")


# ============================================================
# Plot styling
# ============================================================
def _set_plot_theme() -> None:
    """Apply the soft figure style used across the experiment scripts."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10.8,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11.2,
            "legend.fontsize": 10.2,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _style_axis(ax) -> None:
    """Apply a clean axis style with dashed background guides."""
    ax.set_facecolor("#FCFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.85)
    ax.spines["bottom"].set_alpha(0.85)
    ax.minorticks_on()
    ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.24, linewidth=0.7)
    ax.grid(True, which="minor", axis="both", linestyle="--", alpha=0.12, linewidth=0.5)
    ax.set_axisbelow(True)


def _inlier_mask_iqr(values: np.ndarray) -> np.ndarray:
    """
    Return a Tukey-IQR inlier mask for one-dimensional data.

    The mask is used to ignore extreme attacked probabilities inside each
    probability bin so the plotted mean is not dominated by a few outliers.
    """
    values = np.asarray(values, dtype=float)
    if values.size <= 3:
        return np.ones(values.shape, dtype=bool)

    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = q3 - q1
    if iqr <= 0.0:
        return np.ones(values.shape, dtype=bool)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (values >= lower) & (values <= upper)


def _binned_attack_summary(
    clean_prob: np.ndarray,
    target_prob: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group attacked probabilities by the clean probability level.

    The returned arrays contain:
        - x positions: mean clean probability inside each occupied bin
        - y means:     mean attacked probability in that bin after trimming
                       outliers with the IQR rule
        - y lows:      one standard deviation below the trimmed mean
        - y highs:     one standard deviation above the trimmed mean
    """
    clean_prob = np.asarray(clean_prob, dtype=float)
    target_prob = np.asarray(target_prob, dtype=float)

    if clean_prob.shape != target_prob.shape:
        raise ValueError("clean_prob and target_prob must have the same shape")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    x_vals: list[float] = []
    y_means: list[float] = []
    y_lows: list[float] = []
    y_highs: list[float] = []

    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]

        if idx == n_bins - 1:
            mask = (clean_prob >= left) & (clean_prob <= right)
        else:
            mask = (clean_prob >= left) & (clean_prob < right)

        if not np.any(mask):
            continue

        clean_bin = clean_prob[mask]
        target_bin = target_prob[mask]
        inlier_mask = _inlier_mask_iqr(target_bin)
        if np.any(inlier_mask):
            clean_bin = clean_bin[inlier_mask]
            target_bin = target_bin[inlier_mask]

        bin_mean_x = float(np.mean(clean_bin))
        bin_mean_y = float(np.mean(target_bin))
        bin_std_y = float(np.std(target_bin))

        x_vals.append(bin_mean_x)
        y_means.append(bin_mean_y)
        y_lows.append(bin_mean_y - bin_std_y)
        y_highs.append(bin_mean_y + bin_std_y)

    return (
        np.asarray(x_vals),
        np.asarray(y_means),
        np.asarray(y_lows),
        np.asarray(y_highs),
    )


# ============================================================
# Monte Carlo computation
# ============================================================
def _coverage_to_epsilon(levels: np.ndarray) -> np.ndarray:
    """Convert 3D ellipsoid coverage levels into chi-square thresholds."""
    return chi2.ppf(levels, df=3)


def _compute_clean_probability(
    y: np.ndarray,
    u: np.ndarray,
    mats: dict[str, np.ndarray],
    m0: np.ndarray,
    P0: np.ndarray,
) -> float:
    """Return the clean estimate E[g(s_T) | o_0:T] for one run."""
    m_filt, P_filt, m_pred, P_pred = kalman_filter_nd(
        y=y,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=m0,
        P0=P0,
    )
    m_smooth, P_smooth = rts_smoother_nd(
        m_filt=m_filt,
        P_filt=P_filt,
        m_pred=m_pred,
        P_pred=P_pred,
        A_t=mats["A_t"],
    )
    mu_g_clean, _ = estimate_E_g(
        m=m_smooth[ATTACK_T],
        P=P_smooth[ATTACK_T],
        g=g_scalar,
        n_mc=N_MC_EST,
        seed=77,
    )
    return float(mu_g_clean[0])


def _compute_attacked_probability(
    *,
    y: np.ndarray,
    u: np.ndarray,
    mats: dict[str, np.ndarray],
    m0: np.ndarray,
    P0: np.ndarray,
    epsilon: float,
    attack_seed: int,
) -> float:
    """Return E[g(s_T) | o_0:T-1, o_T^*] for one run and one ellipsoid size."""
    y_star, _ = white_box_point_attack_nd(
        t=ATTACK_T,
        y=y,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        P0=P0,
        m0=m0,
        epsilon=epsilon,
        M_star=M_STAR,
        g=g_scalar,
        g_grad=g_scalar_grad,
        eta=ETA,
        n_steps=N_STEPS,
        n_mc=N_MC_OPT,
        seed=attack_seed,
    )

    y_adv = y.copy()
    y_adv[ATTACK_T] = y_star

    m_filt, P_filt, m_pred, P_pred = kalman_filter_nd(
        y=y_adv,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=m0,
        P0=P0,
    )
    m_smooth, P_smooth = rts_smoother_nd(
        m_filt=m_filt,
        P_filt=P_filt,
        m_pred=m_pred,
        P_pred=P_pred,
        A_t=mats["A_t"],
    )
    mu_g_attack, _ = estimate_E_g(
        m=m_smooth[ATTACK_T],
        P=P_smooth[ATTACK_T],
        g=g_scalar,
        n_mc=N_MC_EST,
        seed=77,
    )
    return float(mu_g_attack[0])


def _select_rows_by_levels(
    values_by_level: np.ndarray,
    all_levels: np.ndarray,
    requested_levels: np.ndarray,
) -> np.ndarray:
    """Extract rows associated with selected coverage levels, preserving order."""
    selected_rows: list[np.ndarray] = []
    for level in requested_levels:
        match_idx = int(np.where(np.isclose(all_levels, level))[0][0])
        selected_rows.append(values_by_level[match_idx])
    return np.stack(selected_rows, axis=0)


def compute_summary_data(
    *,
    n_runs: int = N_RUNS,
    probability_levels: np.ndarray = PROBABILITY_COVERAGE_LEVELS,
    call_ratio_levels: np.ndarray = CALL_RATIO_COVERAGE_LEVELS,
) -> dict[str, np.ndarray]:
    """
    Run the clean-versus-attacked comparison for several ellipsoid coverages.

    The clean trajectory is simulated only once per Monte Carlo run, then the
    attacked probability is recomputed for every coverage level on top of that
    same clean realization.
    """
    probability_levels = np.asarray(probability_levels, dtype=float)
    call_ratio_levels = np.asarray(call_ratio_levels, dtype=float)
    all_coverage_levels = np.unique(np.concatenate([probability_levels, call_ratio_levels]))
    epsilon_levels = _coverage_to_epsilon(all_coverage_levels)

    true_prob = np.zeros(n_runs, dtype=float)
    clean_prob = np.zeros(n_runs, dtype=float)
    attack_prob_all = np.zeros((len(all_coverage_levels), n_runs), dtype=float)

    for run_id in range(n_runs):
        pars = get_system_parameters()
        sim_seed = 2025 + run_id

        x_true, o_obs, u_ctrl, mats = simulate_lgssm_nd(
            A0=pars["A0"],
            B0=pars["B0"],
            H0=pars["H0"],
            D0=pars["D0"],
            T=HORIZON_T,
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
        true_prob[run_id] = float(g_scalar(x_true[ATTACK_T]))

        clean_prob[run_id] = _compute_clean_probability(
            y=o_obs,
            u=u_ctrl,
            mats=mats,
            m0=pars["m0"],
            P0=pars["P0"],
        )

        for level_idx, epsilon in enumerate(epsilon_levels):
            attack_seed = 6000 + 1000 * level_idx + run_id
            attack_prob_all[level_idx, run_id] = _compute_attacked_probability(
                y=o_obs,
                u=u_ctrl,
                mats=mats,
                m0=pars["m0"],
                P0=pars["P0"],
                epsilon=float(epsilon),
                attack_seed=attack_seed,
            )

        print(
            f"[{run_id + 1:3d}/{n_runs}] "
            f"clean={clean_prob[run_id]:.4f} | "
            f"attack(max coverage)={attack_prob_all[-1, run_id]:.4f}"
        )

    true_calls = true_prob > CALL_THRESHOLD
    clean_calls = clean_prob > CALL_THRESHOLD
    attack_calls = attack_prob_all > CALL_THRESHOLD
    clean_false_calls = clean_calls & (~true_calls)
    false_calls = attack_calls & (~true_calls[None, :])
    total_call_count = np.sum(attack_calls, axis=1).astype(int)
    false_call_count = np.sum(false_calls, axis=1).astype(int)
    clean_total_call_count = int(np.sum(clean_calls))
    clean_false_call_count = int(np.sum(clean_false_calls))
    false_over_total_pct = np.divide(
        100.0 * false_call_count,
        total_call_count,
        out=np.zeros_like(false_call_count, dtype=float),
        where=total_call_count > 0,
    )
    clean_false_positive_rate = (
        100.0 * clean_false_call_count / clean_total_call_count
        if clean_total_call_count > 0
        else 0.0
    )

    attack_prob_for_plot = _select_rows_by_levels(attack_prob_all, all_coverage_levels, probability_levels)
    false_over_total_for_plot = _select_rows_by_levels(
        false_over_total_pct[:, None],
        all_coverage_levels,
        call_ratio_levels,
    ).reshape(-1)
    total_call_count_for_plot = _select_rows_by_levels(
        total_call_count[:, None],
        all_coverage_levels,
        call_ratio_levels,
    ).reshape(-1)

    return {
        "probability_coverage_levels": probability_levels,
        "call_ratio_coverage_levels": call_ratio_levels,
        "all_coverage_levels": all_coverage_levels,
        "epsilon_levels": epsilon_levels,
        "true_prob": true_prob,
        "clean_prob": clean_prob,
        "attack_prob_for_plot": attack_prob_for_plot,
        "clean_call_count": np.array([int(np.sum(clean_calls))], dtype=int),
        "clean_false_call_count": np.array([clean_false_call_count], dtype=int),
        "clean_false_positive_rate": np.array([clean_false_positive_rate], dtype=float),
        "attack_prob_all": attack_prob_all,
        "total_call_count_all": total_call_count,
        "false_call_count_all": false_call_count,
        "false_over_total_pct_for_plot": false_over_total_for_plot,
        "total_call_count_for_plot": total_call_count_for_plot,
        "call_threshold": np.array([CALL_THRESHOLD], dtype=float),
        "n_runs": np.array([n_runs], dtype=int),
    }


# ============================================================
# Plotting
# ============================================================
def plot_call_summary(
    *,
    true_prob: np.ndarray,
    clean_prob: np.ndarray,
    attack_prob: np.ndarray,
    probability_coverage_levels: np.ndarray,
    call_ratio_coverage_levels: np.ndarray,
    false_over_total_pct: np.ndarray,
    clean_false_positive_rate: float,
    total_call_count: np.ndarray,
    n_runs: int,
    outpath: str,
) -> None:
    """
    Draw the two-panel summary figure requested by the user.

    Left panel:
        clean and attacked call probabilities across runs.
    Right panel:
        false-call percentage among total calls across coverage levels.
    """
    _set_plot_theme()

    true_prob = np.asarray(true_prob, dtype=float)
    clean_prob = np.asarray(clean_prob, dtype=float)
    attack_prob = np.asarray(attack_prob, dtype=float)
    probability_coverage_levels = np.asarray(probability_coverage_levels, dtype=float)
    call_ratio_coverage_levels = np.asarray(call_ratio_coverage_levels, dtype=float)
    false_over_total_pct = np.asarray(false_over_total_pct, dtype=float)
    total_call_count = np.asarray(total_call_count, dtype=int)

    colors = ["#9DB7D5", "#A9D1C2", "#F2C9A1", "#E6B5D0"]
    clean_color = "#525252"
    ratio_color = "#D98F8F"
    n_bins = max(30, min(40, clean_prob.size // 2 if clean_prob.size >= 12 else clean_prob.size))

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.9), constrained_layout=True)
    ax_prob, ax_calls = axes

    for ax in axes:
        _style_axis(ax)

    diag_x = np.linspace(0.0, 1.0, 200)
    ax_prob.plot(
        diag_x,
        diag_x,
        color=clean_color,
        linewidth=2.0,
        linestyle="--",
        label="Clean mean",
        zorder=3,
    )

    for color, level, prob_values in zip(colors, probability_coverage_levels, attack_prob):
        x_bin, prob_mean, prob_low, prob_high = _binned_attack_summary(
            true_prob,
            prob_values,
            n_bins=n_bins,
        )
        ax_prob.fill_between(
            x_bin,
            np.clip(prob_low, 0.0, 1.0),
            np.clip(prob_high, 0.0, 1.0),
            color=color,
            alpha=0.16,
            zorder=1,
        )
        ax_prob.plot(
            x_bin,
            prob_mean,
            color=color,
            linewidth=1.9,
            label=fr"Attack mean ($\epsilon={level:.2f}$)",
            zorder=3,
        )

    ax_prob.axhline(
        CALL_THRESHOLD,
        color="#8A8A8A",
        linewidth=1.1,
        linestyle=":",
        alpha=0.9,
        label="Call threshold",
        zorder=2,
    )
    ax_prob.set_xlabel(r"Real clean probability $g(s_T)$")
    ax_prob.set_ylabel(r"Estimated call probability $\mathbb{E}[g(s_T)\mid o_{0:T}]$")
    ax_prob.set_xlim(0.0, 1.0)
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_prob.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_prob.legend(
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
    )

    ax_calls.plot(
        call_ratio_coverage_levels,
        false_over_total_pct,
        color=ratio_color,
        marker="o",
        linewidth=2.2,
        markersize=6.0,
        label="False positive rate",
        zorder=3,
    )
    ax_calls.axhline(
        clean_false_positive_rate,
        color=clean_color,
        linewidth=1.4,
        linestyle="--",
        alpha=0.9,
        label="Clean false positive rate",
        zorder=2,
    )

    ax_calls.set_xticks(call_ratio_coverage_levels)
    ax_calls.set_xlabel(r"Confidence level ($\epsilon$)")
    ax_calls.set_ylabel("False positive rate (%)")
    ax_calls.set_xlim(float(call_ratio_coverage_levels[0]) - 0.02, float(call_ratio_coverage_levels[-1]) + 0.02)
    ax_calls.set_ylim(0.0, min(100.0, max(np.max(false_over_total_pct), 5.0) + 10.0))
    ax_calls.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_calls.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
    )

    # Add a zoomed inset of the first panel for the high-clean-probability tail.
    high_prob_x = diag_x[diag_x > 0.7]
    if high_prob_x.size > 1:
        inset = ax_calls.inset_axes([0.50, 0.08, 0.42, 0.42])
        _style_axis(inset)

        inset.plot(
            high_prob_x,
            high_prob_x,
            color=clean_color,
            linewidth=1.6,
            linestyle="--",
            zorder=3,
        )

        for color, prob_values in zip(colors, attack_prob):
            x_bin, prob_mean, _, _ = _binned_attack_summary(
                true_prob,
                prob_values,
                n_bins=n_bins,
            )
            high_mask = x_bin > 0.7
            if not np.any(high_mask):
                continue
            inset.plot(
                x_bin[high_mask],
                prob_mean[high_mask],
                color=color,
                linewidth=1.45,
                zorder=3,
            )

        inset.axhline(
            CALL_THRESHOLD,
            color="#8A8A8A",
            linewidth=0.9,
            linestyle=":",
            alpha=0.85,
            zorder=2,
        )
        inset.set_xlim(0.75, 1.0)
        inset.set_ylim(0.72, 1.02)
        inset.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        inset.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        inset.tick_params(axis="both", labelsize=7.5)
        inset.set_xlabel("")
        inset.set_ylabel("")

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(module_dir, FIGURES_DIRNAME)
    os.makedirs(figures_dir, exist_ok=True)

    fig_path = os.path.join(
        figures_dir,
        f"call_summary_false_over_total_3d_T{HORIZON_T}_N{N_RUNS}.png",
    )
    data_path = data_path_for_plot(fig_path)
    required_cache_keys = {
        "true_prob",
        "clean_prob",
        "attack_prob_for_plot",
        "probability_coverage_levels",
        "call_ratio_coverage_levels",
        "false_over_total_pct_for_plot",
        "clean_false_positive_rate",
        "total_call_count_for_plot",
        "n_runs",
    }

    if os.path.exists(data_path):
        print(f"[cache] Loading cached data: {data_path}")
        data = load_npz(data_path)
        if not required_cache_keys.issubset(data.keys()):
            print("[cache] Cache is from an older format. Recomputing data...")
            data = compute_summary_data()
            save_npz(data_path, **data)
            print(f"[cache] Updated data: {data_path}")
    else:
        print("[run] Computing clean and attacked call probabilities...")
        data = compute_summary_data()
        save_npz(data_path, **data)
        print(f"[cache] Saved data: {data_path}")

    plot_call_summary(
        true_prob=data["true_prob"],
        clean_prob=data["clean_prob"],
        attack_prob=data["attack_prob_for_plot"],
        probability_coverage_levels=data["probability_coverage_levels"],
        call_ratio_coverage_levels=data["call_ratio_coverage_levels"],
        false_over_total_pct=data["false_over_total_pct_for_plot"],
        clean_false_positive_rate=float(data["clean_false_positive_rate"][0]),
        total_call_count=data["total_call_count_for_plot"],
        n_runs=int(data["n_runs"][0]),
        outpath=fig_path,
    )
    print(f"[saved] Figure written to: {fig_path}")


if __name__ == "__main__":
    main()
