#!/usr/bin/env python3
"""
Compare both weight pairs in the nonlinear covariance-adaptation defense.

The covariance-inflation magnitude is fixed at ``lambda = lambda_max``.  The
experiment varies one complementary weight pair at a time:

1. Prior mixture: ``(omega_h, omega_o) = (q, 1 - q)`` while the posterior
   evidence weights remain fixed at ``(w_M, w_g) = (0.5, 0.5)``.
2. Posterior evidence mixture: ``(w_M, w_g) = (q, 1 - q)`` while the prior
   weights remain fixed at ``(omega_h, omega_o) = (0.5, 0.5)``.

The resulting 2x2 figure uses one row per weight pair.  Its left column shows
overlaid Monte Carlo distributions of the raw posterior attack probability
``gamma_t`` for q in {0.25, 0.50, 0.75}.  Its right column shows defended
``E[g(s_T) | o_0:T]`` against the actual ``g(s_T)`` over a denser weight grid.

This is a separate experiment script so the original lambda-sweep remains
unchanged.  All experiment sizes are defined in ``main`` for easy editing.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import nonlinear_g_covadapt as base

from shared_ssm.artifacts import cached_npz
from shared_ssm.artifacts import data_path_for_plot
from shared_ssm.artifacts import figures_dir_for


def parse_weight_values(raw_values: str) -> np.ndarray:
    """Return sorted unique weights in the closed interval [0, 1]."""
    pieces = [piece.strip() for piece in raw_values.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("At least one weight value is required.")
    values = np.sort(np.unique(np.asarray([float(piece) for piece in pieces], dtype=float)))
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("All weights must lie in [0, 1].")
    return values


def _defended_probability_and_gamma(
    *,
    y_adv: np.ndarray,
    u_controls: np.ndarray,
    mats: dict[str, np.ndarray],
    m0: np.ndarray,
    P0: np.ndarray,
    attack_t: int,
    attack_target: np.ndarray,
    lam: float,
    omega_h: float,
    omega_o: float,
    mahalanobis_weight: float,
    objective_weight: float,
    epsilon: float,
    objective_attack_score_builder,
    n_mc_est: int,
) -> tuple[float, float]:
    """Run one defended filter and return E[g] and the raw gamma at attack_t."""
    defended_filter = base.kalman_filter_with_online_covariance_adaptation_3d(
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
        attack_targets={attack_t: attack_target},
        lam=float(lam),
        omega_h=float(omega_h),
        omega_o=float(omega_o),
        delta_threshold=base.DEFAULT_POSTERIOR_ATTACK_THRESHOLD,
        objective_attack_score_builder=objective_attack_score_builder,
        mahalanobis_epsilon=float(epsilon),
        mahalanobis_evidence_weight=float(mahalanobis_weight),
        objective_evidence_weight=float(objective_weight),
    )
    _, _, defended_probability = base.compute_smoothed_probability(
        m_filt=defended_filter[0],
        P_filt=defended_filter[1],
        m_pred=defended_filter[2],
        P_pred=defended_filter[3],
        A_t=mats["A_t"],
        attack_t=attack_t,
        n_mc_est=n_mc_est,
    )
    diagnostics = defended_filter[4]
    gamma_t = float(np.asarray(diagnostics["gamma_t"], dtype=float)[attack_t])
    return float(defended_probability), gamma_t


def evaluate_weight_sweeps_for_run(
    *,
    run_seed: int,
    T: int,
    attack_t: int,
    coverage: float,
    lambda_scale: float,
    weight_values: np.ndarray,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_mc_est: int,
) -> dict[str, np.ndarray | float]:
    """Simulate one run, build one attack, and evaluate both weight sweeps."""
    pars = base.get_system_parameters()
    epsilon = base.coverage_to_epsilon(coverage)
    weight_values = np.asarray(weight_values, dtype=float)

    x_true, y_clean, u_controls, mats = base.simulate_lgssm_nd(
        A0=pars["A0"],
        B0=pars["B0"],
        H0=pars["H0"],
        D0=pars["D0"],
        T=T,
        seed=run_seed,
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
    attack_data = base.build_attack_on_g(
        y_clean=y_clean,
        u_controls=u_controls,
        mats=mats,
        m0=pars["m0"],
        P0=pars["P0"],
        epsilon=epsilon,
        attack_t=attack_t,
        attack_seed=6000 + run_seed,
        eta=eta,
        n_steps=n_steps,
        n_mc_opt=n_mc_opt,
        clean_true_g_value=float(base.g_scalar(x_true[attack_t])),
        call_threshold=base.DEFAULT_CALL_THRESHOLD,
    )
    y_adv = np.asarray(attack_data["y_adv"], dtype=float)
    attack_target = np.asarray(attack_data["adv_target"], dtype=float)

    # The clean and attacked filters are shared across both weight sweeps.
    clean_filter = base.kalman_filter_nd(
        y=y_clean,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )
    attack_filter = base.kalman_filter_nd(
        y=y_adv,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )
    _, _, clean_probability = base.compute_smoothed_probability(
        m_filt=clean_filter[0],
        P_filt=clean_filter[1],
        m_pred=clean_filter[2],
        P_pred=clean_filter[3],
        A_t=mats["A_t"],
        attack_t=attack_t,
        n_mc_est=n_mc_est,
    )
    _, _, attack_probability = base.compute_smoothed_probability(
        m_filt=attack_filter[0],
        P_filt=attack_filter[1],
        m_pred=attack_filter[2],
        P_pred=attack_filter[3],
        A_t=mats["A_t"],
        attack_t=attack_t,
        n_mc_est=n_mc_est,
    )
    objective_builder = base.build_nonlinear_objective_attack_score_builder(
        attack_t=attack_t,
        clean_state_mean=clean_filter[0][attack_t],
    )

    lambda_max = base.lambda_max_from_observation_covariance(mats["R_t"][attack_t])
    lam = float(lambda_scale) * float(lambda_max)
    omega_probabilities = np.zeros(weight_values.size, dtype=float)
    omega_gammas = np.zeros(weight_values.size, dtype=float)
    evidence_probabilities = np.zeros(weight_values.size, dtype=float)
    evidence_gammas = np.zeros(weight_values.size, dtype=float)

    # The q=0.5 configuration is common to both sweeps.  Cache it locally to
    # avoid evaluating the identical defended filter twice in every MC run.
    defense_cache: dict[tuple[float, float, float, float], tuple[float, float]] = {}

    def run_defense(omega_h: float, omega_o: float, w_m: float, w_g: float) -> tuple[float, float]:
        key = tuple(round(value, 12) for value in (omega_h, omega_o, w_m, w_g))
        if key not in defense_cache:
            defense_cache[key] = _defended_probability_and_gamma(
                y_adv=y_adv,
                u_controls=u_controls,
                mats=mats,
                m0=pars["m0"],
                P0=pars["P0"],
                attack_t=attack_t,
                attack_target=attack_target,
                lam=lam,
                omega_h=omega_h,
                omega_o=omega_o,
                mahalanobis_weight=w_m,
                objective_weight=w_g,
                epsilon=epsilon,
                objective_attack_score_builder=objective_builder,
                n_mc_est=n_mc_est,
            )
        return defense_cache[key]

    for idx, q_value in enumerate(weight_values):
        omega_probabilities[idx], omega_gammas[idx] = run_defense(
            float(q_value),
            float(1.0 - q_value),
            0.5,
            0.5,
        )
        evidence_probabilities[idx], evidence_gammas[idx] = run_defense(
            0.5,
            0.5,
            float(q_value),
            float(1.0 - q_value),
        )

    return {
        "true_g": float(base.g_scalar(x_true[attack_t])),
        "clean_probability": float(clean_probability),
        "attack_probability": float(attack_probability),
        "lambda_max": float(lambda_max),
        "omega_probabilities": omega_probabilities,
        "omega_gammas": omega_gammas,
        "evidence_probabilities": evidence_probabilities,
        "evidence_gammas": evidence_gammas,
    }


def evaluate_weight_sweeps_from_task(task: dict[str, object]) -> dict[str, np.ndarray | float]:
    """Unpack one plain worker task so it remains picklable on Windows."""
    return evaluate_weight_sweeps_for_run(
        run_seed=int(task["run_seed"]),
        T=int(task["T"]),
        attack_t=int(task["attack_t"]),
        coverage=float(task["coverage"]),
        lambda_scale=float(task["lambda_scale"]),
        weight_values=np.asarray(task["weight_values"], dtype=float),
        eta=float(task["eta"]),
        n_steps=int(task["n_steps"]),
        n_mc_opt=int(task["n_mc_opt"]),
        n_mc_est=int(task["n_mc_est"]),
    )


def run_monte_carlo_weight_sweeps(
    *,
    N_runs: int,
    mc_seed: int,
    T: int,
    attack_t: int,
    coverage: float,
    lambda_scale: float,
    weight_values: np.ndarray,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_mc_est: int,
    n_jobs: int,
) -> dict[str, np.ndarray | float | int]:
    """Evaluate the two weight sweeps over a reproducible MC seed sequence."""
    weight_values = np.asarray(weight_values, dtype=float)
    seed_sequence = np.random.SeedSequence(mc_seed)
    run_seeds = np.asarray(
        [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_sequence.spawn(N_runs)],
        dtype=np.uint32,
    )
    tasks = [
        {
            "run_seed": int(run_seed),
            "T": T,
            "attack_t": attack_t,
            "coverage": coverage,
            "lambda_scale": lambda_scale,
            "weight_values": weight_values,
            "eta": eta,
            "n_steps": n_steps,
            "n_mc_opt": n_mc_opt,
            "n_mc_est": n_mc_est,
        }
        for run_seed in run_seeds
    ]

    results: list[dict[str, np.ndarray | float] | None] = [None] * N_runs
    max_workers = max(1, min(int(n_jobs), N_runs))
    if max_workers == 1:
        for idx, task in enumerate(tasks):
            results[idx] = evaluate_weight_sweeps_from_task(task)
            print(f"[weights] completed {idx + 1}/{N_runs} runs")
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            future_to_idx = {
                executor.submit(evaluate_weight_sweeps_from_task, task): idx
                for idx, task in enumerate(tasks)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                result_idx = future_to_idx[future]
                results[result_idx] = future.result()
                completed += 1
                print(f"[weights] completed {completed}/{N_runs} runs")

    complete_results = [result for result in results if result is not None]
    if len(complete_results) != N_runs:
        raise RuntimeError("At least one Monte Carlo weight-sweep run did not finish.")

    return {
        "N_runs": int(N_runs),
        "mc_seed": int(mc_seed),
        "T": int(T),
        "attack_t": int(attack_t),
        "coverage": float(coverage),
        "lambda_scale": float(lambda_scale),
        "weight_values": weight_values,
        "run_seeds": run_seeds,
        "true_g": np.asarray([result["true_g"] for result in complete_results], dtype=float),
        "clean_probability": np.asarray(
            [result["clean_probability"] for result in complete_results], dtype=float
        ),
        "attack_probability": np.asarray(
            [result["attack_probability"] for result in complete_results], dtype=float
        ),
        "lambda_max": np.asarray([result["lambda_max"] for result in complete_results], dtype=float),
        "omega_probabilities": np.stack(
            [np.asarray(result["omega_probabilities"], dtype=float) for result in complete_results], axis=1
        ),
        "omega_gammas": np.stack(
            [np.asarray(result["omega_gammas"], dtype=float) for result in complete_results], axis=1
        ),
        "evidence_probabilities": np.stack(
            [np.asarray(result["evidence_probabilities"], dtype=float) for result in complete_results], axis=1
        ),
        "evidence_gammas": np.stack(
            [np.asarray(result["evidence_gammas"], dtype=float) for result in complete_results], axis=1
        ),
    }


def _plot_gamma_distributions(
    ax: plt.Axes,
    *,
    weight_values: np.ndarray,
    gamma_values: np.ndarray,
    selected_weights: np.ndarray,
    pair_kind: str,
) -> None:
    """Draw three overlaid gamma histograms with transparent pastel fills."""
    pastel_colors = ["#8EC9D6", "#B8A7D8", "#F0A6A6"]
    bins = np.linspace(0.0, 1.0, 17)
    for color, selected_weight in zip(pastel_colors, selected_weights, strict=True):
        matching = np.flatnonzero(np.isclose(weight_values, selected_weight, atol=1e-12))
        if matching.size != 1:
            raise ValueError(f"Selected weight {selected_weight:g} is missing from weight_values.")
        values = np.asarray(gamma_values[matching[0]], dtype=float)
        values = values[np.isfinite(values)]
        if pair_kind == "omega":
            label = fr"$(\omega_h,\omega_o)=({selected_weight:.2f},{1.0-selected_weight:.2f})$"
            legend_title = "Prior weights"
        else:
            label = fr"$(w_M,w_g)=({selected_weight:.2f},{1.0-selected_weight:.2f})$"
            legend_title = "Evidence weights"
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=color,
            edgecolor=color,
            linewidth=1.5,
            alpha=0.34,
            label=label,
        )

    ax.axvline(
        base.DEFAULT_POSTERIOR_ATTACK_THRESHOLD,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
        label=fr"Threshold $\delta={base.DEFAULT_POSTERIOR_ATTACK_THRESHOLD:.2f}$",
    )
    ax.set_xlabel(r"Posterior attack probability $\gamma_T$")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.96, title=legend_title)


def _plot_probability_weight_sweep(
    ax: plt.Axes,
    *,
    true_g: np.ndarray,
    attack_probability: np.ndarray,
    defended_probabilities: np.ndarray,
    weight_values: np.ndarray,
    pair_kind: str,
    call_threshold: float,
    fig: plt.Figure,
) -> None:
    """Draw the E[g] panel in the style of the reference lambda-sweep plot."""
    true_g = np.asarray(true_g, dtype=float)
    attack_probability = np.asarray(attack_probability, dtype=float)
    defended_probabilities = np.asarray(defended_probabilities, dtype=float)
    weight_values = np.asarray(weight_values, dtype=float)
    n_bins = max(5, min(10, true_g.size // 3))

    diagonal = np.linspace(0.0, 1.0, 200)
    ax.plot(
        diagonal,
        diagonal,
        color="#555555",
        linestyle="--",
        linewidth=1.8,
        label="Non-attacked mean",
        zorder=3,
    )

    attack_x, attack_mean, attack_low, attack_high = base.binned_probability_summary(
        true_g,
        attack_probability,
        n_bins=n_bins,
    )
    attack_mean = np.clip(base.smooth_series(attack_mean, window=3), 0.0, 1.0)
    attack_low = np.clip(base.smooth_series(attack_low, window=3), 0.0, 1.0)
    attack_high = np.clip(base.smooth_series(attack_high, window=3), 0.0, 1.0)
    ax.fill_between(attack_x, attack_low, attack_high, color="#E3A0A0", alpha=0.15, zorder=1)
    ax.plot(
        attack_x,
        attack_mean,
        color="#D88383",
        linewidth=1.9,
        label="Attacked mean",
        zorder=3,
    )

    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=float(np.min(weight_values)), vmax=float(np.max(weight_values)))
    for idx, weight in enumerate(weight_values):
        curve_x, curve_mean, curve_low, curve_high = base.binned_probability_summary(
            true_g,
            defended_probabilities[idx],
            n_bins=n_bins,
        )
        curve_mean = np.clip(base.smooth_series(curve_mean, window=3), 0.0, 1.0)
        curve_low = np.clip(base.smooth_series(curve_low, window=3), 0.0, 1.0)
        curve_high = np.clip(base.smooth_series(curve_high, window=3), 0.0, 1.0)
        color = cmap(norm(float(weight)))
        ax.fill_between(curve_x, curve_low, curve_high, color=color, alpha=0.08, zorder=1)
        ax.plot(
            curve_x,
            curve_mean,
            color=color,
            linewidth=1.65,
            alpha=0.76,
            label="Defended mean" if idx == 0 else None,
            zorder=3,
        )

    ax.axhline(
        call_threshold,
        color="#8A8A8A",
        linewidth=1.1,
        linestyle=":",
        label="Call threshold",
        zorder=2,
    )
    ax.set_xlabel(r"Actual $g(s_T)$")
    ax.set_ylabel(r"Estimated $\mathbb{E}[g(s_T)\mid o_{0:T}]$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    legend_title = "Prior-weight sweep" if pair_kind == "omega" else "Evidence-weight sweep"
    ax.legend(loc="lower right", frameon=True, framealpha=0.96, title=legend_title)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.015)
    colorbar.set_ticks(weight_values)
    colorbar.ax.tick_params(labelsize=8.5)
    if pair_kind == "omega":
        colorbar.set_label(r"$\omega_h$ in $(\omega_h,\omega_o)=(q,1-q)$")
    else:
        colorbar.set_label(r"$w_M$ in $(w_M,w_g)=(q,1-q)$")


def plot_weight_sweep_summary(
    *,
    data: dict[str, np.ndarray | float | int],
    selected_weights: np.ndarray,
    outpath: str,
) -> None:
    """Create the requested 2x2 gamma-distribution and E[g] comparison."""
    base.set_plot_theme()
    weight_values = np.asarray(data["weight_values"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.0), constrained_layout=True)
    for ax in axes.flat:
        base.style_axis(ax)

    _plot_gamma_distributions(
        axes[0, 0],
        weight_values=weight_values,
        gamma_values=np.asarray(data["omega_gammas"], dtype=float),
        selected_weights=selected_weights,
        pair_kind="omega",
    )
    _plot_probability_weight_sweep(
        axes[0, 1],
        true_g=np.asarray(data["true_g"], dtype=float),
        attack_probability=np.asarray(data["attack_probability"], dtype=float),
        defended_probabilities=np.asarray(data["omega_probabilities"], dtype=float),
        weight_values=weight_values,
        pair_kind="omega",
        call_threshold=base.DEFAULT_CALL_THRESHOLD,
        fig=fig,
    )
    _plot_gamma_distributions(
        axes[1, 0],
        weight_values=weight_values,
        gamma_values=np.asarray(data["evidence_gammas"], dtype=float),
        selected_weights=selected_weights,
        pair_kind="evidence",
    )
    _plot_probability_weight_sweep(
        axes[1, 1],
        true_g=np.asarray(data["true_g"], dtype=float),
        attack_probability=np.asarray(data["attack_probability"], dtype=float),
        defended_probabilities=np.asarray(data["evidence_probabilities"], dtype=float),
        weight_values=weight_values,
        pair_kind="evidence",
        call_threshold=base.DEFAULT_CALL_THRESHOLD,
        fig=fig,
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    """
    Run the fixed-lambda weight experiment with production MC settings.
    """
    T = 5
    attack_t = T
    seed = 2025
    coverage = 0.95
    lambda_scale = 1.0
    raw_weight_values = "0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1"
    selected_distribution_weights = np.array([0.25, 0.50, 0.75], dtype=float)

    # Production Monte Carlo configuration aligned with the reference script.
    N_runs = 1000
    mc_seed = seed
    eta = 1.5
    n_steps = 700
    n_mc_opt = 96
    n_mc_est = 1200
    # Limit local parallelism to avoid excessive memory use from MC arrays.
    n_jobs = min(8, max(1, (os.cpu_count() or 1) - 1))
    force_mc = False

    if not (0 <= attack_t <= T):
        raise ValueError("attack_t must satisfy 0 <= attack_t <= T.")
    weight_values = parse_weight_values(raw_weight_values)
    for selected_weight in selected_distribution_weights:
        if not np.any(np.isclose(weight_values, selected_weight, atol=1e-12)):
            raise ValueError(f"Selected weight {selected_weight:g} must occur in raw_weight_values.")

    out_dir = figures_dir_for(CURRENT_DIR)
    outpath = os.path.join(
        out_dir,
        (
            "comparison_g_fixed_lambda_weight_sweeps_"
            f"cov{int(round(100 * coverage))}_t{attack_t}_T{T}_seed{seed}_N{N_runs}.png"
        ),
    )
    data_path = data_path_for_plot(outpath)

    data = cached_npz(
        data_path,
        lambda: run_monte_carlo_weight_sweeps(
            N_runs=N_runs,
            mc_seed=mc_seed,
            T=T,
            attack_t=attack_t,
            coverage=coverage,
            lambda_scale=lambda_scale,
            weight_values=weight_values,
            eta=eta,
            n_steps=n_steps,
            n_mc_opt=n_mc_opt,
            n_mc_est=n_mc_est,
            n_jobs=n_jobs,
        ),
        force=force_mc,
    )
    plot_weight_sweep_summary(
        data=data,
        selected_weights=selected_distribution_weights,
        outpath=outpath,
    )
    print(f"Saved figure to: {outpath}")
    print(f"Saved data to: {data_path}")


if __name__ == "__main__":
    main()
