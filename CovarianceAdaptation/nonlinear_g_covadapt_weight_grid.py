#!/usr/bin/env python3
"""
Joint weight-grid experiment for the nonlinear covariance-adaptation defense.

The four defense weights form two complementary pairs, so only two free
parameters remain:

    (omega_h, omega_o) = (q_h, 1 - q_h)
    (w_M, w_g)         = (q_M, 1 - q_M)

The reusable functions in this script can evaluate every pair ``(q_h, q_M)``
on a 5x5 grid while fixing ``lambda = lambda_max``.  The figure produced by
``main`` focuses on three vertical metric stacks.  The left stack varies the
evidence weights on non-attacked (clean) observations; the middle and right
stacks show attacked observations while varying the prior and evidence pairs,
respectively.  Each stack separates false-positive rate, false-negative rate,
and local hidden-state effect so that all metrics have their own y scale.  The
non-varied pair is fixed at ``(0.5, 0.5)`` in every stack.

False-negative rates use a dedicated larger Monte Carlo sample.  The existing
rate sample is reused as its first block and only the additional independent
runs are computed.  False-positive rates and local effects keep their smaller
sample, avoiding unnecessary repetition of those experiments.

The false-positive statistic matches ``nonlinear_g_covadapt.py``: false calls
are divided by the total number of predicted calls.  The false-negative rate
is divided by the number of true calls.  All experiment values are defined in
``main`` rather than environment variables so they can be edited directly.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import nonlinear_g_covadapt as base
import nonlinear_g_covadapt_weight_sweep as weight_sweep

from shared_ssm.artifacts import cached_npz
from shared_ssm.artifacts import data_path_for_plot
from shared_ssm.artifacts import figures_dir_for


def evaluate_grid_for_run(
    *,
    run_seed: int,
    T: int,
    attack_t: int,
    coverage: float,
    lambda_scale: float,
    omega_h_values: np.ndarray,
    mahalanobis_weight_values: np.ndarray,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_mc_est: int,
) -> dict[str, np.ndarray | float]:
    """Evaluate all attacked cells in the complementary two-weight grid."""
    pars = base.get_system_parameters()
    epsilon = base.coverage_to_epsilon(coverage)
    omega_h_values = np.asarray(omega_h_values, dtype=float)
    mahalanobis_weight_values = np.asarray(mahalanobis_weight_values, dtype=float)

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

    # Compute the two no-defense references once per Monte Carlo realization.
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

    grid_probabilities = np.zeros(
        (omega_h_values.size, mahalanobis_weight_values.size),
        dtype=float,
    )
    grid_gammas = np.zeros_like(grid_probabilities)
    for omega_idx, omega_h in enumerate(omega_h_values):
        for evidence_idx, w_m in enumerate(mahalanobis_weight_values):
            grid_probabilities[omega_idx, evidence_idx], grid_gammas[omega_idx, evidence_idx] = (
                weight_sweep._defended_probability_and_gamma(
                    y_adv=y_adv,
                    u_controls=u_controls,
                    mats=mats,
                    m0=pars["m0"],
                    P0=pars["P0"],
                    attack_t=attack_t,
                    attack_target=attack_target,
                    lam=lam,
                    omega_h=float(omega_h),
                    omega_o=float(1.0 - omega_h),
                    mahalanobis_weight=float(w_m),
                    objective_weight=float(1.0 - w_m),
                    epsilon=epsilon,
                    objective_attack_score_builder=objective_builder,
                    n_mc_est=n_mc_est,
                )
            )

    return {
        "true_g": float(base.g_scalar(x_true[attack_t])),
        "clean_probability": float(clean_probability),
        "attack_probability": float(attack_probability),
        "lambda_max": float(lambda_max),
        "grid_probabilities": grid_probabilities,
        "grid_gammas": grid_gammas,
    }


def evaluate_grid_from_task(task: dict[str, object]) -> dict[str, np.ndarray | float]:
    """Unpack one plain task payload for a picklable worker entry point."""
    return evaluate_grid_for_run(
        run_seed=int(task["run_seed"]),
        T=int(task["T"]),
        attack_t=int(task["attack_t"]),
        coverage=float(task["coverage"]),
        lambda_scale=float(task["lambda_scale"]),
        omega_h_values=np.asarray(task["omega_h_values"], dtype=float),
        mahalanobis_weight_values=np.asarray(task["mahalanobis_weight_values"], dtype=float),
        eta=float(task["eta"]),
        n_steps=int(task["n_steps"]),
        n_mc_opt=int(task["n_mc_opt"]),
        n_mc_est=int(task["n_mc_est"]),
    )


def run_monte_carlo_grid(
    *,
    N_runs: int,
    mc_seed: int,
    T: int,
    attack_t: int,
    coverage: float,
    lambda_scale: float,
    omega_h_values: np.ndarray,
    mahalanobis_weight_values: np.ndarray,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_mc_est: int,
    n_jobs: int,
) -> dict[str, np.ndarray | float | int]:
    """Run the joint grid with reproducible seeds and optional multiprocessing."""
    omega_h_values = np.asarray(omega_h_values, dtype=float)
    mahalanobis_weight_values = np.asarray(mahalanobis_weight_values, dtype=float)
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
            "omega_h_values": omega_h_values,
            "mahalanobis_weight_values": mahalanobis_weight_values,
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
            results[idx] = evaluate_grid_from_task(task)
            print(f"[weight-grid] completed {idx + 1}/{N_runs} runs", flush=True)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            future_to_idx = {
                executor.submit(evaluate_grid_from_task, task): idx
                for idx, task in enumerate(tasks)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                result_idx = future_to_idx[future]
                results[result_idx] = future.result()
                completed += 1
                print(f"[weight-grid] completed {completed}/{N_runs} runs", flush=True)

    complete_results = [result for result in results if result is not None]
    if len(complete_results) != N_runs:
        raise RuntimeError("At least one Monte Carlo grid run did not finish.")

    # Store MC runs along the final axis, matching the convention used by the
    # earlier sweep caches: (parameter axes..., run axis).
    return {
        "N_runs": int(N_runs),
        "mc_seed": int(mc_seed),
        "T": int(T),
        "attack_t": int(attack_t),
        "coverage": float(coverage),
        "lambda_scale": float(lambda_scale),
        "omega_h_values": omega_h_values,
        "mahalanobis_weight_values": mahalanobis_weight_values,
        "run_seeds": run_seeds,
        "true_g": np.asarray([result["true_g"] for result in complete_results], dtype=float),
        "clean_probability": np.asarray(
            [result["clean_probability"] for result in complete_results], dtype=float
        ),
        "attack_probability": np.asarray(
            [result["attack_probability"] for result in complete_results], dtype=float
        ),
        "lambda_max": np.asarray([result["lambda_max"] for result in complete_results], dtype=float),
        "grid_probabilities": np.stack(
            [np.asarray(result["grid_probabilities"], dtype=float) for result in complete_results],
            axis=2,
        ),
        "grid_gammas": np.stack(
            [np.asarray(result["grid_gammas"], dtype=float) for result in complete_results],
            axis=2,
        ),
    }


def _defended_local_effect(
    *,
    y_adv: np.ndarray,
    u_controls: np.ndarray,
    mats: dict[str, np.ndarray],
    m0: np.ndarray,
    P0: np.ndarray,
    x_true: np.ndarray,
    attack_t: int,
    attack_target: np.ndarray,
    lam: float,
    omega_h: float,
    omega_o: float,
    mahalanobis_weight: float,
    objective_weight: float,
    epsilon: float,
    objective_attack_score_builder,
) -> float:
    """Return the attacked local state effect without estimating E[g]."""
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
    defended_smooth, _ = base.rts_smoother_nd(
        m_filt=defended_filter[0],
        P_filt=defended_filter[1],
        m_pred=defended_filter[2],
        P_pred=defended_filter[3],
        A_t=mats["A_t"],
    )
    return base.local_hidden_state_error(x_true, defended_smooth, attack_t=attack_t)


def evaluate_clean_evidence_sweep_for_run(
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
    """Evaluate clean probabilities/local effects while varying ``(w_M,w_g)``."""
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

    # Match the clean branch in nonlinear_g_covadapt.py: the adversarial
    # optimization defines the objective target, but the filter below always
    # receives the untouched clean observations.
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
    attack_target = np.asarray(attack_data["adv_target"], dtype=float)
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
    clean_smooth, _, clean_probability = base.compute_smoothed_probability(
        m_filt=clean_filter[0],
        P_filt=clean_filter[1],
        m_pred=clean_filter[2],
        P_pred=clean_filter[3],
        A_t=mats["A_t"],
        attack_t=attack_t,
        n_mc_est=n_mc_est,
    )
    clean_local_effect = base.local_hidden_state_error(
        x_true,
        clean_smooth,
        attack_t=attack_t,
    )
    objective_builder = base.build_nonlinear_objective_attack_score_builder(
        attack_t=attack_t,
        clean_state_mean=clean_filter[0][attack_t],
    )
    lam = float(lambda_scale) * base.lambda_max_from_observation_covariance(
        mats["R_t"][attack_t]
    )

    clean_probabilities = np.zeros(weight_values.size, dtype=float)
    clean_local_effects = np.zeros(weight_values.size, dtype=float)
    for idx, w_m in enumerate(weight_values):
        defended_filter = base.kalman_filter_with_online_covariance_adaptation_3d(
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
            attack_targets={attack_t: attack_target},
            lam=lam,
            omega_h=0.5,
            omega_o=0.5,
            delta_threshold=base.DEFAULT_POSTERIOR_ATTACK_THRESHOLD,
            objective_attack_score_builder=objective_builder,
            mahalanobis_epsilon=float(epsilon),
            mahalanobis_evidence_weight=float(w_m),
            objective_evidence_weight=float(1.0 - w_m),
        )
        defended_smooth, _, clean_probabilities[idx] = base.compute_smoothed_probability(
            m_filt=defended_filter[0],
            P_filt=defended_filter[1],
            m_pred=defended_filter[2],
            P_pred=defended_filter[3],
            A_t=mats["A_t"],
            attack_t=attack_t,
            n_mc_est=n_mc_est,
        )
        clean_local_effects[idx] = base.local_hidden_state_error(
            x_true,
            defended_smooth,
            attack_t=attack_t,
        )

    return {
        "true_g": float(base.g_scalar(x_true[attack_t])),
        "clean_probability": float(clean_probability),
        "clean_local_effect": float(clean_local_effect),
        "clean_probabilities": clean_probabilities,
        "clean_local_effects": clean_local_effects,
    }


def evaluate_clean_evidence_sweep_from_task(
    task: dict[str, object],
) -> dict[str, np.ndarray | float]:
    """Unpack one clean evidence-weight task for Windows multiprocessing."""
    return evaluate_clean_evidence_sweep_for_run(
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


def run_monte_carlo_clean_evidence_sweep(
    *,
    run_seeds: np.ndarray,
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
    """Compute the clean evidence-weight sweep for a fixed seed collection."""
    run_seeds = np.asarray(run_seeds, dtype=np.uint32)
    weight_values = np.asarray(weight_values, dtype=float)
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
    results: list[dict[str, np.ndarray | float] | None] = [None] * run_seeds.size
    max_workers = max(1, min(int(n_jobs), run_seeds.size))
    progress_interval = max(1, run_seeds.size // 100)
    if max_workers == 1:
        for idx, task in enumerate(tasks):
            results[idx] = evaluate_clean_evidence_sweep_from_task(task)
            if (idx + 1) % progress_interval == 0 or idx + 1 == run_seeds.size:
                print(f"[clean evidence] completed {idx + 1}/{run_seeds.size} runs", flush=True)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            future_to_idx = {
                executor.submit(evaluate_clean_evidence_sweep_from_task, task): idx
                for idx, task in enumerate(tasks)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                result_idx = future_to_idx[future]
                results[result_idx] = future.result()
                completed += 1
                if completed % progress_interval == 0 or completed == run_seeds.size:
                    print(
                        f"[clean evidence] completed {completed}/{run_seeds.size} runs",
                        flush=True,
                    )

    complete_results = [result for result in results if result is not None]
    if len(complete_results) != run_seeds.size:
        raise RuntimeError("At least one clean evidence-weight run did not finish.")
    return {
        "N_runs": int(run_seeds.size),
        "run_seeds": run_seeds,
        "weight_values": weight_values,
        "lambda_scale": float(lambda_scale),
        "true_g": np.asarray([result["true_g"] for result in complete_results], dtype=float),
        "clean_probability": np.asarray(
            [result["clean_probability"] for result in complete_results], dtype=float
        ),
        "clean_local_effect": np.asarray(
            [result["clean_local_effect"] for result in complete_results], dtype=float
        ),
        "clean_probabilities": np.stack(
            [np.asarray(result["clean_probabilities"], dtype=float) for result in complete_results],
            axis=1,
        ),
        "clean_local_effects": np.stack(
            [np.asarray(result["clean_local_effects"], dtype=float) for result in complete_results],
            axis=1,
        ),
    }


def evaluate_local_effect_sweeps_for_run(
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
) -> dict[str, np.ndarray | float]:
    """Evaluate attacked local effects for both 0.5-fixed weight slices."""
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

    # The nominal clean filter supplies the reference point used by the
    # nonlinear objective-evidence builder; no E[g] integration is needed.
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
    attack_smooth, _ = base.rts_smoother_nd(
        m_filt=attack_filter[0],
        P_filt=attack_filter[1],
        m_pred=attack_filter[2],
        P_pred=attack_filter[3],
        A_t=mats["A_t"],
    )
    attack_local_effect = base.local_hidden_state_error(
        x_true,
        attack_smooth,
        attack_t=attack_t,
    )
    objective_builder = base.build_nonlinear_objective_attack_score_builder(
        attack_t=attack_t,
        clean_state_mean=clean_filter[0][attack_t],
    )
    lam = float(lambda_scale) * base.lambda_max_from_observation_covariance(mats["R_t"][attack_t])

    omega_local_effects = np.zeros(weight_values.size, dtype=float)
    evidence_local_effects = np.zeros(weight_values.size, dtype=float)
    defense_cache: dict[tuple[float, float, float, float], float] = {}

    def run_defense(omega_h: float, omega_o: float, w_m: float, w_g: float) -> float:
        key = tuple(round(value, 12) for value in (omega_h, omega_o, w_m, w_g))
        if key not in defense_cache:
            defense_cache[key] = _defended_local_effect(
                y_adv=y_adv,
                u_controls=u_controls,
                mats=mats,
                m0=pars["m0"],
                P0=pars["P0"],
                x_true=x_true,
                attack_t=attack_t,
                attack_target=attack_target,
                lam=lam,
                omega_h=omega_h,
                omega_o=omega_o,
                mahalanobis_weight=w_m,
                objective_weight=w_g,
                epsilon=epsilon,
                objective_attack_score_builder=objective_builder,
            )
        return defense_cache[key]

    for idx, q_value in enumerate(weight_values):
        omega_local_effects[idx] = run_defense(float(q_value), float(1.0 - q_value), 0.5, 0.5)
        evidence_local_effects[idx] = run_defense(0.5, 0.5, float(q_value), float(1.0 - q_value))

    return {
        "attack_local_effect": float(attack_local_effect),
        "omega_local_effects": omega_local_effects,
        "evidence_local_effects": evidence_local_effects,
    }


def evaluate_local_effect_sweeps_from_task(task: dict[str, object]) -> dict[str, np.ndarray | float]:
    """Unpack one local-effect worker task for Windows multiprocessing."""
    return evaluate_local_effect_sweeps_for_run(
        run_seed=int(task["run_seed"]),
        T=int(task["T"]),
        attack_t=int(task["attack_t"]),
        coverage=float(task["coverage"]),
        lambda_scale=float(task["lambda_scale"]),
        weight_values=np.asarray(task["weight_values"], dtype=float),
        eta=float(task["eta"]),
        n_steps=int(task["n_steps"]),
        n_mc_opt=int(task["n_mc_opt"]),
    )


def run_monte_carlo_local_effect_sweeps(
    *,
    run_seeds: np.ndarray,
    T: int,
    attack_t: int,
    coverage: float,
    lambda_scale: float,
    weight_values: np.ndarray,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_jobs: int,
) -> dict[str, np.ndarray | float | int]:
    """Compute only local effects for the exact seeds used by the rate cache."""
    run_seeds = np.asarray(run_seeds, dtype=np.uint32)
    weight_values = np.asarray(weight_values, dtype=float)
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
        }
        for run_seed in run_seeds
    ]
    results: list[dict[str, np.ndarray | float] | None] = [None] * run_seeds.size
    max_workers = max(1, min(int(n_jobs), run_seeds.size))
    if max_workers == 1:
        for idx, task in enumerate(tasks):
            results[idx] = evaluate_local_effect_sweeps_from_task(task)
            print(f"[local-effect] completed {idx + 1}/{run_seeds.size} runs", flush=True)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            future_to_idx = {
                executor.submit(evaluate_local_effect_sweeps_from_task, task): idx
                for idx, task in enumerate(tasks)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                result_idx = future_to_idx[future]
                results[result_idx] = future.result()
                completed += 1
                print(f"[local-effect] completed {completed}/{run_seeds.size} runs", flush=True)

    complete_results = [result for result in results if result is not None]
    if len(complete_results) != run_seeds.size:
        raise RuntimeError("At least one local-effect run did not finish.")
    return {
        "N_runs": int(run_seeds.size),
        "run_seeds": run_seeds,
        "weight_values": weight_values,
        "lambda_scale": float(lambda_scale),
        "attack_local_effect": np.asarray(
            [result["attack_local_effect"] for result in complete_results], dtype=float
        ),
        "omega_local_effects": np.stack(
            [np.asarray(result["omega_local_effects"], dtype=float) for result in complete_results], axis=1
        ),
        "evidence_local_effects": np.stack(
            [np.asarray(result["evidence_local_effects"], dtype=float) for result in complete_results], axis=1
        ),
    }


def evaluate_false_negative_candidate_for_run(
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
    call_threshold: float,
) -> dict[str, np.ndarray | float | bool]:
    """Evaluate defenses only when one realization is a true positive case."""
    pars = base.get_system_parameters()
    x_true, _, _, _ = base.simulate_lgssm_nd(
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
    true_g = float(base.g_scalar(x_true[attack_t]))
    if true_g <= float(call_threshold):
        return {"is_true_call": False, "true_g": true_g}

    # The full attacked/defended calculation is needed only for true calls,
    # because all other realizations have zero contribution to the FN rate.
    result = weight_sweep.evaluate_weight_sweeps_for_run(
        run_seed=run_seed,
        T=T,
        attack_t=attack_t,
        coverage=coverage,
        lambda_scale=lambda_scale,
        weight_values=np.asarray(weight_values, dtype=float),
        eta=eta,
        n_steps=n_steps,
        n_mc_opt=n_mc_opt,
        n_mc_est=n_mc_est,
    )
    return {
        "is_true_call": True,
        "true_g": float(result["true_g"]),
        "attack_probability": float(result["attack_probability"]),
        "omega_probabilities": np.asarray(result["omega_probabilities"], dtype=float),
        "evidence_probabilities": np.asarray(
            result["evidence_probabilities"], dtype=float
        ),
    }


def evaluate_false_negative_candidate_from_task(
    task: dict[str, object],
) -> dict[str, np.ndarray | float | bool]:
    """Unpack one candidate task for Windows multiprocessing."""
    return evaluate_false_negative_candidate_for_run(
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
        call_threshold=float(task["call_threshold"]),
    )


def run_false_negative_candidates(
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
    call_threshold: float,
    n_jobs: int,
) -> dict[str, np.ndarray | int]:
    """Run FN candidates, skipping expensive defenses for non-true calls."""
    weight_values = np.asarray(weight_values, dtype=float)
    seed_sequence = np.random.SeedSequence(mc_seed)
    candidate_run_seeds = np.asarray(
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
            "call_threshold": call_threshold,
        }
        for run_seed in candidate_run_seeds
    ]
    results: list[dict[str, np.ndarray | float | bool] | None] = [None] * N_runs
    max_workers = max(1, min(int(n_jobs), N_runs))
    progress_interval = max(1, N_runs // 100)
    if max_workers == 1:
        for idx, task in enumerate(tasks):
            results[idx] = evaluate_false_negative_candidate_from_task(task)
            if (idx + 1) % progress_interval == 0 or idx + 1 == N_runs:
                print(f"[false-negative] completed {idx + 1}/{N_runs} candidates", flush=True)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            future_to_idx = {
                executor.submit(evaluate_false_negative_candidate_from_task, task): idx
                for idx, task in enumerate(tasks)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                result_idx = future_to_idx[future]
                results[result_idx] = future.result()
                completed += 1
                if completed % progress_interval == 0 or completed == N_runs:
                    print(
                        f"[false-negative] completed {completed}/{N_runs} candidates",
                        flush=True,
                    )

    if any(result is None for result in results):
        raise RuntimeError("At least one false-negative candidate did not finish.")
    true_indices = np.asarray(
        [idx for idx, result in enumerate(results) if bool(result["is_true_call"])],
        dtype=int,
    )
    true_results = [results[idx] for idx in true_indices]
    if not true_results:
        raise RuntimeError("The additional FN sample contains no true calls.")
    return {
        "N_runs": int(N_runs),
        "candidate_run_seeds": candidate_run_seeds,
        "true_run_seeds": candidate_run_seeds[true_indices],
        "true_g": np.asarray([result["true_g"] for result in true_results], dtype=float),
        "attack_probability": np.asarray(
            [result["attack_probability"] for result in true_results], dtype=float
        ),
        "omega_probabilities": np.stack(
            [np.asarray(result["omega_probabilities"], dtype=float) for result in true_results],
            axis=1,
        ),
        "evidence_probabilities": np.stack(
            [
                np.asarray(result["evidence_probabilities"], dtype=float)
                for result in true_results
            ],
            axis=1,
        ),
    }


def extend_rate_data_for_false_negative(
    *,
    base_rate_data: dict[str, np.ndarray | float | int],
    target_N_runs: int,
    extra_mc_seed: int,
    T: int,
    attack_t: int,
    coverage: float,
    lambda_scale: float,
    weight_values: np.ndarray,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_mc_est: int,
    call_threshold: float,
    n_jobs: int,
) -> dict[str, np.ndarray | float | int]:
    """Extend the existing rate cache for a higher-resolution FN estimate."""
    base_count = int(np.asarray(base_rate_data["true_g"]).size)
    if target_N_runs < base_count:
        raise ValueError("target_N_runs cannot be smaller than the existing rate sample.")
    if not np.allclose(
        np.asarray(base_rate_data["weight_values"], dtype=float),
        np.asarray(weight_values, dtype=float),
        atol=1e-12,
    ):
        raise ValueError("The existing and extended rate samples must use the same weights.")

    additional_count = int(target_N_runs - base_count)
    if additional_count == 0:
        return dict(base_rate_data)

    # A separate seed creates an independent block while allowing the already
    # computed base block to remain untouched.  The completed cache records
    # both blocks' seeds, making the combined FN estimate reproducible.
    extra_data = run_false_negative_candidates(
        N_runs=additional_count,
        mc_seed=extra_mc_seed,
        T=T,
        attack_t=attack_t,
        coverage=coverage,
        lambda_scale=lambda_scale,
        weight_values=np.asarray(weight_values, dtype=float),
        eta=eta,
        n_steps=n_steps,
        n_mc_opt=n_mc_opt,
        n_mc_est=n_mc_est,
        call_threshold=call_threshold,
        n_jobs=n_jobs,
    )
    base_seeds = np.asarray(base_rate_data["run_seeds"], dtype=np.uint32)
    extra_seeds = np.asarray(extra_data["candidate_run_seeds"], dtype=np.uint32)
    if np.intersect1d(base_seeds, extra_seeds).size:
        raise RuntimeError("The base and additional Monte Carlo seed blocks overlap.")

    base_true_mask = np.asarray(base_rate_data["true_g"], dtype=float) > float(call_threshold)
    base_true_seeds = base_seeds[base_true_mask]
    extra_true_seeds = np.asarray(extra_data["true_run_seeds"], dtype=np.uint32)

    combined: dict[str, np.ndarray | float | int] = {
        "N_runs": int(target_N_runs),
        "base_N_runs": int(base_count),
        "extra_mc_seed": int(extra_mc_seed),
        "T": int(T),
        "attack_t": int(attack_t),
        "coverage": float(coverage),
        "lambda_scale": float(lambda_scale),
        "weight_values": np.asarray(weight_values, dtype=float),
        "run_seeds": np.concatenate([base_seeds, extra_seeds]),
        "true_run_seeds": np.concatenate([base_true_seeds, extra_true_seeds]),
    }
    for key in ("true_g", "attack_probability"):
        combined[key] = np.concatenate(
            [
                np.asarray(base_rate_data[key], dtype=float)[base_true_mask],
                np.asarray(extra_data[key], dtype=float),
            ]
        )
    for key in ("omega_probabilities", "evidence_probabilities"):
        combined[key] = np.concatenate(
            [
                np.asarray(base_rate_data[key], dtype=float)[:, base_true_mask],
                np.asarray(extra_data[key], dtype=float),
            ],
            axis=1,
        )
    combined["N_true_calls"] = int(np.asarray(combined["true_g"]).size)
    return combined


def false_positive_rate_percent(true_calls: np.ndarray, predicted_calls: np.ndarray) -> float:
    """Return false calls divided by total predicted calls, as in the reference."""
    true_calls = np.asarray(true_calls, dtype=bool)
    predicted_calls = np.asarray(predicted_calls, dtype=bool)
    predicted_count = int(np.sum(predicted_calls))
    if predicted_count == 0:
        return 0.0
    false_call_count = int(np.sum(predicted_calls & (~true_calls)))
    return 100.0 * float(false_call_count) / float(predicted_count)


def false_negative_rate_percent(true_calls: np.ndarray, predicted_calls: np.ndarray) -> float:
    """Return missed true calls divided by the total number of true calls."""
    true_calls = np.asarray(true_calls, dtype=bool)
    predicted_calls = np.asarray(predicted_calls, dtype=bool)
    true_count = int(np.sum(true_calls))
    if true_count == 0:
        return 0.0
    missed_count = int(np.sum((~predicted_calls) & true_calls))
    return 100.0 * float(missed_count) / float(true_count)


def _rate_series(
    *,
    true_calls: np.ndarray,
    defended_probabilities: np.ndarray,
    call_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return false-positive and false-negative curves for one data branch."""
    defended_calls = np.asarray(defended_probabilities, dtype=float) > float(call_threshold)
    false_positive = np.asarray(
        [false_positive_rate_percent(true_calls, calls) for calls in defended_calls],
        dtype=float,
    )
    false_negative = np.asarray(
        [false_negative_rate_percent(true_calls, calls) for calls in defended_calls],
        dtype=float,
    )
    return false_positive, false_negative


def _set_metric_axis_limits(
    ax: plt.Axes,
    values: np.ndarray,
    reference: float,
) -> None:
    """Use a metric-specific y range while retaining the reference line."""
    all_values = np.concatenate(
        [np.asarray(values, dtype=float), np.array([reference], dtype=float)]
    )
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    value_span = value_max - value_min
    display_scale = max(abs(value_min), abs(value_max), 1e-2)
    padding = max(0.12 * value_span, 0.025 * display_scale, 1e-4)
    ax.set_ylim(max(0.0, value_min - padding), value_max + padding)


def _plot_metric_stack(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    *,
    weight_values: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    false_positive_reference: float,
    false_negative_reference: float,
    local_effect: np.ndarray,
    local_effect_reference: float,
    x_label: str,
    legend_title: str,
    curve_labels: tuple[str, str, str],
    reference_label: str,
) -> None:
    """Draw FP, FN, and local effect on three vertically stacked axes."""
    fp_color = "#D98F8F"
    fn_color = "#789BC8"
    local_color = "#8D72B8"
    fp_axis, fn_axis, local_axis = axes

    fp_axis.plot(
        weight_values,
        false_positive,
        color=fp_color,
        marker="o",
        linewidth=1.9,
        label=curve_labels[0],
    )
    fp_axis.axhline(
        false_positive_reference,
        color=fp_color,
        linestyle="--",
        linewidth=1.35,
        label=reference_label,
    )
    fp_axis.set_ylabel("False positive (%)")
    _set_metric_axis_limits(fp_axis, false_positive, false_positive_reference)
    fp_axis.legend(
        loc="best",
        frameon=True,
        framealpha=0.96,
        title=legend_title,
        fontsize=8.5,
        title_fontsize=8.5,
    )

    fn_axis.plot(
        weight_values,
        false_negative,
        color=fn_color,
        marker="s",
        linewidth=1.9,
        label=curve_labels[1],
    )
    fn_axis.axhline(
        false_negative_reference,
        color=fn_color,
        linestyle="--",
        linewidth=1.35,
        label=reference_label,
    )
    fn_axis.set_ylabel("False negative (%)")
    _set_metric_axis_limits(fn_axis, false_negative, false_negative_reference)
    fn_axis.legend(loc="best", frameon=True, framealpha=0.96, fontsize=8.5)

    local_axis.plot(
        weight_values,
        local_effect,
        color=local_color,
        linestyle="-.",
        marker="D",
        markerfacecolor="white",
        linewidth=1.7,
        label=curve_labels[2],
    )
    local_axis.axhline(
        local_effect_reference,
        color=local_color,
        linestyle="--",
        linewidth=1.3,
        label=reference_label,
    )
    local_axis.set_ylabel(r"Local effect on $s_T$")
    _set_metric_axis_limits(local_axis, local_effect, local_effect_reference)
    local_axis.legend(loc="best", frameon=True, framealpha=0.96, fontsize=8.5)

    # The three panels in a column use the same weights.  Only the bottom one
    # shows tick labels so the compact stack remains easy to scan vertically.
    for axis in axes:
        axis.set_xlim(float(weight_values[0]) - 0.03, float(weight_values[-1]) + 0.03)
        axis.set_xticks(weight_values)
        axis.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fp_axis.tick_params(axis="x", labelbottom=False)
    fn_axis.tick_params(axis="x", labelbottom=False)
    local_axis.set_xlabel(x_label)


def plot_combined_weight_summary(
    *,
    clean_evidence_data: dict[str, np.ndarray | float | int],
    rate_data: dict[str, np.ndarray | float | int],
    false_negative_data: dict[str, np.ndarray | float | int],
    local_data: dict[str, np.ndarray | float | int],
    call_threshold: float,
    outpath: str,
) -> None:
    """Draw clean evidence weights and the two attacked weight-pair slices."""
    base.set_plot_theme()
    rate_weight_values = np.asarray(rate_data["weight_values"], dtype=float)
    clean_weight_values = np.asarray(clean_evidence_data["weight_values"], dtype=float)
    false_negative_weight_values = np.asarray(
        false_negative_data["weight_values"], dtype=float
    )
    true_calls = np.asarray(rate_data["true_g"], dtype=float) > float(call_threshold)
    attack_calls = np.asarray(rate_data["attack_probability"], dtype=float) > float(call_threshold)
    if not np.allclose(clean_weight_values, rate_weight_values, atol=1e-12):
        raise ValueError("The clean and attacked sweeps must use the same weights.")
    if not np.allclose(
        rate_weight_values,
        false_negative_weight_values,
        atol=1e-12,
    ):
        raise ValueError("The FP and FN samples must use the same weight values.")
    if not np.array_equal(
        np.asarray(rate_data["run_seeds"], dtype=np.uint32),
        np.asarray(local_data["run_seeds"], dtype=np.uint32),
    ):
        raise ValueError("Rate and local-effect caches must use the same Monte Carlo seeds.")
    if not np.allclose(
        rate_weight_values,
        np.asarray(local_data["weight_values"], dtype=float),
        atol=1e-12,
    ):
        raise ValueError("Rate and local-effect caches must use the same weight values.")
    if not np.array_equal(
        np.asarray(rate_data["run_seeds"], dtype=np.uint32),
        np.asarray(clean_evidence_data["run_seeds"], dtype=np.uint32),
    ):
        raise ValueError("The clean and attacked N=700 sweeps must use the same seeds.")

    # The clean left stack varies only the evidence pair.  Its solid curves
    # show covariance adaptation; dashed lines show the ordinary clean filter.
    clean_true_calls = (
        np.asarray(clean_evidence_data["true_g"], dtype=float) > float(call_threshold)
    )
    clean_reference_calls = (
        np.asarray(clean_evidence_data["clean_probability"], dtype=float)
        > float(call_threshold)
    )
    clean_fp, clean_fn = _rate_series(
        true_calls=clean_true_calls,
        defended_probabilities=np.asarray(
            clean_evidence_data["clean_probabilities"], dtype=float
        ),
        call_threshold=call_threshold,
    )
    clean_fp_reference = false_positive_rate_percent(
        clean_true_calls,
        clean_reference_calls,
    )
    clean_fn_reference = false_negative_rate_percent(
        clean_true_calls,
        clean_reference_calls,
    )
    clean_local_effect = np.mean(
        np.asarray(clean_evidence_data["clean_local_effects"], dtype=float),
        axis=1,
    )
    clean_local_effect_reference = float(
        np.mean(np.asarray(clean_evidence_data["clean_local_effect"], dtype=float))
    )

    # The rate cache evaluates only the two slices with the complementary pair
    # fixed at (0.5, 0.5), avoiding a full N=700 evaluation of all 25 cells.
    omega_probabilities = np.asarray(rate_data["omega_probabilities"], dtype=float)
    evidence_probabilities = np.asarray(rate_data["evidence_probabilities"], dtype=float)
    omega_fp, _ = _rate_series(
        true_calls=true_calls,
        defended_probabilities=omega_probabilities,
        call_threshold=call_threshold,
    )
    evidence_fp, _ = _rate_series(
        true_calls=true_calls,
        defended_probabilities=evidence_probabilities,
        call_threshold=call_threshold,
    )
    fp_reference = false_positive_rate_percent(true_calls, attack_calls)

    # Only false-negative curves use the enlarged sample.  Keeping it separate
    # prevents the FP and local-effect experiments from being recalculated.
    false_negative_true_calls = (
        np.asarray(false_negative_data["true_g"], dtype=float) > float(call_threshold)
    )
    false_negative_attack_calls = (
        np.asarray(false_negative_data["attack_probability"], dtype=float)
        > float(call_threshold)
    )
    _, omega_fn = _rate_series(
        true_calls=false_negative_true_calls,
        defended_probabilities=np.asarray(
            false_negative_data["omega_probabilities"], dtype=float
        ),
        call_threshold=call_threshold,
    )
    _, evidence_fn = _rate_series(
        true_calls=false_negative_true_calls,
        defended_probabilities=np.asarray(
            false_negative_data["evidence_probabilities"], dtype=float
        ),
        call_threshold=call_threshold,
    )
    fn_reference = false_negative_rate_percent(
        false_negative_true_calls,
        false_negative_attack_calls,
    )
    omega_local_effect = np.mean(np.asarray(local_data["omega_local_effects"], dtype=float), axis=1)
    evidence_local_effect = np.mean(
        np.asarray(local_data["evidence_local_effects"], dtype=float), axis=1
    )
    local_effect_reference = float(np.mean(np.asarray(local_data["attack_local_effect"], dtype=float)))

    # Each data branch occupies one equal-width column of three small panels.
    fig = plt.figure(figsize=(20.6, 8.0), constrained_layout=True)
    outer_grid = fig.add_gridspec(1, 3)
    clean_grid = outer_grid[0, 0].subgridspec(3, 1, hspace=0.06)
    omega_grid = outer_grid[0, 1].subgridspec(3, 1, hspace=0.06)
    evidence_grid = outer_grid[0, 2].subgridspec(3, 1, hspace=0.06)
    clean_axes = (
        fig.add_subplot(clean_grid[0, 0]),
        fig.add_subplot(clean_grid[1, 0]),
        fig.add_subplot(clean_grid[2, 0]),
    )
    omega_axes = (
        fig.add_subplot(omega_grid[0, 0]),
        fig.add_subplot(omega_grid[1, 0]),
        fig.add_subplot(omega_grid[2, 0]),
    )
    evidence_axes = (
        fig.add_subplot(evidence_grid[0, 0]),
        fig.add_subplot(evidence_grid[1, 0]),
        fig.add_subplot(evidence_grid[2, 0]),
    )
    for ax in (*clean_axes, *omega_axes, *evidence_axes):
        base.style_axis(ax)

    _plot_metric_stack(
        clean_axes,
        weight_values=clean_weight_values,
        false_positive=clean_fp,
        false_negative=clean_fn,
        false_positive_reference=clean_fp_reference,
        false_negative_reference=clean_fn_reference,
        local_effect=clean_local_effect,
        local_effect_reference=clean_local_effect_reference,
        x_label=r"$w_M$ in $(w_M,w_g)=(q,1-q)$",
        legend_title=r"Clean; $(\omega_h,\omega_o)=(0.5,0.5)$",
        curve_labels=("Clean + cov-adapt", "Clean + cov-adapt", "Clean + cov-adapt"),
        reference_label="Clean reference",
    )
    _plot_metric_stack(
        omega_axes,
        weight_values=rate_weight_values,
        false_positive=omega_fp,
        false_negative=omega_fn,
        false_positive_reference=fp_reference,
        false_negative_reference=fn_reference,
        local_effect=omega_local_effect,
        local_effect_reference=local_effect_reference,
        x_label=r"$\omega_h$ in $(\omega_h,\omega_o)=(q,1-q)$",
        legend_title=r"$(w_M,w_g)=(0.5,0.5)$",
        curve_labels=("False positive", "False negative", "Local effect"),
        reference_label="Reference",
    )
    _plot_metric_stack(
        evidence_axes,
        weight_values=rate_weight_values,
        false_positive=evidence_fp,
        false_negative=evidence_fn,
        false_positive_reference=fp_reference,
        false_negative_reference=fn_reference,
        local_effect=evidence_local_effect,
        local_effect_reference=local_effect_reference,
        x_label=r"$w_M$ in $(w_M,w_g)=(q,1-q)$",
        legend_title=r"$(\omega_h,\omega_o)=(0.5,0.5)$",
        curve_labels=("False positive", "False negative", "Local effect"),
        reference_label="Reference",
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Plot clean evidence weights beside the two attacked weight sweeps."""
    T = 5
    attack_t = T
    seed = 2025
    coverage = 0.95
    lambda_scale = 1.0
    omega_h_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)

    # The clean and standard attacked curves share 700 seeds.  The attacked FN
    # estimate retains its previously generated larger sample of 5000 runs.
    previous_grid_N_runs = 250
    clean_N_runs = 700
    rate_N_runs = 700
    false_negative_N_runs = 5000
    mc_seed = seed
    eta = 1.5
    n_steps = 700
    n_mc_opt = 96
    n_mc_est = 1200
    n_jobs = min(8, max(1, (os.cpu_count() or 1) - 1))
    force_mc = False

    if not (0 <= attack_t <= T):
        raise ValueError("attack_t must satisfy 0 <= attack_t <= T.")
    if not np.any(np.isclose(omega_h_values, 0.5, atol=1e-12)):
        raise ValueError("omega_h_values must contain 0.5 for the rate slice.")

    figure_dir = figures_dir_for(CURRENT_DIR)
    base_tag = f"cov{int(round(100 * coverage))}_t{attack_t}_T{T}_seed{seed}"
    rate_base_name = (
        "comparison_gamma_weight_grid_attacked_rates_"
        f"{base_tag}_Ngrid{previous_grid_N_runs}_Nrates{rate_N_runs}"
    )
    combined_outpath = os.path.join(
        figure_dir,
        (
            "comparison_clean_evidence_attacked_weight_rates_"
            f"{base_tag}_Nclean{clean_N_runs}_Nrates{rate_N_runs}_"
            f"Nfn{false_negative_N_runs}.png"
        ),
    )
    rate_data_path = data_path_for_plot(os.path.join(figure_dir, f"{rate_base_name}_rates.png"))
    local_data_path = data_path_for_plot(
        os.path.join(figure_dir, f"{rate_base_name}_local_effects.png")
    )
    false_negative_data_path = data_path_for_plot(
        os.path.join(
            figure_dir,
            f"{rate_base_name}_Nfn{false_negative_N_runs}_false_negative.png",
        )
    )
    clean_evidence_data_path = data_path_for_plot(
        combined_outpath.replace(".png", "_clean_evidence.png")
    )

    rate_data = cached_npz(
        rate_data_path,
        lambda: weight_sweep.run_monte_carlo_weight_sweeps(
            N_runs=rate_N_runs,
            mc_seed=mc_seed,
            T=T,
            attack_t=attack_t,
            coverage=coverage,
            lambda_scale=lambda_scale,
            weight_values=omega_h_values,
            eta=eta,
            n_steps=n_steps,
            n_mc_opt=n_mc_opt,
            n_mc_est=n_mc_est,
            n_jobs=n_jobs,
        ),
        force=force_mc,
    )
    clean_evidence_data = cached_npz(
        clean_evidence_data_path,
        lambda: run_monte_carlo_clean_evidence_sweep(
            run_seeds=np.asarray(rate_data["run_seeds"], dtype=np.uint32),
            T=T,
            attack_t=attack_t,
            coverage=coverage,
            lambda_scale=lambda_scale,
            weight_values=omega_h_values,
            eta=eta,
            n_steps=n_steps,
            n_mc_opt=n_mc_opt,
            n_mc_est=n_mc_est,
            n_jobs=n_jobs,
        ),
        force=force_mc,
    )
    false_negative_data = cached_npz(
        false_negative_data_path,
        lambda: extend_rate_data_for_false_negative(
            base_rate_data=rate_data,
            target_N_runs=false_negative_N_runs,
            extra_mc_seed=mc_seed + 1,
            T=T,
            attack_t=attack_t,
            coverage=coverage,
            lambda_scale=lambda_scale,
            weight_values=omega_h_values,
            eta=eta,
            n_steps=n_steps,
            n_mc_opt=n_mc_opt,
            n_mc_est=n_mc_est,
            call_threshold=base.DEFAULT_CALL_THRESHOLD,
            n_jobs=n_jobs,
        ),
        force=force_mc,
    )
    local_data = cached_npz(
        local_data_path,
        lambda: run_monte_carlo_local_effect_sweeps(
            run_seeds=np.asarray(rate_data["run_seeds"], dtype=np.uint32),
            T=T,
            attack_t=attack_t,
            coverage=coverage,
            lambda_scale=lambda_scale,
            weight_values=omega_h_values,
            eta=eta,
            n_steps=n_steps,
            n_mc_opt=n_mc_opt,
            n_jobs=n_jobs,
        ),
        force=force_mc,
    )
    plot_combined_weight_summary(
        clean_evidence_data=clean_evidence_data,
        rate_data=rate_data,
        false_negative_data=false_negative_data,
        local_data=local_data,
        call_threshold=base.DEFAULT_CALL_THRESHOLD,
        outpath=combined_outpath,
    )
    print(f"Saved combined figure to: {combined_outpath}")
    print(f"Saved clean evidence-weight data: {clean_evidence_data_path}")
    print(f"Saved rate-curve data: {rate_data_path}")
    print(f"Saved high-resolution false-negative data: {false_negative_data_path}")
    print(f"Saved local-effect data: {local_data_path}")


if __name__ == "__main__":
    main()
