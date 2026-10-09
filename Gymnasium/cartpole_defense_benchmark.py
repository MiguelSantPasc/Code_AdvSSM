#!/usr/bin/env python3
"""
cartpole_defense_benchmark.py

Reproducible CartPole defense benchmark aligned with `RL/defense_benchmark.py`.

Why this module exists:
1. The repository already contains the full wind-defense benchmark in `RL/`,
   plus CartPole comparison scripts for covariance adaptation and WoLF.
2. What was still missing was one CartPole benchmark with the same output
   structure as the RL benchmark:
   - episode-level CSV,
   - step-level diagnostics CSV,
   - compressed NPZ payload,
   - a return-comparison figure,
   - a dynamics/diagnostics figure.
3. This script fills that gap while reusing the exact CartPole attack and
   defense primitives already implemented in
   `Gymnasium/cartpole_covadapt_compare_epsilons_wolf.py`.

Discounted-covariance rationale:
1. The EKF-style predictor first propagates the posterior covariance with the
   local CartPole Jacobian,

      P_raw(t+1|t) = A_t P(t|t) A_t^T.

2. The current CartPole predictor then applies the fixed discount rule

      P(t+1|t) = P_raw(t+1|t) / delta,

   with `0 < delta <= 1`.
3. This is equivalent to an additive state-dependent process covariance

      P(t+1|t)
      = P_raw(t+1|t) + (delta^{-1} - 1) P_raw(t+1|t)
      = P_raw(t+1|t) + Q_eff,t,

   where

      Q_eff,t = (delta^{-1} - 1) P_raw(t+1|t).

   So the discount factor is not arbitrary inflation: it is a compact way of
   saying that the linearized predictor is under-modeling uncertainty and needs
   an uncertainty floor proportional to the propagated covariance.
4. The calibration target is the normalized innovation squared

      NIS_t = nu_t^T S_t^{-1} nu_t,
      nu_t  = o_t - o_hat(t|t-1),
      S_t   = P(t|t-1) + R,

   because the observation model is `o_t = s_t + r_t`.
5. Under a correctly calibrated 4D Gaussian filter, the expected NIS is the
   observation dimension:

      E[NIS_t] = 4.

6. The existing CartPole calibration log in
   `Gymnasium/outputs/data/calibrate_cartpole_discounted_covariances_*.log`
   reports:
   - `delta = 0.93  -> mean NIS = 3.9848`,
   - `delta = 0.94  -> mean NIS = 4.0046`.
   Since `|4.0046 - 4| < |3.9848 - 4|`, the current default `delta = 0.94`
   was the closer choice in that earlier scenario. This script retains 0.94,
   but does not claim a new NIS calibration for the centered policy and stops.

Experimental protocol:
1. The clean environment dynamics are a finer-step CartPole rollout:
   the plant is integrated internally with `tau_real = 0.01` while the EKF
   keeps the coarser nominal step `tau_filter = 0.02`.
2. Comparable randomness across methods is enforced with shared seeds and with
   separate RNG streams for:
   - additive observation noise,
   - attack gating,
   - random ellipsoidal attacks.
3. The initial observation is noisy but never attacked, matching the current
   CartPole attack scripts.
4. From step 1 onward, attacks are activated independently with probability
   `attack_prob`.
5. The benchmark is configured for 95% ellipsoidal coverage by default, so the
   attack radius is

      epsilon = chi2.ppf(0.95, df=4) ~= 9.49.
6. The estimated-return PGD attack evaluates perceived greedy actions at a
   fixed nominal KF estimate from the current observation before manipulation.
   Its Bellman objective matches RL's estimated-return attack; a softmax
   surrogate provides gradients for the discrete DQN, while candidate scores
   use hard actions. Output/cache tags distinguish it from the former
   posterior-value attack and its old WoLF tuning results.
7. The policy is the separately adapted centered DQN. Cart stops are at +/-2.4
   m and reward is 1 - 0.8*(x_next/2.4)^2. All methods, including clean and
   random baselines, share this plant and reward. The first angle failure ends
   an episode; there is no post-failure visualization continuation here.
8. WoLF tuning and evaluation use disjoint seeds. Policy hashes, physics and
   complete settings fingerprint caches and are persisted with the results.
   The sampled hard-action attack score remains distinct from execution of
   the greedy action at the defended posterior mean.
9. The default evaluation uses 200 seeds per method. Windows process workers
   run independent seed batches with one Torch thread each; selection finishes
   before evaluation starts. The worker count changes scheduling only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import cartpole_covadapt_compare_epsilons_wolf as cartpole_mod

from shared_ssm.artifacts import data_dir_for
from shared_ssm.artifacts import figures_dir_for


BASE_SEED = 20260729
ATTACK_PROBABILITY = 0.20
BENCHMARK_LAMBDAS = (0.5, 1.0, 2.0)
BENCHMARK_COVERAGE = 0.95
N_TUNING_EPISODES = 6
N_EPISODES = 200
STRONG_WOLF_WEIGHT_THRESHOLD = 0.25


@dataclass(frozen=True)
class BenchmarkMethod:
    """Describe one attack/defense configuration included in the benchmark."""

    name: str
    attack_type: str
    filter_type: str
    lambda_covariance: float = 0.0
    wolf_kind: str | None = None
    wolf_parameters: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level CartPole benchmark configuration."""

    coverage: float
    attack_eps: float
    attack_prob: float
    lambdas: tuple[float, ...]
    discount_delta: float
    obs_noise_std: np.ndarray
    kf_meas_std: np.ndarray
    kf_proc_std: np.ndarray
    kf_meas_corr: np.ndarray
    kf_proc_corr: np.ndarray
    pgd_steps: int
    pgd_step_size: float
    policy_temperature: float
    mc_samples: int
    pgd_boundary_tol: float
    omega_h: float
    omega_o: float
    gamma_threshold: float
    wolf_imq_soft_threshold_attack: float
    wolf_imq_soft_threshold_random: float
    wolf_tmd_threshold_attack: float
    wolf_tmd_threshold_random: float
    strong_wolf_weight_threshold: float
    n_tuning_episodes: int
    n_episodes: int
    seed0: int
    scenario_tag: str
    model_path: str
    wall_position: float
    wall_restitution: float
    center_reward_weight: float
    filter_tau: float
    real_tau: float
    tuning_seed0: int


@dataclass(frozen=True)
class StepDiagnostics:
    """Step-wise diagnostics recorded for one method rollout."""

    step_index: int
    attacked: bool
    attack_type: str
    filter_type: str
    gamma_t: float
    gamma_bar_t: float
    effective_inflation: float
    state_estimation_error: float
    innovation_norm: float
    nominal_noise_norm: float
    pgd_perturbation_norm: float
    random_contour_perturbation_norm: float
    wolf_weight: float
    wolf_weight_squared: float
    wolf_heavily_discounted: bool


@dataclass(frozen=True)
class MethodEpisodeResult:
    """Aggregate metrics of one method on one environment seed."""

    method: BenchmarkMethod
    episode_seed: int
    episode_return: float
    success: bool
    mean_state_estimation_error: float
    steps: int
    diagnostics: list[StepDiagnostics]


def float_tag(value: float, *, digits: int = 3) -> str:
    """Return a filename-safe float tag."""
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def build_default_config() -> BenchmarkConfig:
    """Build the default 95% CartPole defense benchmark configuration."""
    coverage = float(BENCHMARK_COVERAGE)
    attack_eps = float(chi2.ppf(coverage, df=4))
    obs_noise_std = np.array([0.10, 0.22, 0.05, 0.22], dtype=float)
    kf_proc_std = np.array([0.320, 0.720, 0.160, 0.720], dtype=float)
    kf_meas_corr = np.array(
        [
            [1.00, 0.18, 0.06, 0.00],
            [0.18, 1.00, 0.14, 0.22],
            [0.06, 0.14, 1.00, 0.18],
            [0.00, 0.22, 0.18, 1.00],
        ],
        dtype=float,
    )
    kf_proc_corr = np.array(
        [
            [1.00, 0.24, 0.08, 0.00],
            [0.24, 1.00, 0.16, 0.26],
            [0.08, 0.16, 1.00, 0.22],
            [0.00, 0.26, 0.22, 1.00],
        ],
        dtype=float,
    )
    return BenchmarkConfig(
        coverage=coverage,
        attack_eps=attack_eps,
        attack_prob=float(ATTACK_PROBABILITY),
        lambdas=tuple(float(value) for value in BENCHMARK_LAMBDAS),
        discount_delta=float(cartpole_mod.DEFAULT_GYMNASIUM_DISCOUNT_DELTA),
        obs_noise_std=obs_noise_std.copy(),
        kf_meas_std=obs_noise_std.copy(),
        kf_proc_std=kf_proc_std,
        kf_meas_corr=kf_meas_corr,
        kf_proc_corr=kf_proc_corr,
        pgd_steps=20,
        pgd_step_size=0.34,
        policy_temperature=1.0,
        mc_samples=64,
        pgd_boundary_tol=0.10,
        omega_h=0.50,
        omega_o=0.50,
        gamma_threshold=0.20,
        wolf_imq_soft_threshold_attack=0.60,
        wolf_imq_soft_threshold_random=0.45,
        wolf_tmd_threshold_attack=3.0,
        wolf_tmd_threshold_random=2.6,
        strong_wolf_weight_threshold=float(STRONG_WOLF_WEIGHT_THRESHOLD),
        n_tuning_episodes=int(N_TUNING_EPISODES),
        n_episodes=int(N_EPISODES),
        seed0=int(BASE_SEED),
        scenario_tag="centered_cartpole_v1",
        model_path=str(Path(CURRENT_DIR) / "outputs/saved_models/sb3_dqn_cartpole_centered_v1/dqn-CartPole-centered.zip"),
        wall_position=2.4,
        wall_restitution=0.0,
        center_reward_weight=0.8,
        filter_tau=0.02,
        real_tau=0.01,
        tuning_seed0=int(BASE_SEED) + 100_000,
    )


def benchmark_ssm(config: BenchmarkConfig) -> cartpole_mod.CartPoleLinearSSM:
    """Use the video's physical stops and centered reward in every rollout."""
    return cartpole_mod.build_cartpole_linear_ssm(
        filter_tau=config.filter_tau, real_tau=config.real_tau,
        wall_position=config.wall_position, wall_restitution=config.wall_restitution,
        center_reward_weight=config.center_reward_weight,
    )


def benchmark_metadata(config: BenchmarkConfig) -> dict:
    """Fingerprint weights, physics, attack settings and the actual tuning grid."""
    return cartpole_mod.cartpole_experiment_metadata(
        ssm=benchmark_ssm(config), model_path=config.model_path,
        settings={**asdict(config), "wolf_grid": build_wolf_sweep_configurations()},
    )


def load_benchmark_policy(config: BenchmarkConfig):
    """Reject legacy or mismatched weights before tuning or collecting results."""
    from train_cartpole_centered import verify_centered_checkpoint
    ssm = benchmark_ssm(config)
    verify_centered_checkpoint(Path(config.model_path), ssm=ssm)
    model = cartpole_mod.load_cartpole_policy(config.model_path, torch.device("cpu"))
    model.policy.set_training_mode(False)
    return model, ssm


def benchmark_case_tag(config: BenchmarkConfig) -> str:
    """Return the file tag for the configured benchmark case."""
    return f"{cartpole_mod.cartpole_experiment_tag(benchmark_metadata(config))}_n{config.n_episodes}"


def default_output_prefix(config: BenchmarkConfig) -> str:
    """Return the default output prefix for the saved benchmark artifacts."""
    return f"cartpole_defense_benchmark_{benchmark_case_tag(config)}"


def default_wolf_output_prefix(config: BenchmarkConfig) -> str:
    """Return the default output prefix for the standalone CartPole WoLF sweep."""
    case_tag = benchmark_case_tag(config)
    return f"cartpole_wolf_benchmark_{case_tag}_ntune{int(config.n_tuning_episodes)}"


def best_wolf_json_path(config: BenchmarkConfig) -> str:
    """Return the case-specific JSON path with the selected CartPole WoLF parameters."""
    filename = f"best_cartpole_wolf_params_{benchmark_case_tag(config)}_ntune{int(config.n_tuning_episodes)}.json"
    return os.path.join(data_dir_for(CURRENT_DIR), filename)


def build_benchmark_methods(
    config: BenchmarkConfig,
    wolf_params: dict[str, dict[str, Any]],
) -> list[BenchmarkMethod]:
    """Return the ordered list of benchmark methods."""
    def selected_wolf_parameters(*, kind: str, attack_type: str) -> dict[str, Any]:
        """Return the scenario-specific WoLF parameters, with legacy fallback."""
        scenario_suffix = "random" if attack_type == "random_contour" else "pgd"
        scenario_key = f"wolf_{kind}_{scenario_suffix}"
        legacy_key = f"wolf_{kind}"
        if scenario_key in wolf_params:
            return dict(wolf_params[scenario_key]["selected_parameters"])
        return dict(wolf_params[legacy_key]["selected_parameters"])

    methods = [
        BenchmarkMethod(name="Clean", attack_type="clean", filter_type="clean"),
        BenchmarkMethod(name="Noisy + KF", attack_type="none", filter_type="kf"),
        BenchmarkMethod(name="PGD estimated, no defense", attack_type="pgd_estimated", filter_type="kf"),
    ]
    for lambda_value in config.lambdas:
        methods.append(
            BenchmarkMethod(
                name=f"PGD estimated + CovAdap, lambda={lambda_value:g}",
                attack_type="pgd_estimated",
                filter_type="covadapt",
                lambda_covariance=float(lambda_value),
            )
        )
    methods.append(
        BenchmarkMethod(
            name="PGD estimated + WoLF-IMQ",
            attack_type="pgd_estimated",
            filter_type="wolf_imq",
            wolf_kind="imq",
            wolf_parameters=selected_wolf_parameters(kind="imq", attack_type="pgd_estimated"),
        )
    )
    methods.append(
        BenchmarkMethod(
            name="PGD estimated + WoLF-TMD",
            attack_type="pgd_estimated",
            filter_type="wolf_tmd",
            wolf_kind="tmd",
            wolf_parameters=selected_wolf_parameters(kind="tmd", attack_type="pgd_estimated"),
        )
    )
    methods.append(
        BenchmarkMethod(name="Random contour, no defense", attack_type="random_contour", filter_type="kf")
    )
    for lambda_value in config.lambdas:
        methods.append(
            BenchmarkMethod(
                name=f"Random contour + CovAdap, lambda={lambda_value:g}",
                attack_type="random_contour",
                filter_type="covadapt",
                lambda_covariance=float(lambda_value),
            )
        )
    methods.append(
        BenchmarkMethod(
            name="Random contour + WoLF-IMQ",
            attack_type="random_contour",
            filter_type="wolf_imq",
            wolf_kind="imq",
            wolf_parameters=selected_wolf_parameters(kind="imq", attack_type="random_contour"),
        )
    )
    methods.append(
        BenchmarkMethod(
            name="Random contour + WoLF-TMD",
            attack_type="random_contour",
            filter_type="wolf_tmd",
            wolf_kind="tmd",
            wolf_parameters=selected_wolf_parameters(kind="tmd", attack_type="random_contour"),
        )
    )
    return methods


def wolf_thresholds_for_method(
    method: BenchmarkMethod,
    config: BenchmarkConfig,
) -> tuple[float, float]:
    """Return the WoLF thresholds that match the current attack family."""
    if method.wolf_parameters is not None:
        return (
            float(method.wolf_parameters.get("imq_soft_threshold", 1.0)),
            float(method.wolf_parameters.get("tmd_threshold", 3.0)),
        )
    if method.attack_type == "random_contour":
        return float(config.wolf_imq_soft_threshold_random), float(config.wolf_tmd_threshold_random)
    return float(config.wolf_imq_soft_threshold_attack), float(config.wolf_tmd_threshold_attack)


def success_from_episode(
    *,
    terminated: bool,
    truncated: bool,
    steps: int,
    env: gym.Env,
) -> bool:
    """Return whether the episode reached the time limit without failure."""
    max_steps = int(getattr(env.spec, "max_episode_steps", 500) or 500)
    return bool(truncated or steps >= max_steps) and not bool(terminated)


def build_wolf_sweep_configurations() -> list[dict[str, Any]]:
    """Return the CartPole WoLF hyperparameter grid used before the final benchmark."""
    configs: list[dict[str, Any]] = []
    # The richer IMQ grid keeps extra density around the previously strongest
    # region near `tau ~= 0.45` while still probing milder and stronger
    # discounting.
    for value in (0.30, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.55, 0.60, 0.65):
        configs.append(
            {
                "config_label": f"wolf_imq_tau_{str(value).replace('.', 'p')}",
                "kind": "imq",
                "imq_soft_threshold": float(value),
                "tmd_threshold": 2.8,
                "min_weight": 1e-6,
            }
        )
    # The richer TMD grid mirrors the same idea around `tau ~= 2.8`.
    for value in (2.4, 2.5, 2.6, 2.7, 2.75, 2.8, 2.85, 2.9, 3.0, 3.1, 3.2):
        configs.append(
            {
                "config_label": f"wolf_tmd_tau_{str(value).replace('.', 'p')}",
                "kind": "tmd",
                "imq_soft_threshold": 0.45,
                "tmd_threshold": float(value),
                "min_weight": 1e-6,
            }
        )
    return configs


def rollout_clean_episode(
    env: gym.Env,
    model: cartpole_mod.DQN,
    *,
    seed: int,
    ssm: cartpole_mod.CartPoleLinearSSM,
) -> MethodEpisodeResult:
    """Run the clean baseline with direct policy observations."""
    obs, _info = cartpole_mod.reset_cartpole_rollout(env, seed=int(seed))
    diagnostics: list[StepDiagnostics] = []
    episode_return = 0.0
    steps = 0

    while True:
        action = cartpole_mod.select_action(model, obs)
        diagnostics.append(
            StepDiagnostics(
                step_index=int(steps),
                attacked=False,
                attack_type="clean",
                filter_type="clean",
                gamma_t=0.0,
                gamma_bar_t=0.0,
                effective_inflation=0.0,
                state_estimation_error=0.0,
                innovation_norm=0.0,
                nominal_noise_norm=0.0,
                pgd_perturbation_norm=0.0,
                random_contour_perturbation_norm=0.0,
                wolf_weight=1.0,
                wolf_weight_squared=1.0,
                wolf_heavily_discounted=False,
            )
        )
        obs, reward, terminated, truncated, _info = cartpole_mod.step_cartpole_rollout(env, action, ssm=ssm)
        episode_return += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return MethodEpisodeResult(
        method=BenchmarkMethod(name="Clean", attack_type="clean", filter_type="clean"),
        episode_seed=int(seed),
        episode_return=float(episode_return),
        success=success_from_episode(terminated=terminated, truncated=truncated, steps=steps, env=env),
        mean_state_estimation_error=0.0,
        steps=int(steps),
        diagnostics=diagnostics,
    )


def rollout_benchmark_episode(
    env: gym.Env,
    model: cartpole_mod.DQN,
    method: BenchmarkMethod,
    *,
    seed: int,
    ssm: cartpole_mod.CartPoleLinearSSM,
    R: np.ndarray,
    config: BenchmarkConfig,
    device: str,
) -> MethodEpisodeResult:
    """Run one CartPole method rollout using the paired-noise benchmark protocol."""
    obs, _info = cartpole_mod.reset_cartpole_rollout(env, seed=int(seed))
    diagnostics: list[StepDiagnostics] = []
    episode_return = 0.0
    total_state_error = 0.0

    rng_attack_gate = np.random.default_rng(int(seed) + 707_001)
    rng_nominal_noise = np.random.default_rng(int(seed) + 707_002)
    rng_boundary = np.random.default_rng(int(seed) + 707_003)

    init_noise = rng_nominal_noise.normal(0.0, config.obs_noise_std, size=(4,)).astype(np.float32)
    obs_init_noisy = (np.asarray(obs, dtype=np.float32) + init_noise).astype(np.float32)
    m_post, P_post = cartpole_mod.kf_update_state(
        m_pred=np.asarray(obs, dtype=np.float32),
        P_pred=cartpole_mod.project_to_psd(R.copy()),
        y_obs=obs_init_noisy,
        R=R,
    )

    state_error = float(np.linalg.norm(np.asarray(m_post, dtype=float) - np.asarray(obs, dtype=float)))
    diagnostics.append(
        StepDiagnostics(
            step_index=0,
            attacked=False,
            attack_type=method.attack_type,
            filter_type=method.filter_type,
            gamma_t=0.0,
            gamma_bar_t=0.0,
            effective_inflation=0.0,
            state_estimation_error=state_error,
            innovation_norm=float(np.linalg.norm(np.asarray(obs_init_noisy, dtype=float) - np.asarray(obs, dtype=float))),
            nominal_noise_norm=float(np.linalg.norm(init_noise)),
            pgd_perturbation_norm=0.0,
            random_contour_perturbation_norm=0.0,
            wolf_weight=1.0,
            wolf_weight_squared=1.0,
            wolf_heavily_discounted=False,
        )
    )
    total_state_error += state_error

    action = cartpole_mod.select_action(model, m_post)
    force = cartpole_mod.action_to_force(action, ssm.force_mag)
    m_pred, P_pred = cartpole_mod.kf_predict_state(
        m_post=m_post,
        P_post=P_post,
        force=force,
        Q=None,
        discount_delta=float(config.discount_delta),
        ssm=ssm,
    )

    obs, reward, terminated, truncated, _info = cartpole_mod.step_cartpole_rollout(env, action, ssm=ssm)
    episode_return += float(reward)
    if terminated or truncated:
        return MethodEpisodeResult(
            method=method,
            episode_seed=int(seed),
            episode_return=float(episode_return),
            success=success_from_episode(terminated=terminated, truncated=truncated, steps=1, env=env),
            mean_state_estimation_error=float(total_state_error / max(len(diagnostics), 1)),
            steps=1,
            diagnostics=diagnostics,
        )

    step_idx = 1

    while True:
        y_clean = np.asarray(obs, dtype=np.float32)
        nominal_noise = rng_nominal_noise.normal(0.0, config.obs_noise_std, size=(4,)).astype(np.float32)
        y_noisy = (y_clean + nominal_noise).astype(np.float32)

        attacked = bool(
            method.attack_type in {"pgd_estimated", "random_contour"}
            and rng_attack_gate.random() < float(config.attack_prob)
        )
        y_filter = y_noisy
        gamma_t = 0.0
        gamma_bar_t = 0.0
        effective_inflation = 0.0
        innovation_norm = float(np.linalg.norm(np.asarray(y_noisy, dtype=float) - np.asarray(m_pred, dtype=float)))
        pgd_perturbation_norm = 0.0
        random_contour_perturbation_norm = 0.0
        wolf_weight_sq = 1.0

        if attacked:
            attack_center = np.asarray(m_pred, dtype=np.float32)
            attack_sigma = cartpole_mod.project_to_psd(np.asarray(P_pred, dtype=float) + np.asarray(R, dtype=float))
            if method.attack_type == "pgd_estimated":
                y_filter, _obj_star, m_post_attack, P_post_attack = cartpole_mod.estimated_return_pgd_attack_observation(
                    model=model,
                    obs_nom=y_noisy,
                    m_pred=m_pred,
                    P_pred=P_pred,
                    R=R,
                    attack_center=attack_center,
                    attack_sigma=attack_sigma,
                    attack_eps=float(config.attack_eps),
                    pgd_steps=int(config.pgd_steps),
                    pgd_step_size=float(config.pgd_step_size),
                    mc_samples=int(config.mc_samples),
                    policy_temperature=float(config.policy_temperature),
                    ssm=ssm, current_step_index=int(step_idx),
                    max_episode_steps=int(env.spec.max_episode_steps or 500),
                    rng_seed=int(seed) + 10_000 * int(step_idx),
                    device=device,
                )
                pgd_perturbation_norm = float(
                    np.linalg.norm(np.asarray(y_filter, dtype=float) - np.asarray(y_noisy, dtype=float))
                )
            elif method.attack_type == "random_contour":
                y_filter = cartpole_mod.sample_random_attack_in_ellipsoid(
                    center=attack_center,
                    Sigma=attack_sigma,
                    epsilon=float(config.attack_eps),
                    rng=rng_boundary,
                )
                random_contour_perturbation_norm = float(
                    np.linalg.norm(np.asarray(y_filter, dtype=float) - np.asarray(y_noisy, dtype=float))
                )
                m_post_attack = None
                P_post_attack = None
            else:
                raise ValueError(f"Unsupported attack type: {method.attack_type}")
            innovation_norm = float(np.linalg.norm(np.asarray(y_filter, dtype=float) - np.asarray(m_pred, dtype=float)))
        else:
            m_post_attack = None
            P_post_attack = None

        if method.filter_type == "kf":
            if attacked and method.attack_type == "pgd_estimated" and m_post_attack is not None and P_post_attack is not None:
                m_post = np.asarray(m_post_attack, dtype=np.float32)
                P_post = np.asarray(P_post_attack, dtype=np.float32)
            else:
                m_post, P_post = cartpole_mod.kf_update_state(
                    m_pred=m_pred,
                    P_pred=P_pred,
                    y_obs=y_filter,
                    R=R,
                )
        elif method.filter_type == "covadapt":
            adv_target = np.asarray(y_filter, dtype=np.float32) if attacked else None
            m_post, P_post, cov_diag = cartpole_mod.covariance_adapted_kf_update_state(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=y_filter,
                R=R,
                adv_target=adv_target,
                c_scale=float(method.lambda_covariance),
                omega_h=float(config.omega_h),
                omega_o=float(config.omega_o),
                delta_threshold=float(config.gamma_threshold),
            )
            gamma_t = float(cov_diag["gamma_t"])
            gamma_bar_t = float(cov_diag["bar_gamma_t"])
            effective_inflation = float(method.lambda_covariance) * float(gamma_bar_t)
            innovation_norm = float(cov_diag["innovation_norm"])
        elif method.filter_type in {"wolf_imq", "wolf_tmd"}:
            imq_tau, tmd_tau = wolf_thresholds_for_method(method, config)
            m_post, P_post, wolf_diag = cartpole_mod.wolf_kf_update_state(
                m_pred=m_pred,
                P_pred=P_pred,
                y_obs=y_filter,
                R=R,
                wolf_kind=str(method.wolf_kind),
                imq_soft_threshold=float(imq_tau),
                tmd_threshold=float(tmd_tau),
            )
            wolf_weight_sq = float(wolf_diag["weight_sq"])
            innovation_norm = float(wolf_diag["innovation_norm"])
        else:
            raise ValueError(f"Unsupported filter type: {method.filter_type}")

        state_error = float(np.linalg.norm(np.asarray(m_post, dtype=float) - np.asarray(y_clean, dtype=float)))
        total_state_error += state_error
        wolf_weight = float(np.sqrt(max(wolf_weight_sq, 0.0)))
        diagnostics.append(
            StepDiagnostics(
                step_index=int(step_idx),
                attacked=bool(attacked),
                attack_type=method.attack_type,
                filter_type=method.filter_type,
                gamma_t=float(gamma_t),
                gamma_bar_t=float(gamma_bar_t),
                effective_inflation=float(effective_inflation),
                state_estimation_error=float(state_error),
                innovation_norm=float(innovation_norm),
                nominal_noise_norm=float(np.linalg.norm(nominal_noise)),
                pgd_perturbation_norm=float(pgd_perturbation_norm),
                random_contour_perturbation_norm=float(random_contour_perturbation_norm),
                wolf_weight=float(wolf_weight),
                wolf_weight_squared=float(wolf_weight_sq),
                wolf_heavily_discounted=bool(wolf_weight <= float(config.strong_wolf_weight_threshold)),
            )
        )

        action = cartpole_mod.select_action(model, m_post)
        force = cartpole_mod.action_to_force(action, ssm.force_mag)
        m_pred, P_pred = cartpole_mod.kf_predict_state(
            m_post=m_post,
            P_post=P_post,
            force=force,
            Q=None,
            discount_delta=float(config.discount_delta),
            ssm=ssm,
        )

        obs, reward, terminated, truncated, _info = cartpole_mod.step_cartpole_rollout(env, action, ssm=ssm)
        episode_return += float(reward)
        step_idx += 1
        if terminated or truncated:
            break

    steps = int(step_idx)
    return MethodEpisodeResult(
        method=method,
        episode_seed=int(seed),
        episode_return=float(episode_return),
        success=success_from_episode(terminated=terminated, truncated=truncated, steps=steps, env=env),
        mean_state_estimation_error=float(total_state_error / max(len(diagnostics), 1)),
        steps=steps,
        diagnostics=diagnostics,
    )


def diagnostics_to_dataframe(results: list[MethodEpisodeResult]) -> pd.DataFrame:
    """Convert step diagnostics into one flat DataFrame."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for diag in result.diagnostics:
            rows.append(
                {
                    "method": result.method.name,
                    "attack_type": result.method.attack_type,
                    "filter_or_defense": result.method.filter_type,
                    "lambda": float(result.method.lambda_covariance),
                    "episode_seed": int(result.episode_seed),
                    "step_index": int(diag.step_index),
                    "attacked": bool(diag.attacked),
                    "gamma_t": float(diag.gamma_t),
                    "gamma_bar_t": float(diag.gamma_bar_t),
                    "effective_inflation": float(diag.effective_inflation),
                    "state_estimation_error": float(diag.state_estimation_error),
                    "innovation_norm": float(diag.innovation_norm),
                    "nominal_noise_norm": float(diag.nominal_noise_norm),
                    "pgd_perturbation_norm": float(diag.pgd_perturbation_norm),
                    "random_contour_perturbation_norm": float(diag.random_contour_perturbation_norm),
                    "wolf_weight": float(diag.wolf_weight),
                    "wolf_weight_squared": float(diag.wolf_weight_squared),
                    "wolf_heavily_discounted": bool(diag.wolf_heavily_discounted),
                }
            )
    return pd.DataFrame(rows)


def aggregate_benchmark_results(results: list[MethodEpisodeResult]) -> pd.DataFrame:
    """Aggregate episode returns and step diagnostics into one summary table."""
    diagnostics_df = diagnostics_to_dataframe(results)
    episode_rows: list[dict[str, Any]] = []
    for result in results:
        episode_rows.append(
            {
                "method": result.method.name,
                "filter_or_defense": result.method.filter_type,
                "attack_type": result.method.attack_type,
                "lambda": float(result.method.lambda_covariance),
                "hyperparameters": json.dumps(
                    {
                        "lambda_covariance": float(result.method.lambda_covariance),
                        "wolf_kind": result.method.wolf_kind,
                        "wolf_parameters": result.method.wolf_parameters,
                    },
                    sort_keys=True,
                ),
                "episode_seed": int(result.episode_seed),
                "episode_return": float(result.episode_return),
                "success": float(result.success),
                "mean_state_estimation_error": float(result.mean_state_estimation_error),
                "steps": int(result.steps),
            }
        )
    episode_df = pd.DataFrame(episode_rows)

    summary_rows: list[dict[str, Any]] = []
    for method_name, group_df in episode_df.groupby("method", sort=False):
        method_diag_df = diagnostics_df.loc[diagnostics_df["method"] == method_name]
        summary_rows.append(
            {
                "method": method_name,
                "filter_or_defense": str(group_df["filter_or_defense"].iloc[0]),
                "hyperparameters": str(group_df["hyperparameters"].iloc[0]),
                "attack_type": str(group_df["attack_type"].iloc[0]),
                "lambda": float(group_df["lambda"].iloc[0]),
                "mean_return": float(group_df["episode_return"].mean()),
                "std_return": float(group_df["episode_return"].std(ddof=1)) if len(group_df) > 1 else 0.0,
                "mean_state_estimation_error": float(group_df["mean_state_estimation_error"].mean()),
                "mean_episode_length": float(group_df["steps"].mean()),
                "success_rate": float(group_df["success"].mean()),
                "mean_gamma_t": float(method_diag_df["gamma_t"].mean()) if not method_diag_df.empty else 0.0,
                "gamma_ge_threshold_rate": float(
                    (method_diag_df["gamma_t"] >= 0.3).mean()
                ) if not method_diag_df.empty else 0.0,
                "mean_effective_inflation": float(
                    method_diag_df["effective_inflation"].mean()
                ) if not method_diag_df.empty else 0.0,
                "mean_wolf_weight": float(
                    method_diag_df.loc[method_diag_df["attacked"], "wolf_weight"].mean()
                ) if "wolf" in str(group_df["filter_or_defense"].iloc[0]) else np.nan,
                "wolf_strong_discount_rate": float(
                    method_diag_df.loc[method_diag_df["attacked"], "wolf_heavily_discounted"].mean()
                ) if "wolf" in str(group_df["filter_or_defense"].iloc[0]) else np.nan,
                "num_runs": int(len(group_df)),
            }
        )
    return pd.DataFrame(summary_rows)


def build_episode_dataframe(results: list[MethodEpisodeResult]) -> pd.DataFrame:
    """Convert the per-episode results into one flat DataFrame."""
    return pd.DataFrame(
        [
            {
                "method": result.method.name,
                "attack_type": result.method.attack_type,
                "filter_or_defense": result.method.filter_type,
                "lambda": float(result.method.lambda_covariance),
                "episode_seed": int(result.episode_seed),
                "episode_return": float(result.episode_return),
                "success": float(result.success),
                "mean_state_estimation_error": float(result.mean_state_estimation_error),
                "steps": int(result.steps),
            }
            for result in results
        ]
    )


def selection_score_from_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank WoLF configurations using the same logic as the RL benchmark."""
    attacked_df = summary_df.loc[summary_df["scenario"].isin(["pgd_estimated", "random_contour"])]
    grouped = attacked_df.groupby("config_label", sort=False).agg(
        attacked_mean_return=("mean_return", "mean"),
        attacked_mean_state_estimation_error=("mean_state_estimation_error", "mean"),
        attacked_std_return=("std_return", "mean"),
    )
    noisy_df = summary_df.loc[summary_df["scenario"] == "no_attack"].set_index("config_label")
    grouped["noisy_mean_return"] = noisy_df["mean_return"]
    grouped["noisy_mean_state_estimation_error"] = noisy_df["mean_state_estimation_error"]
    grouped = grouped.reset_index()
    grouped = grouped.sort_values(
        by=[
            "attacked_mean_return",
            "attacked_mean_state_estimation_error",
            "noisy_mean_return",
            "attacked_std_return",
        ],
        ascending=[False, True, False, True],
    )
    return grouped


def wolf_overall_score_table(full_df: pd.DataFrame) -> pd.DataFrame:
    """Join the clean and attacked WoLF metrics into one configuration table."""
    attacked_df = full_df.loc[full_df["scenario"].isin(["pgd_estimated", "random_contour"])]
    attacked_grouped = attacked_df.groupby("config_label", sort=False).agg(
        attacked_mean_return=("mean_return", "mean"),
        attacked_mean_state_estimation_error=("mean_state_estimation_error", "mean"),
        attacked_std_return=("std_return", "mean"),
        attacked_success_rate=("success_rate", "mean"),
    )
    clean_grouped = full_df.loc[full_df["scenario"] == "no_attack"].groupby("config_label", sort=False).agg(
        clean_mean_return=("mean_return", "mean"),
        clean_mean_state_estimation_error=("mean_state_estimation_error", "mean"),
        clean_std_return=("std_return", "mean"),
        clean_success_rate=("success_rate", "mean"),
    )
    parameter_df = (
        full_df.groupby("config_label", sort=False)[["kind", "imq_soft_threshold", "tmd_threshold"]]
        .first()
        .reset_index()
    )
    score_df = parameter_df.merge(attacked_grouped.reset_index(), on="config_label", how="left")
    score_df = score_df.merge(clean_grouped.reset_index(), on="config_label", how="left")
    score_df["overall_mean_return"] = (
        score_df["clean_mean_return"] + 2.0 * score_df["attacked_mean_return"]
    ) / 3.0
    return score_df


def clean_selection_ranking(full_df: pd.DataFrame) -> pd.DataFrame:
    """Rank WoLF configurations by clean robustness before the attacked tie-breakers."""
    ranking_df = wolf_overall_score_table(full_df)
    return ranking_df.sort_values(
        by=[
            "clean_mean_return",
            "attacked_mean_return",
            "attacked_mean_state_estimation_error",
            "attacked_std_return",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def adversarial_selection_ranking(full_df: pd.DataFrame) -> pd.DataFrame:
    """Rank WoLF configurations by attacked robustness with clean performance as a tie-breaker."""
    ranking_df = wolf_overall_score_table(full_df)
    return ranking_df.sort_values(
        by=[
            "attacked_mean_return",
            "attacked_mean_state_estimation_error",
            "clean_mean_return",
            "attacked_std_return",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def selection_payload_from_row(row: pd.Series) -> dict[str, Any]:
    """Convert one ranked WoLF row into the persisted JSON payload format."""
    return {
        "config_label": str(row["config_label"]),
        "selected_parameters": {
            "kind": str(row["kind"]),
            "imq_soft_threshold": float(row["imq_soft_threshold"]),
            "tmd_threshold": float(row["tmd_threshold"]),
            "min_weight": 1e-6,
        },
        "clean_mean_return": float(row["clean_mean_return"]),
        "attacked_mean_return": float(row["attacked_mean_return"]),
    }


def select_distinct_clean_and_adversarial_wolf_parameters(
    *,
    full_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Select one best clean WoLF configuration and one distinct attacked configuration."""
    clean_ranking_df = clean_selection_ranking(full_df)
    adversarial_ranking_df = adversarial_selection_ranking(full_df)
    if clean_ranking_df.empty or adversarial_ranking_df.empty:
        raise RuntimeError("The WoLF sweep did not generate enough rows to rank clean and adversarial settings.")

    clean_row = clean_ranking_df.iloc[0]
    adversarial_candidates = adversarial_ranking_df.loc[
        adversarial_ranking_df["config_label"] != clean_row["config_label"]
    ]
    adversarial_row = adversarial_candidates.iloc[0] if not adversarial_candidates.empty else adversarial_ranking_df.iloc[0]
    return {
        "wolf_clean": selection_payload_from_row(clean_row),
        "wolf_adversarial": selection_payload_from_row(adversarial_row),
    }


def select_best_wolf_parameters(*, full_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Select the best IMQ and TMD WoLF configurations from the tuning table."""
    selected: dict[str, dict[str, Any]] = {}
    for kind_key in ("imq", "tmd"):
        kind_full_df = full_df.loc[full_df["kind"] == kind_key]
        kind_ranking = selection_score_from_summary(kind_full_df)
        if kind_ranking.empty:
            raise RuntimeError(f"No WoLF sweep rows were generated for kind={kind_key}.")
        best_label = str(kind_ranking.iloc[0]["config_label"])
        best_row = kind_full_df.loc[kind_full_df["config_label"] == best_label].iloc[0]
        selected[f"wolf_{kind_key}"] = {
            "selected_parameters": {
                "kind": kind_key,
                "imq_soft_threshold": float(best_row["imq_soft_threshold"]),
                "tmd_threshold": float(best_row["tmd_threshold"]),
                "min_weight": 1e-6,
            }
        }
    return selected


def select_best_wolf_parameters_per_scenario(*, full_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Select one best WoLF configuration for each `(scenario, kind)` pair."""
    selected: dict[str, dict[str, Any]] = {}
    for scenario_key, suffix in (("pgd_estimated", "pgd"), ("random_contour", "random")):
        scenario_df = full_df.loc[full_df["scenario"] == scenario_key]
        for kind_key in ("imq", "tmd"):
            kind_df = scenario_df.loc[scenario_df["kind"] == kind_key]
            if kind_df.empty:
                raise RuntimeError(
                    f"No WoLF sweep rows were generated for scenario={scenario_key}, kind={kind_key}."
                )
            ranking_df = kind_df.sort_values(
                by=["mean_return", "mean_state_estimation_error", "std_return"],
                ascending=[False, True, True],
            )
            best_row = ranking_df.iloc[0]
            selected[f"wolf_{kind_key}_{suffix}"] = {
                "config_label": str(best_row["config_label"]),
                "selected_parameters": {
                    "kind": kind_key,
                    "imq_soft_threshold": float(best_row["imq_soft_threshold"]),
                    "tmd_threshold": float(best_row["tmd_threshold"]),
                    "min_weight": 1e-6,
                },
                "scenario": scenario_key,
                "mean_return": float(best_row["mean_return"]),
                "mean_state_estimation_error": float(best_row["mean_state_estimation_error"]),
            }
    return selected


def plot_wolf_benchmark_results(*, full_df: pd.DataFrame, output_path: str) -> None:
    """Save the standalone CartPole WoLF tuning figure."""
    cartpole_mod.set_plot_theme()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    pivot_df = full_df.pivot(index="config_label", columns="scenario", values="mean_return").fillna(np.nan)
    pivot_df = pivot_df.sort_index()
    x_positions = np.arange(len(pivot_df), dtype=float)
    width = 0.24
    scenario_order = ["no_attack", "pgd_estimated", "random_contour"]
    scenario_colors = {
        "no_attack": "#9DB7D5",
        "pgd_estimated": "#E6A57E",
        "random_contour": "#AFC7E8",
    }
    for idx, scenario in enumerate(scenario_order):
        values = pivot_df[scenario].to_numpy(dtype=float)
        ax.bar(
            x_positions + (idx - 1) * width,
            values,
            width=width,
            color=scenario_colors[scenario],
            edgecolor="white",
            linewidth=0.8,
            label=scenario.replace("_", " "),
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot_df.index.tolist(), rotation=35, ha="right")
    ax.set_ylabel("Mean accumulated reward")
    ax.set_xlabel("WoLF configuration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_wolf_selection_artifacts(
    *,
    config: BenchmarkConfig,
    full_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    selected_parameters: dict[str, dict[str, Any]],
    scenario_selected_parameters: dict[str, dict[str, Any]],
    clean_ranking_df: pd.DataFrame,
    adversarial_ranking_df: pd.DataFrame,
    distinct_selection: dict[str, dict[str, Any]],
    output_prefix: str,
) -> dict[str, str]:
    """Save the CartPole WoLF sweep tables, figure, and selected-parameter JSON."""
    data_dir = data_dir_for(CURRENT_DIR)
    figures_dir = figures_dir_for(CURRENT_DIR)
    full_csv_path = os.path.join(data_dir, f"{output_prefix}_full.csv")
    ranking_csv_path = os.path.join(data_dir, f"{output_prefix}_ranking.csv")
    clean_ranking_csv_path = os.path.join(data_dir, f"{output_prefix}_clean_ranking.csv")
    adversarial_ranking_csv_path = os.path.join(data_dir, f"{output_prefix}_adversarial_ranking.csv")
    figure_path = os.path.join(figures_dir, f"{output_prefix}.png")
    json_path = best_wolf_json_path(config)

    full_df.to_csv(full_csv_path, index=False)
    ranking_df.to_csv(ranking_csv_path, index=False)
    clean_ranking_df.to_csv(clean_ranking_csv_path, index=False)
    adversarial_ranking_df.to_csv(adversarial_ranking_csv_path, index=False)
    plot_wolf_benchmark_results(full_df=full_df, output_path=figure_path)

    payload = {
        "wolf_imq": selected_parameters["wolf_imq"],
        "wolf_tmd": selected_parameters["wolf_tmd"],
        "wolf_imq_pgd": scenario_selected_parameters["wolf_imq_pgd"],
        "wolf_imq_random": scenario_selected_parameters["wolf_imq_random"],
        "wolf_tmd_pgd": scenario_selected_parameters["wolf_tmd_pgd"],
        "wolf_tmd_random": scenario_selected_parameters["wolf_tmd_random"],
        "wolf_clean": distinct_selection["wolf_clean"],
        "wolf_adversarial": distinct_selection["wolf_adversarial"],
        "attack_objective": "nominal_kf_bellman_hard_actions_v1",
        "policy_temperature": float(config.policy_temperature),
        "selection_metric": "maximize attacked mean return; tie-break by lower state error, better noisy return, lower return variability",
        "scenario_selection_metric": "for each scenario and WoLF kind: maximize scenario mean return; tie-break by lower state error, then lower return variability",
        "distinct_selection_metric": "clean: maximize no-attack mean return; adversarial: maximize attacked mean return, excluding the clean winner when possible",
        "coverage": float(config.coverage),
        "epsilon": float(config.attack_eps),
        "n_tuning_episodes": int(config.n_tuning_episodes),
        "tuning_seeds": [int(config.tuning_seed0) + idx for idx in range(int(config.n_tuning_episodes))],
        "experiment": benchmark_metadata(config),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "wolf_full_csv_path": full_csv_path,
        "wolf_ranking_csv_path": ranking_csv_path,
        "wolf_clean_ranking_csv_path": clean_ranking_csv_path,
        "wolf_adversarial_ranking_csv_path": adversarial_ranking_csv_path,
        "wolf_figure_path": figure_path,
        "wolf_json_path": json_path,
    }


def run_wolf_sweep(
    *,
    model: cartpole_mod.DQN,
    ssm: cartpole_mod.CartPoleLinearSSM,
    R: np.ndarray,
    config: BenchmarkConfig,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the WoLF sweep over the clean-noisy, PGD, and random scenarios."""
    rows: list[dict[str, Any]] = []
    for parameters in build_wolf_sweep_configurations():
        for scenario in ("no_attack", "pgd_estimated", "random_contour"):
            method = BenchmarkMethod(
                name=f"{parameters['config_label']}::{scenario}",
                attack_type="none" if scenario == "no_attack" else scenario,
                filter_type=f"wolf_{parameters['kind']}",
                lambda_covariance=0.0,
                wolf_kind=str(parameters["kind"]),
                wolf_parameters=dict(parameters),
            )
            episode_results: list[MethodEpisodeResult] = []
            tuning_indices = range(int(config.n_tuning_episodes))
            if executor is not None:
                jobs = [(config, method, [int(config.tuning_seed0) + idx], 500) for idx in tuning_indices]
                for batch in executor.map(run_episode_batch, jobs):
                    episode_results.extend(batch)
                tuning_indices = ()
            for episode_idx in tuning_indices:
                seed = int(config.tuning_seed0) + episode_idx
                env = gym.make("CartPole-v1")
                try:
                    episode_results.append(
                        rollout_benchmark_episode(
                            env,
                            model,
                            method,
                            seed=seed,
                            ssm=ssm,
                            R=R,
                            config=config,
                            device="cpu",
                        )
                    )
                finally:
                    env.close()
            diagnostics_df = diagnostics_to_dataframe(episode_results)
            episode_returns = np.asarray([result.episode_return for result in episode_results], dtype=float)
            rows.append(
                {
                    "config_label": str(parameters["config_label"]),
                    "kind": str(parameters["kind"]),
                    "scenario": scenario,
                    "imq_soft_threshold": float(parameters["imq_soft_threshold"]),
                    "tmd_threshold": float(parameters["tmd_threshold"]),
                    "mean_return": float(np.mean(episode_returns)),
                    "std_return": float(np.std(episode_returns, ddof=1)) if len(episode_returns) > 1 else 0.0,
                    "mean_state_estimation_error": float(
                        np.mean([result.mean_state_estimation_error for result in episode_results], dtype=float)
                    ),
                    "mean_episode_length": float(np.mean([result.steps for result in episode_results], dtype=float)),
                    "success_rate": float(np.mean([result.success for result in episode_results], dtype=float)),
                    "mean_wolf_weight": float(diagnostics_df["wolf_weight"].mean()) if not diagnostics_df.empty else 1.0,
                    "wolf_strong_discount_rate": float(
                        diagnostics_df["wolf_heavily_discounted"].mean()
                    ) if not diagnostics_df.empty else 0.0,
                    "num_runs": int(len(episode_results)),
                }
            )
            print(
                "[cartpole_wolf_benchmark] "
                f"{parameters['config_label']} | {scenario} | "
                f"mean_return={float(np.mean(episode_returns)):.3f}"
            )

    full_df = pd.DataFrame(rows)
    ranking_df = selection_score_from_summary(full_df)
    return full_df, ranking_df


def ensure_wolf_params_json(
    *,
    config: BenchmarkConfig,
    model: cartpole_mod.DQN,
    ssm: cartpole_mod.CartPoleLinearSSM,
    R: np.ndarray,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Ensure the case-specific CartPole WoLF JSON exists, running tuning if needed."""
    json_path = best_wolf_json_path(config)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("experiment") != benchmark_metadata(config):
            raise ValueError("WoLF cache metadata does not match the requested experiment.")
        return payload, {"wolf_json_path": json_path}

    output_prefix = default_wolf_output_prefix(config)
    print(
        "[cartpole_wolf_benchmark] "
        f"starting tuning | output={output_prefix} | n_tuning_episodes={int(config.n_tuning_episodes)}"
    )
    full_df, ranking_df = run_wolf_sweep(
        model=model,
        ssm=ssm,
        R=R,
        config=config,
        executor=executor,
    )
    selected_parameters = select_best_wolf_parameters(full_df=full_df)
    scenario_selected_parameters = select_best_wolf_parameters_per_scenario(full_df=full_df)
    clean_ranking_df = clean_selection_ranking(full_df)
    adversarial_ranking_df = adversarial_selection_ranking(full_df)
    distinct_selection = select_distinct_clean_and_adversarial_wolf_parameters(full_df=full_df)
    artifact_paths = save_wolf_selection_artifacts(
        config=config,
        full_df=full_df,
        ranking_df=ranking_df,
        selected_parameters=selected_parameters,
        scenario_selected_parameters=scenario_selected_parameters,
        clean_ranking_df=clean_ranking_df,
        adversarial_ranking_df=adversarial_ranking_df,
        distinct_selection=distinct_selection,
        output_prefix=output_prefix,
    )
    payload = {
        "wolf_imq": selected_parameters["wolf_imq"],
        "wolf_tmd": selected_parameters["wolf_tmd"],
        "wolf_imq_pgd": scenario_selected_parameters["wolf_imq_pgd"],
        "wolf_imq_random": scenario_selected_parameters["wolf_imq_random"],
        "wolf_tmd_pgd": scenario_selected_parameters["wolf_tmd_pgd"],
        "wolf_tmd_random": scenario_selected_parameters["wolf_tmd_random"],
        "wolf_clean": distinct_selection["wolf_clean"],
        "wolf_adversarial": distinct_selection["wolf_adversarial"],
    }
    print(f"[cartpole_wolf_benchmark] saved json: {artifact_paths['wolf_json_path']}")
    return payload, artifact_paths


def run_wolf_only(
    *,
    config: BenchmarkConfig,
    output_prefix: str,
    workers: int = 1,
) -> dict[str, str]:
    """Run only the standalone WoLF sweep and persist the richer ranking outputs."""
    model, ssm = load_benchmark_policy(config)
    R, _legacy_Q = cartpole_mod.build_filter_covariances(
        meas_std=config.kf_meas_std,
        proc_std=config.kf_proc_std,
        meas_corr=config.kf_meas_corr,
        proc_corr=config.kf_proc_corr,
    )
    print(
        "[cartpole_wolf_benchmark] "
        f"mode=wolf | output={output_prefix} | n_tuning_episodes={int(config.n_tuning_episodes)}"
    )
    with ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None) as executor:
        full_df, ranking_df = run_wolf_sweep(
            model=model, ssm=ssm, R=R, config=config, executor=executor,
        )
    selected_parameters = select_best_wolf_parameters(full_df=full_df)
    scenario_selected_parameters = select_best_wolf_parameters_per_scenario(full_df=full_df)
    clean_ranking_df = clean_selection_ranking(full_df)
    adversarial_ranking_df = adversarial_selection_ranking(full_df)
    distinct_selection = select_distinct_clean_and_adversarial_wolf_parameters(full_df=full_df)
    artifact_paths = save_wolf_selection_artifacts(
        config=config,
        full_df=full_df,
        ranking_df=ranking_df,
        selected_parameters=selected_parameters,
        scenario_selected_parameters=scenario_selected_parameters,
        clean_ranking_df=clean_ranking_df,
        adversarial_ranking_df=adversarial_ranking_df,
        distinct_selection=distinct_selection,
        output_prefix=output_prefix,
    )
    print(
        "[cartpole_wolf_benchmark] "
        f"selected pgd/imq={scenario_selected_parameters['wolf_imq_pgd']['config_label']} | "
        f"pgd/tmd={scenario_selected_parameters['wolf_tmd_pgd']['config_label']} | "
        f"random/imq={scenario_selected_parameters['wolf_imq_random']['config_label']} | "
        f"random/tmd={scenario_selected_parameters['wolf_tmd_random']['config_label']}"
    )
    return artifact_paths


def plot_benchmark_results(
    *,
    summary_df: pd.DataFrame,
    coverage: float,
    output_path: str,
) -> None:
    """Save the two-panel return benchmark with the RL-style right legend."""
    cartpole_mod.set_plot_theme()
    fig = plt.figure(figsize=(16.8, 6.8))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.78), wspace=0.08)
    axes = np.asarray(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
        ],
        dtype=object,
    )
    axes[1].sharey(axes[0])
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.axis("off")

    def panel_methods(panel_kind: str) -> list[str]:
        if panel_kind == "pgd":
            covadap_lambda_rows = summary_df.loc[
                summary_df["method"].str.startswith("PGD estimated + CovAdap, lambda="),
                ["method", "lambda"],
            ].sort_values("lambda")
            names = [
                "Clean",
                "Noisy + KF",
                "PGD estimated, no defense",
                *covadap_lambda_rows["method"].tolist(),
            ]
            wolf_names = ["PGD estimated + WoLF-IMQ", "PGD estimated + WoLF-TMD"]
        else:
            covadap_lambda_rows = summary_df.loc[
                summary_df["method"].str.startswith("Random contour + CovAdap, lambda="),
                ["method", "lambda"],
            ].sort_values("lambda")
            names = [
                "Clean",
                "Noisy + KF",
                "Random contour, no defense",
                *covadap_lambda_rows["method"].tolist(),
            ]
            wolf_names = ["Random contour + WoLF-IMQ", "Random contour + WoLF-TMD"]
        for wolf_name in wolf_names:
            if wolf_name in summary_df["method"].values:
                names.append(wolf_name)
        return names

    method_colors = {
        "Clean": "#8EC5B5",
        "Noisy + KF": "#A8D3C2",
        "PGD estimated, no defense": "#E6A57E",
        "Random contour, no defense": "#AFC7E8",
        "PGD estimated + WoLF-IMQ": "#D8C3EA",
        "PGD estimated + WoLF-TMD": "#BDA6DB",
        "Random contour + WoLF-IMQ": "#D8C3EA",
        "Random contour + WoLF-TMD": "#BDA6DB",
    }
    pgd_lambda_palette = ["#F6D6B8", "#F0C987", "#D9B977"]
    contour_lambda_palette = ["#D9E5F7", "#BFD3F1", "#96B6E3"]

    def legend_label(method_name: str, panel_kind: str) -> str:
        if method_name == "Clean":
            return "Noiseless"
        if method_name == "Noisy + KF":
            return "Noisy + KF"
        if method_name == "PGD estimated, no defense":
            return "Attack + KF (no defense)"
        if method_name == "Random contour, no defense":
            return r"$\epsilon$ perturbation + KF (no defense)"
        if method_name.startswith("PGD estimated + CovAdap, lambda="):
            return method_name.replace("PGD estimated + CovAdap", "Attack + DirCovAdapt")
        if method_name.startswith("Random contour + CovAdap, lambda="):
            return method_name.replace("Random contour + CovAdap", r"$\epsilon$ perturbation + DirCovAdapt")
        if "WoLF-IMQ" in method_name:
            return ("Attack + " if panel_kind == "pgd" else r"$\epsilon$ perturbation + ") + "WoLF-IMQ"
        if "WoLF-TMD" in method_name:
            return ("Attack + " if panel_kind == "pgd" else r"$\epsilon$ perturbation + ") + "WoLF-TMD"
        return method_name

    panel_specs = [
        ("pgd", "Attack"),
        ("random", r"$\epsilon$ perturbation"),
    ]
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    plotted_mean_values: list[float] = []

    for ax, (panel_kind, panel_note) in zip(axes, panel_specs, strict=False):
        method_names = panel_methods(panel_kind)
        panel_df = summary_df.set_index("method").loc[method_names].reset_index()
        panel_mean_values = panel_df["mean_return"].to_numpy(dtype=float)
        plotted_mean_values.extend(panel_mean_values.tolist())
        colors: list[str] = []
        lambda_idx = 0
        for method_name in panel_df["method"].tolist():
            if method_name in method_colors:
                colors.append(method_colors[method_name])
            elif panel_kind == "pgd":
                colors.append(pgd_lambda_palette[lambda_idx % len(pgd_lambda_palette)])
                lambda_idx += 1
            else:
                colors.append(contour_lambda_palette[lambda_idx % len(contour_lambda_palette)])
                lambda_idx += 1

        x_positions = np.arange(len(panel_df), dtype=float)
        panel_legend_labels = [legend_label(name, panel_kind) for name in panel_df["method"].tolist()]
        bars = ax.bar(
            x_positions,
            panel_mean_values,
            color=colors,
            edgecolor="black",
            linewidth=0.45,
        )
        for bar, label in zip(bars, panel_legend_labels, strict=False):
            if label not in legend_labels:
                legend_handles.append(bar)
                legend_labels.append(label)
        ax.set_xticks([])
        for bar, value in zip(bars, panel_mean_values, strict=False):
            ax.text(
                float(bar.get_x() + 0.5 * bar.get_width()),
                float(value) + 0.004 * max(abs(float(np.min(panel_mean_values))), abs(float(np.max(panel_mean_values))), 1.0),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.2,
                color="#31424F",
            )
        ax.set_xlabel(panel_note, fontsize=13.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.24)
        ax.grid(False, axis="x")
        ax.axhline(0.0, color="#5F6B76", linewidth=1.0, linestyle="--", alpha=0.9, zorder=0)

    axes[0].set_ylabel("Mean accumulated reward")
    if plotted_mean_values:
        ymin = float(np.floor(np.min(plotted_mean_values)) - 10.0)
        ymax = float(np.ceil(np.max(plotted_mean_values)) + 10.0)
        if np.isclose(ymin, ymax):
            ymax = ymin + 1.0
        for ax in axes:
            ax.set_ylim(ymin, ymax)
    if legend_handles:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            fontsize=11.4,
            frameon=True,
            framealpha=0.92,
        )
    legend_ax.text(
        0.0,
        0.96,
        rf"$\epsilon = {100.0 * float(coverage):.0f}\%$",
        ha="left",
        va="top",
        fontsize=12.2,
        transform=legend_ax.transAxes,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "alpha": 0.94,
            "edgecolor": "#BFC8D0",
        },
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.14, wspace=0.08)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_dynamics(
    *,
    diagnostics_df: pd.DataFrame,
    coverage: float,
    output_path: str,
) -> None:
    """Save the attacked-step gamma and perturbation distributions."""
    cartpole_mod.set_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    attacked_df = diagnostics_df.loc[diagnostics_df["attacked"] == True].copy()

    pgd_method_name = "PGD estimated + CovAdap, lambda=0.5"
    contour_method_name = "Random contour + CovAdap, lambda=0.5"
    gamma_label_map = {
        pgd_method_name: "Estimated-return attack + DirCovAdapt",
        contour_method_name: r"$\epsilon$ perturbation + DirCovAdapt",
    }
    perturbation_label_map = {
        pgd_method_name: "Estimated-return attack",
        contour_method_name: r"$\epsilon$ perturbation",
    }
    color_map = {
        pgd_method_name: "#E6A57E",
        contour_method_name: "#AFC7E8",
    }

    def metric_values(method_name: str, metric: str, *, positive_only: bool) -> np.ndarray:
        values = attacked_df.loc[attacked_df["method"] == method_name, metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if positive_only:
            values = values[values > 0.0]
        return values

    def silverman_bandwidth(values: np.ndarray) -> float:
        if values.size <= 1:
            return 0.1
        value_std = float(np.std(values, ddof=1))
        value_iqr = float(np.subtract(*np.percentile(values, [75.0, 25.0])))
        robust_scale = min(value_std, value_iqr / 1.34) if value_iqr > 0.0 else value_std
        if robust_scale <= 1e-12:
            robust_scale = max(abs(float(np.mean(values))), 1.0) * 0.05
        return max(0.9 * robust_scale * values.size ** (-0.2), 1e-3)

    def draw_density(
        *,
        ax: plt.Axes,
        first_values: np.ndarray,
        first_method: str,
        second_values: np.ndarray,
        second_method: str,
        label_map: dict[str, str],
        legend_loc: str,
        xlabel: str,
        lower: float,
        upper: float | None,
    ) -> None:
        if first_values.size == 0 and second_values.size == 0:
            ax.text(0.5, 0.5, "No attacked data", ha="center", va="center", transform=ax.transAxes)
            return

        combined = np.concatenate([values for values in (first_values, second_values) if values.size > 0])
        max_value = float(np.max(combined))
        if upper is None:
            span = max(0.12 * max_value, 1e-3)
            grid = np.linspace(float(lower), max_value + span, 400)
        else:
            grid = np.linspace(float(lower), float(upper), 400)

        for values, method_name in ((first_values, first_method), (second_values, second_method)):
            if values.size == 0:
                continue
            bandwidth = silverman_bandwidth(values)
            density_values = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / bandwidth) ** 2)
            density_values += np.exp(-0.5 * ((grid[:, None] + values[None, :] - 2.0 * float(lower)) / bandwidth) ** 2)
            density_values /= np.sqrt(2.0 * np.pi)
            density_values = np.mean(density_values, axis=1) / bandwidth
            ax.fill_between(grid, density_values, color=color_map[method_name], alpha=0.24, linewidth=0.0)
            ax.plot(grid, density_values, color=color_map[method_name], linewidth=2.0, label=label_map[method_name])

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Distribution")
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.92, fontsize=10.6)
        ax.axhline(0.0, color="#5F6B76", linewidth=0.9, linestyle="--", alpha=0.8, zorder=0)
        cartpole_mod.style_axis(ax)

    draw_density(
        ax=axes[0],
        first_values=metric_values(pgd_method_name, "gamma_t", positive_only=False),
        first_method=pgd_method_name,
        second_values=metric_values(contour_method_name, "gamma_t", positive_only=False),
        second_method=contour_method_name,
        label_map=gamma_label_map,
        legend_loc="upper center",
        xlabel=r"$\gamma_t$",
        lower=0.0,
        upper=1.0,
    )
    draw_density(
        ax=axes[1],
        first_values=metric_values(pgd_method_name, "pgd_perturbation_norm", positive_only=True),
        first_method=pgd_method_name,
        second_values=metric_values(contour_method_name, "random_contour_perturbation_norm", positive_only=True),
        second_method=contour_method_name,
        label_map=perturbation_label_map,
        legend_loc="upper right",
        xlabel=r"Perturbation norm $\|o_t^{adv} - o_t\|_2$",
        lower=0.0,
        upper=None,
    )
    axes[0].text(
        0.03,
        0.96,
        rf"$\epsilon = {100.0 * float(coverage):.0f}\%$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=11.4,
        color="#31424F",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "alpha": 0.94,
            "edgecolor": "#BFC8D0",
        },
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run_episode_batch(job: tuple) -> list[MethodEpisodeResult]:
    """Run independent seeded episodes in a worker, preserving serial ordering.

    Each batch loads fixed weights and creates fresh environments. All rollout
    randomness is already local to the episode seed, so worker scheduling does
    not alter the numerical experiment. A small batch amortizes checkpoint IO.
    The explicit horizon is 500 in production and can be 3 in a smoke check.
    """
    config, method, seeds, max_steps = job
    torch.set_num_threads(1)
    model, ssm = load_benchmark_policy(config)
    R, _ = cartpole_mod.build_filter_covariances(
        meas_std=config.kf_meas_std, proc_std=config.kf_proc_std,
        meas_corr=config.kf_meas_corr, proc_corr=config.kf_proc_corr,
    )
    results = []
    try:
        for seed in seeds:
            env = gym.make("CartPole-v1", max_episode_steps=max_steps)
            try:
                if method.filter_type == "clean":
                    result = rollout_clean_episode(env, model, seed=seed, ssm=ssm)
                else:
                    result = rollout_benchmark_episode(
                        env, model, method, seed=seed, ssm=ssm, R=R,
                        config=config, device="cpu",
                    )
                results.append(result)
            finally:
                env.close()
    finally:
        if model.get_env() is not None:
            model.get_env().close()
    return results


def run_full_benchmark(
    *,
    config: BenchmarkConfig,
    output_prefix: str,
    workers: int = 1,
) -> dict[str, str]:
    """Run the full CartPole defense benchmark and save all artifacts."""
    data_dir = data_dir_for(CURRENT_DIR)
    figures_dir = figures_dir_for(CURRENT_DIR)
    summary_csv_path = os.path.join(data_dir, f"{output_prefix}_summary.csv")
    episodes_csv_path = os.path.join(data_dir, f"{output_prefix}_episodes.csv")
    diagnostics_csv_path = os.path.join(data_dir, f"{output_prefix}_diagnostics.csv")
    results_npz_path = os.path.join(data_dir, f"{output_prefix}.npz")
    returns_figure_path = os.path.join(figures_dir, f"{output_prefix}_returns.png")
    diagnostics_figure_path = os.path.join(figures_dir, f"{output_prefix}_dynamics.png")

    model, ssm = load_benchmark_policy(config)
    R, legacy_Q = cartpole_mod.build_filter_covariances(
        meas_std=config.kf_meas_std,
        proc_std=config.kf_proc_std,
        meas_corr=config.kf_meas_corr,
        proc_corr=config.kf_proc_corr,
    )
    # Complete selection first. The evaluation pool below cannot start before
    # the compatible WoLF selection JSON has been produced and loaded.
    with ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None) as executor:
        wolf_payload, wolf_artifact_paths = ensure_wolf_params_json(
            config=config, model=model, ssm=ssm, R=R, executor=executor,
        )
    methods = build_benchmark_methods(config, wolf_params=wolf_payload)
    all_results: list[MethodEpisodeResult] = []

    print(
        "[cartpole_defense_benchmark] "
        f"delta={config.discount_delta:.4f} | "
        f"coverage={config.coverage:.2f} | "
        f"epsilon={config.attack_eps:.4f} | "
        f"n_episodes={config.n_episodes} | "
        f"n_methods={len(methods)}"
    )
    print(
        "[cartpole_defense_benchmark] "
        "delta=0.94 is retained from the earlier calibration; it has not been "
        "recalibrated for the centered policy and physical stops."
    )
    if "wolf_imq" in wolf_payload and "wolf_tmd" in wolf_payload:
        print(
            "[cartpole_defense_benchmark] "
            "scenario-specific WoLF settings: "
            + " | ".join(
                f"{key}={wolf_payload[key]['selected_parameters']}"
                for key in ("wolf_imq_pgd", "wolf_tmd_pgd", "wolf_imq_random", "wolf_tmd_random")
            )
        )

    with ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None) as executor:
        for method_index, method in enumerate(methods, start=1):
            print(f"[cartpole_defense_benchmark] method {method_index}/{len(methods)} | {method.name}")
            episode_indices = range(int(config.n_episodes))
            if executor is not None:
                seeds = [int(config.seed0) + idx for idx in episode_indices]
                jobs = [(config, method, seeds[start:start + 5], 500) for start in range(0, len(seeds), 5)]
                completed = 0
                for batch in executor.map(run_episode_batch, jobs):
                    all_results.extend(batch)
                    completed += len(batch)
                    if completed % 25 == 0 or completed == len(seeds):
                        print(f"[cartpole_defense_benchmark] {method.name} | episode {completed}/{len(seeds)}", flush=True)
                episode_indices = ()
            for episode_index in episode_indices:
                seed = int(config.seed0) + episode_index
                env = gym.make("CartPole-v1")
                if method.filter_type == "clean":
                    result = rollout_clean_episode(env, model, seed=seed, ssm=ssm)
                else:
                    result = rollout_benchmark_episode(
                        env,
                        model,
                        method,
                        seed=seed,
                        ssm=ssm,
                        R=R,
                        config=config,
                        device="cpu",
                    )
                all_results.append(result)
                env.close()
                if episode_index == 0 or episode_index + 1 == int(config.n_episodes):
                    print(
                        "[cartpole_defense_benchmark] "
                        f"{method.name} | episode {episode_index + 1}/{config.n_episodes}"
                    )

    summary_df = aggregate_benchmark_results(all_results)
    diagnostics_df = diagnostics_to_dataframe(all_results)
    episodes_df = build_episode_dataframe(all_results)

    method_order = {method.name: idx for idx, method in enumerate(methods)}
    summary_df["_order"] = summary_df["method"].map(method_order)
    summary_df = summary_df.sort_values("_order").drop(columns="_order")
    episodes_df["_order"] = episodes_df["method"].map(method_order)
    episodes_df = episodes_df.sort_values(["_order", "episode_seed"]).drop(columns="_order")
    diagnostics_df["_order"] = diagnostics_df["method"].map(method_order)
    diagnostics_df = diagnostics_df.sort_values(["_order", "episode_seed", "step_index"]).drop(columns="_order")

    summary_df.to_csv(summary_csv_path, index=False)
    episodes_df.to_csv(episodes_csv_path, index=False)
    diagnostics_df.to_csv(diagnostics_csv_path, index=False)

    np.savez_compressed(
        results_npz_path,
        experiment_json=np.asarray(json.dumps(benchmark_metadata(config), sort_keys=True)),
        summary_columns=np.asarray(summary_df.columns.tolist(), dtype=object),
        summary_values=summary_df.to_numpy(dtype=object),
        episode_columns=np.asarray(episodes_df.columns.tolist(), dtype=object),
        episode_values=episodes_df.to_numpy(dtype=object),
        diagnostics_columns=np.asarray(diagnostics_df.columns.tolist(), dtype=object),
        diagnostics_values=diagnostics_df.to_numpy(dtype=object),
        coverage=np.asarray([config.coverage], dtype=float),
        attack_eps=np.asarray([config.attack_eps], dtype=float),
        attack_objective=np.asarray("nominal_kf_bellman_hard_actions_v1"),
        policy_temperature=np.asarray([config.policy_temperature], dtype=float),
        discount_delta=np.asarray([config.discount_delta], dtype=float),
        attack_probability=np.asarray([config.attack_prob], dtype=float),
        lambdas=np.asarray(config.lambdas, dtype=float),
        gamma_threshold=np.asarray([config.gamma_threshold], dtype=float),
        obs_noise_std=np.asarray(config.obs_noise_std, dtype=float),
        R=np.asarray(R, dtype=float),
        legacy_Q_reference=np.asarray(legacy_Q, dtype=float),
    )

    plot_benchmark_results(summary_df=summary_df, coverage=float(config.coverage), output_path=returns_figure_path)
    plot_benchmark_dynamics(
        diagnostics_df=diagnostics_df,
        coverage=float(config.coverage),
        output_path=diagnostics_figure_path,
    )

    print(f"[cartpole_defense_benchmark] saved summary: {summary_csv_path}")
    print(f"[cartpole_defense_benchmark] saved diagnostics: {diagnostics_csv_path}")
    print(f"[cartpole_defense_benchmark] saved figures: {returns_figure_path} | {diagnostics_figure_path}")
    return {
        **wolf_artifact_paths,
        "summary_csv_path": summary_csv_path,
        "episodes_csv_path": episodes_csv_path,
        "diagnostics_csv_path": diagnostics_csv_path,
        "results_npz_path": results_npz_path,
        "returns_figure_path": returns_figure_path,
        "diagnostics_figure_path": diagnostics_figure_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse the benchmark command-line arguments."""
    parser = argparse.ArgumentParser(description="CartPole defense benchmark.")
    parser.add_argument("--mode", choices=("benchmark", "wolf"), default="benchmark")
    parser.add_argument("--n-tuning-episodes", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--discount-delta", type=float, default=None)
    parser.add_argument("--pgd-steps", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Override the default config with command-line arguments."""
    config = build_default_config()
    override_dict: dict[str, Any] = {}
    if args.n_tuning_episodes is not None:
        override_dict["n_tuning_episodes"] = int(args.n_tuning_episodes)
    if args.n_episodes is not None:
        override_dict["n_episodes"] = int(args.n_episodes)
    if args.coverage is not None:
        coverage = float(args.coverage)
        override_dict["coverage"] = coverage
        override_dict["attack_eps"] = float(chi2.ppf(coverage, df=4))
    if args.discount_delta is not None:
        override_dict["discount_delta"] = float(args.discount_delta)
    if args.pgd_steps is not None:
        override_dict["pgd_steps"] = int(args.pgd_steps)
    if args.mc_samples is not None:
        override_dict["mc_samples"] = int(args.mc_samples)
    if not override_dict:
        return config
    return BenchmarkConfig(**{**config.__dict__, **override_dict})


def main() -> None:
    """Run the CartPole defense benchmark and print the artifact paths."""
    args = parse_args()
    torch.set_num_threads(1)
    workers = 8 if args.workers is None else args.workers
    if workers < 1:
        raise ValueError("workers must be at least one")
    config = config_from_args(args)
    if args.mode == "wolf":
        output_prefix = args.output_prefix or default_wolf_output_prefix(config)
        artifact_paths = run_wolf_only(config=config, output_prefix=output_prefix, workers=workers)
    else:
        output_prefix = args.output_prefix or default_output_prefix(config)
        artifact_paths = run_full_benchmark(config=config, output_prefix=output_prefix, workers=workers)
    for label, path in artifact_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
