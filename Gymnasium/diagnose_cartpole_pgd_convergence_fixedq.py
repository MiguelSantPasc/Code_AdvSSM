#!/usr/bin/env python3
"""
diagnose_cartpole_pgd_convergence_fixedq.py

Inspect PGD convergence for the main CartPole attack geometries.

Why this script exists:
1. The repository now uses a calibrated discounted predictor in the main
   Gymnasium benchmark, but we still want to compare it against the older
   fixed-`Q` attack setup directly.
2. A recurring practical question is whether the projected PGD attack really
   converges and whether its final iterate ends up close to the ellipsoid
   boundary instead of stopping well inside the feasible set.
3. This script freezes a short batch of attack cases and reruns PGD with
   several `(pgd_steps, pgd_step_size)` pairs on exactly the same cases.

What this script reports:
1. `mean_eps_adv`, the average ellipsoidal radius achieved by the best PGD
   iterate.
2. `mean_gap`, the average slack `epsilon - eps_adv`, so smaller means closer
   to the ellipsoid boundary.
3. `pct_boundary`, the percentage of best iterates whose slack is below a
   small tolerance. We intentionally keep that tolerance visible in `main()`
   because the achieved ellipsoidal radius is computed numerically and may show
   a small approximation error relative to the target `epsilon`.
4. `mean_obj_impr`, the average improvement in the critic objective relative
   to the initial projected observation.

Important setup choice:
1. This script compares both the older fixed-`Q` predictor and the current
   fixed-discount predictor.
2. The goal here is not to compare defenses, but to sanity-check the PGD
   optimizer under the attack geometries that matter for CartPole.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import cartpole_covadapt_compare_epsilons_wolf as cartpole_mod


@dataclass(frozen=True)
class AttackCase:
    """Store one fixed attack instance used across all PGD settings."""

    case_id: int
    episode_idx: int
    epsilon: float
    obs_nom: np.ndarray
    attack_center: np.ndarray
    P_pred: np.ndarray
    R: np.ndarray
    attack_sigma: np.ndarray
    rng_seed: int


@dataclass
class ConfigSummary:
    """Accumulate convergence diagnostics for one PGD configuration."""

    count: int = 0
    eps_adv_sum: float = 0.0
    gap_sum: float = 0.0
    obj_improvement_sum: float = 0.0
    on_boundary: int = 0


@dataclass(frozen=True)
class PredictorGeometry:
    """Describe one CartPole predictor geometry to diagnose."""

    name: str
    display_name: str
    discount_delta: float | None
    use_legacy_q: bool


def quadratic_radius_sq(
    *,
    vector: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """Return `v^T Sigma^{-1} v` using the repository SPD solver."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    Sigma = cartpole_mod.project_to_psd(np.asarray(Sigma, dtype=float))
    return float(np.dot(vector, cartpole_mod.solve_spd(Sigma, vector)))


def achieved_attack_epsilon(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """Return the ellipsoidal radius achieved by one observation."""
    delta = np.asarray(observation, dtype=float) - np.asarray(center, dtype=float)
    return quadratic_radius_sq(vector=delta, Sigma=Sigma)


def build_legacy_fixed_q_covariances() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the original fixed-`Q` CartPole filter settings.

    These are the same numbers that were used by the older inspection script
    before the discounted predictor became the shared Gymnasium default.
    """
    kf_meas_std = np.array([0.080, 0.180, 0.040, 0.180], dtype=np.float32)
    kf_proc_std = np.array([0.320, 0.720, 0.160, 0.720], dtype=np.float32)

    kf_meas_corr = np.array(
        [
            [1.00, 0.22, 0.08, 0.00],
            [0.22, 1.00, 0.18, 0.28],
            [0.08, 0.18, 1.00, 0.24],
            [0.00, 0.28, 0.24, 1.00],
        ],
        dtype=np.float32,
    )
    kf_proc_corr = np.array(
        [
            [1.00, 0.40, 0.16, 0.00],
            [0.40, 1.00, 0.30, 0.42],
            [0.16, 0.30, 1.00, 0.34],
            [0.00, 0.42, 0.34, 1.00],
        ],
        dtype=np.float32,
    )
    return kf_meas_std, kf_proc_std, kf_meas_corr, kf_proc_corr


def expected_objective_at_observation(
    *,
    model: cartpole_mod.DQN,
    observation: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    xi_torch: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the posterior expected critic value at one attacked observation."""
    obs_t = torch.tensor(np.asarray(observation, dtype=np.float32), dtype=torch.float32, device=device)
    mu_value_t, m_post_np, P_post_np = cartpole_mod.expected_critic_value_mc(
        model=model,
        obs_adv_torch=obs_t,
        m_pred=m_pred,
        P_pred=P_pred,
        R=R,
        xi_torch=xi_torch,
    )
    return float(mu_value_t.detach().cpu().item()), m_post_np, P_post_np


def pgd_attack_with_trace(
    *,
    model: cartpole_mod.DQN,
    obs_nom: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    attack_center: np.ndarray,
    attack_sigma: np.ndarray,
    attack_eps: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    rng_seed: int,
    device: str = "cpu",
) -> dict[str, float | np.ndarray]:
    """
    Run projected PGD and return concise convergence diagnostics.

    The best iterate is tracked exactly as in the shared attack code, but this
    helper also records the initial objective and the final boundary slack so
    we can compare optimizer settings directly.
    """
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng_seed))
    xi_torch = torch.randn((mc_samples, 4), generator=gen, device=dev, dtype=torch.float32)

    center = np.asarray(attack_center, dtype=np.float32).copy()
    obs_curr_np = cartpole_mod.project_to_attack_region(
        y_candidate=np.asarray(obs_nom, dtype=np.float32),
        center=center,
        Sigma=attack_sigma,
        epsilon=attack_eps,
    )

    init_obj, _m_post_init, _P_post_init = expected_objective_at_observation(
        model=model,
        observation=obs_curr_np,
        m_pred=m_pred,
        P_pred=P_pred,
        R=R,
        xi_torch=xi_torch,
        device=dev,
    )
    init_eps = achieved_attack_epsilon(
        observation=obs_curr_np,
        center=center,
        Sigma=attack_sigma,
    )

    best_obs = obs_curr_np.copy()
    best_obj = float(init_obj)
    best_m_post = None
    best_P_post = None

    for _ in range(int(pgd_steps)):
        obs_t = torch.tensor(obs_curr_np, dtype=torch.float32, device=dev, requires_grad=True)
        mu_value_t, m_post_np, P_post_np = cartpole_mod.expected_critic_value_mc(
            model=model,
            obs_adv_torch=obs_t,
            m_pred=m_pred,
            P_pred=P_pred,
            R=R,
            xi_torch=xi_torch,
        )
        mu_value_t.backward()
        grad = obs_t.grad.detach().cpu().numpy().astype(np.float32)

        obj_val = float(mu_value_t.detach().cpu().item())
        if obj_val < best_obj:
            best_obj = obj_val
            best_obs = obs_curr_np.copy()
            best_m_post = m_post_np.copy()
            best_P_post = P_post_np.copy()

        obs_next = obs_curr_np - float(pgd_step_size) * grad
        obs_curr_np = cartpole_mod.project_to_attack_region(
            y_candidate=obs_next,
            center=center,
            Sigma=attack_sigma,
            epsilon=attack_eps,
        ).astype(np.float32)

    final_obj, m_post_np, P_post_np = expected_objective_at_observation(
        model=model,
        observation=obs_curr_np,
        m_pred=m_pred,
        P_pred=P_pred,
        R=R,
        xi_torch=xi_torch,
        device=dev,
    )
    if final_obj < best_obj:
        best_obj = float(final_obj)
        best_obs = obs_curr_np.copy()
        best_m_post = m_post_np.copy()
        best_P_post = P_post_np.copy()

    if best_m_post is None or best_P_post is None:
        _obj_tmp, best_m_post, best_P_post = expected_objective_at_observation(
            model=model,
            observation=best_obs,
            m_pred=m_pred,
            P_pred=P_pred,
            R=R,
            xi_torch=xi_torch,
            device=dev,
        )

    best_eps = achieved_attack_epsilon(
        observation=best_obs,
        center=center,
        Sigma=attack_sigma,
    )
    return {
        "init_obj": float(init_obj),
        "best_obj": float(best_obj),
        "init_eps": float(init_eps),
        "best_eps": float(best_eps),
        "best_obs": best_obs.astype(np.float32),
        "best_m_post": best_m_post.astype(np.float32),
        "best_P_post": best_P_post.astype(np.float32),
    }


def collect_attack_cases(
    *,
    geometry: PredictorGeometry,
    seed0: int,
    n_episodes: int,
    max_steps_per_episode: int,
    epsilon_values: tuple[float, ...],
) -> tuple[list[AttackCase], np.ndarray, np.ndarray]:
    """
    Build a shared batch of attack cases under one chosen predictor geometry.
    """
    kf_meas_std, kf_proc_std, kf_meas_corr, kf_proc_corr = build_legacy_fixed_q_covariances()
    R, Q = cartpole_mod.build_filter_covariances(
        meas_std=kf_meas_std,
        proc_std=kf_proc_std,
        meas_corr=kf_meas_corr,
        proc_corr=kf_proc_corr,
    )

    model_path = cartpole_mod.ensure_downloaded_cartpole_checkpoint()
    device = torch.device("cpu")
    model = cartpole_mod.load_cartpole_policy(model_path, device)
    ssm = cartpole_mod.build_cartpole_linear_ssm()

    attack_cases: list[AttackCase] = []
    case_id = 0

    for episode_idx in range(int(n_episodes)):
        seed = int(seed0 + episode_idx)
        env = gym.make("CartPole-v1")
        obs, _info = env.reset(seed=seed)

        rng_nominal_noise = np.random.default_rng(seed + 707_002)
        init_noise = rng_nominal_noise.normal(0.0, kf_meas_std, size=(4,)).astype(np.float32)
        obs_init_noisy = (np.asarray(obs, dtype=np.float32) + init_noise).astype(np.float32)

        m_post, P_post = cartpole_mod.kf_update_state(
            m_pred=np.asarray(obs, dtype=np.float32),
            P_pred=cartpole_mod.project_to_psd(R.copy()),
            y_obs=obs_init_noisy,
            R=R,
        )
        action = cartpole_mod.select_action(model, m_post)
        force = cartpole_mod.action_to_force(action, ssm.force_mag)
        m_pred, P_pred = cartpole_mod.kf_predict_state(
            m_post=m_post,
            P_post=P_post,
            force=force,
            Q=Q if geometry.use_legacy_q else None,
            discount_delta=geometry.discount_delta,
            ssm=ssm,
        )

        obs, reward, terminated, truncated, _info = env.step(action)
        _ = reward
        if terminated or truncated:
            env.close()
            continue

        step_idx = 1
        while step_idx <= int(max_steps_per_episode):
            y_clean = np.asarray(obs, dtype=np.float32)
            nominal_noise = rng_nominal_noise.normal(0.0, kf_meas_std, size=(4,)).astype(np.float32)
            o_real = (y_clean + nominal_noise).astype(np.float32)
            attack_center = np.asarray(m_pred, dtype=np.float32)
            attack_sigma = cartpole_mod.project_to_psd(np.asarray(P_pred, dtype=float) + np.asarray(R, dtype=float))

            for eps_idx, epsilon in enumerate(epsilon_values):
                attack_cases.append(
                    AttackCase(
                        case_id=case_id,
                        episode_idx=int(episode_idx),
                        epsilon=float(epsilon),
                        obs_nom=o_real.copy(),
                        attack_center=attack_center.copy(),
                        P_pred=np.asarray(P_pred, dtype=np.float32).copy(),
                        R=np.asarray(R, dtype=np.float32).copy(),
                        attack_sigma=np.asarray(attack_sigma, dtype=np.float32).copy(),
                        rng_seed=seed + 10_000 * step_idx + eps_idx,
                    )
                )
                case_id += 1

            m_post, P_post = cartpole_mod.kf_update_state(
                m_pred=attack_center,
                P_pred=P_pred,
                y_obs=o_real,
                R=R,
            )
            action = cartpole_mod.select_action(model, m_post)
            force = cartpole_mod.action_to_force(action, ssm.force_mag)
            m_pred, P_pred = cartpole_mod.kf_predict_state(
                m_post=m_post,
                P_post=P_post,
                force=force,
                Q=Q if geometry.use_legacy_q else None,
                discount_delta=geometry.discount_delta,
                ssm=ssm,
            )
            obs, reward, terminated, truncated, _info = env.step(action)
            _ = reward
            step_idx += 1

            if terminated or truncated:
                break

        env.close()

    return attack_cases, R, Q


def summarize_pgd_grid(
    *,
    geometry_label: str,
    attack_cases: list[AttackCase],
    pgd_steps_values: tuple[int, ...],
    pgd_step_sizes: tuple[float, ...],
    mc_samples: int,
    boundary_tol: float,
) -> dict[tuple[int, float], ConfigSummary]:
    """Run every PGD configuration on the same case batch."""
    model_path = cartpole_mod.ensure_downloaded_cartpole_checkpoint()
    device = torch.device("cpu")
    model = cartpole_mod.load_cartpole_policy(model_path, device)

    summaries: dict[tuple[int, float], ConfigSummary] = {
        (int(pgd_steps), float(step_size)): ConfigSummary()
        for pgd_steps in pgd_steps_values
        for step_size in pgd_step_sizes
    }

    for pgd_steps in pgd_steps_values:
        for step_size in pgd_step_sizes:
            summary = summaries[(int(pgd_steps), float(step_size))]
            episode_boundary_hits = 0
            episode_case_count = 0
            current_episode_idx = None

            for case in attack_cases:
                if current_episode_idx is None:
                    current_episode_idx = int(case.episode_idx)
                elif int(case.episode_idx) != current_episode_idx:
                    pct_boundary_episode = 100.0 * float(episode_boundary_hits) / max(episode_case_count, 1)
                    print(
                        f"[pgd-progress] geometry={geometry_label} | "
                        f"pgd_steps={int(pgd_steps)} | "
                        f"step_size={float(step_size):.3f} | "
                        f"episode={current_episode_idx + 1} | "
                        f"pct_boundary={pct_boundary_episode:.2f}%"
                    )
                    current_episode_idx = int(case.episode_idx)
                    episode_boundary_hits = 0
                    episode_case_count = 0

                trace = pgd_attack_with_trace(
                    model=model,
                    obs_nom=case.obs_nom,
                    m_pred=case.attack_center,
                    P_pred=case.P_pred,
                    R=case.R,
                    attack_center=case.attack_center,
                    attack_sigma=case.attack_sigma,
                    attack_eps=float(case.epsilon),
                    pgd_steps=int(pgd_steps),
                    pgd_step_size=float(step_size),
                    mc_samples=mc_samples,
                    rng_seed=int(case.rng_seed),
                    device="cpu",
                )
                best_eps = float(trace["best_eps"])
                gap = max(float(case.epsilon) - best_eps, 0.0)
                summary.count += 1
                summary.eps_adv_sum += best_eps
                summary.gap_sum += gap
                summary.obj_improvement_sum += float(trace["init_obj"]) - float(trace["best_obj"])
                on_boundary = int(gap <= float(boundary_tol))
                summary.on_boundary += on_boundary
                episode_boundary_hits += on_boundary
                episode_case_count += 1

            if current_episode_idx is not None:
                pct_boundary_episode = 100.0 * float(episode_boundary_hits) / max(episode_case_count, 1)
                print(
                    f"[pgd-progress] geometry={geometry_label} | "
                    f"pgd_steps={int(pgd_steps)} | "
                    f"step_size={float(step_size):.3f} | "
                    f"episode={current_episode_idx + 1} | "
                    f"pct_boundary={pct_boundary_episode:.2f}%"
                )

    return summaries


def print_grid_report(
    *,
    geometry: PredictorGeometry,
    attack_cases: list[AttackCase],
    pgd_steps_values: tuple[int, ...],
    pgd_step_sizes: tuple[float, ...],
    summaries: dict[tuple[int, float], ConfigSummary],
    boundary_tol: float,
    epsilon_display_values: tuple[float, ...] | None,
    R: np.ndarray,
    Q: np.ndarray,
) -> None:
    """Print one compact convergence table for the PGD grid."""
    sorted_epsilons = tuple(sorted({float(case.epsilon) for case in attack_cases}))
    if epsilon_display_values is None:
        epsilon_display_values = sorted_epsilons
    print(f"CartPole PGD convergence on {geometry.display_name}")
    print(f"R_trace = {float(np.trace(np.asarray(R, dtype=float))):.4f}")
    if geometry.use_legacy_q:
        print(f"legacy_Q_trace = {float(np.trace(np.asarray(Q, dtype=float))):.4f}")
    else:
        print(f"discount_delta = {float(geometry.discount_delta):.2f}")
    print(
        f"cases={len(attack_cases)}  epsilons_display={tuple(float(value) for value in epsilon_display_values)}  "
        f"epsilons_internal={sorted_epsilons}  "
        f"boundary_tol={boundary_tol:.1e}"
    )
    print("-" * 102)
    print("pgd_steps   step_size   mean_eps_adv   mean_gap   pct_boundary   mean_obj_impr")
    print("-" * 102)

    best_key = None
    best_gap = np.inf
    for pgd_steps in pgd_steps_values:
        for step_size in pgd_step_sizes:
            summary = summaries[(int(pgd_steps), float(step_size))]
            n = max(summary.count, 1)
            mean_gap = summary.gap_sum / n
            if mean_gap < best_gap:
                best_gap = float(mean_gap)
                best_key = (int(pgd_steps), float(step_size))

            print(
                f"{int(pgd_steps):>9d} "
                f"{float(step_size):>11.3f} "
                f"{summary.eps_adv_sum / n:>14.4f} "
                f"{mean_gap:>10.4f} "
                f"{100.0 * summary.on_boundary / n:>13.2f}% "
                f"{summary.obj_improvement_sum / n:>15.4f}"
            )

    if best_key is not None:
        best_summary = summaries[best_key]
        n = max(best_summary.count, 1)
        print("-" * 102)
        print(
            "best_boundary_cfg = "
            f"(pgd_steps={best_key[0]}, step_size={best_key[1]:.3f})  "
            f"mean_gap={best_summary.gap_sum / n:.4f}  "
            f"pct_boundary={100.0 * best_summary.on_boundary / n:.2f}%"
        )


def inspect_pgd_convergence_geometries() -> None:
    """
    Compare several PGD settings under the two CartPole predictor geometries.
    """
    seed0 = 7
    n_episodes = 2
    max_steps_per_episode = 6
    # Focus the diagnosis on the 95% predictive ellipsoid only so we can check
    # whether PGD converges cleanly on the harder CartPole attack radius.
    epsilon_values = (9.49,)
    epsilon_display_values = (0.95,)
    pgd_steps_values = (10, 20, 40, 80)
    pgd_step_sizes = (0.05, 0.10, 0.20, 0.35)
    mc_samples = 48
    # Treat best iterates within 0.1 ellipsoidal-radius units of the boundary
    # as converged for this diagnostic pass.
    boundary_tol = 0.1

    print(f"seed0 = {seed0}")
    print(f"n_episodes = {n_episodes}")
    print(f"max_steps_per_episode = {max_steps_per_episode}")
    print(f"mc_samples = {mc_samples}")

    geometry_list = (
        PredictorGeometry(
            name="legacy_fixed_q",
            display_name="legacy fixed-Q attack geometry",
            discount_delta=None,
            use_legacy_q=True,
        ),
        PredictorGeometry(
            name="discounted_delta_0p86",
            display_name="discounted attack geometry",
            discount_delta=cartpole_mod.DEFAULT_GYMNASIUM_DISCOUNT_DELTA,
            use_legacy_q=False,
        ),
    )

    for geometry in geometry_list:
        attack_cases, R, Q = collect_attack_cases(
            geometry=geometry,
            seed0=seed0,
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps_per_episode,
            epsilon_values=epsilon_values,
        )
        summaries = summarize_pgd_grid(
            geometry_label=geometry.display_name,
            attack_cases=attack_cases,
            pgd_steps_values=pgd_steps_values,
            pgd_step_sizes=pgd_step_sizes,
            mc_samples=mc_samples,
            boundary_tol=boundary_tol,
        )
        print_grid_report(
            geometry=geometry,
            attack_cases=attack_cases,
            pgd_steps_values=pgd_steps_values,
            pgd_step_sizes=pgd_step_sizes,
            summaries=summaries,
            boundary_tol=boundary_tol,
            epsilon_display_values=epsilon_display_values,
            R=R,
            Q=Q,
        )
        print()


def main() -> None:
    """Entry point for the CartPole PGD convergence comparison."""
    inspect_pgd_convergence_geometries()


if __name__ == "__main__":
    main()
