#!/usr/bin/env python3
"""
calibrate_cartpole_discounted_covariances.py

Inspect whether the CartPole EKF geometry becomes better calibrated when the
predictive covariance is inflated with the classical discount principle.

Why this script exists:
1. The current `summarize_cartpole_attack_geometry.py` uses a fixed process
   covariance `Q`, which is useful, but it does not tell us directly whether a
   discount-factor construction would calibrate the predictive ellipsoid more
   naturally.
2. In the discount view, we first propagate the posterior covariance through
   the local CartPole Jacobian,

      C_t = A_t P_{t-1|t-1} A_t^T,

   and then inflate it with a discount factor `delta in (0, 1]`,

      P_{t|t-1} = C_t / delta,
      Q_t^(discount) = P_{t|t-1} - C_t = ((1-delta)/delta) C_t.

3. When `delta < 1`, the filter becomes less overconfident in exactly the
   directions that the local dynamics consider uncertain, instead of adding one
   fixed `Q` everywhere.

What this script reports:
1. A baseline that reproduces the current inspection style with a fixed `Q`.
2. Several fixed-discount filters such as `delta = 0.99, 0.97, 0.95, 0.90`.
3. One adaptive diagnostic mode that chooses the largest `delta_t` whose
   predictive innovation radius is at most a target value. This adaptive mode
   is meant for inspection and calibration, not as a final causal defense.
4. For each mode it prints:
   - the mean and median normalized innovation squared,
   - how often the nominal noisy observation falls inside several ellipsoids,
   - the average effective discount,
   - the average trace of the induced discount covariance `Q_t`.

Important comparison convention:
1. All modes see the same CartPole trajectory and the same noisy observations.
2. The environment is driven by the pretrained DQN policy evaluated on the
   clean state `s_t`, so the comparison isolates covariance calibration rather
   than policy drift.
3. Good calibration should push the mean normalized innovation squared closer
   to the state dimension `4`, while increasing the fraction of observations
   that fall inside reasonable ellipsoidal radii.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for import_path in (CURRENT_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import cartpole_covadapt_compare_epsilons_wolf as cartpole_mod

try:
    from AdvSSM.io_utils import data_dir_for, save_npz
except ModuleNotFoundError:
    from io_utils import data_dir_for, save_npz


@dataclass(frozen=True)
class ModeConfig:
    """Describe one covariance-calibration mode inspected by the script."""

    name: str
    kind: str
    delta: float | None = None
    target_radius: float | None = None
    delta_min: float | None = None
    delta_max: float | None = None
    fixed_q: np.ndarray | None = None


@dataclass
class CalibrationSummary:
    """Accumulate scalar calibration diagnostics for one mode."""

    count: int = 0
    eps_real_values: list[float] = field(default_factory=list)
    delta_values: list[float] = field(default_factory=list)
    trace_q_values: list[float] = field(default_factory=list)
    trace_p_values: list[float] = field(default_factory=list)
    inside_counts: dict[float, int] = field(default_factory=dict)


def quadratic_radius_sq(
    *,
    vector: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """Return `v^T Sigma^{-1} v` using the shared SPD solver."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    Sigma = cartpole_mod.project_to_psd(np.asarray(Sigma, dtype=float))
    return float(np.dot(vector, cartpole_mod.solve_spd(Sigma, vector)))


def achieved_attack_epsilon(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    Sigma: np.ndarray,
) -> float:
    """Return the ellipsoidal radius reached by one observation."""
    delta = np.asarray(observation, dtype=float) - np.asarray(center, dtype=float)
    return quadratic_radius_sq(vector=delta, Sigma=Sigma)


def build_correlated_filter_covariances() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the CartPole noise settings used by the current inspection scripts.

    The fixed-`Q` baseline reuses these same values so the new discount-based
    inspection stays directly comparable to the existing script.
    """
    kf_meas_std = np.array([0.100, 0.220, 0.050, 0.220], dtype=np.float32)
    kf_proc_std = np.array([0.320, 0.720, 0.160, 0.720], dtype=np.float32)

    kf_meas_corr = np.array(
        [
            [1.00, 0.18, 0.06, 0.00],
            [0.18, 1.00, 0.14, 0.22],
            [0.06, 0.14, 1.00, 0.18],
            [0.00, 0.22, 0.18, 1.00],
        ],
        dtype=np.float32,
    )
    kf_proc_corr = np.array(
        [
            [1.00, 0.24, 0.08, 0.00],
            [0.24, 1.00, 0.16, 0.26],
            [0.08, 0.16, 1.00, 0.22],
            [0.00, 0.26, 0.22, 1.00],
        ],
        dtype=np.float32,
    )
    return kf_meas_std, kf_proc_std, kf_meas_corr, kf_proc_corr


def build_mode_list(
    *,
    baseline_q: np.ndarray,
    fixed_discount_values: tuple[float, ...],
    adaptive_target_radius: float,
    adaptive_delta_min: float,
    adaptive_delta_max: float,
) -> list[ModeConfig]:
    """Return the baseline, fixed-discount, and adaptive-discount modes."""
    mode_list = [
        ModeConfig(
            name="fixed_q_inspect",
            kind="fixed_q",
            fixed_q=cartpole_mod.project_to_psd(np.asarray(baseline_q, dtype=float)),
        )
    ]

    for delta in fixed_discount_values:
        mode_list.append(
            ModeConfig(
                name=f"discount_{delta:.2f}",
                kind="fixed_discount",
                delta=float(delta),
            )
        )

    mode_list.append(
        ModeConfig(
            name=f"adaptive_nis_{adaptive_target_radius:.1f}",
            kind="adaptive_discount",
            target_radius=float(adaptive_target_radius),
            delta_min=float(adaptive_delta_min),
            delta_max=float(adaptive_delta_max),
        )
    )
    return mode_list


def initialize_summary_dict(
    *,
    mode_list: list[ModeConfig],
    epsilon_values: tuple[float, ...],
) -> dict[str, CalibrationSummary]:
    """Create one empty summary container per mode."""
    return {
        mode.name: CalibrationSummary(
            inside_counts={float(epsilon): 0 for epsilon in epsilon_values},
        )
        for mode in mode_list
    }


def predict_nominal_transition(
    *,
    m_post: np.ndarray,
    P_post: np.ndarray,
    force: float,
    ssm: cartpole_mod.CartPoleLinearSSM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the EKF mean prediction and the Jacobian-propagated covariance part.

    This isolates the purely dynamical covariance term
    `C_t = A_t P_post A_t^T` before either a fixed `Q` or a discount inflation
    is applied.
    """
    m_post = np.asarray(m_post, dtype=float).reshape(4)
    P_post = cartpole_mod.project_to_psd(np.asarray(P_post, dtype=float))

    m_pred = cartpole_mod.cartpole_discrete_dynamics(
        state=m_post,
        force=force,
        ssm=ssm,
    )
    A_local, _B_local = cartpole_mod.linearize_cartpole_discrete_dynamics(
        state=m_post,
        force=force,
        ssm=ssm,
    )
    transition_cov = cartpole_mod.project_to_psd(A_local @ P_post @ A_local.T)
    return m_pred.astype(np.float32), transition_cov.astype(np.float32)


def discounted_predictive_covariance(
    *,
    transition_cov: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inflate `C_t` with the classical discount rule.

    The returned pair is `(P_pred, Q_discount)` where
        `P_pred = C_t / delta`
    and
        `Q_discount = P_pred - C_t`.
    """
    if not (0.0 < float(delta) <= 1.0):
        raise ValueError("delta must satisfy 0 < delta <= 1.")

    transition_cov = cartpole_mod.project_to_psd(np.asarray(transition_cov, dtype=float))
    P_pred = cartpole_mod.project_to_psd(transition_cov / float(delta))
    Q_discount = cartpole_mod.project_to_psd(P_pred - transition_cov)
    return P_pred.astype(np.float32), Q_discount.astype(np.float32)


def solve_discount_for_target_radius(
    *,
    innovation: np.ndarray,
    transition_cov: np.ndarray,
    R: np.ndarray,
    target_radius: float,
    delta_min: float,
    delta_max: float,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> tuple[float, float]:
    """
    Return the least-inflated discount whose innovation radius stays controlled.

    The function chooses the largest `delta` in `[delta_min, delta_max]` such
    that
        `innovation^T (transition_cov / delta + R)^(-1) innovation <= target`.
    This keeps the predictive covariance as sharp as possible while still
    satisfying the desired calibration target whenever feasible.
    """
    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    transition_cov = cartpole_mod.project_to_psd(np.asarray(transition_cov, dtype=float))
    R = cartpole_mod.project_to_psd(np.asarray(R, dtype=float))

    if not (0.0 < float(delta_min) <= float(delta_max) <= 1.0):
        raise ValueError("Discount bounds must satisfy 0 < delta_min <= delta_max <= 1.")

    target_radius = max(float(target_radius), 1e-8)

    def radius_for_delta(delta_value: float) -> float:
        sigma = cartpole_mod.project_to_psd(transition_cov / float(delta_value) + R)
        return quadratic_radius_sq(vector=innovation, Sigma=sigma)

    radius_at_max = radius_for_delta(float(delta_max))
    if radius_at_max <= target_radius:
        return float(delta_max), float(radius_at_max)

    radius_at_min = radius_for_delta(float(delta_min))
    if radius_at_min > target_radius:
        return float(delta_min), float(radius_at_min)

    lo = float(delta_min)
    hi = float(delta_max)
    chosen_radius = float(radius_at_min)

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        radius_mid = radius_for_delta(mid)

        # The radius increases with delta, so a feasible midpoint lets us keep
        # more confidence and move right.
        if radius_mid <= target_radius:
            lo = mid
            chosen_radius = float(radius_mid)
        else:
            hi = mid

        if hi - lo <= float(tol):
            break

    return float(lo), float(chosen_radius)


def select_mode_predictive_covariance(
    *,
    mode: ModeConfig,
    transition_cov: np.ndarray,
    innovation: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return `(P_pred, Q_eff, delta_used)` for one calibration mode.
    """
    transition_cov = cartpole_mod.project_to_psd(np.asarray(transition_cov, dtype=float))

    if mode.kind == "fixed_q":
        if mode.fixed_q is None:
            raise ValueError("fixed_q mode requires a covariance matrix.")
        q_eff = cartpole_mod.project_to_psd(np.asarray(mode.fixed_q, dtype=float))
        P_pred = cartpole_mod.project_to_psd(transition_cov + q_eff)
        return P_pred.astype(np.float32), q_eff.astype(np.float32), 1.0

    if mode.kind == "fixed_discount":
        if mode.delta is None:
            raise ValueError("fixed_discount mode requires `delta`.")
        P_pred, q_eff = discounted_predictive_covariance(
            transition_cov=transition_cov,
            delta=float(mode.delta),
        )
        return P_pred, q_eff, float(mode.delta)

    if mode.kind == "adaptive_discount":
        if (
            mode.target_radius is None
            or mode.delta_min is None
            or mode.delta_max is None
        ):
            raise ValueError("adaptive_discount mode requires target and bounds.")
        delta_used, _radius = solve_discount_for_target_radius(
            innovation=innovation,
            transition_cov=transition_cov,
            R=R,
            target_radius=float(mode.target_radius),
            delta_min=float(mode.delta_min),
            delta_max=float(mode.delta_max),
        )
        P_pred, q_eff = discounted_predictive_covariance(
            transition_cov=transition_cov,
            delta=delta_used,
        )
        return P_pred, q_eff, float(delta_used)

    raise ValueError(f"Unsupported mode kind: {mode.kind}")


def update_summary(
    *,
    summary: CalibrationSummary,
    eps_real: float,
    delta_used: float,
    q_eff: np.ndarray,
    P_pred: np.ndarray,
    epsilon_values: tuple[float, ...],
) -> None:
    """Accumulate one new calibration sample into the running summary."""
    summary.count += 1
    summary.eps_real_values.append(float(eps_real))
    summary.delta_values.append(float(delta_used))
    summary.trace_q_values.append(float(np.trace(np.asarray(q_eff, dtype=float))))
    summary.trace_p_values.append(float(np.trace(np.asarray(P_pred, dtype=float))))

    for epsilon in epsilon_values:
        summary.inside_counts[float(epsilon)] += int(float(eps_real) <= float(epsilon))


def print_summary_table(
    *,
    mode_list: list[ModeConfig],
    summaries: dict[str, CalibrationSummary],
    epsilon_values: tuple[float, ...],
    target_dimension: float,
) -> None:
    """Print a concise calibration table that is easy to scan."""
    epsilon_target = float(max(epsilon_values))
    print("-" * 104)
    print(
        "mode                 n    mean_nis   |mean_nis-4|   "
        f"pct<={epsilon_target:.2f}   mean_delta   mean_trQ"
    )
    print("-" * 104)

    for mode in mode_list:
        summary = summaries[mode.name]
        n = max(summary.count, 1)
        eps_values = np.asarray(summary.eps_real_values, dtype=float)
        delta_values = np.asarray(summary.delta_values, dtype=float)
        trace_q_values = np.asarray(summary.trace_q_values, dtype=float)
        mean_nis = float(eps_values.mean()) if eps_values.size else 0.0
        abs_gap = abs(mean_nis - float(target_dimension))
        inside_pct = 100.0 * summary.inside_counts[epsilon_target] / n

        print(
            f"{mode.name:<20} "
            f"{summary.count:>4d} "
            f"{mean_nis:>11.4f} "
            f"{abs_gap:>14.4f}   "
            f"{inside_pct:>9.2f}%   "
            f"{delta_values.mean() if delta_values.size else 0.0:>10.4f} "
            f"{trace_q_values.mean() if trace_q_values.size else 0.0:>10.4f}"
        )


def best_fixed_discount_mode(
    *,
    mode_list: list[ModeConfig],
    summaries: dict[str, CalibrationSummary],
    target_dimension: float,
) -> tuple[ModeConfig, float]:
    """Return the fixed-discount mode whose mean NIS is closest to target."""
    candidate_modes = [mode for mode in mode_list if mode.kind == "fixed_discount"]
    if not candidate_modes:
        raise ValueError("No fixed-discount modes were provided.")

    best_mode = candidate_modes[0]
    best_gap = np.inf

    for mode in candidate_modes:
        eps_values = np.asarray(summaries[mode.name].eps_real_values, dtype=float)
        mean_nis = float(eps_values.mean()) if eps_values.size else np.inf
        current_gap = abs(mean_nis - float(target_dimension))
        if current_gap < best_gap:
            best_mode = mode
            best_gap = current_gap

    return best_mode, float(best_gap)


def save_calibration_summary(
    *,
    outpath: str,
    scenario_tag: str,
    mode_list: list[ModeConfig],
    summaries: dict[str, CalibrationSummary],
    epsilon_values: tuple[float, ...],
    fixed_discount_values: tuple[float, ...],
    seed0: int,
    n_episodes: int,
    max_steps_per_episode: int,
    adaptive_target_radius: float,
    adaptive_delta_min: float,
    adaptive_delta_max: float,
    target_dimension: float,
    best_mode: ModeConfig,
    best_gap: float,
) -> None:
    """Save the discounted-covariance calibration summary as one NPZ file."""
    payload: dict[str, object] = {
        "scenario_tag": np.asarray(scenario_tag),
        "epsilon_values": np.asarray(epsilon_values, dtype=float),
        "fixed_discount_values": np.asarray(fixed_discount_values, dtype=float),
        "seed0": int(seed0),
        "n_episodes": int(n_episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "adaptive_target_radius": float(adaptive_target_radius),
        "adaptive_delta_min": float(adaptive_delta_min),
        "adaptive_delta_max": float(adaptive_delta_max),
        "target_dimension": float(target_dimension),
        "best_fixed_delta": float(best_mode.delta if best_mode.delta is not None else np.nan),
        "best_fixed_abs_gap": float(best_gap),
    }

    for mode in mode_list:
        summary = summaries[mode.name]
        safe_name = mode.name.replace(".", "p")
        eps_values = np.asarray(summary.eps_real_values, dtype=float)
        delta_values = np.asarray(summary.delta_values, dtype=float)
        trace_q_values = np.asarray(summary.trace_q_values, dtype=float)
        trace_p_values = np.asarray(summary.trace_p_values, dtype=float)
        payload[f"count_{safe_name}"] = int(summary.count)
        payload[f"mean_nis_{safe_name}"] = float(eps_values.mean()) if eps_values.size else 0.0
        payload[f"median_nis_{safe_name}"] = float(np.median(eps_values)) if eps_values.size else 0.0
        payload[f"mean_delta_{safe_name}"] = float(delta_values.mean()) if delta_values.size else 0.0
        payload[f"mean_trQ_{safe_name}"] = float(trace_q_values.mean()) if trace_q_values.size else 0.0
        payload[f"mean_trP_{safe_name}"] = float(trace_p_values.mean()) if trace_p_values.size else 0.0
        for epsilon in epsilon_values:
            payload[f"inside_pct_eps{str(float(epsilon)).replace('.', 'p')}_{safe_name}"] = 100.0 * float(
                summary.inside_counts[float(epsilon)]
            ) / max(int(summary.count), 1)

    save_npz(outpath, **payload)
    print(f"Saved calibration summary to: {outpath}")


def inspect_discounted_covariances(
    *,
    seed0: int,
    n_episodes: int,
    max_steps_per_episode: int,
    epsilon_values: tuple[float, ...],
    fixed_discount_values: tuple[float, ...],
    adaptive_target_radius: float,
    adaptive_delta_min: float,
    adaptive_delta_max: float,
    scenario_tag: str,
    outpath: str,
) -> None:
    """
    Compare fixed-`Q`, fixed-discount, and adaptive-discount calibration.
    """
    kf_meas_std, kf_proc_std, kf_meas_corr, kf_proc_corr = build_correlated_filter_covariances()
    R, baseline_Q = cartpole_mod.build_filter_covariances(
        meas_std=kf_meas_std,
        proc_std=kf_proc_std,
        meas_corr=kf_meas_corr,
        proc_corr=kf_proc_corr,
    )

    model_path = cartpole_mod.ensure_downloaded_cartpole_checkpoint()
    device = torch.device("cpu")
    model = cartpole_mod.load_cartpole_policy(model_path, device)
    ssm = cartpole_mod.build_cartpole_linear_ssm()

    mode_list = build_mode_list(
        baseline_q=baseline_Q,
        fixed_discount_values=fixed_discount_values,
        adaptive_target_radius=adaptive_target_radius,
        adaptive_delta_min=adaptive_delta_min,
        adaptive_delta_max=adaptive_delta_max,
    )
    summaries = initialize_summary_dict(
        mode_list=mode_list,
        epsilon_values=epsilon_values,
    )

    for episode_idx in range(int(n_episodes)):
        seed = int(seed0 + episode_idx)
        env = gym.make("CartPole-v1")
        obs_clean, _info = env.reset(seed=seed)

        rng_nominal_noise = np.random.default_rng(seed + 707_002)
        init_noise = rng_nominal_noise.normal(0.0, kf_meas_std, size=(4,)).astype(np.float32)
        obs_init_noisy = (np.asarray(obs_clean, dtype=np.float32) + init_noise).astype(np.float32)

        filter_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for mode in mode_list:
            m_post, P_post = cartpole_mod.kf_update_state(
                m_pred=np.asarray(obs_clean, dtype=np.float32),
                P_pred=cartpole_mod.project_to_psd(R.copy()),
                y_obs=obs_init_noisy,
                R=R,
            )
            filter_states[mode.name] = (m_post, P_post)

        step_idx = 1
        while step_idx <= int(max_steps_per_episode):
            action = cartpole_mod.select_action(model, np.asarray(obs_clean, dtype=np.float32))
            force = cartpole_mod.action_to_force(action, ssm.force_mag)

            nominal_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for mode in mode_list:
                m_post, P_post = filter_states[mode.name]
                m_pred_nom, transition_cov = predict_nominal_transition(
                    m_post=m_post,
                    P_post=P_post,
                    force=force,
                    ssm=ssm,
                )
                nominal_predictions[mode.name] = (m_pred_nom, transition_cov)

            obs_clean, reward, terminated, truncated, _info = env.step(action)
            _ = reward
            o_clean = np.asarray(obs_clean, dtype=np.float32)
            nominal_noise = rng_nominal_noise.normal(0.0, kf_meas_std, size=(4,)).astype(np.float32)
            o_real = (o_clean + nominal_noise).astype(np.float32)

            for mode in mode_list:
                m_pred_nom, transition_cov = nominal_predictions[mode.name]
                innovation = o_real - m_pred_nom
                P_pred, q_eff, delta_used = select_mode_predictive_covariance(
                    mode=mode,
                    transition_cov=transition_cov,
                    innovation=innovation,
                    R=R,
                )
                attack_sigma = cartpole_mod.project_to_psd(
                    np.asarray(P_pred, dtype=float) + np.asarray(R, dtype=float)
                )
                eps_real = achieved_attack_epsilon(
                    observation=o_real,
                    center=m_pred_nom,
                    Sigma=attack_sigma,
                )
                update_summary(
                    summary=summaries[mode.name],
                    eps_real=eps_real,
                    delta_used=delta_used,
                    q_eff=q_eff,
                    P_pred=P_pred,
                    epsilon_values=epsilon_values,
                )

                m_post, P_post = cartpole_mod.kf_update_state(
                    m_pred=m_pred_nom,
                    P_pred=P_pred,
                    y_obs=o_real,
                    R=R,
                )
                filter_states[mode.name] = (m_post, P_post)

            step_idx += 1
            if terminated or truncated:
                break

        env.close()

    target_dimension = 4.0
    best_mode, best_gap = best_fixed_discount_mode(
        mode_list=mode_list,
        summaries=summaries,
        target_dimension=target_dimension,
    )
    best_mean_nis = float(np.mean(np.asarray(summaries[best_mode.name].eps_real_values, dtype=float)))

    print("CartPole discounted-covariance inspection")
    print(
        f"episodes={n_episodes}  steps={max_steps_per_episode}  "
        f"target_mean_nis={target_dimension:.1f}  fixed_deltas={fixed_discount_values}"
    )
    print(
        f"best_fixed_delta={float(best_mode.delta):.2f}  "
        f"mean_nis={best_mean_nis:.4f}  abs_gap={best_gap:.4f}"
    )
    print_summary_table(
        mode_list=mode_list,
        summaries=summaries,
        epsilon_values=epsilon_values,
        target_dimension=target_dimension,
    )
    save_calibration_summary(
        outpath=outpath,
        scenario_tag=scenario_tag,
        mode_list=mode_list,
        summaries=summaries,
        epsilon_values=epsilon_values,
        fixed_discount_values=fixed_discount_values,
        seed0=seed0,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps_per_episode,
        adaptive_target_radius=adaptive_target_radius,
        adaptive_delta_min=adaptive_delta_min,
        adaptive_delta_max=adaptive_delta_max,
        target_dimension=target_dimension,
        best_mode=best_mode,
        best_gap=best_gap,
    )


def main() -> None:
    """
    Entry point for the discount-based CartPole covariance inspection.
    """
    seed0 = 7
    n_episodes = 6
    max_steps_per_episode = 24
    # 4D chi-square radii corresponding approximately to 75% and 95%
    # predictive-ellipsoid coverage.
    epsilon_values = (5.39, 9.49)
    fixed_discount_values = tuple(float(value) for value in np.arange(0.80, 0.95, 0.01))
    adaptive_target_radius = 4.0
    adaptive_delta_min = 0.70
    adaptive_delta_max = 1.00
    scenario_tag = "obs010-022-005-022_nofallback"
    out_dir = data_dir_for(CURRENT_DIR)
    outpath = os.path.join(
        out_dir,
        (
            "calibrate_cartpole_discounted_covariances_"
            f"{scenario_tag}_N{n_episodes}_S{max_steps_per_episode}.npz"
        ),
    )

    inspect_discounted_covariances(
        seed0=seed0,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps_per_episode,
        epsilon_values=epsilon_values,
        fixed_discount_values=fixed_discount_values,
        adaptive_target_radius=adaptive_target_radius,
        adaptive_delta_min=adaptive_delta_min,
        adaptive_delta_max=adaptive_delta_max,
        scenario_tag=scenario_tag,
        outpath=outpath,
    )


if __name__ == "__main__":
    main()
