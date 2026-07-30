"""
Compatibility helpers for the original script-oriented repository layout.

The new package uses the canonical notation

    s_t = A_t s_{t-1} + B_t a_{t-1} + w_t
    o_t = F_t s_t     + G_t a_{t-1} + v_t.

Many existing scripts, however, store observations as `y[0:T]` and controls as
either `u[0:T-1]` or `u[0:T]`. This module keeps those scripts working while
centralizing the numerical implementation in one package. New code should
prefer `shared_ssm.linear_gaussian`, `shared_ssm.geometry`, and
`shared_ssm.attacks` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .attacks.linear import solve_max_quadratic_over_ellipsoid
from .constraints import project_to_ellipsoid
from .defenses.covariance_adaptation import clip_attack_evidence_score
from .defenses.covariance_adaptation import compute_contamination_prior
from .defenses.covariance_adaptation import log_mixture_posterior_weight
from .defenses.covariance_adaptation import mahalanobis_attack_evidence
from .defenses.covariance_adaptation import posterior_attack_probability_from_evidence
from .defenses.covariance_adaptation import rank_one_covariance_update
from .linalg import gaussian_logpdf
from .linalg import project_to_psd
from .linalg import solve_spd
from .linalg import spd_inverse
from .linalg import sqrtm_psd


ObservationActionConvention = Literal["previous", "current"]


@dataclass(frozen=True)
class LegacyControlConvention:
    """
    Control indexing convention for legacy arrays.

    `transition_action="current"` means transition `s_k -> s_{k+1}` uses
    `u[k]`, which matches the requested RL-compatible convention.

    `observation_action` controls the direct observation term:
    1. `"previous"`: `y[k]` uses `u[k-1]` for `k >= 1`, and `u[0]` for `k=0`.
    2. `"current"`: `y[k]` uses `u[k]`, clipped at the last available control.
    """

    observation_action: ObservationActionConvention = "previous"


def simulate_lgssm_nd(
    A0: np.ndarray,
    B0: np.ndarray,
    H0: np.ndarray,
    D0: np.ndarray,
    T: int,
    seed: int = 123,
    x0: np.ndarray | None = None,
    Q0: np.ndarray | None = None,
    R0: np.ndarray | None = None,
    dA: np.ndarray | None = None,
    dB: np.ndarray | None = None,
    dH: np.ndarray | None = None,
    dD: np.ndarray | None = None,
    dQ: np.ndarray | None = None,
    dR: np.ndarray | None = None,
    u_low: float = -0.5,
    u_high: float = 0.5,
    control_length: Literal["T", "T_plus_1"] = "T",
    observation_action: ObservationActionConvention = "previous",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Simulate the legacy `y[0:T]` LGSSM format used by the experiment scripts.

    The transition always uses `u[k]` for `s_k -> s_{k+1}`. This is the
    canonical convention selected for the refactor and matches the RL policy
    interpretation.
    """
    if int(T) < 0:
        raise ValueError("T must be non-negative.")

    rng = np.random.default_rng(int(seed))
    A0 = np.asarray(A0, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    H0 = np.asarray(H0, dtype=float)
    D0 = np.asarray(D0, dtype=float)
    n_state = A0.shape[0]
    n_action = B0.shape[1]
    n_obs = H0.shape[0]

    x0 = np.zeros(n_state, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(n_state)
    Q0 = 0.02 * np.eye(n_state) if Q0 is None else np.asarray(Q0, dtype=float)
    R0 = 0.03 * np.eye(n_obs) if R0 is None else np.asarray(R0, dtype=float)

    dA = np.zeros_like(A0) if dA is None else np.asarray(dA, dtype=float)
    dB = np.zeros_like(B0) if dB is None else np.asarray(dB, dtype=float)
    dH = np.zeros_like(H0) if dH is None else np.asarray(dH, dtype=float)
    dD = np.zeros_like(D0) if dD is None else np.asarray(dD, dtype=float)
    dQ = np.zeros_like(Q0) if dQ is None else np.asarray(dQ, dtype=float)
    dR = np.zeros_like(R0) if dR is None else np.asarray(dR, dtype=float)

    A_t = np.zeros((T + 1, n_state, n_state), dtype=float)
    B_t = np.zeros((T + 1, n_state, n_action), dtype=float)
    H_t = np.zeros((T + 1, n_obs, n_state), dtype=float)
    D_t = np.zeros((T + 1, n_obs, n_action), dtype=float)
    Q_t = np.zeros((T + 1, n_state, n_state), dtype=float)
    R_t = np.zeros((T + 1, n_obs, n_obs), dtype=float)

    for time_idx in range(T + 1):
        A_t[time_idx] = A0 + dA * time_idx
        B_t[time_idx] = B0 + dB * time_idx
        H_t[time_idx] = H0 + dH * time_idx
        D_t[time_idx] = D0 + dD * time_idx
        Q_t[time_idx] = project_to_psd(Q0 + dQ * time_idx)
        R_t[time_idx] = project_to_psd(R0 + dR * time_idx)

    u_len = T + 1 if control_length == "T_plus_1" else max(T, 1)
    u = rng.uniform(float(u_low), float(u_high), size=(u_len, n_action))
    if control_length == "T" and T == 0:
        u = u[:0]

    states = np.zeros((T + 1, n_state), dtype=float)
    observations = np.zeros((T + 1, n_obs), dtype=float)
    states[0] = x0

    observations[0] = (
        H_t[0] @ states[0]
        + D_t[0] @ _legacy_observation_action(u, 0, convention=observation_action, action_dim=n_action)
        + rng.multivariate_normal(np.zeros(n_obs), R_t[0])
    )

    for time_idx in range(T):
        action = _legacy_transition_action(u, time_idx, action_dim=n_action)
        states[time_idx + 1] = (
            A_t[time_idx] @ states[time_idx]
            + B_t[time_idx] @ action
            + rng.multivariate_normal(np.zeros(n_state), Q_t[time_idx])
        )
        obs_action = _legacy_observation_action(
            u,
            time_idx + 1,
            convention=observation_action,
            action_dim=n_action,
        )
        observations[time_idx + 1] = (
            H_t[time_idx + 1] @ states[time_idx + 1]
            + D_t[time_idx + 1] @ obs_action
            + rng.multivariate_normal(np.zeros(n_obs), R_t[time_idx + 1])
        )

    mats = {"A_t": A_t, "B_t": B_t, "H_t": H_t, "D_t": D_t, "Q_t": Q_t, "R_t": R_t}
    return states, observations, u, mats


def simulate_lgssm_1d(
    A: float,
    B: float,
    H: float,
    D: float,
    T: int,
    seed: int = 123,
    x0: float = 0.0,
    Q: float = 0.02,
    R: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a scalar legacy LGSSM with transition action `u[k]`."""
    states, observations, controls, _mats = simulate_lgssm_nd(
        np.array([[float(A)]]),
        np.array([[float(B)]]),
        np.array([[float(H)]]),
        np.array([[float(D)]]),
        T=int(T),
        seed=int(seed),
        x0=np.array([float(x0)]),
        Q0=np.array([[float(Q)]]),
        R0=np.array([[float(R)]]),
    )
    return states[:, 0], observations[:, 0], controls[:, 0]


def simulate_lgssm_nd_current_observation(*args, **kwargs):
    """Legacy simulator for scripts where `y[k]` uses the current control `u[k]`."""
    kwargs.setdefault("control_length", "T_plus_1")
    kwargs.setdefault("observation_action", "current")
    return simulate_lgssm_nd(*args, **kwargs)


def simulate_lgssm_nd_previous_observation(*args, **kwargs):
    """Legacy simulator for scripts where `y[k]` uses the previous transition action."""
    kwargs.setdefault("control_length", "T")
    kwargs.setdefault("observation_action", "previous")
    return simulate_lgssm_nd(*args, **kwargs)


def simulate_lgssm(
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    T: int,
    seed: int = 123,
    x0: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility simulator for constant-matrix ND demos."""
    states, observations, controls, _mats = simulate_lgssm_nd(
        A,
        B,
        H,
        D,
        T=int(T),
        seed=int(seed),
        x0=x0,
        Q0=Q,
        R0=R,
        u_low=0.0,
        u_high=1.0,
        control_length="T",
        observation_action="previous",
    )
    return states, observations, controls


def simulate_lgssm_nd_const(
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    T: int,
    seed: int = 123,
    x0: np.ndarray | None = None,
    u_low: float = -0.5,
    u_high: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compatibility simulator for constant-parameter KKT/Grey-box scripts."""
    states, observations, controls, mats = simulate_lgssm_nd(
        A,
        B,
        H,
        D,
        T=int(T),
        seed=int(seed),
        x0=x0,
        Q0=Q,
        R0=R,
        u_low=float(u_low),
        u_high=float(u_high),
        control_length="T",
        observation_action="previous",
    )
    return states, observations, controls, {
        "A": np.asarray(A, dtype=float),
        "B": np.asarray(B, dtype=float),
        "H": np.asarray(H, dtype=float),
        "D": np.asarray(D, dtype=float),
        "Q": project_to_psd(np.asarray(Q, dtype=float)),
        "R": project_to_psd(np.asarray(R, dtype=float)),
        **mats,
    }


def kalman_filter_nd(
    *,
    y: np.ndarray,
    u: np.ndarray,
    A_t: np.ndarray,
    B_t: np.ndarray,
    H_t: np.ndarray,
    D_t: np.ndarray,
    Q_t: np.ndarray,
    R_t: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    obs_mask: np.ndarray | None = None,
    observation_action: ObservationActionConvention = "previous",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Kalman filtering for the legacy `y[0:T]` array format."""
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    A_t = np.asarray(A_t, dtype=float)
    B_t = np.asarray(B_t, dtype=float)
    H_t = np.asarray(H_t, dtype=float)
    D_t = np.asarray(D_t, dtype=float)
    Q_t = np.asarray(Q_t, dtype=float)
    R_t = np.asarray(R_t, dtype=float)
    m0 = np.asarray(m0, dtype=float).reshape(-1)
    P0 = project_to_psd(np.asarray(P0, dtype=float))

    T = y.shape[0] - 1
    n_state = m0.size
    n_action = B_t.shape[2]
    identity = np.eye(n_state, dtype=float)
    if obs_mask is None:
        obs_mask = np.ones(T + 1, dtype=bool)
    else:
        obs_mask = np.asarray(obs_mask, dtype=bool)

    m_pred = np.zeros((T + 1, n_state), dtype=float)
    P_pred = np.zeros((T + 1, n_state, n_state), dtype=float)
    m_filt = np.zeros((T + 1, n_state), dtype=float)
    P_filt = np.zeros((T + 1, n_state, n_state), dtype=float)
    m_pred[0] = m0
    P_pred[0] = P0

    for time_idx in range(T + 1):
        Hk = H_t[time_idx]
        Rk = project_to_psd(R_t[time_idx])
        obs_action = _legacy_observation_action(
            u,
            time_idx,
            convention=observation_action,
            action_dim=n_action,
        )
        y_hat = Hk @ m_pred[time_idx] + D_t[time_idx] @ obs_action
        S = project_to_psd(Hk @ P_pred[time_idx] @ Hk.T + Rk)
        K = solve_spd(S, Hk @ P_pred[time_idx].T).T
        innovation = y[time_idx] - y_hat

        if bool(obs_mask[time_idx]) and not np.any(np.isnan(y[time_idx])):
            m_filt[time_idx] = m_pred[time_idx] + K @ innovation
            left = identity - K @ Hk
            P_filt[time_idx] = project_to_psd(left @ P_pred[time_idx] @ left.T + K @ Rk @ K.T)
        else:
            m_filt[time_idx] = m_pred[time_idx]
            P_filt[time_idx] = P_pred[time_idx]

        if time_idx < T:
            action = _legacy_transition_action(u, time_idx, action_dim=n_action)
            m_pred[time_idx + 1] = A_t[time_idx] @ m_filt[time_idx] + B_t[time_idx] @ action
            P_pred[time_idx + 1] = project_to_psd(
                A_t[time_idx] @ P_filt[time_idx] @ A_t[time_idx].T + Q_t[time_idx]
            )

    return m_filt, P_filt, m_pred, P_pred


def kalman_filter_1d(
    y: np.ndarray,
    u: np.ndarray,
    A: float,
    B: float,
    H: float,
    D: float,
    Q: float,
    R: float,
    m0: float,
    P0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scalar wrapper around the shared legacy Kalman filter."""
    T = np.asarray(y).shape[0] - 1
    A_t, B_t, H_t, D_t, Q_t, R_t = _constant_1d_sequences(A, B, H, D, Q, R, T)
    m_filt, P_filt, m_pred, P_pred = kalman_filter_nd(
        y=np.asarray(y, dtype=float).reshape(T + 1, 1),
        u=np.asarray(u, dtype=float).reshape(-1, 1),
        A_t=A_t,
        B_t=B_t,
        H_t=H_t,
        D_t=D_t,
        Q_t=Q_t,
        R_t=R_t,
        m0=np.array([float(m0)]),
        P0=np.array([[float(P0)]]),
    )
    return m_filt[:, 0], P_filt[:, 0, 0], m_pred[:, 0], P_pred[:, 0, 0]


def kalman_filter_nd_current_observation(*args, **kwargs):
    """Legacy Kalman filter for the current-observation-control convention."""
    kwargs.setdefault("observation_action", "current")
    return kalman_filter_nd(*args, **kwargs)


def kalman_filter_nd_previous_observation(*args, **kwargs):
    """Legacy Kalman filter for the previous-observation-control convention."""
    kwargs.setdefault("observation_action", "previous")
    return kalman_filter_nd(*args, **kwargs)


def kalman_filter(
    y: np.ndarray,
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility Kalman filter for constant-matrix ND demos."""
    T = np.asarray(y).shape[0] - 1
    A_t = np.repeat(np.asarray(A, dtype=float)[None, :, :], T + 1, axis=0)
    B_t = np.repeat(np.asarray(B, dtype=float)[None, :, :], T + 1, axis=0)
    H_t = np.repeat(np.asarray(H, dtype=float)[None, :, :], T + 1, axis=0)
    D_t = np.repeat(np.asarray(D, dtype=float)[None, :, :], T + 1, axis=0)
    Q_t = np.repeat(project_to_psd(Q)[None, :, :], T + 1, axis=0)
    R_t = np.repeat(project_to_psd(R)[None, :, :], T + 1, axis=0)
    return kalman_filter_nd(
        y=y,
        u=u,
        A_t=A_t,
        B_t=B_t,
        H_t=H_t,
        D_t=D_t,
        Q_t=Q_t,
        R_t=R_t,
        m0=m0,
        P0=P0,
    )


def kalman_filter_nd_const(
    *,
    y: np.ndarray,
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keyword-only compatibility Kalman filter for constant KKT scripts."""
    return kalman_filter(y, u, A, B, H, D, Q, R, m0, P0)


def rts_smoother_nd(
    *,
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run RTS smoothing for legacy filtered arrays."""
    m_filt = np.asarray(m_filt, dtype=float)
    P_filt = np.asarray(P_filt, dtype=float)
    m_pred = np.asarray(m_pred, dtype=float)
    P_pred = np.asarray(P_pred, dtype=float)
    A_t = np.asarray(A_t, dtype=float)
    T = m_filt.shape[0] - 1

    m_smooth = np.zeros_like(m_filt)
    P_smooth = np.zeros_like(P_filt)
    m_smooth[T] = m_filt[T]
    P_smooth[T] = P_filt[T]

    for time_idx in range(T - 1, -1, -1):
        J = solve_spd(P_pred[time_idx + 1], A_t[time_idx] @ P_filt[time_idx]).T
        m_smooth[time_idx] = m_filt[time_idx] + J @ (m_smooth[time_idx + 1] - m_pred[time_idx + 1])
        P_smooth[time_idx] = project_to_psd(
            P_filt[time_idx] + J @ (P_smooth[time_idx + 1] - P_pred[time_idx + 1]) @ J.T
        )

    return m_smooth, P_smooth


def rts_smoother_1d(
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar wrapper around the shared legacy RTS smoother."""
    T = np.asarray(m_filt).shape[0] - 1
    A_t = np.repeat(np.array([[[float(A)]]]), T + 1, axis=0)
    m_smooth, P_smooth = rts_smoother_nd(
        m_filt=np.asarray(m_filt, dtype=float).reshape(T + 1, 1),
        P_filt=np.asarray(P_filt, dtype=float).reshape(T + 1, 1, 1),
        m_pred=np.asarray(m_pred, dtype=float).reshape(T + 1, 1),
        P_pred=np.asarray(P_pred, dtype=float).reshape(T + 1, 1, 1),
        A_t=A_t,
    )
    return m_smooth[:, 0], P_smooth[:, 0, 0]


def rts_smoother(
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility RTS smoother for constant-matrix ND demos."""
    T = np.asarray(m_filt).shape[0] - 1
    A_t = np.repeat(np.asarray(A, dtype=float)[None, :, :], T + 1, axis=0)
    return rts_smoother_nd(
        m_filt=m_filt,
        P_filt=P_filt,
        m_pred=m_pred,
        P_pred=P_pred,
        A_t=A_t,
    )


def rts_smoother_nd_const(
    *,
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keyword-only compatibility RTS smoother for constant KKT scripts."""
    return rts_smoother(m_filt, P_filt, m_pred, P_pred, A)


def leave_one_out_attack_stats_nd(
    *,
    t: int,
    y: np.ndarray,
    u: np.ndarray,
    A_t: np.ndarray,
    B_t: np.ndarray,
    H_t: np.ndarray,
    D_t: np.ndarray,
    Q_t: np.ndarray,
    R_t: np.ndarray,
    P0: np.ndarray,
    m0: np.ndarray,
    observation_action: ObservationActionConvention = "previous",
) -> dict[str, np.ndarray]:
    """Return leave-one-out posterior and observation predictive quantities."""
    y = np.asarray(y, dtype=float)
    T = y.shape[0] - 1
    if not (0 <= int(t) <= T):
        raise ValueError("t out of range.")

    obs_mask = np.ones(T + 1, dtype=bool)
    obs_mask[int(t)] = False
    m_filt, P_filt, m_pred, P_pred = kalman_filter_nd(
        y=y,
        u=u,
        A_t=A_t,
        B_t=B_t,
        H_t=H_t,
        D_t=D_t,
        Q_t=Q_t,
        R_t=R_t,
        m0=m0,
        P0=P0,
        obs_mask=obs_mask,
        observation_action=observation_action,
    )
    m_smooth, P_smooth = rts_smoother_nd(
        m_filt=m_filt,
        P_filt=P_filt,
        m_pred=m_pred,
        P_pred=P_pred,
        A_t=A_t,
    )

    t = int(t)
    H_obs = np.asarray(H_t, dtype=float)[t]
    D_obs = np.asarray(D_t, dtype=float)[t]
    R_obs = project_to_psd(np.asarray(R_t, dtype=float)[t])
    n_action = np.asarray(B_t).shape[2]
    obs_action = _legacy_observation_action(
        np.asarray(u, dtype=float),
        t,
        convention=observation_action,
        action_dim=n_action,
    )
    m_minus = m_smooth[t]
    P_minus = project_to_psd(P_smooth[t])
    mu_obs = H_obs @ m_minus + D_obs @ obs_action
    Sigma_obs = project_to_psd(H_obs @ P_minus @ H_obs.T + R_obs)
    K_obs = solve_spd(Sigma_obs, H_obs @ P_minus.T).T
    P_post = project_to_psd(P_minus - K_obs @ H_obs @ P_minus)

    return {
        "m_t_minus": m_minus,
        "P_t_minus": P_minus,
        "mu_t": mu_obs,
        "Sigma_t": Sigma_obs,
        "K_t": K_obs,
        "P_post": P_post,
        "m_smooth_minus_t": m_smooth,
        "P_smooth_minus_t": P_smooth,
    }


def leave_one_out_attack_stats_nd_current_observation(*args, **kwargs):
    """Leave-one-out stats for the current-observation-control convention."""
    kwargs.setdefault("observation_action", "current")
    return leave_one_out_attack_stats_nd(*args, **kwargs)


def leave_one_out_attack_stats_nd_previous_observation(*args, **kwargs):
    """Leave-one-out stats for the previous-observation-control convention."""
    kwargs.setdefault("observation_action", "previous")
    return leave_one_out_attack_stats_nd(*args, **kwargs)


def loo_values_nd(
    *,
    t: int,
    y: np.ndarray,
    u: np.ndarray,
    A_t: np.ndarray,
    B_t: np.ndarray,
    H_t: np.ndarray,
    D_t: np.ndarray,
    Q_t: np.ndarray,
    R_t: np.ndarray,
    P0: np.ndarray,
    m0: np.ndarray,
    observation_action: ObservationActionConvention = "previous",
) -> list[np.ndarray]:
    """
    Return `[X_t, mu_t, Sigma_t]` for legacy AdvSSM-style attacks.

    `X_t` is the affine posterior gain from an attacked observation to the
    smoothed mean at the attacked time. For Gaussian models this is the same
    linear object needed by the KKT attack.
    """
    stats = leave_one_out_attack_stats_nd(
        t=t,
        y=y,
        u=u,
        A_t=A_t,
        B_t=B_t,
        H_t=H_t,
        D_t=D_t,
        Q_t=Q_t,
        R_t=R_t,
        P0=P0,
        m0=m0,
        observation_action=observation_action,
    )
    return [stats["K_t"], stats["mu_t"], stats["Sigma_t"]]


def loo_values_nd_current_observation(*args, **kwargs):
    """Return LOO geometry for the current-observation-control convention."""
    kwargs.setdefault("observation_action", "current")
    return loo_values_nd(*args, **kwargs)


def loo_values_nd_previous_observation(*args, **kwargs):
    """Return LOO geometry for the previous-observation-control convention."""
    kwargs.setdefault("observation_action", "previous")
    return loo_values_nd(*args, **kwargs)


def loo_values_nd_const(
    *,
    t: int,
    y: np.ndarray,
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P0: np.ndarray,
    m0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility LOO geometry for constant-matrix KKT scripts."""
    T = np.asarray(y).shape[0] - 1
    A_t = np.repeat(np.asarray(A, dtype=float)[None, :, :], T + 1, axis=0)
    B_t = np.repeat(np.asarray(B, dtype=float)[None, :, :], T + 1, axis=0)
    H_t = np.repeat(np.asarray(H, dtype=float)[None, :, :], T + 1, axis=0)
    D_t = np.repeat(np.asarray(D, dtype=float)[None, :, :], T + 1, axis=0)
    Q_t = np.repeat(project_to_psd(Q)[None, :, :], T + 1, axis=0)
    R_t = np.repeat(project_to_psd(R)[None, :, :], T + 1, axis=0)
    X, mu, Sigma = loo_values_nd(
        t=t,
        y=y,
        u=u,
        A_t=A_t,
        B_t=B_t,
        H_t=H_t,
        D_t=D_t,
        Q_t=Q_t,
        R_t=R_t,
        P0=P0,
        m0=m0,
    )
    return X, mu, Sigma


def format_loo_line_1d(
    *,
    t: int,
    y: np.ndarray,
    u: np.ndarray,
    A_t: list[float] | np.ndarray,
    B_t: list[float] | np.ndarray,
    H_t: list[float] | np.ndarray,
    D_t: list[float] | np.ndarray,
    Q_t: list[float] | np.ndarray,
    R_t: list[float] | np.ndarray,
    P0: float,
    m0: float = 0.0,
) -> str:
    """Return the formatted scalar leave-one-out line used by legacy demos."""
    A_arr = np.asarray(A_t, dtype=float).reshape(-1, 1, 1)
    B_arr = np.asarray(B_t, dtype=float).reshape(-1, 1, 1)
    H_arr = np.asarray(H_t, dtype=float).reshape(-1, 1, 1)
    D_arr = np.asarray(D_t, dtype=float).reshape(-1, 1, 1)
    Q_arr = np.asarray(Q_t, dtype=float).reshape(-1, 1, 1)
    R_arr = np.asarray(R_t, dtype=float).reshape(-1, 1, 1)
    X, mu, sigma = loo_values_nd(
        t=int(t),
        y=np.asarray(y, dtype=float).reshape(-1, 1),
        u=np.asarray(u, dtype=float).reshape(-1, 1),
        A_t=A_arr,
        B_t=B_arr,
        H_t=H_arr,
        D_t=D_arr,
        Q_t=Q_arr,
        R_t=R_arr,
        P0=np.array([[float(P0)]]),
        m0=np.array([float(m0)]),
    )
    return (
        f"t={int(t)} "
        f"[X={float(np.asarray(X).reshape(-1)[0]): .6f}, "
        f"mu_y_t_given_minus_t={float(np.asarray(mu).reshape(-1)[0]): .6f}, "
        f"Sigma_y_t_given_minus_t={float(np.asarray(sigma).reshape(-1)[0]): .6f}]"
    )


def solve_kkt_max_quadratic_over_ellipsoid(
    *,
    X: np.ndarray,
    y_t: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 250,
) -> tuple[np.ndarray, float]:
    """Compatibility wrapper for the analytic KKT attack."""
    return solve_max_quadratic_over_ellipsoid(
        effect_matrix=X,
        reference_observation=y_t,
        center=mu,
        covariance=Sigma,
        epsilon=float(epsilon),
        tol=float(tol),
        max_iter=int(max_iter),
    )


def project_to_attack_region(
    y_candidate: np.ndarray,
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """Compatibility wrapper for ellipsoid projection."""
    return project_to_ellipsoid(
        observation=y_candidate,
        center=mu_t,
        covariance=Sigma_t,
        epsilon=float(epsilon),
        tol=float(tol),
        max_iter=int(max_iter),
    )


def project_to_attack_region_named(
    *,
    y_candidate: np.ndarray,
    center: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Compatibility wrapper for Gymnasium keyword naming."""
    return project_to_ellipsoid(
        observation=y_candidate,
        center=center,
        covariance=Sigma,
        epsilon=float(epsilon),
    )


def finite_diff_jacobian_g(g, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Return a centered finite-difference Jacobian for scalar/vector `g`."""
    x = np.asarray(x, dtype=float).reshape(-1)
    fx = _as_1d_output(g(x))
    jacobian = np.zeros((fx.size, x.size), dtype=float)
    for dim_idx in range(x.size):
        step = np.zeros_like(x)
        step[dim_idx] = float(h)
        fp = _as_1d_output(g(x + step))
        fm = _as_1d_output(g(x - step))
        jacobian[:, dim_idx] = (fp - fm) / (2.0 * float(h))
    return jacobian


def white_box_point_attack_nd(
    *,
    t: int,
    y: np.ndarray,
    u: np.ndarray,
    A_t: np.ndarray,
    B_t: np.ndarray,
    H_t: np.ndarray,
    D_t: np.ndarray,
    Q_t: np.ndarray,
    R_t: np.ndarray,
    P0: np.ndarray,
    m0: np.ndarray,
    epsilon: float,
    M_star: np.ndarray | float,
    g,
    g_grad=None,
    eta: float = 0.05,
    n_steps: int = 500,
    n_mc: int = 128,
    seed: int = 1234,
    observation_action: ObservationActionConvention = "current",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compatibility nonlinear attack used by AdvNonLinearAttack scripts.

    It optimizes `||E[g(s_t)|o_t'] - M_star||^2` with projected gradient steps.
    """
    rng = np.random.default_rng(int(seed))
    stats = leave_one_out_attack_stats_nd(
        t=t,
        y=y,
        u=u,
        A_t=A_t,
        B_t=B_t,
        H_t=H_t,
        D_t=D_t,
        Q_t=Q_t,
        R_t=R_t,
        P0=P0,
        m0=m0,
        observation_action=observation_action,
    )
    m_minus = stats["m_t_minus"]
    mu_t = stats["mu_t"]
    Sigma_t = stats["Sigma_t"]
    K_t = stats["K_t"]
    P_post = stats["P_post"]
    target = _as_1d_output(M_star)
    y_current = project_to_attack_region(np.asarray(y, dtype=float)[int(t)], mu_t, Sigma_t, epsilon)

    L_post = sqrtm_psd(P_post)
    xi = rng.standard_normal(size=(int(n_mc), m_minus.size))
    y_hist: list[np.ndarray] = []
    obj_hist: list[float] = []
    mu_g_hist: list[np.ndarray] = []

    for _ in range(int(n_steps)):
        m_post = m_minus + K_t @ (y_current - mu_t)
        samples = m_post[None, :] + xi @ L_post.T
        values = np.stack([_as_1d_output(g(sample)) for sample in samples], axis=0)
        mu_g = np.mean(values, axis=0)
        if mu_g.shape != target.shape:
            raise ValueError("g output and M_star have incompatible shapes.")

        if g_grad is None:
            jacobians = np.stack([finite_diff_jacobian_g(g, sample) for sample in samples], axis=0)
        else:
            jacobians = np.stack([_as_2d_jacobian(g_grad(sample), m_minus.size) for sample in samples], axis=0)
        dmu_dy = np.mean(jacobians, axis=0) @ K_t
        diff = mu_g - target
        gradient = 2.0 * (dmu_dy.T @ diff)

        y_current = y_current - float(eta) * gradient
        y_current = project_to_attack_region(y_current, mu_t, Sigma_t, epsilon)
        y_hist.append(y_current.copy())
        obj_hist.append(float(np.dot(diff, diff)))
        mu_g_hist.append(mu_g.copy())

    history = {
        "y_hist": np.asarray(y_hist),
        "obj_hist": np.asarray(obj_hist),
        "mu_g_hist": np.asarray(mu_g_hist),
        "mu_t": mu_t,
        "Sigma_t": Sigma_t,
        "m_t_minus": m_minus,
        "K_t": K_t,
        "P_post": P_post,
    }
    return y_current, history


def estimate_E_g(
    *,
    m: np.ndarray,
    P: np.ndarray,
    g,
    n_mc: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate `E[g(s)]` and `g(E[s])` under a Gaussian state law."""
    rng = np.random.default_rng(int(seed))
    mean = np.asarray(m, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(P, dtype=float))
    samples = mean[None, :] + rng.standard_normal(size=(int(n_mc), mean.size)) @ sqrtm_psd(covariance).T
    values = np.stack([_as_1d_output(g(sample)) for sample in samples], axis=0)
    return np.mean(values, axis=0), _as_1d_output(g(mean))


def compute_filter_predictive_quantities(
    *,
    y: np.ndarray,
    u: np.ndarray,
    A_t: np.ndarray,
    B_t: np.ndarray,
    H_t: np.ndarray,
    D_t: np.ndarray,
    Q_t: np.ndarray,
    R_t: np.ndarray,
    attack_t: int,
    m0: np.ndarray,
    P0: np.ndarray,
    observation_action: ObservationActionConvention = "previous",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return KF-only attack geometry at one legacy time index."""
    _m_filt, _P_filt, m_pred, P_pred = kalman_filter_nd(
        y=y,
        u=u,
        A_t=A_t,
        B_t=B_t,
        H_t=H_t,
        D_t=D_t,
        Q_t=Q_t,
        R_t=R_t,
        m0=m0,
        P0=P0,
        observation_action=observation_action,
    )
    attack_t = int(attack_t)
    n_action = np.asarray(B_t).shape[2]
    obs_action = _legacy_observation_action(
        np.asarray(u, dtype=float),
        attack_t,
        convention=observation_action,
        action_dim=n_action,
    )
    H_obs = np.asarray(H_t, dtype=float)[attack_t]
    D_obs = np.asarray(D_t, dtype=float)[attack_t]
    P_obs = project_to_psd(P_pred[attack_t])
    R_obs = project_to_psd(np.asarray(R_t, dtype=float)[attack_t])
    mu_obs = H_obs @ m_pred[attack_t] + D_obs @ obs_action
    Sigma_obs = project_to_psd(H_obs @ P_obs @ H_obs.T + R_obs)
    K_obs = solve_spd(Sigma_obs, H_obs @ P_obs.T).T
    return K_obs, mu_obs, Sigma_obs


def build_kf_attack(
    *,
    y_clean: np.ndarray,
    u_controls: np.ndarray,
    mats: dict[str, np.ndarray],
    attack_t: int,
    m0: np.ndarray,
    P0: np.ndarray,
    epsilon: float,
    observation_action: ObservationActionConvention = "previous",
) -> dict[str, np.ndarray | float]:
    """Build a single-time KF-only KKT attack for covariance-adaptation scripts."""
    X_t, mu_t, Sigma_t = compute_filter_predictive_quantities(
        y=y_clean,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        attack_t=attack_t,
        m0=m0,
        P0=P0,
        observation_action=observation_action,
    )
    y_adv = np.asarray(y_clean, dtype=float).copy()
    y_adv[int(attack_t)], obj_star = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_t,
        y_t=np.asarray(y_clean, dtype=float)[int(attack_t)],
        mu=mu_t,
        Sigma=Sigma_t,
        epsilon=float(epsilon),
    )
    return {
        "X_t": X_t,
        "mu_t": mu_t,
        "Sigma_t": Sigma_t,
        "y_adv": y_adv,
        "adv_target": np.asarray(y_adv[int(attack_t)], dtype=float),
        "obj_star": float(obj_star),
    }


def kalman_filter_with_online_covariance_adaptation(
    *,
    y: np.ndarray,
    u: np.ndarray,
    A_t: np.ndarray,
    B_t: np.ndarray,
    H_t: np.ndarray,
    D_t: np.ndarray,
    Q_t: np.ndarray,
    R_t: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    attack_targets: dict[int, np.ndarray] | None,
    lam: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    objective_attack_score_builder: Callable[
        [
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray,
        ],
        float | None,
    ]
    | None = None,
    mahalanobis_epsilon: float = 1.0,
    mahalanobis_evidence_weight: float = 0.5,
    objective_evidence_weight: float = 0.5,
    direction_eps: float = 1e-10,
    observation_action: ObservationActionConvention = "previous",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Legacy online directional covariance adaptation.

    This mirrors the historical return signature while using the shared
    covariance-adaptation primitives.
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    A_t = np.asarray(A_t, dtype=float)
    B_t = np.asarray(B_t, dtype=float)
    H_t = np.asarray(H_t, dtype=float)
    D_t = np.asarray(D_t, dtype=float)
    Q_t = np.asarray(Q_t, dtype=float)
    R_t = np.asarray(R_t, dtype=float)
    m0 = np.asarray(m0, dtype=float).reshape(-1)
    P0 = project_to_psd(np.asarray(P0, dtype=float))

    T = y.shape[0] - 1
    n_state = m0.size
    n_obs = y.shape[1]
    n_action = B_t.shape[2]
    identity = np.eye(n_state, dtype=float)

    m_pred = np.zeros((T + 1, n_state), dtype=float)
    P_pred = np.zeros((T + 1, n_state, n_state), dtype=float)
    m_filt = np.zeros((T + 1, n_state), dtype=float)
    P_filt = np.zeros((T + 1, n_state, n_state), dtype=float)
    diagnostics = {
        "pi_t": np.zeros(T + 1, dtype=float),
        "gamma_t": np.zeros(T + 1, dtype=float),
        "bar_gamma_t": np.zeros(T + 1, dtype=float),
        "r_hidden_t": np.zeros(T + 1, dtype=float),
        "r_obs_t": np.zeros(T + 1, dtype=float),
        "u_t": np.zeros((T + 1, n_obs), dtype=float),
        "V_tilde_t": np.zeros((T + 1, n_obs, n_obs), dtype=float),
        "K_tilde_t": np.zeros((T + 1, n_state, n_obs), dtype=float),
        "mu_poe_t": np.zeros((T + 1, n_obs), dtype=float),
        "Sigma_poe_t": np.zeros((T + 1, n_obs, n_obs), dtype=float),
    }

    m_pred[0] = m0
    P_pred[0] = P0

    for time_idx in range(T + 1):
        H_obs = H_t[time_idx]
        R_obs = project_to_psd(R_t[time_idx])
        obs_action = _legacy_observation_action(
            u,
            time_idx,
            convention=observation_action,
            action_dim=n_action,
        )
        y_hat = H_obs @ m_pred[time_idx] + D_t[time_idx] @ obs_action
        S_nom = project_to_psd(H_obs @ P_pred[time_idx] @ H_obs.T + R_obs)
        innovation = y[time_idx] - y_hat

        V_tilde = R_obs.copy()
        S_tilde = S_nom.copy()
        K_tilde = solve_spd(S_tilde, H_obs @ P_pred[time_idx].T).T
        mu_poe = y_hat.copy()
        Sigma_poe = S_nom.copy()
        pi_t = gamma_t = bar_gamma_t = r_hidden = r_obs = 0.0
        direction = np.zeros(n_obs, dtype=float)

        if attack_targets is not None and time_idx in attack_targets:
            target = np.asarray(attack_targets[time_idx], dtype=float).reshape(n_obs)
            delta_adv = target - y_hat
            norm_innovation = float(np.linalg.norm(innovation))
            if norm_innovation >= float(direction_eps):
                direction = innovation / norm_innovation
                pi_t, r_hidden, r_obs = compute_contamination_prior(
                    delta_adv=delta_adv,
                    predictive_observation_covariance=S_nom,
                    predictive_state_covariance=P_pred[time_idx],
                    observation_matrix=H_obs,
                    omega_h=float(omega_h),
                    omega_o=float(omega_o),
                )
                K_nominal = solve_spd(S_nom, H_obs @ P_pred[time_idx].T).T
                m_post_observed = m_pred[time_idx] + K_nominal @ innovation
                m_post_target = m_pred[time_idx] + K_nominal @ (target - y_hat)
                P_post_nominal = project_to_psd(
                    (identity - K_nominal @ H_obs) @ P_pred[time_idx] @ (identity - K_nominal @ H_obs).T
                    + K_nominal @ R_obs @ K_nominal.T
                )
                objective_attack_score = 0.0
                if objective_attack_score_builder is not None:
                    built_score = objective_attack_score_builder(
                        int(time_idx),
                        np.asarray(y[time_idx], dtype=float).reshape(n_obs),
                        np.asarray(y_hat, dtype=float).reshape(n_obs),
                        target,
                        np.asarray(direction, dtype=float).reshape(n_obs),
                        np.asarray(m_post_observed, dtype=float).reshape(n_state),
                        np.asarray(m_post_target, dtype=float).reshape(n_state),
                        np.asarray(P_post_nominal, dtype=float),
                    )
                    if built_score is not None:
                        objective_attack_score = clip_attack_evidence_score(float(built_score))
                gamma_t = posterior_attack_probability_from_evidence(
                    prior_probability=float(pi_t),
                    mahalanobis_attack_score=float(
                        mahalanobis_attack_evidence(
                            innovation=innovation,
                            predictive_observation_covariance=S_nom,
                            epsilon=float(mahalanobis_epsilon),
                        )
                    ),
                    objective_attack_score=float(objective_attack_score),
                    mahalanobis_weight=float(mahalanobis_evidence_weight),
                    objective_weight=float(objective_evidence_weight),
                )
                bar_gamma_t = gamma_t if gamma_t >= float(delta_threshold) else 0.0
                V_tilde = rank_one_covariance_update(R_obs, float(lam), direction, weight=bar_gamma_t)
                S_tilde = project_to_psd(H_obs @ P_pred[time_idx] @ H_obs.T + V_tilde)
                K_tilde = solve_spd(S_tilde, H_obs @ P_pred[time_idx].T).T
                mu_poe = target.copy()
                Sigma_poe = P_post_nominal.copy()

        m_filt[time_idx] = m_pred[time_idx] + K_tilde @ innovation
        left = identity - K_tilde @ H_obs
        P_filt[time_idx] = project_to_psd(left @ P_pred[time_idx] @ left.T + K_tilde @ V_tilde @ K_tilde.T)

        diagnostics["pi_t"][time_idx] = pi_t
        diagnostics["gamma_t"][time_idx] = gamma_t
        diagnostics["bar_gamma_t"][time_idx] = bar_gamma_t
        diagnostics["r_hidden_t"][time_idx] = r_hidden
        diagnostics["r_obs_t"][time_idx] = r_obs
        diagnostics["u_t"][time_idx] = direction
        diagnostics["V_tilde_t"][time_idx] = V_tilde
        diagnostics["K_tilde_t"][time_idx] = K_tilde
        diagnostics["mu_poe_t"][time_idx] = mu_poe
        diagnostics["Sigma_poe_t"][time_idx] = Sigma_poe

        if time_idx < T:
            action = _legacy_transition_action(u, time_idx, action_dim=n_action)
            m_pred[time_idx + 1] = A_t[time_idx] @ m_filt[time_idx] + B_t[time_idx] @ action
            P_pred[time_idx + 1] = project_to_psd(
                A_t[time_idx] @ P_filt[time_idx] @ A_t[time_idx].T + Q_t[time_idx]
            )

    return m_filt, P_filt, m_pred, P_pred, diagnostics


def kalman_filter_with_online_covariance_adaptation_current_observation(*args, **kwargs):
    """Covariance adaptation with current-control direct observations."""
    kwargs.setdefault("observation_action", "current")
    return kalman_filter_with_online_covariance_adaptation(*args, **kwargs)


def _legacy_transition_action(u: np.ndarray, time_idx: int, *, action_dim: int) -> np.ndarray:
    """Return `u[time_idx]` for transition `s_t -> s_{t+1}`."""
    if u.size == 0:
        return np.zeros(action_dim, dtype=float)
    clipped_idx = min(max(int(time_idx), 0), u.shape[0] - 1)
    return np.asarray(u[clipped_idx], dtype=float).reshape(action_dim)


def _legacy_observation_action(
    u: np.ndarray,
    time_idx: int,
    *,
    convention: ObservationActionConvention,
    action_dim: int,
) -> np.ndarray:
    """Return the direct observation action under a legacy convention."""
    if u.size == 0:
        return np.zeros(action_dim, dtype=float)
    if convention == "current":
        action_idx = int(time_idx)
    elif convention == "previous":
        action_idx = 0 if int(time_idx) == 0 else int(time_idx) - 1
    else:
        raise ValueError("Unsupported observation action convention.")
    action_idx = min(max(action_idx, 0), u.shape[0] - 1)
    return np.asarray(u[action_idx], dtype=float).reshape(action_dim)


def _constant_1d_sequences(
    A: float,
    B: float,
    H: float,
    D: float,
    Q: float,
    R: float,
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return constant scalar matrices as legacy stacked arrays."""
    A_t = np.repeat(np.array([[[float(A)]]]), T + 1, axis=0)
    B_t = np.repeat(np.array([[[float(B)]]]), T + 1, axis=0)
    H_t = np.repeat(np.array([[[float(H)]]]), T + 1, axis=0)
    D_t = np.repeat(np.array([[[float(D)]]]), T + 1, axis=0)
    Q_t = np.repeat(np.array([[[float(Q)]]]), T + 1, axis=0)
    R_t = np.repeat(np.array([[[float(R)]]]), T + 1, axis=0)
    return A_t, B_t, H_t, D_t, Q_t, R_t


def _as_1d_output(value: np.ndarray | float) -> np.ndarray:
    """Normalize function output to a one-dimensional array."""
    return np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)


def _as_2d_jacobian(jacobian: np.ndarray, n_state: int) -> np.ndarray:
    """Normalize gradient/Jacobian output to shape `(d_out, d_state)`."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim == 1:
        if jacobian.shape[0] != int(n_state):
            raise ValueError("Gradient has incompatible shape.")
        return jacobian[None, :]
    if jacobian.ndim == 2:
        if jacobian.shape[1] != int(n_state):
            raise ValueError("Jacobian has incompatible state dimension.")
        return jacobian
    raise ValueError("Jacobian must be one- or two-dimensional.")
