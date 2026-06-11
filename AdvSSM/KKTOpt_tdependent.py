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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from AdvSSM.io_utils import data_dir_for, figures_dir_for
except ModuleNotFoundError:
    from io_utils import data_dir_for, figures_dir_for


# ============================================================
# Linear algebra helpers (PSD-safe)
# ============================================================
def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def project_to_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=float))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(w) @ V.T


def sqrtm_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=float))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(np.sqrt(w)) @ V.T


# ============================================================
# ND LGSSM simulator (time-varying matrices via optional drift)
# ============================================================
def simulate_lgssm_nd(
    A0: np.ndarray,              # (n_x, n_x)
    B0: np.ndarray,              # (n_x, n_u)
    H0: np.ndarray,              # (n_y, n_x)
    D0: np.ndarray,              # (n_y, n_u)
    T: int,
    seed: int = 123,
    x0: np.ndarray | None = None,    # (n_x,)
    Q0: np.ndarray | None = None,    # (n_x, n_x)
    R0: np.ndarray | None = None,    # (n_y, n_y)
    dA: np.ndarray | None = None,
    dB: np.ndarray | None = None,
    dH: np.ndarray | None = None,
    dD: np.ndarray | None = None,
    dQ: np.ndarray | None = None,
    dR: np.ndarray | None = None,
    u_low: float = -1.5,
    u_high: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Model (t=0..T-1):
      x_{t+1} = A_t x_t + B_t u_t + w_{t+1},   w_{t+1} ~ N(0, Q_t)
      y_t     = H_t x_t + D_t u_t + v_t,       v_t     ~ N(0, R_t)

    Drift:
      A_t = A0 + dA*t, etc. (if provided)
    """
    if T < 0:
        raise ValueError("T must be >= 0")

    rng = np.random.default_rng(seed)

    A0 = np.asarray(A0, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    H0 = np.asarray(H0, dtype=float)
    D0 = np.asarray(D0, dtype=float)

    n_x = A0.shape[0]
    n_u = B0.shape[1]
    n_y = H0.shape[0]

    if x0 is None:
        x0 = np.zeros(n_x, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (n_x,):
            raise ValueError("x0 must be (n_x,)")

    if Q0 is None:
        Q0 = 0.02 * np.eye(n_x, dtype=float)
    else:
        Q0 = np.asarray(Q0, dtype=float)

    if R0 is None:
        R0 = 0.03 * np.eye(n_y, dtype=float)
    else:
        R0 = np.asarray(R0, dtype=float)

    Q0 = project_to_psd(Q0)
    R0 = project_to_psd(R0)

    dA = np.zeros_like(A0) if dA is None else np.asarray(dA, dtype=float)
    dB = np.zeros_like(B0) if dB is None else np.asarray(dB, dtype=float)
    dH = np.zeros_like(H0) if dH is None else np.asarray(dH, dtype=float)
    dD = np.zeros_like(D0) if dD is None else np.asarray(dD, dtype=float)
    dQ = np.zeros_like(Q0) if dQ is None else np.asarray(dQ, dtype=float)
    dR = np.zeros_like(R0) if dR is None else np.asarray(dR, dtype=float)

    A_t = np.zeros((T + 1, n_x, n_x), dtype=float)
    B_t = np.zeros((T + 1, n_x, n_u), dtype=float)
    H_t = np.zeros((T + 1, n_y, n_x), dtype=float)
    D_t = np.zeros((T + 1, n_y, n_u), dtype=float)
    Q_t = np.zeros((T + 1, n_x, n_x), dtype=float)
    R_t = np.zeros((T + 1, n_y, n_y), dtype=float)

    for t in range(T + 1):
        A_t[t] = A0 + dA * t
        B_t[t] = B0 + dB * t
        H_t[t] = H0 + dH * t
        D_t[t] = D0 + dD * t
        Q_t[t] = project_to_psd(Q0 + dQ * t)
        R_t[t] = project_to_psd(R0 + dR * t)

    u = rng.uniform(u_low, u_high, size=(T, n_u))

    x = np.zeros((T + 1, n_x), dtype=float)
    y = np.zeros((T + 1, n_y), dtype=float)
    x[0] = x0

    if T > 0:
        y[0] = H_t[0] @ x[0] + D_t[0] @ u[0] + rng.multivariate_normal(np.zeros(n_y), R_t[0])
    else:
        y[0] = H_t[0] @ x[0] + rng.multivariate_normal(np.zeros(n_y), R_t[0])

    for t in range(T):
        w_next = rng.multivariate_normal(np.zeros(n_x), Q_t[t])
        v_next = rng.multivariate_normal(np.zeros(n_y), R_t[t + 1])
        x[t + 1] = A_t[t] @ x[t] + B_t[t] @ u[t] + w_next
        y[t + 1] = H_t[t + 1] @ x[t + 1] + D_t[t + 1] @ u[t] + v_next

    mats = {"A_t": A_t, "B_t": B_t, "H_t": H_t, "D_t": D_t, "Q_t": Q_t, "R_t": R_t}
    return x, y, u, mats


# ============================================================
# Kalman Filter + RTS smoother (ND)
# ============================================================
def kalman_filter_nd(
    *,
    y: np.ndarray,          # (T+1, n_y)
    u: np.ndarray,          # (T,   n_u)
    A_t: np.ndarray,        # (T+1, n_x, n_x)
    B_t: np.ndarray,        # (T+1, n_x, n_u)
    H_t: np.ndarray,        # (T+1, n_y, n_x)
    D_t: np.ndarray,        # (T+1, n_y, n_u)
    Q_t: np.ndarray,        # (T+1, n_x, n_x)
    R_t: np.ndarray,        # (T+1, n_y, n_y)
    m0: np.ndarray,         # (n_x,)
    P0: np.ndarray,         # (n_x,n_x)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = y.shape[0] - 1
    n_x = P0.shape[0]
    I = np.eye(n_x)

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

    m_pred = np.zeros((T + 1, n_x))
    P_pred = np.zeros((T + 1, n_x, n_x))
    m_filt = np.zeros((T + 1, n_x))
    P_filt = np.zeros((T + 1, n_x, n_x))

    m_pred[0] = m0
    P_pred[0] = P0

    for k in range(T + 1):
        Hk = H_t[k]
        Dk = D_t[k]
        Rk = project_to_psd(R_t[k])
        uk = u_at(k)

        y_hat = Hk @ m_pred[k] + Dk @ uk
        S = Hk @ P_pred[k] @ Hk.T + Rk
        K = P_pred[k] @ Hk.T @ np.linalg.inv(S)

        innov = y[k] - y_hat
        m_filt[k] = m_pred[k] + K @ innov
        P_filt[k] = (I - K @ Hk) @ P_pred[k]

        if k < T:
            Ak = A_t[k]
            Bk = B_t[k]
            Qk = project_to_psd(Q_t[k])
            m_pred[k + 1] = Ak @ m_filt[k] + Bk @ u_at(k + 1)
            P_pred[k + 1] = Ak @ P_filt[k] @ Ak.T + Qk

    return m_filt, P_filt, m_pred, P_pred


def rts_smoother_nd(
    *,
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = m_filt.shape[0] - 1
    m_smooth = np.zeros_like(m_filt)
    P_smooth = np.zeros_like(P_filt)

    m_smooth[T] = m_filt[T]
    P_smooth[T] = P_filt[T]

    for k in range(T - 1, -1, -1):
        Ak = A_t[k]
        Ck = P_filt[k] @ Ak.T @ np.linalg.inv(P_pred[k + 1])
        m_smooth[k] = m_filt[k] + Ck @ (m_smooth[k + 1] - m_pred[k + 1])
        P_smooth[k] = P_filt[k] + Ck @ (P_smooth[k + 1] - P_pred[k + 1]) @ Ck.T

    return m_smooth, P_smooth


# ============================================================
# Leave-one-out p(y_t | y_-t) + X_t (your formula)
# ============================================================
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
) -> list[np.ndarray]:
    """
    Returns [X_t, mu_y_t_given_minus_t, Sigma_y_t_given_minus_t].
    """
    T = int(y.shape[0] - 1)
    if not (0 <= t <= T):
        raise ValueError("t out of range")

    n_x = P0.shape[0]
    I_x = np.eye(n_x)

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

    # KF covariances + gains (for X_t)
    P_pred = [None] * (T + 1)
    P_filt = [None] * (T + 1)
    K_kf = [None] * (T + 1)

    P_pred[0] = P0.copy()

    for k in range(T + 1):
        Hk = H_t[k]
        Rk = project_to_psd(R_t[k])

        S = Hk @ P_pred[k] @ Hk.T + Rk
        K = P_pred[k] @ Hk.T @ np.linalg.inv(S)

        P_filt[k] = (I_x - K @ Hk) @ P_pred[k]
        K_kf[k] = K

        if k < T:
            Ak = A_t[k]
            Qk = project_to_psd(Q_t[k])
            P_pred[k + 1] = Ak @ P_filt[k] @ Ak.T + Qk

    # RTS gains
    J = [np.zeros((n_x, n_x)) for _ in range(T + 1)]
    for k in range(T):
        Ak = A_t[k]
        J[k] = P_filt[k] @ Ak.T @ np.linalg.inv(P_pred[k + 1])

    def prod_right(mats: list[np.ndarray]) -> np.ndarray:
        out = np.eye(n_x)
        for M in mats:
            out = out @ M
        return out

    def prod_left(mats: list[np.ndarray]) -> np.ndarray:
        out = np.eye(n_x)
        for M in reversed(mats):
            out = out @ M
        return out

    # Your X_t formula
    l = T - t
    total = np.zeros((n_x, n_x))
    for i in range(l + 1):
        prodJ = np.eye(n_x) if i == 0 else prod_right([J[t + j] for j in range(i)])

        if t + i >= T:
            mid = np.eye(n_x)
        else:
            mid = np.eye(n_x) - J[t + i] @ A_t[t + i + 1]

        factors = [((np.eye(n_x) - K_kf[t + j] @ H_t[t + j]) @ A_t[t + j]) for j in range(i + 1)]
        prodKH = prod_left(factors)

        total = total + (prodJ @ mid @ prodKH)

    X_t_out = total @ K_kf[t]

    # Forward means with y_t excluded
    m_pred = [None] * (T + 1)
    m_filt = [None] * (T + 1)
    m_pred[0] = m0.copy()

    for k in range(T + 1):
        if k == t:
            m_filt[k] = m_pred[k]
        else:
            Hk = H_t[k]
            Dk = D_t[k]
            Rk = project_to_psd(R_t[k])
            uk = u_at(k)

            y_hat = Hk @ m_pred[k] + Dk @ uk
            S = Hk @ P_pred[k] @ Hk.T + Rk
            K = P_pred[k] @ Hk.T @ np.linalg.inv(S)

            m_filt[k] = m_pred[k] + K @ (y[k] - y_hat)

        if k < T:
            Ak = A_t[k]
            Bk = B_t[k]
            m_pred[k + 1] = Ak @ m_filt[k] + Bk @ u_at(k + 1)

    # Backward info-form messages excluding measurement at t
    Lambda = [None] * (T + 1)
    eta = [None] * (T + 1)
    Lambda[T] = np.zeros((n_x, n_x))
    eta[T] = np.zeros((n_x,))

    for k in range(T - 1, -1, -1):
        kp1 = k + 1

        Akp1 = A_t[kp1]
        Bkp1 = B_t[kp1]
        Qkp1 = project_to_psd(Q_t[kp1])
        Hkp1 = H_t[kp1]
        Dkp1 = D_t[kp1]
        Rkp1 = project_to_psd(R_t[kp1])
        ukp1 = u_at(kp1)

        if kp1 == t:
            barLambda = Lambda[kp1]
            barEta = eta[kp1]
        else:
            tilde_y = y[kp1] - Dkp1 @ ukp1
            Rinv = np.linalg.inv(Rkp1)
            barLambda = Lambda[kp1] + Hkp1.T @ Rinv @ Hkp1
            barEta = eta[kp1] + Hkp1.T @ Rinv @ tilde_y

        Qinv = np.linalg.inv(Qkp1)
        S_back = Qinv + barLambda
        S_back_inv = np.linalg.inv(S_back)

        core = Qinv - Qinv @ S_back_inv @ Qinv
        Lambda[k] = Akp1.T @ core @ Akp1

        term1 = Akp1.T @ Qinv @ S_back_inv @ barEta
        term2 = Akp1.T @ Qinv @ S_back_inv @ barLambda @ (Bkp1 @ ukp1)
        eta[k] = term1 - term2

    # Combine at time t to get p(y_t | y_-t)
    P_t_minus = np.linalg.inv(np.linalg.inv(P_pred[t]) + Lambda[t])
    m_t_minus = P_t_minus @ (np.linalg.inv(P_pred[t]) @ m_pred[t] + eta[t])

    mu_y = H_t[t] @ m_t_minus + D_t[t] @ u_at(t)
    Sigma_y = H_t[t] @ P_t_minus @ H_t[t].T + project_to_psd(R_t[t])

    return [X_t_out, mu_y, Sigma_y]


# ============================================================
# KKT attack solver
# ============================================================
def solve_kkt_max_quadratic_over_ellipsoid(
    *,
    X: np.ndarray,          # (n_x, n_y)
    y_t: np.ndarray,        # (n_y,)
    mu: np.ndarray,         # (n_y,)
    Sigma: np.ndarray,      # (n_y, n_y)
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 250,
) -> tuple[np.ndarray, float]:
    """
    Solve:
      maximize_y ||X (y - y_t)||^2
      s.t.       (y - mu)^T Sigma^{-1} (y - mu) <= epsilon
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    Sigma = project_to_psd(Sigma)
    S = sqrtm_psd(Sigma)
    M = project_to_psd(symmetrize(X.T @ X))

    d = (mu - y_t).reshape(-1)
    A = symmetrize(S.T @ M @ S)
    b = (S.T @ M @ d).reshape(-1)

    a, U = np.linalg.eigh(A)
    a_max = float(np.max(a))
    bp = U.T @ b

    if np.linalg.norm(b) < 1e-14:
        idx = int(np.argmax(a))
        zp = np.zeros_like(bp)
        zp[idx] = np.sqrt(epsilon)
        z = U @ zp
    else:
        def g(lam: float) -> float:
            zi = -bp / (a - lam)
            return float(np.dot(zi, zi) - epsilon)

        lam_low = a_max + 1e-12
        f_low = g(lam_low)
        if f_low <= 0:
            lam_low = a_max + 1e-16
            f_low = g(lam_low)

        lam_high = a_max + 1.0
        f_high = g(lam_high)
        while f_high > 0:
            lam_high *= 2.0
            f_high = g(lam_high)
            if lam_high > 1e14:
                raise RuntimeError("Failed to bracket lambda in KKT solve.")

        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_low + lam_high)
            f_mid = g(lam_mid)
            if abs(f_mid) < tol:
                lam_low = lam_high = lam_mid
                break
            if f_mid > 0:
                lam_low = lam_mid
            else:
                lam_high = lam_mid

        lam_star = 0.5 * (lam_low + lam_high)
        zp = -bp / (a - lam_star)
        z = U @ zp

        nz = np.linalg.norm(z)
        if nz > 0:
            z = z * (np.sqrt(epsilon) / nz)

    y_star = mu + S @ z
    obj_star = float(np.linalg.norm(X @ (y_star - y_t)) ** 2)
    return y_star, obj_star


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
    Plot the mean local and global attack effects together in one panel so
    their dependence on the attacked time can be compared directly.
    """
    _set_plot_theme()

    c_local_line = "#5E738F"
    c_global_line = "#6E9181"
    t_offset = 0.05

    fig, ax = plt.subplots(figsize=(15.5, 6.2), constrained_layout=True)
    _style_axis(ax)
    ax.spines["left"].set_position(("axes", 0.03))
    ax_right = ax.twinx()
    ax_right.set_facecolor("none")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_color("black")
    ax_right.spines["right"].set_alpha(0.9)
    ax_right.spines["right"].set_position(("axes", 0.97))
    ax_right.grid(False)
    
    local_means = np.array([
        np.nanmean(local_mat[:, i]) if np.any(np.isfinite(local_mat[:, i])) else np.nan
        for i in range(local_mat.shape[1])
    ], dtype=float)
    global_means = np.array([
        np.nanmean(global_mat[:, i]) if np.any(np.isfinite(global_mat[:, i])) else np.nan
        for i in range(global_mat.shape[1])
    ], dtype=float)
    local_t_values = t_values.astype(float) - t_offset
    global_t_values = t_values.astype(float) + t_offset

    ax.plot(
        local_t_values,
        local_means,
        color=c_local_line,
        marker="o",
        markersize=5.5,
        linewidth=2.3,
        zorder=3,
        label="Local mean",
    )

    ax_right.plot(
        global_t_values,
        global_means,
        color=c_global_line,
        marker="o",
        markersize=5.5,
        linewidth=2.3,
        zorder=3,
        label="Global mean",
    )

    finite_local = local_means[np.isfinite(local_means)]
    if finite_local.size > 0:
        y_min = float(np.min(finite_local))
        y_max = float(np.max(finite_local))
        span = max(y_max - y_min, 1e-8)
        pad = 0.12 * span

        if span < 1e-6:
            pad = 0.01 * max(abs(y_min), abs(y_max), 1.0)

        ax.set_ylim(y_min - pad, y_max + pad)

    finite_global = global_means[np.isfinite(global_means)]
    if finite_global.size > 0:
        y_min = float(np.min(finite_global))
        y_max = float(np.max(finite_global))
        span = max(y_max - y_min, 1e-8)
        pad = 0.12 * span

        if span < 1e-6:
            pad = 0.01 * max(abs(y_min), abs(y_max), 1.0)

        ax_right.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel("Attacked time step $t$")
    ax.set_ylabel("Mean local attack effect", color="black")
    ax_right.set_ylabel("Mean global attack effect", color="black")
    ax.set_xticks(t_values)
    ax.set_xlim(float(t_values[0]) - 0.75, float(t_values[-1]) + 0.75)
    ax.tick_params(axis="y", colors="black", direction="in", pad=8)
    ax_right.tick_params(axis="y", colors="black", direction="in", pad=8)

    legend_handles = [
        Line2D([0], [0], color=c_local_line, marker="o", linewidth=2.3, markersize=5.5, label="Local mean"),
        Line2D([0], [0], color=c_global_line, marker="o", linewidth=2.3, markersize=5.5, label="Global mean"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
    )

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
    T = 8
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
        f"mc_attack_effects_dual_axis_means_N{N_runs}_T{T}_eps{epsilon:.3f}.png"
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

if __name__ == "__main__":
    main()
