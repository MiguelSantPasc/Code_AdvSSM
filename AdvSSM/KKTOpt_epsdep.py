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
import numpy as np
import matplotlib.pyplot as plt

try:
    from AdvSSM.io_utils import cached_npz, data_dir_for, figures_dir_for
except ModuleNotFoundError:
    from io_utils import cached_npz, data_dir_for, figures_dir_for


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

    def u_state_at(k: int) -> np.ndarray:
        """
        Input used in state transition x[k] -> x[k+1]:
            x[k+1] = A_t[k] x[k] + B_t[k] u[k] + w
        """
        if not (0 <= k < T):
            raise IndexError(f"u_state_at({k}) out of range for T={T}")
        return u[k]

    def u_meas_at(k: int) -> np.ndarray:
        """
        Input used in measurement y[k].

        Consistent with simulate_lgssm_nd:
          y[0] = H_0 x[0] + D_0 u[0] + v_0
          y[k] = H_k x[k] + D_k u[k-1] + v_k   for k >= 1
        """
        if T == 0:
            return np.zeros(D_t.shape[2], dtype=float)
        if k == 0:
            return u[0]
        if 1 <= k <= T:
            return u[k - 1]
        raise IndexError(f"u_meas_at({k}) out of range for T={T}")

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
        uk_meas = u_meas_at(k)

        y_hat = Hk @ m_pred[k] + Dk @ uk_meas
        S = Hk @ P_pred[k] @ Hk.T + Rk
        K = P_pred[k] @ Hk.T @ np.linalg.inv(S)

        innov = y[k] - y_hat
        m_filt[k] = m_pred[k] + K @ innov
        P_filt[k] = (I - K @ Hk) @ P_pred[k]

        if k < T:
            Ak = A_t[k]
            Bk = B_t[k]
            Qk = project_to_psd(Q_t[k])
            uk_state = u_state_at(k)

            # FIXED: use u[k], not u[k+1]
            m_pred[k + 1] = Ak @ m_filt[k] + Bk @ uk_state
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
# Leave-one-out p(y_t | y_-t) + X_t (corrected indexing)
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
    n_u = D_t.shape[2]
    I_x = np.eye(n_x)

    def u_state_at(k: int) -> np.ndarray:
        if not (0 <= k < T):
            raise IndexError(f"u_state_at({k}) out of range for T={T}")
        return u[k]

    def u_meas_at(k: int) -> np.ndarray:
        if T == 0:
            return np.zeros(n_u, dtype=float)
        if k == 0:
            return u[0]
        if 1 <= k <= T:
            return u[k - 1]
        raise IndexError(f"u_meas_at({k}) out of range for T={T}")

    # --------------------------------------------------------
    # KF covariances + gains (needed for X_t)
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # X_t formula
    # --------------------------------------------------------
    l = T - t
    total = np.zeros((n_x, n_x))
    for i in range(l + 1):
        prodJ = np.eye(n_x) if i == 0 else prod_right([J[t + j] for j in range(i)])

        if t + i >= T:
            mid = np.eye(n_x)
        else:
            mid = np.eye(n_x) - J[t + i] @ A_t[t + i]

        factors = [((np.eye(n_x) - K_kf[t + j] @ H_t[t + j]) @ A_t[t + j]) for j in range(i + 1) if (t + j) < T]
        prodKH = prod_left(factors) if len(factors) > 0 else np.eye(n_x)

        total = total + (prodJ @ mid @ prodKH)

    X_t_out = total @ K_kf[t]

    # --------------------------------------------------------
    # Forward means with y_t excluded
    # --------------------------------------------------------
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
            uk_meas = u_meas_at(k)

            y_hat = Hk @ m_pred[k] + Dk @ uk_meas
            S = Hk @ P_pred[k] @ Hk.T + Rk
            K = P_pred[k] @ Hk.T @ np.linalg.inv(S)

            m_filt[k] = m_pred[k] + K @ (y[k] - y_hat)

        if k < T:
            Ak = A_t[k]
            Bk = B_t[k]
            uk_state = u_state_at(k)

            # FIXED: use u[k], not u[k+1]
            m_pred[k + 1] = Ak @ m_filt[k] + Bk @ uk_state

    # --------------------------------------------------------
    # Backward info-form messages excluding measurement at t
    # --------------------------------------------------------
    Lambda = [None] * (T + 1)
    eta = [None] * (T + 1)
    Lambda[T] = np.zeros((n_x, n_x))
    eta[T] = np.zeros((n_x,))

    for k in range(T - 1, -1, -1):
        kp1 = k + 1

        # Transition x_k -> x_{k+1} uses index k
        Ak = A_t[k]
        Bk = B_t[k]
        Qk = project_to_psd(Q_t[k])

        # Measurement at time k+1 uses index kp1
        Hkp1 = H_t[kp1]
        Dkp1 = D_t[kp1]
        Rkp1 = project_to_psd(R_t[kp1])
        ukp1_meas = u_meas_at(kp1)
        uk_state = u_state_at(k)

        if kp1 == t:
            barLambda = Lambda[kp1]
            barEta = eta[kp1]
        else:
            tilde_y = y[kp1] - Dkp1 @ ukp1_meas
            Rinv = np.linalg.inv(Rkp1)
            barLambda = Lambda[kp1] + Hkp1.T @ Rinv @ Hkp1
            barEta = eta[kp1] + Hkp1.T @ Rinv @ tilde_y

        Qinv = np.linalg.inv(Qk)
        S_back = Qinv + barLambda
        S_back_inv = np.linalg.inv(S_back)

        core = Qinv - Qinv @ S_back_inv @ Qinv
        Lambda[k] = Ak.T @ core @ Ak

        term1 = Ak.T @ Qinv @ S_back_inv @ barEta
        term2 = Ak.T @ Qinv @ S_back_inv @ barLambda @ (Bk @ uk_state)
        eta[k] = term1 - term2

    # --------------------------------------------------------
    # Combine at time t to get p(y_t | y_-t)
    # --------------------------------------------------------
    P_t_minus = np.linalg.inv(np.linalg.inv(P_pred[t]) + Lambda[t])
    m_t_minus = P_t_minus @ (np.linalg.inv(P_pred[t]) @ m_pred[t] + eta[t])

    mu_y = H_t[t] @ m_t_minus + D_t[t] @ u_meas_at(t)
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


if __name__ == "__main__":
    main()
