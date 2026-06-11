#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensitivity of the KKT observation attack with respect to epsilon.

For each run, the script simulates an ND linear Gaussian SSM,

    x_{k+1} = A_k x_k + B_k u_k + w_{k+1},    w_{k+1} ~ N(0, Q_k)
    y_k     = H_k x_k + D_k u_k + v_k,        v_k     ~ N(0, R_k),

then attacks one observation y_t inside the leave-one-out ellipsoid

    (y_t* - mu_{t|-t})^T Sigma_{t|-t}^{-1} (y_t* - mu_{t|-t}) <= epsilon,

where p(y_t | y_{-t}) = N(mu_{t|-t}, Sigma_{t|-t}). The KKT solution maximizes
the quadratic state perturbation ||X_t (y_t* - y_t)||^2 under that constraint.

The plotted response is the Euclidean distance at the attacked time between
the true state and either the baseline RTS smoother or the adversarial RTS
smoother. Repeating this for many seeds gives a bootstrap 95% confidence
interval for the mean sensitivity curve.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from AdvSSM.io_utils import cached_npz, data_path_for_plot
except ModuleNotFoundError:
    from io_utils import cached_npz, data_path_for_plot

# -----------------------------
# Utilities: PSD symmetrize + sqrt
# -----------------------------
def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def project_to_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(M)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(w) @ V.T

def sqrtm_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(M)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(np.sqrt(w)) @ V.T

# -----------------------------
# Simulator with drift (yours)
# -----------------------------
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
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

    if Q0 is None:
        Q0 = 0.02 * np.eye(n_x)
    else:
        Q0 = np.asarray(Q0, dtype=float)

    if R0 is None:
        R0 = 0.03 * np.eye(n_y)
    else:
        R0 = np.asarray(R0, dtype=float)

    # Ensure covariances are valid
    Q0 = project_to_psd(Q0)
    R0 = project_to_psd(R0)

    # Drift defaults
    dA = np.zeros_like(A0) if dA is None else np.asarray(dA, dtype=float)
    dB = np.zeros_like(B0) if dB is None else np.asarray(dB, dtype=float)
    dH = np.zeros_like(H0) if dH is None else np.asarray(dH, dtype=float)
    dD = np.zeros_like(D0) if dD is None else np.asarray(dD, dtype=float)
    dQ = np.zeros_like(Q0) if dQ is None else np.asarray(dQ, dtype=float)
    dR = np.zeros_like(R0) if dR is None else np.asarray(dR, dtype=float)

    A_t = np.zeros((T + 1, n_x, n_x))
    B_t = np.zeros((T + 1, n_x, n_u))
    H_t = np.zeros((T + 1, n_y, n_x))
    D_t = np.zeros((T + 1, n_y, n_u))
    Q_t = np.zeros((T + 1, n_x, n_x))
    R_t = np.zeros((T + 1, n_y, n_y))

    for t in range(T + 1):
        A_t[t] = A0 + dA * t
        B_t[t] = B0 + dB * t
        H_t[t] = H0 + dH * t
        D_t[t] = D0 + dD * t
        Q_t[t] = project_to_psd(Q0 + dQ * t)
        R_t[t] = project_to_psd(R0 + dR * t)

    u = rng.uniform(u_low, u_high, size=(T, n_u))

    x = np.zeros((T + 1, n_x))
    y = np.zeros((T + 1, n_y))
    x[0] = x0

    # y0
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


# -----------------------------
# LOO + X_t (yours, compact)
# -----------------------------
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
    T = int(y.shape[0] - 1)
    if not (0 <= t <= T):
        raise ValueError("t out of range")
    n_x = P0.shape[0]
    n_y = y.shape[1]
    I_x = np.eye(n_x)

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

    # KF covariances (for X)
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
    J[T] = np.zeros((n_x, n_x))

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

    X_t_out = total @ K_kf[t]  # (n_x, n_y)

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

    # Backward info messages (exclude measurement at time t)
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


# -----------------------------
# KKT solver (yours)
# -----------------------------
def solve_kkt_max_quadratic_over_ellipsoid(
    *,
    X: np.ndarray,
    y_t: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    Sigma = project_to_psd(Sigma)
    S = sqrtm_psd(Sigma)
    M = X.T @ X

    d = (mu - y_t).reshape(-1)
    A = symmetrize(S.T @ M @ S)
    b = (S.T @ M @ d).reshape(-1)

    a, U = np.linalg.eigh(A)
    a_max = float(np.max(a))
    bp = U.T @ b

    if np.linalg.norm(b) < 1e-14:
        idx = int(np.argmax(a))
        z_star = np.zeros_like(b)
        z_star[idx] = np.sqrt(epsilon)
        z_star = U @ z_star
    else:
        def norm2_minus_eps(lam: float) -> float:
            zi = -bp / (a - lam)
            return float(np.dot(zi, zi) - epsilon)

        lam_low = a_max + 1e-12
        f_low = norm2_minus_eps(lam_low)
        if f_low <= 0:
            lam_low = a_max + 1e-16
            f_low = norm2_minus_eps(lam_low)

        lam_high = a_max + 1.0
        f_high = norm2_minus_eps(lam_high)
        while f_high > 0:
            lam_high *= 2.0
            f_high = norm2_minus_eps(lam_high)
            if lam_high > 1e12:
                raise RuntimeError("Failed to bracket lambda.")

        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_low + lam_high)
            f_mid = norm2_minus_eps(lam_mid)
            if abs(f_mid) < tol:
                lam_low = lam_high = lam_mid
                break
            if f_mid > 0:
                lam_low = lam_mid
            else:
                lam_high = lam_mid

        lam_star = 0.5 * (lam_low + lam_high)
        z_star = U @ (-bp / (a - lam_star))

        nz = np.linalg.norm(z_star)
        if nz > 0:
            z_star *= (np.sqrt(epsilon) / nz)

    y_star = mu + S @ z_star
    obj_star = float(np.linalg.norm(X @ (y_star - y_t)) ** 2)
    return y_star, obj_star


# -----------------------------
# Kalman + RTS (yours)
# -----------------------------
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


# -----------------------------
# Bootstrap CI for the mean
# -----------------------------
def bootstrap_ci_of_mean(
    x: np.ndarray,
    n_boot: int = 600,
    alpha: float = 0.05,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x shape: (n_samples,) or (n_samples, n_eps)
    returns: mean, lo, hi (bootstrap percentile CI for mean)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = x.shape[0]

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)  # (n_boot,) or (n_boot, n_eps)

    mean = x.mean(axis=0)
    lo = np.quantile(boot_means, alpha / 2, axis=0)
    hi = np.quantile(boot_means, 1 - alpha / 2, axis=0)
    return mean, lo, hi


# -----------------------------
# Sensitivity experiment runner
# -----------------------------
def run_sensitivity_two_views(
    *,
    T: int,
    t_selected: list[int],
    eps_grid: np.ndarray,
    eps_fixed: float,
    n_seeds: int = 70,
    seed0: int = 2026,
    n_boot: int = 600,
    seed_boot_left: int = 999,
    seed_boot_right: int = 1234,
) -> dict:
    """
    Devuelve resultados para:
      (A) ratio vs eps en t_selected
      (B) ratio vs t (todos los t=0..T) para eps_fixed
    """

    t_selected = [int(t) for t in t_selected]
    for t in t_selected:
        if not (0 <= t <= T):
            raise ValueError(f"t_selected contiene t={t} fuera de rango [0, T] con T={T}")

    eps_grid = np.asarray(eps_grid, dtype=float)
    eps_grid = eps_grid[eps_grid > 0]
    if eps_grid.size == 0:
        raise ValueError("eps_grid debe tener epsilons > 0")

    eps_fixed = float(eps_fixed)
    if eps_fixed <= 0:
        raise ValueError("eps_fixed debe ser > 0")

    n_sel = len(t_selected)
    n_eps = eps_grid.size

    # -----------------------
    # MODELO (igual que tu ejemplo)
    # -----------------------
    n_x = n_y = n_u = 2

    A0 = np.array([[0.65, 0.40],
                   [-0.15, 0.70]], dtype=float)

    B0 = np.array([[1.65, 0.40],
                   [-0.15, 0.70]], dtype=float)

    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.03 * np.array([[1.6, -0.40],
                          [0.15, 0.70]], dtype=float)

    R0 = 0.02 * np.array([[0.65, 0.40],
                          [-0.15, 1.70]], dtype=float)

    Q0 = project_to_psd(Q0)
    R0 = project_to_psd(R0)

    dA = np.zeros_like(A0)
    dB = np.zeros_like(B0)
    dH = np.zeros_like(H0)
    dD = np.zeros_like(D0)
    dQ = np.zeros_like(Q0)
    dR = np.zeros_like(R0)

    x0 = np.array([0.5, 0.5])
    m0 = x0.copy()
    P0 = 0.05 * np.eye(n_x)

    # -----------------------
    # ARRAYS: por semilla
    # -----------------------
    # Vista A (t_selected x eps_grid)
    d_base_sel = np.zeros((n_seeds, n_sel), dtype=float)
    d_adv0_sel = np.zeros((n_seeds, n_sel), dtype=float)
    d_adv_sel  = np.zeros((n_seeds, n_sel, n_eps), dtype=float)

    # Vista B (todos los t) para eps_fixed
    d_base_all = np.zeros((n_seeds, T + 1), dtype=float)
    d_adv0_all = np.zeros((n_seeds, T + 1), dtype=float)
    d_adv_fix  = np.zeros((n_seeds, T + 1), dtype=float)

    for i in range(n_seeds):
        seed = seed0 + i

        # --- simula
        x, y, u, mats = simulate_lgssm_nd(
            A0=A0, B0=B0, H0=H0, D0=D0,
            T=T, seed=seed, x0=x0,
            Q0=Q0, R0=R0,
            dA=dA, dB=dB, dH=dH, dD=dD, dQ=dQ, dR=dR,
            u_low=-0.5, u_high=0.5,
        )

        # --- baseline RTS (una vez)
        m_filt_b, P_filt_b, m_pred_b, P_pred_b = kalman_filter_nd(
            y=y, u=u,
            A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
            Q_t=mats["Q_t"], R_t=mats["R_t"],
            m0=m0, P0=P0
        )
        m_smooth_b, P_smooth_b = rts_smoother_nd(
            m_filt=m_filt_b, P_filt=P_filt_b,
            m_pred=m_pred_b, P_pred=P_pred_b,
            A_t=mats["A_t"]
        )

        # baseline error para todos los t
        for t in range(T + 1):
            d_base_all[i, t] = float(np.linalg.norm(m_smooth_b[t] - x[t]))
        # baseline para t_selected
        for k, t in enumerate(t_selected):
            d_base_sel[i, k] = d_base_all[i, t]

        # -----------------------
        # Vista A: para t_selected y eps_grid
        # -----------------------
        for k, t in enumerate(t_selected):
            X_t, mu_t, Sigma_t = loo_values_nd(
                t=t, y=y, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                P0=P0, m0=m0,
            )
            y_t = y[t].copy()

            # d_adv(0): eps=0 => y[t] = mu_t
            y_adv0 = y.copy()
            y_adv0[t] = mu_t
            m_filt0, P_filt0, m_pred0, P_pred0 = kalman_filter_nd(
                y=y_adv0, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=m0, P0=P0
            )
            m_smooth0, P_smooth0 = rts_smoother_nd(
                m_filt=m_filt0, P_filt=P_filt0,
                m_pred=m_pred0, P_pred=P_pred0,
                A_t=mats["A_t"]
            )
            d_adv0_sel[i, k] = float(np.linalg.norm(m_smooth0[t] - x[t]))

            # barrido eps
            for j, eps in enumerate(eps_grid):
                y_star, _ = solve_kkt_max_quadratic_over_ellipsoid(
                    X=X_t, y_t=y_t, mu=mu_t, Sigma=Sigma_t, epsilon=float(eps)
                )
                y_adv = y.copy()
                y_adv[t] = y_star

                m_filtd, P_filtd, m_predd, P_predd = kalman_filter_nd(
                    y=y_adv, u=u,
                    A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                    Q_t=mats["Q_t"], R_t=mats["R_t"],
                    m0=m0, P0=P0
                )
                m_smoothd, P_smoothd = rts_smoother_nd(
                    m_filt=m_filtd, P_filt=P_filtd,
                    m_pred=m_predd, P_pred=P_predd,
                    A_t=mats["A_t"]
                )
                d_adv_sel[i, k, j] = float(np.linalg.norm(m_smoothd[t] - x[t]))

        # -----------------------
        # Vista B: para todos los t con eps_fixed
        # -----------------------
        for t in range(T + 1):
            X_t, mu_t, Sigma_t = loo_values_nd(
                t=t, y=y, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                P0=P0, m0=m0,
            )
            y_t = y[t].copy()

            # d_adv0(t): y[t]=mu_t
            y_adv0 = y.copy()
            y_adv0[t] = mu_t
            m_filt0, P_filt0, m_pred0, P_pred0 = kalman_filter_nd(
                y=y_adv0, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=m0, P0=P0
            )
            m_smooth0, P_smooth0 = rts_smoother_nd(
                m_filt=m_filt0, P_filt=P_filt0,
                m_pred=m_pred0, P_pred=P_pred0,
                A_t=mats["A_t"]
            )
            d_adv0_all[i, t] = float(np.linalg.norm(m_smooth0[t] - x[t]))

            # d_adv_fix(t): y[t]=y*(eps_fixed)
            y_star, _ = solve_kkt_max_quadratic_over_ellipsoid(
                X=X_t, y_t=y_t, mu=mu_t, Sigma=Sigma_t, epsilon=eps_fixed
            )
            y_adv = y.copy()
            y_adv[t] = y_star

            m_filtd, P_filtd, m_predd, P_predd = kalman_filter_nd(
                y=y_adv, u=u,
                A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
                Q_t=mats["Q_t"], R_t=mats["R_t"],
                m0=m0, P0=P0
            )
            m_smoothd, P_smoothd = rts_smoother_nd(
                m_filt=m_filtd, P_filt=P_filtd,
                m_pred=m_predd, P_pred=P_predd,
                A_t=mats["A_t"]
            )
            d_adv_fix[i, t] = float(np.linalg.norm(m_smoothd[t] - x[t]))

    # -----------------------
    # Ratios
    # -----------------------
    # Vista A: (S, n_sel, n_eps)
    denA = np.where(np.abs(d_base_sel[:, :, None]) < 1e-12, 1e-12, d_base_sel[:, :, None])
    ratio_sel = (d_adv_sel - d_adv0_sel[:, :, None]) / denA

    meanA, loA, hiA = bootstrap_ci_of_mean(ratio_sel, n_boot=n_boot, seed=seed_boot_left)

    # Vista B: (S, T+1)
    denB = np.where(np.abs(d_base_all) < 1e-12, 1e-12, d_base_all)
    ratio_t = (d_adv_fix - d_adv0_all) / denB

    meanB, loB, hiB = bootstrap_ci_of_mean(ratio_t, n_boot=n_boot, seed=seed_boot_right)

    return dict(
        T=T,
        eps_grid=eps_grid,
        eps_fixed=eps_fixed,
        t_selected=np.array(t_selected, dtype=int),
        meanA=meanA, loA=loA, hiA=hiA,        # (n_sel, n_eps)
        t_all=np.arange(T + 1, dtype=int),
        meanB=meanB, loB=loB, hiB=hiB,        # (T+1,)
    )


def plot_two_panel(
    res2: dict,
    *,
    use_logx_left: bool = True,
    eps_ref: float | None = 5.991,
    savepath: str | None = None,
    show: bool = True,
) -> str:
    if savepath is None:
        savepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/two_panel_ratio.png")
    """
    Figura con 2 paneles:
      Izq: ratio vs eps para t_selected
      Der: ratio vs t para eps_fixed
    """

    eps = np.asarray(res2["eps_grid"], dtype=float)
    t_sel = np.asarray(res2["t_selected"], dtype=int)
    meanA = np.asarray(res2["meanA"], dtype=float)
    loA   = np.asarray(res2["loA"], dtype=float)
    hiA   = np.asarray(res2["hiA"], dtype=float)

    t_all = np.asarray(res2["t_all"], dtype=int)
    meanB = np.asarray(res2["meanB"], dtype=float)
    loB   = np.asarray(res2["loB"], dtype=float)
    hiB   = np.asarray(res2["hiB"], dtype=float)

    # ordenar eps
    order = np.argsort(eps)
    eps = eps[order]
    meanA = meanA[:, order]
    loA   = loA[:, order]
    hiA   = hiA[:, order]

    if use_logx_left:
        mask = eps > 0
        eps = eps[mask]
        meanA = meanA[:, mask]
        loA   = loA[:, mask]
        hiA   = hiA[:, mask]

    plt.rcParams.update({
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.2, 6.2))

    # ---- Left: ratio vs eps for selected t
    for k, t in enumerate(t_sel):
        (line,) = axL.plot(eps, meanA[k], linewidth=2.0, label=fr"$t={t}$")
        axL.fill_between(eps, loA[k], hiA[k], alpha=0.18, color=line.get_color())

    axL.axhline(0.0, linestyle=":", linewidth=1.1, alpha=0.9)
    if eps_ref is not None:
        axL.axvline(float(eps_ref), linestyle="--", linewidth=1.0, alpha=0.35)

    if use_logx_left:
        axL.set_xscale("log")

    axL.set_xlabel("epsilon (ε)")
    axL.set_ylabel(r"$(d_{adv}(\epsilon)-d_{adv}(0))/d_{base}$")
    axL.set_title("Ratio vs ε (95% bootstrap CI)")
    axL.grid(True, which="major", alpha=0.28)
    axL.grid(True, which="minor", alpha=0.12)
    axL.legend(frameon=True, loc="best")

    # ---- Right: ratio vs time t for fixed eps
    (lineR,) = axR.plot(t_all, meanB, linewidth=2.2, marker="o", markersize=4.5,
                        label=fr"$\epsilon={res2['eps_fixed']}$")
    axR.fill_between(t_all, loB, hiB, alpha=0.18, color=lineR.get_color())

    axR.axhline(0.0, linestyle=":", linewidth=1.1, alpha=0.9)
    axR.set_xlabel("time index t")
    axR.set_ylabel(r"$(d_{adv}(\epsilon_{fix})-d_{adv}(0))/d_{base}$")
    axR.set_title("Ratio vs time (ε fixed, 95% bootstrap CI)")
    axR.grid(True, alpha=0.25)
    axR.legend(frameon=True, loc="best")

    fig.suptitle("Adversarial impact sensitivity: ε-sweep and time profile", y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return savepath


def main():
    T = 50
    t_selected = [1, int(T/2), T]
    eps_grid = np.r_[np.linspace(0.1, 2.0, 10, endpoint=False),
                    np.linspace(2.0, 12.0, 16)]
    eps_fixed = 5.991
    n_seeds = 70
    seed0 = 2026
    force_recompute = False

    savepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output/two_panel_ratio_eps_and_time.png",
    )
    data_path = data_path_for_plot(savepath)

    def compute_sensitivity_data() -> dict:
        return run_sensitivity_two_views(
            T=T,
            t_selected=t_selected,
            eps_grid=eps_grid,
            eps_fixed=eps_fixed,
            n_seeds=n_seeds,
            seed0=seed0,
        )

    res2 = cached_npz(data_path, compute_sensitivity_data, force=force_recompute)

    out = plot_two_panel(
        res2,
        use_logx_left=False,
        eps_ref=5.991,
        savepath=savepath,
        show=False,   # pon False si solo quieres guardar
    )
    print("Saved:", out)

if __name__ == "__main__":
    main()
