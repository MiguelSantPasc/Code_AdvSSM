#!/usr/bin/env python3
"""
attack_direction_density_lgssm_3d.py

Runs N independent simulations of a 3D LGSSM, attacks the last time step t=T
via a leave-one-out white-box point attack on E[g(x_T) | y_T', y_-T], and
builds a 3D density-colored scatter of the attack displacements in observation space.

Main output:
- A 3D density plot of attack displacements:
      delta_i = y_T'_i - y_T_i

Notes:
- We attack the last observation y_T, which changes the posterior on x_T.
- The feasible attack region is the ellipsoid:
      (y_T' - mu_T)^T Sigma_T^{-1} (y_T' - mu_T) <= epsilon
  where p(y_T | y_-T) = N(mu_T, Sigma_T).
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Global configuration
# ============================================================
N_RUNS = 500
T = 5
ATTACK_T = T

# 95% chi-square threshold for 3D is about 7.8147
EPSILON = 7.814727903251179

# Attack target for scalar g(x) in [0,1]
M_STAR = np.array([1.00], dtype=float)

# Optimization settings
ETA = 0.05
N_STEPS = 800
N_MC_OPT = 128
N_MC_EST = 2000

# Output
OUTPUT_DIRNAME = "output"


# ============================================================
# PSD / linear algebra helpers
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


def inv_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(np.asarray(M, dtype=float))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(1.0 / w) @ V.T


# ============================================================
# ND LGSSM simulator
# ============================================================
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
    """
    Model for k = 0,...,T:

      y_k     = H_k x_k + D_k u_k + v_k
      x_{k+1} = A_k x_k + B_k u_k + w_{k+1}      for k = 0,...,T-1

    with:
      v_k     ~ N(0, R_k)
      w_{k+1} ~ N(0, Q_k)

    Controls u_k exist for k=0,...,T, so u has shape (T+1, n_u).
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
            raise ValueError("x0 must have shape (n_x,)")

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

    for k in range(T + 1):
        A_t[k] = A0 + dA * k
        B_t[k] = B0 + dB * k
        H_t[k] = H0 + dH * k
        D_t[k] = D0 + dD * k
        Q_t[k] = project_to_psd(Q0 + dQ * k)
        R_t[k] = project_to_psd(R0 + dR * k)

    u = rng.uniform(u_low, u_high, size=(T + 1, n_u))

    x = np.zeros((T + 1, n_x), dtype=float)
    y = np.zeros((T + 1, n_y), dtype=float)
    x[0] = x0

    y[0] = H_t[0] @ x[0] + D_t[0] @ u[0] + rng.multivariate_normal(np.zeros(n_y), R_t[0])

    for k in range(T):
        w_next = rng.multivariate_normal(np.zeros(n_x), Q_t[k])
        x[k + 1] = A_t[k] @ x[k] + B_t[k] @ u[k] + w_next

        v_next = rng.multivariate_normal(np.zeros(n_y), R_t[k + 1])
        y[k + 1] = H_t[k + 1] @ x[k + 1] + D_t[k + 1] @ u[k + 1] + v_next

    mats = {
        "A_t": A_t,
        "B_t": B_t,
        "H_t": H_t,
        "D_t": D_t,
        "Q_t": Q_t,
        "R_t": R_t,
    }
    return x, y, u, mats


# ============================================================
# Kalman filter + RTS smoother with optional missing observations
# ============================================================
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = y.shape[0] - 1
    n_x = P0.shape[0]
    I = np.eye(n_x)

    if obs_mask is None:
        obs_mask = np.ones(T + 1, dtype=bool)
    else:
        obs_mask = np.asarray(obs_mask, dtype=bool)
        if obs_mask.shape != (T + 1,):
            raise ValueError("obs_mask must have shape (T+1,)")

    m_pred = np.zeros((T + 1, n_x), dtype=float)
    P_pred = np.zeros((T + 1, n_x, n_x), dtype=float)
    m_filt = np.zeros((T + 1, n_x), dtype=float)
    P_filt = np.zeros((T + 1, n_x, n_x), dtype=float)

    m_pred[0] = np.asarray(m0, dtype=float)
    P_pred[0] = project_to_psd(P0)

    for k in range(T + 1):
        Hk = H_t[k]
        Dk = D_t[k]
        Rk = project_to_psd(R_t[k])

        if obs_mask[k]:
            y_hat = Hk @ m_pred[k] + Dk @ u[k]
            S = Hk @ P_pred[k] @ Hk.T + Rk
            K = P_pred[k] @ Hk.T @ np.linalg.inv(S)

            innov = y[k] - y_hat
            m_filt[k] = m_pred[k] + K @ innov
            P_filt[k] = (I - K @ Hk) @ P_pred[k]
            P_filt[k] = project_to_psd(P_filt[k])
        else:
            m_filt[k] = m_pred[k]
            P_filt[k] = P_pred[k]

        if k < T:
            Ak = A_t[k]
            Bk = B_t[k]
            Qk = project_to_psd(Q_t[k])

            m_pred[k + 1] = Ak @ m_filt[k] + Bk @ u[k]
            P_pred[k + 1] = Ak @ P_filt[k] @ Ak.T + Qk
            P_pred[k + 1] = project_to_psd(P_pred[k + 1])

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
        P_smooth[k] = project_to_psd(P_smooth[k])

    return m_smooth, P_smooth


# ============================================================
# Leave-one-out attack stats
# ============================================================
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
) -> dict[str, np.ndarray]:
    """
    Remove y_t from the update:
      p(x_t | y_-t) = N(m_t_minus, P_t_minus)
      p(y_t | y_-t) = N(mu_t, Sigma_t)

    Then:
      m_post(y_t') = m_t_minus + K_t (y_t' - mu_t)
      P_post       = (I - K_t H_t) P_t_minus
    """
    T = y.shape[0] - 1
    if not (0 <= t <= T):
        raise ValueError("t out of range")

    obs_mask = np.ones(T + 1, dtype=bool)
    obs_mask[t] = False

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
    )
    m_smooth, P_smooth = rts_smoother_nd(
        m_filt=m_filt,
        P_filt=P_filt,
        m_pred=m_pred,
        P_pred=P_pred,
        A_t=A_t,
    )

    m_t_minus = m_smooth[t]
    P_t_minus = project_to_psd(P_smooth[t])

    mu_t = H_t[t] @ m_t_minus + D_t[t] @ u[t]
    Sigma_t = H_t[t] @ P_t_minus @ H_t[t].T + project_to_psd(R_t[t])
    Sigma_t = project_to_psd(Sigma_t)

    K_t = P_t_minus @ H_t[t].T @ np.linalg.inv(Sigma_t)
    P_post = (np.eye(P_t_minus.shape[0]) - K_t @ H_t[t]) @ P_t_minus
    P_post = project_to_psd(P_post)

    return {
        "m_t_minus": m_t_minus,
        "P_t_minus": P_t_minus,
        "mu_t": mu_t,
        "Sigma_t": Sigma_t,
        "K_t": K_t,
        "P_post": P_post,
        "m_smooth_minus_t": m_smooth,
        "P_smooth_minus_t": P_smooth,
    }


# ============================================================
# Finite-difference Jacobian helpers
# ============================================================
def _as_1d_output(v: np.ndarray | float) -> np.ndarray:
    return np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1)


def _as_2d_jacobian(J: np.ndarray, n_x: int) -> np.ndarray:
    J = np.asarray(J, dtype=float)
    if J.ndim == 1:
        if J.shape[0] != n_x:
            raise ValueError(f"1D gradient must have shape ({n_x},)")
        return J[None, :]
    if J.ndim == 2:
        if J.shape[1] != n_x:
            raise ValueError(f"Jacobian second dimension must be {n_x}")
        return J
    raise ValueError("Gradient/Jacobian must be 1D or 2D.")


def finite_diff_jacobian_g(
    g,
    x: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    fx = _as_1d_output(g(x))
    p = fx.size
    n = x.size

    J = np.zeros((p, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = h
        fp = _as_1d_output(g(x + e))
        fm = _as_1d_output(g(x - e))
        J[:, j] = (fp - fm) / (2.0 * h)
    return J


# ============================================================
# Projection onto ellipsoidal region
# ============================================================
def project_to_attack_region(
    y_candidate: np.ndarray,
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Exact Euclidean projection onto:
        {y : (y-mu)^T Sigma^{-1} (y-mu) <= epsilon}
    """
    y_candidate = np.asarray(y_candidate, dtype=float).reshape(-1)
    mu_t = np.asarray(mu_t, dtype=float).reshape(-1)
    Sigma_t = project_to_psd(Sigma_t)

    Sinv = inv_psd(Sigma_t)
    diff = y_candidate - mu_t
    maha = float(diff.T @ Sinv @ diff)

    if maha <= epsilon + tol:
        return y_candidate.copy()

    s, U = np.linalg.eigh(Sigma_t)
    s = np.maximum(s, 1e-12)
    r = U.T @ diff

    def f(lam: float) -> float:
        return float(np.sum((s * r**2) / (s + lam) ** 2) - epsilon)

    lam_low = 0.0
    lam_high = 1.0
    while f(lam_high) > 0:
        lam_high *= 2.0
        if lam_high > 1e14:
            raise RuntimeError("Could not bracket lambda in ellipsoid projection.")

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_low + lam_high)
        val = f(lam_mid)
        if abs(val) < tol:
            lam_low = lam_high = lam_mid
            break
        if val > 0:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    lam_star = 0.5 * (lam_low + lam_high)
    z = (s / (s + lam_star)) * r
    y_proj = mu_t + U @ z
    return y_proj


# ============================================================
# White-box attack
# ============================================================
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
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Solve:
        min_{y_t' in X_epsilon} || E[g(x_t) | y_t', y_-t] - M_star ||^2
    by projected gradient descent.
    """
    rng = np.random.default_rng(seed)

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
    )

    m_t_minus = stats["m_t_minus"]
    mu_t = stats["mu_t"]
    Sigma_t = stats["Sigma_t"]
    K_t = stats["K_t"]
    P_post = stats["P_post"]

    n_x = m_t_minus.size
    y_curr = project_to_attack_region(y[t].copy(), mu_t, Sigma_t, epsilon)

    M_star = _as_1d_output(M_star)
    L_post = sqrtm_psd(P_post)

    xi = rng.standard_normal(size=(n_mc, n_x))

    y_hist = []
    obj_hist = []
    mu_g_hist = []

    for _ in range(n_steps):
        m_post = m_t_minus + K_t @ (y_curr - mu_t)

        x_samples = m_post[None, :] + xi @ L_post.T

        g_vals = np.stack([_as_1d_output(g(xs)) for xs in x_samples], axis=0)
        mu_g = np.mean(g_vals, axis=0)

        if mu_g.shape != M_star.shape:
            raise ValueError(
                f"Shape mismatch: g returns shape {mu_g.shape}, "
                f"but M_star has shape {M_star.shape}"
            )

        if g_grad is not None:
            Jg_vals = np.stack(
                [_as_2d_jacobian(g_grad(xs), n_x=n_x) for xs in x_samples],
                axis=0,
            )
        else:
            Jg_vals = np.stack(
                [finite_diff_jacobian_g(g, xs) for xs in x_samples],
                axis=0,
            )

        Jg_mean = np.mean(Jg_vals, axis=0)
        dmu_dy = Jg_mean @ K_t

        diff = mu_g - M_star
        grad = 2.0 * (dmu_dy.T @ diff)

        y_curr = y_curr - eta * grad
        y_curr = project_to_attack_region(y_curr, mu_t, Sigma_t, epsilon)

        y_hist.append(y_curr.copy())
        obj_hist.append(float(np.dot(diff, diff)))
        mu_g_hist.append(mu_g.copy())

    history = {
        "y_hist": np.asarray(y_hist),
        "obj_hist": np.asarray(obj_hist),
        "mu_g_hist": np.asarray(mu_g_hist),
        "mu_t": mu_t,
        "Sigma_t": Sigma_t,
        "m_t_minus": m_t_minus,
        "K_t": K_t,
        "P_post": P_post,
    }
    return y_curr, history


# ============================================================
# Utility: estimate E[g(x)]
# ============================================================
def estimate_E_g(
    *,
    m: np.ndarray,
    P: np.ndarray,
    g,
    n_mc: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = np.asarray(m, dtype=float).reshape(-1)
    P = project_to_psd(P)

    n_x = m.size
    L = sqrtm_psd(P)

    xi = rng.standard_normal(size=(n_mc, n_x))
    x_samples = m[None, :] + xi @ L.T

    g_vals = np.stack(
        [np.atleast_1d(np.asarray(g(xs), dtype=float)).reshape(-1) for xs in x_samples],
        axis=0,
    )
    mu_g = np.mean(g_vals, axis=0)
    g_at_mean = np.atleast_1d(np.asarray(g(m), dtype=float)).reshape(-1)
    return mu_g, g_at_mean


# ============================================================
# Risk model g(x) in 3D
# ============================================================
def sigmoid(z: float | np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def g_scalar(x_vec: np.ndarray) -> float:
    """
    3D risk / alarm probability.

    Designed so that the (x2, x3) plane dominates clearly:
    - weak dependence on x1
    - strong quadratic and interaction terms on x2, x3
    """
    x_vec = np.asarray(x_vec, dtype=float)
    x1, x2, x3 = x_vec[0], x_vec[1], x_vec[2]

    beta0 = -3.2

    # x1 has small influence
    beta1 = 0.15

    # x2/x3 plane dominates
    beta1 = 0.05
    beta2 = 1.60
    beta3 = 1.60
    beta23 = 3.50
    beta22 = 3.00
    beta33 = 3.00

    # tiny couplings with x1
    beta12 = 0.10
    beta13 = 0.08

    z = (
        beta0
        + beta1 * x1
        + beta2 * x2
        + beta3 * x3
        + beta23 * x2 * x3
        + beta22 * x2**2
        + beta33 * x3**2
        + beta12 * x1 * x2
        + beta13 * x1 * x3
    )
    return float(sigmoid(z))


def g_scalar_grad(x_vec: np.ndarray) -> np.ndarray:
    """
    Gradient of scalar g wrt x = (x1, x2, x3).
    Returns shape (3,)
    """
    x_vec = np.asarray(x_vec, dtype=float)
    x1, x2, x3 = x_vec[0], x_vec[1], x_vec[2]

    beta0 = -3.2
    beta1 = 0.05
    beta2 = 1.60
    beta3 = 1.60
    beta23 = 3.50
    beta22 = 3.00
    beta33 = 3.00
    # tiny couplings with x1
    beta12 = 0.10
    beta13 = 0.08

    z = (
        beta0
        + beta1 * x1
        + beta2 * x2
        + beta3 * x3
        + beta23 * x2 * x3
        + beta22 * x2**2
        + beta33 * x3**2
        + beta12 * x1 * x2
        + beta13 * x1 * x3
    )
    s = float(sigmoid(z))
    common = s * (1.0 - s)

    dz_dx1 = beta1 + beta12 * x2 + beta13 * x3
    dz_dx2 = beta2 + beta23 * x3 + 2.0 * beta22 * x2 + beta12 * x1
    dz_dx3 = beta3 + beta23 * x2 + 2.0 * beta33 * x3 + beta13 * x1

    return np.array(
        [
            common * dz_dx1,
            common * dz_dx2,
            common * dz_dx3,
        ],
        dtype=float,
    )


# ============================================================
# Experiment setup
# ============================================================
def get_system_parameters() -> dict[str, np.ndarray]:
    A0 = np.array(
        [
            [0.92, 0.10, 0.03],
            [0.02, 0.90, 0.12],
            [0.01, 0.08, 0.88],
        ],
        dtype=float,
    )

    B0 = np.array(
        [
            [-0.22, 0.10, 0.06],
            [-0.10, 0.20, 0.12],
            [0.04, -0.08, 0.18],
        ],
        dtype=float,
    )

    H0 = np.array(
        [
            [1.10, 0.20, 0.10],
            [0.15, 1.05, 0.25],
            [0.08, 0.30, 0.95],
        ],
        dtype=float,
    )

    D0 = np.array(
        [
            [-0.04, 0.10, 0.05],
            [-0.08, 0.14, 0.10],
            [0.02, -0.03, 0.12],
        ],
        dtype=float,
    )

    Q0 = np.array(
        [
            [0.018, 0.004, 0.002],
            [0.004, 0.020, 0.006],
            [0.002, 0.006, 0.019],
        ],
        dtype=float,
    )

    R0 = np.array(
        [
            [0.040, 0.008, 0.004],
            [0.008, 0.042, 0.010],
            [0.004, 0.010, 0.038],
        ],
        dtype=float,
    )

    x0 = np.array([0.15, 0.35, 0.30], dtype=float)
    m0 = x0.copy()

    P0 = np.array(
        [
            [0.035, 0.006, 0.003],
            [0.006, 0.038, 0.008],
            [0.003, 0.008, 0.036],
        ],
        dtype=float,
    )

    return {
        "A0": A0,
        "B0": B0,
        "H0": H0,
        "D0": D0,
        "Q0": project_to_psd(Q0),
        "R0": project_to_psd(R0),
        "x0": x0,
        "m0": m0,
        "P0": project_to_psd(P0),
        "dA": np.zeros_like(A0),
        "dB": np.zeros_like(B0),
        "dH": np.zeros_like(H0),
        "dD": np.zeros_like(D0),
        "dQ": np.zeros_like(Q0),
        "dR": np.zeros_like(R0),
    }


# ============================================================
# Single run
# ============================================================
def run_one_experiment(run_id: int) -> dict[str, np.ndarray | float]:
    pars = get_system_parameters()

    sim_seed = 2025 + run_id
    attack_seed = 6000 + run_id

    x, y, u, mats = simulate_lgssm_nd(
        A0=pars["A0"],
        B0=pars["B0"],
        H0=pars["H0"],
        D0=pars["D0"],
        T=T,
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

    y_star, attack_hist = white_box_point_attack_nd(
        t=ATTACK_T,
        y=y,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        P0=pars["P0"],
        m0=pars["m0"],
        epsilon=EPSILON,
        M_star=M_STAR,
        g=g_scalar,
        g_grad=g_scalar_grad,
        eta=ETA,
        n_steps=N_STEPS,
        n_mc=N_MC_OPT,
        seed=attack_seed,
    )

    y_T = y[ATTACK_T].copy()
    delta_y = y_star - y_T
    delta_norm = float(np.linalg.norm(delta_y))

    if delta_norm > 1e-12:
        direction = delta_y / delta_norm
    else:
        direction = np.zeros_like(delta_y)

    m_filt_b, P_filt_b, m_pred_b, P_pred_b = kalman_filter_nd(
        y=y,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )
    m_smooth_b, P_smooth_b = rts_smoother_nd(
        m_filt=m_filt_b,
        P_filt=P_filt_b,
        m_pred=m_pred_b,
        P_pred=P_pred_b,
        A_t=mats["A_t"],
    )

    y_adv = y.copy()
    y_adv[ATTACK_T] = y_star

    m_filt_a, P_filt_a, m_pred_a, P_pred_a = kalman_filter_nd(
        y=y_adv,
        u=u,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )
    m_smooth_a, P_smooth_a = rts_smoother_nd(
        m_filt=m_filt_a,
        P_filt=P_filt_a,
        m_pred=m_pred_a,
        P_pred=P_pred_a,
        A_t=mats["A_t"],
    )

    mu_g_base, _ = estimate_E_g(
        m=m_smooth_b[ATTACK_T],
        P=P_smooth_b[ATTACK_T],
        g=g_scalar,
        n_mc=N_MC_EST,
        seed=77,
    )
    mu_g_adv, _ = estimate_E_g(
        m=m_smooth_a[ATTACK_T],
        P=P_smooth_a[ATTACK_T],
        g=g_scalar,
        n_mc=N_MC_EST,
        seed=77,
    )

    return {
        "run_id": run_id,
        "y_T": y_T,
        "y_star": y_star,
        "delta_y": delta_y,
        "delta_norm": delta_norm,
        "direction": direction,
        "final_obj": float(attack_hist["obj_hist"][-1]),
        "risk_base": float(mu_g_base[0]),
        "risk_adv": float(mu_g_adv[0]),
    }


def plot_delta_density_pairwise(
    deltas: np.ndarray,
    outpath: str,
    gridsize: int = 220,
    scatter_size: float = 18.0,
) -> None:
    """
    Pairwise 2D KDE density plots of 3D attack displacements:
        delta_y = y_T' - y_T

    Creates a 1x3 figure with panels:
        (Δy1, Δy2), (Δy1, Δy3), (Δy2, Δy3)

    deltas: shape (N, 3)
    """
    deltas = np.asarray(deltas, dtype=float)

    if deltas.ndim != 2 or deltas.shape[1] != 3:
        raise ValueError("deltas must have shape (N, 3)")

    pairs = [
        (0, 1, r"$\Delta y_1$", r"$\Delta y_2$"),
        (0, 2, r"$\Delta y_1$", r"$\Delta y_3$"),
        (1, 2, r"$\Delta y_2$", r"$\Delta y_3$"),
    ]

    # common symmetric axis limit across all coordinates
    max_abs = max(np.max(np.abs(deltas[:, 0])),
                  np.max(np.abs(deltas[:, 1])),
                  np.max(np.abs(deltas[:, 2])),
                  1e-3)
    lim = 1.10 * max_abs

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)

    last_im = None

    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        x = deltas[:, i]
        y = deltas[:, j]

        values = np.vstack([x, y])
        kde = gaussian_kde(values)

        xi = np.linspace(-lim, lim, gridsize)
        yi = np.linspace(-lim, lim, gridsize)
        XX, YY = np.meshgrid(xi, yi)

        grid_coords = np.vstack([XX.ravel(), YY.ravel()])
        Z = kde(grid_coords).reshape(XX.shape)

        im = ax.imshow(
            Z,
            origin="lower",
            extent=[-lim, lim, -lim, lim],
            cmap="viridis",
            aspect="equal",
        )
        last_im = im

        # optional point overlay
        ax.scatter(x, y, s=scatter_size, alpha=0.35, edgecolors="none")

        # mark origin
        ax.scatter(0.0, 0.0, marker="x", s=80, linewidths=2.0, color="white")
        ax.axhline(0.0, linewidth=0.8, alpha=0.35, color="white")
        ax.axvline(0.0, linewidth=0.8, alpha=0.35, color="white")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.12)

    axes[0].set_title(r"Density of $(\Delta y_1, \Delta y_2)$")
    axes[1].set_title(r"Density of $(\Delta y_1, \Delta y_3)$")
    axes[2].set_title(r"Density of $(\Delta y_2, \Delta y_3)$")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.90, pad=0.02)
    cbar.set_label("density")

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(
        out_dir,
        f"delta_y_points_3d_N{N_RUNS}_T{T}_t{ATTACK_T}.npz",
    )

    density_path = os.path.join(
        out_dir,
        f"delta_y_density_3d_N{N_RUNS}_T{T}_t{ATTACK_T}.png",
    )

    if os.path.exists(data_path):
        print("\nData already exists. Skipping simulation...")

        data = np.load(data_path)
        deltas = data["deltas"]

        plot_delta_density_pairwise(
            deltas=deltas,
            outpath=density_path,
        )

        print(f"3D density plot regenerated from existing data: {density_path}")
        return

    all_results: list[dict[str, np.ndarray | float]] = []

    print("\nRunning repeated attacks...")
    print(f"N_RUNS   = {N_RUNS}")
    print(f"T        = {T}")
    print(f"attack t = {ATTACK_T}")
    print(f"epsilon  = {EPSILON:.6f}")
    print(f"M_STAR   = {M_STAR}\n")

    for run_id in range(N_RUNS):
        res = run_one_experiment(run_id)
        all_results.append(res)

        if (run_id + 1) % 10 == 0 or run_id == 0:
            print(
                f"[{run_id + 1:3d}/{N_RUNS}] "
                f"delta_y = {res['delta_y']} | "
                f"||delta|| = {res['delta_norm']:.4f} | "
                f"risk: {res['risk_base']:.3f} -> {res['risk_adv']:.3f} | "
                f"final obj = {res['final_obj']:.4e}"
            )

    deltas = np.stack(
        [np.asarray(r["delta_y"], dtype=float) for r in all_results],
        axis=0,
    )

    plot_delta_density_pairwise(
        deltas=deltas,
        outpath=density_path,
    )

    np.savez(
        data_path,
        deltas=deltas,
    )

    print(f"\nSaved 3D density plot to: {density_path}")
    print(f"Saved raw delta points to: {data_path}")


if __name__ == "__main__":
    main()