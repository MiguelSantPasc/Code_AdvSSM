#!/usr/bin/env python3
"""
Gray-box ND attack demo:

- True system simulates (x,y,u) with known Q,R.
- White-box attacker: uses true A,B,H,D to compute (X_t, mu_t, Sigma_t), then KKT attack y*.
- Gray-box attacker: does NOT know A,B,H,D, but knows Q,R.
  Uses Gibbs with NUTS:
    1) sample x_{0:T} | params, y,u (NUTS)
    2) sample params | x_{0:T}, y,u (NUTS)
  Uses posterior mean params to compute (X_t, mu_t, Sigma_t), then KKT attack y*.

Plots: ONLY "first column" style plots (geometry + x1/x2 time series), DOUBLED:
  Left column  = white-box (true params)
  Right column = gray-box  (estimated params)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Linear algebra utilities
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

def inv_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = symmetrize(M)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return V @ np.diag(1.0 / w) @ V.T


# -----------------------------
# Simulator (constant params; you can add drift back if needed)
# -----------------------------
def simulate_lgssm_nd(
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
    rng = np.random.default_rng(seed)

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    H = np.asarray(H, float)
    D = np.asarray(D, float)
    Q = project_to_psd(np.asarray(Q, float))
    R = project_to_psd(np.asarray(R, float))

    n_x = A.shape[0]
    n_u = B.shape[1]
    n_y = H.shape[0]

    if x0 is None:
        x0 = np.zeros(n_x)
    else:
        x0 = np.asarray(x0, float)

    u = rng.uniform(u_low, u_high, size=(T, n_u))

    x = np.zeros((T + 1, n_x))
    y = np.zeros((T + 1, n_y))
    x[0] = x0

    # y0 uses u0 (if T>0) for consistency with your original
    if T > 0:
        y[0] = H @ x[0] + D @ u[0] + rng.multivariate_normal(np.zeros(n_y), R)
    else:
        y[0] = H @ x[0] + rng.multivariate_normal(np.zeros(n_y), R)

    for t in range(T):
        w = rng.multivariate_normal(np.zeros(n_x), Q)
        v = rng.multivariate_normal(np.zeros(n_y), R)
        x[t + 1] = A @ x[t] + B @ u[t] + w
        y[t + 1] = H @ x[t + 1] + D @ u[t] + v

    mats = {"A": A, "B": B, "H": H, "D": D, "Q": Q, "R": R}
    return x, y, u, mats


# -----------------------------
# KF + RTS (constant params)
# -----------------------------
def kalman_filter_nd(
    *,
    y: np.ndarray,      # (T+1, n_y)
    u: np.ndarray,      # (T, n_u)
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = y.shape[0] - 1
    n_x = P0.shape[0]
    I = np.eye(n_x)

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    H = np.asarray(H, float)
    D = np.asarray(D, float)
    Q = project_to_psd(np.asarray(Q, float))
    R = project_to_psd(np.asarray(R, float))

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

    m_pred = np.zeros((T + 1, n_x))
    P_pred = np.zeros((T + 1, n_x, n_x))
    m_filt = np.zeros((T + 1, n_x))
    P_filt = np.zeros((T + 1, n_x, n_x))

    m_pred[0] = m0
    P_pred[0] = P0

    for k in range(T + 1):
        uk = u_at(k)
        y_hat = H @ m_pred[k] + D @ uk
        S = H @ P_pred[k] @ H.T + R
        K = P_pred[k] @ H.T @ np.linalg.inv(S)

        innov = y[k] - y_hat
        m_filt[k] = m_pred[k] + K @ innov
        P_filt[k] = (I - K @ H) @ P_pred[k]

        if k < T:
            m_pred[k + 1] = A @ m_filt[k] + B @ u_at(k + 1)
            P_pred[k + 1] = A @ P_filt[k] @ A.T + Q

    return m_filt, P_filt, m_pred, P_pred


def rts_smoother_nd(
    *,
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = m_filt.shape[0] - 1
    m_smooth = np.zeros_like(m_filt)
    P_smooth = np.zeros_like(P_filt)

    m_smooth[T] = m_filt[T]
    P_smooth[T] = P_filt[T]

    for k in range(T - 1, -1, -1):
        Ck = P_filt[k] @ A.T @ np.linalg.inv(P_pred[k + 1])
        m_smooth[k] = m_filt[k] + Ck @ (m_smooth[k + 1] - m_pred[k + 1])
        P_smooth[k] = P_filt[k] + Ck @ (P_smooth[k + 1] - P_pred[k + 1]) @ Ck.T

    return m_smooth, P_smooth


# -----------------------------
# LOO (constant params version) returning X_t, mu_t, Sigma_t
# (kept close to your version; no drift arrays)
# -----------------------------
def loo_values_nd_const(
    *,
    t: int,
    y: np.ndarray,      # (T+1, n_y)
    u: np.ndarray,      # (T, n_u)
    A: np.ndarray,
    B: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P0: np.ndarray,
    m0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = int(y.shape[0] - 1)
    if not (0 <= t <= T):
        raise ValueError("t out of range")

    n_x = P0.shape[0]
    n_y = y.shape[1]
    I_x = np.eye(n_x)

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    H = np.asarray(H, float)
    D = np.asarray(D, float)
    Q = project_to_psd(np.asarray(Q, float))
    R = project_to_psd(np.asarray(R, float))

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

    # ---- KF covariances (for X)
    P_pred = [None] * (T + 1)
    P_filt = [None] * (T + 1)
    K_kf = [None] * (T + 1)

    P_pred[0] = P0.copy()
    for k in range(T + 1):
        S = H @ P_pred[k] @ H.T + R
        K = P_pred[k] @ H.T @ np.linalg.inv(S)
        P_filt[k] = (I_x - K @ H) @ P_pred[k]
        K_kf[k] = K
        if k < T:
            P_pred[k + 1] = A @ P_filt[k] @ A.T + Q

    # RTS gains
    J = [np.zeros((n_x, n_x)) for _ in range(T + 1)]
    for k in range(T):
        J[k] = P_filt[k] @ A.T @ np.linalg.inv(P_pred[k + 1])
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
            mid = np.eye(n_x) - J[t + i] @ A
        factors = [((np.eye(n_x) - K_kf[t + j] @ H) @ A) for j in range(i + 1)]
        prodKH = prod_left(factors)
        total = total + (prodJ @ mid @ prodKH)

    X_t_out = total @ K_kf[t]  # (n_x, n_y)

    # ---- Forward means with y_t excluded
    m_pred = [None] * (T + 1)
    m_filt = [None] * (T + 1)
    m_pred[0] = m0.copy()

    for k in range(T + 1):
        if k == t:
            m_filt[k] = m_pred[k]
        else:
            uk = u_at(k)
            y_hat = H @ m_pred[k] + D @ uk
            S = H @ P_pred[k] @ H.T + R
            K = P_pred[k] @ H.T @ np.linalg.inv(S)
            m_filt[k] = m_pred[k] + K @ (y[k] - y_hat)
        if k < T:
            m_pred[k + 1] = A @ m_filt[k] + B @ u_at(k + 1)

    # ---- Backward info messages (exclude measurement at time t)
    Lambda = [None] * (T + 1)
    eta = [None] * (T + 1)
    Lambda[T] = np.zeros((n_x, n_x))
    eta[T] = np.zeros((n_x,))

    for k in range(T - 1, -1, -1):
        kp1 = k + 1
        ukp1 = u_at(kp1)

        if kp1 == t:
            barLambda = Lambda[kp1]
            barEta = eta[kp1]
        else:
            tilde_y = y[kp1] - D @ ukp1
            Rinv = np.linalg.inv(R)
            barLambda = Lambda[kp1] + H.T @ Rinv @ H
            barEta = eta[kp1] + H.T @ Rinv @ tilde_y

        Qinv = np.linalg.inv(Q)
        S_back = Qinv + barLambda
        S_back_inv = np.linalg.inv(S_back)

        core = Qinv - Qinv @ S_back_inv @ Qinv
        Lambda[k] = A.T @ core @ A

        term1 = A.T @ Qinv @ S_back_inv @ barEta
        term2 = A.T @ Qinv @ S_back_inv @ barLambda @ (B @ ukp1)
        eta[k] = term1 - term2

    # ---- Combine at time t to get p(y_t | y_-t)
    P_t_minus = np.linalg.inv(np.linalg.inv(P_pred[t]) + Lambda[t])
    m_t_minus = P_t_minus @ (np.linalg.inv(P_pred[t]) @ m_pred[t] + eta[t])

    mu_y = H @ m_t_minus + D @ u_at(t)
    Sigma_y = H @ P_t_minus @ H.T + R

    return X_t_out, mu_y, project_to_psd(Sigma_y)


# -----------------------------
# KKT / trust-region solver (your version, unchanged)
# -----------------------------
def solve_kkt_max_quadratic_over_ellipsoid(
    *,
    X: np.ndarray,          # (n_x, n_y)
    y_t: np.ndarray,        # (n_y,)
    mu: np.ndarray,         # (n_y,)
    Sigma: np.ndarray,      # (n_y, n_y), PSD/PD
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
            denom = (a - lam)
            zi = -bp / denom
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
                raise RuntimeError("Failed to bracket lambda for KKT root finding.")

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
            z_star = z_star * (np.sqrt(epsilon) / nz)

    y_star = mu + S @ z_star
    obj_star = float(np.linalg.norm(X @ (y_star - y_t)) ** 2)
    return y_star, obj_star


# -----------------------------
# Geometry helpers (2D only for y-plot)
# -----------------------------
def _ellipse_points_from_quad(center: np.ndarray, shape_inv: np.ndarray, level: float, n: int = 320) -> np.ndarray:
    center = np.asarray(center, dtype=float).reshape(2,)
    M = project_to_psd(symmetrize(np.asarray(shape_inv, dtype=float).reshape(2, 2)))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-14)
    radii = np.sqrt(level / w)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    pts = (V @ (radii[:, None] * circle)).T + center[None, :]
    return pts


def graybox_gibbs_nuts(
    *,
    y: np.ndarray,
    u: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    num_gibbs: int = 8,
    num_warmup_x: int = 600,
    num_samples_x: int = 600,
    num_warmup_p: int = 700,
    num_samples_p: int = 700,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Gray-box attacker: unknown (A,B,H,D), known (Q,R).
    Gibbs with NUTS:
      1) sample x_{0:T} | params, y,u
      2) sample params  | x_{0:T}, y,u

    Returns posterior draws from the LAST Gibbs iteration:
      - x_samples: (Sx, T+1, n_x)
      - A_samples, B_samples, H_samples, D_samples: (Sp, ...)
      - A_mean, B_mean, H_mean, D_mean: posterior means (from last iter)
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.contrib.control_flow import scan

    # CPU (native Windows)
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(1)

    y_j = jnp.asarray(y)
    u_j = jnp.asarray(u)
    Q_j = jnp.asarray(Q)
    R_j = jnp.asarray(R)
    m0_j = jnp.asarray(m0)
    P0_j = jnp.asarray(P0)

    T = int(y.shape[0] - 1)
    n_y = int(y.shape[1])
    n_u = int(u.shape[1])
    n_x = int(m0.shape[0])

    def u_at(k: int):
        # keep your convention for k==T
        return u_j[k] if k < T else u_j[T - 1]

    # Soft bounding (demo-friendly)
    def bounded(mat_raw, scale=0.95):
        return scale * jnp.tanh(mat_raw)

    # -------------------------
    # x-step model: sample x given params
    # -------------------------
    def model_x(y_obs, u_in, A, B, H, D):
        # prior on x0
        x0 = numpyro.sample("x0", dist.MultivariateNormal(loc=m0_j, covariance_matrix=P0_j))

        # y0 likelihood
        yhat0 = H @ x0 + D @ u_at(0)
        numpyro.sample("y0", dist.MultivariateNormal(loc=yhat0, covariance_matrix=R_j), obs=y_obs[0])

        def transition(x_prev, t_idx):
            # state
            mean_xt1 = A @ x_prev + B @ u_in[t_idx]
            x_t1 = numpyro.sample(
                f"x_{t_idx+1}",
                dist.MultivariateNormal(loc=mean_xt1, covariance_matrix=Q_j),
            )

            # emission uses u[t_idx] to predict y[t_idx+1]
            yhat = H @ x_t1 + D @ u_in[t_idx]
            numpyro.sample(
                f"y_{t_idx+1}",
                dist.MultivariateNormal(loc=yhat, covariance_matrix=R_j),
                obs=y_obs[t_idx + 1],
            )

            return x_t1, x_t1

        # scan over t=0..T-1, returns x_1..x_T
        _, x_path = scan(transition, x0, jnp.arange(T))
        x_all = jnp.concatenate([x0[None, :], x_path], axis=0)  # (T+1, n_x)

        numpyro.deterministic("x", x_all)

    # -------------------------
    # param-step model: sample params given x
    # -------------------------
    def model_params(y_obs, u_in, x_in):
        # priors
        A_raw = numpyro.sample("A_raw", dist.Normal(0.0, 1.0).expand([n_x, n_x]))
        H_raw = numpyro.sample("H_raw", dist.Normal(0.0, 1.0).expand([n_y, n_x]))
        B = numpyro.sample("B", dist.Normal(0.0, 1.0).expand([n_x, n_u]))
        D = numpyro.sample("D", dist.Normal(0.0, 1.0).expand([n_y, n_u]))

        A = bounded(A_raw, scale=0.95)
        H = bounded(H_raw, scale=1.50)

        numpyro.deterministic("A", A)
        numpyro.deterministic("H", H)

        # y0 likelihood
        yhat0 = H @ x_in[0] + D @ u_at(0)
        numpyro.sample("y0", dist.MultivariateNormal(loc=yhat0, covariance_matrix=R_j), obs=y_obs[0])

        def obs_step(carry, t_idx):
            # transition likelihood for x_{t+1}
            mean_xt1 = A @ x_in[t_idx] + B @ u_in[t_idx]
            numpyro.sample(
                f"x_like_{t_idx+1}",
                dist.MultivariateNormal(loc=mean_xt1, covariance_matrix=Q_j),
                obs=x_in[t_idx + 1],
            )

            # emission likelihood for y_{t+1}
            yhat = H @ x_in[t_idx + 1] + D @ u_in[t_idx]
            numpyro.sample(
                f"y_{t_idx+1}",
                dist.MultivariateNormal(loc=yhat, covariance_matrix=R_j),
                obs=y_obs[t_idx + 1],
            )

            return carry, None

        scan(obs_step, None, jnp.arange(T))

    # -------------------------
    # Gibbs loop
    # -------------------------
    rng = jax.random.PRNGKey(seed)

    # init params
    A_cur = jnp.eye(n_x) * 0.7
    H_cur = jnp.eye(n_y, n_x)
    B_cur = jnp.zeros((n_x, n_u))
    D_cur = jnp.zeros((n_y, n_u))

    # init state path
    x_cur = jnp.tile(m0_j[None, :], (T + 1, 1))

    last: dict[str, np.ndarray] = {}

    for g in range(num_gibbs):
        # ---- x step (sample x | params)
        nuts_x = NUTS(lambda y_obs, u_in: model_x(y_obs, u_in, A_cur, B_cur, H_cur, D_cur))
        mcmc_x = MCMC(nuts_x, num_warmup=num_warmup_x, num_samples=num_samples_x, num_chains=1)

        rng, kx = jax.random.split(rng)
        mcmc_x.run(kx, y_obs=y_j, u_in=u_j)

        sx = mcmc_x.get_samples()
        x_samps = sx["x"]  # (Sx, T+1, n_x)
        x_cur = jnp.mean(x_samps, axis=0)

        # ---- param step (sample params | x)
        nuts_p = NUTS(lambda y_obs, u_in: model_params(y_obs, u_in, x_cur))
        mcmc_p = MCMC(nuts_p, num_warmup=num_warmup_p, num_samples=num_samples_p, num_chains=1)

        rng, kp = jax.random.split(rng)
        mcmc_p.run(kp, y_obs=y_j, u_in=u_j)

        sp = mcmc_p.get_samples()

        # update current params via posterior mean
        A_cur = jnp.mean(sp["A"], axis=0)
        H_cur = jnp.mean(sp["H"], axis=0)
        B_cur = jnp.mean(sp["B"], axis=0)
        D_cur = jnp.mean(sp["D"], axis=0)

        last = {
            "x_samples": np.array(x_samps),
            "A_samples": np.array(sp["A"]),
            "B_samples": np.array(sp["B"]),
            "H_samples": np.array(sp["H"]),
            "D_samples": np.array(sp["D"]),
            "A_mean": np.array(A_cur),
            "B_mean": np.array(B_cur),
            "H_mean": np.array(H_cur),
            "D_mean": np.array(D_cur),
        }

        print(f"[Gibbs {g+1}/{num_gibbs}] done.")

    return last


# -----------------------------
# Plot: doubled "first-column" panels (white-box vs gray-box)
# -----------------------------
def plot_attack_overlay_one_column(
    *,
    t: int,
    epsilon_w: float,
    epsilon_g: float,
    # geometry: white
    y_t: np.ndarray,
    mu_w: np.ndarray,
    Sigma_w: np.ndarray,
    X_w: np.ndarray,
    y_star_w: np.ndarray,
    obj_w: float,
    # geometry: gray
    mu_g: np.ndarray,
    Sigma_g: np.ndarray,
    X_g: np.ndarray,
    y_star_g: np.ndarray,
    obj_g: float,
    # time series: attacked smoothers (both evaluated under TRUE params)
    m_smooth_white: np.ndarray,
    P_smooth_white: np.ndarray,
    m_smooth_gray: np.ndarray,
    P_smooth_gray: np.ndarray,
    # truth
    x_true: np.ndarray,
    outpath: str | None = None,
) -> None:
    if y_t.shape != (2,) or mu_w.shape != (2,) or mu_g.shape != (2,) or y_star_w.shape != (2,) or y_star_g.shape != (2,):
        raise ValueError("This plot expects n_y=2 (y_t, mu_w, mu_g, y_star_w, y_star_g must be shape (2,)).")

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    T = x_true.shape[0] - 1
    tt = np.arange(T + 1)
    z = 1.96

    fig = plt.figure(figsize=(10.5, 10.8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.35, 1.0, 1.0], hspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])  # geometry
    ax2 = fig.add_subplot(gs[1, 0])  # x1
    ax3 = fig.add_subplot(gs[2, 0])  # x2

    # -----------------
    # Panel 1: Geometry overlay
    # -----------------
    Sigma_w = project_to_psd(Sigma_w)
    Sigma_g = project_to_psd(Sigma_g)

    # WHITE constraint ellipse uses epsilon_w
    pts_constraint_w = _ellipse_points_from_quad(mu_w, inv_psd(Sigma_w), epsilon_w)
    M_w = project_to_psd(symmetrize(X_w.T @ X_w))
    pts_obj_w = _ellipse_points_from_quad(y_t, M_w, obj_w)

    # GRAY constraint ellipse uses epsilon_g
    pts_constraint_g = _ellipse_points_from_quad(mu_g, inv_psd(Sigma_g), epsilon_g)
    M_g = project_to_psd(symmetrize(X_g.T @ X_g))
    pts_obj_g = _ellipse_points_from_quad(y_t, M_g, obj_g)

    # Draw WHITE
    ax1.fill(pts_constraint_w[:, 0], pts_constraint_w[:, 1], alpha=0.14, label=f"WHITE constraint (ε={epsilon_w:.3f})")
    ax1.plot(pts_constraint_w[:, 0], pts_constraint_w[:, 1], linewidth=1.8, alpha=0.9)

    ax1.fill(pts_obj_w[:, 0], pts_obj_w[:, 1], alpha=0.10, label="WHITE objective level-set")
    ax1.plot(pts_obj_w[:, 0], pts_obj_w[:, 1], linewidth=1.8, linestyle="--", alpha=0.9)

    # Draw GRAY
    ax1.fill(pts_constraint_g[:, 0], pts_constraint_g[:, 1], alpha=0.14, label=f"GRAY constraint (ε={epsilon_g:.3f})")
    ax1.plot(pts_constraint_g[:, 0], pts_constraint_g[:, 1], linewidth=1.8, alpha=0.9)

    ax1.fill(pts_obj_g[:, 0], pts_obj_g[:, 1], alpha=0.10, label="GRAY objective level-set")
    ax1.plot(pts_obj_g[:, 0], pts_obj_g[:, 1], linewidth=1.8, linestyle="--", alpha=0.9)

    # Points
    ax1.scatter([y_t[0]], [y_t[1]], s=65, marker="x", label=r"$y_t$", zorder=10)
    ax1.scatter([mu_w[0]], [mu_w[1]], s=55, marker="o", label=r"$\mu_t$ (WHITE)", zorder=10)
    ax1.scatter([y_star_w[0]], [y_star_w[1]], s=95, marker="*", label=r"$y^\star$ (WHITE)", zorder=11)

    ax1.scatter([mu_g[0]], [mu_g[1]], s=55, marker="o", label=r"$\mu_t$ (GRAY)", zorder=10)
    ax1.scatter([y_star_g[0]], [y_star_g[1]], s=95, marker="*", label=r"$y^\star$ (GRAY)", zorder=11)

    ax1.set_title(f"Attack geometry overlay at t={t} (ε_W={epsilon_w:.3f}, ε_G={epsilon_g:.3f})")
    ax1.set_xlabel("y1")
    ax1.set_ylabel("y2")
    ax1.grid(True, alpha=0.20)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(loc="best", frameon=True)

    # -----------------
    # Panels 2 & 3: time series overlay (WHITE vs GRAY)
    # -----------------
    def plot_state_overlay(ax, idx: int, title: str) -> None:
        ax.plot(tt, x_true[:, idx], marker="o", markersize=2.6, linewidth=1.05, label=f"True x[{idx}]")

        mW = m_smooth_white[:, idx]
        sdW = np.sqrt(np.maximum(P_smooth_white[:, idx, idx], 0.0))
        ax.fill_between(tt, mW - z * sdW, mW + z * sdW, alpha=0.16, label="WHITE attacked 95% CI")
        ax.plot(tt, mW, linewidth=1.4, label="WHITE attacked mean")

        mG = m_smooth_gray[:, idx]
        sdG = np.sqrt(np.maximum(P_smooth_gray[:, idx, idx], 0.0))
        ax.fill_between(tt, mG - z * sdG, mG + z * sdG, alpha=0.12, label="GRAY attacked 95% CI")
        ax.plot(tt, mG, linewidth=1.4, linestyle="--", label="GRAY attacked mean")

        ax.axvline(t, linewidth=1.0, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("time t")
        ax.set_ylabel(f"x[{idx}]")
        ax.grid(True, alpha=0.20)
        ax.legend(loc="best", frameon=True)

    plot_state_overlay(ax2, 0, "State impact overlay on x1 (component 0)")
    plot_state_overlay(ax3, 1, "State impact overlay on x2 (component 1)")

    if outpath is not None:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # dims
    n_x = n_y = n_u = 2
    T = 20
    seed = 20241

    # True parameters
    A_true = np.array([[0.65, 0.40],
                       [-0.15, 0.70]], dtype=float)

    B_true = np.array([[1.65, 0.40],
                       [-0.15, 0.70]], dtype=float)

    H_true = np.eye(n_y, n_x)
    D_true = np.zeros((n_y, n_u), dtype=float)

    Q_true = project_to_psd(0.03 * np.array([[1.6, -0.40],
                                            [0.15, 0.70]], dtype=float))
    R_true = project_to_psd(0.02 * np.array([[0.65, 1.80],
                                            [-0.15, 1.70]], dtype=float))

    x0 = np.array([0.5, 0.5], dtype=float)

    # prior for filtering/smoothing (also used by attacker)
    m0 = x0.copy()
    P0 = 0.05 * np.eye(n_x)

    # simulate
    x_true, y, u, mats = simulate_lgssm_nd(
        A=A_true, B=B_true, H=H_true, D=D_true,
        Q=Q_true, R=R_true,
        T=T, seed=seed, x0=x0,
        u_low=-0.5, u_high=0.5
    )

    # attack time
    t = 10
    y_t = y[t].copy()

    # attack constraint
    epsilon_w = 5.991
    epsilon_g = 3.991  

    # =========================================================
    # WHITE-BOX attacker (knows true params)
    # =========================================================
    X_w, mu_w, Sig_w = loo_values_nd_const(
        t=t, y=y, u=u,
        A=A_true, B=B_true, H=H_true, D=D_true,
        Q=Q_true, R=R_true,
        P0=P0, m0=m0
    )
    y_star_w, obj_w = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_w, y_t=y_t, mu=mu_w, Sigma=Sig_w, epsilon=epsilon_w
    )

    # baseline RTS on original y (true params)
    m_f_b, P_f_b, m_p_b, P_p_b = kalman_filter_nd(
        y=y, u=u,
        A=A_true, B=B_true, H=H_true, D=D_true,
        Q=Q_true, R=R_true,
        m0=m0, P0=P0
    )
    m_s_b, P_s_b = rts_smoother_nd(
        m_filt=m_f_b, P_filt=P_f_b,
        m_pred=m_p_b, P_pred=P_p_b,
        A=A_true
    )

    # white-box attacked series
    y_adv_w = y.copy()
    y_adv_w[t] = y_star_w

    # attacked RTS (true params)
    m_f_aw, P_f_aw, m_p_aw, P_p_aw = kalman_filter_nd(
        y=y_adv_w, u=u,
        A=A_true, B=B_true, H=H_true, D=D_true,
        Q=Q_true, R=R_true,
        m0=m0, P0=P0
    )
    m_s_aw, P_s_aw = rts_smoother_nd(
        m_filt=m_f_aw, P_filt=P_f_aw,
        m_pred=m_p_aw, P_pred=P_p_aw,
        A=A_true
    )

    # =========================================================
    # GRAY-BOX attacker (estimates params via Gibbs+NUTS ONCE)
    # =========================================================
    print("\nRunning gray-box Gibbs with NUTS (CPU on native Windows)...\n")
    post = graybox_gibbs_nuts(
        y=y, u=u, Q=Q_true, R=R_true,
        m0=m0, P0=P0,
        num_gibbs=20,                 # many passes (increase if you want)
        num_warmup_x=500, num_samples_x=500,
        num_warmup_p=600, num_samples_p=600,
        seed=0,
    )

    A_hat = post["A_mean"]
    B_hat = post["B_mean"]
    H_hat = post["H_mean"]
    D_hat = post["D_mean"]

    # ---- PRINT true vs estimated matrices (attacker)
    np.set_printoptions(precision=4, suppress=True)

    print("\n==================== TRUE PARAMETERS ====================")
    print("A_true =\n", A_true)
    print("B_true =\n", B_true)
    print("H_true =\n", H_true)
    print("D_true =\n", D_true)

    print("\n==================== ATTACKER ESTIMATES ==================")
    print("A_hat  =\n", A_hat)
    print("B_hat  =\n", B_hat)
    print("H_hat  =\n", H_hat)
    print("D_hat  =\n", D_hat)

    print("\n==================== ESTIMATION ERRORS ===================")
    print("A_hat - A_true =\n", A_hat - A_true)
    print("B_hat - B_true =\n", B_hat - B_true)
    print("H_hat - H_true =\n", H_hat - H_true)
    print("D_hat - D_true =\n", D_hat - D_true)
    print("=========================================================\n")

    # --- attacker uses estimated params to compute attack objects
    X_g, mu_g, Sig_g = loo_values_nd_const(
        t=t, y=y, u=u,
        A=A_hat, B=B_hat, H=H_hat, D=D_hat,
        Q=Q_true, R=R_true,
        P0=P0, m0=m0
    )
    y_star_g, obj_g = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_g, y_t=y_t, mu=mu_g, Sigma=Sig_g, epsilon=epsilon_g
    )

    # gray-box attacked series (only change y[t])
    y_adv_g = y.copy()
    y_adv_g[t] = y_star_g

    # =========================================================
    # IMPORTANT: Evaluate gray-box attack impact under TRUE params
    # (this is what you asked for)
    # =========================================================
    m_f_ag, P_f_ag, m_p_ag, P_p_ag = kalman_filter_nd(
        y=y_adv_g, u=u,
        A=A_true, B=B_true, H=H_true, D=D_true,   # <-- TRUE params here
        Q=Q_true, R=R_true,
        m0=m0, P0=P0
    )
    m_s_ag, P_s_ag = rts_smoother_nd(
        m_filt=m_f_ag, P_filt=P_f_ag,
        m_pred=m_p_ag, P_pred=P_p_ag,
        A=A_true
    )

    # =========================================================
    # Plot doubled "first-column" panels only
    # =========================================================
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"attack_overlay_onecol_t{t}_T{T}_seed{seed}.png")

    plot_attack_overlay_one_column(
    t=t,
    epsilon_w=epsilon_w,
    epsilon_g=epsilon_g,
    # geometry white
    y_t=y_t,
    mu_w=mu_w, Sigma_w=Sig_w, X_w=X_w,
    y_star_w=y_star_w, obj_w=obj_w,
    # geometry gray
    mu_g=mu_g, Sigma_g=Sig_g, X_g=X_g,
    y_star_g=y_star_g, obj_g=obj_g,
    # time series
    m_smooth_white=m_s_aw, P_smooth_white=P_s_aw,
    m_smooth_gray=m_s_ag,  P_smooth_gray=P_s_ag,
    x_true=x_true,
    outpath=outpath,
    )


    print(f"Saved figure: {outpath}")

if __name__ == "__main__":
    main()
