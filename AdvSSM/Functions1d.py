"""
One-dimensional leave-one-out helper for linear Gaussian SSM experiments.

The scalar model is:

    x_{k+1} = A_k x_k + B_k u_k + w_{k+1},    w_{k+1} ~ N(0, Q_k)
    y_k     = H_k x_k + D_k u_k + v_k,        v_k     ~ N(0, R_k)

For a requested index t, this module computes:

    X_t          : sensitivity of the RTS-smoothed state to perturbing y_t.
    mu_{t|-t}    : E[y_t | y_{-t}], the leave-one-out predictive mean.
    Sigma_{t|-t}: Var[y_t | y_{-t}], the leave-one-out predictive variance.

The calculation uses standard Kalman covariance recursions, RTS gains J_k, and
a backward information message beta_k(x_k) that skips the likelihood term at
the removed observation y_t.
"""

from __future__ import annotations

import numpy as np


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
    """
    Importable helper: returns a single formatted line

        t=.. [X=..., mu_y_t_given_minus_t=..., Sigma_y_t_given_minus_t=...]

    for the requested time index t.

    Assumptions (1D, potentially time-varying):
      x_{k+1} = A_{k+1} x_k + B_{k+1} u_{k+1} + w_{k+1},  w_{k+1} ~ N(0, Q_{k+1})
      y_k     = H_k x_k + D_k u_k + v_k,                 v_k ~ N(0, R_k)

    Inputs:
      - y: length T+1
      - u: length T
      - A_t,B_t,H_t,D_t,Q_t,R_t: length T+1 lists/arrays
      - P0,m0: prior for x_0

    What it computes:
      - X(t): your “image formula” value computed from KF covariances + RTS gains (time-varying generalization)
      - mu_y_t_given_minus_t and Sigma_y_t_given_minus_t via LOO info-form messages (exclude y_t only)
    """
    
    T = int(len(y) - 1)
    if T < 0:
        raise ValueError("y must have length >= 1")
    if len(u) != T:
        raise ValueError(f"u must have length T={T}")
    for name, arr in [("A_t", A_t), ("B_t", B_t), ("H_t", H_t), ("D_t", D_t), ("Q_t", Q_t), ("R_t", R_t)]:
        if len(arr) != T + 1:
            raise ValueError(f"{name} must have length T+1={T+1}")
    if not (0 <= t <= T):
        raise ValueError(f"t must be between 0 and T={T}")

    # Convert to numpy for easy indexing
    A_t = np.asarray(A_t, dtype=float)
    B_t = np.asarray(B_t, dtype=float)
    H_t = np.asarray(H_t, dtype=float)
    D_t = np.asarray(D_t, dtype=float)
    Q_t = np.asarray(Q_t, dtype=float)
    R_t = np.asarray(R_t, dtype=float)

    def u_at(k: int) -> float:
        return float(u[k]) if k < T else float(u[T - 1])

    # ----------------------------
    # 1) Compute X(t) (your image formula), generalized to time-varying scalars
    # ----------------------------
    # Build KF covariance sequences needed: P_{k|k-1}, P_{k|k}, K_k, and then RTS gains J_k.

    P_pred = np.zeros(T + 1)  # P_{k|k-1}
    P_filt = np.zeros(T + 1)  # P_{k|k}
    K_kf = np.zeros(T + 1)

    P_pred[0] = float(P0)

    for k in range(T + 1):
        S = (H_t[k] ** 2) * P_pred[k] + R_t[k]
        K_kf[k] = (P_pred[k] * H_t[k]) / S
        P_filt[k] = (1.0 - K_kf[k] * H_t[k]) * P_pred[k]

        if k < T:
            P_pred[k + 1] = (A_t[k] ** 2) * P_filt[k] + Q_t[k]

    # RTS gains J_k (define J_T = 0)
    J = np.zeros(T + 1)
    for k in range(T):
        if P_pred[k + 1] <= 0:
            raise ValueError(f"P_pred[{k+1}] must be > 0 to compute J.")
        J[k] = (P_filt[k] * A_t[k]) / P_pred[k + 1]
    J[T] = 0.0

    def prod_right(vals: list[float]) -> float:
        out = 1.0
        for v in vals:
            out *= v
        return out

    def prod_left(vals: list[float]) -> float:
        out = 1.0
        for v in reversed(vals):
            out *= v
        return out

    l = T - t
    total = 0.0
    for i in range(l + 1):
        # Π_{j=0}^{i-1}→ J_{t+j}
        prodJ = 1.0 if i == 0 else prod_right([float(J[t + j]) for j in range(i)])

        # (1 - J_{t+i} A_{t+i+1})
        if t + i >= T:
            mid = 1.0  # because J_T = 0
        else:
            mid = 1.0 - float(J[t + i]) * float(A_t[t + i + 1])

        # Π_{j=0}^{i}← (1 - K_{t+j} H_{t+j}) A_{t+j}
        factors = [float((1.0 - K_kf[t + j] * H_t[t + j]) * A_t[t + j]) for j in range(i + 1)]
        prodKH = prod_left(factors)

        total += prodJ * mid * prodKH

    X_val = float(total * K_kf[t])

    # ----------------------------
    # 2) Compute mu_y_t| -t and Sigma_y_t| -t using your info-form backward message
    #    (exclude y_t only) + forward prediction p(x_t|y_{1:t-1})
    # ----------------------------
    # Forward prediction mean/cov excluding y_t in the update only affects m_{k|k}, not P_pred.
    # Here we only need m_pred[t] and P_pred[t] and then beta message at time t.

    m_pred = np.zeros(T + 1)
    m_filt = np.zeros(T + 1)

    m_pred[0] = float(m0)

    for k in range(T + 1):
        if k == t:
            # skip update
            m_filt[k] = m_pred[k]
        else:
            y_hat = H_t[k] * m_pred[k] + D_t[k] * u_at(k)
            S = (H_t[k] ** 2) * P_pred[k] + R_t[k]
            K = (P_pred[k] * H_t[k]) / S
            m_filt[k] = m_pred[k] + K * (y[k] - y_hat)

        if k < T:
            m_pred[k + 1] = A_t[k] * m_filt[k] + B_t[k] * u_at(k + 1)

    # Backward info message beta_k(x_k) = exp(-1/2 x^T Lambda_k x + eta_k^T x)
    Lambda = np.zeros(T + 1)
    eta = np.zeros(T + 1)
    Lambda[T] = 0.0
    eta[T] = 0.0

    for k in range(T - 1, -1, -1):
        kp1 = k + 1

        # bar terms at time kp1
        if kp1 == t:
            barLambda = Lambda[kp1]
            barEta = eta[kp1]
        else:
            tilde_y = y[kp1] - D_t[kp1] * u_at(kp1)
            barLambda = Lambda[kp1] + (H_t[kp1] ** 2) / R_t[kp1]
            barEta = eta[kp1] + (H_t[kp1] / R_t[kp1]) * tilde_y

        S_back = (1.0 / Q_t[kp1]) + barLambda

        Lambda[k] = (A_t[kp1] ** 2) * (
            (1.0 / Q_t[kp1]) - (1.0 / (Q_t[kp1] ** 2)) * (1.0 / S_back)
        )

        eta[k] = (A_t[kp1] * (1.0 / Q_t[kp1]) * (1.0 / S_back) * barEta) - (
            A_t[kp1]
            * (1.0 / Q_t[kp1])
            * (1.0 / S_back)
            * barLambda
            * B_t[kp1]
            * u_at(kp1)
        )

    # Combine at time t:
    P_t_minus = 1.0 / ((1.0 / P_pred[t]) + Lambda[t])
    m_t_minus = P_t_minus * ((m_pred[t] / P_pred[t]) + eta[t])

    mu_val = float(H_t[t] * m_t_minus + D_t[t] * u_at(t))
    Sig_val = float((H_t[t] ** 2) * P_t_minus + R_t[t])

    return [X_val, mu_val, Sig_val]

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
    rng = np.random.default_rng(seed)

    u = rng.uniform(-0.5, 0.5, size=T)
    w = rng.normal(loc=0.0, scale=np.sqrt(Q), size=T + 1)
    v = rng.normal(loc=0.0, scale=np.sqrt(R), size=T + 1)

    x = np.zeros(T + 1)
    y = np.zeros(T + 1)

    x[0] = x0
    y[0] = H * x[0] + (D * u[0] if T > 0 else 0.0) + v[0]

    for t in range(T):
        x[t + 1] = A * x[t] + B * u[t] + w[t + 1]
        y[t + 1] = H * x[t + 1] + D * u[t] + v[t + 1]

    return x, y, u


def main() -> None:
    # --- Params
    A = 0.35
    B = 0.85
    H = 2.5
    D = 0.0

    T = 15
    seed = 2026
    x0 = 0.5

    Q = 0.8
    R = 0.5

    # --- Simulate
    x, y, u = simulate_lgssm_1d(A=A, B=B, H=H, D=D, T=T, seed=seed, x0=x0, Q=Q, R=R)

    # --- Build time-varying lists (here constant over time)
    A_t = [A] * (T + 1)
    B_t = [B] * (T + 1)
    H_t = [H] * (T + 1)
    D_t = [D] * (T + 1)
    Q_t = [Q] * (T + 1)
    R_t = [R] * (T + 1)

    # --- Prior
    m0 = x0
    P0 = 0.05

    # --- Print 5 random t's using format_loo_line_1d
    rng = np.random.default_rng(seed + 12345)
    t_list = np.sort(rng.choice(np.arange(T + 1), size=min(5, T + 1), replace=False))

    for t in t_list:
        line = format_loo_line_1d(
            t=int(t),
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
        print(line)


if __name__ == "__main__":
    main()


