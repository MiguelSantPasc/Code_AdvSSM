"""
Leave-one-out quantities for a multidimensional linear Gaussian SSM.

The model is indexed on k = 0,...,T with controls u_k for k < T:

    x_{k+1} = A_k x_k + B_k u_k + w_{k+1},    w_{k+1} ~ N(0, Q_k)
    y_k     = H_k x_k + D_k u_k + v_k,        v_k     ~ N(0, R_k)

For a selected observation time t, the main helper computes:

    X_t          : linear map from an observation perturbation at time t to the
                   induced RTS-smoothed state perturbation.
    mu_{t|-t}    : E[y_t | y_{-t}], the leave-one-out predictive mean.
    Sigma_{t|-t}: Cov[y_t | y_{-t}], the leave-one-out predictive covariance.

The implementation combines Kalman filter covariance recursions, RTS smoother
gains J_k, and a backward information-form message that excludes y_t only.
"""

from __future__ import annotations

import numpy as np

def loo_values_nd(
    *,
    t: int,
    y: np.ndarray,          # (T+1, n_y)
    u: np.ndarray,          # (T,   n_u)
    A_t: list[np.ndarray] | np.ndarray,  # (T+1, n_x, n_x)  (we use A_t[k] for k=0..T-1)
    B_t: list[np.ndarray] | np.ndarray,  # (T+1, n_x, n_u)
    H_t: list[np.ndarray] | np.ndarray,  # (T+1, n_y, n_x)
    D_t: list[np.ndarray] | np.ndarray,  # (T+1, n_y, n_u)
    Q_t: list[np.ndarray] | np.ndarray,  # (T+1, n_x, n_x)
    R_t: list[np.ndarray] | np.ndarray,  # (T+1, n_y, n_y)
    P0: np.ndarray,         # (n_x, n_x)
    m0: np.ndarray,         # (n_x,)
) -> list[np.ndarray]:
    """
    Multidimensional version.

    Returns:
        [X_t, mu_y_t_given_minus_t, Sigma_y_t_given_minus_t]

    where:
        X_t:     (n_x, n_y)   your "image formula" coefficient (matrix)
        mu_y:    (n_y,)
        Sigma_y: (n_y, n_y)

    Conventions (time indices 0..T):
      - y has length T+1
      - u has length T (controls for 0..T-1); we use u_at(T)=u[T-1]
      - Prediction covariance recursion uses:
            P_pred[0] = P0
            Update at k uses (H_k, R_k)
            Predict next uses (A_k, Q_k): P_pred[k+1] = A_k P_filt[k] A_k^T + Q_k
      - RTS gain:
            J_k = P_filt[k] A_k^T (P_pred[k+1])^{-1}  for k=0..T-1 ; J_T = 0
      - Leave-one-out: exclude only y_t (skip update at time t and skip the measurement term in backward message at time t).
    """
    # ---- checks & shapes
    T = int(y.shape[0] - 1)
    if T < 0:
        raise ValueError("y must have shape (T+1, n_y) with T>=0")
    if u.shape[0] != T:
        raise ValueError(f"u must have shape (T, n_u) with T={T}")
    if not (0 <= t <= T):
        raise ValueError(f"t must be between 0 and T={T}")

    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    P0 = np.asarray(P0, dtype=float)
    m0 = np.asarray(m0, dtype=float)

    # allow list-of-matrices or stacked arrays
    A_t = np.asarray(A_t, dtype=float)
    B_t = np.asarray(B_t, dtype=float)
    H_t = np.asarray(H_t, dtype=float)
    D_t = np.asarray(D_t, dtype=float)
    Q_t = np.asarray(Q_t, dtype=float)
    R_t = np.asarray(R_t, dtype=float)

    n_x = P0.shape[0]
    n_y = y.shape[1]
    I_x = np.eye(n_x)

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

    # ============================================================
    # 1) Build KF covariance sequences (no missing needed for X_t)
    # ============================================================
    P_pred = [None] * (T + 1)
    P_filt = [None] * (T + 1)
    K_kf = [None] * (T + 1)

    P_pred[0] = P0.copy()

    for k in range(T + 1):
        Hk = H_t[k]
        Rk = R_t[k]
        # S = H P H^T + R
        S = Hk @ P_pred[k] @ Hk.T + Rk
        # K = P H^T S^{-1}
        K = P_pred[k] @ Hk.T @ np.linalg.inv(S)
        # P_filt = (I - K H) P_pred
        P_filt[k] = (I_x - K @ Hk) @ P_pred[k]
        K_kf[k] = K

        if k < T:
            Ak = A_t[k]
            Qk = Q_t[k]
            P_pred[k + 1] = Ak @ P_filt[k] @ Ak.T + Qk

    # RTS gains J_k
    J = [np.zeros((n_x, n_x)) for _ in range(T + 1)]
    for k in range(T):
        Ak = A_t[k]
        # J_k = P_filt[k] A_k^T (P_pred[k+1])^{-1}
        J[k] = P_filt[k] @ Ak.T @ np.linalg.inv(P_pred[k + 1])
    J[T] = np.zeros((n_x, n_x))  # convention

    # ============================================================
    # 1b) Compute X_t via your "image formula" (matrix products)
    # ============================================================
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
        # Π_{j=0}^{i-1}→ J_{t+j}
        prodJ = np.eye(n_x) if i == 0 else prod_right([J[t + j] for j in range(i)])

        # (I - J_{t+i} A_{t+i+1})
        # boundary: if t+i==T => use I (since J_T=0)
        if t + i >= T:
            mid = np.eye(n_x)
        else:
            mid = np.eye(n_x) - J[t + i] @ A_t[t + i + 1]

        # Π_{j=0}^{i}← (I - K_{t+j} H_{t+j}) A_{t+j}
        factors = []
        for j in range(i + 1):
            tj = t + j
            factors.append((np.eye(n_x) - K_kf[tj] @ H_t[tj]) @ A_t[tj])
        prodKH = prod_left(factors)

        total = total + (prodJ @ mid @ prodKH)

    # X_t = (sum ...) K_t
    X_t = total @ K_kf[t]   # (n_x,n_x)@(n_x,n_y) => (n_x,n_y)

    # ============================================================
    # 2) Leave-one-out p(y_t | y_-t): forward (skip update at t) + backward info-form
    # ============================================================
    m_pred = [None] * (T + 1)
    m_filt = [None] * (T + 1)

    m_pred[0] = m0.copy()

    for k in range(T + 1):
        if k == t:
            # skip update
            m_filt[k] = m_pred[k]
        else:
            Hk = H_t[k]
            Dk = D_t[k]
            Rk = R_t[k]
            uk = u_at(k)
            y_hat = Hk @ m_pred[k] + Dk @ uk
            S = Hk @ P_pred[k] @ Hk.T + Rk
            K = P_pred[k] @ Hk.T @ np.linalg.inv(S)
            m_filt[k] = m_pred[k] + K @ (y[k] - y_hat)

        if k < T:
            Ak = A_t[k]
            Bk = B_t[k]
            ukp1 = u_at(k + 1)
            m_pred[k + 1] = Ak @ m_filt[k] + Bk @ ukp1

    # Backward beta_k(x_k) ∝ exp(-1/2 x^T Λ_k x + η_k^T x)
    Lambda = [None] * (T + 1)
    eta = [None] * (T + 1)
    Lambda[T] = np.zeros((n_x, n_x))
    eta[T] = np.zeros((n_x,))

    for k in range(T - 1, -1, -1):
        kp1 = k + 1

        Akp1 = A_t[kp1]
        Bkp1 = B_t[kp1]
        Qkp1 = Q_t[kp1]
        Hkp1 = H_t[kp1]
        Dkp1 = D_t[kp1]
        Rkp1 = R_t[kp1]
        ukp1 = u_at(kp1)

        if kp1 == t:
            # exclude measurement term at time kp1
            barLambda = Lambda[kp1]
            barEta = eta[kp1]
        else:
            tilde_y = y[kp1] - Dkp1 @ ukp1
            barLambda = Lambda[kp1] + Hkp1.T @ np.linalg.inv(Rkp1) @ Hkp1
            barEta = eta[kp1] + Hkp1.T @ np.linalg.inv(Rkp1) @ tilde_y

        # S = Q^{-1} + barLambda
        Qinv = np.linalg.inv(Qkp1)
        S_back = Qinv + barLambda
        S_back_inv = np.linalg.inv(S_back)

        # Λ_k = A_{k+1}^T [Q^{-1} - Q^{-1} S^{-1} Q^{-1}] A_{k+1}
        core = Qinv - Qinv @ S_back_inv @ Qinv
        Lambda[k] = Akp1.T @ core @ Akp1

        # η_k = A^T Q^{-1} S^{-1} barEta - A^T Q^{-1} S^{-1} barLambda B u
        term1 = Akp1.T @ Qinv @ S_back_inv @ barEta
        term2 = Akp1.T @ Qinv @ S_back_inv @ barLambda @ (Bkp1 @ ukp1)
        eta[k] = term1 - term2

    # Combine at time t:
    P_t_minus = np.linalg.inv(np.linalg.inv(P_pred[t]) + Lambda[t])
    m_t_minus = P_t_minus @ (np.linalg.inv(P_pred[t]) @ m_pred[t] + eta[t])

    # Predictive LOO for y_t
    Ht = H_t[t]
    Dt = D_t[t]
    Rt = R_t[t]
    ut = u_at(t)

    mu_y = Ht @ m_t_minus + Dt @ ut
    Sigma_y = Ht @ P_t_minus @ Ht.T + Rt

    return [X_t, mu_y, Sigma_y]


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
    # --- optional linear drift (matrix-valued) ---
    dA: np.ndarray | None = None,    # (n_x, n_x)
    dB: np.ndarray | None = None,    # (n_x, n_u)
    dH: np.ndarray | None = None,    # (n_y, n_x)
    dD: np.ndarray | None = None,    # (n_y, n_u)
    dQ: np.ndarray | None = None,    # (n_x, n_x)
    dR: np.ndarray | None = None,    # (n_y, n_y)
    # --- control distribution ---
    u_low: float = -0.5,
    u_high: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Multidimensional LGSSM simulator with optional *linear drift* in matrices.

    Model (time index t=0..T):
        x_{t+1} = A_t x_t + B_t u_t + w_{t+1},   w_{t+1} ~ N(0, Q_t)
        y_t     = H_t x_t + D_t u_t + v_t,       v_t     ~ N(0, R_t)

    Drift parameterization (linear in time):
        A_t = A0 + dA * t    (if dA provided else constant)
        B_t = B0 + dB * t
        H_t = H0 + dH * t
        D_t = D0 + dD * t
        Q_t = Q0 + dQ * t    (you are responsible for keeping Q_t PSD)
        R_t = R0 + dR * t    (you are responsible for keeping R_t PSD)

    Returns:
        x: (T+1, n_x) states
        y: (T+1, n_y) observations
        u: (T,   n_u) controls
        mats: dict with stacked matrices over time (each (T+1, ...)):
              A_t, B_t, H_t, D_t, Q_t, R_t
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

    if A0.shape != (n_x, n_x):
        raise ValueError("A0 must be (n_x, n_x)")
    if B0.shape != (n_x, n_u):
        raise ValueError("B0 must be (n_x, n_u)")
    if H0.shape != (n_y, n_x):
        raise ValueError("H0 must be (n_y, n_x)")
    if D0.shape != (n_y, n_u):
        raise ValueError("D0 must be (n_y, n_u)")

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
        if Q0.shape != (n_x, n_x):
            raise ValueError("Q0 must be (n_x, n_x)")

    if R0 is None:
        R0 = 0.03 * np.eye(n_y, dtype=float)
    else:
        R0 = np.asarray(R0, dtype=float)
        if R0.shape != (n_y, n_y):
            raise ValueError("R0 must be (n_y, n_y)")

    # Drift defaults to zero matrices
    dA = np.zeros_like(A0) if dA is None else np.asarray(dA, dtype=float)
    dB = np.zeros_like(B0) if dB is None else np.asarray(dB, dtype=float)
    dH = np.zeros_like(H0) if dH is None else np.asarray(dH, dtype=float)
    dD = np.zeros_like(D0) if dD is None else np.asarray(dD, dtype=float)
    dQ = np.zeros_like(Q0) if dQ is None else np.asarray(dQ, dtype=float)
    dR = np.zeros_like(R0) if dR is None else np.asarray(dR, dtype=float)

    # Basic shape checks for drift
    if dA.shape != A0.shape:
        raise ValueError("dA must match A0 shape")
    if dB.shape != B0.shape:
        raise ValueError("dB must match B0 shape")
    if dH.shape != H0.shape:
        raise ValueError("dH must match H0 shape")
    if dD.shape != D0.shape:
        raise ValueError("dD must match D0 shape")
    if dQ.shape != Q0.shape:
        raise ValueError("dQ must match Q0 shape")
    if dR.shape != R0.shape:
        raise ValueError("dR must match R0 shape")

    # Build time-varying matrices (stacked)
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
        Q_t[t] = Q0 + dQ * t
        R_t[t] = R0 + dR * t

    # Controls
    u = rng.uniform(u_low, u_high, size=(T, n_u))

    # Simulate
    x = np.zeros((T + 1, n_x), dtype=float)
    y = np.zeros((T + 1, n_y), dtype=float)

    x[0] = x0
    # y_0 uses u_0 if T>0 else zeros
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

def main() -> None:
    """
    Demo main for the last two functions:
      1) simulate_lgssm_nd (with drift)
      2) loo_values_nd (multidimensional leave-one-out + X_t)

    It simulates a 2D state, 2D observation, 2D control system,
    then prints results for 5 random time indices.
    """
    # -----------------------
    # Dimensions + horizon
    # -----------------------
    n_x, n_y, n_u = 2, 2, 2
    T = 15
    seed = 2026
    rng = np.random.default_rng(seed)

    # -----------------------
    # Base matrices
    # -----------------------
    A0 = np.array([[0.65, 0.10],
                   [-0.15, 0.70]], dtype=float)

    B0 = 0.25 * np.eye(n_x, n_u)

    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.05 * np.eye(n_x)
    R0 = 0.08 * np.eye(n_y)

    # -----------------------
    # Linear drifts (optional)
    # -----------------------
    dA = np.array([[0.000, 0.000],
                   [0.000, 0.000]], dtype=float)

    dB = np.zeros_like(B0)
    dH = np.zeros_like(H0)
    dD = np.zeros_like(D0)

    dQ = np.zeros_like(Q0)  
    dR = np.zeros_like(R0)  

    # Prior
    x0 = np.array([0.5, 0.5], dtype=float)
    m0 = x0.copy()
    P0 = 0.05 * np.eye(n_x)

    # -----------------------
    # 1) Simulate ND LGSSM with drift
    # -----------------------
    x, y, u, mats = simulate_lgssm_nd(
        A0=A0, B0=B0, H0=H0, D0=D0,
        T=T, seed=seed, x0=x0,
        Q0=Q0, R0=R0,
        dA=dA, dB=dB, dH=dH, dD=dD, dQ=dQ, dR=dR,
        u_low=-0.5, u_high=0.5,
    )

    # -----------------------
    # 2) For 5 random t's: compute [X_t, mu_y, Sigma_y]
    #    where X_t is (n_x,n_y), mu_y is (n_y,), Sigma_y is (n_y,n_y)
    # -----------------------
    t_list = np.sort(rng.choice(np.arange(T + 1), size=min(5, T + 1), replace=False))

    for t in t_list:
        X_t, mu_y, Sigma_y = loo_values_nd(
            t=int(t),
            y=y,
            u=u,
            A_t=mats["A_t"],
            B_t=mats["B_t"],
            H_t=mats["H_t"],
            D_t=mats["D_t"],
            Q_t=mats["Q_t"],
            R_t=mats["R_t"],
            P0=P0,
            m0=m0,
        )

        print(f"\n=== t={int(t)} ===")
        print("X_t shape:", X_t.shape)
        print("X_t:\n", X_t)
        print("mu_y_t_given_minus_t:", mu_y)
        print("Sigma_y_t_given_minus_t:\n", Sigma_y)


if __name__ == "__main__":
    main()
