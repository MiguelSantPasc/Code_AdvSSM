from __future__ import annotations
import os
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


import matplotlib.pyplot as plt


def _ellipse_points_2d(center: np.ndarray, cov: np.ndarray, chi2_val: float = 5.991, n: int = 240) -> np.ndarray:
    """
    Return (n,2) points of the ellipse:
        (z-center)^T cov^{-1} (z-center) = chi2_val
    For 95% in 2D: chi2_val ≈ 5.991.
    """
    center = np.asarray(center, dtype=float).reshape(2,)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)

    # eigendecomposition (cov must be PSD)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)

    # radii = sqrt(chi2 * eigenvalues)
    radii = np.sqrt(chi2_val * vals)

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2,n)

    # transform circle -> ellipse
    ellipse = (vecs @ (radii[:, None] * circle)).T + center[None, :]  # (n,2)
    return ellipse


def plot_two_ellipses_for_t(
    *,
    t: int,
    y: np.ndarray,          # (T+1, 2)
    mu_y: np.ndarray,       # (2,)
    Sigma_y: np.ndarray,    # (2,2)
    X_t: np.ndarray,        # (n_x, 2)
    out_path: str | None = None,
    chi2_val: float = 5.991,  # 95% in 2D
) -> None:
    """
    One figure with:
      - ellipse centered at mu_y with covariance Sigma_y
      - ellipse centered at y[t] with covariance (X_t^T X_t)
    plus markers for mu_y and y[t].

    Requires n_y = 2.
    """
    if y.shape[1] != 2:
        raise ValueError("This plotting helper requires 2D observations (n_y=2).")
    if mu_y.shape != (2,):
        raise ValueError("mu_y must be shape (2,).")
    if Sigma_y.shape != (2, 2):
        raise ValueError("Sigma_y must be shape (2,2).")
    if X_t.shape[1] != 2:
        raise ValueError("X_t must have shape (n_x, 2) so that X_t^T X_t is (2,2).")
    if not (0 <= t < y.shape[0]):
        raise ValueError("t out of range for y")

    y_t = y[t].astype(float)
    cov2 = (X_t.T @ X_t).astype(float)

    e1 = _ellipse_points_2d(mu_y, Sigma_y, chi2_val=chi2_val)
    e2 = _ellipse_points_2d(y_t, cov2, chi2_val=chi2_val)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_subplot(111)

    # Ellipse 1: (mu_y, Sigma_y)
    ax.plot(e1[:, 0], e1[:, 1], linewidth=1.6, label=r"Ellipse: $(\mu_y,\Sigma_y)$")
    ax.scatter([mu_y[0]], [mu_y[1]], s=40, marker="o", label=r"Center $\mu_y$")

    # Ellipse 2: (y_t, X^T X)
    ax.plot(e2[:, 0], e2[:, 1], linewidth=1.6, linestyle="--", label=r"Ellipse: $(y_t, X_t^\top X_t)$")
    ax.scatter([y_t[0]], [y_t[1]], s=50, marker="x", label=r"Point $y_t$")

    ax.set_title(f"Ellipses at t={t} (chi2={chi2_val:.3f})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.20)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


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
    A0 = np.array([[0.65, 0.40],
                   [-0.15, 0.70]], dtype=float)

    B0 = np.array([[1.65, 0.40],
                   [-0.15, 0.70]], dtype=float)
    
    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 =  0.3* np.array([[1.6, -0.40],
                   [0.15, 0.70]], dtype=float)
    
    R0 =  0.2* np.array([[0.65, 1.40],
                   [-0.15, 1.70]], dtype=float)
    
    Q0 = 0.5 * (Q0 + Q0.T)
    R0 = 0.5 * (R0 + R0.T)  

    # -----------------------
    # Linear drifts (optional)
    # -----------------------
    dA = np.array([[0.002, 0.000],
                   [0.000, -0.001]], dtype=float)

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

    t = 7

    X_t, mu_y, Sigma_y = loo_values_nd(
        t=t,
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

    plot_two_ellipses_for_t(
        t=t,
        y=y,              # (T+1,2)
        mu_y=mu_y,        # (2,)
        Sigma_y=Sigma_y,  # (2,2)
        X_t=X_t,          # (n_x,2)
        out_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/ellipses_t5.png"),
    )

    

if __name__ == "__main__":
    main()
