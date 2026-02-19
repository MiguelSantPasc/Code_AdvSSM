#!/usr/bin/env python3
"""
Linear Gaussian state-space model (2D state, 2D observation, 2D control).

x_{t+1} = A x_t + B u_t + w_{t+1}
y_t     = H x_t + D u_t + v_t

- u_t ~ Uniform([0,1]x[0,1]) i.i.d.
- w_t, v_t independent Gaussian noises
- Defaults: H = I, D = 0
- Simulate and plot x_t and y_t in the plane.
- Time index is annotated on both x_t and y_t.
- Also plots Kalman filter (filtered means) and RTS smoother (smoothed means).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


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
    rng = np.random.default_rng(seed)

    n = A.shape[0]
    m = H.shape[0]
    k = B.shape[1]

    if x0 is None:
        x0 = np.zeros(n)
    if Q is None:
        Q = 0.02 * np.eye(n)
    if R is None:
        R = 0.03 * np.eye(m)

    u = rng.uniform(0.0, 1.0, size=(T, k))
    w = rng.multivariate_normal(mean=np.zeros(n), cov=Q, size=T + 1)
    v = rng.multivariate_normal(mean=np.zeros(m), cov=R, size=T + 1)

    x = np.zeros((T + 1, n))
    y = np.zeros((T + 1, m))

    x[0] = x0
    y[0] = H @ x[0] + (D @ u[0] if T > 0 else 0.0) + v[0]

    for t in range(T):
        x[t + 1] = A @ x[t] + B @ u[t] + w[t + 1]
        y[t + 1] = H @ x[t + 1] + D @ u[t] + v[t + 1]

    return x, y, u


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
    """
    Returns:
        m_filt: (T+1, n) filtered means E[x_t | y_0..y_t]
        P_filt: (T+1, n, n) filtered covariances
        m_pred: (T+1, n) predicted means E[x_t | y_0..y_{t-1}] (m_pred[0]=m0)
        P_pred: (T+1, n, n) predicted covariances
    """
    T = y.shape[0] - 1
    n = A.shape[0]

    m_pred = np.zeros((T + 1, n))
    P_pred = np.zeros((T + 1, n, n))
    m_filt = np.zeros((T + 1, n))
    P_filt = np.zeros((T + 1, n, n))

    m_pred[0] = m0
    P_pred[0] = P0

    for t in range(T + 1):
        # Update using y_t
        u_t = u[t] if t < T else u[T - 1]  # for t=T, D@u not essential when D=0; keep defined
        y_hat = H @ m_pred[t] + D @ u_t
        S = H @ P_pred[t] @ H.T + R
        K = P_pred[t] @ H.T @ np.linalg.inv(S)

        innov = y[t] - y_hat
        m_filt[t] = m_pred[t] + K @ innov
        P_filt[t] = (np.eye(n) - K @ H) @ P_pred[t]

        # Predict next
        if t < T:
            m_pred[t + 1] = A @ m_filt[t] + B @ u[t]
            P_pred[t + 1] = A @ P_filt[t] @ A.T + Q

    return m_filt, P_filt, m_pred, P_pred


def rts_smoother(
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rauch–Tung–Striebel smoother.
    Returns:
        m_smooth: (T+1, n) smoothed means E[x_t | y_0..y_T]
        P_smooth: (T+1, n, n) smoothed covariances
    """
    T = m_filt.shape[0] - 1
    n = A.shape[0]

    m_smooth = np.zeros_like(m_filt)
    P_smooth = np.zeros_like(P_filt)

    m_smooth[T] = m_filt[T]
    P_smooth[T] = P_filt[T]

    for t in range(T - 1, -1, -1):
        C = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t + 1])
        m_smooth[t] = m_filt[t] + C @ (m_smooth[t + 1] - m_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred[t + 1]) @ C.T

    return m_smooth, P_smooth


def plot_all(
    x: np.ndarray,
    y: np.ndarray,
    m_filt: np.ndarray,
    m_smooth: np.ndarray,
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    fig = plt.figure(figsize=(9.2, 6.2))
    ax = fig.add_subplot(111)

    # Muted palette
    state_color = "#4C72B0"
    obs_color = "#55A868"
    filt_color = "#8172B2"   # muted purple
    smooth_color = "#C44E52" # muted red

    # Thinner lines
    ax.plot(
        x[:, 0], x[:, 1],
        marker="o", markersize=4.0,
        linewidth=1.25,
        color=state_color,
        label="True state $x_t$",
        alpha=0.95,
    )
    ax.plot(
        y[:, 0], y[:, 1],
        marker="s", markersize=3.7,
        linestyle="--",
        linewidth=1.05,
        color=obs_color,
        label="Observation $y_t$",
        alpha=0.85,
    )
    ax.plot(
        m_filt[:, 0], m_filt[:, 1],
        marker=".", markersize=5.0,
        linestyle="-",
        linewidth=1.15,
        color=filt_color,
        label="Kalman filtered mean",
        alpha=0.95,
    )
    ax.plot(
        m_smooth[:, 0], m_smooth[:, 1],
        marker=".", markersize=5.0,
        linestyle="-.",
        linewidth=1.15,
        color=smooth_color,
        label="RTS smoothed mean",
        alpha=0.95,
    )

    ax.scatter([x[0, 0]], [x[0, 1]], s=105, marker="*", color=state_color, label="Start ($x_0$)", zorder=3)
    ax.scatter([x[-1, 0]], [x[-1, 1]], s=65, marker="X", color=state_color, label="End ($x_T$)", zorder=3)

    # Time labels on both x_t and y_t
    for t in range(x.shape[0]):
        ax.text(
            x[t, 0], x[t, 1],
            f"{t}",
            ha="center", va="center",
            fontsize=8,
            alpha=0.70,
            color="black",
            zorder=4,
        )
        ax.annotate(
            f"{t}",
            xy=(y[t, 0], y[t, 1]),
            xytext=(5, -6),
            textcoords="offset points",
            fontsize=8,
            alpha=0.65,
            color="black",
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", frameon=True)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    plt.show()


def main() -> None:
    A = np.array([[0.85, 0.20],
                  [-0.25, 0.90]], dtype=float)

    B = np.array([[0.05, 0.005],
                  [0.001, 0.02]], dtype=float)

    H = np.eye(2, dtype=float)
    D = np.zeros((2, 2), dtype=float)

    T = 8
    seed = 2026
    x0 = np.array([1.0, 1.0], dtype=float)

    Q = 0.002 * np.eye(2)
    R = 0.003 * np.eye(2)

    x, y, u = simulate_lgssm(A=A, B=B, H=H, D=D, T=T, seed=seed, x0=x0, Q=Q, R=R)

    # Prior for Kalman filter
    m0 = x0.copy()
    P0 = 0.05 * np.eye(2)

    m_filt, P_filt, m_pred, P_pred = kalman_filter(
        y=y, u=u, A=A, B=B, H=H, D=D, Q=Q, R=R, m0=m0, P0=P0
    )
    m_smooth, P_smooth = rts_smoother(
        m_filt=m_filt, P_filt=P_filt, m_pred=m_pred, P_pred=P_pred, A=A
    )

    plot_all(
        x=x,
        y=y,
        m_filt=m_filt,
        m_smooth=m_smooth,
        title=f"LGSSM simulation (T={T}, fixed seed) — true/obs + Kalman filter/smoother",
    )


if __name__ == "__main__":
    main()
