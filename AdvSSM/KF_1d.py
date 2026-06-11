#!/usr/bin/env python3
"""
1D Linear Gaussian state-space model with 1D observation and 1D control.

x_{t+1} = A x_t + B u_t + w_{t+1}
y_t     = H x_t + D u_t + v_t

This script simulates (x, y) but plots ONLY:
- true state x_t
- RTS smoothed mean
- shaded confidence interval based on smoothed variance
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


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
    """
    Simulate a scalar LGSSM trajectory.

    The hidden state and observation are generated as:

        x_{t+1} = A x_t + B u_t + w_{t+1},    w_{t+1} ~ N(0, Q)
        y_t     = H x_t + D u_t + v_t,        v_t     ~ N(0, R)

    Returns:
        x: true latent states with shape (T+1,)
        y: noisy observations with shape (T+1,)
        u: controls with shape (T,)
    """
    rng = np.random.default_rng(seed)

    u = rng.uniform(0.0, 1.0, size=T)
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
    """
    Run the scalar Kalman filter.

    Prediction:
        m_{t|t-1} = A m_{t-1|t-1} + B u_{t-1}
        P_{t|t-1} = A^2 P_{t-1|t-1} + Q

    Update:
        S_t = H^2 P_{t|t-1} + R
        K_t = P_{t|t-1} H / S_t
        m_{t|t} = m_{t|t-1} + K_t (y_t - H m_{t|t-1} - D u_t)
        P_{t|t} = (1 - K_t H) P_{t|t-1}
    """
    T = y.shape[0] - 1

    m_pred = np.zeros(T + 1)
    P_pred = np.zeros(T + 1)
    m_filt = np.zeros(T + 1)
    P_filt = np.zeros(T + 1)

    m_pred[0] = m0
    P_pred[0] = P0

    for t in range(T + 1):
        u_t = u[t] if t < T else u[T - 1]  # only matters if D != 0
        y_hat = H * m_pred[t] + D * u_t
        innovation_variance = H * P_pred[t] * H + R
        kalman_gain = P_pred[t] * H / innovation_variance

        innovation = y[t] - y_hat
        m_filt[t] = m_pred[t] + kalman_gain * innovation
        P_filt[t] = (1.0 - kalman_gain * H) * P_pred[t]

        if t < T:
            m_pred[t + 1] = A * m_filt[t] + B * u[t]
            P_pred[t + 1] = A * P_filt[t] * A + Q

    return m_filt, P_filt, m_pred, P_pred


def rts_smoother_1d(
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the Rauch-Tung-Striebel smoother for the scalar filtered sequence.

    Backward recursion for t = T-1,...,0:

        C_t = P_{t|t} A / P_{t+1|t}
        m_{t|T} = m_{t|t} + C_t (m_{t+1|T} - m_{t+1|t})
        P_{t|T} = P_{t|t} + C_t^2 (P_{t+1|T} - P_{t+1|t})
    """
    T = m_filt.shape[0] - 1

    m_smooth = np.zeros_like(m_filt)
    P_smooth = np.zeros_like(P_filt)

    m_smooth[T] = m_filt[T]
    P_smooth[T] = P_filt[T]

    for t in range(T - 1, -1, -1):
        smoothing_gain = P_filt[t] * A / P_pred[t + 1]
        m_smooth[t] = m_filt[t] + smoothing_gain * (m_smooth[t + 1] - m_pred[t + 1])
        P_smooth[t] = P_filt[t] + smoothing_gain * (P_smooth[t + 1] - P_pred[t + 1]) * smoothing_gain

    return m_smooth, P_smooth


def plot_true_and_smoothed_with_ci(
    x: np.ndarray,
    m_smooth: np.ndarray,
    P_smooth: np.ndarray,
    title: str,
    ci_sigma: float = 1.96,  # ~95% if Gaussian
) -> None:
    """Plot the true state and RTS smoothed mean with a Gaussian confidence band."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    t = np.arange(x.shape[0])

    # Muted colors
    true_state_color = "#4C72B0"
    smoothed_mean_color = "#C44E52"

    smooth_std = np.sqrt(np.maximum(P_smooth, 0.0))
    lower = m_smooth - ci_sigma * smooth_std
    upper = m_smooth + ci_sigma * smooth_std

    fig = plt.figure(figsize=(10.2, 4.9))
    ax = fig.add_subplot(111)

    ax.fill_between(
        t, lower, upper,
        alpha=0.20,
        label=f"{int(round(100 * (1 - 2 * (1 - 0.975))))}% CI (+/-{ci_sigma:.2f} sigma)"
        if abs(ci_sigma - 1.96) < 1e-6 else f"CI (+/-{ci_sigma:.2f} sigma)",
    )
    ax.plot(t, m_smooth, linewidth=1.25, color=smoothed_mean_color, label="RTS smoothed mean")
    ax.plot(t, x, marker="o", markersize=3.2, linewidth=1.05, color=true_state_color, label="True state $x_t$")

    ax.set_title(title)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    plt.show()


def main() -> None:
    A = 0.45
    B = 0.15
    H = 1.1
    D = 0.0

    T = 50
    seed = 2026
    x0 = 1.0

    Q = 0.02
    R = 0.03

    x, y, u = simulate_lgssm_1d(A=A, B=B, H=H, D=D, T=T, seed=seed, x0=x0, Q=Q, R=R)

    m0 = x0
    P0 = 0.05

    m_filt, P_filt, m_pred, P_pred = kalman_filter_1d(
        y=y, u=u, A=A, B=B, H=H, D=D, Q=Q, R=R, m0=m0, P0=P0
    )
    m_smooth, P_smooth = rts_smoother_1d(
        m_filt=m_filt, P_filt=P_filt, m_pred=m_pred, P_pred=P_pred, A=A
    )

    plot_true_and_smoothed_with_ci(
        x=x,
        m_smooth=m_smooth,
        P_smooth=P_smooth,
        title=f"RTS smoother vs true state (T={T}, fixed seed) with confidence band",
        ci_sigma=1.96,
    )


if __name__ == "__main__":
    main()
