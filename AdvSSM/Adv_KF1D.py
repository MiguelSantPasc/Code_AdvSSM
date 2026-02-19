#!/usr/bin/env python3
"""
1D LGSSM:
x_{t+1} = A x_t + B u_t + w_{t+1}
y_t     = H x_t + D u_t + v_t

Figure:
True x_t, RTS smoothed mean, and shaded CI from smoothed variance.

Also computes and returns a dictionary (lists over t=0..T) with:
P_{t|t}, P_{t|t-1}, A_t (with drift), B_t (with drift), H_t (with drift),
Q_t, R_t, S_t and K_t, plus J_t (RTS gain, 1D).

In 1D:
S_t = H_t^2 * P_{t|t-1} + R_t
K_t = P_{t|t-1} * H_t / S_t

Additionally, computes for a given t the formula you provided (with l = T - t):

    value(t) = ( sum_{i=0}^{l} [
                   ( Π_{j=0}^{i-1}→ J_{t+j} )
                   ( 1 - J_{t+i} A_{t+i+1} )
                   ( Π_{j=0}^{i}← (1 - K_{t+j} H_{t+j}) A_{t+j} )
                ] ) * K_t

Conventions in 1D:
- "I" = 1
- J_T is set to 0 (so when t+i == T the middle factor becomes 1 automatically).
"""

from __future__ import annotations

import os
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
    T = y.shape[0] - 1

    m_pred = np.zeros(T + 1)
    P_pred = np.zeros(T + 1)
    m_filt = np.zeros(T + 1)
    P_filt = np.zeros(T + 1)

    m_pred[0] = m0
    P_pred[0] = P0

    for t in range(T + 1):
        u_t = u[t] if t < T else u[T - 1]

        if np.isnan(y[t]):
            m_filt[t] = m_pred[t]
            P_filt[t] = P_pred[t]
        else:
            y_hat = H * m_pred[t] + D * u_t
            S = H * P_pred[t] * H + R
            K = P_pred[t] * H / S

            innov = y[t] - y_hat
            m_filt[t] = m_pred[t] + K * innov
            P_filt[t] = (1.0 - K * H) * P_pred[t]

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
    T = m_filt.shape[0] - 1

    m_smooth = np.zeros_like(m_filt)
    P_smooth = np.zeros_like(P_filt)

    m_smooth[T] = m_filt[T]
    P_smooth[T] = P_filt[T]

    for t in range(T - 1, -1, -1):
        C = P_filt[t] * A / P_pred[t + 1]
        m_smooth[t] = m_filt[t] + C * (m_smooth[t + 1] - m_pred[t + 1])
        P_smooth[t] = P_filt[t] + C * (P_smooth[t + 1] - P_pred[t + 1]) * C

    return m_smooth, P_smooth


def plot_true_and_smoothed_with_ci_save(
    x: np.ndarray,
    m_smooth: np.ndarray,
    P_smooth: np.ndarray,
    title: str,
    outpath: str,
    ci_sigma: float = 1.96,
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

    t = np.arange(x.shape[0])

    c_true = "#4C72B0"
    c_smooth = "#C44E52"

    sd = np.sqrt(np.maximum(P_smooth, 0.0))
    lower = m_smooth - ci_sigma * sd
    upper = m_smooth + ci_sigma * sd

    fig = plt.figure(figsize=(10.6, 5.0))
    ax = fig.add_subplot(111)

    ax.fill_between(t, lower, upper, alpha=0.20, label=f"Smoothed CI (±{ci_sigma:.2f}σ)")
    ax.plot(t, m_smooth, linewidth=1.25, color=c_smooth, label="RTS smoothed mean")
    ax.plot(t, x, marker="o", markersize=3.0, linewidth=1.05, color=c_true, label="True state $x_t$")

    ax.set_title(title)
    ax.set_xlabel("Time t")
    ax.set_ylabel("State value")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def build_kf_cov_sequences_1d(
    T: int,
    P0: float,
    A: float,
    B: float,
    H: float,
    Q: float,
    R: float,
    dA: float = 0.0,
    dB: float = 0.0,
    dH: float = 0.0,
    dQ: float = 0.0,
    dR: float = 0.0,
) -> dict[str, list[float]]:
    if T < 0:
        raise ValueError("T must be >= 0")
    if P0 < 0:
        raise ValueError("P0 must be >= 0")

    A_t = [float(A + dA * t) for t in range(T + 1)]
    B_t = [float(B + dB * t) for t in range(T + 1)]
    H_t = [float(H + dH * t) for t in range(T + 1)]
    Q_t = [float(Q + dQ * t) for t in range(T + 1)]
    R_t = [float(R + dR * t) for t in range(T + 1)]

    P_tt1 = [0.0] * (T + 1)
    P_tt = [0.0] * (T + 1)
    S_t = [0.0] * (T + 1)
    K_t = [0.0] * (T + 1)

    P_tt1[0] = float(P0)

    for t in range(T + 1):
        # S_t = H_t^2 * P_{t|t-1} + R_t
        S_t[t] = (H_t[t] ** 2) * P_tt1[t] + R_t[t]
        if S_t[t] <= 0:
            raise ValueError(f"S_t must be > 0 (got S_{t}={S_t[t]}).")

        # K_t = P_{t|t-1} * H_t / S_t
        K_t[t] = (P_tt1[t] * H_t[t]) / S_t[t]

        # P_{t|t} = (1 - K_t H_t) P_{t|t-1}
        P_tt[t] = (1.0 - K_t[t] * H_t[t]) * P_tt1[t]

        if t < T:
            P_tt1[t + 1] = (A_t[t] ** 2) * P_tt[t] + Q_t[t]

    # RTS gain J_t (1D): J_t = P_{t|t} A_t / P_{t+1|t}, J_T=0
    J_t = [0.0] * (T + 1)
    for t in range(T):
        denom = P_tt1[t + 1]
        if denom <= 0:
            raise ValueError(f"P_tt1[{t+1}] must be > 0 to compute J_t.")
        J_t[t] = (P_tt[t] * A_t[t]) / denom
    J_t[T] = 0.0

    return {
        "T": T,
        "A_t": A_t,
        "B_t": B_t,
        "H_t": H_t,
        "Q_t": Q_t,
        "R_t": R_t,
        "P_tt1": P_tt1,
        "P_tt": P_tt,
        "S_t": S_t,
        "K_t": K_t,
        "J_t": J_t,
    }


def print_sequences(seq: dict[str, list[float]], keys: list[str], max_print: int = 10) -> None:
    for k in keys:
        vals = seq[k]
        if len(vals) <= max_print:
            show = vals
            suffix = ""
        else:
            show = vals[:max_print]
            suffix = f" ... (total {len(vals)} values)"
        print(f"{k}: {[round(v, 6) for v in show]}{suffix}")


def _prod_right(vals: list[float]) -> float:
    out = 1.0
    for v in vals:
        out *= v
    return out


def _prod_left(vals: list[float]) -> float:
    out = 1.0
    for v in reversed(vals):
        out *= v
    return out


def compute_X(t: int, seq: dict[str, list[float]]) -> float:
    """
    value(t) = ( sum_{i=0}^{l} [
                   ( Π_{j=0}^{i-1}→ J_{t+j} )
                   ( 1 - J_{t+i} A_{t+i+1} )
                   ( Π_{j=0}^{i}← (1 - K_{t+j} H_{t+j}) A_{t+j} )
                ] ) * K_t
    with l = T - t.

    1D conventions:
      I = 1
      J_T = 0 (so if t+i == T, the middle factor is 1)
    """
    T = int(seq["T"])
    if not (0 <= t <= T):
        raise ValueError(f"t must be between 0 and T={T}")

    A_t = seq["A_t"]
    H_t = seq["H_t"]
    K_t = seq["K_t"]
    J_t = seq["J_t"]

    l = T - t
    total = 0.0

    for i in range(l + 1):
        prodJ = 1.0 if i == 0 else _prod_right([J_t[t + j] for j in range(i)])

        if t + i >= T:
            mid = 1.0
        else:
            mid = 1.0 - J_t[t + i] * A_t[t + i + 1]

        factors = [((1.0 - K_t[t + j] * H_t[t + j]) * A_t[t + j]) for j in range(i + 1)]
        prodKH = _prod_left(factors)

        total += prodJ * mid * prodKH

    return total * K_t[t]


def main() -> None:
    A = 0.35
    B = 0.85
    H = 2.5
    D = 0.0

    T = 25
    seed = 2026
    x0 = 0.5

    Q = 0.8
    R = 0.5

    x, y, u = simulate_lgssm_1d(A=A, B=B, H=H, D=D, T=T, seed=seed, x0=x0, Q=Q, R=R)

    m0 = x0
    P0 = 0.05

    m_filt, P_filt, m_pred, P_pred = kalman_filter_1d(
        y=y, u=u, A=A, B=B, H=H, D=D, Q=Q, R=R, m0=m0, P0=P0
    )
    m_smooth, P_smooth = rts_smoother_1d(
        m_filt=m_filt, P_filt=P_filt, m_pred=m_pred, P_pred=P_pred, A=A
    )

    # Save plot into ./output/ (created if needed)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rts_smoother_T{T}_seed{seed}.png")

    plot_true_and_smoothed_with_ci_save(
        x=x,
        m_smooth=m_smooth,
        P_smooth=P_smooth,
        title=f"1D LGSSM (T={T}, fixed seed): RTS smoother",
        outpath=out_path,
        ci_sigma=1.96,
    )
    print(f"Saved plot to: {out_path}")

    seq = build_kf_cov_sequences_1d(
        T=T, P0=P0,
        A=A, B=B, H=H, Q=Q, R=R,
        dA=0.0, dB=0.0, dH=0.0, dQ=0.0, dR=0.0,
    )

    print("\nSequences (print-all if <=10 else first 10):")
    print_sequences(
        seq,
        keys=["A_t", "B_t", "H_t", "Q_t", "R_t", "P_tt1", "P_tt", "S_t", "K_t", "J_t"],
        max_print=10,
    )

    print("\nFormula values (your image), for each t=0..T:")
    for t in range(T + 1):
        val = compute_X(t=t, seq=seq)
        print(f"t={t:2d}  value={val:.10f}")


if __name__ == "__main__":
    main()
