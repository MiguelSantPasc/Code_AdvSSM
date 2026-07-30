#!/usr/bin/env python3
"""
1D LGSSM:
x_{t+1} = A x_t + B u_t + w_{t+1}
y_t     = H x_t + D u_t + v_t

Figure with two stacked panels:
(Top)  True x_t, RTS smoothed mean, and shaded CI from smoothed variance.
(Bottom) For every t: boxplot of p(y_t | y_{-t}) (leave-one-out), with the realized y_t overlaid.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def compute_loo_y_box_samples(
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
    seed: int,
    n_samples: int = 800,
) -> list[np.ndarray]:
    """
    For each t, approximate p(y_t | y_{-t}) by:
    - set y[t]=NaN, run KF+RTS => x_t | y_{-t} ~ N(m, P)
    - then y_t | y_{-t} ~ N(H m + D u_t, H^2 P + R)
    - sample n_samples points for boxplot
    """
    T = y.shape[0] - 1
    rng = np.random.default_rng(seed)

    samples_list: list[np.ndarray] = []

    for t0 in range(T + 1):
        y_loo = y.copy()
        y_loo[t0] = np.nan

        m_filt_loo, P_filt_loo, m_pred_loo, P_pred_loo = kalman_filter_1d(
            y=y_loo, u=u, A=A, B=B, H=H, D=D, Q=Q, R=R, m0=m0, P0=P0
        )
        m_smooth_loo, P_smooth_loo = rts_smoother_1d(
            m_filt=m_filt_loo, P_filt=P_filt_loo, m_pred=m_pred_loo, P_pred=P_pred_loo, A=A
        )

        u_t0 = u[t0] if t0 < T else u[T - 1]
        mu_y = H * m_smooth_loo[t0] + D * u_t0
        var_y = (H * H) * P_smooth_loo[t0] + R

        s = rng.normal(loc=mu_y, scale=np.sqrt(max(var_y, 0.0)), size=n_samples)
        samples_list.append(s)

    return samples_list


def plot_stacked(
    x: np.ndarray,
    y: np.ndarray,
    m_smooth: np.ndarray,
    P_smooth: np.ndarray,
    y_box_samples: list[np.ndarray],
    title: str,
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
    c_y = "#444444"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.8, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.35]}
    )

    # --- Top: x_t vs smoother with CI
    sd = np.sqrt(np.maximum(P_smooth, 0.0))
    lower = m_smooth - ci_sigma * sd
    upper = m_smooth + ci_sigma * sd

    ax_top.fill_between(t, lower, upper, alpha=0.20, label=f"Smoothed CI (±{ci_sigma:.2f}σ)")
    ax_top.plot(t, m_smooth, linewidth=1.25, color=c_smooth, label="RTS smoothed mean")
    ax_top.plot(t, x, marker="o", markersize=3.0, linewidth=1.05, color=c_true, label="True state $x_t$")

    ax_top.set_title(title)
    ax_top.set_ylabel("State value")
    ax_top.grid(True, alpha=0.20)
    ax_top.legend(loc="best", frameon=True)

    # --- Bottom: boxplots of p(y_t | y_{-t}) + realized y_t
    positions = np.arange(len(y_box_samples))
    bp = ax_bot.boxplot(
        y_box_samples,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
    )

    for box in bp["boxes"]:
        box.set_alpha(0.18)
    for median in bp["medians"]:
        median.set_linewidth(1.05)

    # Overlay realized y_t
    ax_bot.plot(
        t, y,
        marker="o",
        markersize=2.8,
        linewidth=0.9,
        color=c_y,
        alpha=0.85,
        label="Realized $y_t$",
        zorder=3
    )

    ax_bot.set_xlabel("Time t")
    ax_bot.set_ylabel(r"$y_t \mid y_{-t}$")
    ax_bot.grid(True, axis="y", alpha=0.20)
    ax_bot.legend(loc="best", frameon=True)

    fig.tight_layout()
    plt.show()


def main() -> None:
    A = 0.65
    B = 1.85
    H = 2.5
    D = 0.0

    T = 50
    seed = 2026
    x0 = 0.5

    Q = 0.08
    R = 0.05

    x, y, u = simulate_lgssm_1d(A=A, B=B, H=H, D=D, T=T, seed=seed, x0=x0, Q=Q, R=R)

    m0 = x0
    P0 = 0.05

    # RTS smoother with all data
    m_filt, P_filt, m_pred, P_pred = kalman_filter_1d(
        y=y, u=u, A=A, B=B, H=H, D=D, Q=Q, R=R, m0=m0, P0=P0
    )
    m_smooth, P_smooth = rts_smoother_1d(
        m_filt=m_filt, P_filt=P_filt, m_pred=m_pred, P_pred=P_pred, A=A
    )

    # Leave-one-out boxplot samples for all t
    y_box_samples = compute_loo_y_box_samples(
        y=y, u=u, A=A, B=B, H=H, D=D, Q=Q, R=R, m0=m0, P0=P0,
        seed=seed + 999, n_samples=800
    )

    plot_stacked(
        x=x,
        y=y,
        m_smooth=m_smooth,
        P_smooth=P_smooth,
        y_box_samples=y_box_samples,
        title=f"1D LGSSM (T={T}, fixed seed): RTS smoother + leave-one-out $p(y_t|y_{{-t}})$",
        ci_sigma=1.96,
    )


import os as _os
import sys as _sys

# Make `shared_ssm` importable when this legacy script is run directly.
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from shared_ssm.legacy import kalman_filter_1d
from shared_ssm.legacy import rts_smoother_1d
from shared_ssm.legacy import simulate_lgssm_1d


if __name__ == "__main__":
    main()
