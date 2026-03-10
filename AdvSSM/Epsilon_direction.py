#!/usr/bin/env python3
"""
KKTOpt_multi_epsilon_tangent.py

Single geometry plot in observation space for multiple epsilon values:
- constraint ellipses superposed (NOT filled)
- objective level-set ellipses superposed (NOT filled)
- tangent points y*(epsilon) clearly shown
- soft / muted colors
- output saved into ./output/
"""

from __future__ import annotations

import os
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt


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
# ND LGSSM simulator with optional linear drift
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
            raise ValueError("x0 must be (n_x,)")

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

    for t in range(T + 1):
        A_t[t] = A0 + dA * t
        B_t[t] = B0 + dB * t
        H_t[t] = H0 + dH * t
        D_t[t] = D0 + dD * t
        Q_t[t] = project_to_psd(Q0 + dQ * t)
        R_t[t] = project_to_psd(R0 + dR * t)

    u = rng.uniform(u_low, u_high, size=(T, n_u))

    x = np.zeros((T + 1, n_x), dtype=float)
    y = np.zeros((T + 1, n_y), dtype=float)
    x[0] = x0

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


# ============================================================
# Leave-one-out p(y_t | y_-t) + X_t
# ============================================================
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
    I_x = np.eye(n_x)

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

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

    J = [np.zeros((n_x, n_x)) for _ in range(T + 1)]
    for k in range(T):
        Ak = A_t[k]
        J[k] = P_filt[k] @ Ak.T @ np.linalg.inv(P_pred[k + 1])

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

    X_t_out = total @ K_kf[t]

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

    P_t_minus = np.linalg.inv(np.linalg.inv(P_pred[t]) + Lambda[t])
    m_t_minus = P_t_minus @ (np.linalg.inv(P_pred[t]) @ m_pred[t] + eta[t])

    mu_y = H_t[t] @ m_t_minus + D_t[t] @ u_at(t)
    Sigma_y = H_t[t] @ P_t_minus @ H_t[t].T + project_to_psd(R_t[t])

    return [X_t_out, mu_y, Sigma_y]


# ============================================================
# KKT solver
# ============================================================
def solve_kkt_max_quadratic_over_ellipsoid(
    *,
    X: np.ndarray,
    y_t: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    tol: float = 1e-12,
    max_iter: int = 250,
) -> tuple[np.ndarray, float]:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    Sigma = project_to_psd(Sigma)
    S = sqrtm_psd(Sigma)
    M = project_to_psd(symmetrize(X.T @ X))

    d = (mu - y_t).reshape(-1)
    A = symmetrize(S.T @ M @ S)
    b = (S.T @ M @ d).reshape(-1)

    a, U = np.linalg.eigh(A)
    a_max = float(np.max(a))
    bp = U.T @ b

    if np.linalg.norm(b) < 1e-14:
        idx = int(np.argmax(a))
        zp = np.zeros_like(bp)
        zp[idx] = np.sqrt(epsilon)
        z = U @ zp
    else:
        def g(lam: float) -> float:
            denom = (a - lam)
            zi = -bp / denom
            return float(np.dot(zi, zi) - epsilon)

        lam_low = a_max + 1e-12
        lam_high = a_max + 1.0

        while g(lam_high) > 0:
            lam_high *= 2.0
            if lam_high > 1e14:
                raise RuntimeError("Failed to bracket lambda in KKT solve.")

        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_low + lam_high)
            f_mid = g(lam_mid)
            if abs(f_mid) < tol:
                lam_low = lam_high = lam_mid
                break
            if f_mid > 0:
                lam_low = lam_mid
            else:
                lam_high = lam_mid

        lam_star = 0.5 * (lam_low + lam_high)
        zp = -bp / (a - lam_star)
        z = U @ zp

        nz = np.linalg.norm(z)
        if nz > 0:
            z = z * (np.sqrt(epsilon) / nz)

    y_star = mu + S @ z
    obj_star = float(np.linalg.norm(X @ (y_star - y_t)) ** 2)
    return y_star, obj_star


# ============================================================
# Geometry helpers
# ============================================================
def _ellipse_points_from_quad(
    center: np.ndarray,
    shape_inv: np.ndarray,
    level: float,
    n: int = 400,
) -> np.ndarray:
    center = np.asarray(center, dtype=float).reshape(2,)
    M = project_to_psd(symmetrize(np.asarray(shape_inv, dtype=float).reshape(2, 2)))

    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-14)
    radii = np.sqrt(level / w)

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    pts = (V @ (radii[:, None] * circle)).T + center[None, :]
    return pts


def _set_plot_theme() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
    })


def _style_axis(ax) -> None:
    ax.set_facecolor("#FCFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.50)
    ax.spines["bottom"].set_alpha(0.50)
    ax.grid(True, alpha=0.18)


def _points_limits(points: list[np.ndarray], pad_frac: float = 0.12):
    arrs = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p[None, :]
        if p.size > 0:
            arrs.append(p[:, :2])

    P = np.vstack(arrs)
    xmin, ymin = np.min(P[:, 0]), np.min(P[:, 1])
    xmax, ymax = np.max(P[:, 0]), np.max(P[:, 1])

    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    d = max(dx, dy)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    pad = pad_frac * d + 1e-6

    return (cx - 0.5 * d - pad, cx + 0.5 * d + pad), (cy - 0.5 * d - pad, cy + 0.5 * d + pad)


# ============================================================
# Plot
# ============================================================
def plot_multi_epsilon_geometry_with_tangency(
    *,
    y_t: np.ndarray,
    mu_t: np.ndarray,
    Sigma_t: np.ndarray,
    X_t: np.ndarray,
    epsilons: list[float],
    outpath: str,
    t: int,
) -> None:
    if y_t.shape != (2,) or mu_t.shape != (2,):
        raise ValueError("This plot expects n_y=2.")
    if Sigma_t.shape != (2, 2):
        raise ValueError("Sigma_t must be (2,2).")
    if X_t.shape[1] != 2:
        raise ValueError("X_t must be (n_x, 2).")

    _set_plot_theme()
    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    _style_axis(ax)

    Sigma_t = project_to_psd(Sigma_t)
    Sigma_inv = inv_psd(Sigma_t)

    # objective quadratic form
    M = project_to_psd(symmetrize(X_t.T @ X_t), eps=1e-10)

    # soft palette
    soft_colors = [
        "#7C8DA6",  # muted blue-gray
        "#9A8C98",  # mauve gray
        "#8FAE9D",  # muted green
        "#B39B7D",  # muted sand
        "#8C7C74",  # warm gray-brown
        "#6F8F8D",  # desaturated teal
    ]

    all_points = [y_t, mu_t]

    # base points (not added to automatic legend)
    ax.scatter(
        [mu_t[0]], [mu_t[1]],
        s=80, marker="o", color="#6C6F7D",
        edgecolor="black", linewidths=0.45, zorder=8
    )
    ax.scatter(
        [y_t[0]], [y_t[1]],
        s=95, marker="x", color="#222222",
        linewidths=2.0, zorder=9
    )

    ax.annotate(
        r"$\mu_t$",
        xy=mu_t,
        xytext=(7, 8),
        textcoords="offset points",
        color="#4A4A4A",
    )
    ax.annotate(
        r"$y_t$",
        xy=y_t,
        xytext=(7, -14),
        textcoords="offset points",
        color="#2A2A2A",
    )

    # store handles for epsilon legend
    epsilon_handles = []

    # draw each epsilon
    for i, eps in enumerate(epsilons):
        color = soft_colors[i % len(soft_colors)]

        y_star, obj_star = solve_kkt_max_quadratic_over_ellipsoid(
            X=X_t, y_t=y_t, mu=mu_t, Sigma=Sigma_t, epsilon=eps
        )

        # constraint ellipse centered at mu_t
        pts_constraint = _ellipse_points_from_quad(mu_t, Sigma_inv, eps)
        all_points.extend([pts_constraint, y_star])

        ax.plot(
            pts_constraint[:, 0], pts_constraint[:, 1],
            color=color, linewidth=2.0, alpha=0.95, zorder=1
        )

        # objective level-set centered at y_t that passes through y_star
        pts_obj = _ellipse_points_from_quad(y_t, M, max(obj_star, 1e-12))
        all_points.append(pts_obj)

        ax.plot(
            pts_obj[:, 0], pts_obj[:, 1],
            color=color, linewidth=1.5, linestyle="--", alpha=0.90, zorder=2
        )

        # tangent point
        ax.scatter(
            [y_star[0]], [y_star[1]],
            s=68, color=color, edgecolor="black", linewidths=0.45, zorder=10
        )

        # attack vector: y_t -> y_star
        ax.annotate(
            "",
            xy=y_star,
            xytext=y_t,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.9,
                color=color,
                mutation_scale=14,
                alpha=0.95,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=6,
        )

        # optional dotted segment underneath for extra visibility
        ax.plot(
            [y_t[0], y_star[0]],
            [y_t[1], y_star[1]],
            linestyle=":",
            linewidth=1.2,
            color=color,
            alpha=0.75,
            zorder=5,
        )

        # tangent point label
        ax.annotate(
            fr"$y^\star_{{{i+1}}}$",
            xy=y_star,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

        constr_val = float((y_star - mu_t).T @ Sigma_inv @ (y_star - mu_t))
        delta_adv = y_star - y_t
        print(
            f"epsilon={eps:.6f} | constraint={constr_val:.6f} | "
            f"obj={obj_star:.6f} | delta_adv={delta_adv}"
        )

        epsilon_handles.append(
            Line2D([0], [0], color=color, lw=2.2, label=fr"$\epsilon={eps:.3f}$")
        )

    xlim, ylim = _points_limits(all_points, pad_frac=0.14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y[0]")
    ax.set_ylabel("y[1]")
    ax.set_title(
        f"Constraint ellipses and tangent objective level-sets at t={t}",
        loc="left",
        fontweight="semibold",
    )

    # Legend 1: meaning of each visual element
    style_handles = [
        Line2D([0], [0], color="#7A7A7A", lw=2.0, linestyle="-",
               label="Constraint ellipse"),
        Line2D([0], [0], color="#7A7A7A", lw=1.5, linestyle="--",
               label="Objective level-set"),
        Line2D([0], [0], color="#7A7A7A", lw=0, linestyle="None",
               marker="o", markersize=6,
               markerfacecolor="#6C6F7D", markeredgecolor="black",
               label=r"$\mu_t$"),
        Line2D([0], [0], color="#222222", lw=0, linestyle="None",
               marker="x", markersize=8, markeredgewidth=2.0,
               label=r"$y_t$"),
        Line2D([0], [0], color="#7A7A7A", lw=0, linestyle="None",
               marker="o", markersize=6,
               markerfacecolor="#BBBBBB", markeredgecolor="black",
               label=r"Tangent point $y^\star$"),
        Line2D([0], [0], color="#7A7A7A", lw=1.8, linestyle="-",
               label=r"Attack vector $y_t \rightarrow y_t^{adv}$"),
    ]

    legend_style = ax.legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        framealpha=0.96,
        fontsize=8.3,
        title="Meaning",
        title_fontsize=9,
        borderpad=0.35,
        labelspacing=0.28,
        handlelength=2.0,
        handletextpad=0.55,
    )

    # Legend 2: epsilon/color map
    legend_eps = ax.legend(
        handles=epsilon_handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        frameon=True,
        framealpha=0.96,
        fontsize=8.3,
        title=r"$\epsilon$ values",
        title_fontsize=9,
        borderpad=0.35,
        labelspacing=0.22,
        handlelength=1.8,
        handletextpad=0.45,
        ncol=1,
    )

    ax.add_artist(legend_style)

    fig.tight_layout()

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(outpath, facecolor="white", dpi=300)
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    n_x = n_y = n_u = 2
    T = 25
    seed = 2026

    A0 = np.array([[0.65, 0.40],
                   [-0.15, 0.70]], dtype=float)

    B0 = np.array([[1.65, 1.40],
                   [-0.15, 0.70]], dtype=float)

    H0 = np.array([[0.65, 0.40],
                   [-0.15, 0.370]], dtype=float)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.3 * np.array([[1.6, -1.40],
                         [0.15, 0.70]], dtype=float)

    R0 = 0.2 * np.array([[0.65, 0.40],
                         [-0.15, 1.70]], dtype=float)

    Q0 = project_to_psd(Q0)
    R0 = project_to_psd(R0)

    dA = np.zeros_like(A0)
    dB = np.zeros_like(B0)
    dH = np.zeros_like(H0)
    dD = np.zeros_like(D0)
    dQ = np.zeros_like(Q0)
    dR = np.zeros_like(R0)

    x0 = np.array([0.5, 0.5], dtype=float)
    m0 = x0.copy()
    P0 = 0.05 * np.eye(n_x)

    x, y, u, mats = simulate_lgssm_nd(
        A0=A0, B0=B0, H0=H0, D0=D0,
        T=T, seed=seed, x0=x0,
        Q0=Q0, R0=R0,
        dA=dA, dB=dB, dH=dH, dD=dD, dQ=dQ, dR=dR,
        u_low=-0.5, u_high=0.5,
    )

    t = 5

    X_t, mu_t, Sigma_t = loo_values_nd(
        t=t,
        y=y, u=u,
        A_t=mats["A_t"], B_t=mats["B_t"], H_t=mats["H_t"], D_t=mats["D_t"],
        Q_t=mats["Q_t"], R_t=mats["R_t"],
        P0=P0, m0=m0,
    )
    y_t = y[t].copy()

    epsilons = [1.0, 2.0, 3.84, 5.991, 9.210]  # varios niveles de confianza chi-cuadrado para 2 grados de libertad

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"multi_epsilon_tangent_geometry_t{t}_T{T}_seed{seed}.png")

    plot_multi_epsilon_geometry_with_tangency(
        y_t=y_t,
        mu_t=mu_t,
        Sigma_t=Sigma_t,
        X_t=X_t,
        epsilons=epsilons,
        outpath=outpath,
        t=t,
    )

    print(f"\nSaved figure to: {outpath}")


if __name__ == "__main__":
    main()