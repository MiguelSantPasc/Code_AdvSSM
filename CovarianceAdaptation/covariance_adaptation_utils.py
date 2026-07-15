#!/usr/bin/env python3
"""
covariance_adaptation_utils.py

Shared covariance-adaptation helpers for the scripts that remain in
`CovarianceAdaptation`.

Why this file exists:
1. `comparison.py` originally mixed reusable utilities with one specific
   experiment.
2. The remaining scripts only need the reusable pieces: plotting helpers,
   stable SPD linear algebra, KF-only attack construction, and the online
   covariance-adaptation filter.
3. Moving those pieces here keeps the folder tidier while letting us delete
   older one-off experiment scripts without breaking imports.

What this module intentionally contains:
- the deterministic 2D reference setup used by the lambda sweep,
- shared Matplotlib styling,
- numerically stable Gaussian / SPD helpers,
- construction of the single-time KF attack,
- the online Bayesian covariance-adaptation KF defense.

What this module intentionally does not contain:
- figure-generation entry points,
- Monte Carlo experiment drivers tied to one specific script,
- nonlinear `g` experiment logic,
- RL experiment logic.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from AdvSSM.KKTOpt import (
        kalman_filter_nd,
        project_to_psd,
        solve_kkt_max_quadratic_over_ellipsoid,
    )
except ModuleNotFoundError:
    from KKTOpt import (
        kalman_filter_nd,
        project_to_psd,
        solve_kkt_max_quadratic_over_ellipsoid,
    )


# ============================================================
# Reference setup and plotting helpers
# ============================================================
def build_reference_setup() -> dict[str, np.ndarray | float | int]:
    """
    Return the deterministic 2D LGSSM used as the shared reference case.

    Reusing the same matrices across covariance-adaptation scripts keeps the
    lambda-sweep figures directly comparable.
    """
    n_x = n_y = n_u = 2

    A0 = np.array([[0.65, 0.40], [-0.15, 0.70]], dtype=float)
    B0 = np.array([[1.65, 1.40], [-0.15, 0.70]], dtype=float)
    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.3 * np.array([[1.6, -0.40], [-1.15, 0.70]], dtype=float)
    R0 = 0.42 * np.array([[0.65, 0.40], [-0.15, 1.70]], dtype=float)

    return {
        "A0": A0,
        "B0": B0,
        "H0": H0,
        "D0": D0,
        "Q0": project_to_psd(Q0),
        "R0": project_to_psd(R0),
        "dA": np.zeros_like(A0),
        "dB": np.zeros_like(B0),
        "dH": np.zeros_like(H0),
        "dD": np.zeros_like(D0),
        "dQ": np.zeros((n_x, n_x), dtype=float),
        "dR": np.zeros((n_y, n_y), dtype=float),
        "x0": np.array([0.5, 0.5], dtype=float),
        "P0": 0.05 * np.eye(n_x),
    }


def set_plot_theme() -> None:
    """Apply the shared pastel plotting theme used across kept scripts."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10.1,
            "axes.labelsize": 10.6,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 9.1,
            "ytick.labelsize": 9.1,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.26,
            "grid.linewidth": 0.78,
            "grid.linestyle": "-",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    """Apply the shared axis styling without adding subplot titles."""
    ax.set_facecolor("#FCFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.60)
    ax.spines["bottom"].set_alpha(0.60)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.26, linewidth=0.78)
    ax.grid(True, which="minor", alpha=0.11, linewidth=0.52)
    ax.set_axisbelow(True)


# ============================================================
# Stable SPD linear algebra helpers
# ============================================================
def stabilized_cholesky(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Return a Cholesky factor of an SPD approximation of `M`.

    A small diagonal jitter is increased only when needed.
    """
    M = project_to_psd(np.asarray(M, dtype=float), eps=eps)
    eye = np.eye(M.shape[0], dtype=float)
    jitter = eps

    for _ in range(8):
        try:
            return np.linalg.cholesky(M + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    raise np.linalg.LinAlgError("Failed to compute a stable Cholesky factor.")


def solve_spd(M: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Solve `M X = B` for SPD `M` using Cholesky factorization.

    This avoids direct inversion and supports vector or matrix right-hand
    sides.
    """
    chol = stabilized_cholesky(M, eps=eps)
    y = np.linalg.solve(chol, np.asarray(B, dtype=float))
    return np.linalg.solve(chol.T, y)


def spd_inverse(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return the inverse of an SPD matrix via solves against the identity."""
    I = np.eye(np.asarray(M).shape[0], dtype=float)
    return project_to_psd(solve_spd(M, I, eps=eps), eps=eps)


def gaussian_logpdf(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Evaluate the multivariate Gaussian log-density with Cholesky solves.

    The covariance is projected to PSD and stabilized before factorization.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = project_to_psd(np.asarray(cov, dtype=float), eps=eps)

    chol = stabilized_cholesky(cov, eps=eps)
    diff = x - mean
    whitened = np.linalg.solve(chol, diff)
    quad = float(np.dot(whitened, whitened))
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    dim = x.size
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad)


def quad_form_spd(M: np.ndarray, x: np.ndarray, eps: float = 1e-10) -> float:
    """Return `x^T M^{-1} x` for SPD `M` using a linear solve."""
    x = np.asarray(x, dtype=float).reshape(-1)
    solved = solve_spd(M, x, eps=eps)
    return float(np.dot(x, solved))


def log_mix_posterior_weight(
    pi: float,
    log_p0: float,
    log_p1: float,
    eps: float = 1e-12,
) -> float:
    """
    Return the stable posterior contamination probability in log-space.

    This computes:
        gamma = pi p1 / ((1 - pi) p0 + pi p1)
    """
    pi = float(np.clip(pi, eps, 1.0 - eps))
    log_num = np.log(pi) + log_p1
    log_den = np.logaddexp(np.log1p(-pi) + log_p0, log_num)
    return float(np.exp(log_num - log_den))


def safe_unit_direction(v: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, float]:
    """Return a safe unit direction and its norm for `v`."""
    v = np.asarray(v, dtype=float).reshape(-1)
    norm_v = float(np.linalg.norm(v))
    if norm_v < eps:
        return np.zeros_like(v), norm_v
    return v / norm_v, norm_v


def rank_one_covariance_update(
    base: np.ndarray,
    lam: float,
    u_vec: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """Add `lam * weight * u u^T` to a covariance and reproject to PSD."""
    base = np.asarray(base, dtype=float)
    u_vec = np.asarray(u_vec, dtype=float).reshape(-1)
    if np.linalg.norm(u_vec) < 1e-12 or lam <= 0.0 or weight <= 0.0:
        return project_to_psd(base)
    return project_to_psd(base + (lam * weight) * np.outer(u_vec, u_vec))


# ============================================================
# KF-only attack construction
# ============================================================
def compute_filter_predictive_quantities(
    *,
    y: np.ndarray,
    u: np.ndarray,
    A_t: np.ndarray,
    B_t: np.ndarray,
    H_t: np.ndarray,
    D_t: np.ndarray,
    Q_t: np.ndarray,
    R_t: np.ndarray,
    attack_t: int,
    m0: np.ndarray,
    P0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the KF-only attack ingredients at the attacked time.

    The returned tuple is:
    - `X_t`: the Kalman gain used by the local attack geometry,
    - `mu_t`: the predictive observation mean,
    - `Sigma_t`: the predictive observation covariance.
    """
    _, _, m_pred, P_pred = kalman_filter_nd(
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
    )

    T = y.shape[0] - 1

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

    Hk = H_t[attack_t]
    Dk = D_t[attack_t]
    Rk = project_to_psd(R_t[attack_t])
    Pk_pred = np.asarray(P_pred[attack_t], dtype=float)

    mu_pred = Hk @ np.asarray(m_pred[attack_t], dtype=float) + Dk @ u_at(attack_t)
    Sigma_pred = project_to_psd(Hk @ Pk_pred @ Hk.T + Rk)
    kalman_gain = solve_spd(Sigma_pred, Hk @ Pk_pred.T).T
    return kalman_gain, mu_pred, Sigma_pred


def build_kf_attack(
    *,
    y_clean: np.ndarray,
    u_controls: np.ndarray,
    mats: dict[str, np.ndarray],
    attack_t: int,
    m0: np.ndarray,
    P0: np.ndarray,
    epsilon: float,
) -> dict[str, np.ndarray | float]:
    """Build the single-time adversarial observation using KF-only geometry."""
    X_t, mu_t, Sigma_t = compute_filter_predictive_quantities(
        y=y_clean,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        attack_t=attack_t,
        m0=m0,
        P0=P0,
    )

    y_adv = y_clean.copy()
    y_adv[attack_t], obj_star = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_t,
        y_t=y_clean[attack_t],
        mu=mu_t,
        Sigma=Sigma_t,
        epsilon=epsilon,
    )

    return {
        "X_t": X_t,
        "mu_t": mu_t,
        "Sigma_t": Sigma_t,
        "y_adv": y_adv,
        "adv_target": np.asarray(y_adv[attack_t], dtype=float),
        "obj_star": float(obj_star),
    }


# ============================================================
# Online Bayesian covariance adaptation for KF
# ============================================================
def compute_contamination_prior(
    *,
    delta_adv: np.ndarray,
    S_t: np.ndarray,
    P_pred_t: np.ndarray,
    H_t: np.ndarray,
    omega_h: float,
    omega_o: float,
) -> tuple[float, float, float]:
    """
    Compute the prior contamination probability and its two component scores.

    `r_t^(o)` measures closeness in observation space.
    `r_t^(h)` measures hidden-state risk after mapping through the nominal
    Kalman gain.
    """
    delta_adv = np.asarray(delta_adv, dtype=float).reshape(-1)
    S_t = project_to_psd(S_t)
    P_pred_t = project_to_psd(P_pred_t)

    obs_mahal_sq = quad_form_spd(S_t, delta_adv)
    r_obs = float(np.exp(-0.5 * obs_mahal_sq))

    K_nom = solve_spd(S_t, H_t @ P_pred_t.T).T
    delta_state = K_nom @ delta_adv
    state_mahal_sq = quad_form_spd(P_pred_t, delta_state)
    r_hidden = float(1.0 - np.exp(-0.5 * state_mahal_sq))

    pi_t = float(np.clip(omega_h * r_hidden + omega_o * r_obs, 0.0, 1.0))
    return pi_t, r_hidden, r_obs


def kalman_filter_with_online_covariance_adaptation(
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
    attack_targets: dict[int, np.ndarray] | None,
    lam: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    direction_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Run KF with the online Bayesian covariance-adaptation defense.

    The defense adapts only along the attack direction and only at times for
    which an adversarial target is provided in `attack_targets`.
    """
    T = y.shape[0] - 1
    n_x = P0.shape[0]
    n_y = y.shape[1]
    I_x = np.eye(n_x, dtype=float)

    def u_at(k: int) -> np.ndarray:
        return u[k] if k < T else u[T - 1]

    m_pred = np.zeros((T + 1, n_x), dtype=float)
    P_pred = np.zeros((T + 1, n_x, n_x), dtype=float)
    m_filt = np.zeros((T + 1, n_x), dtype=float)
    P_filt = np.zeros((T + 1, n_x, n_x), dtype=float)

    diagnostics = {
        "pi_t": np.zeros(T + 1, dtype=float),
        "gamma_t": np.zeros(T + 1, dtype=float),
        "bar_gamma_t": np.zeros(T + 1, dtype=float),
        "r_hidden_t": np.zeros(T + 1, dtype=float),
        "r_obs_t": np.zeros(T + 1, dtype=float),
        "u_t": np.zeros((T + 1, n_y), dtype=float),
        "V_tilde_t": np.zeros((T + 1, n_y, n_y), dtype=float),
        "K_tilde_t": np.zeros((T + 1, n_x, n_y), dtype=float),
        "mu_poe_t": np.zeros((T + 1, n_y), dtype=float),
        "Sigma_poe_t": np.zeros((T + 1, n_y, n_y), dtype=float),
    }

    m_pred[0] = np.asarray(m0, dtype=float)
    P_pred[0] = project_to_psd(np.asarray(P0, dtype=float))

    for k in range(T + 1):
        Hk = np.asarray(H_t[k], dtype=float)
        Dk = np.asarray(D_t[k], dtype=float)
        Rk = project_to_psd(np.asarray(R_t[k], dtype=float))
        uk = u_at(k)

        y_hat = Hk @ m_pred[k] + Dk @ uk
        S_nom = project_to_psd(Hk @ P_pred[k] @ Hk.T + Rk)
        innov = y[k] - y_hat

        V_tilde = Rk.copy()
        S_tilde = S_nom.copy()
        K_tilde = solve_spd(S_tilde, Hk @ P_pred[k].T).T
        mu_poe = y_hat.copy()
        Sigma_poe = S_nom.copy()
        pi_t = 0.0
        gamma_t = 0.0
        bar_gamma_t = 0.0
        r_hidden = 0.0
        r_obs = 0.0
        u_dir = np.zeros(n_y, dtype=float)

        if attack_targets is not None and k in attack_targets:
            adv_target = np.asarray(attack_targets[k], dtype=float).reshape(n_y)
            delta_adv = adv_target - y_hat
            u_dir, delta_norm = safe_unit_direction(delta_adv, eps=direction_eps)

            if delta_norm >= direction_eps:
                pi_t, r_hidden, r_obs = compute_contamination_prior(
                    delta_adv=delta_adv,
                    S_t=S_nom,
                    P_pred_t=P_pred[k],
                    H_t=Hk,
                    omega_h=omega_h,
                    omega_o=omega_o,
                )

                S_adv = rank_one_covariance_update(S_nom, lam, u_dir)
                precision_poe = spd_inverse(S_adv) + spd_inverse(Rk)
                Sigma_poe = spd_inverse(precision_poe)
                rhs_poe = solve_spd(S_adv, y_hat) + solve_spd(Rk, adv_target)
                mu_poe = solve_spd(precision_poe, rhs_poe)

                log_p0 = gaussian_logpdf(y[k], y_hat, S_nom)
                log_p1 = gaussian_logpdf(y[k], mu_poe, Sigma_poe)
                gamma_t = log_mix_posterior_weight(pi_t, log_p0, log_p1)
                bar_gamma_t = gamma_t if gamma_t >= delta_threshold else 0.0

                V_tilde = rank_one_covariance_update(Rk, lam, u_dir, weight=bar_gamma_t)
                S_tilde = rank_one_covariance_update(S_nom, lam, u_dir, weight=bar_gamma_t)
                K_tilde = solve_spd(S_tilde, Hk @ P_pred[k].T).T

        m_filt[k] = m_pred[k] + K_tilde @ innov

        # Joseph form keeps the posterior covariance PSD after online
        # adaptation of the observation covariance.
        joseph_left = I_x - K_tilde @ Hk
        P_filt[k] = project_to_psd(
            joseph_left @ P_pred[k] @ joseph_left.T + K_tilde @ V_tilde @ K_tilde.T
        )

        diagnostics["pi_t"][k] = pi_t
        diagnostics["gamma_t"][k] = gamma_t
        diagnostics["bar_gamma_t"][k] = bar_gamma_t
        diagnostics["r_hidden_t"][k] = r_hidden
        diagnostics["r_obs_t"][k] = r_obs
        diagnostics["u_t"][k] = u_dir
        diagnostics["V_tilde_t"][k] = V_tilde
        diagnostics["K_tilde_t"][k] = K_tilde
        diagnostics["mu_poe_t"][k] = mu_poe
        diagnostics["Sigma_poe_t"][k] = Sigma_poe

        if k < T:
            Ak = np.asarray(A_t[k], dtype=float)
            Bk = np.asarray(B_t[k], dtype=float)
            Qk = project_to_psd(np.asarray(Q_t[k], dtype=float))
            m_pred[k + 1] = Ak @ m_filt[k] + Bk @ u_at(k + 1)
            P_pred[k + 1] = project_to_psd(Ak @ P_filt[k] @ Ak.T + Qk)

    return m_filt, P_filt, m_pred, P_pred, diagnostics
