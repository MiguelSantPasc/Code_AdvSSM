#!/usr/bin/env python3
"""
comparison.py

Compact covariance-adaptation comparison for the KF-only attack pipeline.

This script mirrors the role of `AdvSSM/comparison.py`, but now the attacker
and the estimator both stay in the filtering setting:

1. The adversarial observation at a fixed time `t` is built from the KF
   predictive observation law, without RTS smoothing.
2. The attacked trajectory is then estimated in two ways:
   - standard Kalman filtering, and
   - Kalman filtering with an online Bayesian covariance-adaptation defense.
3. A single combined figure is produced with three panels:
   - left: first hidden-state dimension over time,
   - middle: Monte Carlo local effect,
   - right: Monte Carlo global effect.

Implementation notes for the defense:
- The contamination posterior uses log-likelihoods for numerical stability.
- SPD linear algebra uses Cholesky factorization and linear solves instead of
  direct matrix inversion.
- The attack direction `u_t` is derived from the actual KF adversarial target
  via `(o_t^adv - o_hat_t) / ||o_t^adv - o_hat_t||`.
- If the target is nearly identical to the predictive mean, the code falls
  back to the nominal update and records zero adaptation.

Assumption used here:
- The defense is activated at the attacked time where the adversarial target
  is known from the constructed KF attack. All other times use the nominal
  Kalman update.
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
        simulate_lgssm_nd,
        solve_kkt_max_quadratic_over_ellipsoid,
    )
    from AdvSSM.KKTOpt_tdependent import sample_random_ssm_run_params
    from AdvSSM.io_utils import cached_npz, data_path_for_plot, figures_dir_for
except ModuleNotFoundError:
    from KKTOpt import (
        kalman_filter_nd,
        project_to_psd,
        simulate_lgssm_nd,
        solve_kkt_max_quadratic_over_ellipsoid,
    )
    from KKTOpt_tdependent import sample_random_ssm_run_params
    from io_utils import cached_npz, data_path_for_plot, figures_dir_for


# ============================================================
# Reference setup and plotting helpers
# ============================================================
def build_reference_setup() -> dict[str, np.ndarray | float | int]:
    """
    Return the deterministic 2D LGSSM used as the reference comparison case.

    Reusing the same matrices as the AdvSSM comparison keeps the new
    covariance-adaptation figure directly comparable to the rest of the repo.
    """
    n_x = n_y = n_u = 2

    A0 = np.array([[0.65, 0.40],
                   [-0.15, 0.70]], dtype=float)
    B0 = np.array([[1.65, 1.40],
                   [-0.15, 0.70]], dtype=float)
    H0 = np.eye(n_y, n_x)
    D0 = np.zeros((n_y, n_u), dtype=float)

    Q0 = 0.3 * np.array([[1.6, -0.40],
                         [-1.15, 0.70]], dtype=float)
    R0 = 0.42 * np.array([[0.65, 0.40],
                          [-0.15, 1.70]], dtype=float)

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
    """Apply a compact pastel plotting theme shared by the three panels."""
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
    """Apply the shared axis styling without adding any subplot title."""
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

    A small diagonal jitter is increased progressively only if needed.
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

    This avoids direct inversion and works for both vector and matrix right
    hand sides.
    """
    chol = stabilized_cholesky(M, eps=eps)
    y = np.linalg.solve(chol, np.asarray(B, dtype=float))
    return np.linalg.solve(chol.T, y)


def spd_inverse(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return the inverse of an SPD matrix via solves against the identity."""
    I = np.eye(np.asarray(M).shape[0], dtype=float)
    return project_to_psd(solve_spd(M, I, eps=eps), eps=eps)


def gaussian_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, eps: float = 1e-10) -> float:
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
    """Return `x^T M^{-1} x` for SPD `M` using a solve."""
    x = np.asarray(x, dtype=float).reshape(-1)
    solved = solve_spd(M, x, eps=eps)
    return float(np.dot(x, solved))


def log_mix_posterior_weight(pi: float, log_p0: float, log_p1: float, eps: float = 1e-12) -> float:
    """
    Return the stable posterior contamination probability.

    This computes:
      gamma = pi p1 / ((1-pi) p0 + pi p1)
    in log-space.
    """
    pi = float(np.clip(pi, eps, 1.0 - eps))
    log_num = np.log(pi) + log_p1
    log_den = np.logaddexp(np.log1p(-pi) + log_p0, log_num)
    return float(np.exp(log_num - log_den))


def safe_unit_direction(v: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, float]:
    """Return the unit direction and norm, handling near-zero vectors safely."""
    v = np.asarray(v, dtype=float).reshape(-1)
    norm_v = float(np.linalg.norm(v))
    if norm_v < eps:
        return np.zeros_like(v), norm_v
    return v / norm_v, norm_v


def rank_one_covariance_update(base: np.ndarray, lam: float, u_vec: np.ndarray, weight: float = 1.0) -> np.ndarray:
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

    The attack geometry matches the filter-only branch used in the AdvSSM
    comparison:
    - `X_t` is the Kalman gain at time `t`,
    - `mu_t` is the KF predictive observation mean,
    - `Sigma_t` is the KF predictive observation covariance.
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
    """
    Build the single-time adversarial observation using the KF-only geometry.
    """
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

    `r_t^(o)` is a predictive closeness score to the adversarial target in
    observation space.

    `r_t^(h)` is a hidden-state risk score induced by the same target after the
    nominal Kalman gain maps the observation-space deviation into state space.
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

        # Joseph-form covariance update keeps the covariance PSD after the
        # observation covariance has been adapted online.
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


# ============================================================
# Reference run and Monte Carlo evaluation
# ============================================================
def evaluate_reference_scenario(
    *,
    T: int,
    attack_t: int,
    seed: int,
    epsilon: float,
    lam: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
) -> dict[str, np.ndarray | float | int]:
    """
    Simulate one deterministic reference trajectory and compare three filters:
    clean KF, attacked KF, and attacked KF with covariance adaptation.
    """
    setup = build_reference_setup()
    m0 = np.asarray(setup["x0"], dtype=float).copy()
    P0 = np.asarray(setup["P0"], dtype=float).copy()

    x_true, y_clean, u_controls, mats = simulate_lgssm_nd(
        A0=np.asarray(setup["A0"], dtype=float),
        B0=np.asarray(setup["B0"], dtype=float),
        H0=np.asarray(setup["H0"], dtype=float),
        D0=np.asarray(setup["D0"], dtype=float),
        T=T,
        seed=seed,
        x0=np.asarray(setup["x0"], dtype=float),
        Q0=np.asarray(setup["Q0"], dtype=float),
        R0=np.asarray(setup["R0"], dtype=float),
        dA=np.asarray(setup["dA"], dtype=float),
        dB=np.asarray(setup["dB"], dtype=float),
        dH=np.asarray(setup["dH"], dtype=float),
        dD=np.asarray(setup["dD"], dtype=float),
        dQ=np.asarray(setup["dQ"], dtype=float),
        dR=np.asarray(setup["dR"], dtype=float),
        u_low=-0.5,
        u_high=0.5,
    )

    attack_data = build_kf_attack(
        y_clean=y_clean,
        u_controls=u_controls,
        mats=mats,
        attack_t=attack_t,
        m0=m0,
        P0=P0,
        epsilon=epsilon,
    )
    y_adv = np.asarray(attack_data["y_adv"], dtype=float)

    clean_m, clean_P, _, _ = kalman_filter_nd(
        y=y_clean,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=m0,
        P0=P0,
    )
    attack_m, attack_P, _, _ = kalman_filter_nd(
        y=y_adv,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=m0,
        P0=P0,
    )
    adapt_m, adapt_P, _, _, adapt_diag = kalman_filter_with_online_covariance_adaptation(
        y=y_adv,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=m0,
        P0=P0,
        attack_targets={attack_t: np.asarray(attack_data["adv_target"], dtype=float)},
        lam=lam,
        omega_h=omega_h,
        omega_o=omega_o,
        delta_threshold=delta_threshold,
    )

    return {
        "T": T,
        "attack_t": attack_t,
        "seed": seed,
        "epsilon": epsilon,
        "lambda": lam,
        "omega_h": omega_h,
        "omega_o": omega_o,
        "delta_threshold": delta_threshold,
        "x_true": x_true,
        "y_clean": y_clean,
        "y_adv": y_adv,
        "clean_m": clean_m,
        "clean_P": clean_P,
        "attack_m": attack_m,
        "attack_P": attack_P,
        "adapt_m": adapt_m,
        "adapt_P": adapt_P,
        "attack_obj_star": float(attack_data["obj_star"]),
        "attack_mu_t": np.asarray(attack_data["mu_t"], dtype=float),
        "attack_Sigma_t": np.asarray(attack_data["Sigma_t"], dtype=float),
        "pi_t": np.asarray(adapt_diag["pi_t"], dtype=float),
        "gamma_t": np.asarray(adapt_diag["gamma_t"], dtype=float),
        "bar_gamma_t": np.asarray(adapt_diag["bar_gamma_t"], dtype=float),
        "u_t": np.asarray(adapt_diag["u_t"], dtype=float),
        "V_tilde_t": np.asarray(adapt_diag["V_tilde_t"], dtype=float),
        "K_tilde_t": np.asarray(adapt_diag["K_tilde_t"], dtype=float),
    }


def evaluate_single_monte_carlo_run(
    *,
    run_seed: int,
    T: int,
    attack_t: int,
    epsilon: float,
    lam: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    var_entries: float = 8.0,
) -> dict[str, float]:
    """
    Evaluate one random SSM under attacked KF and attacked KF + adaptation.

    The local and global effects follow the same definition as in
    `AdvSSM/comparison.py`: absolute hidden-state error at `t`, and accumulated
    absolute hidden-state error over the full trajectory.
    """
    rng = np.random.default_rng(run_seed)
    std = float(np.sqrt(var_entries))
    params = sample_random_ssm_run_params(rng, n_x=2, n_y=2, n_u=2, std=std)

    zero_drifts = {
        "dA": np.zeros_like(params["A0"]),
        "dB": np.zeros_like(params["B0"]),
        "dH": np.zeros_like(params["H0"]),
        "dD": np.zeros_like(params["D0"]),
        "dQ": np.zeros_like(params["Q0"]),
        "dR": np.zeros_like(params["R0"]),
    }

    x_true, y_clean, u_controls, mats = simulate_lgssm_nd(
        A0=params["A0"],
        B0=params["B0"],
        H0=params["H0"],
        D0=params["D0"],
        T=T,
        seed=run_seed,
        x0=params["x0"],
        Q0=params["Q0"],
        R0=params["R0"],
        dA=zero_drifts["dA"],
        dB=zero_drifts["dB"],
        dH=zero_drifts["dH"],
        dD=zero_drifts["dD"],
        dQ=zero_drifts["dQ"],
        dR=zero_drifts["dR"],
        u_low=-0.5,
        u_high=0.5,
    )

    attack_data = build_kf_attack(
        y_clean=y_clean,
        u_controls=u_controls,
        mats=mats,
        attack_t=attack_t,
        m0=params["m0"],
        P0=params["P0"],
        epsilon=epsilon,
    )
    y_adv = np.asarray(attack_data["y_adv"], dtype=float)

    attack_m, _, _, _ = kalman_filter_nd(
        y=y_adv,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=params["m0"],
        P0=params["P0"],
    )
    adapt_m, _, _, _, _ = kalman_filter_with_online_covariance_adaptation(
        y=y_adv,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=params["m0"],
        P0=params["P0"],
        attack_targets={attack_t: np.asarray(attack_data["adv_target"], dtype=float)},
        lam=lam,
        omega_h=omega_h,
        omega_o=omega_o,
        delta_threshold=delta_threshold,
    )

    local_attack = float(np.sum(np.abs(x_true[attack_t] - attack_m[attack_t])))
    global_attack = float(np.sum(np.abs(x_true - attack_m)))
    local_adapt = float(np.sum(np.abs(x_true[attack_t] - adapt_m[attack_t])))
    global_adapt = float(np.sum(np.abs(x_true - adapt_m)))

    return {
        "local_attack": local_attack,
        "global_attack": global_attack,
        "local_adapt": local_adapt,
        "global_adapt": global_adapt,
    }


def run_monte_carlo_effect_comparison(
    *,
    N_runs: int,
    T: int,
    attack_t: int,
    epsilon: float,
    lam: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    base_seed: int,
) -> dict[str, np.ndarray | float | int]:
    """
    Run the compact Monte Carlo comparison between attacked KF and defended KF.
    """
    local_attack = np.full(N_runs, np.nan, dtype=float)
    global_attack = np.full(N_runs, np.nan, dtype=float)
    local_adapt = np.full(N_runs, np.nan, dtype=float)
    global_adapt = np.full(N_runs, np.nan, dtype=float)

    for run_idx in range(N_runs):
        run_seed = base_seed + 1000 * run_idx
        print(f"[MC] run {run_idx + 1}/{N_runs} (seed={run_seed})")
        try:
            effects = evaluate_single_monte_carlo_run(
                run_seed=run_seed,
                T=T,
                attack_t=attack_t,
                epsilon=epsilon,
                lam=lam,
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
            )
            local_attack[run_idx] = effects["local_attack"]
            global_attack[run_idx] = effects["global_attack"]
            local_adapt[run_idx] = effects["local_adapt"]
            global_adapt[run_idx] = effects["global_adapt"]
        except Exception as exc:
            print(f"[WARN seed={run_seed}] {type(exc).__name__}: {exc}")

    return {
        "N_runs": N_runs,
        "T": T,
        "attack_t": attack_t,
        "epsilon": epsilon,
        "lambda": lam,
        "omega_h": omega_h,
        "omega_o": omega_o,
        "delta_threshold": delta_threshold,
        "base_seed": base_seed,
        "local_attack": local_attack,
        "global_attack": global_attack,
        "local_adapt": local_adapt,
        "global_adapt": global_adapt,
    }


# ============================================================
# Figure drawing
# ============================================================
def draw_effect_boxplot(
    *,
    ax: plt.Axes,
    attack_values: np.ndarray,
    adapt_values: np.ndarray,
    ylabel: str,
) -> None:
    """Draw one Monte Carlo effect panel with pastel colors and inside legend."""
    style_axis(ax)

    attack_color = "#F3B3AA"
    adapt_color = "#B7D8A8"
    mean_color = "#303030"

    finite_attack = np.asarray(attack_values, dtype=float)[np.isfinite(attack_values)]
    finite_adapt = np.asarray(adapt_values, dtype=float)[np.isfinite(adapt_values)]

    box = ax.boxplot(
        [finite_attack, finite_adapt],
        positions=[1.0, 1.7],
        widths=0.26,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#3A3A3A", "linewidth": 1.25},
        whiskerprops={"color": "#6B6B6B", "linewidth": 1.0},
        capprops={"color": "#6B6B6B", "linewidth": 1.0},
    )

    for patch, face in zip(box["boxes"], [attack_color, adapt_color], strict=True):
        patch.set_facecolor(face)
        patch.set_edgecolor(face)
        patch.set_alpha(0.60)
        patch.set_linewidth(1.15)

    rng = np.random.default_rng(2026)
    for xpos, values, color in [(1.0, finite_attack, attack_color), (1.7, finite_adapt, adapt_color)]:
        jitter = rng.uniform(-0.045, 0.045, size=values.size)
        ax.scatter(
            xpos + jitter,
            values,
            s=13,
            color=color,
            edgecolors="none",
            alpha=0.56,
            zorder=3,
        )
        ax.scatter(
            [xpos],
            [float(np.mean(values))],
            s=54,
            marker="D",
            color=mean_color,
            zorder=4,
            label="Mean" if xpos == 1.0 else None,
        )

    ax.set_xlim(0.72, 1.98)
    ax.set_xticks([1.0, 1.7])
    ax.set_xticklabels(["KF attack", "KF + cov-adapt"])
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, borderpad=0.35)


def plot_combined_comparison_figure(
    *,
    x_true: np.ndarray,
    clean_m: np.ndarray,
    clean_P: np.ndarray,
    attack_m: np.ndarray,
    attack_P: np.ndarray,
    adapt_m: np.ndarray,
    adapt_P: np.ndarray,
    local_attack: np.ndarray,
    global_attack: np.ndarray,
    local_adapt: np.ndarray,
    global_adapt: np.ndarray,
    attack_t: int,
    outpath: str,
) -> None:
    """
    Plot the 3-panel comparison requested for covariance adaptation.

    The left axis shows the first hidden-state dimension with:
    - true state,
    - clean non-attack KF baseline,
    - attacked KF,
    - attacked KF with covariance adaptation.
    """
    set_plot_theme()

    tt = np.arange(x_true.shape[0])
    idx = 0
    z_value = 1.96

    true_color = "#6A6F77"
    clean_color = "#9BCBE7"
    attack_color = "#F0A79D"
    adapt_color = "#B9D9A9"
    attack_marker_color = "#D8BE74"

    fig = plt.figure(figsize=(13.6, 3.95), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[2.2, 0.86, 0.86], wspace=0.08)
    ax_state = fig.add_subplot(grid[0, 0])
    ax_local = fig.add_subplot(grid[0, 1])
    ax_global = fig.add_subplot(grid[0, 2], sharey=ax_local)

    style_axis(ax_state)

    clean_sd = np.sqrt(np.maximum(clean_P[:, idx, idx], 0.0))
    attack_sd = np.sqrt(np.maximum(attack_P[:, idx, idx], 0.0))
    adapt_sd = np.sqrt(np.maximum(adapt_P[:, idx, idx], 0.0))

    ax_state.axvspan(attack_t - 0.32, attack_t + 0.32, color="#F6E7B4", alpha=0.38, zorder=0)
    ax_state.axvline(attack_t, color=attack_marker_color, linewidth=1.08, zorder=1)

    ax_state.fill_between(
        tt,
        clean_m[:, idx] - z_value * clean_sd,
        clean_m[:, idx] + z_value * clean_sd,
        color=clean_color,
        alpha=0.18,
        zorder=1,
    )
    ax_state.plot(tt, clean_m[:, idx], color=clean_color, linewidth=1.85, label="Clean KF", zorder=3)

    ax_state.fill_between(
        tt,
        attack_m[:, idx] - z_value * attack_sd,
        attack_m[:, idx] + z_value * attack_sd,
        color=attack_color,
        alpha=0.18,
        zorder=1,
    )
    ax_state.plot(
        tt,
        attack_m[:, idx],
        color=attack_color,
        linewidth=1.85,
        linestyle="--",
        label="Attacked KF",
        zorder=3,
    )

    ax_state.fill_between(
        tt,
        adapt_m[:, idx] - z_value * adapt_sd,
        adapt_m[:, idx] + z_value * adapt_sd,
        color=adapt_color,
        alpha=0.20,
        zorder=1,
    )
    ax_state.plot(
        tt,
        adapt_m[:, idx],
        color=adapt_color,
        linewidth=1.9,
        label="KF + cov-adapt",
        zorder=4,
    )

    ax_state.plot(
        tt,
        x_true[:, idx],
        color=true_color,
        linewidth=1.35,
        marker="o",
        markersize=2.6,
        label=r"True $s_t^{(1)}$",
        zorder=2,
    )

    state_curves = [
        x_true[:, idx],
        clean_m[:, idx] - z_value * clean_sd,
        clean_m[:, idx] + z_value * clean_sd,
        attack_m[:, idx] - z_value * attack_sd,
        attack_m[:, idx] + z_value * attack_sd,
        adapt_m[:, idx] - z_value * adapt_sd,
        adapt_m[:, idx] + z_value * adapt_sd,
    ]
    state_min = float(np.min([np.min(curve) for curve in state_curves]))
    state_max = float(np.max([np.max(curve) for curve in state_curves]))
    state_pad = 0.06 * max(state_max - state_min, 1e-8)
    ax_state.set_ylim(state_min - state_pad, state_max + state_pad)

    ax_state.set_xlim(-0.15, x_true.shape[0] - 0.85)
    ax_state.set_xlabel("time t")
    ax_state.set_ylabel(r"$s_t^{(1)}$")
    ax_state.legend(loc="upper left", frameon=True, framealpha=0.95, borderpad=0.35)

    draw_effect_boxplot(
        ax=ax_local,
        attack_values=local_attack,
        adapt_values=local_adapt,
        ylabel="Local effect",
    )
    draw_effect_boxplot(
        ax=ax_global,
        attack_values=global_attack,
        adapt_values=global_adapt,
        ylabel="Global effect",
    )

    effect_arrays = [
        np.asarray(local_attack, dtype=float),
        np.asarray(global_attack, dtype=float),
        np.asarray(local_adapt, dtype=float),
        np.asarray(global_adapt, dtype=float),
    ]
    finite_effects = np.concatenate([arr[np.isfinite(arr)] for arr in effect_arrays])
    effect_min = float(np.min(finite_effects))
    effect_max = float(np.max(finite_effects))
    effect_pad = 0.06 * max(effect_max - effect_min, 1e-8)
    ax_local.set_ylim(effect_min - effect_pad, effect_max + effect_pad)
    ax_global.set_ylim(effect_min - effect_pad, effect_max + effect_pad)

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(outpath, facecolor="white", dpi=300)
    plt.close(fig)


# ============================================================
# Main entry point
# ============================================================
def main() -> None:
    """
    Generate the covariance-adaptation comparison figure and its caches.

    Environment variables are supported so the experiment can be rerun quickly
    with either tiny validation settings or richer Monte Carlo studies.
    """
    T = int(os.environ.get("COVADAPT_T", "10"))
    attack_t = int(os.environ.get("COVADAPT_ATTACK_T", "5"))
    seed = int(os.environ.get("COVADAPT_SEED", "2026"))
    epsilon = float(os.environ.get("COVADAPT_EPS", "5.991"))
    lam = float(os.environ.get("COVADAPT_LAMBDA", "18.0"))
    omega_h = float(os.environ.get("COVADAPT_OMEGA_H", "0.50"))
    omega_o = float(os.environ.get("COVADAPT_OMEGA_O", "0.50"))
    delta_threshold = float(os.environ.get("COVADAPT_DELTA", "0.20"))
    N_runs = int(os.environ.get("COVADAPT_MC_RUNS", "100"))
    mc_seed = int(os.environ.get("COVADAPT_MC_BASE_SEED", "2026"))
    force_reference = os.environ.get("COVADAPT_FORCE_REFERENCE", "0") == "1"
    force_mc = os.environ.get("COVADAPT_FORCE_MC", "0") == "1"

    if not (0 <= attack_t <= T):
        raise ValueError("attack_t must satisfy 0 <= attack_t <= T")
    if not np.isclose(omega_h + omega_o, 1.0, atol=1e-9):
        raise ValueError("omega_h and omega_o must sum to 1.")
    if not (0.0 <= delta_threshold <= 1.0):
        raise ValueError("delta_threshold must lie in [0, 1].")

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    outpath = os.path.join(
        out_dir,
        f"comparison_covadapt_t{attack_t}_T{T}_seed{seed}_N{N_runs}.png",
    )
    reference_data_path = data_path_for_plot(outpath.replace(".png", "_reference.png"))
    mc_data_path = data_path_for_plot(outpath.replace(".png", "_mc.png"))

    def compute_reference_data() -> dict[str, np.ndarray | float | int]:
        return evaluate_reference_scenario(
            T=T,
            attack_t=attack_t,
            seed=seed,
            epsilon=epsilon,
            lam=lam,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
        )

    def compute_mc_data() -> dict[str, np.ndarray | float | int]:
        return run_monte_carlo_effect_comparison(
            N_runs=N_runs,
            T=T,
            attack_t=attack_t,
            epsilon=epsilon,
            lam=lam,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            base_seed=mc_seed,
        )

    reference_data = cached_npz(reference_data_path, compute_reference_data, force=force_reference)
    mc_data = cached_npz(mc_data_path, compute_mc_data, force=force_mc)

    plot_combined_comparison_figure(
        x_true=np.asarray(reference_data["x_true"], dtype=float),
        clean_m=np.asarray(reference_data["clean_m"], dtype=float),
        clean_P=np.asarray(reference_data["clean_P"], dtype=float),
        attack_m=np.asarray(reference_data["attack_m"], dtype=float),
        attack_P=np.asarray(reference_data["attack_P"], dtype=float),
        adapt_m=np.asarray(reference_data["adapt_m"], dtype=float),
        adapt_P=np.asarray(reference_data["adapt_P"], dtype=float),
        local_attack=np.asarray(mc_data["local_attack"], dtype=float),
        global_attack=np.asarray(mc_data["global_attack"], dtype=float),
        local_adapt=np.asarray(mc_data["local_adapt"], dtype=float),
        global_adapt=np.asarray(mc_data["global_adapt"], dtype=float),
        attack_t=attack_t,
        outpath=outpath,
    )
    print(f"Saved figure to: {outpath}")


if __name__ == "__main__":
    main()
