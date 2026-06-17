#!/usr/bin/env python3
"""
comparison_g.py

Covariance-adaptation comparison for the 3D nonlinear `g`-attack example.

This script reuses the 3D nonlinear attack setup from
`AdvNonLinearAttack/AttackSense3D.py`, but replaces its original visualization
with a compact lambda-sweep comparison inside `CovarianceAdaptation`.

Design choices:
1. The attack is the same white-box point attack on the last observation
   `o_T`, targeting the posterior quantity `E[g(s_T) | o]`.
2. The ellipsoid size is fixed through a 95% 3D chi-square coverage, i.e.
   `epsilon = chi2_ppf(0.95; df = 3)`.
3. The defense modifies the attacked-time observation covariance only along
   the online adversarial direction, then runs the Kalman update and finally
   the RTS smoother so the downstream posterior remains aligned with the
   original nonlinear example.
4. The figure intentionally mirrors `AttackSense3D_CallSummary.py`, but the
   varying parameter is now `lambda` instead of `epsilon`:
   - left panel: clean, attacked, and defended call-probability curves,
   - right panel: false-positive and false-negative rates versus `lambda`.

Plotting conventions followed here:
- hidden states are written as `s_t`,
- no subplot titles are used,
- legends stay inside each axis,
- pastel colors are used by default,
- comments explain the nontrivial computational steps.
"""

from __future__ import annotations

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import chi2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
NONLINEAR_DIR = os.path.join(REPO_ROOT, "AdvNonLinearAttack")

for import_path in (REPO_ROOT, NONLINEAR_DIR):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

try:
    from AdvNonLinearAttack.AttackSense3D import (
        estimate_E_g,
        g_scalar,
        g_scalar_grad,
        get_system_parameters,
        kalman_filter_nd,
        project_to_psd,
        rts_smoother_nd,
        simulate_lgssm_nd,
        white_box_point_attack_nd,
    )
    from AdvSSM.io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from CovarianceAdaptation.comparison import (
        compute_contamination_prior,
        gaussian_logpdf,
        log_mix_posterior_weight,
        rank_one_covariance_update,
        safe_unit_direction,
        set_plot_theme,
        solve_spd,
        spd_inverse,
        style_axis,
    )
except ModuleNotFoundError:
    from AttackSense3D import (
        estimate_E_g,
        g_scalar,
        g_scalar_grad,
        get_system_parameters,
        kalman_filter_nd,
        project_to_psd,
        rts_smoother_nd,
        simulate_lgssm_nd,
        white_box_point_attack_nd,
    )
    from io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from comparison import (
        compute_contamination_prior,
        gaussian_logpdf,
        log_mix_posterior_weight,
        rank_one_covariance_update,
        safe_unit_direction,
        set_plot_theme,
        solve_spd,
        spd_inverse,
        style_axis,
    )


# ============================================================
# Shared experiment constants
# ============================================================
DEFAULT_T = 5
DEFAULT_ATTACK_T = DEFAULT_T
DEFAULT_SEED = 2025
DEFAULT_COVERAGE = 0.95
DEFAULT_M_STAR = np.array([0.9], dtype=float)
DEFAULT_ETA = 1.5
DEFAULT_N_STEPS = 500
DEFAULT_N_MC_OPT = 96
DEFAULT_N_MC_EST = 1200
DEFAULT_CALL_THRESHOLD = 0.90
DEFAULT_POSTERIOR_ATTACK_THRESHOLD = 0.50


# ============================================================
# Small plotting / statistics helpers
# ============================================================
def parse_lambda_scales(raw_values: str) -> np.ndarray:
    """Parse a comma-separated list of nonnegative lambda scales."""
    pieces = [piece.strip() for piece in raw_values.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("At least one lambda-scale value is required.")

    lambda_scales = np.array([float(piece) for piece in pieces], dtype=float)
    if np.any(lambda_scales < 0.0):
        raise ValueError("All lambda-scale values must be nonnegative.")
    return np.sort(np.unique(lambda_scales))


def build_lambda_colormap(
    lambda_scales: np.ndarray,
) -> tuple[mcolors.Colormap, mcolors.Normalize, np.ndarray]:
    """
    Build the viridis mapping shared by lambda curves and the colorbar.

    Using the same normalization in both places keeps each defended curve
    visually aligned with the scale shown outside the main legend.
    """
    lambda_scales = np.asarray(lambda_scales, dtype=float)
    positive_scales = lambda_scales[lambda_scales > 0.0]
    if positive_scales.size == 0:
        positive_scales = np.array([1.0], dtype=float)

    if positive_scales.size == 1:
        center = float(positive_scales[0])
        vmin = max(center / 1.5, 1e-12)
        vmax = max(center * 1.5, vmin * 1.001)
        norm: mcolors.Normalize = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.LogNorm(vmin=float(np.min(positive_scales)), vmax=float(np.max(positive_scales)))
    return plt.get_cmap("viridis"), norm, positive_scales


def lambda_colors(lambda_scales: np.ndarray) -> list[str]:
    """Return viridis colors matched to the requested lambda scales."""
    lambda_scales = np.asarray(lambda_scales, dtype=float)
    if lambda_scales.size == 0:
        return []

    cmap, norm, positive_scales = build_lambda_colormap(lambda_scales)
    if positive_scales.size == 1 and lambda_scales.size == 1:
        return [mcolors.to_hex(cmap(0.58)[:3])]
    return [mcolors.to_hex(cmap(float(norm(max(scale, positive_scales[0]))))[:3]) for scale in lambda_scales]


def tukey_inliers(values: np.ndarray, whisker_scale: float = 1.5) -> np.ndarray:
    """
    Return the Tukey-IQR inliers used for the Monte Carlo summaries.

    The effect panels use these inliers only for visualization so a few
    extreme Monte Carlo runs do not dominate the plotted means and bands.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 3:
        return values

    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = q3 - q1
    lower = q1 - whisker_scale * iqr
    upper = q3 + whisker_scale * iqr
    inliers = values[(values >= lower) & (values <= upper)]
    return inliers if inliers.size > 0 else values


def summarize_effect_series(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean and interquartile envelope after Tukey trimming."""
    inliers = tukey_inliers(values)
    return (
        float(np.mean(inliers)),
        float(np.percentile(inliers, 25.0)),
        float(np.percentile(inliers, 75.0)),
    )


def add_lambda_colorbar(
    *,
    fig: plt.Figure,
    axes: list[plt.Axes],
    lambda_scales: np.ndarray,
) -> None:
    """
    Add a dedicated viridis colorbar for the lambda sweep outside the legend.

    This keeps the in-axis legend focused on baseline references while the
    lambda values are communicated through a separate visual scale.
    """
    _, norm, positive_scales = build_lambda_colormap(lambda_scales)
    if positive_scales.size == 0:
        return

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("viridis"))
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        fraction=0.040,
        pad=0.020,
    )
    colorbar.set_label(r"$c$ in $\lambda = c\,\lambda_{\max}$")
    colorbar.set_ticks(positive_scales)
    colorbar.set_ticklabels([f"{scale:g}" for scale in positive_scales])


def lambda_max_from_observation_covariance(R_tk: np.ndarray) -> float:
    """Return the largest eigenvalue of the attacked-time observation covariance."""
    eigvals = np.linalg.eigvalsh(np.asarray(R_tk, dtype=float))
    return float(np.max(eigvals))


def coverage_to_epsilon(coverage: float) -> float:
    """Convert 3D ellipsoid coverage into the corresponding chi-square threshold."""
    if not (0.0 < coverage < 1.0):
        raise ValueError("coverage must lie in (0, 1)")
    return float(chi2.ppf(coverage, df=3))


def inlier_mask_iqr(values: np.ndarray) -> np.ndarray:
    """
    Return a Tukey-IQR inlier mask for one-dimensional data.

    This matches the trimming logic used in the 3D call-summary script so the
    binned probability curves remain stable when a few runs are extreme.
    """
    values = np.asarray(values, dtype=float)
    if values.size <= 3:
        return np.ones(values.shape, dtype=bool)

    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = q3 - q1
    if iqr <= 0.0:
        return np.ones(values.shape, dtype=bool)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (values >= lower) & (values <= upper)


def binned_probability_summary(
    true_prob: np.ndarray,
    estimated_prob: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Group estimated probabilities by the true clean probability level.

    The returned arrays contain the bin centers in true-probability space, the
    trimmed mean estimate in each occupied bin, and one-standard-deviation
    lower/upper envelopes around that trimmed mean.
    """
    true_prob = np.asarray(true_prob, dtype=float)
    estimated_prob = np.asarray(estimated_prob, dtype=float)
    if true_prob.shape != estimated_prob.shape:
        raise ValueError("true_prob and estimated_prob must have the same shape")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    x_vals: list[float] = []
    y_means: list[float] = []
    y_lows: list[float] = []
    y_highs: list[float] = []

    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]

        if idx == n_bins - 1:
            mask = (true_prob >= left) & (true_prob <= right)
        else:
            mask = (true_prob >= left) & (true_prob < right)

        if not np.any(mask):
            continue

        true_bin = true_prob[mask]
        estimate_bin = estimated_prob[mask]
        keep_mask = inlier_mask_iqr(estimate_bin)
        if np.any(keep_mask):
            true_bin = true_bin[keep_mask]
            estimate_bin = estimate_bin[keep_mask]

        mean_x = float(np.mean(true_bin))
        mean_y = float(np.mean(estimate_bin))
        std_y = float(np.std(estimate_bin))

        x_vals.append(mean_x)
        y_means.append(mean_y)
        y_lows.append(mean_y - std_y)
        y_highs.append(mean_y + std_y)

    return (
        np.asarray(x_vals),
        np.asarray(y_means),
        np.asarray(y_lows),
        np.asarray(y_highs),
    )


def smooth_series(values: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Smooth a one-dimensional series with a centered moving average.

    The edge values are padded so the output length matches the input length
    and the curve remains visually stable near 0 and 1.
    """
    values = np.asarray(values, dtype=float)
    if values.size <= 2:
        return values.copy()

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if values.size < window:
        window = values.size if values.size % 2 == 1 else max(3, values.size - 1)
    if window <= 1:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


# ============================================================
# Covariance-adaptation filter for the 3D nonlinear example
# ============================================================
def kalman_filter_with_online_covariance_adaptation_3d(
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
    posterior_attack_threshold: float = DEFAULT_POSTERIOR_ATTACK_THRESHOLD,
    direction_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Run the online covariance-adaptation defense on the 3D AttackSense model.

    Unlike the earlier AdvSSM helper, this version follows the `u` indexing
    convention used by `AttackSense3D.py`, where controls have shape `(T+1,n_u)`
    and the transition from `k` to `k+1` uses `u[k]`.

    In this thresholded variant, the covariance adaptation is activated only
    when the posterior attack probability `gamma_t` exceeds 0.5.
    """
    T = y.shape[0] - 1
    n_x = P0.shape[0]
    n_y = y.shape[1]
    I_x = np.eye(n_x, dtype=float)

    m_pred = np.zeros((T + 1, n_x), dtype=float)
    P_pred = np.zeros((T + 1, n_x, n_x), dtype=float)
    m_filt = np.zeros((T + 1, n_x), dtype=float)
    P_filt = np.zeros((T + 1, n_x, n_x), dtype=float)

    diagnostics = {
        "pi_t": np.zeros(T + 1, dtype=float),
        "gamma_t": np.zeros(T + 1, dtype=float),
        "bar_gamma_t": np.zeros(T + 1, dtype=float),
        "u_t": np.zeros((T + 1, n_y), dtype=float),
        "V_tilde_t": np.zeros((T + 1, n_y, n_y), dtype=float),
    }

    m_pred[0] = np.asarray(m0, dtype=float)
    P_pred[0] = project_to_psd(np.asarray(P0, dtype=float))

    for k in range(T + 1):
        Hk = np.asarray(H_t[k], dtype=float)
        Dk = np.asarray(D_t[k], dtype=float)
        Rk = project_to_psd(np.asarray(R_t[k], dtype=float))

        y_hat = Hk @ m_pred[k] + Dk @ u[k]
        S_nom = project_to_psd(Hk @ P_pred[k] @ Hk.T + Rk)
        innov = y[k] - y_hat

        V_tilde = Rk.copy()
        S_tilde = S_nom.copy()
        K_tilde = solve_spd(S_tilde, Hk @ P_pred[k].T).T
        pi_t = 0.0
        gamma_t = 0.0
        bar_gamma_t = 0.0
        u_dir = np.zeros(n_y, dtype=float)

        if attack_targets is not None and k in attack_targets:
            adv_target = np.asarray(attack_targets[k], dtype=float).reshape(n_y)
            delta_adv = adv_target - y_hat
            u_dir, delta_norm = safe_unit_direction(delta_adv, eps=direction_eps)

            if delta_norm >= direction_eps:
                pi_t, _, _ = compute_contamination_prior(
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
                bar_gamma_t = gamma_t if gamma_t > posterior_attack_threshold else 0.0

                V_tilde = rank_one_covariance_update(Rk, lam, u_dir, weight=bar_gamma_t)
                S_tilde = rank_one_covariance_update(S_nom, lam, u_dir, weight=bar_gamma_t)
                K_tilde = solve_spd(S_tilde, Hk @ P_pred[k].T).T

        m_filt[k] = m_pred[k] + K_tilde @ innov
        joseph_left = I_x - K_tilde @ Hk
        P_filt[k] = project_to_psd(
            joseph_left @ P_pred[k] @ joseph_left.T + K_tilde @ V_tilde @ K_tilde.T
        )

        diagnostics["pi_t"][k] = pi_t
        diagnostics["gamma_t"][k] = gamma_t
        diagnostics["bar_gamma_t"][k] = bar_gamma_t
        diagnostics["u_t"][k] = u_dir
        diagnostics["V_tilde_t"][k] = V_tilde

        if k < T:
            Ak = np.asarray(A_t[k], dtype=float)
            Bk = np.asarray(B_t[k], dtype=float)
            Qk = project_to_psd(np.asarray(Q_t[k], dtype=float))
            m_pred[k + 1] = Ak @ m_filt[k] + Bk @ u[k]
            P_pred[k + 1] = project_to_psd(Ak @ P_filt[k] @ Ak.T + Qk)

    return m_filt, P_filt, m_pred, P_pred, diagnostics


# ============================================================
# Clean / attack / defense evaluation
# ============================================================
def compute_smoothed_probability(
    *,
    m_filt: np.ndarray,
    P_filt: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    A_t: np.ndarray,
    attack_t: int,
    n_mc_est: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Smooth the filtered trajectory and estimate `E[g(s_T) | o]`.

    Returning the smoothed means and covariances lets the same helper support
    both the state-trajectory panels and the scalar `g` summaries.
    """
    m_smooth, P_smooth = rts_smoother_nd(
        m_filt=m_filt,
        P_filt=P_filt,
        m_pred=m_pred,
        P_pred=P_pred,
        A_t=A_t,
    )
    mu_g, _ = estimate_E_g(
        m=m_smooth[attack_t],
        P=P_smooth[attack_t],
        g=g_scalar,
        n_mc=n_mc_est,
        seed=77,
    )
    return m_smooth, P_smooth, float(mu_g[0])


def build_attack_on_g(
    *,
    y_clean: np.ndarray,
    u_controls: np.ndarray,
    mats: dict[str, np.ndarray],
    m0: np.ndarray,
    P0: np.ndarray,
    epsilon: float,
    attack_t: int,
    attack_seed: int,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
) -> dict[str, np.ndarray | float]:
    """
    Build the last-time attacked observation using the 3D nonlinear objective.
    """
    y_star, history = white_box_point_attack_nd(
        t=attack_t,
        y=y_clean,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        P0=P0,
        m0=m0,
        epsilon=epsilon,
        M_star=DEFAULT_M_STAR,
        g=g_scalar,
        g_grad=g_scalar_grad,
        eta=eta,
        n_steps=n_steps,
        n_mc=n_mc_opt,
        seed=attack_seed,
    )

    y_adv = np.asarray(y_clean, dtype=float).copy()
    y_adv[attack_t] = np.asarray(y_star, dtype=float)

    return {
        "y_adv": y_adv,
        "adv_target": np.asarray(y_star, dtype=float),
        "mu_t": np.asarray(history["mu_t"], dtype=float),
        "Sigma_t": np.asarray(history["Sigma_t"], dtype=float),
        "obj_hist": np.asarray(history["obj_hist"], dtype=float),
    }


def evaluate_reference_g_lambda_sweep(
    *,
    T: int,
    attack_t: int,
    seed: int,
    coverage: float,
    lambda_scales: np.ndarray,
    omega_h: float,
    omega_o: float,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_mc_est: int,
) -> dict[str, np.ndarray | float | int]:
    """
    Simulate one deterministic 3D nonlinear run and evaluate the lambda sweep.

    The two state panels later show `s_t^(2)` and `s_t^(3)` because those are
    the dominant variables in the definition of `g` in `AttackSense3D.py`.
    """
    pars = get_system_parameters()
    epsilon = coverage_to_epsilon(coverage)

    x_true, y_clean, u_controls, mats = simulate_lgssm_nd(
        A0=pars["A0"],
        B0=pars["B0"],
        H0=pars["H0"],
        D0=pars["D0"],
        T=T,
        seed=seed,
        x0=pars["x0"],
        Q0=pars["Q0"],
        R0=pars["R0"],
        dA=pars["dA"],
        dB=pars["dB"],
        dH=pars["dH"],
        dD=pars["dD"],
        dQ=pars["dQ"],
        dR=pars["dR"],
        u_low=-0.5,
        u_high=0.5,
    )

    attack_data = build_attack_on_g(
        y_clean=y_clean,
        u_controls=u_controls,
        mats=mats,
        m0=pars["m0"],
        P0=pars["P0"],
        epsilon=epsilon,
        attack_t=attack_t,
        attack_seed=6000 + seed,
        eta=eta,
        n_steps=n_steps,
        n_mc_opt=n_mc_opt,
    )
    y_adv = np.asarray(attack_data["y_adv"], dtype=float)

    clean_filt = kalman_filter_nd(
        y=y_clean,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )
    attack_filt = kalman_filter_nd(
        y=y_adv,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )

    clean_m, _, clean_prob = compute_smoothed_probability(
        m_filt=clean_filt[0],
        P_filt=clean_filt[1],
        m_pred=clean_filt[2],
        P_pred=clean_filt[3],
        A_t=mats["A_t"],
        attack_t=attack_t,
        n_mc_est=n_mc_est,
    )
    attack_m, _, attack_prob = compute_smoothed_probability(
        m_filt=attack_filt[0],
        P_filt=attack_filt[1],
        m_pred=attack_filt[2],
        P_pred=attack_filt[3],
        A_t=mats["A_t"],
        attack_t=attack_t,
        n_mc_est=n_mc_est,
    )

    lambda_max = lambda_max_from_observation_covariance(mats["R_t"][attack_t])
    lambda_values = lambda_scales * lambda_max
    adapt_means = np.zeros((lambda_scales.size, T + 1, x_true.shape[1]), dtype=float)
    adapt_prob = np.zeros(lambda_scales.size, dtype=float)

    for lam_idx, lam in enumerate(lambda_values):
        adapt_filt = kalman_filter_with_online_covariance_adaptation_3d(
            y=y_adv,
            u=u_controls,
            A_t=mats["A_t"],
            B_t=mats["B_t"],
            H_t=mats["H_t"],
            D_t=mats["D_t"],
            Q_t=mats["Q_t"],
            R_t=mats["R_t"],
            m0=pars["m0"],
            P0=pars["P0"],
            attack_targets={attack_t: np.asarray(attack_data["adv_target"], dtype=float)},
            lam=float(lam),
            omega_h=omega_h,
            omega_o=omega_o,
        )
        adapt_means[lam_idx], _, adapt_prob[lam_idx] = compute_smoothed_probability(
            m_filt=adapt_filt[0],
            P_filt=adapt_filt[1],
            m_pred=adapt_filt[2],
            P_pred=adapt_filt[3],
            A_t=mats["A_t"],
            attack_t=attack_t,
            n_mc_est=n_mc_est,
        )

    return {
        "T": T,
        "attack_t": attack_t,
        "seed": seed,
        "coverage": coverage,
        "epsilon": epsilon,
        "lambda_scales": lambda_scales,
        "lambda_max": float(lambda_max),
        "lambda_values": lambda_values,
        "x_true": x_true,
        "clean_m": clean_m,
        "attack_m": attack_m,
        "adapt_means": adapt_means,
        "posterior_attack_threshold": DEFAULT_POSTERIOR_ATTACK_THRESHOLD,
        "true_prob": np.array([float(g_scalar(x_true[attack_t]))], dtype=float),
        "clean_prob": np.array([clean_prob], dtype=float),
        "attack_prob": np.array([attack_prob], dtype=float),
        "adapt_prob": adapt_prob,
    }


def evaluate_single_mc_g_run(
    *,
    run_seed: int,
    T: int,
    attack_t: int,
    coverage: float,
    epsilon_override: float | None,
    lambda_scales: np.ndarray,
    omega_h: float,
    omega_o: float,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_mc_est: int,
) -> dict[str, np.ndarray | float]:
    """
    Evaluate one 3D nonlinear Monte Carlo run for all lambda values.
    """
    pars = get_system_parameters()
    epsilon = float(epsilon_override) if epsilon_override is not None else coverage_to_epsilon(coverage)

    x_true, y_clean, u_controls, mats = simulate_lgssm_nd(
        A0=pars["A0"],
        B0=pars["B0"],
        H0=pars["H0"],
        D0=pars["D0"],
        T=T,
        seed=run_seed,
        x0=pars["x0"],
        Q0=pars["Q0"],
        R0=pars["R0"],
        dA=pars["dA"],
        dB=pars["dB"],
        dH=pars["dH"],
        dD=pars["dD"],
        dQ=pars["dQ"],
        dR=pars["dR"],
        u_low=-0.5,
        u_high=0.5,
    )

    attack_data = build_attack_on_g(
        y_clean=y_clean,
        u_controls=u_controls,
        mats=mats,
        m0=pars["m0"],
        P0=pars["P0"],
        epsilon=epsilon,
        attack_t=attack_t,
        attack_seed=6000 + run_seed,
        eta=eta,
        n_steps=n_steps,
        n_mc_opt=n_mc_opt,
    )
    y_adv = np.asarray(attack_data["y_adv"], dtype=float)

    true_prob = float(g_scalar(x_true[attack_t]))
    lambda_max = lambda_max_from_observation_covariance(mats["R_t"][attack_t])
    lambda_values = lambda_scales * lambda_max

    clean_filt = kalman_filter_nd(
        y=y_clean,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )
    attack_filt = kalman_filter_nd(
        y=y_adv,
        u=u_controls,
        A_t=mats["A_t"],
        B_t=mats["B_t"],
        H_t=mats["H_t"],
        D_t=mats["D_t"],
        Q_t=mats["Q_t"],
        R_t=mats["R_t"],
        m0=pars["m0"],
        P0=pars["P0"],
    )

    _, _, clean_prob = compute_smoothed_probability(
        m_filt=clean_filt[0],
        P_filt=clean_filt[1],
        m_pred=clean_filt[2],
        P_pred=clean_filt[3],
        A_t=mats["A_t"],
        attack_t=attack_t,
        n_mc_est=n_mc_est,
    )
    _, _, attack_prob = compute_smoothed_probability(
        m_filt=attack_filt[0],
        P_filt=attack_filt[1],
        m_pred=attack_filt[2],
        P_pred=attack_filt[3],
        A_t=mats["A_t"],
        attack_t=attack_t,
        n_mc_est=n_mc_est,
    )

    clean_adapt_prob = np.zeros(lambda_scales.size, dtype=float)
    adapt_prob = np.zeros(lambda_scales.size, dtype=float)
    for lam_idx, lam in enumerate(lambda_values):
        clean_adapt_filt = kalman_filter_with_online_covariance_adaptation_3d(
            y=y_clean,
            u=u_controls,
            A_t=mats["A_t"],
            B_t=mats["B_t"],
            H_t=mats["H_t"],
            D_t=mats["D_t"],
            Q_t=mats["Q_t"],
            R_t=mats["R_t"],
            m0=pars["m0"],
            P0=pars["P0"],
            attack_targets={attack_t: np.asarray(attack_data["adv_target"], dtype=float)},
            lam=float(lam),
            omega_h=omega_h,
            omega_o=omega_o,
        )
        adapt_filt = kalman_filter_with_online_covariance_adaptation_3d(
            y=y_adv,
            u=u_controls,
            A_t=mats["A_t"],
            B_t=mats["B_t"],
            H_t=mats["H_t"],
            D_t=mats["D_t"],
            Q_t=mats["Q_t"],
            R_t=mats["R_t"],
            m0=pars["m0"],
            P0=pars["P0"],
            attack_targets={attack_t: np.asarray(attack_data["adv_target"], dtype=float)},
            lam=float(lam),
            omega_h=omega_h,
            omega_o=omega_o,
        )
        _, _, clean_adapt_prob[lam_idx] = compute_smoothed_probability(
            m_filt=clean_adapt_filt[0],
            P_filt=clean_adapt_filt[1],
            m_pred=clean_adapt_filt[2],
            P_pred=clean_adapt_filt[3],
            A_t=mats["A_t"],
            attack_t=attack_t,
            n_mc_est=n_mc_est,
        )
        _, _, adapt_prob[lam_idx] = compute_smoothed_probability(
            m_filt=adapt_filt[0],
            P_filt=adapt_filt[1],
            m_pred=adapt_filt[2],
            P_pred=adapt_filt[3],
            A_t=mats["A_t"],
            attack_t=attack_t,
            n_mc_est=n_mc_est,
        )

    return {
        "true_prob": true_prob,
        "clean_prob": clean_prob,
        "attack_prob": attack_prob,
        "posterior_attack_threshold": DEFAULT_POSTERIOR_ATTACK_THRESHOLD,
        "clean_adapt_prob": clean_adapt_prob,
        "adapt_prob": adapt_prob,
    }


def run_monte_carlo_g_lambda_sweep(
    *,
    N_runs: int,
    T: int,
    attack_t: int,
    coverage: float,
    epsilon_override: float | None,
    lambda_scales: np.ndarray,
    omega_h: float,
    omega_o: float,
    eta: float,
    n_steps: int,
    n_mc_opt: int,
    n_mc_est: int,
    base_seed: int,
) -> dict[str, np.ndarray | float | int]:
    """
    Run the Monte Carlo comparison for the nonlinear `g`-attack case.
    """
    true_prob = np.full(N_runs, np.nan, dtype=float)
    clean_prob = np.full(N_runs, np.nan, dtype=float)
    attack_prob = np.full(N_runs, np.nan, dtype=float)
    clean_adapt_prob = np.full((lambda_scales.size, N_runs), np.nan, dtype=float)
    adapt_prob = np.full((lambda_scales.size, N_runs), np.nan, dtype=float)

    for run_idx in range(N_runs):
        run_seed = base_seed + 1000 * run_idx
        print(f"[MC-g] run {run_idx + 1}/{N_runs} (seed={run_seed})")
        try:
            result = evaluate_single_mc_g_run(
                run_seed=run_seed,
                T=T,
                attack_t=attack_t,
                coverage=coverage,
                epsilon_override=epsilon_override,
                lambda_scales=lambda_scales,
                omega_h=omega_h,
                omega_o=omega_o,
                eta=eta,
                n_steps=n_steps,
                n_mc_opt=n_mc_opt,
                n_mc_est=n_mc_est,
            )
            true_prob[run_idx] = float(result["true_prob"])
            clean_prob[run_idx] = float(result["clean_prob"])
            attack_prob[run_idx] = float(result["attack_prob"])
            clean_adapt_prob[:, run_idx] = np.asarray(result["clean_adapt_prob"], dtype=float)
            adapt_prob[:, run_idx] = np.asarray(result["adapt_prob"], dtype=float)
        except Exception as exc:
            print(f"[WARN seed={run_seed}] {type(exc).__name__}: {exc}")

    clean_abs_error = np.abs(clean_prob - true_prob)
    attack_abs_error = np.abs(attack_prob - true_prob)
    adapt_abs_error = np.abs(adapt_prob - true_prob[None, :])

    return {
        "N_runs": N_runs,
        "T": T,
        "attack_t": attack_t,
        "coverage": coverage,
        "epsilon": float(epsilon_override) if epsilon_override is not None else coverage_to_epsilon(coverage),
        "lambda_scales": lambda_scales,
        "posterior_attack_threshold": DEFAULT_POSTERIOR_ATTACK_THRESHOLD,
        "true_prob": true_prob,
        "clean_prob": clean_prob,
        "attack_prob": attack_prob,
        "clean_adapt_prob": clean_adapt_prob,
        "adapt_prob": adapt_prob,
        "clean_abs_error": clean_abs_error,
        "attack_abs_error": attack_abs_error,
        "adapt_abs_error": adapt_abs_error,
    }


# ============================================================
# Figure drawing
# ============================================================
def plot_g_call_summary_vs_lambda(
    *,
    left_true_prob: np.ndarray,
    left_clean_prob: np.ndarray,
    left_attack_prob: np.ndarray,
    left_adapt_prob: np.ndarray,
    left_epsilon: float,
    right_true_prob: np.ndarray,
    right_clean_prob: np.ndarray,
    right_attack_prob: np.ndarray,
    right_clean_adapt_prob: np.ndarray,
    right_adapt_prob: np.ndarray,
    right_epsilon: float,
    lambda_scales: np.ndarray,
    call_threshold: float,
    outpath: str,
) -> None:
    """
    Draw the two-panel nonlinear comparison requested by the user.

    This intentionally mirrors `AttackSense3D_CallSummary.py`, but fixes
    epsilon at 95% coverage and uses different `lambda` values for the defense.
    """
    set_plot_theme()

    left_true_prob = np.asarray(left_true_prob, dtype=float)
    left_clean_prob = np.asarray(left_clean_prob, dtype=float)
    left_attack_prob = np.asarray(left_attack_prob, dtype=float)
    left_adapt_prob = np.asarray(left_adapt_prob, dtype=float)
    right_true_prob = np.asarray(right_true_prob, dtype=float)
    right_clean_prob = np.asarray(right_clean_prob, dtype=float)
    right_attack_prob = np.asarray(right_attack_prob, dtype=float)
    right_clean_adapt_prob = np.asarray(right_clean_adapt_prob, dtype=float)
    right_adapt_prob = np.asarray(right_adapt_prob, dtype=float)
    lambda_scales = np.asarray(lambda_scales, dtype=float)

    colors = lambda_colors(lambda_scales)
    clean_color = "#525252"
    attack_color = "#D98F8F"
    n_bins = max(30, min(40, left_clean_prob.size // 2 if left_clean_prob.size >= 12 else left_clean_prob.size))

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.9), constrained_layout=True)
    ax_prob, ax_calls = axes
    for ax in axes:
        style_axis(ax)

    diag_x = np.linspace(0.0, 1.0, 200)
    ax_prob.plot(
        diag_x,
        diag_x,
        color=clean_color,
        linewidth=2.0,
        linestyle="--",
        label="Clean mean",
        zorder=3,
    )

    attack_x, attack_mean, attack_low, attack_high = binned_probability_summary(
        left_true_prob,
        left_attack_prob,
        n_bins=n_bins,
    )
    attack_mean = np.clip(smooth_series(attack_mean, window=7), 0.0, 1.0)
    attack_low = np.clip(smooth_series(attack_low, window=7), 0.0, 1.0)
    attack_high = np.clip(smooth_series(attack_high, window=7), 0.0, 1.0)
    ax_prob.fill_between(
        attack_x,
        attack_low,
        attack_high,
        color=attack_color,
        alpha=0.14,
        zorder=1,
    )
    ax_prob.plot(
        attack_x,
        attack_mean,
        color=attack_color,
        linewidth=1.9,
        label="Attack mean",
        zorder=3,
    )

    for color, scale, prob_values in zip(colors, lambda_scales, left_adapt_prob, strict=True):
        x_bin, prob_mean, prob_low, prob_high = binned_probability_summary(
            left_true_prob,
            prob_values,
            n_bins=n_bins,
        )
        prob_mean = np.clip(smooth_series(prob_mean, window=7), 0.0, 1.0)
        prob_low = np.clip(smooth_series(prob_low, window=7), 0.0, 1.0)
        prob_high = np.clip(smooth_series(prob_high, window=7), 0.0, 1.0)
        ax_prob.fill_between(
            x_bin,
            prob_low,
            prob_high,
            color=color,
            alpha=0.12,
            zorder=1,
        )
        ax_prob.plot(
            x_bin,
            prob_mean,
            color=color,
            linewidth=1.8,
            alpha=0.72,
            zorder=3,
        )

    ax_prob.axhline(
        call_threshold,
        color="#8A8A8A",
        linewidth=1.1,
        linestyle=":",
        alpha=0.9,
        label="Call threshold",
        zorder=2,
    )
    ax_prob.set_xlabel(r"Real clean probability $g(s_T)$")
    ax_prob.set_ylabel(r"Estimated call probability $\mathbb{E}[g(s_T)\mid o_{0:T}]$")
    ax_prob.set_xlim(0.0, 1.0)
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_prob.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_prob.text(
        0.03,
        0.97,
        rf"$\epsilon={left_epsilon:.2f}$",
        transform=ax_prob.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#D0D0D0", alpha=0.92, boxstyle="round,pad=0.25"),
    )
    ax_prob.legend(loc="lower right", frameon=True, framealpha=0.95)
    add_lambda_colorbar(fig=fig, axes=[ax_prob, ax_calls], lambda_scales=lambda_scales)

    true_calls = right_true_prob > call_threshold
    clean_calls = right_clean_prob > call_threshold
    attack_calls = right_attack_prob > call_threshold
    clean_adapt_calls = right_clean_adapt_prob > call_threshold
    adapt_calls = right_adapt_prob > call_threshold

    clean_false_calls = clean_calls & (~true_calls)
    attack_false_calls = attack_calls & (~true_calls)
    clean_adapt_false_calls = clean_adapt_calls & (~true_calls[None, :])
    adapt_false_calls = adapt_calls & (~true_calls[None, :])
    clean_false_negatives = (~clean_calls) & true_calls
    attack_false_negatives = (~attack_calls) & true_calls
    clean_adapt_false_negatives = (~clean_adapt_calls) & true_calls[None, :]
    adapt_false_negatives = (~adapt_calls) & true_calls[None, :]

    clean_total_call_count = int(np.sum(clean_calls))
    attack_total_call_count = int(np.sum(attack_calls))
    clean_adapt_total_call_count = np.sum(clean_adapt_calls, axis=1).astype(int)
    adapt_total_call_count = np.sum(adapt_calls, axis=1).astype(int)
    true_call_count = int(np.sum(true_calls))

    clean_false_positive_rate = (
        100.0 * int(np.sum(clean_false_calls)) / clean_total_call_count
        if clean_total_call_count > 0
        else 0.0
    )
    attack_false_positive_rate = (
        100.0 * int(np.sum(attack_false_calls)) / attack_total_call_count
        if attack_total_call_count > 0
        else 0.0
    )
    clean_false_over_total_pct = np.divide(
        100.0 * np.sum(clean_adapt_false_calls, axis=1),
        clean_adapt_total_call_count,
        out=np.zeros_like(clean_adapt_total_call_count, dtype=float),
        where=clean_adapt_total_call_count > 0,
    )
    false_over_total_pct = np.divide(
        100.0 * np.sum(adapt_false_calls, axis=1),
        adapt_total_call_count,
        out=np.zeros_like(adapt_total_call_count, dtype=float),
        where=adapt_total_call_count > 0,
    )
    clean_false_negative_rate = (
        100.0 * int(np.sum(clean_false_negatives)) / true_call_count
        if true_call_count > 0
        else 0.0
    )
    attack_false_negative_rate = (
        100.0 * int(np.sum(attack_false_negatives)) / true_call_count
        if true_call_count > 0
        else 0.0
    )
    clean_false_negative_rate_pct = (
        100.0 * np.sum(clean_adapt_false_negatives, axis=1) / true_call_count
        if true_call_count > 0
        else np.zeros(right_clean_adapt_prob.shape[0], dtype=float)
    )
    false_negative_rate_pct = (
        100.0 * np.sum(adapt_false_negatives, axis=1) / true_call_count
        if true_call_count > 0
        else np.zeros(right_adapt_prob.shape[0], dtype=float)
    )

    positive_mask = lambda_scales > 0.0
    positive_lambda_scales = lambda_scales[positive_mask]
    positive_clean_false_over_total_pct = clean_false_over_total_pct[positive_mask]
    positive_clean_false_negative_rate_pct = clean_false_negative_rate_pct[positive_mask]
    positive_false_over_total_pct = false_over_total_pct[positive_mask]
    positive_false_negative_rate_pct = false_negative_rate_pct[positive_mask]
    # Apply a light moving-average smoothing only at plotting time so the
    # lambda-rate trends are easier to read without changing the raw metrics.
    smooth_window_calls = 5
    plot_clean_false_over_total_pct = smooth_series(
        positive_clean_false_over_total_pct,
        window=smooth_window_calls,
    )
    plot_clean_false_negative_rate_pct = smooth_series(
        positive_clean_false_negative_rate_pct,
        window=smooth_window_calls,
    )
    plot_false_over_total_pct = smooth_series(
        positive_false_over_total_pct,
        window=smooth_window_calls,
    )
    plot_false_negative_rate_pct = smooth_series(
        positive_false_negative_rate_pct,
        window=smooth_window_calls,
    )

    ax_calls.plot(
        positive_lambda_scales,
        plot_clean_false_over_total_pct,
        color="#6EA7C6",
        marker="o",
        linewidth=1.9,
        markersize=5.2,
        linestyle="--",
        label="Clean + cov-adapt false positive rate",
        zorder=3,
    )
    ax_calls.plot(
        positive_lambda_scales,
        plot_clean_false_negative_rate_pct,
        color="#6F79C9",
        marker="s",
        linewidth=1.8,
        markersize=4.8,
        linestyle="--",
        label="Clean + cov-adapt false negative rate",
        zorder=3,
    )
    ax_calls.plot(
        positive_lambda_scales,
        plot_false_over_total_pct,
        color="#7FB26A",
        marker="o",
        linewidth=2.2,
        markersize=6.0,
        label="Cov-adapt false positive rate",
        zorder=3,
    )
    ax_calls.plot(
        positive_lambda_scales,
        plot_false_negative_rate_pct,
        color="#5B84C4",
        marker="s",
        linewidth=2.0,
        markersize=5.4,
        label="Cov-adapt false negative rate",
        zorder=3,
    )
    ax_calls.axhline(
        clean_false_positive_rate,
        color=clean_color,
        linewidth=1.4,
        linestyle="--",
        alpha=0.9,
        label="Clean false positive rate",
        zorder=2,
    )
    ax_calls.axhline(
        attack_false_positive_rate,
        color=attack_color,
        linewidth=1.4,
        linestyle="--",
        alpha=0.9,
        label="Attack false positive rate",
        zorder=2,
    )
    ax_calls.axhline(
        clean_false_negative_rate,
        color="#6A6A6A",
        linewidth=1.2,
        linestyle=":",
        alpha=0.92,
        label="Clean false negative rate",
        zorder=2,
    )
    ax_calls.axhline(
        attack_false_negative_rate,
        color="#C46D61",
        linewidth=1.2,
        linestyle=":",
        alpha=0.92,
        label="Attack false negative rate",
        zorder=2,
    )

    ax_calls.set_xscale("log")
    ax_calls.set_xlim(
        float(np.min(positive_lambda_scales)) * 0.92,
        float(np.max(positive_lambda_scales)) * 1.08,
    )
    ax_calls.xaxis.set_major_locator(mticker.FixedLocator(positive_lambda_scales))
    ax_calls.xaxis.set_major_formatter(mticker.FixedFormatter([f"{scale:g}" for scale in positive_lambda_scales]))
    ax_calls.xaxis.set_minor_locator(mticker.NullLocator())
    ax_calls.minorticks_off()
    ax_calls.set_xlabel(r"$c$ in $\lambda = c\,\lambda_{\max}$")
    ax_calls.set_ylabel("False positive / negative rate (%)")
    max_rate = max(
        float(np.max(plot_clean_false_over_total_pct)) if plot_clean_false_over_total_pct.size > 0 else 0.0,
        float(np.max(plot_clean_false_negative_rate_pct)) if plot_clean_false_negative_rate_pct.size > 0 else 0.0,
        float(np.max(plot_false_over_total_pct)) if plot_false_over_total_pct.size > 0 else 0.0,
        float(np.max(plot_false_negative_rate_pct)) if plot_false_negative_rate_pct.size > 0 else 0.0,
        clean_false_positive_rate,
        attack_false_positive_rate,
        clean_false_negative_rate,
        attack_false_negative_rate,
        5.0,
    )
    ax_calls.set_ylim(0.0, min(100.0, max_rate + 10.0))
    ax_calls.text(
        0.03,
        0.97,
        rf"$\epsilon={right_epsilon:.2f}$",
        transform=ax_calls.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="#D0D0D0", alpha=0.92, boxstyle="round,pad=0.25"),
    )
    ax_calls.legend(loc="upper left", frameon=True, framealpha=0.95)

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
    Generate the nonlinear `g` call-summary comparison for epsilon 95%.

    The default setup follows the 3D `AttackSense3D` example, fixes the
    ellipsoid coverage at 95%, and varies the covariance-adaptation lambda.
    """
    T = int(os.environ.get("COVADAPT_G_T", str(DEFAULT_T)))
    attack_t = int(os.environ.get("COVADAPT_G_ATTACK_T", str(DEFAULT_ATTACK_T)))
    seed = int(os.environ.get("COVADAPT_G_SEED", str(DEFAULT_SEED)))
    coverage = float(os.environ.get("COVADAPT_G_COVERAGE", "0.95"))
    raw_lambda_scales = os.environ.get("COVADAPT_G_LAMBDA_SCALES", "0.2,0.5,1.0,2.0,5.0,10.0")
    omega_h = float(os.environ.get("COVADAPT_G_OMEGA_H", "0.50"))
    omega_o = float(os.environ.get("COVADAPT_G_OMEGA_O", "0.50"))
    N_runs = int(os.environ.get("COVADAPT_G_MC_RUNS", "500"))
    mc_seed = int(os.environ.get("COVADAPT_G_MC_BASE_SEED", "2025"))
    eta = float(os.environ.get("COVADAPT_G_ETA", str(DEFAULT_ETA)))
    n_steps = int(os.environ.get("COVADAPT_G_N_STEPS", str(DEFAULT_N_STEPS)))
    n_mc_opt = int(os.environ.get("COVADAPT_G_N_MC_OPT", str(DEFAULT_N_MC_OPT)))
    n_mc_est = int(os.environ.get("COVADAPT_G_N_MC_EST", str(DEFAULT_N_MC_EST)))
    force_mc = os.environ.get("COVADAPT_G_FORCE_MC", "0") == "1"

    lambda_scales = parse_lambda_scales(raw_lambda_scales)
    base_epsilon = coverage_to_epsilon(coverage)
    epsilon_left = float(os.environ.get("COVADAPT_G_EPS_LEFT", str(0.5 * base_epsilon)))
    epsilon_right = float(os.environ.get("COVADAPT_G_EPS_RIGHT", str(1.5 * base_epsilon)))

    if not (0 <= attack_t <= T):
        raise ValueError("attack_t must satisfy 0 <= attack_t <= T")
    if not np.isclose(omega_h + omega_o, 1.0, atol=1e-9):
        raise ValueError("omega_h and omega_o must sum to 1.")

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    lambda_tag = "-".join(f"{value:g}" for value in lambda_scales).replace(".", "p")
    outpath = os.path.join(
        out_dir,
        f"comparison_g_callsummary_cov{int(round(100 * coverage))}_t{attack_t}_T{T}_seed{seed}_N{N_runs}_{lambda_tag}.png",
    )
    left_tag = str(epsilon_left).replace(".", "p")
    right_tag = str(epsilon_right).replace(".", "p")
    left_mc_data_path = data_path_for_plot(outpath.replace(".png", f"_left_eps{left_tag}_mc.png"))
    right_mc_data_path = data_path_for_plot(outpath.replace(".png", f"_right_eps{right_tag}_mc.png"))

    def compute_left_mc_data() -> dict[str, np.ndarray | float | int]:
        return run_monte_carlo_g_lambda_sweep(
            N_runs=N_runs,
            T=T,
            attack_t=attack_t,
            coverage=coverage,
            epsilon_override=epsilon_left,
            lambda_scales=lambda_scales,
            omega_h=omega_h,
            omega_o=omega_o,
            eta=eta,
            n_steps=n_steps,
            n_mc_opt=n_mc_opt,
            n_mc_est=n_mc_est,
            base_seed=mc_seed,
        )
    def compute_right_mc_data() -> dict[str, np.ndarray | float | int]:
        return run_monte_carlo_g_lambda_sweep(
            N_runs=N_runs,
            T=T,
            attack_t=attack_t,
            coverage=coverage,
            epsilon_override=epsilon_right,
            lambda_scales=lambda_scales,
            omega_h=omega_h,
            omega_o=omega_o,
            eta=eta,
            n_steps=n_steps,
            n_mc_opt=n_mc_opt,
            n_mc_est=n_mc_est,
            base_seed=mc_seed,
        )

    left_mc_data = cached_npz(left_mc_data_path, compute_left_mc_data, force=force_mc)
    right_mc_data = cached_npz(right_mc_data_path, compute_right_mc_data, force=force_mc)
    cached_threshold_left = float(left_mc_data.get("posterior_attack_threshold", np.nan))
    cached_threshold_right = float(right_mc_data.get("posterior_attack_threshold", np.nan))
    if not np.isclose(cached_threshold_left, DEFAULT_POSTERIOR_ATTACK_THRESHOLD, atol=1e-12):
        print("[cache] left Monte Carlo cache uses a different posterior threshold; recomputing.")
        left_mc_data = cached_npz(left_mc_data_path, compute_left_mc_data, force=True)
    if not np.isclose(cached_threshold_right, DEFAULT_POSTERIOR_ATTACK_THRESHOLD, atol=1e-12):
        print("[cache] right Monte Carlo cache uses a different posterior threshold; recomputing.")
        right_mc_data = cached_npz(right_mc_data_path, compute_right_mc_data, force=True)
    required_mc_keys = {"clean_adapt_prob"}
    if not required_mc_keys.issubset(left_mc_data):
        print("[cache] left Monte Carlo cache is missing clean cov-adapt data; recomputing.")
        left_mc_data = cached_npz(left_mc_data_path, compute_left_mc_data, force=True)
    if not required_mc_keys.issubset(right_mc_data):
        print("[cache] right Monte Carlo cache is missing clean cov-adapt data; recomputing.")
        right_mc_data = cached_npz(right_mc_data_path, compute_right_mc_data, force=True)

    plot_g_call_summary_vs_lambda(
        left_true_prob=np.asarray(left_mc_data["true_prob"], dtype=float),
        left_clean_prob=np.asarray(left_mc_data["clean_prob"], dtype=float),
        left_attack_prob=np.asarray(left_mc_data["attack_prob"], dtype=float),
        left_adapt_prob=np.asarray(left_mc_data["adapt_prob"], dtype=float),
        left_epsilon=float(left_mc_data["epsilon"]),
        right_true_prob=np.asarray(right_mc_data["true_prob"], dtype=float),
        right_clean_prob=np.asarray(right_mc_data["clean_prob"], dtype=float),
        right_attack_prob=np.asarray(right_mc_data["attack_prob"], dtype=float),
        right_clean_adapt_prob=np.asarray(right_mc_data["clean_adapt_prob"], dtype=float),
        right_adapt_prob=np.asarray(right_mc_data["adapt_prob"], dtype=float),
        right_epsilon=float(right_mc_data["epsilon"]),
        lambda_scales=np.asarray(left_mc_data["lambda_scales"], dtype=float),
        call_threshold=DEFAULT_CALL_THRESHOLD,
        outpath=outpath,
    )
    print(f"Saved figure to: {outpath}")


if __name__ == "__main__":
    main()
