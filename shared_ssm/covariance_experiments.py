"""Shared helpers for covariance-adaptation experiment scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .defenses.covariance_adaptation import compute_contamination_prior
from .defenses.covariance_adaptation import log_mixture_posterior_weight
from .defenses.covariance_adaptation import rank_one_covariance_update
from .legacy import build_kf_attack
from .legacy import compute_filter_predictive_quantities
from .legacy import kalman_filter_with_online_covariance_adaptation
from .linalg import gaussian_logpdf
from .linalg import project_to_psd
from .linalg import quad_form_spd
from .linalg import solve_spd
from .linalg import spd_inverse
from .linalg import stabilized_cholesky


def build_reference_setup() -> dict[str, np.ndarray]:
    """Return the deterministic 2D LGSSM used by KF covariance sweeps."""
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
    """Apply the shared pastel plotting theme used by kept scripts."""
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
    """Apply compact axis styling with legends remaining inside each plot."""
    ax.set_facecolor("#FBFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.26, linewidth=0.78)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.45)
    ax.set_axisbelow(True)


def safe_unit_direction(v: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, float]:
    """Return a unit direction and its original norm, or zeros for tiny inputs."""
    vector = np.asarray(v, dtype=float).reshape(-1)
    norm_value = float(np.linalg.norm(vector))
    if norm_value < float(eps):
        return np.zeros_like(vector), norm_value
    return vector / norm_value, norm_value


def log_mix_posterior_weight(prior_probability: float, log_clean: float, log_attack: float) -> float:
    """Compatibility wrapper for the shared log-mixture posterior weight."""
    return log_mixture_posterior_weight(
        prior_probability=prior_probability,
        log_clean=log_clean,
        log_attack=log_attack,
    )
