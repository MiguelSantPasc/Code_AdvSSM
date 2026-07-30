"""
Compare leave-one-out predictive regions in an ND linear Gaussian SSM.

The model is:

    x_{k+1} = A_k x_k + B_k u_k + w_{k+1},    w_{k+1} ~ N(0, Q_k)
    y_k     = H_k x_k + D_k u_k + v_k,        v_k     ~ N(0, R_k)

For a selected time t, the script builds two observation-space ellipses:

    p(y_t | y_{0:t-1})       from the forward Kalman prediction only.
    p(y_t | y_{-t})          from the forward prediction combined with the
                              backward information message from future data.

Plotting both ellipses makes the information contributed by future
observations visible: the leave-one-out region generally contracts or rotates
relative to the forward-only predictive region.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def _ellipse_points_2d(center: np.ndarray, cov: np.ndarray, chi2_val: float = 5.991, n: int = 240) -> np.ndarray:
    """
    Return (n,2) points of the ellipse:
        (z-center)^T cov^{-1} (z-center) = chi2_val
    For 95% in 2D: chi2_val is approximately 5.991.
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
        out_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "outputs",
            "figures",
            "ellipses_t5.png",
        ),
    )

    

import os as _os
import sys as _sys

# Make `shared_ssm` importable when this legacy script is run directly.
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from shared_ssm.legacy import loo_values_nd_previous_observation as loo_values_nd
from shared_ssm.legacy import simulate_lgssm_nd_previous_observation as simulate_lgssm_nd


if __name__ == "__main__":
    main()
