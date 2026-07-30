#!/usr/bin/env python3
"""
Gray-box ND attack demo:

- True system simulates (x,y,u) with known Q,R.
- White-box attacker: uses true A,B,H,D to compute (X_t, mu_t, Sigma_t), then KKT attack y*.
- Gray-box attacker: does NOT know A,B,H,D, but knows Q,R.
  Uses Gibbs with NUTS:
    1) sample x_{0:T} | params, y,u (NUTS)
    2) sample params | x_{0:T}, y,u (NUTS)
  Uses posterior mean params to compute (X_t, mu_t, Sigma_t), then KKT attack y*.

Plots: ONLY "first column" style plots (geometry + x1/x2 time series), DOUBLED:
  Left column  = white-box (true params)
  Right column = gray-box  (estimated params)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Linear algebra utilities
# -----------------------------


# -----------------------------
# Simulator (constant params; you can add drift back if needed)
# -----------------------------


# -----------------------------
# KF + RTS (constant params)
# -----------------------------


# -----------------------------
# LOO (constant params version) returning X_t, mu_t, Sigma_t
# (kept close to your version; no drift arrays)
# -----------------------------


# -----------------------------
# KKT / trust-region solver (your version, unchanged)
# -----------------------------


# -----------------------------
# Geometry helpers (2D only for y-plot)
# -----------------------------
def _ellipse_points_from_quad(center: np.ndarray, shape_inv: np.ndarray, level: float, n: int = 320) -> np.ndarray:
    center = np.asarray(center, dtype=float).reshape(2,)
    M = project_to_psd(symmetrize(np.asarray(shape_inv, dtype=float).reshape(2, 2)))
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-14)
    radii = np.sqrt(level / w)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    pts = (V @ (radii[:, None] * circle)).T + center[None, :]
    return pts


def graybox_gibbs_nuts(
    *,
    y: np.ndarray,
    u: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    num_gibbs: int = 8,
    num_warmup_x: int = 600,
    num_samples_x: int = 600,
    num_warmup_p: int = 700,
    num_samples_p: int = 700,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Gray-box attacker: unknown (A,B,H,D), known (Q,R).
    Gibbs with NUTS:
      1) sample x_{0:T} | params, y,u
      2) sample params  | x_{0:T}, y,u

    Returns posterior draws from the LAST Gibbs iteration:
      - x_samples: (Sx, T+1, n_x)
      - A_samples, B_samples, H_samples, D_samples: (Sp, ...)
      - A_mean, B_mean, H_mean, D_mean: posterior means (from last iter)
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.contrib.control_flow import scan

    # CPU (native Windows)
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(1)

    y_j = jnp.asarray(y)
    u_j = jnp.asarray(u)
    Q_j = jnp.asarray(Q)
    R_j = jnp.asarray(R)
    m0_j = jnp.asarray(m0)
    P0_j = jnp.asarray(P0)

    T = int(y.shape[0] - 1)
    n_y = int(y.shape[1])
    n_u = int(u.shape[1])
    n_x = int(m0.shape[0])

    def u_at(k: int):
        # keep your convention for k==T
        return u_j[k] if k < T else u_j[T - 1]

    # Soft bounding (demo-friendly)
    def bounded(mat_raw, scale=0.95):
        return scale * jnp.tanh(mat_raw)

    # -------------------------
    # x-step model: sample x given params
    # -------------------------
    def model_x(y_obs, u_in, A, B, H, D):
        # prior on x0
        x0 = numpyro.sample("x0", dist.MultivariateNormal(loc=m0_j, covariance_matrix=P0_j))

        # y0 likelihood
        yhat0 = H @ x0 + D @ u_at(0)
        numpyro.sample("y0", dist.MultivariateNormal(loc=yhat0, covariance_matrix=R_j), obs=y_obs[0])

        def transition(x_prev, t_idx):
            # state
            mean_xt1 = A @ x_prev + B @ u_in[t_idx]
            x_t1 = numpyro.sample(
                f"x_{t_idx+1}",
                dist.MultivariateNormal(loc=mean_xt1, covariance_matrix=Q_j),
            )

            # emission uses u[t_idx] to predict y[t_idx+1]
            yhat = H @ x_t1 + D @ u_in[t_idx]
            numpyro.sample(
                f"y_{t_idx+1}",
                dist.MultivariateNormal(loc=yhat, covariance_matrix=R_j),
                obs=y_obs[t_idx + 1],
            )

            return x_t1, x_t1

        # scan over t=0..T-1, returns x_1..x_T
        _, x_path = scan(transition, x0, jnp.arange(T))
        x_all = jnp.concatenate([x0[None, :], x_path], axis=0)  # (T+1, n_x)

        numpyro.deterministic("x", x_all)

    # -------------------------
    # param-step model: sample params given x
    # -------------------------
    def model_params(y_obs, u_in, x_in):
        # priors
        A_raw = numpyro.sample("A_raw", dist.Normal(0.0, 1.0).expand([n_x, n_x]))
        H_raw = numpyro.sample("H_raw", dist.Normal(0.0, 1.0).expand([n_y, n_x]))
        B = numpyro.sample("B", dist.Normal(0.0, 1.0).expand([n_x, n_u]))
        D = numpyro.sample("D", dist.Normal(0.0, 1.0).expand([n_y, n_u]))

        A = bounded(A_raw, scale=0.95)
        H = bounded(H_raw, scale=1.50)

        numpyro.deterministic("A", A)
        numpyro.deterministic("H", H)

        # y0 likelihood
        yhat0 = H @ x_in[0] + D @ u_at(0)
        numpyro.sample("y0", dist.MultivariateNormal(loc=yhat0, covariance_matrix=R_j), obs=y_obs[0])

        def obs_step(carry, t_idx):
            # transition likelihood for x_{t+1}
            mean_xt1 = A @ x_in[t_idx] + B @ u_in[t_idx]
            numpyro.sample(
                f"x_like_{t_idx+1}",
                dist.MultivariateNormal(loc=mean_xt1, covariance_matrix=Q_j),
                obs=x_in[t_idx + 1],
            )

            # emission likelihood for y_{t+1}
            yhat = H @ x_in[t_idx + 1] + D @ u_in[t_idx]
            numpyro.sample(
                f"y_{t_idx+1}",
                dist.MultivariateNormal(loc=yhat, covariance_matrix=R_j),
                obs=y_obs[t_idx + 1],
            )

            return carry, None

        scan(obs_step, None, jnp.arange(T))

    # -------------------------
    # Gibbs loop
    # -------------------------
    rng = jax.random.PRNGKey(seed)

    # init params
    A_cur = jnp.eye(n_x) * 0.7
    H_cur = jnp.eye(n_y, n_x)
    B_cur = jnp.zeros((n_x, n_u))
    D_cur = jnp.zeros((n_y, n_u))

    # init state path
    x_cur = jnp.tile(m0_j[None, :], (T + 1, 1))

    last: dict[str, np.ndarray] = {}

    for g in range(num_gibbs):
        # ---- x step (sample x | params)
        nuts_x = NUTS(lambda y_obs, u_in: model_x(y_obs, u_in, A_cur, B_cur, H_cur, D_cur))
        mcmc_x = MCMC(nuts_x, num_warmup=num_warmup_x, num_samples=num_samples_x, num_chains=1)

        rng, kx = jax.random.split(rng)
        mcmc_x.run(kx, y_obs=y_j, u_in=u_j)

        sx = mcmc_x.get_samples()
        x_samps = sx["x"]  # (Sx, T+1, n_x)
        x_cur = jnp.mean(x_samps, axis=0)

        # ---- param step (sample params | x)
        nuts_p = NUTS(lambda y_obs, u_in: model_params(y_obs, u_in, x_cur))
        mcmc_p = MCMC(nuts_p, num_warmup=num_warmup_p, num_samples=num_samples_p, num_chains=1)

        rng, kp = jax.random.split(rng)
        mcmc_p.run(kp, y_obs=y_j, u_in=u_j)

        sp = mcmc_p.get_samples()

        # update current params via posterior mean
        A_cur = jnp.mean(sp["A"], axis=0)
        H_cur = jnp.mean(sp["H"], axis=0)
        B_cur = jnp.mean(sp["B"], axis=0)
        D_cur = jnp.mean(sp["D"], axis=0)

        last = {
            "x_samples": np.array(x_samps),
            "A_samples": np.array(sp["A"]),
            "B_samples": np.array(sp["B"]),
            "H_samples": np.array(sp["H"]),
            "D_samples": np.array(sp["D"]),
            "A_mean": np.array(A_cur),
            "B_mean": np.array(B_cur),
            "H_mean": np.array(H_cur),
            "D_mean": np.array(D_cur),
        }

        print(f"[Gibbs {g+1}/{num_gibbs}] done.")

    return last


# -----------------------------
# Plot: doubled "first-column" panels (white-box vs gray-box)
# -----------------------------
def plot_attack_overlay_one_column(
    *,
    t: int,
    epsilon_w: float,
    epsilon_g: float,
    # geometry: white
    y_t: np.ndarray,
    mu_w: np.ndarray,
    Sigma_w: np.ndarray,
    X_w: np.ndarray,
    y_star_w: np.ndarray,
    obj_w: float,
    # geometry: gray
    mu_g: np.ndarray,
    Sigma_g: np.ndarray,
    X_g: np.ndarray,
    y_star_g: np.ndarray,
    obj_g: float,
    # time series: attacked smoothers (both evaluated under TRUE params)
    m_smooth_white: np.ndarray,
    P_smooth_white: np.ndarray,
    m_smooth_gray: np.ndarray,
    P_smooth_gray: np.ndarray,
    # truth
    x_true: np.ndarray,
    outpath: str | None = None,
) -> None:
    if y_t.shape != (2,) or mu_w.shape != (2,) or mu_g.shape != (2,) or y_star_w.shape != (2,) or y_star_g.shape != (2,):
        raise ValueError("This plot expects n_y=2 (y_t, mu_w, mu_g, y_star_w, y_star_g must be shape (2,)).")

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    T = x_true.shape[0] - 1
    tt = np.arange(T + 1)
    z = 1.96

    fig = plt.figure(figsize=(10.5, 10.8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.35, 1.0, 1.0], hspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])  # geometry
    ax2 = fig.add_subplot(gs[1, 0])  # x1
    ax3 = fig.add_subplot(gs[2, 0])  # x2

    # -----------------
    # Panel 1: Geometry overlay
    # -----------------
    Sigma_w = project_to_psd(Sigma_w)
    Sigma_g = project_to_psd(Sigma_g)

    # WHITE constraint ellipse uses epsilon_w
    pts_constraint_w = _ellipse_points_from_quad(mu_w, inv_psd(Sigma_w), epsilon_w)
    M_w = project_to_psd(symmetrize(X_w.T @ X_w))
    pts_obj_w = _ellipse_points_from_quad(y_t, M_w, obj_w)

    # GRAY constraint ellipse uses epsilon_g
    pts_constraint_g = _ellipse_points_from_quad(mu_g, inv_psd(Sigma_g), epsilon_g)
    M_g = project_to_psd(symmetrize(X_g.T @ X_g))
    pts_obj_g = _ellipse_points_from_quad(y_t, M_g, obj_g)

    # Draw WHITE
    ax1.fill(pts_constraint_w[:, 0], pts_constraint_w[:, 1], alpha=0.14, label=f"WHITE constraint (ε={epsilon_w:.3f})")
    ax1.plot(pts_constraint_w[:, 0], pts_constraint_w[:, 1], linewidth=1.8, alpha=0.9)

    ax1.fill(pts_obj_w[:, 0], pts_obj_w[:, 1], alpha=0.10, label="WHITE objective level-set")
    ax1.plot(pts_obj_w[:, 0], pts_obj_w[:, 1], linewidth=1.8, linestyle="--", alpha=0.9)

    # Draw GRAY
    ax1.fill(pts_constraint_g[:, 0], pts_constraint_g[:, 1], alpha=0.14, label=f"GRAY constraint (ε={epsilon_g:.3f})")
    ax1.plot(pts_constraint_g[:, 0], pts_constraint_g[:, 1], linewidth=1.8, alpha=0.9)

    ax1.fill(pts_obj_g[:, 0], pts_obj_g[:, 1], alpha=0.10, label="GRAY objective level-set")
    ax1.plot(pts_obj_g[:, 0], pts_obj_g[:, 1], linewidth=1.8, linestyle="--", alpha=0.9)

    # Points
    ax1.scatter([y_t[0]], [y_t[1]], s=65, marker="x", label=r"$y_t$", zorder=10)
    ax1.scatter([mu_w[0]], [mu_w[1]], s=55, marker="o", label=r"$\mu_t$ (WHITE)", zorder=10)
    ax1.scatter([y_star_w[0]], [y_star_w[1]], s=95, marker="*", label=r"$y^\star$ (WHITE)", zorder=11)

    ax1.scatter([mu_g[0]], [mu_g[1]], s=55, marker="o", label=r"$\mu_t$ (GRAY)", zorder=10)
    ax1.scatter([y_star_g[0]], [y_star_g[1]], s=95, marker="*", label=r"$y^\star$ (GRAY)", zorder=11)

    ax1.set_title(f"Attack geometry overlay at t={t} (ε_W={epsilon_w:.3f}, ε_G={epsilon_g:.3f})")
    ax1.set_xlabel("y1")
    ax1.set_ylabel("y2")
    ax1.grid(True, alpha=0.20)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(loc="best", frameon=True)

    # -----------------
    # Panels 2 & 3: time series overlay (WHITE vs GRAY)
    # -----------------
    def plot_state_overlay(ax, idx: int, title: str) -> None:
        ax.plot(tt, x_true[:, idx], marker="o", markersize=2.6, linewidth=1.05, label=f"True x[{idx}]")

        mW = m_smooth_white[:, idx]
        sdW = np.sqrt(np.maximum(P_smooth_white[:, idx, idx], 0.0))
        ax.fill_between(tt, mW - z * sdW, mW + z * sdW, alpha=0.16, label="WHITE attacked 95% CI")
        ax.plot(tt, mW, linewidth=1.4, label="WHITE attacked mean")

        mG = m_smooth_gray[:, idx]
        sdG = np.sqrt(np.maximum(P_smooth_gray[:, idx, idx], 0.0))
        ax.fill_between(tt, mG - z * sdG, mG + z * sdG, alpha=0.12, label="GRAY attacked 95% CI")
        ax.plot(tt, mG, linewidth=1.4, linestyle="--", label="GRAY attacked mean")

        ax.axvline(t, linewidth=1.0, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("time t")
        ax.set_ylabel(f"x[{idx}]")
        ax.grid(True, alpha=0.20)
        ax.legend(loc="best", frameon=True)

    plot_state_overlay(ax2, 0, "State impact overlay on x1 (component 0)")
    plot_state_overlay(ax3, 1, "State impact overlay on x2 (component 1)")

    if outpath is not None:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # dims
    n_x = n_y = n_u = 2
    T = 20
    seed = 20241

    # True parameters
    A_true = np.array([[0.65, 0.40],
                       [-0.15, 0.70]], dtype=float)

    B_true = np.array([[1.65, 0.40],
                       [-0.15, 0.70]], dtype=float)

    H_true = np.eye(n_y, n_x)
    D_true = np.zeros((n_y, n_u), dtype=float)

    Q_true = project_to_psd(0.03 * np.array([[1.6, -0.40],
                                            [0.15, 0.70]], dtype=float))
    R_true = project_to_psd(0.02 * np.array([[0.65, 1.80],
                                            [-0.15, 1.70]], dtype=float))

    x0 = np.array([0.5, 0.5], dtype=float)

    # prior for filtering/smoothing (also used by attacker)
    m0 = x0.copy()
    P0 = 0.05 * np.eye(n_x)

    # simulate
    x_true, y, u, mats = simulate_lgssm_nd(
        A=A_true, B=B_true, H=H_true, D=D_true,
        Q=Q_true, R=R_true,
        T=T, seed=seed, x0=x0,
        u_low=-0.5, u_high=0.5
    )

    # attack time
    t = 10
    y_t = y[t].copy()

    # attack constraint
    epsilon_w = 5.991
    epsilon_g = 3.991  

    # =========================================================
    # WHITE-BOX attacker (knows true params)
    # =========================================================
    X_w, mu_w, Sig_w = loo_values_nd_const(
        t=t, y=y, u=u,
        A=A_true, B=B_true, H=H_true, D=D_true,
        Q=Q_true, R=R_true,
        P0=P0, m0=m0
    )
    y_star_w, obj_w = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_w, y_t=y_t, mu=mu_w, Sigma=Sig_w, epsilon=epsilon_w
    )

    # baseline RTS on original y (true params)
    m_f_b, P_f_b, m_p_b, P_p_b = kalman_filter_nd(
        y=y, u=u,
        A=A_true, B=B_true, H=H_true, D=D_true,
        Q=Q_true, R=R_true,
        m0=m0, P0=P0
    )
    m_s_b, P_s_b = rts_smoother_nd(
        m_filt=m_f_b, P_filt=P_f_b,
        m_pred=m_p_b, P_pred=P_p_b,
        A=A_true
    )

    # white-box attacked series
    y_adv_w = y.copy()
    y_adv_w[t] = y_star_w

    # attacked RTS (true params)
    m_f_aw, P_f_aw, m_p_aw, P_p_aw = kalman_filter_nd(
        y=y_adv_w, u=u,
        A=A_true, B=B_true, H=H_true, D=D_true,
        Q=Q_true, R=R_true,
        m0=m0, P0=P0
    )
    m_s_aw, P_s_aw = rts_smoother_nd(
        m_filt=m_f_aw, P_filt=P_f_aw,
        m_pred=m_p_aw, P_pred=P_p_aw,
        A=A_true
    )

    # =========================================================
    # GRAY-BOX attacker (estimates params via Gibbs+NUTS ONCE)
    # =========================================================
    print("\nRunning gray-box Gibbs with NUTS (CPU on native Windows)...\n")
    post = graybox_gibbs_nuts(
        y=y, u=u, Q=Q_true, R=R_true,
        m0=m0, P0=P0,
        num_gibbs=20,                 # many passes (increase if you want)
        num_warmup_x=500, num_samples_x=500,
        num_warmup_p=600, num_samples_p=600,
        seed=0,
    )

    A_hat = post["A_mean"]
    B_hat = post["B_mean"]
    H_hat = post["H_mean"]
    D_hat = post["D_mean"]

    # ---- PRINT true vs estimated matrices (attacker)
    np.set_printoptions(precision=4, suppress=True)

    print("\n==================== TRUE PARAMETERS ====================")
    print("A_true =\n", A_true)
    print("B_true =\n", B_true)
    print("H_true =\n", H_true)
    print("D_true =\n", D_true)

    print("\n==================== ATTACKER ESTIMATES ==================")
    print("A_hat  =\n", A_hat)
    print("B_hat  =\n", B_hat)
    print("H_hat  =\n", H_hat)
    print("D_hat  =\n", D_hat)

    print("\n==================== ESTIMATION ERRORS ===================")
    print("A_hat - A_true =\n", A_hat - A_true)
    print("B_hat - B_true =\n", B_hat - B_true)
    print("H_hat - H_true =\n", H_hat - H_true)
    print("D_hat - D_true =\n", D_hat - D_true)
    print("=========================================================\n")

    # --- attacker uses estimated params to compute attack objects
    X_g, mu_g, Sig_g = loo_values_nd_const(
        t=t, y=y, u=u,
        A=A_hat, B=B_hat, H=H_hat, D=D_hat,
        Q=Q_true, R=R_true,
        P0=P0, m0=m0
    )
    y_star_g, obj_g = solve_kkt_max_quadratic_over_ellipsoid(
        X=X_g, y_t=y_t, mu=mu_g, Sigma=Sig_g, epsilon=epsilon_g
    )

    # gray-box attacked series (only change y[t])
    y_adv_g = y.copy()
    y_adv_g[t] = y_star_g

    # =========================================================
    # IMPORTANT: Evaluate gray-box attack impact under TRUE params
    # (this is what you asked for)
    # =========================================================
    m_f_ag, P_f_ag, m_p_ag, P_p_ag = kalman_filter_nd(
        y=y_adv_g, u=u,
        A=A_true, B=B_true, H=H_true, D=D_true,   # <-- TRUE params here
        Q=Q_true, R=R_true,
        m0=m0, P0=P0
    )
    m_s_ag, P_s_ag = rts_smoother_nd(
        m_filt=m_f_ag, P_filt=P_f_ag,
        m_pred=m_p_ag, P_pred=P_p_ag,
        A=A_true
    )

    # =========================================================
    # Plot doubled "first-column" panels only
    # =========================================================
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"attack_overlay_onecol_t{t}_T{T}_seed{seed}.png")

    plot_attack_overlay_one_column(
    t=t,
    epsilon_w=epsilon_w,
    epsilon_g=epsilon_g,
    # geometry white
    y_t=y_t,
    mu_w=mu_w, Sigma_w=Sig_w, X_w=X_w,
    y_star_w=y_star_w, obj_w=obj_w,
    # geometry gray
    mu_g=mu_g, Sigma_g=Sig_g, X_g=X_g,
    y_star_g=y_star_g, obj_g=obj_g,
    # time series
    m_smooth_white=m_s_aw, P_smooth_white=P_s_aw,
    m_smooth_gray=m_s_ag,  P_smooth_gray=P_s_ag,
    x_true=x_true,
    outpath=outpath,
    )


    print(f"Saved figure: {outpath}")

import os as _os
import sys as _sys

# Make `shared_ssm` importable when this legacy script is run directly.
_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from shared_ssm.linalg import project_to_psd
from shared_ssm.linalg import spd_inverse as inv_psd
from shared_ssm.linalg import sqrtm_psd
from shared_ssm.linalg import symmetrize
from shared_ssm.legacy import kalman_filter_nd_const as kalman_filter_nd
from shared_ssm.legacy import loo_values_nd_const
from shared_ssm.legacy import rts_smoother_nd_const as rts_smoother_nd
from shared_ssm.legacy import simulate_lgssm_nd_const as simulate_lgssm_nd
from shared_ssm.legacy import solve_kkt_max_quadratic_over_ellipsoid


if __name__ == "__main__":
    main()
