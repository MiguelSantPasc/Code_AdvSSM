#!/usr/bin/env python3
"""
cartpole_covadapt_compare_epsilons_wolf.py

Benchmark `Gymnasium/CartPole-v1` under adversarial observation attacks using:
1. a nominal EKF-style defense,
2. covariance adaptation,
3. WoLF-IMQ,
4. WoLF-TMD.

Why this environment is a good fit for the repository:
1. `CartPole-v1` is the standard inverted-pendulum benchmark in Gymnasium.
2. The hidden dynamics are nonlinear. We keep the upright linearization as a
   reference SSM, but the actual predictor used by the defense is EKF-like:
   the nonlinear CartPole dynamics propagate the mean and a local Jacobian is
   recomputed at every step to propagate the covariance.
3. The policy observes the full 4D state
      s_t = [x_t, xdot_t, theta_t, thetadot_t],
   so the attacked quantity and the defended latent state live in the same
   coordinates, just like the 4D RL benchmark already present in the repo.

Reference inverted-pendulum SSM used here:
1. Continuous-time local dynamics around `theta = 0`:

      ds_t / dt = A_c s_t + B_c F_t + q_t
      o_t       = H s_t + r_t

   with `o_t = s_t + r_t` and, for the default Gymnasium parameters
   `M = 1.0`, `m = 0.1`, `l = 0.5`, `g = 9.8`,

      A_c =
      [[0, 1, 0, 0],
       [0, 0, -0.71707317, 0],
       [0, 0, 0, 1],
       [0, 0, 15.77560976, 0]]

      B_c =
      [[0],
       [0.97560976],
       [0],
       [-1.46341463]]

      H = I_4.

2. Gymnasium integrates CartPole with Euler step `tau = 0.02`, so the upright
   reference discrete model is

      s_{t+1} = A_d s_t + B_d F_t + q_t
      o_t     = H s_t + r_t

   where

      A_d = I + tau * A_c
      B_d = tau * B_c.

3. During filtering we do not keep `A_d` and `B_d` fixed. Instead, at each
   time step we:
   - propagate the posterior mean through the nonlinear CartPole dynamics,
   - build local Jacobians `A_t` and `B_t` by finite differences,
   - propagate the covariance with that local linearization.

Practical policy/critic note:
1. The script does not train anything.
2. It loads a pretrained DQN checkpoint downloaded from Hugging Face into
   `Gymnasium/outputs/saved_models/sb3_dqn_cartpole_v1/`.
3. The DQN `Q-network` acts as the critic. When the attack optimizes the
   expected value under the defended posterior, it minimizes

      E[max_a Q(s_t, a) | o_t', history].

Figure layout:
1. same three-panel grouped-bar layout as the RL covariance-adaptation
   comparison with WoLF,
2. same two-epsilon comparison using hatches,
3. no figure title and legends kept inside the axes.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from stable_baselines3 import DQN


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from AdvSSM.io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from CovarianceAdaptation.covariance_adaptation_utils import (
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
    from io_utils import cached_npz, data_path_for_plot, figures_dir_for
    from covariance_adaptation_utils import (
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


DEFAULT_GYMNASIUM_DISCOUNT_DELTA = 0.86


def normalized_final_reward(series: np.ndarray, *, n_episodes: int) -> float:
    """Return the final accumulated reward standardized by episode count."""
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return 0.0
    return float(series[-1] / float(n_episodes))


def darken_hex(hex_color: str, factor: float = 0.88) -> tuple[float, float, float]:
    """Return a slightly darker RGB color for bar edges and legends."""
    raw = hex_color.lstrip("#")
    rgb = tuple(int(raw[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))
    return tuple(max(0.0, min(1.0, factor * channel)) for channel in rgb)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def project_to_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone."""
    matrix = symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(np.float32)


def sqrtm_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return the PSD square root of a covariance matrix."""
    matrix = symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)


def mahalanobis_radius_sq(
    *,
    observation: np.ndarray,
    center: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return the ellipsoidal radius of one observation around a given center."""
    observation = np.asarray(observation, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    covariance = project_to_psd(np.asarray(covariance, dtype=float))
    delta = observation - center
    return float(np.dot(delta, solve_spd(covariance, delta)))


class SeparateObservationNoise(gym.ObservationWrapper):
    """
    Add observation noise with its own RNG so env randomness stays paired.

    This mirrors the RL benchmark pattern: the process randomness of the
    environment should be identical across clean, noisy, attacked, and defended
    rollouts when they reuse the same base seed.
    """

    def __init__(self, env: gym.Env, sigma: np.ndarray | float, seed: int):
        super().__init__(env)
        self.rng_obs = np.random.default_rng(int(seed))
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 0:
            sigma_arr = np.full(self.observation_space.shape, float(sigma_arr), dtype=float)
        self.sigma = sigma_arr.astype(np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        noise = self.rng_obs.normal(0.0, self.sigma, size=observation.shape).astype(np.float32)
        return (np.asarray(observation, dtype=np.float32) + noise).astype(np.float32)


@dataclass(frozen=True)
class CartPoleLinearSSM:
    """Container for the linearized CartPole matrices and physical constants."""

    A_c: np.ndarray
    B_c: np.ndarray
    A_d: np.ndarray
    B_d: np.ndarray
    H: np.ndarray
    tau: float
    gravity: float
    masscart: float
    masspole: float
    length: float
    force_mag: float


def build_cartpole_linear_ssm() -> CartPoleLinearSSM:
    """
    Build the local linearized SSM around the upright CartPole equilibrium.

    The coefficients are derived from the exact Gymnasium CartPole equations and
    then discretized with the same Euler step used by the environment.
    """
    env = gym.make("CartPole-v1")
    base_env = env.unwrapped

    tau = float(base_env.tau)
    gravity = float(base_env.gravity)
    masscart = float(base_env.masscart)
    masspole = float(base_env.masspole)
    length = float(base_env.length)
    force_mag = float(base_env.force_mag)
    env.close()

    total_mass = masscart + masspole
    denom = length * (4.0 / 3.0 - masspole / total_mass)

    a23 = -(masspole * gravity) / (total_mass * (4.0 / 3.0 - masspole / total_mass))
    a43 = gravity / denom
    b2 = (1.0 / total_mass) * (1.0 + (masspole / total_mass) / (4.0 / 3.0 - masspole / total_mass))
    b4 = -1.0 / (total_mass * denom)

    A_c = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, a23, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, a43, 0.0],
        ],
        dtype=np.float32,
    )
    B_c = np.array(
        [
            [0.0],
            [b2],
            [0.0],
            [b4],
        ],
        dtype=np.float32,
    )

    A_d = (np.eye(4, dtype=np.float32) + tau * A_c).astype(np.float32)
    B_d = (tau * B_c).astype(np.float32)
    H = np.eye(4, dtype=np.float32)
    return CartPoleLinearSSM(
        A_c=A_c,
        B_c=B_c,
        A_d=A_d,
        B_d=B_d,
        H=H,
        tau=tau,
        gravity=gravity,
        masscart=masscart,
        masspole=masspole,
        length=length,
        force_mag=force_mag,
    )


def print_cartpole_ssm_summary(ssm: CartPoleLinearSSM) -> None:
    """Print the upright reference SSM used to initialize the EKF intuition."""
    np.set_printoptions(precision=8, suppress=True)
    print("CartPole upright reference SSM (the EKF predictor re-linearizes locally)")
    print(f"tau = {ssm.tau:.4f}")
    print("A_c =")
    print(ssm.A_c)
    print("B_c =")
    print(ssm.B_c)
    print("A_d =")
    print(ssm.A_d)
    print("B_d =")
    print(ssm.B_d)
    print("H =")
    print(ssm.H)


def default_download_dir() -> str:
    """Return the pretrained-model directory used by this benchmark."""
    return os.path.join(CURRENT_DIR, "outputs", "saved_models", "sb3_dqn_cartpole_v1")


def ensure_downloaded_cartpole_checkpoint() -> str:
    """
    Ensure the pretrained CartPole checkpoint exists locally.

    The benchmark prefers the downloaded Hugging Face checkpoint. If downloading
    is unavailable, it falls back to the compatible local checkpoint already
    stored in `RL/others/outputs/saved_models/`.
    """
    download_dir = default_download_dir()
    os.makedirs(download_dir, exist_ok=True)

    checkpoint_path = os.path.join(download_dir, "dqn-CartPole-v1.zip")
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    files = {
        "dqn-CartPole-v1.zip": "https://huggingface.co/sb3/dqn-CartPole-v1/resolve/main/dqn-CartPole-v1.zip",
        "args.yml": "https://huggingface.co/sb3/dqn-CartPole-v1/resolve/main/args.yml",
        "config.yml": "https://huggingface.co/sb3/dqn-CartPole-v1/resolve/main/config.yml",
        "results.json": "https://huggingface.co/sb3/dqn-CartPole-v1/resolve/main/results.json",
    }

    try:
        for filename, url in files.items():
            target_path = os.path.join(download_dir, filename)
            if os.path.exists(target_path):
                continue
            print(f"[download] {url}")
            urllib.request.urlretrieve(url, target_path)
        return checkpoint_path
    except Exception as exc:
        print(f"[warning] Could not download Hugging Face checkpoint: {exc}")

    fallback_path = os.path.join(
        REPO_ROOT,
        "RL",
        "others",
        "outputs",
        "saved_models",
        "dqn_cartpole_clean.zip",
    )
    if os.path.exists(fallback_path):
        print(f"[fallback] Using local CartPole DQN checkpoint: {fallback_path}")
        return fallback_path

    raise FileNotFoundError(
        "No pretrained CartPole checkpoint is available. "
        "Expected either the downloaded Hugging Face file or the local fallback."
    )


def load_cartpole_policy(model_path: str, device: torch.device) -> DQN:
    """
    Load the pretrained DQN policy while bridging old `gym` checkpoints.

    The Hugging Face checkpoint was serialized against the older `gym` module,
    whereas this repository now uses `gymnasium`. Mapping `gym -> gymnasium`
    and replacing a few deserialized objects is enough to recover the policy
    and its critic without training anything.
    """
    sys.modules.setdefault("gym", gym)

    env = gym.make("CartPole-v1")
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "exploration_schedule": lambda _: 0.0,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    model = DQN.load(
        model_path,
        env=env,
        custom_objects=custom_objects,
        device=device,
    )
    return model


def select_action(model: DQN, obs: np.ndarray) -> int:
    """Return the deterministic greedy DQN action."""
    action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
    return int(np.asarray(action).item())


def action_to_force(action: int, force_mag: float) -> float:
    """Map the discrete CartPole action to the physical horizontal force."""
    return float(force_mag) if int(action) == 1 else -float(force_mag)


def build_filter_covariances(
    *,
    meas_std: np.ndarray,
    proc_std: np.ndarray,
    meas_corr: np.ndarray | None = None,
    proc_corr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build measurement and process covariances in 4D state coordinates.

    By default the noises are diagonal. If correlation matrices are provided,
    the covariance is built as `diag(std) @ Corr @ diag(std)`.
    """
    meas_std = np.asarray(meas_std, dtype=float).reshape(4)
    proc_std = np.asarray(proc_std, dtype=float).reshape(4)
    meas_scale = np.diag(meas_std).astype(np.float32)
    proc_scale = np.diag(proc_std).astype(np.float32)

    if meas_corr is None:
        R = np.diag(meas_std**2).astype(np.float32)
    else:
        meas_corr = symmetrize(np.asarray(meas_corr, dtype=float).reshape(4, 4))
        R = (meas_scale @ meas_corr @ meas_scale).astype(np.float32)

    if proc_corr is None:
        Q = np.diag(proc_std**2).astype(np.float32)
    else:
        proc_corr = symmetrize(np.asarray(proc_corr, dtype=float).reshape(4, 4))
        Q = (proc_scale @ proc_corr @ proc_scale).astype(np.float32)

    return project_to_psd(R), project_to_psd(Q)


def critic_state_values(model: DQN, obs_batch: torch.Tensor) -> torch.Tensor:
    """
    Return the DQN state value proxy `max_a Q(s, a)` for a batch of states.
    """
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.unsqueeze(0)
    q_values = model.q_net(obs_batch)
    return q_values.max(dim=1).values


def kf_update_state(
    *,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one KF update with `H = I_4` in CartPole state coordinates."""
    m_pred = np.asarray(m_pred, dtype=float).reshape(4)
    P_pred = project_to_psd(np.asarray(P_pred, dtype=float))
    y_obs = np.asarray(y_obs, dtype=float).reshape(4)
    R = project_to_psd(np.asarray(R, dtype=float))

    I4 = np.eye(4, dtype=float)
    S = project_to_psd(P_pred + R)
    K = solve_spd(S, P_pred.T).T
    innovation = y_obs - m_pred
    m_post = m_pred + K @ innovation

    joseph_left = I4 - K
    P_post = project_to_psd(joseph_left @ P_pred @ joseph_left.T + K @ R @ K.T)
    return m_post.astype(np.float32), P_post.astype(np.float32)


def kf_predict_state(
    *,
    m_post: np.ndarray,
    P_post: np.ndarray,
    force: float,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    discount_delta: float | None = None,
    ssm: CartPoleLinearSSM | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one prediction step for the hidden CartPole state.

    Two model structures are supported:
    1. If `ssm` is provided, use an EKF-style nonlinear prediction:
       - propagate the mean with the exact CartPole Euler step,
       - propagate the covariance with the local Jacobian.
    2. Otherwise fall back to the constant linear prediction specified by
       `A` and `B`.

    For the covariance itself, the predictor supports:
    1. a fixed additive process covariance `Q`,
    2. a discount rule `P_pred = (A_t P_post A_t^T) / discount_delta`.

    When both are provided, the discount inflation is applied first and `Q`
    acts only as an extra covariance floor on top of that discounted predictor.
    """
    m_post = np.asarray(m_post, dtype=float).reshape(4)
    P_post = project_to_psd(np.asarray(P_post, dtype=float))
    if Q is not None:
        Q = project_to_psd(np.asarray(Q, dtype=float))
    if discount_delta is not None and not (0.0 < float(discount_delta) <= 1.0):
        raise ValueError("discount_delta must satisfy 0 < discount_delta <= 1.")

    def apply_predictive_uncertainty(transition_cov: np.ndarray) -> np.ndarray:
        """
        Add the configured uncertainty to the propagated covariance.

        The shared Gymnasium default uses the globally calibrated fixed
        discount chosen from the CartPole NIS sweep. Older `Q`-based scripts
        can still add an explicit covariance floor when needed.
        """
        P_next_local = project_to_psd(np.asarray(transition_cov, dtype=float))

        if discount_delta is not None:
            P_next_local = project_to_psd(P_next_local / float(discount_delta))

        if Q is not None:
            P_next_local = project_to_psd(P_next_local + Q)

        return P_next_local.astype(np.float32)

    if ssm is not None:
        m_next = cartpole_discrete_dynamics(
            state=m_post,
            force=force,
            ssm=ssm,
        )
        A_local, _B_local = linearize_cartpole_discrete_dynamics(
            state=m_post,
            force=force,
            ssm=ssm,
        )
        transition_cov = project_to_psd(A_local @ P_post @ A_local.T)
        P_next = apply_predictive_uncertainty(transition_cov)
        return m_next.astype(np.float32), P_next.astype(np.float32)

    if A is None or B is None:
        raise ValueError("Either provide `ssm` for EKF prediction or provide both `A` and `B`.")

    A = np.asarray(A, dtype=float).reshape(4, 4)
    B = np.asarray(B, dtype=float).reshape(4, 1)
    u = np.array([float(force)], dtype=float)
    m_next = A @ m_post + B @ u
    transition_cov = project_to_psd(A @ P_post @ A.T)
    P_next = apply_predictive_uncertainty(transition_cov)
    return m_next.astype(np.float32), P_next.astype(np.float32)


def cartpole_continuous_accelerations(
    *,
    state: np.ndarray,
    force: float,
    ssm: CartPoleLinearSSM,
) -> tuple[float, float]:
    """
    Return the exact CartPole accelerations used by Gymnasium.

    The observation is the full state, so only the state-transition Jacobian
    needs to adapt online; the measurement matrix stays equal to `I_4`.
    """
    x_pos, xdot, theta, thetadot = np.asarray(state, dtype=float).reshape(4)
    _ = x_pos

    total_mass = float(ssm.masscart + ssm.masspole)
    polemass_length = float(ssm.masspole * ssm.length)
    sintheta = float(np.sin(theta))
    costheta = float(np.cos(theta))

    temp = (float(force) + polemass_length * (thetadot**2) * sintheta) / total_mass
    denom = float(ssm.length * (4.0 / 3.0 - (ssm.masspole * costheta**2) / total_mass))
    thetaacc = (float(ssm.gravity) * sintheta - costheta * temp) / denom
    xacc = temp - (polemass_length * thetaacc * costheta) / total_mass
    return float(xacc), float(thetaacc)


def cartpole_discrete_dynamics(
    *,
    state: np.ndarray,
    force: float,
    ssm: CartPoleLinearSSM,
) -> np.ndarray:
    """
    Propagate one CartPole state with the same Euler step as Gymnasium.
    """
    x_pos, xdot, theta, thetadot = np.asarray(state, dtype=float).reshape(4)
    xacc, thetaacc = cartpole_continuous_accelerations(
        state=np.array([x_pos, xdot, theta, thetadot], dtype=float),
        force=force,
        ssm=ssm,
    )
    tau = float(ssm.tau)
    x_next = x_pos + tau * xdot
    xdot_next = xdot + tau * xacc
    theta_next = theta + tau * thetadot
    thetadot_next = thetadot + tau * thetaacc
    return np.array([x_next, xdot_next, theta_next, thetadot_next], dtype=np.float32)


def linearize_cartpole_discrete_dynamics(
    *,
    state: np.ndarray,
    force: float,
    ssm: CartPoleLinearSSM,
    state_eps: float = 1e-5,
    force_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the local discrete-time Jacobians `(A_t, B_t)` for CartPole.

    We use centered finite differences around the current posterior mean. This
    keeps the implementation short, robust, and faithful to the exact
    nonlinear dynamics used by Gymnasium.
    """
    state = np.asarray(state, dtype=float).reshape(4)
    A_local = np.zeros((4, 4), dtype=np.float32)

    for dim_idx in range(4):
        step = np.zeros((4,), dtype=float)
        step[dim_idx] = float(state_eps)
        f_plus = cartpole_discrete_dynamics(
            state=state + step,
            force=force,
            ssm=ssm,
        ).astype(np.float64)
        f_minus = cartpole_discrete_dynamics(
            state=state - step,
            force=force,
            ssm=ssm,
        ).astype(np.float64)
        A_local[:, dim_idx] = ((f_plus - f_minus) / (2.0 * float(state_eps))).astype(np.float32)

    f_u_plus = cartpole_discrete_dynamics(
        state=state,
        force=float(force) + float(force_eps),
        ssm=ssm,
    ).astype(np.float64)
    f_u_minus = cartpole_discrete_dynamics(
        state=state,
        force=float(force) - float(force_eps),
        ssm=ssm,
    ).astype(np.float64)
    B_local = (((f_u_plus - f_u_minus) / (2.0 * float(force_eps))).reshape(4, 1)).astype(np.float32)
    return A_local, B_local


def covariance_adapted_kf_update_state(
    *,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
    adv_target: np.ndarray | None,
    c_scale: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    direction_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | np.ndarray]]:
    """
    Run one 4D state-space KF update with online covariance adaptation.

    The hidden defended quantity is the CartPole state itself, and the attacked
    observation also lives in those same coordinates, so the measurement model
    is simply `o_t = s_t + r_t`.
    """
    m_pred = np.asarray(m_pred, dtype=float).reshape(4)
    P_pred = project_to_psd(np.asarray(P_pred, dtype=float))
    y_obs = np.asarray(y_obs, dtype=float).reshape(4)
    R = project_to_psd(np.asarray(R, dtype=float))

    I4 = np.eye(4, dtype=float)
    y_hat = m_pred.copy()
    S_nom = project_to_psd(P_pred + R)
    innovation = y_obs - y_hat

    V_tilde = R.copy()
    S_tilde = S_nom.copy()
    K_tilde = solve_spd(S_tilde, P_pred.T).T
    pi_t = 0.0
    gamma_t = 0.0
    bar_gamma_t = 0.0
    lambda_t = 0.0
    u_dir = np.zeros(4, dtype=float)

    if adv_target is not None:
        adv_target = np.asarray(adv_target, dtype=float).reshape(4)
        delta_adv = adv_target - y_hat
        u_dir, delta_norm = safe_unit_direction(delta_adv, eps=direction_eps)

        if delta_norm >= direction_eps:
            lambda_t = float(c_scale) * float(np.max(np.linalg.eigvalsh(S_nom)))
            pi_t, _, _ = compute_contamination_prior(
                delta_adv=delta_adv,
                S_t=S_nom,
                P_pred_t=P_pred,
                H_t=I4,
                omega_h=omega_h,
                omega_o=omega_o,
            )

            S_adv = rank_one_covariance_update(S_nom, lambda_t, u_dir)
            precision_poe = spd_inverse(S_adv) + spd_inverse(R)
            Sigma_poe = spd_inverse(precision_poe)
            rhs_poe = solve_spd(S_adv, y_hat) + solve_spd(R, adv_target)
            mu_poe = solve_spd(precision_poe, rhs_poe)

            log_p0 = gaussian_logpdf(y_obs, y_hat, S_nom)
            log_p1 = gaussian_logpdf(y_obs, mu_poe, Sigma_poe)
            gamma_t = log_mix_posterior_weight(pi_t, log_p0, log_p1)
            bar_gamma_t = gamma_t if gamma_t >= delta_threshold else 0.0

            V_tilde = rank_one_covariance_update(R, lambda_t, u_dir, weight=bar_gamma_t)
            S_tilde = rank_one_covariance_update(S_nom, lambda_t, u_dir, weight=bar_gamma_t)
            K_tilde = solve_spd(S_tilde, P_pred.T).T

    m_post = m_pred + K_tilde @ innovation

    joseph_left = I4 - K_tilde
    P_post = project_to_psd(joseph_left @ P_pred @ joseph_left.T + K_tilde @ V_tilde @ K_tilde.T)

    diagnostics = {
        "pi_t": float(pi_t),
        "gamma_t": float(gamma_t),
        "bar_gamma_t": float(bar_gamma_t),
        "lambda_t": float(lambda_t),
        "innovation_norm": float(np.linalg.norm(innovation)),
        "direction": u_dir.astype(np.float32),
    }
    return m_post.astype(np.float32), P_post.astype(np.float32), diagnostics


def wolf_imq_weight_squared(
    innovation: np.ndarray,
    *,
    soft_threshold: float,
    min_weight: float = 1e-6,
) -> float:
    """Return the WoLF-IMQ observation weight `w_t^2`."""
    if soft_threshold <= 0.0:
        raise ValueError("soft_threshold must be positive.")

    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    threshold_sq = float(soft_threshold) ** 2
    weight_sq = threshold_sq / (threshold_sq + float(np.dot(innovation, innovation)))
    return float(max(weight_sq, min_weight))


def wolf_tmd_weight_squared(
    innovation: np.ndarray,
    *,
    innovation_covariance: np.ndarray,
    threshold: float,
    min_weight: float = 1e-6,
) -> float:
    """Return the WoLF-TMD observation weight `w_t^2`."""
    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")

    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    innovation_covariance = project_to_psd(np.asarray(innovation_covariance, dtype=float))
    mahal_sq = float(np.dot(innovation, solve_spd(innovation_covariance, innovation)))
    mahal = float(np.sqrt(max(mahal_sq, 0.0)))
    weight_sq = 1.0 if mahal < float(threshold) else min_weight
    return float(weight_sq)


def wolf_kf_update_state(
    *,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    y_obs: np.ndarray,
    R: np.ndarray,
    wolf_kind: str,
    imq_soft_threshold: float,
    tmd_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run one 4D state-space KF update with a WoLF weighting rule."""
    m_pred = np.asarray(m_pred, dtype=float).reshape(4)
    P_pred = project_to_psd(np.asarray(P_pred, dtype=float))
    y_obs = np.asarray(y_obs, dtype=float).reshape(4)
    R = project_to_psd(np.asarray(R, dtype=float))

    innovation = y_obs - m_pred
    innovation_covariance = project_to_psd(P_pred + R)

    if wolf_kind == "imq":
        weight_sq = wolf_imq_weight_squared(
            innovation,
            soft_threshold=imq_soft_threshold,
        )
    elif wolf_kind == "tmd":
        weight_sq = wolf_tmd_weight_squared(
            innovation,
            innovation_covariance=innovation_covariance,
            threshold=tmd_threshold,
        )
    else:
        raise ValueError(f"Unsupported wolf_kind: {wolf_kind}")

    effective_R = project_to_psd(R / float(weight_sq))
    S_eff = project_to_psd(P_pred + effective_R)
    K_eff = solve_spd(S_eff, P_pred.T).T
    m_post = m_pred + K_eff @ innovation

    I4 = np.eye(4, dtype=float)
    joseph_left = I4 - K_eff
    P_post = project_to_psd(joseph_left @ P_pred @ joseph_left.T + K_eff @ effective_R @ K_eff.T)

    diagnostics = {
        "weight_sq": float(weight_sq),
        "innovation_norm": float(np.linalg.norm(innovation)),
    }
    return m_post.astype(np.float32), P_post.astype(np.float32), diagnostics


def sample_boundary_attack(
    *,
    center: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one point on the boundary of an ellipsoid in 4D observation space."""
    center = np.asarray(center, dtype=float).reshape(4)
    Sigma = project_to_psd(np.asarray(Sigma, dtype=float))
    direction = rng.normal(0.0, 1.0, size=(4,))
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    delta = np.sqrt(float(epsilon)) * (sqrtm_psd(Sigma) @ direction)
    return (center + delta).astype(np.float32)


def sample_random_attack_in_ellipsoid(
    *,
    center: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample one random point inside the attack ellipsoid.

    The sampled observation satisfies
        (y-center)^T Sigma^{-1} (y-center) <= epsilon
    up to numerical tolerance.

    The construction is:
    1. sample a random direction on the 4D sphere,
    2. sample a random radius with the correct `r^(d-1)` volume scaling,
    3. map that isotropic sample through `sqrtm(Sigma)` and the confidence
       radius `sqrt(epsilon)`.
    """
    center = np.asarray(center, dtype=float).reshape(4)
    Sigma = project_to_psd(np.asarray(Sigma, dtype=float))
    direction = rng.normal(0.0, 1.0, size=(4,))
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    radial_scale = float(rng.random()) ** (1.0 / 4.0)
    delta = radial_scale * np.sqrt(float(epsilon)) * (sqrtm_psd(Sigma) @ direction)
    attacked_candidate = center + delta
    return project_to_attack_region(
        y_candidate=attacked_candidate,
        center=center,
        Sigma=Sigma,
        epsilon=epsilon,
    )


def project_to_attack_region(
    *,
    y_candidate: np.ndarray,
    center: np.ndarray,
    Sigma: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Project one observation onto the attack ellipsoid.

    The attack set is:
        { y : (y-center)^T Sigma^{-1} (y-center) <= epsilon }.
    """
    y_candidate = np.asarray(y_candidate, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    Sigma = project_to_psd(np.asarray(Sigma, dtype=float))

    Sigma_inv = np.linalg.inv(Sigma.astype(np.float64))
    delta = y_candidate - center
    maha = float(delta.T @ Sigma_inv @ delta)
    if maha <= float(epsilon):
        return y_candidate.astype(np.float32)

    scale = np.sqrt(float(epsilon) / max(maha, 1e-12))
    return (center + scale * delta).astype(np.float32)


def expected_critic_value_mc(
    *,
    model: DQN,
    obs_adv_torch: torch.Tensor,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    xi_torch: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Return the posterior expected critic value induced by an attacked state observation.
    """
    dev = obs_adv_torch.device

    m_pred_t = torch.tensor(m_pred, dtype=torch.float32, device=dev)
    P_pred_t = torch.tensor(P_pred, dtype=torch.float32, device=dev)
    R_t = torch.tensor(R, dtype=torch.float32, device=dev)

    S_t = P_pred_t + R_t
    K_t = P_pred_t @ torch.linalg.inv(S_t)
    innovation_t = obs_adv_torch - m_pred_t
    m_post_t = m_pred_t + K_t @ innovation_t

    I4 = torch.eye(4, dtype=torch.float32, device=dev)
    P_post_t = (I4 - K_t) @ P_pred_t @ (I4 - K_t).T + K_t @ R_t @ K_t.T

    P_post_np = project_to_psd(P_post_t.detach().cpu().numpy())
    L_np = sqrtm_psd(P_post_np)
    L_t = torch.tensor(L_np, dtype=torch.float32, device=dev)

    x_samples = m_post_t.unsqueeze(0) + xi_torch @ L_t.T
    value_batch = critic_state_values(model, x_samples)
    mu_value = value_batch.mean()

    return mu_value, m_post_t.detach().cpu().numpy().astype(np.float32), P_post_np


def pgd_attack_on_expected_value(
    *,
    model: DQN,
    obs_nom: np.ndarray,
    m_pred: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    attack_center: np.ndarray,
    attack_sigma: np.ndarray,
    attack_eps: float,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    rng_seed: int,
    device: str = "cpu",
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Approximately solve the worst-case observation attack inside the ellipsoid.
    """
    dev = torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(rng_seed))
    xi_torch = torch.randn((mc_samples, 4), generator=gen, device=dev, dtype=torch.float32)

    center = np.asarray(attack_center, dtype=np.float32).copy()
    obs_curr_np = project_to_attack_region(
        y_candidate=np.asarray(obs_nom, dtype=np.float32),
        center=center,
        Sigma=attack_sigma,
        epsilon=attack_eps,
    )

    best_obs = obs_curr_np.copy()
    best_obj = None
    best_m_post = None
    best_P_post = None

    for _ in range(pgd_steps):
        obs_t = torch.tensor(obs_curr_np, dtype=torch.float32, device=dev, requires_grad=True)
        mu_value_t, m_post_np, P_post_np = expected_critic_value_mc(
            model=model,
            obs_adv_torch=obs_t,
            m_pred=m_pred,
            P_pred=P_pred,
            R=R,
            xi_torch=xi_torch,
        )

        mu_value_t.backward()
        grad = obs_t.grad.detach().cpu().numpy().astype(np.float32)
        obs_next = obs_curr_np - float(pgd_step_size) * grad

        obs_next = project_to_attack_region(
            y_candidate=obs_next,
            center=center,
            Sigma=attack_sigma,
            epsilon=attack_eps,
        )

        obj_val = float(mu_value_t.detach().cpu().item())
        if best_obj is None or obj_val < best_obj:
            best_obj = obj_val
            best_obs = obs_curr_np.copy()
            best_m_post = m_post_np.copy()
            best_P_post = P_post_np.copy()

        obs_curr_np = obs_next.astype(np.float32)

    obs_t = torch.tensor(obs_curr_np, dtype=torch.float32, device=dev, requires_grad=True)
    mu_value_t, m_post_np, P_post_np = expected_critic_value_mc(
        model=model,
        obs_adv_torch=obs_t,
        m_pred=m_pred,
        P_pred=P_pred,
        R=R,
        xi_torch=xi_torch,
    )
    obj_val = float(mu_value_t.detach().cpu().item())

    if best_obj is None or obj_val < best_obj:
        best_obj = obj_val
        best_obs = obs_curr_np.copy()
        best_m_post = m_post_np.copy()
        best_P_post = P_post_np.copy()

    return best_obs, float(best_obj), best_m_post, best_P_post


@torch.no_grad()
def rollout_episode_return_clean(
    env: gym.Env,
    model: DQN,
    *,
    seed: int,
) -> float:
    """Evaluate the clean pretrained policy without any defense layer."""
    obs, _info = env.reset(seed=int(seed))
    ep_return = 0.0

    while True:
        action = select_action(model, obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        ep_return += float(reward)
        if terminated or truncated:
            break

    return float(ep_return)


@torch.no_grad()
def rollout_episode_return_noisy_kf(
    env: gym.Env,
    model: DQN,
    *,
    seed: int,
    ssm: CartPoleLinearSSM,
    R: np.ndarray,
    discount_delta: float,
) -> float:
    """Evaluate under noisy observations with the shared discounted predictor."""
    obs, _info = env.reset(seed=int(seed))

    m_pred = np.asarray(obs, dtype=np.float32).copy()
    P_pred = project_to_psd(R.copy())
    ep_return = 0.0

    while True:
        m_post, P_post = kf_update_state(
            m_pred=m_pred,
            P_pred=P_pred,
            y_obs=obs,
            R=R,
        )
        action = select_action(model, m_post)
        force = action_to_force(action, ssm.force_mag)
        m_pred, P_pred = kf_predict_state(
            m_post=m_post,
            P_post=P_post,
            force=force,
            Q=None,
            discount_delta=discount_delta,
            ssm=ssm,
        )

        obs, reward, terminated, truncated, _info = env.step(action)
        ep_return += float(reward)
        if terminated or truncated:
            break

    return float(ep_return)


def rollout_episode_return_attacked(
    env: gym.Env,
    model: DQN,
    *,
    seed: int,
    ssm: CartPoleLinearSSM,
    R: np.ndarray,
    discount_delta: float,
    obs_noise_std: np.ndarray,
    attack_eps: float,
    attack_prob: float,
    attack_mode: str,
    defense: str,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    c_scale: float,
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    wolf_kind: str,
    wolf_imq_soft_threshold: float,
    wolf_tmd_threshold: float,
    device: str = "cpu",
) -> tuple[float, dict[str, int]]:
    """
    Roll out one episode with probabilistic attacks and one chosen defense.

    Protocol matched to the RL benchmark:
    1. the initial observation is never attacked but it is noised,
    2. from the second step onward, attacks happen with probability `attack_prob`,
    3. when no attack happens, the filter sees the same Gaussian noisy observation
       as the noisy baseline,
    4. when an attack happens:
       - the PGD attack starts from that same noisy observation,
       - the random baseline samples a random point from the attack ellipsoid.
    """
    obs, _info = env.reset(seed=int(seed))
    fallback_stats = {
        "attack_attempts": 0,
        "fallback_to_real": 0,
    }

    rng_attack_gate = np.random.default_rng(int(seed) + 707_001)
    rng_nominal_noise = np.random.default_rng(int(seed) + 707_002)
    rng_boundary = np.random.default_rng(int(seed) + 707_003)
    init_noise = rng_nominal_noise.normal(0.0, obs_noise_std, size=(4,)).astype(np.float32)
    obs_init_noisy = (np.asarray(obs, dtype=np.float32) + init_noise).astype(np.float32)

    m_post, P_post = kf_update_state(
        m_pred=np.asarray(obs, dtype=np.float32),
        P_pred=project_to_psd(R.copy()),
        y_obs=obs_init_noisy,
        R=R,
    )

    ep_return = 0.0
    step_idx = 0

    action = select_action(model, m_post)
    force = action_to_force(action, ssm.force_mag)
    m_pred, P_pred = kf_predict_state(
        m_post=m_post,
        P_post=P_post,
        force=force,
        Q=None,
        discount_delta=discount_delta,
        ssm=ssm,
    )

    obs, reward, terminated, truncated, _info = env.step(action)
    ep_return += float(reward)
    if terminated or truncated:
        return float(ep_return), fallback_stats

    step_idx = 1

    while True:
        y_clean = np.asarray(obs, dtype=np.float32)
        do_attack = bool(rng_attack_gate.random() < float(attack_prob))
        nominal_noise = rng_nominal_noise.normal(0.0, obs_noise_std, size=(4,)).astype(np.float32)
        y_noisy = (y_clean + nominal_noise).astype(np.float32)
        attack_center = np.asarray(m_pred, dtype=np.float32)
        attack_sigma = project_to_psd(np.asarray(P_pred, dtype=float) + np.asarray(R, dtype=float))

        if do_attack:
            fallback_stats["attack_attempts"] += 1
            if attack_mode == "pgd":
                y_used, _obj_star, m_post_attack, P_post_attack = pgd_attack_on_expected_value(
                    model=model,
                    obs_nom=y_noisy,
                    m_pred=m_pred,
                    P_pred=P_pred,
                    R=R,
                    attack_center=attack_center,
                    attack_sigma=attack_sigma,
                    attack_eps=attack_eps,
                    pgd_steps=pgd_steps,
                    pgd_step_size=pgd_step_size,
                    mc_samples=mc_samples,
                    rng_seed=int(seed) + 10_000 * step_idx,
                    device=device,
                )
            elif attack_mode == "random":
                y_used = sample_random_attack_in_ellipsoid(
                    center=attack_center,
                    Sigma=attack_sigma,
                    epsilon=attack_eps,
                    rng=rng_boundary,
                )
                m_post_attack = None
                P_post_attack = None
            else:
                raise ValueError(f"Unsupported attack_mode: {attack_mode}")

            eps_real = mahalanobis_radius_sq(
                observation=y_noisy,
                center=attack_center,
                covariance=attack_sigma,
            )
            eps_attack = mahalanobis_radius_sq(
                observation=y_used,
                center=attack_center,
                covariance=attack_sigma,
            )
            use_real_observation = bool(eps_attack < eps_real)
            if use_real_observation:
                fallback_stats["fallback_to_real"] += 1
                y_used = y_noisy.copy()
                m_post_attack = None
                P_post_attack = None
        else:
            use_real_observation = True

        if use_real_observation:
            if defense == "kf":
                m_post, P_post = kf_update_state(
                    m_pred=m_pred,
                    P_pred=P_pred,
                    y_obs=y_noisy,
                    R=R,
                )
            elif defense == "covadapt":
                m_post, P_post, _diag = covariance_adapted_kf_update_state(
                    m_pred=m_pred,
                    P_pred=P_pred,
                    y_obs=y_noisy,
                    R=R,
                    adv_target=None,
                    c_scale=c_scale,
                    omega_h=omega_h,
                    omega_o=omega_o,
                    delta_threshold=delta_threshold,
                )
            elif defense == "wolf":
                m_post, P_post, _diag = wolf_kf_update_state(
                    m_pred=m_pred,
                    P_pred=P_pred,
                    y_obs=y_noisy,
                    R=R,
                    wolf_kind=wolf_kind,
                    imq_soft_threshold=wolf_imq_soft_threshold,
                    tmd_threshold=wolf_tmd_threshold,
                )
            else:
                raise ValueError(f"Unsupported defense: {defense}")
        else:
            if defense == "kf":
                if attack_mode == "pgd" and m_post_attack is not None and P_post_attack is not None:
                    m_post = m_post_attack.astype(np.float32)
                    P_post = P_post_attack.astype(np.float32)
                else:
                    m_post, P_post = kf_update_state(
                        m_pred=m_pred,
                        P_pred=P_pred,
                        y_obs=y_used,
                        R=R,
                    )
            elif defense == "covadapt":
                m_post, P_post, _diag = covariance_adapted_kf_update_state(
                    m_pred=m_pred,
                    P_pred=P_pred,
                    y_obs=y_used,
                    R=R,
                    adv_target=y_used,
                    c_scale=c_scale,
                    omega_h=omega_h,
                    omega_o=omega_o,
                    delta_threshold=delta_threshold,
                )
            elif defense == "wolf":
                m_post, P_post, _diag = wolf_kf_update_state(
                    m_pred=m_pred,
                    P_pred=P_pred,
                    y_obs=y_used,
                    R=R,
                    wolf_kind=wolf_kind,
                    imq_soft_threshold=wolf_imq_soft_threshold,
                    tmd_threshold=wolf_tmd_threshold,
                )
            else:
                raise ValueError(f"Unsupported defense: {defense}")

        action = select_action(model, m_post)
        force = action_to_force(action, ssm.force_mag)
        m_pred, P_pred = kf_predict_state(
            m_post=m_post,
            P_post=P_post,
            force=force,
            Q=None,
            discount_delta=discount_delta,
            ssm=ssm,
        )

        obs, reward, terminated, truncated, _info = env.step(action)
        ep_return += float(reward)
        step_idx += 1

        if terminated or truncated:
            break

    return float(ep_return), fallback_stats


def compute_accumulated_reward_data_with_wolf(
    *,
    n_episodes: int,
    seed0: int,
    model_path: str,
    obs_noise_std: np.ndarray,
    attack_eps: float,
    attack_prob: float,
    discount_delta: float,
    kf_meas_std: np.ndarray,
    kf_proc_std: np.ndarray,
    kf_meas_corr: np.ndarray | None,
    kf_proc_corr: np.ndarray | None,
    pgd_steps: int,
    pgd_step_size: float,
    mc_samples: int,
    c_scales: tuple[float, ...],
    omega_h: float,
    omega_o: float,
    delta_threshold: float,
    wolf_imq_soft_threshold: float,
    wolf_tmd_threshold: float,
    device: str,
) -> dict[str, np.ndarray | float | int]:
    """Compute all accumulated-reward curves for one CartPole attack radius."""
    if not np.isclose(omega_h + omega_o, 1.0, atol=1e-9):
        raise ValueError("omega_h and omega_o must sum to 1.")
    if not (0.0 <= delta_threshold <= 1.0):
        raise ValueError("delta_threshold must lie in [0, 1].")
    if wolf_imq_soft_threshold <= 0.0:
        raise ValueError("wolf_imq_soft_threshold must be positive.")
    if wolf_tmd_threshold <= 0.0:
        raise ValueError("wolf_tmd_threshold must be positive.")

    device_t = torch.device(device)
    model = load_cartpole_policy(model_path, device_t)
    ssm = build_cartpole_linear_ssm()
    R, legacy_Q = build_filter_covariances(
        meas_std=kf_meas_std,
        proc_std=kf_proc_std,
        meas_corr=kf_meas_corr,
        proc_corr=kf_proc_corr,
    )

    totals: dict[str, float] = {
        "clean": 0.0,
        "noisy_kf": 0.0,
        "attack_kf": 0.0,
        "random_kf": 0.0,
        "attack_wolf_imq": 0.0,
        "attack_wolf_tmd": 0.0,
        "random_wolf_imq": 0.0,
        "random_wolf_tmd": 0.0,
    }
    for c_scale in c_scales:
        totals[f"attack_cov_{c_scale:g}"] = 0.0
        totals[f"random_cov_{c_scale:g}"] = 0.0

    fallback_by_mode = {
        "pgd": {"attack_attempts": 0, "fallback_to_real": 0},
        "random": {"attack_attempts": 0, "fallback_to_real": 0},
    }

    series: dict[str, list[float]] = {key: [] for key in totals}

    for episode_idx in range(n_episodes):
        seed = int(seed0 + episode_idx)

        env_clean = gym.make("CartPole-v1")
        env_noisy = SeparateObservationNoise(
            gym.make("CartPole-v1"),
            sigma=obs_noise_std,
            seed=seed + 25_000,
        )
        env_attack = gym.make("CartPole-v1")
        env_random = gym.make("CartPole-v1")
        env_attack_wolf_imq = gym.make("CartPole-v1")
        env_attack_wolf_tmd = gym.make("CartPole-v1")
        env_random_wolf_imq = gym.make("CartPole-v1")
        env_random_wolf_tmd = gym.make("CartPole-v1")

        ret_clean = rollout_episode_return_clean(
            env_clean,
            model,
            seed=seed,
        )
        ret_noisy = rollout_episode_return_noisy_kf(
            env_noisy,
            model,
            seed=seed,
            ssm=ssm,
            R=R,
            discount_delta=discount_delta,
        )
        ret_attack, stats_attack = rollout_episode_return_attacked(
            env_attack,
            model,
            seed=seed,
            ssm=ssm,
            R=R,
            discount_delta=discount_delta,
            obs_noise_std=np.asarray(kf_meas_std, dtype=float),
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            attack_mode="pgd",
            defense="kf",
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            mc_samples=mc_samples,
            c_scale=0.0,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            wolf_kind="imq",
            wolf_imq_soft_threshold=wolf_imq_soft_threshold,
            wolf_tmd_threshold=wolf_tmd_threshold,
            device=device,
        )
        ret_random, stats_random = rollout_episode_return_attacked(
            env_random,
            model,
            seed=seed,
            ssm=ssm,
            R=R,
            discount_delta=discount_delta,
            obs_noise_std=np.asarray(kf_meas_std, dtype=float),
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            attack_mode="random",
            defense="kf",
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            mc_samples=mc_samples,
            c_scale=0.0,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            wolf_kind="imq",
            wolf_imq_soft_threshold=wolf_imq_soft_threshold,
            wolf_tmd_threshold=wolf_tmd_threshold,
            device=device,
        )
        ret_attack_wolf_imq, stats_attack_wolf_imq = rollout_episode_return_attacked(
            env_attack_wolf_imq,
            model,
            seed=seed,
            ssm=ssm,
            R=R,
            discount_delta=discount_delta,
            obs_noise_std=np.asarray(kf_meas_std, dtype=float),
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            attack_mode="pgd",
            defense="wolf",
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            mc_samples=mc_samples,
            c_scale=0.0,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            wolf_kind="imq",
            wolf_imq_soft_threshold=wolf_imq_soft_threshold,
            wolf_tmd_threshold=wolf_tmd_threshold,
            device=device,
        )
        ret_attack_wolf_tmd, stats_attack_wolf_tmd = rollout_episode_return_attacked(
            env_attack_wolf_tmd,
            model,
            seed=seed,
            ssm=ssm,
            R=R,
            discount_delta=discount_delta,
            obs_noise_std=np.asarray(kf_meas_std, dtype=float),
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            attack_mode="pgd",
            defense="wolf",
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            mc_samples=mc_samples,
            c_scale=0.0,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            wolf_kind="tmd",
            wolf_imq_soft_threshold=wolf_imq_soft_threshold,
            wolf_tmd_threshold=wolf_tmd_threshold,
            device=device,
        )
        ret_random_wolf_imq, stats_random_wolf_imq = rollout_episode_return_attacked(
            env_random_wolf_imq,
            model,
            seed=seed,
            ssm=ssm,
            R=R,
            discount_delta=discount_delta,
            obs_noise_std=np.asarray(kf_meas_std, dtype=float),
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            attack_mode="random",
            defense="wolf",
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            mc_samples=mc_samples,
            c_scale=0.0,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            wolf_kind="imq",
            wolf_imq_soft_threshold=wolf_imq_soft_threshold,
            wolf_tmd_threshold=wolf_tmd_threshold,
            device=device,
        )
        ret_random_wolf_tmd, stats_random_wolf_tmd = rollout_episode_return_attacked(
            env_random_wolf_tmd,
            model,
            seed=seed,
            ssm=ssm,
            R=R,
            discount_delta=discount_delta,
            obs_noise_std=np.asarray(kf_meas_std, dtype=float),
            attack_eps=attack_eps,
            attack_prob=attack_prob,
            attack_mode="random",
            defense="wolf",
            pgd_steps=pgd_steps,
            pgd_step_size=pgd_step_size,
            mc_samples=mc_samples,
            c_scale=0.0,
            omega_h=omega_h,
            omega_o=omega_o,
            delta_threshold=delta_threshold,
            wolf_kind="tmd",
            wolf_imq_soft_threshold=wolf_imq_soft_threshold,
            wolf_tmd_threshold=wolf_tmd_threshold,
            device=device,
        )

        totals["clean"] += float(ret_clean)
        totals["noisy_kf"] += float(ret_noisy)
        totals["attack_kf"] += float(ret_attack)
        totals["random_kf"] += float(ret_random)
        totals["attack_wolf_imq"] += float(ret_attack_wolf_imq)
        totals["attack_wolf_tmd"] += float(ret_attack_wolf_tmd)
        totals["random_wolf_imq"] += float(ret_random_wolf_imq)
        totals["random_wolf_tmd"] += float(ret_random_wolf_tmd)

        for stats_block in (stats_attack, stats_attack_wolf_imq, stats_attack_wolf_tmd):
            fallback_by_mode["pgd"]["attack_attempts"] += int(stats_block["attack_attempts"])
            fallback_by_mode["pgd"]["fallback_to_real"] += int(stats_block["fallback_to_real"])
        for stats_block in (stats_random, stats_random_wolf_imq, stats_random_wolf_tmd):
            fallback_by_mode["random"]["attack_attempts"] += int(stats_block["attack_attempts"])
            fallback_by_mode["random"]["fallback_to_real"] += int(stats_block["fallback_to_real"])

        series["clean"].append(totals["clean"])
        series["noisy_kf"].append(totals["noisy_kf"])
        series["attack_kf"].append(totals["attack_kf"])
        series["random_kf"].append(totals["random_kf"])
        series["attack_wolf_imq"].append(totals["attack_wolf_imq"])
        series["attack_wolf_tmd"].append(totals["attack_wolf_tmd"])
        series["random_wolf_imq"].append(totals["random_wolf_imq"])
        series["random_wolf_tmd"].append(totals["random_wolf_tmd"])

        env_clean.close()
        env_noisy.close()
        env_attack.close()
        env_random.close()
        env_attack_wolf_imq.close()
        env_attack_wolf_tmd.close()
        env_random_wolf_imq.close()
        env_random_wolf_tmd.close()

        for c_scale in c_scales:
            env_attack_cov = gym.make("CartPole-v1")
            env_random_cov = gym.make("CartPole-v1")

            ret_attack_cov, stats_attack_cov = rollout_episode_return_attacked(
                env_attack_cov,
                model,
                seed=seed,
                ssm=ssm,
                R=R,
                discount_delta=discount_delta,
                obs_noise_std=np.asarray(kf_meas_std, dtype=float),
                attack_eps=attack_eps,
                attack_prob=attack_prob,
                attack_mode="pgd",
                defense="covadapt",
                pgd_steps=pgd_steps,
                pgd_step_size=pgd_step_size,
                mc_samples=mc_samples,
                c_scale=float(c_scale),
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
                wolf_kind="imq",
                wolf_imq_soft_threshold=wolf_imq_soft_threshold,
                wolf_tmd_threshold=wolf_tmd_threshold,
                device=device,
            )
            ret_random_cov, stats_random_cov = rollout_episode_return_attacked(
                env_random_cov,
                model,
                seed=seed,
                ssm=ssm,
                R=R,
                discount_delta=discount_delta,
                obs_noise_std=np.asarray(kf_meas_std, dtype=float),
                attack_eps=attack_eps,
                attack_prob=attack_prob,
                attack_mode="random",
                defense="covadapt",
                pgd_steps=pgd_steps,
                pgd_step_size=pgd_step_size,
                mc_samples=mc_samples,
                c_scale=float(c_scale),
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
                wolf_kind="imq",
                wolf_imq_soft_threshold=wolf_imq_soft_threshold,
                wolf_tmd_threshold=wolf_tmd_threshold,
                device=device,
            )

            attack_key = f"attack_cov_{c_scale:g}"
            random_key = f"random_cov_{c_scale:g}"
            totals[attack_key] += float(ret_attack_cov)
            totals[random_key] += float(ret_random_cov)
            series[attack_key].append(totals[attack_key])
            series[random_key].append(totals[random_key])
            fallback_by_mode["pgd"]["attack_attempts"] += int(stats_attack_cov["attack_attempts"])
            fallback_by_mode["pgd"]["fallback_to_real"] += int(stats_attack_cov["fallback_to_real"])
            fallback_by_mode["random"]["attack_attempts"] += int(stats_random_cov["attack_attempts"])
            fallback_by_mode["random"]["fallback_to_real"] += int(stats_random_cov["fallback_to_real"])

            env_attack_cov.close()
            env_random_cov.close()

        print(f"[eps={attack_eps:.2f} run {episode_idx + 1:4d}/{n_episodes}]")

    pgd_replace_pct = 100.0 * float(fallback_by_mode["pgd"]["fallback_to_real"]) / max(
        int(fallback_by_mode["pgd"]["attack_attempts"]), 1
    )
    random_replace_pct = 100.0 * float(fallback_by_mode["random"]["fallback_to_real"]) / max(
        int(fallback_by_mode["random"]["attack_attempts"]), 1
    )
    print(
        f"[eps={attack_eps:.2f}] replace_by_real_pct: "
        f"pgd={pgd_replace_pct:.2f}% | boundary={random_replace_pct:.2f}%"
    )

    data: dict[str, np.ndarray | float | int] = {
        "n_episodes": int(n_episodes),
        "seed0": int(seed0),
        "attack_eps": float(attack_eps),
        "attack_prob": float(attack_prob),
        "pgd_replace_by_real_pct": float(pgd_replace_pct),
        "boundary_replace_by_real_pct": float(random_replace_pct),
        "discount_delta": float(discount_delta),
        "pgd_steps": int(pgd_steps),
        "pgd_step_size": float(pgd_step_size),
        "mc_samples": int(mc_samples),
        "omega_h": float(omega_h),
        "omega_o": float(omega_o),
        "delta_threshold": float(delta_threshold),
        "wolf_imq_soft_threshold": float(wolf_imq_soft_threshold),
        "wolf_tmd_threshold": float(wolf_tmd_threshold),
        "obs_noise_std": np.asarray(obs_noise_std, dtype=float),
        "kf_meas_std": np.asarray(kf_meas_std, dtype=float),
        "kf_proc_std": np.asarray(kf_proc_std, dtype=float),
        "kf_meas_corr": np.asarray(kf_meas_corr if kf_meas_corr is not None else np.eye(4), dtype=float),
        "kf_proc_corr": np.asarray(kf_proc_corr if kf_proc_corr is not None else np.eye(4), dtype=float),
        "R": np.asarray(R, dtype=float),
        "legacy_Q_reference": np.asarray(legacy_Q, dtype=float),
        "predictor_kind": np.asarray("ekf_local_jacobian_discounted"),
        "A_d": ssm.A_d.astype(np.float32),
        "B_d": ssm.B_d.astype(np.float32),
        "H": ssm.H.astype(np.float32),
        "A_c": ssm.A_c.astype(np.float32),
        "B_c": ssm.B_c.astype(np.float32),
        "c_scales": np.asarray(c_scales, dtype=float),
        "acc_clean": np.asarray(series["clean"], dtype=float),
        "acc_noisy_kf": np.asarray(series["noisy_kf"], dtype=float),
        "acc_attack_kf": np.asarray(series["attack_kf"], dtype=float),
        "acc_random_kf": np.asarray(series["random_kf"], dtype=float),
        "acc_attack_wolf_imq": np.asarray(series["attack_wolf_imq"], dtype=float),
        "acc_attack_wolf_tmd": np.asarray(series["attack_wolf_tmd"], dtype=float),
        "acc_random_wolf_imq": np.asarray(series["random_wolf_imq"], dtype=float),
        "acc_random_wolf_tmd": np.asarray(series["random_wolf_tmd"], dtype=float),
    }
    for c_scale in c_scales:
        data[f"acc_attack_cov_{c_scale:g}"] = np.asarray(series[f"attack_cov_{c_scale:g}"], dtype=float)
        data[f"acc_random_cov_{c_scale:g}"] = np.asarray(series[f"random_cov_{c_scale:g}"], dtype=float)
    return data


def build_panel_specifications(
    c_scales: tuple[float, ...],
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """Return the label/key map for the three grouped-bar panels."""
    baseline_spec = [
        ("Noise-Free", "acc_clean"),
        ("Noise + KF", "acc_noisy_kf"),
        ("Attack + KF", "acc_attack_kf"),
        (r"Boundary $\epsilon$-perturbation + KF", "acc_random_kf"),
    ]

    attack_spec = [
        ("Noise-Free", "acc_clean"),
        ("Noise + KF", "acc_noisy_kf"),
        ("Attack + KF", "acc_attack_kf"),
    ]
    attack_spec.extend(
        [
            (
                rf"Attack + cov-adapt ($\lambda={c_scale:g}\lambda_{{\max}}$)",
                f"acc_attack_cov_{c_scale:g}",
            )
            for c_scale in c_scales
        ]
    )
    attack_spec.extend(
        [
            ("Attack + WoLF-IMQ", "acc_attack_wolf_imq"),
            ("Attack + WoLF-TMD", "acc_attack_wolf_tmd"),
        ]
    )

    random_spec = [
        ("Noise-Free", "acc_clean"),
        ("Noise + KF", "acc_noisy_kf"),
        (r"Boundary $\epsilon$-perturbation + KF", "acc_random_kf"),
    ]
    random_spec.extend(
        [
            (
                rf"Boundary $\epsilon$-perturbation + cov-adapt ($\lambda={c_scale:g}\lambda_{{\max}}$)",
                f"acc_random_cov_{c_scale:g}",
            )
            for c_scale in c_scales
        ]
    )
    random_spec.extend(
        [
            (r"Boundary $\epsilon$-perturbation + WoLF-IMQ", "acc_random_wolf_imq"),
            (r"Boundary $\epsilon$-perturbation + WoLF-TMD", "acc_random_wolf_tmd"),
        ]
    )
    return baseline_spec, attack_spec, random_spec


def plot_accumulated_reward_comparison_two_epsilons_with_wolf(
    *,
    data_by_epsilon: dict[float, dict[str, np.ndarray | float | int]],
    epsilon_values: tuple[float, ...],
    epsilon_display_values: tuple[float, ...] | None,
    outpath: str,
) -> None:
    """Draw the same three-panel comparison used by the RL benchmark."""
    set_plot_theme()

    reference_data = data_by_epsilon[float(epsilon_values[0])]
    c_scales = tuple(float(value) for value in np.asarray(reference_data["c_scales"], dtype=float))
    baseline_spec, attack_spec, random_spec = build_panel_specifications(c_scales)

    method_colors = {
        "acc_clean": "#6FAF8F",
        "acc_noisy_kf": "#A7D37A",
        "acc_attack_kf": "#E9B188",
        "acc_random_kf": "#8FAEDF",
        "acc_attack_wolf_imq": "#B69BD9",
        "acc_attack_wolf_tmd": "#8C72BF",
        "acc_random_wolf_imq": "#B69BD9",
        "acc_random_wolf_tmd": "#8C72BF",
    }
    attack_cov_colors = {
        f"acc_attack_cov_{c_scale:g}": color
        for c_scale, color in zip(c_scales, ["#E4B2AA", "#C96F5C", "#A54E42"], strict=False)
    }
    random_cov_colors = {
        f"acc_random_cov_{c_scale:g}": color
        for c_scale, color in zip(c_scales, ["#B8C9EE", "#6F92D8", "#4D73BD"], strict=False)
    }
    method_colors.update(attack_cov_colors)
    method_colors.update(random_cov_colors)

    hatch_by_epsilon = {
        float(epsilon_values[0]): "",
        float(epsilon_values[1]): "//////" if len(epsilon_values) > 1 else "",
    }
    if epsilon_display_values is None:
        epsilon_display_values = epsilon_values
    epsilon_label_by_value = {
        float(epsilon): float(display_epsilon)
        for epsilon, display_epsilon in zip(epsilon_values, epsilon_display_values, strict=False)
    }
    epsilon_invariant_keys = {"acc_clean", "acc_noisy_kf"}

    def panel_values(spec: list[tuple[str, str]]) -> dict[float, list[float]]:
        values_by_epsilon: dict[float, list[float]] = {}
        for epsilon in epsilon_values:
            panel_data = data_by_epsilon[float(epsilon)]
            panel_n_episodes = int(panel_data["n_episodes"])
            values_by_epsilon[float(epsilon)] = [
                normalized_final_reward(np.asarray(panel_data[key], dtype=float), n_episodes=panel_n_episodes)
                for _label, key in spec
            ]
        return values_by_epsilon

    baseline_values = panel_values(baseline_spec)
    attack_values = panel_values(attack_spec)
    random_values = panel_values(random_spec)

    all_values = []
    for values_by_epsilon in (baseline_values, attack_values, random_values):
        for epsilon in epsilon_values:
            all_values.extend(values_by_epsilon[float(epsilon)])
    all_values_np = np.asarray(all_values, dtype=float)
    value_abs_max = float(np.max(np.abs(all_values_np))) if all_values_np.size > 0 else 1.0
    value_abs_max = max(value_abs_max, 1.0)
    y_pad = 0.18 * value_abs_max
    y_min = float(np.min(all_values_np)) - y_pad
    y_max = float(np.max(all_values_np)) + y_pad
    y_min = min(y_min, -5.0)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(22.4, 6.5),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.82, 1.18, 1.18]},
    )
    ax_baseline, ax_attack, ax_random = axes

    def draw_grouped_bar_panel(
        *,
        ax: plt.Axes,
        spec: list[tuple[str, str]],
        values_by_epsilon: dict[float, list[float]],
        legend_width: str = "normal",
        legend_loc: str = "lower left",
    ) -> None:
        labels = [label for label, _key in spec]
        keys = [key for _label, key in spec]
        x = np.arange(len(labels), dtype=float)
        if x.size >= 2:
            x[1:] += 0.28
            x[1] -= 0.48

        n_eps = len(epsilon_values)
        group_width = 0.76
        bar_width = group_width / float(max(1, n_eps))
        offsets = (
            np.linspace(-0.5 * group_width + 0.5 * bar_width, 0.5 * group_width - 0.5 * bar_width, n_eps)
            if n_eps > 1
            else np.array([0.0], dtype=float)
        )
        key_by_index = {idx: key for idx, (_label, key) in enumerate(spec)}

        single_indices = [idx for idx, key in key_by_index.items() if key in epsilon_invariant_keys]
        if single_indices:
            single_positions = x[single_indices]
            single_values = np.asarray(values_by_epsilon[float(epsilon_values[0])], dtype=float)[single_indices]
            single_colors = [method_colors[key_by_index[idx]] for idx in single_indices]
            single_bars = ax.bar(
                single_positions,
                single_values,
                width=0.66 * group_width,
                color=single_colors,
                edgecolor=[darken_hex(color) for color in single_colors],
                linewidth=1.0,
                alpha=0.97,
                zorder=3,
            )
            for bar, value in zip(single_bars, single_values):
                offset = 0.028 * (y_max - y_min)
                text_y = value + offset if value >= 0.0 else value - offset
                ax.text(
                    float(bar.get_x() + 0.5 * bar.get_width()),
                    float(text_y),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0.0 else "top",
                    fontsize=8.6,
                    color="#2F2F2F",
                )

        for eps_idx, epsilon in enumerate(epsilon_values):
            varying_indices = [idx for idx, key in key_by_index.items() if key not in epsilon_invariant_keys]
            if not varying_indices:
                continue

            bar_positions = x[varying_indices] + offsets[eps_idx]
            panel_values_eps = np.asarray(values_by_epsilon[float(epsilon)], dtype=float)[varying_indices]
            panel_colors = [method_colors[key_by_index[idx]] for idx in varying_indices]
            bars = ax.bar(
                bar_positions,
                panel_values_eps,
                width=0.92 * bar_width,
                color=panel_colors,
                edgecolor=[darken_hex(color) for color in panel_colors],
                linewidth=1.0,
                hatch=hatch_by_epsilon[float(epsilon)],
                alpha=0.97,
                zorder=3,
            )

            for bar, value in zip(bars, panel_values_eps):
                offset = 0.028 * (y_max - y_min)
                text_y = value + offset if value >= 0.0 else value - offset
                ax.text(
                    float(bar.get_x() + 0.5 * bar.get_width()),
                    float(text_y),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0.0 else "top",
                    fontsize=8.6,
                    color="#2F2F2F",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.axhline(0.0, color="#4E4E4E", linewidth=1.15, alpha=0.92, zorder=1)
        ax.set_ylim(y_min, y_max)
        ax.set_axisbelow(True)

        method_handles = [
            Patch(
                facecolor=method_colors[key],
                edgecolor=darken_hex(method_colors[key]),
                linewidth=1.0,
            )
            for key in keys
        ]

        if legend_width == "wide":
            borderpad = 0.80
            handlelength = 3.1
            handletextpad = 0.98
            labelspacing = 0.63
        elif legend_width == "medium":
            borderpad = 0.64
            handlelength = 2.7
            handletextpad = 0.88
            labelspacing = 0.55
        else:
            borderpad = 0.54
            handlelength = 2.3
            handletextpad = 0.78
            labelspacing = 0.50

        method_legend = ax.legend(
            method_handles,
            labels,
            loc=legend_loc,
            frameon=True,
            framealpha=0.97,
            borderpad=borderpad,
            fontsize=9.3,
            handlelength=handlelength,
            handletextpad=handletextpad,
            labelspacing=labelspacing,
        )
        ax.add_artist(method_legend)

        epsilon_handles = [
            Patch(
                facecolor="#FFFFFF",
                edgecolor="#666666",
                linewidth=1.0,
                hatch=hatch_by_epsilon[float(epsilon)],
            )
            for epsilon in epsilon_values
        ]
        epsilon_labels = [
            rf"$\epsilon={epsilon_label_by_value[float(epsilon)]:g}$"
            for epsilon in epsilon_values
        ]
        ax.legend(
            epsilon_handles,
            epsilon_labels,
            loc="upper right",
            frameon=True,
            framealpha=0.97,
            borderpad=0.40,
            fontsize=9.1,
            handlelength=2.0,
            handletextpad=0.70,
            labelspacing=0.42,
        )

    for ax in axes:
        style_axis(ax)

    draw_grouped_bar_panel(
        ax=ax_baseline,
        spec=baseline_spec,
        values_by_epsilon=baseline_values,
    )
    draw_grouped_bar_panel(
        ax=ax_attack,
        spec=attack_spec,
        values_by_epsilon=attack_values,
        legend_width="wide",
        legend_loc="lower right",
    )
    draw_grouped_bar_panel(
        ax=ax_random,
        spec=random_spec,
        values_by_epsilon=random_values,
        legend_width="wide",
    )

    ax_baseline.set_ylabel("Mean accumulated reward")

    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(outpath, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    """
    Generate the two-epsilon CartPole comparison including WoLF baselines.

    All configuration stays in plain Python variables so it is easy to tweak,
    matching the style requested in this repository.
    """
    device = "cpu"
    model_path = ensure_downloaded_cartpole_checkpoint()

    # Observation noise in raw CartPole state coordinates:
    # [x, xdot, theta, thetadot].
    obs_noise_std = np.array([0.08, 0.18, 0.04, 0.18], dtype=float)
    attack_prob = 0.20
    # 4D chi-square radii corresponding approximately to 75% and 95%
    # predictive-ellipsoid coverage.
    attack_eps_values = (5.39, 9.49)
    attack_eps_display_values = (0.75, 0.95)
    discount_delta = DEFAULT_GYMNASIUM_DISCOUNT_DELTA

    # The attack geometry now follows the AdvSSM convention:
    # the ellipsoid is centered at the predictive mean and uses
    # Sigma_t = P_pred + R, while epsilon controls the confidence radius.

    # KF / EKF-like local linear model assumptions.
    # The shared Gymnasium predictor now uses the globally calibrated fixed
    # discount selected from the CartPole NIS sweep.
    kf_meas_std = obs_noise_std.copy()
    kf_proc_std = np.array([0.320, 0.720, 0.160, 0.720], dtype=float)
    kf_meas_corr = np.array(
        [
            [1.00, 0.18, 0.06, 0.00],
            [0.18, 1.00, 0.14, 0.22],
            [0.06, 0.14, 1.00, 0.18],
            [0.00, 0.22, 0.18, 1.00],
        ],
        dtype=float,
    )
    kf_proc_corr = np.array(
        [
            [1.00, 0.24, 0.08, 0.00],
            [0.24, 1.00, 0.16, 0.26],
            [0.08, 0.16, 1.00, 0.22],
            [0.00, 0.26, 0.22, 1.00],
        ],
        dtype=float,
    )

    # PGD on the posterior expected DQN value.
    pgd_steps = 20
    pgd_step_size = 0.34
    mc_samples = 64

    # Episodes used for the benchmark. Increase them for a more stable figure.
    n_episodes = 10
    seed0 = 100

    # Covariance-adaptation and WoLF hyperparameters.
    c_scales = (0.5, 1.0)
    omega_h = 0.50
    omega_o = 0.50
    delta_threshold = 0.20
    wolf_imq_soft_threshold = 0.20
    wolf_tmd_threshold = 2.8
    force_cache = True

    ssm = build_cartpole_linear_ssm()

    out_dir = figures_dir_for(os.path.dirname(os.path.abspath(__file__)))
    c_tag = "-".join(f"{value:g}" for value in c_scales).replace(".", "p")
    eps_tag = "-".join(str(value).replace(".", "p") for value in attack_eps_values)
    imq_tag = str(wolf_imq_soft_threshold).replace(".", "p")
    tmd_tag = str(wolf_tmd_threshold).replace(".", "p")

    outpath = os.path.join(
        out_dir,
        (
            "comparison_cartpole_two_eps_wolf_"
            f"N{n_episodes}_p{attack_prob}_eps{eps_tag}_c{c_tag}_"
            f"imq{imq_tag}_tmd{tmd_tag}.png"
        ),
    )

    data_by_epsilon: dict[float, dict[str, np.ndarray | float | int]] = {}
    for attack_eps in attack_eps_values:
        single_eps_outpath = os.path.join(
            out_dir,
            (
                "comparison_cartpole_wolf_"
                f"N{n_episodes}_p{attack_prob}_eps{str(attack_eps).replace('.', 'p')}_"
                f"c{c_tag}_imq{imq_tag}_tmd{tmd_tag}.png"
            ),
        )
        data_path = data_path_for_plot(single_eps_outpath)

        def compute_data_for_epsilon(attack_eps_value: float = float(attack_eps)) -> dict[str, np.ndarray | float | int]:
            return compute_accumulated_reward_data_with_wolf(
                n_episodes=n_episodes,
                seed0=seed0,
                model_path=model_path,
                obs_noise_std=obs_noise_std,
                attack_eps=attack_eps_value,
                attack_prob=attack_prob,
                discount_delta=discount_delta,
                kf_meas_std=kf_meas_std,
                kf_proc_std=kf_proc_std,
                kf_meas_corr=kf_meas_corr,
                kf_proc_corr=kf_proc_corr,
                pgd_steps=pgd_steps,
                pgd_step_size=pgd_step_size,
                mc_samples=mc_samples,
                c_scales=c_scales,
                omega_h=omega_h,
                omega_o=omega_o,
                delta_threshold=delta_threshold,
                wolf_imq_soft_threshold=wolf_imq_soft_threshold,
                wolf_tmd_threshold=wolf_tmd_threshold,
                device=device,
            )

        data_by_epsilon[float(attack_eps)] = cached_npz(
            data_path,
            compute_data_for_epsilon,
            force=force_cache,
        )

    plot_accumulated_reward_comparison_two_epsilons_with_wolf(
        data_by_epsilon=data_by_epsilon,
        epsilon_values=tuple(float(value) for value in attack_eps_values),
        epsilon_display_values=tuple(float(value) for value in attack_eps_display_values),
        outpath=outpath,
    )
    print(f"Saved figure to: {outpath}")


if __name__ == "__main__":
    main()
