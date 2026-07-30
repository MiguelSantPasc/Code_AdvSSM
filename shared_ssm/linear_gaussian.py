"""
Reusable Kalman filtering and RTS smoothing for linear-Gaussian SSMs.

Model and notation:
1. The latent state is `s_t in R^{d_s}`.
2. The observation is `o_t in R^{d_o}`.
3. The control input is `a_{t-1} in R^{d_a}` and is applied between `t-1`
   and `t`.
4. The model follows, for `t = 1, ..., T`,

       s_t = A_t s_{t-1} + B_t a_{t-1} + w_t,
       w_t ~ N(0, W_t)

       o_t = F_t s_t + G_t a_{t-1} + v_t,
       v_t ~ N(0, V_t)

   with prior `s_0 ~ N(m_0, P_0)`.

Design goals of this module:
1. Support constant or time-varying matrices through one common API.
2. Keep the implementation explicit and well-commented so later attack and
   defense modules can reuse the same intermediate quantities.
3. Expose both single-step recursions and full-sequence inference results.
4. Distinguish causal `online` inference from `offline` inference with
   optional RTS smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .linalg import project_to_psd
from .linalg import solve_spd


InferenceMode = Literal["online", "offline"]


@dataclass(frozen=True)
class LinearGaussianStateSpaceModel:
    """
    High-level linear-Gaussian SSM specification.

    The model accepts either constant matrices, such as `A.shape == (d_s, d_s)`,
    or time-varying sequences, such as `A.shape == (T, d_s, d_s)`.
    Each sequence entry at index `k` corresponds to model time `t = k + 1`.
    """

    A: np.ndarray
    F: np.ndarray
    W: np.ndarray
    V: np.ndarray
    m0: np.ndarray
    P0: np.ndarray
    B: np.ndarray | None = None
    G: np.ndarray | None = None

    def resolve(self, num_steps: int, action_dim: int | None = None) -> "ResolvedLinearGaussianStateSpaceModel":
        """Broadcast constant matrices and validate the model for `num_steps`."""
        num_steps = int(num_steps)
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative.")

        m0 = np.asarray(self.m0, dtype=float).reshape(-1)
        P0 = project_to_psd(np.asarray(self.P0, dtype=float))
        state_dim = m0.size
        if P0.shape != (state_dim, state_dim):
            raise ValueError("P0 must have shape (d_s, d_s).")

        F_seq = _resolve_matrix_sequence(
            self.F,
            num_steps=num_steps,
            trailing_shape=None,
            name="F",
        )
        if num_steps == 0:
            if F_seq.ndim != 3:
                raise ValueError("Resolved observation matrices must have rank 3.")
            obs_dim = F_seq.shape[1]
        else:
            obs_dim = F_seq.shape[1]

        A_seq = _resolve_matrix_sequence(
            self.A,
            num_steps=num_steps,
            trailing_shape=(state_dim, state_dim),
            name="A",
        )
        if A_seq.shape != (num_steps, state_dim, state_dim):
            raise ValueError("Resolved A sequence has inconsistent shape.")

        F_seq = _resolve_matrix_sequence(
            self.F,
            num_steps=num_steps,
            trailing_shape=(obs_dim, state_dim),
            name="F",
        )
        W_seq = _resolve_covariance_sequence(
            self.W,
            num_steps=num_steps,
            dim=state_dim,
            name="W",
        )
        V_seq = _resolve_covariance_sequence(
            self.V,
            num_steps=num_steps,
            dim=obs_dim,
            name="V",
        )

        inferred_action_dim = _infer_action_dim(self.B, self.G, action_dim)
        B_seq = _resolve_optional_control_sequence(
            self.B,
            num_steps=num_steps,
            rows=state_dim,
            cols=inferred_action_dim,
            name="B",
        )
        G_seq = _resolve_optional_control_sequence(
            self.G,
            num_steps=num_steps,
            rows=obs_dim,
            cols=inferred_action_dim,
            name="G",
        )

        return ResolvedLinearGaussianStateSpaceModel(
            A=A_seq,
            B=B_seq,
            F=F_seq,
            G=G_seq,
            W=W_seq,
            V=V_seq,
            m0=m0,
            P0=P0,
        )


@dataclass(frozen=True)
class ResolvedLinearGaussianStateSpaceModel:
    """
    Fully expanded linear-Gaussian SSM with one matrix per step.

    Shapes:
    1. `A, B, F, G, W, V` all have leading dimension `T`.
    2. `m0` has shape `(d_s,)`.
    3. `P0` has shape `(d_s, d_s)`.
    """

    A: np.ndarray
    B: np.ndarray
    F: np.ndarray
    G: np.ndarray
    W: np.ndarray
    V: np.ndarray
    m0: np.ndarray
    P0: np.ndarray

    @property
    def num_steps(self) -> int:
        """Return the number of observation/update steps."""
        return int(self.A.shape[0])

    @property
    def state_dim(self) -> int:
        """Return the latent-state dimension `d_s`."""
        return int(self.m0.size)

    @property
    def obs_dim(self) -> int:
        """Return the observation dimension `d_o`."""
        return int(self.F.shape[1])

    @property
    def action_dim(self) -> int:
        """Return the control dimension `d_a`."""
        return int(self.B.shape[2])


@dataclass(frozen=True)
class KalmanPredictResult:
    """One-step predictive belief and induced predictive observation law."""

    state_mean: np.ndarray
    state_covariance: np.ndarray
    observation_mean: np.ndarray
    observation_covariance: np.ndarray


@dataclass(frozen=True)
class KalmanUpdateResult:
    """One-step posterior update after observing `o_t`."""

    state_mean: np.ndarray
    state_covariance: np.ndarray
    kalman_gain: np.ndarray
    innovation: np.ndarray


@dataclass(frozen=True)
class KalmanInferenceResult:
    """
    Full-sequence inference outputs for the shared Kalman library.

    Array conventions:
    1. Filtered and smoothed state arrays have length `T + 1` and include the
       prior/posterior at `t = 0`.
    2. Predictive and observation-space arrays have length `T` and correspond
       to times `t = 1, ..., T`.
    3. `kalman_gains[k]` is the gain used at model time `t = k + 1`.
    4. `smoothing_gains[k]` maps information from `t = k + 1` back to `t = k`.
    """

    mode: InferenceMode
    observations: np.ndarray
    actions: np.ndarray
    observation_mask: np.ndarray
    predictive_state_means: np.ndarray
    predictive_state_covariances: np.ndarray
    predicted_observation_means: np.ndarray
    predicted_observation_covariances: np.ndarray
    filtered_state_means: np.ndarray
    filtered_state_covariances: np.ndarray
    kalman_gains: np.ndarray
    innovations: np.ndarray
    smoothed_state_means: np.ndarray | None = None
    smoothed_state_covariances: np.ndarray | None = None
    smoothing_gains: np.ndarray | None = None

    @property
    def num_steps(self) -> int:
        """Return the number of filtering steps."""
        return int(self.observations.shape[0])


def predict_observation_distribution(
    *,
    state_mean: np.ndarray,
    state_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    observation_control_matrix: np.ndarray,
    observation_covariance: np.ndarray,
    action_prev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the one-step predictive observation distribution.

    This corresponds to:
        p(o_t | o_{1:t-1}, a_{1:t-1}) = N(o_hat_t, S_t).
    """
    state_mean = np.asarray(state_mean, dtype=float).reshape(-1)
    state_covariance = project_to_psd(np.asarray(state_covariance, dtype=float))
    observation_matrix = np.asarray(observation_matrix, dtype=float)
    observation_control_matrix = np.asarray(observation_control_matrix, dtype=float)
    observation_covariance = project_to_psd(np.asarray(observation_covariance, dtype=float))
    action_prev = np.asarray(action_prev, dtype=float).reshape(-1)

    observation_mean = observation_matrix @ state_mean + observation_control_matrix @ action_prev
    innovation_covariance = project_to_psd(
        observation_matrix @ state_covariance @ observation_matrix.T + observation_covariance
    )
    return observation_mean, innovation_covariance


def kalman_predict_step(
    *,
    prev_state_mean: np.ndarray,
    prev_state_covariance: np.ndarray,
    transition_matrix: np.ndarray,
    control_matrix: np.ndarray,
    process_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    observation_control_matrix: np.ndarray,
    observation_covariance: np.ndarray,
    action_prev: np.ndarray,
) -> KalmanPredictResult:
    """
    Run the one-step Kalman prediction from `t-1` to `t`.

    The action `action_prev` is the control `a_{t-1}` that drives both the
    transition into `s_t` and the direct observation term of `o_t`.
    """
    prev_state_mean = np.asarray(prev_state_mean, dtype=float).reshape(-1)
    prev_state_covariance = project_to_psd(np.asarray(prev_state_covariance, dtype=float))
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    control_matrix = np.asarray(control_matrix, dtype=float)
    process_covariance = project_to_psd(np.asarray(process_covariance, dtype=float))
    action_prev = np.asarray(action_prev, dtype=float).reshape(-1)

    pred_state_mean = transition_matrix @ prev_state_mean + control_matrix @ action_prev
    pred_state_covariance = project_to_psd(
        transition_matrix @ prev_state_covariance @ transition_matrix.T + process_covariance
    )
    pred_obs_mean, pred_obs_covariance = predict_observation_distribution(
        state_mean=pred_state_mean,
        state_covariance=pred_state_covariance,
        observation_matrix=observation_matrix,
        observation_control_matrix=observation_control_matrix,
        observation_covariance=observation_covariance,
        action_prev=action_prev,
    )
    return KalmanPredictResult(
        state_mean=pred_state_mean,
        state_covariance=pred_state_covariance,
        observation_mean=pred_obs_mean,
        observation_covariance=pred_obs_covariance,
    )


def kalman_update_step(
    *,
    pred_state_mean: np.ndarray,
    pred_state_covariance: np.ndarray,
    observation: np.ndarray,
    observation_matrix: np.ndarray,
    observation_control_matrix: np.ndarray,
    observation_covariance: np.ndarray,
    action_prev: np.ndarray,
    observe: bool = True,
) -> KalmanUpdateResult:
    """
    Run the one-step Kalman update at time `t`.

    When `observe` is `False`, the function skips the measurement correction
    and returns the predictive belief unchanged. This is useful for causal
    leave-one-out constructions and for handling missing observations.
    """
    pred_state_mean = np.asarray(pred_state_mean, dtype=float).reshape(-1)
    pred_state_covariance = project_to_psd(np.asarray(pred_state_covariance, dtype=float))
    observation = np.asarray(observation, dtype=float).reshape(-1)
    observation_matrix = np.asarray(observation_matrix, dtype=float)
    observation_control_matrix = np.asarray(observation_control_matrix, dtype=float)
    observation_covariance = project_to_psd(np.asarray(observation_covariance, dtype=float))
    action_prev = np.asarray(action_prev, dtype=float).reshape(-1)

    obs_mean, innovation_covariance = predict_observation_distribution(
        state_mean=pred_state_mean,
        state_covariance=pred_state_covariance,
        observation_matrix=observation_matrix,
        observation_control_matrix=observation_control_matrix,
        observation_covariance=observation_covariance,
        action_prev=action_prev,
    )
    innovation = observation - obs_mean
    kalman_gain = solve_spd(innovation_covariance, observation_matrix @ pred_state_covariance.T).T

    if not observe:
        zero_gain = np.zeros_like(kalman_gain)
        return KalmanUpdateResult(
            state_mean=pred_state_mean.copy(),
            state_covariance=pred_state_covariance.copy(),
            kalman_gain=zero_gain,
            innovation=innovation,
        )

    post_state_mean = pred_state_mean + kalman_gain @ innovation

    # The Joseph form is algebraically equivalent to the textbook update and
    # keeps the covariance PSD under finite-precision arithmetic.
    identity = np.eye(pred_state_covariance.shape[0], dtype=float)
    left_factor = identity - kalman_gain @ observation_matrix
    post_state_covariance = project_to_psd(
        left_factor @ pred_state_covariance @ left_factor.T
        + kalman_gain @ observation_covariance @ kalman_gain.T
    )
    return KalmanUpdateResult(
        state_mean=post_state_mean,
        state_covariance=post_state_covariance,
        kalman_gain=kalman_gain,
        innovation=innovation,
    )


def run_kalman_inference(
    *,
    model: LinearGaussianStateSpaceModel | ResolvedLinearGaussianStateSpaceModel,
    observations: np.ndarray,
    actions: np.ndarray | None = None,
    observation_mask: np.ndarray | None = None,
    mode: InferenceMode = "offline",
    smooth: bool | None = None,
) -> KalmanInferenceResult:
    """
    Run full-sequence Kalman inference.

    Modes:
    1. `online`: causal filtering only, with no smoothing.
    2. `offline`: filtering plus optional RTS smoothing.

    Sequence convention:
    1. `observations[k]` stores `o_{k+1}`.
    2. `actions[k]` stores `a_k`, the control applied between `k` and `k+1`.
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2:
        raise ValueError("observations must have shape (T, d_o).")

    num_steps = int(observations.shape[0])
    resolved_model = _resolve_model_instance(model, num_steps=num_steps, actions=actions)
    action_sequence = _prepare_action_sequence(
        actions,
        num_steps=num_steps,
        action_dim=resolved_model.action_dim,
    )
    obs_mask = _prepare_observation_mask(observation_mask, num_steps=num_steps)

    if mode not in ("online", "offline"):
        raise ValueError("mode must be either 'online' or 'offline'.")
    if smooth is None:
        smooth = mode == "offline"
    if mode == "online" and smooth:
        raise ValueError("online mode cannot request RTS smoothing.")

    predictive_state_means = np.zeros((num_steps, resolved_model.state_dim), dtype=float)
    predictive_state_covariances = np.zeros(
        (num_steps, resolved_model.state_dim, resolved_model.state_dim),
        dtype=float,
    )
    predicted_observation_means = np.zeros((num_steps, resolved_model.obs_dim), dtype=float)
    predicted_observation_covariances = np.zeros(
        (num_steps, resolved_model.obs_dim, resolved_model.obs_dim),
        dtype=float,
    )
    filtered_state_means = np.zeros((num_steps + 1, resolved_model.state_dim), dtype=float)
    filtered_state_covariances = np.zeros(
        (num_steps + 1, resolved_model.state_dim, resolved_model.state_dim),
        dtype=float,
    )
    kalman_gains = np.zeros(
        (num_steps, resolved_model.state_dim, resolved_model.obs_dim),
        dtype=float,
    )
    innovations = np.zeros((num_steps, resolved_model.obs_dim), dtype=float)

    filtered_state_means[0] = resolved_model.m0
    filtered_state_covariances[0] = resolved_model.P0

    for step_idx in range(num_steps):
        predict_result = kalman_predict_step(
            prev_state_mean=filtered_state_means[step_idx],
            prev_state_covariance=filtered_state_covariances[step_idx],
            transition_matrix=resolved_model.A[step_idx],
            control_matrix=resolved_model.B[step_idx],
            process_covariance=resolved_model.W[step_idx],
            observation_matrix=resolved_model.F[step_idx],
            observation_control_matrix=resolved_model.G[step_idx],
            observation_covariance=resolved_model.V[step_idx],
            action_prev=action_sequence[step_idx],
        )
        predictive_state_means[step_idx] = predict_result.state_mean
        predictive_state_covariances[step_idx] = predict_result.state_covariance
        predicted_observation_means[step_idx] = predict_result.observation_mean
        predicted_observation_covariances[step_idx] = predict_result.observation_covariance

        update_result = kalman_update_step(
            pred_state_mean=predict_result.state_mean,
            pred_state_covariance=predict_result.state_covariance,
            observation=observations[step_idx],
            observation_matrix=resolved_model.F[step_idx],
            observation_control_matrix=resolved_model.G[step_idx],
            observation_covariance=resolved_model.V[step_idx],
            action_prev=action_sequence[step_idx],
            observe=bool(obs_mask[step_idx]),
        )
        filtered_state_means[step_idx + 1] = update_result.state_mean
        filtered_state_covariances[step_idx + 1] = update_result.state_covariance
        kalman_gains[step_idx] = update_result.kalman_gain
        innovations[step_idx] = update_result.innovation

    smoothed_state_means = None
    smoothed_state_covariances = None
    smoothing_gains = None
    if smooth and num_steps > 0:
        smoothed_state_means = filtered_state_means.copy()
        smoothed_state_covariances = filtered_state_covariances.copy()
        smoothing_gains = np.zeros(
            (num_steps, resolved_model.state_dim, resolved_model.state_dim),
            dtype=float,
        )

        for step_idx in range(num_steps - 1, -1, -1):
            pred_cov_next = predictive_state_covariances[step_idx]
            smoothing_gain = solve_spd(
                pred_cov_next,
                resolved_model.A[step_idx] @ filtered_state_covariances[step_idx],
            ).T
            smoothing_gains[step_idx] = smoothing_gain

            smoothed_state_means[step_idx] = filtered_state_means[step_idx] + smoothing_gain @ (
                smoothed_state_means[step_idx + 1] - predictive_state_means[step_idx]
            )
            smoothed_state_covariances[step_idx] = project_to_psd(
                filtered_state_covariances[step_idx]
                + smoothing_gain
                @ (smoothed_state_covariances[step_idx + 1] - pred_cov_next)
                @ smoothing_gain.T
            )

    return KalmanInferenceResult(
        mode=mode,
        observations=observations,
        actions=action_sequence,
        observation_mask=obs_mask,
        predictive_state_means=predictive_state_means,
        predictive_state_covariances=predictive_state_covariances,
        predicted_observation_means=predicted_observation_means,
        predicted_observation_covariances=predicted_observation_covariances,
        filtered_state_means=filtered_state_means,
        filtered_state_covariances=filtered_state_covariances,
        kalman_gains=kalman_gains,
        innovations=innovations,
        smoothed_state_means=smoothed_state_means,
        smoothed_state_covariances=smoothed_state_covariances,
        smoothing_gains=smoothing_gains,
    )


def _resolve_model_instance(
    model: LinearGaussianStateSpaceModel | ResolvedLinearGaussianStateSpaceModel,
    *,
    num_steps: int,
    actions: np.ndarray | None,
) -> ResolvedLinearGaussianStateSpaceModel:
    """Normalize either model representation into a resolved step-wise model."""
    if isinstance(model, ResolvedLinearGaussianStateSpaceModel):
        if model.num_steps != num_steps:
            raise ValueError("Resolved model length does not match the observation sequence.")
        return model

    inferred_action_dim = None
    if actions is not None:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2:
            raise ValueError("actions must have shape (T, d_a).")
        inferred_action_dim = int(actions.shape[1])
    return model.resolve(num_steps=num_steps, action_dim=inferred_action_dim)


def _prepare_action_sequence(
    actions: np.ndarray | None,
    *,
    num_steps: int,
    action_dim: int,
) -> np.ndarray:
    """Return a validated action sequence with shape `(T, d_a)`."""
    if actions is None:
        return np.zeros((num_steps, action_dim), dtype=float)

    actions = np.asarray(actions, dtype=float)
    if actions.shape != (num_steps, action_dim):
        raise ValueError(
            f"actions must have shape ({num_steps}, {action_dim}), "
            f"received {actions.shape}."
        )
    return actions


def _prepare_observation_mask(observation_mask: np.ndarray | None, *, num_steps: int) -> np.ndarray:
    """Return a validated observation-usage mask."""
    if observation_mask is None:
        return np.ones((num_steps,), dtype=bool)

    observation_mask = np.asarray(observation_mask, dtype=bool)
    if observation_mask.shape != (num_steps,):
        raise ValueError(
            f"observation_mask must have shape ({num_steps},), "
            f"received {observation_mask.shape}."
        )
    return observation_mask


def _infer_action_dim(
    control_matrix: np.ndarray | None,
    observation_control_matrix: np.ndarray | None,
    action_dim: int | None,
) -> int:
    """Infer the control dimension from `B`, `G`, or an external action array."""
    candidate_dims: list[int] = []
    if action_dim is not None:
        candidate_dims.append(int(action_dim))
    if control_matrix is not None:
        candidate_dims.append(int(np.asarray(control_matrix).shape[-1]))
    if observation_control_matrix is not None:
        candidate_dims.append(int(np.asarray(observation_control_matrix).shape[-1]))

    if not candidate_dims:
        return 0

    unique_dims = sorted(set(candidate_dims))
    if len(unique_dims) != 1:
        raise ValueError(f"Inconsistent action dimensions detected: {unique_dims}.")
    return unique_dims[0]


def _resolve_matrix_sequence(
    matrix: np.ndarray,
    *,
    num_steps: int,
    trailing_shape: tuple[int, int] | None,
    name: str,
) -> np.ndarray:
    """Broadcast a constant matrix or validate a time-varying matrix sequence."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim == 2:
        if trailing_shape is not None and matrix.shape != trailing_shape:
            raise ValueError(f"{name} must have shape {trailing_shape} or (T, *shape).")
        return np.repeat(matrix[None, :, :], num_steps, axis=0)

    if matrix.ndim == 3:
        if matrix.shape[0] != num_steps:
            raise ValueError(f"{name} sequence must have leading dimension {num_steps}.")
        if trailing_shape is not None and matrix.shape[1:] != trailing_shape:
            raise ValueError(f"{name} sequence entries must have shape {trailing_shape}.")
        return matrix

    raise ValueError(f"{name} must have rank 2 or 3.")


def _resolve_covariance_sequence(
    covariance: np.ndarray,
    *,
    num_steps: int,
    dim: int,
    name: str,
) -> np.ndarray:
    """Broadcast and PSD-project a covariance matrix sequence."""
    covariance_seq = _resolve_matrix_sequence(
        covariance,
        num_steps=num_steps,
        trailing_shape=(dim, dim),
        name=name,
    )
    return np.stack([project_to_psd(covariance_seq[idx]) for idx in range(num_steps)], axis=0)


def _resolve_optional_control_sequence(
    matrix: np.ndarray | None,
    *,
    num_steps: int,
    rows: int,
    cols: int,
    name: str,
) -> np.ndarray:
    """Resolve an optional control matrix sequence, defaulting to zeros."""
    if matrix is None:
        return np.zeros((num_steps, rows, cols), dtype=float)

    return _resolve_matrix_sequence(
        matrix,
        num_steps=num_steps,
        trailing_shape=(rows, cols),
        name=name,
    )
