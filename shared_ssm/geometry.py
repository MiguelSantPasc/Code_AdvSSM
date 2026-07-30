"""
Attack geometry for online and offline linear-Gaussian SSM inference.

The central object is the distribution obtained by removing one observation:

    p(s_t | o_{-t}, a_{0:T-1})

for offline attacks, or the causal predictive distribution:

    p(s_t | o_{1:t-1}, a_{0:t-1})

for online attacks. Both induce an observation-space ellipsoid that can be
shared by linear, nonlinear, and RL attack objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .constraints import EllipsoidConstraint
from .linalg import project_to_psd
from .linalg import solve_spd
from .linear_gaussian import KalmanInferenceResult
from .linear_gaussian import LinearGaussianStateSpaceModel
from .linear_gaussian import ResolvedLinearGaussianStateSpaceModel
from .linear_gaussian import run_kalman_inference


AttackInferenceMode = Literal["online", "offline"]


@dataclass(frozen=True)
class AttackGeometry:
    """
    Shared geometry for attacks on one observation.

    `observation_index` is zero-based in the API and corresponds to model time
    `t = observation_index + 1`.
    """

    mode: AttackInferenceMode
    observation_index: int
    state_mean_without_observation: np.ndarray
    state_covariance_without_observation: np.ndarray
    observation_mean_without_observation: np.ndarray
    observation_covariance_without_observation: np.ndarray
    posterior_gain: np.ndarray
    posterior_covariance: np.ndarray
    constraint: EllipsoidConstraint
    inference_without_observation: KalmanInferenceResult

    def posterior_mean_for_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Return `E[s_t | o_t = observation, context]` for this attack geometry.

        The context is either past-only information in online mode or all other
        observations in offline mode.
        """
        observation = np.asarray(observation, dtype=float).reshape(-1)
        innovation = observation - self.observation_mean_without_observation
        return self.state_mean_without_observation + self.posterior_gain @ innovation


def build_attack_geometry(
    *,
    model: LinearGaussianStateSpaceModel | ResolvedLinearGaussianStateSpaceModel,
    observations: np.ndarray,
    actions: np.ndarray | None,
    observation_index: int,
    epsilon: float,
    mode: AttackInferenceMode,
) -> AttackGeometry:
    """
    Build the plausible observation region and posterior affine map.

    In `online` mode the region is the causal predictive observation law at
    the attacked instant. In `offline` mode the attacked observation is masked
    out and RTS smoothing supplies the leave-one-out state distribution.
    """
    observations = np.asarray(observations, dtype=float)
    num_steps = int(observations.shape[0])
    observation_index = int(observation_index)
    if not (0 <= observation_index < num_steps):
        raise ValueError("observation_index must satisfy 0 <= index < T.")
    if mode not in ("online", "offline"):
        raise ValueError("mode must be either 'online' or 'offline'.")
    action_dim = None if actions is None else int(np.asarray(actions, dtype=float).shape[1])
    resolved_model = model.resolve(num_steps, action_dim) if isinstance(model, LinearGaussianStateSpaceModel) else model

    if mode == "online":
        causal_mask = np.zeros((num_steps,), dtype=bool)
        causal_mask[:observation_index] = True
        inference = run_kalman_inference(
            model=model,
            observations=observations,
            actions=actions,
            observation_mask=causal_mask,
            mode="online",
        )
        state_mean = inference.predictive_state_means[observation_index]
        state_covariance = inference.predictive_state_covariances[observation_index]
        observation_mean = inference.predicted_observation_means[observation_index]
        observation_covariance = inference.predicted_observation_covariances[observation_index]
    else:
        loo_mask = np.ones((num_steps,), dtype=bool)
        loo_mask[observation_index] = False
        inference = run_kalman_inference(
            model=model,
            observations=observations,
            actions=actions,
            observation_mask=loo_mask,
            mode="offline",
            smooth=True,
        )
        if inference.smoothed_state_means is None or inference.smoothed_state_covariances is None:
            raise RuntimeError("offline attack geometry requires smoothed inference.")

        state_time_index = observation_index + 1
        action_sequence = inference.actions
        state_mean = inference.smoothed_state_means[state_time_index]
        state_covariance = inference.smoothed_state_covariances[state_time_index]
        F_t = resolved_model.F[observation_index]
        G_t = resolved_model.G[observation_index]
        V_t = resolved_model.V[observation_index]
        action_prev = action_sequence[observation_index]
        observation_mean = F_t @ state_mean + G_t @ action_prev
        observation_covariance = project_to_psd(F_t @ state_covariance @ F_t.T + V_t)

    F_obs = resolved_model.F[observation_index]
    V_obs = resolved_model.V[observation_index]
    observation_covariance = project_to_psd(observation_covariance)
    posterior_gain = solve_spd(observation_covariance, F_obs @ state_covariance.T).T
    posterior_covariance = project_to_psd(
        state_covariance - posterior_gain @ F_obs @ state_covariance
    )

    constraint = EllipsoidConstraint(
        center=observation_mean,
        covariance=observation_covariance,
        epsilon=float(epsilon),
    )
    return AttackGeometry(
        mode=mode,
        observation_index=observation_index,
        state_mean_without_observation=np.asarray(state_mean, dtype=float),
        state_covariance_without_observation=project_to_psd(state_covariance),
        observation_mean_without_observation=np.asarray(observation_mean, dtype=float),
        observation_covariance_without_observation=observation_covariance,
        posterior_gain=posterior_gain,
        posterior_covariance=posterior_covariance,
        constraint=constraint,
        inference_without_observation=inference,
    )
