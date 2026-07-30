"""
Result containers and sequence helpers for adversarial SSM experiments.

The helpers here keep experiment scripts small and, importantly, avoid silent
mutation of the clean observation sequence when inserting adversarial values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linear_gaussian import KalmanInferenceResult
from .linear_gaussian import LinearGaussianStateSpaceModel
from .linear_gaussian import ResolvedLinearGaussianStateSpaceModel
from .linear_gaussian import run_kalman_inference


@dataclass(frozen=True)
class AttackApplicationResult:
    """Comparable clean and attacked inference outputs for one attack."""

    clean_observations: np.ndarray
    attacked_observations: np.ndarray
    clean_inference: KalmanInferenceResult
    attacked_inference: KalmanInferenceResult


def replace_observation(
    *,
    observations: np.ndarray,
    observation_index: int,
    adversarial_observation: np.ndarray,
) -> np.ndarray:
    """Return a copy of `observations` with one adversarial observation inserted."""
    observations = np.asarray(observations, dtype=float)
    attacked = observations.copy()
    attacked[int(observation_index)] = np.asarray(adversarial_observation, dtype=float)
    return attacked


def apply_attack_and_rerun(
    *,
    model: LinearGaussianStateSpaceModel | ResolvedLinearGaussianStateSpaceModel,
    observations: np.ndarray,
    actions: np.ndarray | None,
    observation_index: int,
    adversarial_observation: np.ndarray,
    mode: str = "offline",
    smooth: bool | None = None,
) -> AttackApplicationResult:
    """
    Insert an adversarial observation into a copy and rerun inference.

    The original observation sequence is preserved unchanged.
    """
    observations = np.asarray(observations, dtype=float)
    attacked_observations = replace_observation(
        observations=observations,
        observation_index=observation_index,
        adversarial_observation=adversarial_observation,
    )
    clean_inference = run_kalman_inference(
        model=model,
        observations=observations,
        actions=actions,
        mode=mode,  # type: ignore[arg-type]
        smooth=smooth,
    )
    attacked_inference = run_kalman_inference(
        model=model,
        observations=attacked_observations,
        actions=actions,
        mode=mode,  # type: ignore[arg-type]
        smooth=smooth,
    )
    return AttackApplicationResult(
        clean_observations=observations.copy(),
        attacked_observations=attacked_observations,
        clean_inference=clean_inference,
        attacked_inference=attacked_inference,
    )
