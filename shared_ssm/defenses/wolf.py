"""
Reusable WoLF robust-measurement updates for linear-Gaussian filters.

This module collects the small amount of reusable logic needed by the WoLF
baselines requested in the RL and Gymnasium benchmarks. The implementation
follows the practical form used in the previous repository experiments:

1. Compute a nominal innovation from the predictive measurement law.
2. Convert that innovation into a squared observation weight `w_t^2`.
3. Inflate the observation covariance as `V_t / w_t^2`.
4. Run the Kalman measurement update with that effective covariance.

The two supported weighting rules are:
1. `IMQ`: inverse-multiquadric soft downweighting.
2. `TMD`: transport/Mahalanobis-distance hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..linalg import project_to_psd
from ..linalg import solve_spd
from ..linear_gaussian import KalmanPredictResult
from ..linear_gaussian import KalmanUpdateResult


WoLFKind = Literal["imq", "tmd"]


@dataclass(frozen=True)
class WoLFConfig:
    """Configuration of one WoLF robust measurement update."""

    kind: WoLFKind
    imq_soft_threshold: float = 1.0
    tmd_threshold: float = 3.0
    min_weight: float = 1e-6


@dataclass(frozen=True)
class WoLFDiagnostics:
    """Diagnostics emitted by one WoLF measurement update."""

    weight: float
    weight_squared: float
    innovation_norm: float
    mahalanobis_squared: float
    effective_observation_covariance: np.ndarray


@dataclass(frozen=True)
class WoLFUpdateResult:
    """Kalman update result together with the robust-weight diagnostics."""

    update: KalmanUpdateResult
    diagnostics: WoLFDiagnostics


def wolf_imq_weight_squared(
    innovation: np.ndarray,
    *,
    soft_threshold: float,
    min_weight: float = 1e-6,
) -> float:
    """
    Return the IMQ squared weight `w_t^2`.

    The IMQ rule is implemented in its numerically convenient squared form:

        w_t^2 = c^2 / (c^2 + ||innovation||^2).
    """
    if float(soft_threshold) <= 0.0:
        raise ValueError("soft_threshold must be positive.")
    if float(min_weight) <= 0.0 or float(min_weight) > 1.0:
        raise ValueError("min_weight must lie in (0, 1].")

    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    threshold_sq = float(soft_threshold) ** 2
    innovation_sq = float(np.dot(innovation, innovation))
    weight_sq = threshold_sq / (threshold_sq + innovation_sq)
    return float(np.clip(weight_sq, float(min_weight), 1.0))


def wolf_tmd_weight_squared(
    innovation: np.ndarray,
    *,
    innovation_covariance: np.ndarray,
    threshold: float,
    min_weight: float = 1e-6,
) -> float:
    """
    Return the TMD squared weight `w_t^2`.

    The TMD rule acts as a hard gate on the innovation Mahalanobis distance:
    accept the observation with weight one, otherwise strongly discount it.
    """
    if float(threshold) <= 0.0:
        raise ValueError("threshold must be positive.")
    if float(min_weight) <= 0.0 or float(min_weight) > 1.0:
        raise ValueError("min_weight must lie in (0, 1].")

    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    innovation_covariance = project_to_psd(np.asarray(innovation_covariance, dtype=float))
    mahalanobis_squared = float(np.dot(innovation, solve_spd(innovation_covariance, innovation)))
    mahalanobis_distance = float(np.sqrt(max(mahalanobis_squared, 0.0)))
    if mahalanobis_distance < float(threshold):
        return 1.0
    return float(min_weight)


def run_wolf_measurement_update(
    *,
    predict_result: KalmanPredictResult,
    observation: np.ndarray,
    observation_matrix: np.ndarray,
    observation_covariance: np.ndarray,
    config: WoLFConfig,
) -> WoLFUpdateResult:
    """
    Run one Kalman measurement update with WoLF observation downweighting.

    The caller provides the predictive quantities from the shared Kalman code.
    The robustification only changes the measurement update, leaving the
    prediction model untouched.
    """
    observation = np.asarray(observation, dtype=float).reshape(-1)
    observation_matrix = np.asarray(observation_matrix, dtype=float)
    observation_covariance = project_to_psd(np.asarray(observation_covariance, dtype=float))
    predictive_state_mean = np.asarray(predict_result.state_mean, dtype=float).reshape(-1)
    predictive_state_covariance = project_to_psd(np.asarray(predict_result.state_covariance, dtype=float))
    predictive_observation_mean = np.asarray(predict_result.observation_mean, dtype=float).reshape(-1)
    predictive_observation_covariance = project_to_psd(
        np.asarray(predict_result.observation_covariance, dtype=float)
    )

    innovation = observation - predictive_observation_mean
    innovation_norm = float(np.linalg.norm(innovation))
    mahalanobis_squared = float(np.dot(innovation, solve_spd(predictive_observation_covariance, innovation)))

    if config.kind == "imq":
        weight_squared = wolf_imq_weight_squared(
            innovation,
            soft_threshold=float(config.imq_soft_threshold),
            min_weight=float(config.min_weight),
        )
    elif config.kind == "tmd":
        weight_squared = wolf_tmd_weight_squared(
            innovation,
            innovation_covariance=predictive_observation_covariance,
            threshold=float(config.tmd_threshold),
            min_weight=float(config.min_weight),
        )
    else:
        raise ValueError(f"Unsupported WoLF kind: {config.kind}")

    effective_observation_covariance = project_to_psd(
        observation_covariance / float(weight_squared)
    )
    effective_predictive_observation_covariance = project_to_psd(
        observation_matrix @ predictive_state_covariance @ observation_matrix.T
        + effective_observation_covariance
    )
    kalman_gain = solve_spd(
        effective_predictive_observation_covariance,
        observation_matrix @ predictive_state_covariance.T,
    ).T
    posterior_state_mean = predictive_state_mean + kalman_gain @ innovation

    state_dim = predictive_state_mean.size
    identity = np.eye(state_dim, dtype=float)
    joseph_left = identity - kalman_gain @ observation_matrix
    posterior_state_covariance = project_to_psd(
        joseph_left @ predictive_state_covariance @ joseph_left.T
        + kalman_gain @ effective_observation_covariance @ kalman_gain.T
    )

    diagnostics = WoLFDiagnostics(
        weight=float(np.sqrt(weight_squared)),
        weight_squared=float(weight_squared),
        innovation_norm=float(innovation_norm),
        mahalanobis_squared=float(mahalanobis_squared),
        effective_observation_covariance=effective_observation_covariance,
    )
    update = KalmanUpdateResult(
        state_mean=posterior_state_mean,
        state_covariance=posterior_state_covariance,
        kalman_gain=kalman_gain,
        innovation=innovation,
    )
    return WoLFUpdateResult(update=update, diagnostics=diagnostics)
