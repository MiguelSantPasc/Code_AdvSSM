"""
Online directional covariance adaptation for linear-Gaussian SSM filters.

This module centralizes the defense logic that was previously specialized in
several experiment scripts. The repository now uses one consistent online
defense direction: the normalized observation innovation
`o_t - \\hat o_t`. Adversarial targets are still useful, but only as
reference information for the prior and posterior attack experts, not as the
inflation direction itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..linalg import project_to_psd
from ..linalg import quad_form_spd
from ..linalg import solve_spd
from ..linear_gaussian import KalmanInferenceResult
from ..linear_gaussian import KalmanPredictResult
from ..linear_gaussian import LinearGaussianStateSpaceModel
from ..linear_gaussian import ResolvedLinearGaussianStateSpaceModel
from ..linear_gaussian import kalman_predict_step


AttackTargetBuilder = Callable[
    [int, KalmanPredictResult],
    np.ndarray | None,
]
ObjectiveAttackScoreBuilder = Callable[
    [int, KalmanPredictResult, np.ndarray, np.ndarray | None, np.ndarray | None],
    float | None,
]


@dataclass(frozen=True)
class CovarianceAdaptationConfig:
    """Configuration for the online covariance-adaptation defense."""

    lam: float
    omega_h: float
    omega_o: float
    posterior_attack_threshold: float
    mahalanobis_epsilon: float = 1.0
    mahalanobis_evidence_weight: float = 0.5
    objective_evidence_weight: float = 0.5
    direction_source: str = "observed_innovation"
    direction_eps: float = 1e-10


@dataclass(frozen=True)
class CovarianceAdaptationDiagnostics:
    """Step-wise diagnostics emitted by the online defense."""

    prior_attack_probability: np.ndarray
    posterior_attack_probability: np.ndarray
    applied_attack_probability: np.ndarray
    hidden_risk_score: np.ndarray
    observation_plausibility_score: np.ndarray
    directions: np.ndarray
    original_observation_covariances: np.ndarray
    adapted_observation_covariances: np.ndarray
    predictive_observation_covariances: np.ndarray
    adapted_predictive_observation_covariances: np.ndarray
    adapted_kalman_gains: np.ndarray
    adversarial_targets: np.ndarray
    has_attack_information: np.ndarray


@dataclass(frozen=True)
class CovarianceAdaptationResult:
    """Filtered inference result and diagnostics for covariance adaptation."""

    inference: KalmanInferenceResult
    diagnostics: CovarianceAdaptationDiagnostics


def run_online_covariance_adaptation(
    *,
    model: LinearGaussianStateSpaceModel | ResolvedLinearGaussianStateSpaceModel,
    observations: np.ndarray,
    actions: np.ndarray | None,
    config: CovarianceAdaptationConfig,
    attack_targets: dict[int, np.ndarray] | None = None,
    attack_directions: dict[int, np.ndarray] | None = None,
    attack_target_builder: AttackTargetBuilder | None = None,
    objective_attack_score_builder: ObjectiveAttackScoreBuilder | None = None,
) -> CovarianceAdaptationResult:
    """
    Run causal Kalman filtering with directional covariance adaptation.

    The function supports three ways to supply attack-reference information:
    1. `attack_targets`: adversarial observations already computed elsewhere.
    2. `attack_directions`: optional legacy markers for attacked steps.
    3. `attack_target_builder`: callback that can compute an adversarial target
       from the current prediction.
    4. The defended direction itself is always the current innovation
       `o_t - \\hat o_t`, normalized when nonzero.
    5. `objective_attack_score_builder`: optional callback that returns an
       extra attack-evidence score in `[0, 1]` to be mixed with the
       Mahalanobis evidence in the posterior.
    """
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2:
        raise ValueError("observations must have shape (T, d_o).")
    if float(config.lam) < 0.0:
        raise ValueError("lam must be non-negative.")
    if not (0.0 <= float(config.posterior_attack_threshold) <= 1.0):
        raise ValueError("posterior_attack_threshold must lie in [0, 1].")
    if float(config.mahalanobis_epsilon) <= 0.0:
        raise ValueError("mahalanobis_epsilon must be positive.")
    if str(config.direction_source) not in {"attack_information", "observed_innovation"}:
        raise ValueError(
            "direction_source must be either 'attack_information' or "
            "'observed_innovation'."
        )
    if not np.isclose(
        float(config.mahalanobis_evidence_weight) + float(config.objective_evidence_weight),
        1.0,
        atol=1e-9,
    ):
        raise ValueError("Mahalanobis and objective evidence weights must sum to one.")

    num_steps = int(observations.shape[0])
    action_dim = None if actions is None else int(np.asarray(actions, dtype=float).shape[1])
    resolved_model = model.resolve(num_steps, action_dim) if isinstance(model, LinearGaussianStateSpaceModel) else model
    action_sequence = _prepare_actions(actions, num_steps=num_steps, action_dim=resolved_model.action_dim)

    state_dim = resolved_model.state_dim
    obs_dim = resolved_model.obs_dim

    predictive_state_means = np.zeros((num_steps, state_dim), dtype=float)
    predictive_state_covariances = np.zeros((num_steps, state_dim, state_dim), dtype=float)
    predicted_observation_means = np.zeros((num_steps, obs_dim), dtype=float)
    predicted_observation_covariances = np.zeros((num_steps, obs_dim, obs_dim), dtype=float)
    filtered_state_means = np.zeros((num_steps + 1, state_dim), dtype=float)
    filtered_state_covariances = np.zeros((num_steps + 1, state_dim, state_dim), dtype=float)
    kalman_gains = np.zeros((num_steps, state_dim, obs_dim), dtype=float)
    innovations = np.zeros((num_steps, obs_dim), dtype=float)

    prior_attack_probability = np.zeros(num_steps, dtype=float)
    posterior_attack_probability = np.zeros(num_steps, dtype=float)
    applied_attack_probability = np.zeros(num_steps, dtype=float)
    hidden_risk_score = np.zeros(num_steps, dtype=float)
    observation_plausibility_score = np.zeros(num_steps, dtype=float)
    directions = np.zeros((num_steps, obs_dim), dtype=float)
    original_observation_covariances = np.zeros((num_steps, obs_dim, obs_dim), dtype=float)
    adapted_observation_covariances = np.zeros((num_steps, obs_dim, obs_dim), dtype=float)
    adapted_predictive_observation_covariances = np.zeros((num_steps, obs_dim, obs_dim), dtype=float)
    adapted_kalman_gains = np.zeros((num_steps, state_dim, obs_dim), dtype=float)
    adversarial_targets = np.full((num_steps, obs_dim), np.nan, dtype=float)
    has_attack_information = np.zeros(num_steps, dtype=bool)

    filtered_state_means[0] = resolved_model.m0
    filtered_state_covariances[0] = resolved_model.P0
    identity = np.eye(state_dim, dtype=float)

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

        F_t = resolved_model.F[step_idx]
        V_original = project_to_psd(resolved_model.V[step_idx])
        S_original = project_to_psd(predict_result.observation_covariance)
        observation = observations[step_idx]
        innovation = observation - predict_result.observation_mean

        target = _resolve_attack_target(
            step_idx=step_idx,
            predict_result=predict_result,
            attack_targets=attack_targets,
            attack_target_builder=attack_target_builder,
        )
        has_reference_information = bool(target is not None) or (
            attack_directions is not None and int(step_idx) in attack_directions
        )
        direction = _resolve_attack_direction(
            prediction_mean=predict_result.observation_mean,
            observation=observation,
            direction_eps=float(config.direction_eps),
        )

        V_adapted = V_original.copy()
        S_adapted = S_original.copy()
        posterior_probability = 0.0
        applied_probability = 0.0
        prior_probability = 0.0
        hidden_score = 0.0
        obs_score = 0.0

        if has_reference_information and direction is not None:
            has_attack_information[step_idx] = True
            directions[step_idx] = direction
            delta_reference = innovation if target is None else np.asarray(target, dtype=float).reshape(obs_dim) - np.asarray(
                predict_result.observation_mean,
                dtype=float,
            ).reshape(obs_dim)
            prior_probability, hidden_score, obs_score = compute_contamination_prior(
                delta_adv=delta_reference,
                predictive_observation_covariance=S_original,
                predictive_state_covariance=predict_result.state_covariance,
                observation_matrix=F_t,
                omega_h=float(config.omega_h),
                omega_o=float(config.omega_o),
            )
            if target is not None:
                target = np.asarray(target, dtype=float).reshape(obs_dim)
                adversarial_targets[step_idx] = target
            objective_attack_score = 0.0
            if objective_attack_score_builder is not None:
                built_score = objective_attack_score_builder(
                    int(step_idx),
                    predict_result,
                    np.asarray(observation, dtype=float).reshape(obs_dim),
                    None if target is None else np.asarray(target, dtype=float).reshape(obs_dim),
                    np.asarray(direction, dtype=float).reshape(obs_dim),
                )
                if built_score is not None:
                    objective_attack_score = clip_attack_evidence_score(float(built_score))
            mahalanobis_attack_score = mahalanobis_attack_evidence(
                innovation=innovation,
                predictive_observation_covariance=S_original,
                epsilon=float(config.mahalanobis_epsilon),
            )
            posterior_probability = posterior_attack_probability_from_evidence(
                prior_probability=float(prior_probability),
                mahalanobis_attack_score=float(mahalanobis_attack_score),
                objective_attack_score=float(objective_attack_score),
                mahalanobis_weight=float(config.mahalanobis_evidence_weight),
                objective_weight=float(config.objective_evidence_weight),
            )
            applied_probability = (
                posterior_probability
                if posterior_probability >= float(config.posterior_attack_threshold)
                else 0.0
            )

            V_adapted = rank_one_covariance_update(
                V_original,
                float(config.lam),
                direction,
                weight=applied_probability,
            )
            S_adapted = project_to_psd(F_t @ predict_result.state_covariance @ F_t.T + V_adapted)

        K_adapted = solve_spd(S_adapted, F_t @ predict_result.state_covariance.T).T
        filtered_state_means[step_idx + 1] = predict_result.state_mean + K_adapted @ innovation
        joseph_left = identity - K_adapted @ F_t
        filtered_state_covariances[step_idx + 1] = project_to_psd(
            joseph_left @ predict_result.state_covariance @ joseph_left.T
            + K_adapted @ V_adapted @ K_adapted.T
        )

        kalman_gains[step_idx] = K_adapted
        adapted_kalman_gains[step_idx] = K_adapted
        innovations[step_idx] = innovation
        prior_attack_probability[step_idx] = prior_probability
        posterior_attack_probability[step_idx] = posterior_probability
        applied_attack_probability[step_idx] = applied_probability
        hidden_risk_score[step_idx] = hidden_score
        observation_plausibility_score[step_idx] = obs_score
        original_observation_covariances[step_idx] = V_original
        adapted_observation_covariances[step_idx] = V_adapted
        adapted_predictive_observation_covariances[step_idx] = S_adapted

    inference = KalmanInferenceResult(
        mode="online",
        observations=observations,
        actions=action_sequence,
        observation_mask=np.ones(num_steps, dtype=bool),
        predictive_state_means=predictive_state_means,
        predictive_state_covariances=predictive_state_covariances,
        predicted_observation_means=predicted_observation_means,
        predicted_observation_covariances=predicted_observation_covariances,
        filtered_state_means=filtered_state_means,
        filtered_state_covariances=filtered_state_covariances,
        kalman_gains=kalman_gains,
        innovations=innovations,
    )
    diagnostics = CovarianceAdaptationDiagnostics(
        prior_attack_probability=prior_attack_probability,
        posterior_attack_probability=posterior_attack_probability,
        applied_attack_probability=applied_attack_probability,
        hidden_risk_score=hidden_risk_score,
        observation_plausibility_score=observation_plausibility_score,
        directions=directions,
        original_observation_covariances=original_observation_covariances,
        adapted_observation_covariances=adapted_observation_covariances,
        predictive_observation_covariances=predicted_observation_covariances,
        adapted_predictive_observation_covariances=adapted_predictive_observation_covariances,
        adapted_kalman_gains=adapted_kalman_gains,
        adversarial_targets=adversarial_targets,
        has_attack_information=has_attack_information,
    )
    return CovarianceAdaptationResult(inference=inference, diagnostics=diagnostics)


def compute_contamination_prior(
    *,
    delta_adv: np.ndarray,
    predictive_observation_covariance: np.ndarray,
    predictive_state_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    omega_h: float,
    omega_o: float,
) -> tuple[float, float, float]:
    """
    Compute the prior attack probability and its hidden/observation scores.

    The observation score favors plausible perturbations; the hidden score
    favors perturbations with a large effect after the nominal Kalman gain.
    """
    delta_adv = np.asarray(delta_adv, dtype=float).reshape(-1)
    S_t = project_to_psd(np.asarray(predictive_observation_covariance, dtype=float))
    P_t = project_to_psd(np.asarray(predictive_state_covariance, dtype=float))
    F_t = np.asarray(observation_matrix, dtype=float)

    obs_mahal_sq = quad_form_spd(S_t, delta_adv)
    observation_score = float(np.exp(-0.5 * obs_mahal_sq))

    nominal_gain = solve_spd(S_t, F_t @ P_t.T).T
    delta_state = nominal_gain @ delta_adv
    hidden_mahal_sq = quad_form_spd(P_t, delta_state)
    hidden_score = float(1.0 - np.exp(-0.5 * hidden_mahal_sq))

    prior_probability = float(np.clip(float(omega_h) * hidden_score + float(omega_o) * observation_score, 0.0, 1.0))
    return prior_probability, hidden_score, observation_score


def clip_attack_evidence_score(score: float) -> float:
    """Return one attack-evidence score clipped to `[0, 1]`."""
    return float(np.clip(float(score), 0.0, 1.0))


def mahalanobis_attack_evidence(
    *,
    innovation: np.ndarray,
    predictive_observation_covariance: np.ndarray,
    epsilon: float,
) -> float:
    """Return the bounded Mahalanobis attack score `a_md` in `[0, 1]`."""
    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    predictive_observation_covariance = project_to_psd(
        np.asarray(predictive_observation_covariance, dtype=float)
    )
    epsilon = max(float(epsilon), 1e-12)
    mahalanobis_squared = quad_form_spd(predictive_observation_covariance, innovation)
    return clip_attack_evidence_score(float(mahalanobis_squared / (epsilon + mahalanobis_squared)))


def combined_attack_evidence(
    *,
    mahalanobis_attack_score: float,
    objective_attack_score: float,
    mahalanobis_weight: float,
    objective_weight: float,
) -> float:
    """Return the mixed attack score from the two bounded evidence terms."""
    if not np.isclose(float(mahalanobis_weight) + float(objective_weight), 1.0, atol=1e-9):
        raise ValueError("Mahalanobis and objective evidence weights must sum to one.")
    return clip_attack_evidence_score(
        float(mahalanobis_weight) * clip_attack_evidence_score(mahalanobis_attack_score)
        + float(objective_weight) * clip_attack_evidence_score(objective_attack_score)
    )


def posterior_attack_probability_from_evidence(
    *,
    prior_probability: float,
    mahalanobis_attack_score: float,
    objective_attack_score: float,
    mahalanobis_weight: float,
    objective_weight: float,
) -> float:
    """Return the posterior attack probability from prior plus mixed evidence."""
    prior_probability = clip_attack_evidence_score(prior_probability)
    attack_score = combined_attack_evidence(
        mahalanobis_attack_score=float(mahalanobis_attack_score),
        objective_attack_score=float(objective_attack_score),
        mahalanobis_weight=float(mahalanobis_weight),
        objective_weight=float(objective_weight),
    )
    clean_score = float(1.0 - attack_score)
    numerator = float(prior_probability) * float(attack_score)
    denominator = numerator + float(1.0 - prior_probability) * float(clean_score)
    return float(numerator / max(denominator, 1e-12))


def rank_one_covariance_update(
    covariance: np.ndarray,
    lam: float,
    direction: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """Return `covariance + lam * weight * direction direction^T`, projected to PSD."""
    covariance = np.asarray(covariance, dtype=float)
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if float(lam) <= 0.0 or float(weight) <= 0.0 or float(np.linalg.norm(direction)) < 1e-12:
        return project_to_psd(covariance)
    unit_direction = direction / float(np.linalg.norm(direction))
    return project_to_psd(covariance + float(lam) * float(weight) * np.outer(unit_direction, unit_direction))


def compute_adapted_observation_covariance(
    *,
    observation_covariance: np.ndarray,
    lam: float,
    gamma: float,
    direction: np.ndarray,
    gamma_threshold: float,
) -> tuple[np.ndarray, float]:
    """
    Return the defended observation covariance and the applied `gamma_bar_t`.

    This helper exposes the exact measurement-only inflation used by the online
    covariance-adaptation defense:

        V_adapted_t = V_t + lam * gamma_bar_t * u_t u_t^T

    where `gamma_bar_t = gamma_t` above the threshold and zero otherwise.
    """
    if float(gamma_threshold) < 0.0 or float(gamma_threshold) > 1.0:
        raise ValueError("gamma_threshold must lie in [0, 1].")

    gamma_bar = float(gamma) if float(gamma) >= float(gamma_threshold) else 0.0
    adapted = rank_one_covariance_update(
        observation_covariance,
        float(lam),
        direction,
        weight=float(gamma_bar),
    )
    return adapted, gamma_bar


def log_mixture_posterior_weight(
    *,
    prior_probability: float,
    log_clean: float,
    log_attack: float,
    eps: float = 1e-12,
) -> float:
    """Return the posterior attack probability using a stable log mixture."""
    prior_probability = float(np.clip(prior_probability, eps, 1.0 - eps))
    log_num = np.log(prior_probability) + float(log_attack)
    log_den = np.logaddexp(np.log1p(-prior_probability) + float(log_clean), log_num)
    return float(np.exp(log_num - log_den))


def _resolve_attack_target(
    *,
    step_idx: int,
    predict_result: KalmanPredictResult,
    attack_targets: dict[int, np.ndarray] | None,
    attack_target_builder: AttackTargetBuilder | None,
) -> np.ndarray | None:
    """Return the adversarial target for one step, if available."""
    if attack_targets is not None and int(step_idx) in attack_targets:
        return np.asarray(attack_targets[int(step_idx)], dtype=float)
    if attack_target_builder is not None:
        target = attack_target_builder(int(step_idx), predict_result)
        if target is not None:
            return np.asarray(target, dtype=float)
    return None


def _resolve_attack_direction(
    *,
    prediction_mean: np.ndarray,
    observation: np.ndarray,
    direction_eps: float,
) -> np.ndarray | None:
    """Return the unit defense direction aligned with the current innovation."""
    raw_direction = np.asarray(observation, dtype=float).reshape(-1) - np.asarray(
        prediction_mean,
        dtype=float,
    ).reshape(-1)
    norm_value = float(np.linalg.norm(raw_direction))
    if norm_value < float(direction_eps):
        return None
    return raw_direction / norm_value


def _prepare_actions(
    actions: np.ndarray | None,
    *,
    num_steps: int,
    action_dim: int,
) -> np.ndarray:
    """Return a validated action array for the online defense loop."""
    if actions is None:
        return np.zeros((num_steps, action_dim), dtype=float)
    actions = np.asarray(actions, dtype=float)
    if actions.shape != (num_steps, action_dim):
        raise ValueError(f"actions must have shape ({num_steps}, {action_dim}).")
    return actions
