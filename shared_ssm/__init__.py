"""
Shared linear-Gaussian state-space utilities for the AdvSSM repository.

This package centralizes the reusable numerical pieces that were previously
reimplemented across `AdvSSM`, `AdvNonLinearAttack`, `CovarianceAdaptation`,
and the RL/Gymnasium experiments.

Notation used across the package:
1. `s_t` is the latent state.
2. `o_t` is the observation.
3. `a_{t-1}` is the action or control applied between `t-1` and `t`.
4. The model therefore evolves as:
       s_t = A_t s_{t-1} + B_t a_{t-1} + w_t
       o_t = F_t s_t     + G_t a_{t-1} + v_t
   for `t = 1, ..., T`, with `s_0 ~ N(m_0, P_0)`.

The package exposes:
1. stable SPD linear-algebra helpers,
2. a reusable linear-Gaussian SSM model container,
3. Kalman prediction and update steps,
4. full-sequence filtering in `online` and `offline` modes,
5. optional RTS smoothing for offline inference,
6. shared online/offline attack geometry,
7. linear, nonlinear, RL, and covariance-adaptation helpers.
"""

from .linalg import gaussian_logpdf
from .linalg import project_to_psd
from .linalg import quad_form_spd
from .linalg import solve_spd
from .linalg import spd_inverse
from .linalg import sqrtm_psd
from .linalg import stabilized_cholesky
from .linalg import symmetrize
from .constraints import EllipsoidConstraint
from .constraints import gaussian_likelihood
from .constraints import gaussian_log_likelihood
from .constraints import mahalanobis_distance_squared
from .constraints import project_to_ellipsoid
from .artifacts import cached_npz
from .artifacts import data_dir_for
from .artifacts import data_path_for_plot
from .artifacts import figures_dir_for
from .artifacts import load_npz
from .artifacts import save_npz
from .geometry import AttackGeometry
from .geometry import build_attack_geometry
from .defenses import CovarianceAdaptationConfig
from .defenses import CovarianceAdaptationDiagnostics
from .defenses import CovarianceAdaptationResult
from .defenses import clip_attack_evidence_score
from .defenses import combined_attack_evidence
from .defenses import WoLFConfig
from .defenses import WoLFDiagnostics
from .defenses import WoLFUpdateResult
from .defenses import compute_adapted_observation_covariance
from .defenses import compute_contamination_prior
from .defenses import mahalanobis_attack_evidence
from .defenses import posterior_attack_probability_from_evidence
from .defenses import rank_one_covariance_update
from .defenses import run_online_covariance_adaptation
from .defenses import run_wolf_measurement_update
from .defenses import wolf_imq_weight_squared
from .defenses import wolf_tmd_weight_squared
from .linear_gaussian import KalmanInferenceResult
from .linear_gaussian import KalmanPredictResult
from .linear_gaussian import KalmanUpdateResult
from .linear_gaussian import LinearGaussianStateSpaceModel
from .linear_gaussian import ResolvedLinearGaussianStateSpaceModel
from .linear_gaussian import kalman_predict_step
from .linear_gaussian import kalman_update_step
from .linear_gaussian import predict_observation_distribution
from .linear_gaussian import run_kalman_inference
from .results import AttackApplicationResult
from .results import apply_attack_and_rerun
from .results import replace_observation

__all__ = [
    "AttackApplicationResult",
    "AttackGeometry",
    "CovarianceAdaptationConfig",
    "CovarianceAdaptationDiagnostics",
    "CovarianceAdaptationResult",
    "EllipsoidConstraint",
    "KalmanInferenceResult",
    "KalmanPredictResult",
    "KalmanUpdateResult",
    "LinearGaussianStateSpaceModel",
    "ResolvedLinearGaussianStateSpaceModel",
    "WoLFConfig",
    "WoLFDiagnostics",
    "WoLFUpdateResult",
    "clip_attack_evidence_score",
    "combined_attack_evidence",
    "compute_adapted_observation_covariance",
    "compute_contamination_prior",
    "gaussian_logpdf",
    "gaussian_log_likelihood",
    "gaussian_likelihood",
    "build_attack_geometry",
    "cached_npz",
    "data_dir_for",
    "data_path_for_plot",
    "figures_dir_for",
    "kalman_predict_step",
    "kalman_update_step",
    "load_npz",
    "mahalanobis_attack_evidence",
    "mahalanobis_distance_squared",
    "posterior_attack_probability_from_evidence",
    "predict_observation_distribution",
    "project_to_ellipsoid",
    "project_to_psd",
    "quad_form_spd",
    "rank_one_covariance_update",
    "apply_attack_and_rerun",
    "replace_observation",
    "run_online_covariance_adaptation",
    "run_wolf_measurement_update",
    "run_kalman_inference",
    "save_npz",
    "solve_spd",
    "spd_inverse",
    "sqrtm_psd",
    "stabilized_cholesky",
    "symmetrize",
    "wolf_imq_weight_squared",
    "wolf_tmd_weight_squared",
]
