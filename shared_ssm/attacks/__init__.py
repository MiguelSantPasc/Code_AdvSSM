"""Shared attack interfaces for linear-Gaussian SSM experiments."""

from .linear import LinearAttackResult
from .linear import solve_linear_state_attack
from .linear import solve_max_quadratic_over_ellipsoid
from .nonlinear import NonlinearAttackConfig
from .nonlinear import NonlinearAttackResult
from .nonlinear import estimate_expectation
from .nonlinear import finite_difference_jacobian
from .nonlinear import solve_nonlinear_expectation_attack
from .rl import TorchRLAttackConfig
from .rl import TorchRLAttackResult
from .rl import solve_torch_rl_expectation_attack

__all__ = [
    "LinearAttackResult",
    "NonlinearAttackConfig",
    "NonlinearAttackResult",
    "TorchRLAttackConfig",
    "TorchRLAttackResult",
    "estimate_expectation",
    "finite_difference_jacobian",
    "solve_linear_state_attack",
    "solve_max_quadratic_over_ellipsoid",
    "solve_nonlinear_expectation_attack",
    "solve_torch_rl_expectation_attack",
]
