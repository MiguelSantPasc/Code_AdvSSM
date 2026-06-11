#!/usr/bin/env python3
"""
GradientAttackNoGrad3D.py

3D counterpart of the finite-difference attack script.

It reuses the 3D experiment/plotting pipeline from `GradientAttack3D.py`, but
forces `g_grad=None` so the optimizer estimates the Jacobian of `g` via
central finite differences.
"""

from __future__ import annotations

import os

try:
    from GradientAttack3D import run_attack_experiment
except ModuleNotFoundError:
    import sys

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)

    from GradientAttack3D import run_attack_experiment


def main() -> None:
    run_attack_experiment(use_analytic_grad=False)


if __name__ == "__main__":
    main()
