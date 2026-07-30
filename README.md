# Adversarial State-Space Models

This repository studies adversarial attacks and online defenses for
state-space models, with experiments ranging from classical linear-Gaussian
filtering to nonlinear objectives, covariance adaptation, and RL agents that
act from noisy observations.

The most important idea in this repository is the separation between:

1. `shared_ssm/`, which is the core reusable library.
2. The other top-level folders, which are experiment suites and worked
   examples built on top of that library.

If you want to understand or extend the project, start with `shared_ssm/`.
If you want to reproduce figures or inspect concrete attack/defense workflows,
then move to the experiment folders.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run scripts from the repository root:

```bash
cd C:\Users\Usuario\Desktop\Doctoradooo\AdvSSMs\Code_AdvSSM
```

## Repository Philosophy

This is a script-oriented research repository with a shared library at its
center.

- `shared_ssm/` contains the reusable numerical and modeling building blocks.
- `AdvSSM/`, `AdvNonLinearAttack/`, `CovarianceAdaptation/`, `Gymnasium/`,
  and `RL/` contain concrete studies, benchmarks, and plotting scripts.
- Most experiment files have their own `main()`-style execution path and can
  be run directly.

That means the repository should be read in this order:

1. `shared_ssm/` to understand the common abstractions.
2. One experiment folder to see how those abstractions are applied in practice.
3. The corresponding `outputs/`, `data/`, `figures/`, or model folders if you
   want cached results or generated artifacts.

## The Core Library: `shared_ssm/`

`shared_ssm/` is the backbone of the repository. It centralizes the reusable
logic that would otherwise be duplicated across experiments.

At a high level, the package provides:

- Linear-Gaussian state-space model containers and validation.
- Kalman prediction and update recursions.
- Full-sequence filtering in online and offline modes.
- Optional RTS smoothing for offline inference.
- Leave-one-out and causal attack geometry in observation space.
- Analytic attacks for linear objectives over ellipsoidal constraints.
- Projected-gradient attacks for nonlinear expectations.
- Torch-based attacks for RL return objectives.
- Online defenses such as covariance adaptation and WoLF-style measurement
  updates.
- Shared SPD / PSD linear-algebra utilities and Gaussian likelihood helpers.
- Shared artifact and cache helpers for figures, data, and saved outputs.
- Compatibility helpers in `legacy.py` so older experiment scripts can keep
  their existing interfaces while reusing the shared implementation.

### Internal Map of `shared_ssm/`

```text
shared_ssm/
  __init__.py
    Main public API used by the experiments.

  linear_gaussian.py
    LGSSM model definitions, Kalman prediction/update, filtering, smoothing.

  geometry.py
    Shared attack geometry for online and offline observation attacks.

  attacks/
    linear.py      Analytic linear/KKT-style attacks.
    nonlinear.py   PGD-style attacks on E[g(s_t)] objectives.
    rl.py          Torch-based attacks for RL return objectives.

  defenses/
    covariance_adaptation.py   Online covariance adaptation utilities.
    wolf.py                    WoLF robust measurement-update utilities.

  constraints.py
    Ellipsoidal feasibility regions, Mahalanobis distances, projections.

  linalg.py
    Stable SPD/PSD helpers used throughout the repository.

  artifacts.py
    Shared output-directory and `.npz` cache helpers.

  results.py
    Helpers to inject an attacked observation and rerun inference.

  covariance_experiments.py
    Shared utilities used by covariance-adaptation experiment scripts.

  legacy.py
    Compatibility wrappers for older script conventions.
```

### Why `shared_ssm/` Matters

This folder is not just a helper collection. It is the place where the common
mathematical and numerical contract of the repository lives.

When different folders study:

- KKT attacks on linear SSMs,
- nonlinear attacks on posterior expectations,
- covariance-adaptation defenses,
- CartPole observation attacks,
- wind-navigation RL attacks and defenses,

they are all reusing the same shared notions of:

- state and observation conventions,
- Kalman inference,
- attack regions,
- Gaussian geometry,
- PSD stabilization,
- and output/cache organization.

So if you add new reusable functionality, it should usually go into
`shared_ssm/` first, and only then be consumed by a specific experiment
script.

### Typical Import Surface

Most experiment scripts use `shared_ssm` like a small library:

```python
from shared_ssm import LinearGaussianStateSpaceModel
from shared_ssm import build_attack_geometry
from shared_ssm import run_kalman_inference
from shared_ssm.attacks import solve_torch_rl_expectation_attack
from shared_ssm.defenses import run_online_covariance_adaptation
```

For package-specific details, see [`shared_ssm/README.md`](shared_ssm/README.md).

## Experiment Folders

The remaining top-level directories are best understood as example suites or
application domains built around the shared library.

### `AdvSSM/`

This folder contains the classical linear-Gaussian attack experiments. It is
the most direct illustration of the original adversarial SSM workflow:

- simulate a trajectory,
- run Kalman filtering and smoothing,
- remove or perturb one observation,
- constrain the perturbation to a plausible ellipsoid,
- and measure how the posterior state estimate changes.

Representative scripts:

- `AdvSSM/KKTOpt.py`: fixed-time KKT attack with the main multi-panel figure.
- `AdvSSM/KKTOpt_epsdep.py`: attack effect as a function of `epsilon`.
- `AdvSSM/KKTOpt_tdependent.py`: time-dependent attack studies.
- `AdvSSM/KKTOptSensitivity.py`: sensitivity analyses.
- `AdvSSM/Epsilon_direction.py`: geometry of attack directions across budgets.
- `AdvSSM/Regions.py`: feasible-region exploration.

In practice, `AdvSSM/` is the cleanest place to study the linear attack story
before moving to nonlinear or RL settings.

### `AdvNonLinearAttack/`

This folder extends the attack idea from linear state objectives to nonlinear
functionals such as `E[g(s_t) | o_t', o_-t]`.

The focus here is:

- white-box attacks on nonlinear expectations,
- projected gradient descent inside ellipsoidal attack regions,
- sensitivity and Jacobian-based analyses,
- 2D and 3D illustrative cases.

Representative scripts:

- `AdvNonLinearAttack/GradientAttack.py`
- `AdvNonLinearAttack/GradientAttackNoGrad.py`
- `AdvNonLinearAttack/GradientAttack3D.py`
- `AdvNonLinearAttack/AttackSense.py`
- `AdvNonLinearAttack/AttackSense3D.py`
- `AdvNonLinearAttack/AttackSense3D_CallSummary.py`

Conceptually, this folder shows how the same `shared_ssm` geometry can support
objectives that are no longer simple quadratic linear-state displacements.

### `CovarianceAdaptation/`

This folder contains experiments for online defenses based on adapting the
observation covariance when attacks are suspected.

The main themes are:

- defended Kalman filtering under attacked observations,
- lambda sweeps for directional covariance inflation,
- comparisons between clean, attacked, and defended trajectories,
- extension from linear settings to nonlinear `g(s_t)` studies.

Representative scripts:

- `CovarianceAdaptation/kf_covadapt_lambda_sweep.py`
- `CovarianceAdaptation/nonlinear_g_covadapt.py`

This directory is best read as the defense-focused counterpart to the attack
experiments in `AdvSSM/` and `AdvNonLinearAttack/`.

### `Gymnasium/`

This folder adapts the shared attack/defense ideas to Gymnasium-based control
problems, especially CartPole.

The emphasis is on:

- observation attacks in a standard control benchmark,
- CartPole covariance-adaptation and WoLF comparisons,
- benchmark-style outputs with figures, CSV summaries, and cached data.

Representative scripts:

- `Gymnasium/cartpole_defense_benchmark.py`
- `Gymnasium/cartpole_covadapt_compare_epsilons_wolf.py`
- `Gymnasium/sweep_cartpole_wolf_only.py`

This is the bridge between the core SSM machinery and a familiar benchmark
environment from control/RL tooling.

### `RL/`

This folder contains the RL experiments for the wind-navigation setting.
Here the attacked observation does not just change a posterior estimate; it can
change the agent's control decisions and long-horizon return.

The folder includes:

- environment and model setup for the wind-navigation task,
- online attacks on observations,
- defense benchmarks under matched randomness,
- WoLF tuning and comparison scripts,
- trajectory and value-function visualization utilities.

Representative scripts:

- `RL/wind_rl_setup.py`
- `RL/compare_wind_online_attack_rewards.py`
- `RL/defense_benchmark.py`
- `RL/wolf_benchmark.py`
- `RL/random_contour_benchmark.py`
- `RL/final_comparison.py`

If `AdvSSM/` shows the attack geometry in the simplest setting, `RL/` shows
the most application-driven end of the repository.

## Outputs and Cached Artifacts

Several folders contain generated artifacts such as:

- `outputs/figures/`
- `outputs/data/`
- `outputs/saved_models/`
- `RL/figures/`
- `RL/data/`
- `RL/model/`

The shared cache helpers live in `shared_ssm/artifacts.py`. They standardize:

- where figures are saved,
- where numerical `.npz` payloads are cached,
- and how expensive computations are reused when only the plotting layer
  changes.

This is especially useful for Monte Carlo studies, PGD-based attacks, and RL
benchmarks that are expensive to rerun.

## Suggested Starting Points

If you are new to the repository, a good reading order is:

1. `shared_ssm/README.md`
2. `shared_ssm/__init__.py`
3. `shared_ssm/linear_gaussian.py`
4. `shared_ssm/geometry.py`
5. One of the following example scripts, depending on your interest:
   - `AdvSSM/KKTOpt.py`
   - `AdvNonLinearAttack/GradientAttack.py`
   - `CovarianceAdaptation/kf_covadapt_lambda_sweep.py`
   - `Gymnasium/cartpole_defense_benchmark.py`
   - `RL/defense_benchmark.py`

## Example Commands

Linear-Gaussian fixed-time attack:

```bash
python AdvSSM\KKTOpt.py
```

Nonlinear expectation attack:

```bash
python AdvNonLinearAttack\GradientAttack.py
```

Covariance-adaptation lambda sweep:

```bash
python CovarianceAdaptation\kf_covadapt_lambda_sweep.py
```

CartPole defense benchmark:

```bash
python Gymnasium\cartpole_defense_benchmark.py
```

Wind-navigation RL defense benchmark:

```bash
python RL\defense_benchmark.py
```

## In One Sentence

`shared_ssm/` is the reusable library and mathematical core of the project;
the other folders are specialized experiment suites that demonstrate how to
apply that core to linear attacks, nonlinear attacks, online defenses,
Gymnasium benchmarks, and RL policies.
