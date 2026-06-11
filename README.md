# Adversarial State-Space Models

Research code for adversarial attacks on state-space models, with a focus on
Linear Gaussian State-Space Models, Kalman filtering, RTS smoothing, and RL
agents that act from noisy or adversarial observations.

The repository is script-oriented rather than package-oriented: each experiment
has a `main()` and can be run directly from the repository root.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use the repository root as the working directory:

```bash
cd C:\Users\Usuario\Desktop\Doctoradooo\AdvSSMs\Code_AdvSSM
```

## Repository Layout

```text
AdvSSM/
  Core Kalman/RTS, leave-one-out, KKT attack, sensitivity, and Monte Carlo scripts.

AdvNonLinearAttack/
  White-box attacks on nonlinear functions E[g(x_t) | y_t', y_-t].

RL/experiments/v2_wind/
  PPO-style point-agent with wind, plus clean/noisy/KF/adversarial evaluation.

RL/experiments/AdvRL_policy/
  Earlier no-wind RL point-agent experiments.

RL/experiments/others/
  Side experiments: Gaussian-sign bandit/POMDP, CartPole noise, contour plots.

RL/results/, AdvSSM/output/, AdvNonLinearAttack/output/
  Generated figures and cached experiment data.
```

## Core Ideas

Most SSM scripts use the model:

```text
x_{t+1} = A_t x_t + B_t u_t + w_t
y_t     = H_t x_t + D_t u_t + v_t
```

The attack workflow is:

1. Simulate or load a trajectory.
2. Run Kalman filtering and RTS smoothing.
3. Remove one observation `y_t` and compute `p(y_t | y_-t) = N(mu_t, Sigma_t)`.
4. Pick an adversarial observation `y_t*` inside the plausible ellipsoid
   `(y_t* - mu_t)^T Sigma_t^-1 (y_t* - mu_t) <= epsilon`.
5. Re-run smoothing or RL evaluation and compare the result.

For the KKT scripts, the attack usually maximizes:

```text
||X_t (y_t* - y_t)||^2
```

For the nonlinear scripts, the attack uses projected gradient descent to move:

```text
E[g(x_t) | y_t*, y_-t]
```

toward a target value.

## Cached Outputs

Plots should have matching data files whenever the script is expensive or used
for reported figures. The cache convention is:

```text
some_plot.png
some_plot.npz
```

If the `.npz` exists, the script loads the cached data and redraws the plot.
This lets you edit labels, colors, layouts, or figure style without rerunning
Monte Carlo, PGD, KF rollouts, or RL evaluation.

The shared cache helpers live in:

```text
AdvSSM/io_utils.py
```

Cached scripts currently include:

```text
AdvSSM/KKTOpt.py
AdvSSM/Epsilon_direction.py
AdvSSM/KKTOpt_epsdep.py
AdvSSM/KKTOpt_tdependent.py
AdvSSM/KKTOptSensitivity.py
AdvNonLinearAttack/GradientAttack.py
AdvNonLinearAttack/GradientAttackNoGrad.py
AdvNonLinearAttack/AttackSense.py
AdvNonLinearAttack/AttackSense3D.py
RL/experiments/v2_wind/AdvRL_wind_noise.py
RL/experiments/v2_wind/AdvRL_wind_KF.py
RL/experiments/v2_wind/AdvRL_wind_AttackSSM.py
RL/experiments/v2_wind/AdvRL_wind_AttackSSM_plottraj.py
```

Most scripts use a local `force_recompute = False` flag. Change it to `True`
inside the script, or delete the matching `.npz`, when you intentionally want
fresh data.

## Common Commands

One-shot 2D KKT attack:

```bash
python AdvSSM\KKTOpt.py
```

Multi-epsilon tangent geometry:

```bash
python AdvSSM\Epsilon_direction.py
```

Monte Carlo attack effect over epsilon:

```bash
python AdvSSM\KKTOpt_epsdep.py
```

Nonlinear attack on `E[g(x_t)]`:

```bash
python AdvNonLinearAttack\GradientAttack.py
python AdvNonLinearAttack\GradientAttackNoGrad.py
```

Train or load the wind RL policy and generate basic plots:

```bash
python RL\experiments\v2_wind\AdvRL_wind.py
```

Evaluate clean vs noisy vs KF/adversarial observations:

```bash
python RL\experiments\v2_wind\AdvRL_wind_noise.py
python RL\experiments\v2_wind\AdvRL_wind_KF.py
python RL\experiments\v2_wind\AdvRL_wind_AttackSSM.py
python RL\experiments\v2_wind\AdvRL_wind_AttackSSM_plottraj.py
```

## Notes For Future Changes

- Prefer adding shared helpers in `AdvSSM/io_utils.py` rather than repeating
  cache and output-directory code.
- Keep mathematical refactors conservative. Several scripts use slightly
  different control-indexing conventions, especially `u.shape == (T, n_u)` vs
  `u.shape == (T+1, n_u)`.
- Save the numerical arrays needed to recreate each figure before styling the
  plot. That keeps experiments reproducible and cheap to redraw.
- Generated model checkpoints under `RL/saved_models/` are ignored by git.
