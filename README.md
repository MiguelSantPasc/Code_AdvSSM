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

This folder evaluates observation attacks and KF/covariance-adaptation/WoLF
defenses on a **centered CartPole variant with physical end stops**. The videos
and benchmark entry points use the same adapted discrete DQN and shared plant.

#### Plant, reward and learned policy

The hidden state is $s_t=(x_t,\dot x_t,\theta_t,\dot\theta_t)$. The two actions
apply forces of $-10$ N and $+10$ N; they are not continuous actions. Cart end
stops constrain $x_t$ to $\pm2.4$ m. Inelastic contact stops outward cart motion
and transfers the collision impulse to the pole. Inward motion remains possible;
there is no pole/ground collision model.

The reward is evaluated at the resulting position:

$$
r_t=1-0.8\left(\frac{x_{t+1}}{2.4}\right)^2.
$$

It equals 1 at the center and 0.2 at either stop. Crossing $|\theta|>12^\circ$
terminates an episode; wall contact does not. The maximum episode length is
500 control steps. Return therefore differs from episode length, and success
means reaching the time limit without angle termination. The plant uses two
0.01 s Euler substeps per action; the EKF predictor uses one 0.02 s step.

`Gymnasium/train_cartpole_centered.py` initializes a separate DQN from the
original pretrained weights and trains it with this plant and reward. The
current checkpoint was selected at 80,000 environment steps during a
120,000-step training run, using validation seeds 901 and 902. Training uses
true-state feedback; evaluation feeds the filtered estimate into the policy.
It is not adversarial training. The original downloaded checkpoint is preserved.

The network outputs two real-valued estimates $Q_\phi(s,0),Q_\phi(s,1)$.
Executed actions use $\arg\max_a Q_\phi(m_t,a)$; there is no separate actor or
value network. The value proxy is $\widehat V(s)=\max_a Q_\phi(s,a)$, with DQN
discount $\gamma=0.99$. Policy weights remain fixed throughout attacks and
benchmarks. The centered checkpoint and its training metadata are in
`Gymnasium/outputs/saved_models/sb3_dqn_cartpole_centered_v1/`.

#### Current observation attack: hard scores, soft search gradient

Given the current prior $(m_T^-,P_T^-)$, first update the nominal KF using the
original noisy observation $o_T^{\mathrm{nom}}$:

$$
K_T=P_T^-(P_T^-+R)^{-1},\qquad
m_T^{\mathrm{nom}}=m_T^-+K_T(o_T^{\mathrm{nom}}-m_T^-).
$$

This reference is held fixed during the attack. It is an estimate, not the true
simulator state. Its prior can contain effects of earlier attacks: only the
current observation is unmanipulated in this reference update.

For both actions, enumerate the deterministic transition from that reference:

$$
\bar s_{T+1}^{a}=f(m_T^{\mathrm{nom}},a),\qquad
B_a=r(\bar s_{T+1}^{a})+
\gamma\,\mathbf 1_{\mathrm{continuation}}\max_b Q_\phi(\bar s_{T+1}^{a},b).
$$

The transition includes the same end stops and fine integration as the plant.
The indicator removes the bootstrap on angle termination or at the time limit.
Both $B_a$ stay constant during observation optimization.

An observation candidate $o$ must satisfy the predictive-ellipsoid constraint

$$
(o-m_T^-)^\top(P_T^-+R)^{-1}(o-m_T^-)\leq\rho_\epsilon.
$$

It induces the posterior mean $m_T(o)=m_T^-+K_T(o-m_T^-)$ and covariance
$P_T^+$, which is independent of the candidate under this nominal KF update.
With $N=64$, draw fixed base Gaussian samples $\xi_i$ and form
$s_i(o)=m_T(o)+L_T\xi_i$, where $L_TL_T^\top=P_T^+$. The samples move with
the candidate mean; the base random numbers are reused throughout the search.

Candidate scoring uses discrete actions:

$$
\widehat J_{\mathrm{hard}}(o)
=\frac1N\sum_i B_{\arg\max_a Q_\phi(s_i(o),a)}.
$$

Because this finite-sample score is piecewise constant, its argmax does not
supply the search gradient. Instead, differentiate the surrogate

$$
\widetilde J(o)=\frac1N\sum_i\sum_a
\operatorname{softmax}\!\left(Q_\phi(s_i(o),\cdot)/\tau_\pi\right)_a B_a,
\qquad \tau_\pi=1.0.
$$

Run 20 projected gradient steps with step size 0.34, starting from the original
noisy observation projected into the ellipsoid. Return the visited candidate
with the lowest **hard** score; the soft score only breaks ties. Temperature 1
is a fixed starting choice, not a calibrated optimum. This procedure is an
approximate local search, not a reachability certificate or a global optimizer.

The attacker scores actions across posterior samples, but the deployed agent
chooses its single action at the **posterior mean**. These objectives are not
identical: lowering the sampled score need not change the mean's greedy action.
Softmax is used only for search and does not make deployed actions stochastic.
For defended benchmark branches the attack is constructed using the nominal KF
update, then the selected defense processes the manipulated observation.

The random baseline samples uniformly inside the same ellipsoid; it is not
restricted to the boundary despite the legacy `random_contour` method key.

Defaults are attack probability 0.20 (excluding the initial observation),
$\rho_\epsilon\simeq9.49$ for 95% four-dimensional coverage, and observation
noise standard deviations $(0.10,0.22,0.05,0.22)$. Simulated sensor noise is
independent across components, whereas the filter assumes the correlated $R$
defined in each script. The covariance discount $\delta=0.94$ is separate from
both $\gamma$ and $\tau_\pi$; it is retained from the earlier calibration, not
recalibrated for the centered policy and stops.

#### Entry points and reproducibility

- `Gymnasium/cartpole_defense_benchmark.py`: 200 evaluation episodes per method;
  retunes 22 WoLF configurations on 6 separate tuning seeds for each of the
  no-attack, PGD and random-contour scenarios. Saves episode and diagnostic CSVs,
  summary tables, NPZ data, a WoLF selection JSON and figures.
  Selection completes before comparison starts. Eight process workers run
  independent seeded episodes by default (`--workers 1` runs serially); a
  three-step validation checked identical serial/worker results. The return
  figure preserves the two-panel attack/random layout of the earlier n200
  benchmark, and the dynamics figure shows the same two diagnostic densities.
- `Gymnasium/cartpole_covadapt_compare_epsilons_wolf.py`: 15 episodes per method
  with explicitly fixed WoLF reference thresholds; saves the grouped comparison
  and its data. These thresholds are not claimed to be the newly tuned optimum.
- `Gymnasium/sweep_cartpole_wolf_only.py`: 12 episodes per configuration, sweeping
  5 IMQ and 5 TMD thresholds for PGD and random perturbations.
- `Gymnasium/cartpole_videos.py`: seed-100 noisy+KF and noisy+attack videos at
  0.25x playback, with translucent observations, visible stops and a center band.
- `Gymnasium/cartpole_angle_density.py`: empirical true-pole-angle densities for
  those trajectories, counting each physical step once.

The benchmarks stop at the first episode failure. Videos alone continue the
closed loop after failure to display the physical fall, without adding reward;
the policy then operates outside its training termination range.

```powershell
.\.venv\Scripts\python.exe Gymnasium/cartpole_defense_benchmark.py
.\.venv\Scripts\python.exe Gymnasium/cartpole_covadapt_compare_epsilons_wolf.py
.\.venv\Scripts\python.exe Gymnasium/sweep_cartpole_wolf_only.py
```

Configuration values remain editable in each script. Benchmarks verify the
checkpoint's reward and physics metadata before running. New artifacts use a
`centered_...` identifier derived from the policy SHA-256, physical model and
experiment settings; full metadata accompanies the data. Legacy results and
WoLF selections are not reused for this variant. Returns from the old constant
reward benchmark are not directly comparable with centered-reward returns.

The 2026-09-29 WoLF rerun evaluated all 22 configurations in three scenarios,
with six tuning seeds per configuration/scenario (396 episodes). Its four
scenario-specific selections are:

| Scenario | IMQ soft threshold | TMD threshold |
|---|---:|---:|
| Estimated-return PGD | 0.30 | 2.8 |
| Random ellipsoid perturbation | 0.50 | 3.0 |

These are defense thresholds, distinct from the attack softmax temperature
$\tau_\pi=1$. Selection was checked against the full ranking table; the
serial and parallel tuning results agree within CSV rounding precision.

To reproduce this selection followed by the 200-episode, 14-method comparison:

```powershell
.\.venv\Scripts\python.exe Gymnasium/cartpole_defense_benchmark.py --mode wolf --n-episodes 200 --n-tuning-episodes 6 --pgd-steps 20 --mc-samples 64 --workers 8 --output-prefix cartpole_wolf_benchmark_centered_eps95_delta0p94_n200_tune6_pgd20_mc64
.\.venv\Scripts\python.exe Gymnasium/cartpole_defense_benchmark.py --mode benchmark --n-episodes 200 --n-tuning-episodes 6 --pgd-steps 20 --mc-samples 64 --workers 8 --output-prefix cartpole_defense_benchmark_centered_eps95_delta0p94_n200_tune6_pgd20_mc64
```

The comparison retains the return and diagnostic layouts of the earlier
`eps95_delta0p94_n200_tune6_pgd2_mc4` figures. Its new names use `centered` and
`pgd20_mc64` to identify the actual reward/plant and attack budget; the original
reference figures remain separate. The corresponding artifacts are:

- [WoLF tuning table](Gymnasium/outputs/data/cartpole_wolf_benchmark_centered_eps95_delta0p94_n200_tune6_pgd20_mc64_full.csv)
- [Selected WoLF parameters and metadata](Gymnasium/outputs/data/best_cartpole_wolf_params_centered_e80a52ed4706_n200_ntune6.json)
- [Return comparison](Gymnasium/outputs/figures/cartpole_defense_benchmark_centered_eps95_delta0p94_n200_tune6_pgd20_mc64_returns.png)
- [Diagnostic distributions](Gymnasium/outputs/figures/cartpole_defense_benchmark_centered_eps95_delta0p94_n200_tune6_pgd20_mc64_dynamics.png)

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
