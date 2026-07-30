# shared_ssm

Reusable linear-Gaussian state-space utilities for the AdvSSM repository.

## Canonical notation

This package adopts the same notation requested for the refactor:

1. `s_t` is the latent state.
2. `o_t` is the observation.
3. `a_{t-1}` is the control or RL action applied between `t-1` and `t`.

The canonical model is

```text
s_t = A_t s_{t-1} + B_t a_{t-1} + w_t
o_t = F_t s_t     + G_t a_{t-1} + v_t
```

with

```text
s_0 ~ N(m_0, P_0)
w_t ~ N(0, W_t)
v_t ~ N(0, V_t)
```

## Sequence convention used by the API

To keep the code explicit and compatible with RL:

1. `observations[k]` stores `o_{k+1}`.
2. `actions[k]` stores `a_k`, the action applied between times `k` and `k+1`.
3. Filtered and smoothed state arrays include the prior at time `t = 0`.

That means:

1. `filtered_state_means[0]` is the prior mean `m_0`.
2. `filtered_state_means[k + 1]` is the posterior mean at time `t = k + 1`.
3. `predictive_state_means[k]` is the predictive mean for time `t = k + 1`.
4. In RL language, `u_state[k]` or `actions[k]` is the policy action used in
   the transition `s_k -> s_{k+1}`.

## Current scope

The shared package includes:

1. SPD linear-algebra helpers in `linalg.py`.
2. A reusable LGSSM container in `linear_gaussian.py`.
3. Single-step Kalman prediction and update.
4. Full-sequence Kalman filtering in `online` or `offline` mode.
5. Optional RTS smoothing for offline inference.
6. Observation masking so later leave-one-out attacks can reuse the same filter.
7. Shared likelihood, log-likelihood, and Mahalanobis constraints.
8. Online/offline attack geometry.
9. Analytic linear attacks over ellipsoids.
10. Nonlinear projected-gradient attacks on `E[g(s_t)]`.
11. Torch-based RL attacks for differentiable posterior objectives.
12. Helpers that insert adversarial observations into a copy and rerun inference.
13. Online directional covariance adaptation that can receive attack targets,
    directions, or a target-building callback.

## Repository integration

The existing experiment folders now reuse this package through compatibility
wrappers while keeping their public function names stable:

1. `AdvSSM` uses the shared simulators, Kalman/RTS wrappers, LOO geometry, and
   analytic KKT attack helper.
2. `AdvNonLinearAttack` uses the shared current-control simulator, filters,
   attack regions, finite-difference Jacobians, and nonlinear attack helper.
3. `CovarianceAdaptation` uses the shared Kalman attack and online covariance
   adaptation helpers.
4. `Gymnasium` reuses the shared PSD and attack-region utilities where the
   CartPole covariance-adaptation script needs them.
5. `RL` uses the online posterior attack interface for wind-policy evaluation;
   the trajectory script sets `attack_prob = 0.1` inside `main`.

The old standalone utility/demo scripts that duplicated simulators, Kalman
filters, RTS smoothers, leave-one-out geometry, PSD helpers, artifact-cache
helpers, or covariance-adaptation helpers have been removed or compacted. New
reusable code should be added here instead of inside experiment scripts.
