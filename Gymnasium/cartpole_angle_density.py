#!/usr/bin/env python3
"""Compare smooth true CartPole angle and position densities across paired seeds.

Protocol
--------
Reuse cartpole_videos.simulate_episode for noisy observations without attack,
PGD attack without defense, and PGD attack with covariance adaptation at
lambda=2.0. The thirty default seeds are 100--129. Each condition shares the
initial seed, observation noise stream, DQN, plant, and total horizon. Keep
the videos' controlled continuation beyond the standard 12-degree threshold;
stop at a horizontal pole or the shared 500-step horizon, whichever comes first.

Only the actual hidden state s_t is sampled, never observations or estimates.
Exclude the reset state and include each resulting physical state once.
Video pauses and repeated frames do not contribute samples. Each seed has
equal weight within each condition, irrespective of its trajectory length.
These are empirical time distributions, not confidence intervals.

Figure and output
-----------------
Use two side-by-side Gaussian kernel density estimates for all three cases:
angle in degrees (displayed from -18 to 18) and cart position in meters.
Normalize by ALL samples of each seed before cropping the angle display, so
out-of-view angles do not artificially inflate the visible density. Use the
same bandwidth across cases: 1 degree for angle and 0.15 m for position.
Each full KDE integrates to one over the real line; Gaussian smoothing can
extend slightly beyond physical limits. Report the raw out-of-view sample
mass in the console. Dashed angle lines mark +/-12 degrees.
English labels, pastel colors, inside legends, and no panel titles are used.
Save a PNG figure and the raw per-seed angle/position samples in an NPZ.

All experiment defaults are editable in main(). Run from the repository root:
    .\\.venv\\Scripts\\python.exe Gymnasium/cartpole_angle_density.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cartpole_videos import VideoConfig, cartpole, simulate_episode


def seed_balanced_density(samples_by_seed: list[np.ndarray], grid: np.ndarray,
                          *, bandwidth: float) -> np.ndarray:
    """Average Gaussian KDEs with equal seed weights and a fixed bandwidth.

    Evaluating one trajectory at a time bounds memory use. An explicit Gaussian
    mixture also supports constant or single-sample trajectories, unlike a KDE
    that estimates bandwidth from a potentially singular sample covariance.
    """
    grid = np.asarray(grid, dtype=float)
    if not samples_by_seed or grid.ndim != 1 or grid.size < 2:
        raise ValueError("Provide nonempty trajectories and a one-dimensional grid.")
    if not np.all(np.isfinite(grid)) or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Provide a finite grid and a positive finite bandwidth.")
    density = np.zeros_like(grid)
    for samples in samples_by_seed:
        values = np.asarray(samples, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Each seed must provide nonempty, finite samples.")
        standardized = (grid[:, None] - values[None, :]) / bandwidth
        # Normalize before limiting the display; out-of-view samples still count.
        density += np.exp(-0.5 * standardized**2).mean(axis=1) / (
            bandwidth * np.sqrt(2 * np.pi)
        )
    return density / len(samples_by_seed)


def plot_densities(samples_by_case: dict[str, list[np.ndarray]], *,
                    labels: dict[str, str], colors: tuple[str, ...],
                    angle_grid: np.ndarray, position_grid: np.ndarray,
                    angle_bandwidth: float, position_bandwidth: float,
                    failure_angle: float, output: Path) -> None:
    """Plot smooth angle/position densities using shared bandwidths per panel."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    try:
        for ax, column, grid, bandwidth, xlabel, ylabel in (
            (axes[0], 0, angle_grid, angle_bandwidth,
             r"Angle of $s_t$ (deg)", r"Density (deg$^{-1}$)"),
            (axes[1], 1, position_grid, position_bandwidth,
             r"Position of $s_t$ (m)", r"Density (m$^{-1}$)"),
        ):
            for (case, trajectories), color in zip(samples_by_case.items(), colors):
                density = seed_balanced_density(
                    [trajectory[:, column] for trajectory in trajectories], grid,
                    bandwidth=bandwidth,
                )
                ax.fill_between(grid, density, color=color, alpha=0.20)
                ax.plot(grid, density, color=color, linewidth=2.2, label=labels[case])
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlim(float(grid[0]), float(grid[-1]))
            # Leave space above the curves for the inside legend.
            ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E6E6EC", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.legend(loc="upper right", fontsize=9, frameon=True,
                      facecolor="white", edgecolor="#E6E6EC")
        axes[0].set_xticks(np.arange(angle_grid[0], angle_grid[-1] + 0.1, 6))
        for threshold in (-failure_angle, failure_angle):
            axes[0].axvline(threshold, color="#858593", linestyle="--", linewidth=1.1)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=240)
    finally:
        plt.close(fig)


def main() -> None:
    """Run thirty paired seeds for each of the three video conditions."""
    seed0 = 100
    n_seeds = 30
    max_steps = 500
    device = "cpu"
    obs_noise_std = np.array([0.10, 0.22, 0.05, 0.22], dtype=float)
    discount_delta = 0.94
    attack_prob = 0.20
    attack_eps = 9.49
    pgd_steps = 20
    pgd_step_size = 0.34
    mc_samples = 64
    policy_temperature = 1.0
    defense_lambda = 2.0
    omega_h = 0.50
    omega_o = 0.50
    gamma_threshold = 0.20
    filter_tau = 0.02
    real_tau = 0.01
    wall_position = 2.4
    wall_restitution = 0.0
    center_reward_weight = 0.8
    post_failure_seconds = 10.0
    angle_grid = np.linspace(-18.0, 18.0, 801)
    position_grid = np.linspace(-2.5, 2.5, 801)
    angle_bandwidth = 1.0  # Degrees; shared across all three cases.
    position_bandwidth = 0.15  # Meters; shared across all three cases.
    failure_angle_degrees = 12.0
    colors = ("#8BBBDD", "#DF9DA8", "#91BFA8")
    cases = (("noisy_kf", False, False), ("noisy_attack", True, False),
             ("noisy_attack_defense", True, True))
    labels = {
        "noisy_kf": "No attack",
        "noisy_attack": "Attack without defense",
        "noisy_attack_defense": rf"Attack with defense ($\lambda={defense_lambda:.1f}$)",
    }
    meas_corr = np.array([
        [1.00, 0.18, 0.06, 0.00],
        [0.18, 1.00, 0.14, 0.22],
        [0.06, 0.14, 1.00, 0.18],
        [0.00, 0.22, 0.18, 1.00],
    ], dtype=float)
    directory = Path(__file__).resolve().parent
    model_path = directory / "outputs/saved_models/sb3_dqn_cartpole_centered_v1/dqn-CartPole-centered.zip"
    lambda_tag = str(float(defense_lambda)).replace(".", "p")
    stem = f"cartpole_angle_position_density_n{n_seeds}_seed{seed0}_lambda{lambda_tag}"
    output = directory / "outputs/figures" / f"{stem}.png"
    data_output = directory / "outputs/data" / f"{stem}.npz"

    torch.set_num_threads(1)
    config = VideoConfig(
        seed0, max_steps, obs_noise_std, discount_delta, attack_prob, attack_eps,
        pgd_steps, pgd_step_size, mc_samples, device,
        post_failure_steps=round(post_failure_seconds / filter_tau),
        policy_temperature=policy_temperature, defense_lambda=defense_lambda,
        omega_h=omega_h, omega_o=omega_o, gamma_threshold=gamma_threshold,
    )
    ssm = cartpole.build_cartpole_linear_ssm(
        filter_tau=filter_tau, real_tau=real_tau, wall_position=wall_position,
        wall_restitution=wall_restitution, center_reward_weight=center_reward_weight,
    )
    from train_cartpole_centered import verify_centered_checkpoint
    verify_centered_checkpoint(model_path, ssm=ssm)
    scale = np.diag(obs_noise_std).astype(np.float32)
    R = cartpole.project_to_psd((scale @ meas_corr @ scale).astype(np.float32))
    model = cartpole.load_cartpole_policy(str(model_path), torch.device(device))
    model.policy.set_training_mode(False)
    samples_by_case = {case: [] for case, _, _ in cases}
    # Store numeric arrays only, so the archive can be opened without pickle.
    archive = {
        "seeds": np.arange(seed0, seed0 + n_seeds),
        "columns": np.array(["angle_degrees", "position_meters"]),
        "angle_grid": angle_grid, "position_grid": position_grid,
        "angle_bandwidth": np.array(angle_bandwidth),
        "position_bandwidth": np.array(position_bandwidth),
        "defense_lambda": np.array(defense_lambda),
        "weighting": np.array("equal weight per seed; full-trajectory normalization"),
    }
    try:
        for seed in range(seed0, seed0 + n_seeds):
            for case, with_attack, with_defense in cases:
                snapshots = simulate_episode(
                    model=model, ssm=ssm, R=R, config=replace(config, seed=seed),
                    with_attack=with_attack, with_defense=with_defense,
                )
                # Reset is excluded; every resulting physical state occurs once.
                states = np.asarray([snapshot.state for snapshot in snapshots[1:]])
                samples = np.column_stack((np.degrees(states[:, 2]), states[:, 0]))
                samples_by_case[case].append(samples)
                archive[f"{case}_seed{seed}"] = samples
                print(f"seed={seed}; {case}; steps={len(samples)}; "
                      f"return={snapshots[-1].reward:.2f}", flush=True)
    finally:
        if model.get_env() is not None:
            model.get_env().close()
    data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_output, **archive)
    plot_densities(samples_by_case, labels=labels, colors=colors,
                    angle_grid=angle_grid, position_grid=position_grid,
                    angle_bandwidth=angle_bandwidth, position_bandwidth=position_bandwidth,
                    failure_angle=failure_angle_degrees, output=output)
    for case, trajectories in samples_by_case.items():
        outside = np.mean([np.mean((values[:, 0] < angle_grid[0]) |
                                  (values[:, 0] > angle_grid[-1]))
                           for values in trajectories])
        print(f"{case}: {len(trajectories)} seeds; "
              f"{100 * outside:.2f}% of seed-balanced angle mass outside [-18, 18] deg")
    print(f"Saved {output}\nSaved {data_output}", flush=True)


if __name__ == "__main__":
    main()
