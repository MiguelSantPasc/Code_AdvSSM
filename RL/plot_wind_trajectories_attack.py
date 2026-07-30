#!/usr/bin/env python3
"""
plot_wind_trajectories_attack.py

Plot wind-navigation trajectories under an online observation attack.

Why this script exists:
1. The reward bar plots show aggregate degradation, but they do not show how an
   attack changes the actual path followed by the agent.
2. This script collects trajectories using the same online attack machinery as
   the final RL comparison, especially the `estimated-return PGD` attack.
3. Each subplot marks the real state at attacked decision times and the
   corresponding perceived state used by the policy, making the sensor-attack
   mechanism visible.
4. Figures are saved in `RL/figures/`; numerical trajectory arrays are saved in
   `RL/data/` so the figures directory remains image-only.
"""

from __future__ import annotations

from dataclasses import replace
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


RL_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_MODULE_DIR)
if RL_MODULE_DIR not in sys.path:
    sys.path.insert(0, RL_MODULE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compare_wind_online_attack_rewards import coverage_to_mahalanobis_epsilon
from compare_wind_online_attack_rewards import current_policy_state
from compare_wind_online_attack_rewards import is_inside_ellipsoid
from compare_wind_online_attack_rewards import is_on_ellipsoid_contour
from compare_wind_online_attack_rewards import noisy_state_observation
from wind_rl_setup import WindNavigationBatch
from wind_rl_setup import build_transition_matrix
from wind_rl_setup import default_device
from wind_rl_setup import load_agent_checkpoint
from wind_rl_setup import rl_data_dir
from wind_rl_setup import rl_figures_dir
from wind_rl_setup import rl_model_dir


def set_plot_theme() -> None:
    """Apply a compact pastel style for attacked-trajectory figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9.6,
            "axes.labelsize": 10.6,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.68,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    """Style one trajectory panel while keeping the legend inside the axes."""
    ax.set_facecolor("#FCFCFD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.22, linewidth=0.68)
    ax.grid(True, which="minor", alpha=0.09, linewidth=0.42)
    ax.set_axisbelow(True)


def collect_online_attack_trajectory(
    *,
    policy: torch.nn.Module,
    simulation_config: Any,
    filter_config: Any,
    attack_setting: str,
    coverage: float,
    seed: int,
    observation_noise_std: float,
    gamma: float,
    attack_prob: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    real_attack_transition_samples: int,
    heavy_tail_degrees_of_freedom: float,
    uniform_annulus_min_coverage: float,
    contour_relative_tolerance: float,
    device: torch.device,
) -> dict[str, Any]:
    """Collect one trajectory using the shared online attack implementation."""
    env = WindNavigationBatch(simulation_config, num_envs=1, seed=int(seed))
    env.reset_all()
    initial_hidden_state = env.hidden_state[0].copy()
    goal_xy = env.goal_xy[0].copy()

    obs_rng = np.random.default_rng(int(seed) + 17_003)
    attack_rng = np.random.default_rng(int(seed) + 100_003)
    random_rng = np.random.default_rng(int(seed) + 201_337)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    transitions: list[np.ndarray] = []
    current_noisy_observation = noisy_state_observation(
        state=env.hidden_state[0],
        noise_std=float(observation_noise_std),
        rng=obs_rng,
    )
    mahalanobis_epsilon = coverage_to_mahalanobis_epsilon(float(coverage), obs_dim=4)

    positions = [env.hidden_state[0, :2].copy()]
    winds = [env.hidden_state[0, 2:].copy()]
    hidden_states = [env.hidden_state[0].copy()]
    policy_states = []
    rewards = []
    attack_flags = []
    attack_records: list[dict[str, float | int | str | bool]] = []

    done = False
    reached_goal = False
    timed_out = False
    while not done:
        step_index = int(env.step_index[0])
        policy_state, attacked, attack_diagnostics = current_policy_state(
            setting=attack_setting,
            policy=policy,
            simulation_config=simulation_config,
            filter_config=filter_config,
            initial_hidden_state=initial_hidden_state,
            actual_hidden_state=env.hidden_state[0].copy(),
            goal_xy=goal_xy,
            observations=observations,
            actions=actions,
            transitions=transitions,
            current_noisy_observation=current_noisy_observation,
            current_step_index=int(step_index),
            mahalanobis_epsilon=float(mahalanobis_epsilon),
            gamma=float(gamma),
            attack_prob=float(attack_prob),
            attack_step_size=float(attack_step_size),
            attack_num_steps=int(attack_num_steps),
            attack_mc_samples=int(attack_mc_samples),
            real_attack_transition_samples=int(real_attack_transition_samples),
            heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
            uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
            attack_rng=attack_rng,
            random_rng=random_rng,
            seed=int(seed) + len(actions) + 1_000,
            device=device,
        )

        policy_states.append(np.asarray(policy_state, dtype=float).copy())
        attack_flags.append(bool(attacked))
        if attacked and attack_diagnostics is not None:
            mahalanobis_sq = float(attack_diagnostics["mahalanobis_sq"])
            perception_error = float(np.linalg.norm(policy_state[:2] - env.hidden_state[0, :2]))
            attack_records.append(
                {
                    "step": int(step_index),
                    "mahalanobis_sq": mahalanobis_sq,
                    "mahalanobis_epsilon": float(mahalanobis_epsilon),
                    "inside_ellipsoid": is_inside_ellipsoid(
                        mahalanobis_sq=mahalanobis_sq,
                        mahalanobis_epsilon=float(mahalanobis_epsilon),
                        relative_tolerance=float(contour_relative_tolerance),
                    ),
                    "on_contour": is_on_ellipsoid_contour(
                        mahalanobis_sq=mahalanobis_sq,
                        mahalanobis_epsilon=float(mahalanobis_epsilon),
                        relative_tolerance=float(contour_relative_tolerance),
                    ),
                    "perception_error": perception_error,
                    "state_position_perturbation_norm": float(
                        attack_diagnostics["state_position_perturbation_norm"]
                    ),
                    "state_wind_perturbation_norm": float(
                        attack_diagnostics["state_wind_perturbation_norm"]
                    ),
                    "observation_position_perturbation_norm": float(
                        attack_diagnostics["observation_position_perturbation_norm"]
                    ),
                    "observation_wind_perturbation_norm": float(
                        attack_diagnostics["observation_wind_perturbation_norm"]
                    ),
                    "value_noisy_kf": float(attack_diagnostics["value_noisy_kf"]),
                    "value_attacked": float(attack_diagnostics["value_attacked"]),
                    "delta_value": float(attack_diagnostics["delta_value"]),
                }
            )

        policy_observation = np.concatenate([policy_state, goal_xy]).astype(np.float32)
        obs_tensor = torch.as_tensor(policy_observation[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            action = policy.deterministic_action(obs_tensor).cpu().numpy()[0]

        _next_obs, reward, done_mask, info = env.step(action[None, :])
        current_noisy_observation = noisy_state_observation(
            state=env.hidden_state[0],
            noise_std=float(observation_noise_std),
            rng=obs_rng,
        )
        observations.append(current_noisy_observation.copy())
        actions.append(action.copy())
        transitions.append(
            build_transition_matrix(
                rho_w=float(simulation_config.rho_w),
                delta_psi=float(info["delta_psi"][0]),
            )
        )

        positions.append(env.hidden_state[0, :2].copy())
        winds.append(env.hidden_state[0, 2:].copy())
        hidden_states.append(env.hidden_state[0].copy())
        rewards.append(float(reward[0]))
        done = bool(done_mask[0])
        reached_goal = bool(info["reached_goal"][0])
        timed_out = bool(info["timed_out"][0])

    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "winds": np.asarray(winds, dtype=np.float32),
        "hidden_states": np.asarray(hidden_states, dtype=np.float32),
        "policy_states": np.asarray(policy_states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "attack_flags": np.asarray(attack_flags, dtype=bool),
        "attack_records": attack_records,
        "goal_xy": goal_xy.astype(np.float32),
        "return": float(np.sum(rewards)),
        "length": int(len(actions)),
        "reached_goal": bool(reached_goal),
        "timed_out": bool(timed_out),
        "seed": int(seed),
    }


def collect_trajectories_with_attacks(
    *,
    policy: torch.nn.Module,
    simulation_config: Any,
    filter_config: Any,
    attack_setting: str,
    coverage: float,
    base_seed: int,
    num_trajectories: int,
    min_attacks_per_trajectory: int,
    max_seed_attempts: int,
    observation_noise_std: float,
    gamma: float,
    attack_prob: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    real_attack_transition_samples: int,
    heavy_tail_degrees_of_freedom: float,
    uniform_annulus_min_coverage: float,
    contour_relative_tolerance: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Search seeds until enough trajectories contain visible attacks."""
    trajectories: list[dict[str, Any]] = []
    for attempt_idx in range(int(max_seed_attempts)):
        seed = int(base_seed) + int(attempt_idx)
        trajectory = collect_online_attack_trajectory(
            policy=policy,
            simulation_config=simulation_config,
            filter_config=filter_config,
            attack_setting=attack_setting,
            coverage=float(coverage),
            seed=seed,
            observation_noise_std=float(observation_noise_std),
            gamma=float(gamma),
            attack_prob=float(attack_prob),
            attack_step_size=float(attack_step_size),
            attack_num_steps=int(attack_num_steps),
            attack_mc_samples=int(attack_mc_samples),
            real_attack_transition_samples=int(real_attack_transition_samples),
            heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
            uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
            contour_relative_tolerance=float(contour_relative_tolerance),
            device=device,
        )
        if int(trajectory["attack_flags"].sum()) >= int(min_attacks_per_trajectory):
            trajectories.append(trajectory)
            attack_records = trajectory["attack_records"]
            mean_position_perturbation = 0.0
            mean_wind_perturbation = 0.0
            if attack_records:
                mean_position_perturbation = float(
                    np.mean(
                        [
                            record.get("state_position_perturbation_norm", 0.0)
                            for record in attack_records
                        ]
                    )
                )
                mean_wind_perturbation = float(
                    np.mean(
                        [
                            record.get("state_wind_perturbation_norm", 0.0)
                            for record in attack_records
                        ]
                    )
                )
            print(
                f"selected seed={seed} "
                f"length={trajectory['length']} "
                f"return={trajectory['return']:.2f} "
                f"attacks={int(trajectory['attack_flags'].sum())} "
                f"mean|dp|={mean_position_perturbation:.3f} "
                f"mean|dw|={mean_wind_perturbation:.3f}"
            )
        if len(trajectories) >= int(num_trajectories):
            break

    if len(trajectories) < int(num_trajectories):
        raise RuntimeError(
            "Could not collect enough attacked trajectories. "
            "Increase max_seed_attempts or attack_prob."
        )
    return trajectories


def plot_attacked_trajectories(
    *,
    trajectories: list[dict[str, Any]],
    radius_max: float,
    goal_radius: float,
    outpath: str,
) -> None:
    """Render a `2 x 3` grid of attacked trajectories."""
    set_plot_theme()
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.5), sharex=True, sharey=True)
    axes_flat = axes.reshape(-1)

    path_color = "#7BC8A4"
    wind_color = "#6C91BF"
    action_color = "#D66A4E"
    start_color = "#F4A261"
    goal_color = "#E76F51"
    attack_color = "#C77DA4"
    perceived_color = "#8A6FB3"
    position_perturbation_color = "#F0C987"
    wind_perturbation_color = "#9DB7D5"

    for panel_idx, (ax, trajectory) in enumerate(zip(axes_flat, trajectories), start=1):
        style_axis(ax)
        positions = np.asarray(trajectory["positions"], dtype=float)
        winds = np.asarray(trajectory["winds"], dtype=float)
        actions = np.asarray(trajectory["actions"], dtype=float)
        policy_states = np.asarray(trajectory["policy_states"], dtype=float)
        attack_flags = np.asarray(trajectory["attack_flags"], dtype=bool)
        goal_xy = np.asarray(trajectory["goal_xy"], dtype=float)

        wind_skip = max(1, len(winds) // 8)
        action_skip = max(1, len(actions) // 8)

        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                float(radius_max),
                facecolor="none",
                edgecolor="#D9E3EB",
                linewidth=1.0,
            )
        )
        ax.add_patch(
            plt.Circle(
                goal_xy,
                float(goal_radius),
                facecolor="none",
                edgecolor=goal_color,
                linewidth=1.2,
                linestyle="--",
            )
        )

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color=path_color,
            linewidth=2.1,
            alpha=0.96,
            label="real trajectory",
        )
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            s=42,
            color=start_color,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
            label="start",
        )
        ax.scatter(
            goal_xy[0],
            goal_xy[1],
            s=46,
            color=goal_color,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
            label="goal",
        )

        ax.quiver(
            positions[:-1:wind_skip, 0],
            positions[:-1:wind_skip, 1],
            winds[:-1:wind_skip, 0],
            winds[:-1:wind_skip, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            color=wind_color,
            alpha=0.68,
            label="wind",
        )
        ax.quiver(
            positions[:-1:action_skip, 0],
            positions[:-1:action_skip, 1],
            actions[::action_skip, 0],
            actions[::action_skip, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            color=action_color,
            alpha=0.76,
            label="action",
        )

        real_attack_positions = positions[:-1][attack_flags]
        perceived_attack_positions = policy_states[attack_flags, :2]
        if real_attack_positions.size > 0:
            for real_xy, perceived_xy in zip(real_attack_positions, perceived_attack_positions):
                ax.plot(
                    [real_xy[0], perceived_xy[0]],
                    [real_xy[1], perceived_xy[1]],
                    color=attack_color,
                    linewidth=0.9,
                    linestyle=":",
                    alpha=0.55,
                    zorder=2,
                )
            ax.scatter(
                real_attack_positions[:, 0],
                real_attack_positions[:, 1],
                s=34,
                marker="x",
                color=attack_color,
                linewidths=1.4,
                zorder=5,
                label="attacked real $s_t$",
            )
            ax.scatter(
                perceived_attack_positions[:, 0],
                perceived_attack_positions[:, 1],
                s=26,
                marker="D",
                facecolors="none",
                edgecolors=perceived_color,
                linewidths=1.1,
                zorder=5,
                label="perceived $s_t$",
            )

        outcome_text = "goal" if trajectory["reached_goal"] else "timeout"
        mean_error = 0.0
        mean_position_perturbation = 0.0
        mean_wind_perturbation = 0.0
        if trajectory["attack_records"]:
            mean_error = float(np.mean([record["perception_error"] for record in trajectory["attack_records"]]))
            mean_position_perturbation = float(
                np.mean(
                    [
                        record.get("state_position_perturbation_norm", 0.0)
                        for record in trajectory["attack_records"]
                    ]
                )
            )
            mean_wind_perturbation = float(
                np.mean(
                    [
                        record.get("state_wind_perturbation_norm", 0.0)
                        for record in trajectory["attack_records"]
                    ]
                )
            )
            inset_ax = ax.inset_axes([0.055, 0.055, 0.31, 0.18])
            inset_ax.bar(
                [0, 1],
                [mean_position_perturbation, mean_wind_perturbation],
                color=[position_perturbation_color, wind_perturbation_color],
                edgecolor="white",
                linewidth=0.6,
            )
            inset_ax.set_xticks([0, 1])
            inset_ax.set_xticklabels([r"$|\Delta \hat p_t|$", r"$|\Delta \hat w_t|$"], fontsize=6.2)
            inset_ax.tick_params(axis="y", labelsize=6.2, length=2)
            inset_ax.tick_params(axis="x", length=0)
            inset_ax.grid(True, axis="y", alpha=0.18, linewidth=0.45)
            inset_ax.grid(False, axis="x")
            inset_ax.set_facecolor((1.0, 1.0, 1.0, 0.82))
            inset_ax.spines["top"].set_visible(False)
            inset_ax.spines["right"].set_visible(False)
            inset_ax.spines["left"].set_alpha(0.35)
            inset_ax.spines["bottom"].set_alpha(0.35)
        ax.text(
            0.03,
            0.97,
            f"run {panel_idx} | {outcome_text}\n"
            f"T={trajectory['length']} | G={trajectory['return']:.1f} | "
            f"A={int(attack_flags.sum())}\n"
            f"mean real shift={mean_error:.2f}\n"
            f"mean |dp|={mean_position_perturbation:.2f} | "
            f"|dw|={mean_wind_perturbation:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            color="#31424F",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "alpha": 0.88,
                "edgecolor": "#D8E0E6",
            },
        )

        ax.set_xlim(-float(radius_max), float(radius_max))
        ax.set_ylim(-float(radius_max), float(radius_max))
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", frameon=True, framealpha=0.90)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$p_{x,t}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p_{y,t}$")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_data(
    *,
    trajectories: list[dict[str, Any]],
    outpath: str,
    metadata: dict[str, float | int | str],
) -> None:
    """Save variable-length trajectories and attack metadata in `RL/data/`."""
    np.savez_compressed(
        outpath,
        metadata=np.asarray([metadata], dtype=object),
        seeds=np.asarray([trajectory["seed"] for trajectory in trajectories], dtype=int),
        returns=np.asarray([trajectory["return"] for trajectory in trajectories], dtype=float),
        lengths=np.asarray([trajectory["length"] for trajectory in trajectories], dtype=int),
        reached_goal=np.asarray([trajectory["reached_goal"] for trajectory in trajectories], dtype=bool),
        positions=np.asarray([trajectory["positions"] for trajectory in trajectories], dtype=object),
        winds=np.asarray([trajectory["winds"] for trajectory in trajectories], dtype=object),
        hidden_states=np.asarray([trajectory["hidden_states"] for trajectory in trajectories], dtype=object),
        policy_states=np.asarray([trajectory["policy_states"] for trajectory in trajectories], dtype=object),
        actions=np.asarray([trajectory["actions"] for trajectory in trajectories], dtype=object),
        rewards=np.asarray([trajectory["rewards"] for trajectory in trajectories], dtype=object),
        attack_flags=np.asarray([trajectory["attack_flags"] for trajectory in trajectories], dtype=object),
        attack_records=np.asarray([trajectory["attack_records"] for trajectory in trajectories], dtype=object),
        goal_xy=np.asarray([trajectory["goal_xy"] for trajectory in trajectories], dtype=float),
    )


def run_attack_trajectory_plot(
    *,
    num_trajectories: int,
    min_attacks_per_trajectory: int,
    max_seed_attempts: int,
    attack_setting: str,
    coverage: float,
    attack_prob: float,
    observation_noise_std: float,
    process_position_std: float,
    process_wind_std: float,
    contour_relative_tolerance: float,
    attack_step_size: float,
    attack_num_steps: int,
    attack_mc_samples: int,
    real_attack_transition_samples: int,
    heavy_tail_degrees_of_freedom: float,
    uniform_annulus_min_coverage: float,
    base_seed: int,
    output_stem: str,
    max_steps_override: int | None = None,
) -> dict[str, str]:
    """Collect attacked trajectories, save their plot, and save the data."""
    device = default_device()
    model_path = os.path.join(rl_model_dir(), "wind_navigation_ppo.pt")
    policy, env_config, _net_config, train_config, _history = load_agent_checkpoint(model_path, device)
    policy.eval()

    simulation_config = replace(
        env_config,
        max_steps=int(max_steps_override) if max_steps_override is not None else int(env_config.max_steps),
        observation_noise_std=0.0,
        process_position_std=float(process_position_std),
        process_wind_std=float(process_wind_std),
    )
    filter_config = replace(simulation_config, observation_noise_std=float(observation_noise_std))

    print(
        "Trajectory attack plot: "
        f"setting={attack_setting}, coverage={coverage:.2f}, "
        f"attack_prob={attack_prob:.3f}, pgd_steps={attack_num_steps}"
    )
    trajectories = collect_trajectories_with_attacks(
        policy=policy,
        simulation_config=simulation_config,
        filter_config=filter_config,
        attack_setting=attack_setting,
        coverage=float(coverage),
        base_seed=int(base_seed),
        num_trajectories=int(num_trajectories),
        min_attacks_per_trajectory=int(min_attacks_per_trajectory),
        max_seed_attempts=int(max_seed_attempts),
        observation_noise_std=float(observation_noise_std),
        gamma=float(train_config.gamma),
        attack_prob=float(attack_prob),
        attack_step_size=float(attack_step_size),
        attack_num_steps=int(attack_num_steps),
        attack_mc_samples=int(attack_mc_samples),
        real_attack_transition_samples=int(real_attack_transition_samples),
        heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
        uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
        contour_relative_tolerance=float(contour_relative_tolerance),
        device=device,
    )

    figure_path = os.path.join(rl_figures_dir(), f"{output_stem}.png")
    plot_attacked_trajectories(
        trajectories=trajectories,
        radius_max=float(simulation_config.radius_max),
        goal_radius=float(simulation_config.goal_radius),
        outpath=figure_path,
    )

    data_path = os.path.join(rl_data_dir(), f"{output_stem}.npz")
    save_trajectory_data(
        trajectories=trajectories,
        outpath=data_path,
        metadata={
            "attack_setting": attack_setting,
            "coverage": float(coverage),
            "attack_prob": float(attack_prob),
            "observation_noise_std": float(observation_noise_std),
            "process_position_std": float(process_position_std),
            "process_wind_std": float(process_wind_std),
            "attack_step_size": float(attack_step_size),
            "attack_num_steps": int(attack_num_steps),
            "attack_mc_samples": int(attack_mc_samples),
            "real_attack_transition_samples": int(real_attack_transition_samples),
            "base_seed": int(base_seed),
        },
    )

    print(f"Saved attacked trajectory figure to: {figure_path}")
    print(f"Saved attacked trajectory data to:   {data_path}")
    return {"figure_path": figure_path, "data_path": data_path}


def main() -> None:
    """Configure and plot six estimated-return PGD attacked trajectories."""
    num_trajectories = 6
    min_attacks_per_trajectory = 1
    max_seed_attempts = 250
    attack_setting = "estimated-return PGD"
    coverage = 0.75
    attack_prob = 0.10
    observation_noise_std = 0.75
    process_position_std = 0.14
    process_wind_std = 0.10
    contour_relative_tolerance = 0.01
    attack_step_size = 0.05
    attack_num_steps = 120
    attack_mc_samples = 16
    real_attack_transition_samples = 8
    heavy_tail_degrees_of_freedom = 3.0
    uniform_annulus_min_coverage = 0.50
    base_seed = 20260727
    output_stem = "wind_trajectories_estimated_return_pgd_attack"

    run_attack_trajectory_plot(
        num_trajectories=int(num_trajectories),
        min_attacks_per_trajectory=int(min_attacks_per_trajectory),
        max_seed_attempts=int(max_seed_attempts),
        attack_setting=attack_setting,
        coverage=float(coverage),
        attack_prob=float(attack_prob),
        observation_noise_std=float(observation_noise_std),
        process_position_std=float(process_position_std),
        process_wind_std=float(process_wind_std),
        contour_relative_tolerance=float(contour_relative_tolerance),
        attack_step_size=float(attack_step_size),
        attack_num_steps=int(attack_num_steps),
        attack_mc_samples=int(attack_mc_samples),
        real_attack_transition_samples=int(real_attack_transition_samples),
        heavy_tail_degrees_of_freedom=float(heavy_tail_degrees_of_freedom),
        uniform_annulus_min_coverage=float(uniform_annulus_min_coverage),
        base_seed=int(base_seed),
        output_stem=output_stem,
    )


if __name__ == "__main__":
    main()
