#!/usr/bin/env python3
"""
train_wind_ppo.py

Train the custom wind-navigation actor-critic with PPO and save the model.

Why this script exists:
1. The user wants a trained PPO policy for the 4D wind-navigation SSM.
2. The policy must receive the hidden state and perform the goal-relative
   transformation internally before the neural network layers.
3. The resulting actor and critic should be saved in `RL/model/` so the value
   and trajectory figure scripts can reuse the exact same checkpoint.
"""

from __future__ import annotations

import os

from wind_rl_setup import PolicyNetworkConfig
from wind_rl_setup import PPOTrainingConfig
from wind_rl_setup import WindNavigationConfig
from wind_rl_setup import default_device
from wind_rl_setup import load_training_checkpoint
from wind_rl_setup import rl_model_dir
from wind_rl_setup import save_agent_checkpoint
from wind_rl_setup import train_ppo_agent


def main() -> None:
    """Train the PPO agent and save it under `RL/model/`."""
    resume_training = True
    additional_updates = 1000

    env_config = WindNavigationConfig(
        start_xy=(0.0, 0.0),
        radius_max=12.0,
        start_radius=0.0,
        goal_distance_min=3.0,
        goal_distance_max=10.0,
        goal_radius=0.65,
        max_steps=60,
        rho_w=1.0,
        wind_turn_std=0.10,
        process_position_std=0.030,
        process_wind_std=0.020,
        initial_wind_magnitude=0.85,
        action_limit=1.0,
        step_cost=-3.5,
        progress_reward_weight=0.0,
        near_goal_progress_bonus_weight=0.0,
        near_goal_progress_power=1.0,
        distance_reward_weight=1.0,
        goal_reward=50.0,
        timeout_penalty=-6.0,
        observation_noise_std=0.0,
    )

    net_config = PolicyNetworkConfig(
        hidden_size=96,
        action_std=0.65,
    )

    train_config = PPOTrainingConfig(
        seed=2026,
        num_envs=128,
        rollout_length=64,
        total_updates=500,
        ppo_epochs=4,
        minibatch_size=512,
        gamma=0.995,
        gae_lambda=0.97,
        clip_ratio=0.20,
        value_loss_weight=0.50,
        entropy_weight=0.01,
        learning_rate=3.0e-4,
        max_grad_norm=0.80,
        log_every=10,
    )

    device = default_device()
    print(f"Training on device: {device}")
    model_dir = rl_model_dir()
    model_path = os.path.join(model_dir, "wind_navigation_ppo.pt")

    resume_payload = None
    if resume_training and os.path.exists(model_path):
        resume_payload = load_training_checkpoint(model_path, device)
        previous_net_config = resume_payload["net_config"]
        net_config = PolicyNetworkConfig(
            hidden_size=previous_net_config.hidden_size,
            action_std=net_config.action_std,
        )
        previous_train_config = resume_payload["train_config"]
        train_config = PPOTrainingConfig(
            seed=previous_train_config.seed,
            num_envs=previous_train_config.num_envs,
            rollout_length=previous_train_config.rollout_length,
            total_updates=additional_updates,
            ppo_epochs=previous_train_config.ppo_epochs,
            minibatch_size=previous_train_config.minibatch_size,
            gamma=previous_train_config.gamma,
            gae_lambda=previous_train_config.gae_lambda,
            clip_ratio=previous_train_config.clip_ratio,
            value_loss_weight=previous_train_config.value_loss_weight,
            entropy_weight=train_config.entropy_weight,
            learning_rate=previous_train_config.learning_rate,
            max_grad_norm=previous_train_config.max_grad_norm,
            log_every=previous_train_config.log_every,
        )
        resume_payload["policy"].action_std = float(net_config.action_std)
        print(
            "Resuming existing checkpoint with "
            f"{resume_payload['completed_updates']} completed PPO updates. "
            f"Running {additional_updates} extra updates now "
            "using the current environment configuration and exploration settings from this script."
        )
    else:
        print(
            "Starting a fresh PPO run with "
            f"{train_config.total_updates} updates."
        )

    policy, history, optimizer_state_dict, completed_updates = train_ppo_agent(
        env_config=env_config,
        net_config=net_config,
        train_config=train_config,
        device=device,
        initial_policy=None if resume_payload is None else resume_payload["policy"],
        initial_optimizer_state_dict=None if resume_payload is None else resume_payload["optimizer_state_dict"],
        initial_history=None if resume_payload is None else resume_payload["history"],
        completed_updates=0 if resume_payload is None else resume_payload["completed_updates"],
    )

    save_agent_checkpoint(
        path=model_path,
        policy=policy,
        env_config=env_config,
        net_config=net_config,
        train_config=train_config,
        history=history,
        optimizer_state_dict=optimizer_state_dict,
        completed_updates=completed_updates,
    )

    print(f"Saved trained PPO agent to: {model_path}")
    print(f"Accumulated PPO updates stored in checkpoint: {completed_updates}")
    if history["success_rate"]:
        print(f"Final logged success rate: {history['success_rate'][-1]:.3f}")
        print(f"Final logged mean return: {history['mean_return'][-1]:.3f}")


if __name__ == "__main__":
    main()
