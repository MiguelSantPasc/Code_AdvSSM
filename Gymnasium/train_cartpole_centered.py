#!/usr/bin/env python3
"""Adapt the existing discrete DQN to a rail with physical stops and centering.

The original downloaded checkpoint is kept intact. A fresh DQN optimizer and
replay buffer are initialized from its policy weights, then trained with:
* the same two actions, forces -10 N / +10 N;
* the fine-step plant and inelastic cart/end-stop collisions from the shared
  CartPole model, including the collision impulse transferred to the pole;
* reward 1 - center_reward_weight * (x / wall_position)**2;
* ordinary pole-angle termination at 12 degrees and a 500-step time limit.

Training uses true-state feedback as did the original policy; the videos add
the existing observation noise, KF and observation attack during evaluation.
The actor is still argmax Q. This is not a continuous-action agent.

Two fixed validation seeds, separate from video seed 100, select the checkpoint
with the highest mean centered return among trained checkpoints. The original
weights are evaluated only as a baseline, since their critic has not learned
the new reward. This is model selection during training,
not a search for a seed on which an attack succeeds. Training settings are
plain variables in main(). A JSON companion records the physical/reward model
and the selected checkpoint's validation performance.

Run: .\\.venv\\Scripts\\python.exe Gymnasium/train_cartpole_centered.py
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

import cartpole_covadapt_compare_epsilons_wolf as cartpole


class CenteredCartPoleEnv(gym.Env):
    """Discrete-action training environment sharing the exact rollout plant."""

    def __init__(self, *, ssm: cartpole.CartPoleLinearSSM, max_steps: int):
        super().__init__()
        self.base = gym.make("CartPole-v1", max_episode_steps=max_steps)
        self.ssm = ssm
        self.action_space = self.base.action_space
        self.observation_space = self.base.observation_space

    def reset(self, *, seed=None, options=None):
        """Use Gymnasium's seeded small initial perturbation near equilibrium."""
        super().reset(seed=seed)
        return cartpole.reset_cartpole_rollout(self.base, seed=seed) if seed is not None else self._reset_unseeded()

    def _reset_unseeded(self):
        """Continue the environment RNG stream across training episodes."""
        observation, info = self.base.reset()
        setattr(self.base, "_advssm_coarse_steps", 0)
        return np.asarray(observation, dtype=np.float32), info

    def step(self, action):
        """Apply the shared bounded plant and the shared centered reward."""
        return cartpole.step_cartpole_rollout(self.base, int(action), ssm=self.ssm)

    def close(self):
        self.base.close()


def evaluate_centered_policy(model, *, ssm, seeds: tuple[int, ...], max_steps: int) -> dict:
    """Measure survival, centering and reward on fixed training-selection seeds."""
    env = CenteredCartPoleEnv(ssm=ssm, max_steps=max_steps)
    returns, lengths, mean_abs_x = [], [], []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total, positions = 0.0, []
            for step in range(max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += reward
                positions.append(abs(float(observation[0])))
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(step + 1)
            mean_abs_x.append(float(np.mean(positions)))
    finally:
        env.close()
    return dict(mean_return=float(np.mean(returns)), mean_steps=float(np.mean(lengths)),
                mean_abs_x=float(np.mean(mean_abs_x)), returns=returns, lengths=lengths)


class CenteringCheckpoint(BaseCallback):
    """Persist the best centered-return policy and concise reproducibility data."""

    def __init__(self, *, ssm, seeds, max_steps, interval, output, metadata):
        super().__init__()
        self.ssm, self.seeds, self.max_steps = ssm, seeds, max_steps
        self.interval, self.output, self.metadata = interval, output, metadata
        self.best_score = -float("inf")

    def record(self):
        metrics = evaluate_centered_policy(self.model, ssm=self.ssm, seeds=self.seeds, max_steps=self.max_steps)
        print(f"[training {self.num_timesteps}] return={metrics['mean_return']:.2f}; "
              f"steps={metrics['mean_steps']:.0f}; mean |x|={metrics['mean_abs_x']:.3f} m", flush=True)
        # Keep the source-policy measurement, but select a critic that has
        # actually received updates using the centered reward and bounded plant.
        if self.num_timesteps == 0:
            self.metadata["source_policy_validation"] = metrics
            return
        if metrics["mean_return"] > self.best_score:
            self.best_score = metrics["mean_return"]
            self.model.save(str(self.output))
            payload = {**self.metadata, "selected_training_steps": self.num_timesteps,
                       "validation": metrics}
            self.output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _on_training_start(self):
        self.record()

    def _on_step(self):
        if self.num_timesteps % self.interval == 0:
            self.record()
        return True

    def _on_training_end(self):
        if self.num_timesteps % self.interval:
            self.record()


def verify_centered_checkpoint(model_path: Path, *, ssm: cartpole.CartPoleLinearSSM) -> None:
    """Reject a checkpoint trained for different wall/reward/time-step settings."""
    if not model_path.is_file() or not model_path.with_suffix(".json").is_file():
        raise FileNotFoundError("Train the centered DQN first: python Gymnasium/train_cartpole_centered.py")
    payload = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    if payload.get("selected_training_steps", 0) <= 0:
        raise ValueError("The centered checkpoint must have been trained with the new reward.")
    for name in ("wall_position", "wall_restitution", "center_reward_weight", "tau", "real_tau"):
        if not np.isclose(payload["environment"][name], getattr(ssm, name)):
            raise ValueError(f"Checkpoint mismatch for {name}; retrain for the requested environment.")


def main():
    """Train a centered discrete DQN while preserving the original checkpoint."""
    seed = 20260928
    total_timesteps = 120_000
    max_steps = 500
    filter_tau = 0.02
    real_tau = 0.01
    wall_position = 2.4
    wall_restitution = 0.0
    center_reward_weight = 0.8
    learning_rate = 1e-4
    buffer_size = 50_000
    learning_starts = 2_000
    batch_size = 64
    train_freq = 4
    gradient_steps = 1
    target_update_interval = 1_000
    exploration_fraction = 0.3
    exploration_initial_eps = 0.15
    exploration_final_eps = 0.02
    validation_interval = 10_000
    validation_seeds = (901, 902)
    device = "cpu"
    directory = Path(__file__).resolve().parent
    source = directory / "outputs/saved_models/sb3_dqn_cartpole_v1/dqn-CartPole-v1.zip"
    output = directory / "outputs/saved_models/sb3_dqn_cartpole_centered_v1/dqn-CartPole-centered.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ssm = cartpole.build_cartpole_linear_ssm(
        filter_tau=filter_tau, real_tau=real_tau, wall_position=wall_position,
        wall_restitution=wall_restitution, center_reward_weight=center_reward_weight,
    )
    pretrained = cartpole.load_cartpole_policy(str(source), torch.device(device))
    env = CenteredCartPoleEnv(ssm=ssm, max_steps=max_steps)
    try:
        model = DQN(
            "MlpPolicy", env, policy_kwargs=pretrained.policy_kwargs,
            learning_rate=learning_rate, buffer_size=buffer_size,
            learning_starts=learning_starts, batch_size=batch_size,
            train_freq=train_freq, gradient_steps=gradient_steps,
            target_update_interval=target_update_interval, gamma=pretrained.gamma,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps, seed=seed, device=device,
        )
        model.policy.load_state_dict(pretrained.policy.state_dict())
        physics = asdict(ssm)
        physics = {name: value for name, value in physics.items() if not isinstance(value, np.ndarray)}
        callback = CenteringCheckpoint(
            ssm=ssm, seeds=validation_seeds, max_steps=max_steps,
            interval=validation_interval, output=output,
            metadata=dict(environment=physics, seed=seed, total_training_steps=total_timesteps,
                          validation_seeds=validation_seeds, gamma=float(model.gamma),
                          reward=f"1 - {center_reward_weight} * (x / {wall_position})^2",
                          source_checkpoint=str(source)),
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        print(f"Saved best centered DQN: {output}", flush=True)
    finally:
        env.close()
        if pretrained.get_env() is not None:
            pretrained.get_env().close()


if __name__ == "__main__":
    main()
