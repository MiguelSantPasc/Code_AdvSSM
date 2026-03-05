import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


# -------------------------
# Wrapper: observación con ruido i.i.d.
# sigma puede ser escalar o vector de 4 dims
# -------------------------
class NoisyObs(gym.ObservationWrapper):
    def __init__(self, env, sigma=0.0, seed=0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        if np.isscalar(sigma):
            self.sigma = np.array([sigma] * int(np.prod(self.observation_space.shape)), dtype=np.float32)
        else:
            self.sigma = np.array(sigma, dtype=np.float32)

    def observation(self, obs):
        noise = self.rng.normal(0.0, self.sigma, size=obs.shape).astype(np.float32)
        return (obs + noise).astype(np.float32)


def evaluate_returns(model, env, n_episodes=200, seed=123):
    """Return (reward acumulado por episodio) en n_episodes."""
    returns = np.zeros(n_episodes, dtype=np.float32)
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_return += float(reward)
            done = terminated or truncated
        returns[ep] = ep_return
    return returns


def main():
    # Project structure
    experiments_dir = Path(__file__).resolve().parent
    root_dir = experiments_dir.parent.parent
    saved_models_dir = root_dir / "saved_models"
    results_dir = root_dir / "results"
    results_data_dir = results_dir / "cartpole_evaluation_data"

    # Create directories if they don't exist
    saved_models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    results_data_dir.mkdir(exist_ok=True)

    # Archivos de salida
    model_path = saved_models_dir / "dqn_cartpole_clean.zip"
    fig_path = results_dir / "cartpole_returns_vs_noise.png"

    # -------------------------
    # Entrenar (solo si NO existe el modelo)
    # -------------------------
    if model_path.exists():
        # Cargar modelo existente
        model = DQN.load(str(model_path))
        print(f"[OK] Modelo cargado: {model_path.name}")
    else:
        # Entrenar desde cero SIN ruido
        train_env = gym.make("CartPole-v1")
        train_env = Monitor(train_env)

        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.2,
            exploration_final_eps=0.02,
            verbose=0,  # <- silencio
            seed=0,
        )

        total_timesteps = 200_000
        model.learn(total_timesteps=total_timesteps)

        model.save(str(model_path))
        train_env.close()
        print(f"[OK] Modelo entrenado y guardado: {model_path.name}")

    # -------------------------
    # Evaluar en 4 niveles de ruido (0 + crecientes)
    # Ajusta estos niveles a gusto
    # -------------------------
    noise_levels = [0.0, 0.005, 0.05, 0.50]  # 4 ruidos: 0 y crecientes
    n_eval_episodes = 200
    seed_eval = 10

    results = {}  # sigma -> returns
    for i, sigma in enumerate(noise_levels):
        env = gym.make("CartPole-v1")
        if sigma > 0:
            # seed distinto por nivel de ruido para que el ruido sea reproducible
            env = NoisyObs(env, sigma=sigma, seed=999 + i)

        returns = evaluate_returns(model, env, n_episodes=n_eval_episodes, seed=seed_eval)
        env.close()
        results[sigma] = returns

        # guardo arrays por si quieres inspeccionar
        np.save(results_data_dir / f"returns_sigma_{sigma:.2f}.npy", returns)

    # -------------------------
    # Plot: UNA figura con 2 gráficas (subplots)
    #   arriba: return por episodio
    #   abajo: return acumulado
    # con 4 curvas en cada una
    # -------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1, ax2 = axes

    # Return por episodio
    for sigma in noise_levels:
        returns = results[sigma]
        ax1.plot(returns, label=f"sigma={sigma:.2f}")
    ax1.set_title("CartPole: Return por episodio (policy entrenada sin ruido)")
    ax1.set_ylabel("Return episodio (suma rewards)")
    ax1.legend()

    # Return acumulado
    for sigma in noise_levels:
        returns = results[sigma]
        cum_returns = np.cumsum(returns)
        ax2.plot(cum_returns, label=f"sigma={sigma:.2f}")
    ax2.set_title("CartPole: Return acumulado (cumsum) en evaluación")
    ax2.set_xlabel("Episodio (evaluación)")
    ax2.set_ylabel("Return acumulado")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Resumen mínimo (una sola línea)
    means = " | ".join([f"{s:.2f}:{results[s].mean():.1f}" for s in noise_levels])
    print(f"[OK] Figura guardada: {fig_path.name}")
    print(f"[INFO] Media return por sigma -> {means}")


if __name__ == "__main__":
    main()
