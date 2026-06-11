"""
Gaussian sign POMDP with Bayesian belief updates and attack plots.

The hidden bandit mean mu is fixed during one episode. At each step:

    y_k | mu, sigma^2 ~ N(mu, sigma^2)

and the agent chooses among negative, abstain/continue, and positive actions.
The belief uses the normal-inverse-gamma conjugate prior:

    sigma^2 ~ InvGamma(alpha, beta)
    mu | sigma^2 ~ N(mu_0, sigma^2 / kappa).

The attack wrapper perturbs observations before the Bayesian update, allowing
the plots to compare clean versus attacked posterior means, credible intervals,
and cumulative rewards.

TOP subplot (ONE episode, fixed mu_b = 0.075):
  - plots BOTH clean and attacked observation streams on the same axes:
      * true mu line
      * observations y_k (clean + attacked) with small x-jitter so points don't overlap
      * posterior mean E[mu|data] (clean + attacked)
      * ~95% credible interval bands (clean + attacked)

BOTTOM subplot (MANY episodes):
  - cumulative reward across episodes for:
      * clean simulation
      * attacked simulation (p_attack=0.75, additive Gaussian attack noise)

Install:
  pip install gymnasium numpy matplotlib
Optional (exact Student-t CDF/intervals):
  pip install scipy

IMPORTANT: don't name this file gym.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

# Optional SciPy for exact Student-t calculations
USE_SCIPY = False
try:
    from scipy.stats import t as student_t  # type: ignore
    USE_SCIPY = True
except Exception:
    USE_SCIPY = False


# ============================================================
# 1) Environment
# ============================================================
class GaussianSignPOMDP(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        rho: float = 0.01,
        max_steps: int = 1000,
        reward_correct: float = 1.0,
        reward_wrong: float = 1.0,
        mu_range: float = 1.0,
        sigma_log_range: tuple[float, float] = (-2.0, 0.0),
        seed: int | None = None,
    ):
        super().__init__()
        self.rho = float(rho)
        self.max_steps = int(max_steps)
        self.reward_correct = float(reward_correct)
        self.reward_wrong = float(reward_wrong)

        self.mu_range = float(mu_range)
        self.sigma_log_range = (float(sigma_log_range[0]), float(sigma_log_range[1]))
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.mu_b: float | None = None
        self.sigma_b: float | None = None
        self.t = 0
        self.last_y = 0.0
        self.n_samples = 0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        fixed_mu = None
        fixed_sigma = None
        if options is not None:
            fixed_mu = options.get("fixed_mu", None)
            fixed_sigma = options.get("fixed_sigma", None)

        if fixed_mu is None:
            self.mu_b = float(self.rng.uniform(-self.mu_range, self.mu_range))
        else:
            self.mu_b = float(fixed_mu)

        if fixed_sigma is None:
            log_sigma = float(self.rng.uniform(self.sigma_log_range[0], self.sigma_log_range[1]))
            self.sigma_b = float(np.exp(log_sigma))
        else:
            self.sigma_b = float(fixed_sigma)

        self.t = 0
        self.last_y = 0.0
        self.n_samples = 0

        obs = np.array([self.last_y], dtype=np.float32)
        info = {"mu_b": self.mu_b, "sigma_b": self.sigma_b}
        return obs, info

    def step(self, action: int):
        assert self.mu_b is not None and self.sigma_b is not None, "Call reset() first."

        self.t += 1
        terminated = False
        truncated = False
        reward = 0.0

        if action == 0:
            y = float(self.rng.normal(loc=self.mu_b, scale=self.sigma_b))
            self.last_y = y
            self.n_samples += 1
            reward = -self.rho

        elif action == 1:
            terminated = True
            correct = self.mu_b > 0.0
            reward = self.reward_correct if correct else -self.reward_wrong

        elif action == 2:
            terminated = True
            correct = self.mu_b < 0.0
            reward = self.reward_correct if correct else -self.reward_wrong

        else:
            raise ValueError(f"Invalid action: {action}")

        if (not terminated) and self.t >= self.max_steps:
            truncated = True
            reward = -self.reward_wrong

        obs = np.array([self.last_y], dtype=np.float32)
        info = {"n_samples": self.n_samples}
        return obs, float(reward), terminated, truncated, info


# ============================================================
# 2) Observation attack wrapper
# ============================================================
class ObservationAttackWrapper(gym.Wrapper):
    """
    With probability p_attack, when action==0 (sample), add additive Gaussian noise:
      y_attacked = y + eps, eps ~ N(0, sigma_attack^2)
    """

    def __init__(self, env: gym.Env, p_attack: float = 0.75, sigma_attack: float = 1.0, seed: int = 0):
        super().__init__(env)
        self.p_attack = float(p_attack)
        self.sigma_attack = float(sigma_attack)
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if action == 0 and (self.rng.random() < self.p_attack):
            eps = float(self.rng.normal(0.0, self.sigma_attack))
            obs = np.array([float(obs[0]) + eps], dtype=np.float32)

        return obs, reward, terminated, truncated, info


# ============================================================
# 3) Belief: Normal–Inverse-Gamma
# ============================================================
@dataclass
class NIGPrior:
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 2.0
    beta0: float = 1.0


class NIGBelief:
    def __init__(self, prior: NIGPrior):
        self.prior = prior
        self.reset()

    def reset(self):
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0

    def update(self, y: float):
        y = float(y)
        self.n += 1
        self.sum_y += y
        self.sum_y2 += y * y

    def posterior_params(self) -> tuple[float, float, float, float]:
        mu0, kappa0, alpha0, beta0 = self.prior.mu0, self.prior.kappa0, self.prior.alpha0, self.prior.beta0
        n = self.n
        if n == 0:
            return float(mu0), float(kappa0), float(alpha0), float(beta0)

        ybar = self.sum_y / n
        sse = self.sum_y2 - n * (ybar**2)

        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * ybar) / kappa_n
        alpha_n = alpha0 + 0.5 * n
        beta_n = beta0 + 0.5 * sse + (kappa0 * n * (ybar - mu0) ** 2) / (2.0 * kappa_n)
        return float(mu_n), float(kappa_n), float(alpha_n), float(beta_n)


def prob_mu_positive(mu_n: float, kappa_n: float, alpha_n: float, beta_n: float) -> float:
    if USE_SCIPY:
        df = 2.0 * alpha_n
        scale = float(np.sqrt(beta_n / (alpha_n * kappa_n)))
        return float(1.0 - student_t.cdf(0.0, df=df, loc=mu_n, scale=scale))
    if alpha_n <= 1.0:
        return 0.5
    var_mu = beta_n / ((alpha_n - 1.0) * kappa_n)
    z = (0.0 - mu_n) / (np.sqrt(var_mu) + 1e-12)
    return float(0.5 * (1.0 - np.math.erf(z / np.sqrt(2.0))))


def mu_ci(mu_n: float, kappa_n: float, alpha_n: float, beta_n: float, level=0.95) -> tuple[float, float]:
    if USE_SCIPY:
        df = 2.0 * alpha_n
        scale = float(np.sqrt(beta_n / (alpha_n * kappa_n)))
        q_lo = (1.0 - level) / 2.0
        q_hi = 1.0 - q_lo
        lo = float(student_t.ppf(q_lo, df=df, loc=mu_n, scale=scale))
        hi = float(student_t.ppf(q_hi, df=df, loc=mu_n, scale=scale))
        return lo, hi
    if alpha_n <= 1.0:
        return mu_n, mu_n
    var_mu = beta_n / ((alpha_n - 1.0) * kappa_n)
    z = 1.96
    lo = float(mu_n - z * np.sqrt(var_mu))
    hi = float(mu_n + z * np.sqrt(var_mu))
    return lo, hi


# ============================================================
# 4) Policy
# ============================================================
def bayes_threshold_policy(belief: NIGBelief, delta: float = 0.05) -> int:
    mu_n, kappa_n, alpha_n, beta_n = belief.posterior_params()
    p_pos = prob_mu_positive(mu_n, kappa_n, alpha_n, beta_n)
    if p_pos >= 1.0 - delta:
        return 1
    if p_pos <= delta:
        return 2
    return 0


# ============================================================
# 5) Rollouts
# ============================================================
def run_one_episode_trace_fixed_mu(
    env: gym.Env,
    prior: NIGPrior,
    fixed_mu: float,
    fixed_sigma: float | None,
    seed: int,
    delta: float,
):
    obs, info = env.reset(seed=seed, options={"fixed_mu": fixed_mu, "fixed_sigma": fixed_sigma})
    mu_true = float(info.get("mu_b", fixed_mu))

    belief = NIGBelief(prior)

    k = 0
    ts, ys, mu_hats, ci_los, ci_his = [], [], [], [], []
    total_env_steps = 0
    ep_return = 0.0
    decided_action = None
    done = False

    while not done:
        a = bayes_threshold_policy(belief, delta=delta)
        obs, r, terminated, truncated, _ = env.step(a)
        ep_return += float(r)
        total_env_steps += 1

        if a == 0:
            y = float(obs[0])
            belief.update(y)
            k += 1

            mu_n, kappa_n, alpha_n, beta_n = belief.posterior_params()
            lo, hi = mu_ci(mu_n, kappa_n, alpha_n, beta_n, level=0.95)

            ts.append(k)
            ys.append(y)
            mu_hats.append(mu_n)
            ci_los.append(lo)
            ci_his.append(hi)
        else:
            decided_action = a

        done = terminated or truncated

    return {
        "mu_true": mu_true,
        "ts": np.array(ts, dtype=np.int32),
        "ys": np.array(ys, dtype=np.float64),
        "mu_hats": np.array(mu_hats, dtype=np.float64),
        "ci_los": np.array(ci_los, dtype=np.float64),
        "ci_his": np.array(ci_his, dtype=np.float64),
        "decided_action": decided_action,
        "total_env_steps": total_env_steps,
        "n_samples": len(ys),
        "episode_return": ep_return,
    }


def run_many_episodes_cumreward(
    env_ctor,
    prior: NIGPrior,
    n_episodes: int,
    seed: int,
    delta: float,
):
    episode_returns = np.zeros(n_episodes, dtype=np.float64)
    for ep in range(n_episodes):
        env = env_ctor(ep)
        obs, _ = env.reset(seed=seed + ep)
        belief = NIGBelief(prior)

        done = False
        ep_return = 0.0
        while not done:
            a = bayes_threshold_policy(belief, delta=delta)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_return += float(r)
            if a == 0:
                belief.update(float(obs[0]))
            done = terminated or truncated

        episode_returns[ep] = ep_return
        env.close()

    return episode_returns, np.cumsum(episode_returns)


# ============================================================
# 6) Plotting
# ============================================================
def plot_top_two_traces_and_bottom_cumrewards(
    trace_clean,
    trace_attack,
    cum_clean,
    cum_attack,
    out_path: Path,
    p_attack: float,
    sigma_attack: float,
    x_jitter: float = 0.12,  # separates points so they don't overlap
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # --- TOP: both traces in one axes ---
    ax = axes[0]
    mu_true = trace_clean["mu_true"]

    ax.axhline(mu_true, linestyle="--", linewidth=1.8, label=f"mu_b verdadero = {mu_true:.3f}")
    ax.axhline(0.0, linestyle=":", linewidth=1.5, label="_nolegend_")

    # Clean
    t_c = trace_clean["ts"].astype(np.float64)
    y_c = trace_clean["ys"]
    mh_c = trace_clean["mu_hats"]
    lo_c = trace_clean["ci_los"]
    hi_c = trace_clean["ci_his"]

    # Attack
    t_a = trace_attack["ts"].astype(np.float64)
    y_a = trace_attack["ys"]
    mh_a = trace_attack["mu_hats"]
    lo_a = trace_attack["ci_los"]
    hi_a = trace_attack["ci_his"]

    if len(t_c) > 0:
        ax.scatter(t_c - x_jitter, y_c, s=12, marker="o", label="y_k (clean)")
        ax.plot(t_c, mh_c, linewidth=2.0, label="E[mu|datos] (clean)")
        ax.fill_between(t_c, lo_c, hi_c, alpha=0.18, label="IC~95% (clean)")

    if len(t_a) > 0:
        ax.scatter(t_a + x_jitter, y_a, s=12, marker="x", label="y_k (attack)")
        ax.plot(t_a, mh_a, linewidth=2.0, label="E[mu|datos] (attack)")
        ax.fill_between(t_a, lo_a, hi_a, alpha=0.18, label="IC~95% (attack)")

    def decision_txt(tr):
        d = tr["decided_action"]
        if d is None:
            return "TIMEOUT"
        return "DECIDE: mu > 0" if d == 1 else "DECIDE: mu < 0"

    ax.set_title(
        "Un episodio (mu_b fijo=0.075): clean vs attacked\n"
        f"clean: muestras={trace_clean['n_samples']} return={trace_clean['episode_return']:.3f} | "
        f"attack: muestras={trace_attack['n_samples']} return={trace_attack['episode_return']:.3f}\n"
        f"attack params: p={p_attack:.2f}, eps~N(0,{sigma_attack**2:.2f})"
    )
    ax.set_xlabel("Tiempo (k = #muestras solicitadas)")
    ax.set_ylabel("Valor")
    ax.legend(loc="best")

    # --- BOTTOM: cumulative rewards ---
    ax2 = axes[1]
    ax2.plot(cum_clean, label="Sin ataque (limpio)")
    ax2.plot(cum_attack, label=f"Con ataque (p={p_attack:.2f}, sigma_attack={sigma_attack:.2f})")
    ax2.set_title("Reward acumulado a lo largo de episodios (cumsum de returns)")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Reward acumulado")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Figura guardada en: {out_path}")


# ============================================================
# 7) Main
# ============================================================
def main():
    # Env params
    rho = 0.01
    max_steps = 1000
    reward_correct = 1.0
    reward_wrong = 1.0
    mu_range = 1.0
    sigma_log_range = (-2.0, 0.0)

    # Attack params
    p_attack = 0.5
    sigma_attack = 0.85

    # Fixed episode mu
    fixed_mu = 0.75
    fixed_sigma = None  # or set e.g. 0.2

    # Policy/belief params
    prior = NIGPrior(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    delta = 0.05

    # ---- TOP: run one fixed-mu episode clean + attacked (same seed) ----
    env_clean_trace = GaussianSignPOMDP(
        rho=rho,
        max_steps=max_steps,
        reward_correct=reward_correct,
        reward_wrong=reward_wrong,
        mu_range=mu_range,
        sigma_log_range=sigma_log_range,
        seed=0,
    )
    trace_clean = run_one_episode_trace_fixed_mu(
        env=env_clean_trace,
        prior=prior,
        fixed_mu=fixed_mu,
        fixed_sigma=fixed_sigma,
        seed=123,
        delta=delta,
    )
    env_clean_trace.close()

    env_attack_trace_base = GaussianSignPOMDP(
        rho=rho,
        max_steps=max_steps,
        reward_correct=reward_correct,
        reward_wrong=reward_wrong,
        mu_range=mu_range,
        sigma_log_range=sigma_log_range,
        seed=0,
    )
    env_attack_trace = ObservationAttackWrapper(
        env_attack_trace_base,
        p_attack=p_attack,
        sigma_attack=sigma_attack,
        seed=999,  # attack RNG seed
    )
    trace_attack = run_one_episode_trace_fixed_mu(
        env=env_attack_trace,
        prior=prior,
        fixed_mu=fixed_mu,
        fixed_sigma=fixed_sigma,
        seed=123,  # same latent seed; observations differ due to attack noise
        delta=delta,
    )
    env_attack_trace.close()

    # ---- BOTTOM: many episodes cumulative reward clean vs attacked ----
    def clean_env_ctor(ep_idx: int):
        return GaussianSignPOMDP(
            rho=rho,
            max_steps=max_steps,
            reward_correct=reward_correct,
            reward_wrong=reward_wrong,
            mu_range=mu_range,
            sigma_log_range=sigma_log_range,
            seed=None,
        )

    def attacked_env_ctor(ep_idx: int):
        base = GaussianSignPOMDP(
            rho=rho,
            max_steps=max_steps,
            reward_correct=reward_correct,
            reward_wrong=reward_wrong,
            mu_range=mu_range,
            sigma_log_range=sigma_log_range,
            seed=None,
        )
        return ObservationAttackWrapper(base, p_attack=p_attack, sigma_attack=sigma_attack, seed=999 + ep_idx)

    _, cum_clean = run_many_episodes_cumreward(clean_env_ctor, prior, n_episodes=2000, seed=0, delta=delta)
    _, cum_attack = run_many_episodes_cumreward(attacked_env_ctor, prior, n_episodes=2000, seed=0, delta=delta)

    out_dir = Path(__file__).resolve().parent / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bandit_attack_comparison.png"

    plot_top_two_traces_and_bottom_cumrewards(
        trace_clean=trace_clean,
        trace_attack=trace_attack,
        cum_clean=cum_clean,
        cum_attack=cum_attack,
        out_path=out_path,
        p_attack=p_attack,
        sigma_attack=sigma_attack,
        x_jitter=0.12,  # points won't overlap
    )


if __name__ == "__main__":
    main()
