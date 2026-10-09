#!/usr/bin/env python3
"""Record paired CartPole episodes, optionally with covariance defense.

Protocol
--------
* ``noisy_kf``: noisy observations, nominal KF, centered discrete DQN actions.
* ``noisy_attack``: the same noise and KF, with the estimated-return PGD observation
  attack. This is the benchmark's ``Attack + KF`` case, without WoLF or
  covariance adaptation. The initial observation is never attacked. Perceived
  actions are scored by their Bellman return from KF(o_nominal), not from the
  attacked belief. Posterior sampling matches RL; a softmax gradient surrogate
  handles the discrete DQN while candidate scores use actual greedy actions.
* Both episodes use the same initial seed and independent, identically seeded
  noise streams. Their physical trajectories can diverge after their actions
  diverge. No seeds are searched for a more dramatic outcome.
* ``noisy_attack_defense`` uses the same PGD attack with the benchmark's
  covariance-adapted update. Its lambda is the covariance scale: at each step,
  lambda_t = defense_lambda * lambda_max(P_pred + R). The initial update is
  nominal; subsequent updates use the attacked observation as adv_target when
  an attack occurs, and None otherwise, exactly as in the benchmark.
* Reuse the repository's nonlinear plant (two 0.01 s substeps), EKF predictor
  (0.02 s), posterior update and attack, with physical cart stops at +/-2.4 m.
  Inelastic collisions transfer the cart impulse to the pole. Both the plant
  and the attack's Bellman model use these contacts and the centered reward
  r = 1 - 0.8*(x/2.4)^2. Return is therefore not the number of steps.
* Load the separate DQN adapted by train_cartpole_centered.py. This renderer
  never retrains implicitly or overwrites the original downloaded checkpoint.

Display and output
------------------
Gymnasium renders the actual hidden state s_t over blurred, translucent poses:
blue for the noisy observation o_t, red for its attacked replacement when PGD
is active. The posterior estimate still drives the controller but is not drawn,
keeping the actual simulation clearly visible.
Only a short English title, mathematical legend, step and action annotate the scene.
The compact footer turns red during attacks and green otherwise. Mathtext
renders LaTeX notation locally without requiring a separate TeX installation.
Playback runs at quarter speed and pauses briefly on each attacked decision.
With physical stops the camera is fixed, showing both stops, a position scale
and the target x=0. The green central band is a visual target, not a constraint.
For optional legacy rollouts without stops, the camera follows the cart.

CartPole normally ends an episode at 12 degrees. For these videos only, the
simulation may continue beyond that threshold with the SAME DQN controller,
KF updates, noise RNG and attack RNG still active, until the pole is horizontal
or the continuation time cap or shared total horizon is hit. Falling is never forced: the controller
may recover. The first standard failure is recorded and its benchmark return
is frozen; subsequent controlled motion is outside the standard benchmark.

All experiment defaults are editable in main(). Run from the repository root:
    .\\.venv\\Scripts\\python.exe Gymnasium/cartpole_videos.py
Optional overrides:
    ... cartpole_videos.py --seed 100 --max-steps 500
    ... cartpole_videos.py --mode noisy_attack_defense --defense-lambda 2.0
    ... cartpole_videos.py --mode all --defense-lambda 2.0
Install the video encoder with:
    .\\.venv\\Scripts\\python.exe -m pip install -r Gymnasium/requirements-video.txt
Videos are written to Gymnasium/outputs/videos/, independently of the cwd.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch

import cartpole_covadapt_compare_epsilons_wolf as cartpole


@dataclass(frozen=True)
class VideoConfig:
    """Explicit local settings shared by the two paired episodes."""

    seed: int
    max_steps: int
    obs_noise_std: np.ndarray
    discount_delta: float
    attack_prob: float
    attack_eps: float
    pgd_steps: int
    pgd_step_size: float
    mc_samples: int
    device: str
    post_failure_steps: int = 0
    policy_temperature: float = 1.0
    defense_lambda: float = 2.0
    omega_h: float = 0.50
    omega_o: float = 0.50
    gamma_threshold: float = 0.20


@dataclass(frozen=True)
class Snapshot:
    """State and current decision at one control instant, before its action."""

    step: int
    state: np.ndarray
    noisy: np.ndarray | None
    observation: np.ndarray | None
    estimate: np.ndarray | None
    action: int | None
    attacked: bool
    reward: float
    outcome: str
    benchmark_failure_step: int | None = None


def simulate_episode(*, model, ssm, R: np.ndarray, config: VideoConfig,
                     with_attack: bool, with_defense: bool = False) -> list[Snapshot]:
    """Save one trajectory using the same update order as the benchmark.

    Small state arrays are kept in memory, rather than rendered frames. This
    also separates simulation time from playback speed and permits validation
    of the trajectory without opening a rendering window.
    """
    env = gym.make("CartPole-v1", max_episode_steps=config.max_steps)
    rng_noise = np.random.default_rng(config.seed + 707_002)
    rng_gate = np.random.default_rng(config.seed + 707_001)
    snapshots = []
    reward_sum = 0.0
    failure_step = None
    try:
        state, _ = cartpole.reset_cartpole_rollout(env, seed=config.seed)
        # Match the benchmark's initialization, including its initial prior.
        m_pred = np.asarray(state, dtype=np.float32)
        P_pred = cartpole.project_to_psd(R.copy())
        for step in range(config.max_steps):
            noise = rng_noise.normal(0.0, config.obs_noise_std, size=4).astype(np.float32)
            noisy = (state + noise).astype(np.float32)
            attacked = bool(with_attack and step > 0 and rng_gate.random() < config.attack_prob)
            observation = noisy
            if attacked:
                observation, _, m_post, P_post = cartpole.estimated_return_pgd_attack_observation(
                    model=model, obs_nom=noisy, m_pred=m_pred, P_pred=P_pred, R=R,
                    attack_center=np.asarray(m_pred, dtype=np.float32),
                    attack_sigma=cartpole.project_to_psd(
                        np.asarray(P_pred, dtype=float) + np.asarray(R, dtype=float)
                    ),
                    attack_eps=config.attack_eps, pgd_steps=config.pgd_steps,
                    pgd_step_size=config.pgd_step_size, mc_samples=config.mc_samples,
                    rng_seed=config.seed + 10_000 * step, device=config.device,
                    ssm=ssm, current_step_index=step, max_episode_steps=config.max_steps,
                    policy_temperature=config.policy_temperature,
                )
                m_post = np.asarray(m_post, dtype=np.float32)
                P_post = np.asarray(P_post, dtype=np.float32)
            else:
                m_post, P_post = cartpole.kf_update_state(
                    m_pred=m_pred, P_pred=P_pred, y_obs=observation, R=R,
                )

            if with_defense and step > 0:
                # Recompute from the prior: PGD's nominal posterior scores the
                # attack, while the defended posterior drives the actual DQN.
                m_post, P_post, _ = cartpole.covariance_adapted_kf_update_state(
                    m_pred=m_pred, P_pred=P_pred, y_obs=observation, R=R,
                    adv_target=observation if attacked else None,
                    c_scale=config.defense_lambda, omega_h=config.omega_h,
                    omega_o=config.omega_o, delta_threshold=config.gamma_threshold,
                )

            action = cartpole.select_action(model, m_post)
            snapshots.append(Snapshot(
                step, state.copy(), noisy.copy(), observation.copy(), m_post.copy(),
                action, attacked, reward_sum, "", failure_step,
            ))
            m_pred, P_pred = cartpole.kf_predict_state(
                m_post=m_post, P_post=P_post,
                force=cartpole.action_to_force(action, ssm.force_mag), Q=None,
                discount_delta=config.discount_delta, ssm=ssm,
            )
            if failure_step is None:
                state, reward, terminated, truncated, _ = cartpole.step_cartpole_rollout(
                    env, action, ssm=ssm,
                )
                reward_sum += reward
                if terminated:
                    failure_step = step + 1
            else:
                # Beyond Gymnasium's terminal threshold, keep the ENTIRE closed
                # loop active. Do not use env.step() on a terminated environment,
                # reset the RNGs, remove the force, or add benchmark rewards.
                state = cartpole.cartpole_real_dynamics(
                    state=state,
                    force=cartpole.action_to_force(action, ssm.force_mag), ssm=ssm,
                )
                truncated = False

            outcome = ""
            if failure_step is not None:
                if abs(float(state[2])) >= np.pi / 2:
                    outcome = "Palo horizontal pese al control activo"
                elif step + 1 >= min(config.max_steps, failure_step + config.post_failure_steps):
                    outcome = ("Fallo del episodio CartPole" if config.post_failure_steps == 0
                               else "Fin de continuacion: no alcanzo 90 grados")
            elif truncated:
                outcome = "Limite de pasos alcanzado"
            if outcome:
                snapshots.append(Snapshot(
                    step + 1, state.copy(), None, None, None, None, False,
                    reward_sum, outcome, failure_step,
                ))
                break
    finally:
        env.close()
    return snapshots


def observation_ghost(env, state: np.ndarray, *, color: tuple[int, int, int],
                      opacity: int, blur_radius: float, camera_center: float) -> Image.Image:
    """Tint the native CartPole silhouette, excluding its background and rail.

    Reusing Gymnasium rendering guarantees the observed and actual poses share
    the same position/angle convention and pixel scale. Blur affects only the
    visual overlay; it never smooths the observations used by the controller.
    """
    env.unwrapped.state = np.asarray(state, dtype=np.float64).copy()
    env.unwrapped.state[0] -= camera_center
    rgb = env.render()
    foreground = np.any(rgb < 245, axis=2)
    # The rail spans the entire image width; omit it from the ghost mask.
    foreground[foreground.mean(axis=1) > 0.95] = False
    alpha = Image.fromarray((foreground * opacity).astype(np.uint8))
    alpha = alpha.filter(ImageFilter.GaussianBlur(blur_radius))
    ghost = Image.new("RGBA", (rgb.shape[1], rgb.shape[0]), color + (0,))
    ghost.putalpha(alpha)
    return ghost


@lru_cache(maxsize=32)
def mathematical_label(text: str, size: int = 14) -> Image.Image:
    """Cache transparent math/text labels; render each once, never per frame."""
    from matplotlib import rc_context
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    buffer = BytesIO()
    with rc_context({"savefig.transparent": True}):
        math_to_image(text, buffer, prop=FontProperties(size=size), dpi=100,
                      format="png", color="#303747")
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        return rendered.convert("RGBA")


def annotated_frame(env, snapshot: Snapshot, *, label: str, font,
                    ghost_opacity: int, ghost_blur_radius: float,
                    show_attack_legend: bool, wall_position: float | None = None,
                    goal_half_width: float = 0.25) -> np.ndarray:
    """Draw observation ghosts behind s_t with a short title and minimal text."""
    blue = (125, 180, 226)
    red = (229, 123, 141)
    width, height = env.unwrapped.screen_width, env.unwrapped.screen_height
    if wall_position is not None:
        # This field controls ONLY the native renderer's scale here. The wider
        # fixed camera fits both stops and a full fallen pole beside either one.
        env.unwrapped.x_threshold = wall_position + 1.35
    # Beyond the ordinary failure threshold the cart may leave the stock
    # renderer's field of view. Pan all poses together, leaving a pole length
    # of room on either side. This is only a rendering coordinate transform.
    camera_center = (0.0 if wall_position is not None else
                     float(snapshot.state[0] - np.clip(snapshot.state[0], -1.0, 1.0)))
    scene = Image.new("RGBA", (width, height), "white")
    scale = width / (2 * env.unwrapped.x_threshold)
    rail_y = height - 101
    if wall_position is not None:
        background = ImageDraw.Draw(scene)
        background.rectangle((width / 2 - goal_half_width * scale, 140,
                              width / 2 + goal_half_width * scale, rail_y + 31),
                             fill="#EDF6EF")
    if snapshot.noisy is not None:
        scene = Image.alpha_composite(scene, observation_ghost(
            env, snapshot.noisy, color=blue, opacity=ghost_opacity,
            blur_radius=ghost_blur_radius, camera_center=camera_center,
        ))
    if snapshot.attacked:
        scene = Image.alpha_composite(scene, observation_ghost(
            env, snapshot.observation, color=red, opacity=ghost_opacity,
            blur_radius=ghost_blur_radius, camera_center=camera_center,
        ))
    # Replay saved true states directly: stepping again would create a different
    # trajectory and would bypass the benchmark's finer plant integration.
    env.unwrapped.state = np.asarray(snapshot.state, dtype=np.float64).copy()
    env.unwrapped.state[0] -= camera_center
    rgb = env.render()
    # Paste only the opaque real geometry, so the ghosts remain behind it.
    foreground = Image.fromarray((np.any(rgb < 255, axis=2) * 255).astype(np.uint8))
    scene.paste(Image.fromarray(rgb), (0, 0), foreground)
    canvas = Image.new("RGB", (width, height + 42), "#F5F5FA")
    canvas.paste(scene.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    ink = "#303747"
    if wall_position is not None:
        # The stops' inner faces meet the native renderer's 50-pixel cart body
        # exactly when its CENTER reaches +/-wall_position. They are low enough
        # to represent cart end stops, not unmodeled walls struck by the pole.
        for side in (-1, 1):
            face = width / 2 + side * (wall_position * scale + 25)
            outer = face + side * 14
            draw.rectangle((min(face, outer), rail_y + 3, max(face, outer), rail_y + 34),
                           fill="#B9C1CE", outline="#788699", width=2)
        for position in np.linspace(-wall_position, wall_position, 5):
            pixel = width / 2 + position * scale
            draw.line((pixel, rail_y + 34, pixel, rail_y + 40), fill="#89959A", width=1)
            draw.text((pixel, rail_y + 44), f"{position:+.1f}", anchor="mt", fill=ink, font=font)
        for y in range(145, rail_y + 32, 12):
            draw.line((width / 2, y, width / 2, y + 5), fill="#94B9A1", width=1)
    # User-requested short title and an in-scene legend with proper subscripts.
    title = mathematical_label(label, size=16)
    canvas.paste(title, ((width - title.width) // 2, 12), title)
    legend = [(r"True state: $s_t$", (202, 152, 101)), (r"Noisy: $o_t$", blue)]
    if show_attack_legend:
        legend.append((r"Attacked: $o_t^{\mathrm{adv}}$", red))
    draw.rounded_rectangle((8, 48, 204, 56 + 28 * len(legend)), radius=6, fill="#F5F5FA")
    for index, (text, color) in enumerate(legend):
        y = 56 + 28 * index
        draw.rectangle((18, y, 32, y + 13), fill=color)
        legend_label = mathematical_label(text)
        canvas.paste(legend_label, (42, y - 2), legend_label)
    badge = "#F2C4C4" if snapshot.attacked else "#CBE5DC"
    draw.rectangle((0, height, width, height + 42), fill=badge)
    draw.text((16, height + 12), f"Step {snapshot.step}", fill=ink, font=font)
    action = "--" if snapshot.action is None else ("right" if snapshot.action else "left")
    draw.text((width - 16, height + 12), f"Action: {action}", anchor="rt", fill=ink, font=font)
    return np.asarray(canvas)


def save_video(snapshots: list[Snapshot], *, output: Path, label: str,
               seed: int, tau: float, final_hold_seconds: float,
               playback_speed: float, attack_hold_seconds: float,
               ghost_opacity: int, ghost_blur_radius: float,
               wall_position: float | None = None, goal_half_width: float = 0.25) -> None:
    """Encode slow playback with attack pauses, without changing the simulation."""
    import imageio_ffmpeg

    if not snapshots or not 0 < playback_speed <= 1:
        raise ValueError("Provide a trajectory and a playback speed in (0, 1].")
    if min(final_hold_seconds, attack_hold_seconds, ghost_blur_radius) < 0:
        raise ValueError("Hold times and ghost blur must be nonnegative.")
    if not 0 <= ghost_opacity <= 255:
        raise ValueError("Ghost opacity must be between 0 and 255.")
    fps = 1.0 / tau
    show_attack_legend = any(snapshot.attacked for snapshot in snapshots)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary sibling prevents an interrupted run from replacing a complete
    # video with a partial one. Only the successful encode replaces the target.
    temporary = output.with_name(output.stem + ".partial.mp4")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    writer = None
    try:
        env.reset(seed=seed)
        font = ImageFont.load_default(size=16)
        for index, snapshot in enumerate(snapshots):
            frame = annotated_frame(
                env, snapshot, label=label, font=font, ghost_opacity=ghost_opacity,
                ghost_blur_radius=ghost_blur_radius, show_attack_legend=show_attack_legend,
                wall_position=wall_position, goal_half_width=goal_half_width,
            )
            if writer is None:
                writer = imageio_ffmpeg.write_frames(
                    str(temporary), (frame.shape[1], frame.shape[0]), fps=fps,
                    codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
                    output_params=["-movflags", "+faststart"],
                )
                writer.send(None)
            # Cumulative rounding also supports non-integer slow-motion factors.
            repeats = round((index + 1) / playback_speed) - round(index / playback_speed)
            if snapshot.attacked:
                repeats += round(attack_hold_seconds * fps)
            for _ in range(repeats):
                writer.send(np.ascontiguousarray(frame))
        # Holding the terminal frame adds presentation time, not simulation steps.
        for _ in range(round(final_hold_seconds / tau)):
            writer.send(np.ascontiguousarray(frame))
        writer.close()
        writer = None
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not produce a video: {temporary}")
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.close()
        env.close()
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Define editable defaults and render the selected single-seed episodes."""
    seed = 100
    max_steps = 500
    device = "cpu"
    obs_noise_std = np.array([0.10, 0.22, 0.05, 0.22], dtype=float)
    discount_delta = 0.94
    attack_prob = 0.20
    attack_eps = 9.49
    pgd_steps = 20
    pgd_step_size = 0.34
    mc_samples = 64
    policy_temperature = 1.0  # Search gradient only; DQN actions stay discrete.
    mode = "original"  # Original pair, noisy_attack_defense, or all three.
    defense_lambda = 2.0  # Multiplier of lambda_max(P_pred + R), as in benchmark.
    omega_h = 0.50
    omega_o = 0.50
    gamma_threshold = 0.20
    filter_tau = 0.02
    real_tau = 0.01
    wall_position = 2.4
    wall_restitution = 0.0
    center_reward_weight = 0.8
    goal_half_width = 0.25  # Visual central target band, in meters.
    final_hold_seconds = 1.5
    playback_speed = 0.25  # 0.25x: four times slower than physical time.
    attack_hold_seconds = 0.40  # Extra pause on each attacked observation.
    # Keep the full closed loop active after 12 deg, capped by the SAME total
    # max_steps horizon as the baseline. Never extend until a desired outcome.
    post_failure_seconds = 10.0
    ghost_opacity = 180  # Alpha in [0, 255]; real geometry remains opaque.
    ghost_blur_radius = 1.0  # Pixels; observation values are never smoothed.
    output_dir = Path(__file__).resolve().parent / "outputs" / "videos"
    model_path = (Path(__file__).resolve().parent / "outputs" / "saved_models"
                  / "sb3_dqn_cartpole_centered_v1" / "dqn-CartPole-centered.zip")
    meas_corr = np.array([
        [1.00, 0.18, 0.06, 0.00],
        [0.18, 1.00, 0.14, 0.22],
        [0.06, 0.14, 1.00, 0.18],
        [0.00, 0.22, 0.18, 1.00],
    ], dtype=float)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--max-steps", type=int, default=max_steps)
    parser.add_argument("--mode", choices=("original", "noisy_attack_defense", "all"), default=mode)
    parser.add_argument("--defense-lambda", type=float, default=defense_lambda)
    args = parser.parse_args()
    if args.seed < 0 or not 1 <= args.max_steps <= 500:
        parser.error("seed must be nonnegative and max-steps must be between 1 and 500")
    if not np.isfinite(args.defense_lambda) or args.defense_lambda < 0:
        parser.error("defense-lambda must be finite and nonnegative")
    # Fail before simulating if the encoder is missing; no automatic downloads.
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit("Install video support: python -m pip install -r Gymnasium/requirements-video.txt") from exc
    imageio_ffmpeg.get_ffmpeg_exe()
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing pretrained DQN checkpoint: {model_path}")

    torch.set_num_threads(1)
    config = VideoConfig(args.seed, args.max_steps, obs_noise_std, discount_delta,
                         attack_prob, attack_eps, pgd_steps, pgd_step_size, mc_samples, device,
                         post_failure_steps=round(post_failure_seconds / filter_tau),
                         policy_temperature=policy_temperature,
                         defense_lambda=args.defense_lambda, omega_h=omega_h,
                         omega_o=omega_o, gamma_threshold=gamma_threshold)
    ssm = cartpole.build_cartpole_linear_ssm(
        filter_tau=filter_tau, real_tau=real_tau, wall_position=wall_position,
        wall_restitution=wall_restitution, center_reward_weight=center_reward_weight,
    )
    from train_cartpole_centered import verify_centered_checkpoint
    verify_centered_checkpoint(model_path, ssm=ssm)
    # The predictor uses discount inflation (Q=None), so only R is needed here.
    meas_scale = np.diag(obs_noise_std).astype(np.float32)
    R = cartpole.project_to_psd((meas_scale @ meas_corr @ meas_scale).astype(np.float32))
    model = cartpole.load_cartpole_policy(str(model_path), torch.device(device))
    model.policy.set_training_mode(False)
    episodes = [
        (False, False, "noisy_kf", "Control with noisy observations"),
        (True, False, "noisy_attack", "Control under attack without defense"),
    ]
    if args.mode in ("noisy_attack_defense", "all"):
        lambda_tag = str(float(config.defense_lambda)).replace(".", "p")
        defended_episode = (True, True, f"noisy_attack_defense_lambda{lambda_tag}",
                            rf"Control under attack with defense ($\lambda={config.defense_lambda:.1f}$)")
        episodes = episodes + [defended_episode] if args.mode == "all" else [defended_episode]
    try:
        for with_attack, with_defense, name, label in episodes:
            print(f"Simulating {label}, seed={config.seed}...", flush=True)
            snapshots = simulate_episode(model=model, ssm=ssm, R=R,
                                         config=config, with_attack=with_attack,
                                         with_defense=with_defense)
            # Count physical samples, not repeated frames used for slow motion.
            positions = np.asarray([snapshot.state[0] for snapshot in snapshots[1:]])
            contact_steps = int(np.sum(np.abs(positions) >= wall_position - 1e-5))
            output = output_dir / f"cartpole_seed{config.seed}_{name}.mp4"
            save_video(snapshots, output=output, label=label, seed=config.seed,
                       tau=ssm.tau, final_hold_seconds=final_hold_seconds,
                       playback_speed=playback_speed, attack_hold_seconds=attack_hold_seconds,
                       ghost_opacity=ghost_opacity, ghost_blur_radius=ghost_blur_radius,
                       wall_position=wall_position, goal_half_width=goal_half_width)
            print(f"Saved {output}\n  Return={snapshots[-1].reward:.2f}; "
                  f"attacks={sum(s.attacked for s in snapshots)}; "
                  f"benchmark_failure_step={snapshots[-1].benchmark_failure_step}; "
                  f"final_step={snapshots[-1].step}; "
                  f"mean_abs_x={np.mean(np.abs(positions)):.3f} m; "
                  f"max_abs_x={np.max(np.abs(positions)):.3f} m; "
                  f"wall_contact_steps={contact_steps}; "
                  f"theta={np.degrees(snapshots[-1].state[2]):.2f} deg; "
                  f"{snapshots[-1].outcome}", flush=True)
    finally:
        if model.get_env() is not None:
            model.get_env().close()


if __name__ == "__main__":
    main()
