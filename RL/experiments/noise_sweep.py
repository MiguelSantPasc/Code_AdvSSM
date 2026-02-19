# RL/experiments/noise_sweep.py
# Evalúa un modelo entrenado bajo distintos niveles de OBS NOISE, comparando:
#   (1) RAW: policy con observación ruidosa directa
#   (2) KF matched: KF usa el sigma real (R correcto)
#   (3) KF mismatched: KF asume sigmas fijos para R (lista en config)
#
# Guarda:
#   RL/results/obs_noise_sweep.csv
#   RL/results/obs_noise_sweep_success_summary.png

import os
import sys
import csv
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIG (EDITA AQUÍ)
# ------------------------------------------------------------

DEVICE = "cpu"
EPISODES = 300
SEED = 0

# Sweep SOLO de ruido de observación (sigma REAL):
OBS_NOISE_LIST = [
    0.0, 0.1, 1.0, 10.0
]

# Ruido de proceso fijo (en el entorno):
PROC_NOISE_STD = 0.075

# Lista de sigmas que ASUME el KF (mismatch). Cada uno será una curva.
KF_ASSUMED_NOISE_LIST = [0.01, 1.0, 10.0 ]  # ajusta a lo que quieras

# Parámetros del entorno (consistentes con tu training)
GOAL_R_MIN = 2.5
GOAL_R_MAX = 15.0
MAX_STEPS = 25
GOAL_RADIUS = 2.5

MODEL_PATH = os.path.join("RL", "saved_models", "AdvRL_policy.pt")
OUT_DIR = os.path.join("RL", "results")

# Determinista recomendado para comparar robustez
DETERMINISTIC = True

# Parámetros del Kalman filter (mínimos numéricos)
KF_Q_FLOOR = 1e-6  # var mínima por paso (en delta-space)
KF_R_FLOOR = 1e-6  # var mínima en medición


# ------------------------------------------------------------
# Import robusto de AdvRL.py (asume que está en la raíz del repo)
# ------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from AdvRL import AdvRLEnvConfig, AdvRL2DEnv, ActorCritic  # type: ignore
except Exception as e:
    raise ImportError(
        "No pude importar AdvRL.py. Asegúrate de que AdvRL.py está en la raíz del proyecto "
        f"(ruta esperada: {_PROJECT_ROOT}/AdvRL.py). Error original: {e}"
    )


# ------------------------------------------------------------
# Utilidades KF (2D, lineal)
# Estado: delta_t = goal - x_t
# Dinámica: delta_{t+1} = delta_t - u_t + w
# Obs:      z_t = delta_t + v
# ------------------------------------------------------------

def unit_dir(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)

def kf_update(xhat: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # H = I
    S = P + R
    K = P @ np.linalg.inv(S)
    xhat = xhat + K @ (z - xhat)
    P = (np.eye(2, dtype=np.float32) - K) @ P
    return xhat, P

def kf_predict(xhat: np.ndarray, P: np.ndarray, u: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # A = I, B = -I  => delta_{t+1} = delta_t - u_t
    xhat = xhat - u
    P = P + Q
    return xhat, P


@torch.no_grad()
def eval_setting_raw(model: ActorCritic, device: torch.device, obs_noise_std: float) -> Dict[str, Any]:
    cfg = AdvRLEnvConfig(
        goal_r_min=GOAL_R_MIN,
        goal_r_max=GOAL_R_MAX,
        obs_noise_std=obs_noise_std,
        proc_noise_std=PROC_NOISE_STD,
        max_steps=MAX_STEPS,
        goal_radius=GOAL_RADIUS,
        seed=SEED,
    )
    env = AdvRL2DEnv(cfg)

    successes = 0
    for _ in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            if DETERMINISTIC:
                action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)
            else:
                action, _, _ = model.act(obs_t)

            obs, _, done, info = env.step(action)
            if done:
                successes += int(info.get("success", False))

    return {"success_rate": successes / float(EPISODES)}


@torch.no_grad()
def eval_setting_kf(
    model: ActorCritic,
    device: torch.device,
    obs_noise_std: float,
    assumed_obs_noise_std: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Simulación (ruido REAL):
      z_t = (goal - x_t) + v_t,  v_t ~ N(0, obs_noise_std^2 I)
    KF (ruido ASUMIDO):
      R construido con assumed_obs_noise_std (si None, usa el real => matched)
    Policy recibe unit_dir(delta_hat).
    """
    cfg = AdvRLEnvConfig(
        goal_r_min=GOAL_R_MIN,
        goal_r_max=GOAL_R_MAX,
        obs_noise_std=obs_noise_std,
        proc_noise_std=PROC_NOISE_STD,
        max_steps=MAX_STEPS,
        goal_radius=GOAL_RADIUS,
        seed=SEED,
    )
    env = AdvRL2DEnv(cfg)

    sigma_real = float(obs_noise_std)
    sigma_assumed = float(assumed_obs_noise_std) if assumed_obs_noise_std is not None else sigma_real

    r_var = max(sigma_assumed**2, KF_R_FLOOR)
    q_var = max(PROC_NOISE_STD**2, KF_Q_FLOOR)

    R = (r_var * np.eye(2)).astype(np.float32)
    Q = (q_var * np.eye(2)).astype(np.float32)

    rng = np.random.default_rng(SEED + 12345)

    successes = 0
    for _ in range(EPISODES):
        _ = env.reset()
        done = False

        # init KF con primera medición real
        true_delta = (env.goal - env.x).astype(np.float32)
        z = true_delta + rng.normal(0.0, sigma_real, size=(2,)).astype(np.float32)

        xhat = z.copy()
        P = R.copy()

        while not done:
            # update
            xhat, P = kf_update(xhat, P, z, R)

            # policy input
            obs_for_policy = unit_dir(xhat)
            obs_t = torch.tensor(obs_for_policy, dtype=torch.float32, device=device)

            if DETERMINISTIC:
                action = model.mean_action(obs_t).cpu().numpy().astype(np.float32)
            else:
                action, _, _ = model.act(obs_t)

            theta = float(action[0])
            u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

            # env step
            _, _, done, info = env.step(action)
            if done:
                successes += int(info.get("success", False))
                break

            # predict
            xhat, P = kf_predict(xhat, P, u, Q)

            # next measurement (real)
            true_delta = (env.goal - env.x).astype(np.float32)
            z = true_delta + rng.normal(0.0, sigma_real, size=(2,)).astype(np.float32)

    return {"success_rate": successes / float(EPISODES)}


def save_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _nice_ticks_for_many(xs: List[float]) -> List[float]:
    """Elige un subconjunto de ticks razonable (sin saturar el eje)."""
    if len(xs) <= 12:
        return xs

    # candidatos típicos
    candidates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
    ticks = [t for t in candidates if t in xs]

    # si sigue habiendo demasiados, recorta
    if len(ticks) > 9:
        ticks = [ticks[0]] + ticks[2:-2:2] + [ticks[-1]]

    # fallback: espaciado uniforme
    if len(ticks) < 5:
        idx = np.linspace(0, len(xs) - 1, 7).astype(int)
        ticks = [xs[i] for i in idx]
    return ticks


def plot_success_cream_paper(
    obs_noise_list: List[float],
    curves: Dict[str, List[float]],
    out_path: str,
) -> None:
    """
    curves: dict {label -> y_values aligned with obs_noise_list}
    Un solo gráfico, estilo artículo, crema, pastel más fuerte.
    """

    # --- Look & feel tipo artículo
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    cream = "#F7F1E3"
    cream2 = "#FBF7EE"
    ink = "#1F1F1F"
    grid = "#D9D2C3"

    # Colores “pastel fuerte”
    palette = [
        "#C65D5D",  # RAW (rojo pastel fuerte)
        "#2F7F6F",  # KF matched (verde/teal fuerte)
        "#D4A017",  # KF mismatch 1 (mostaza)
        "#4A86C5",  # KF mismatch 2 (azul)
        "#8E5AAE",  # KF mismatch 3 (morado)
        "#C46A2B",  # extra
        "#2E8B57",  # extra
    ]

    markers = ["o", "o", "s", "D", "^", "v", "P"]
    linestyles = ["-", "-", "--", "--", "--", "--", "--"]

    fig = plt.figure(figsize=(10.8, 5.6))
    fig.patch.set_facecolor(cream)
    ax = plt.gca()
    ax.set_facecolor(cream2)

    # Grid & spines
    ax.grid(True, which="major", color=grid, linewidth=0.9, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BFB8A8")
    ax.spines["bottom"].set_color("#BFB8A8")
    ax.tick_params(colors=ink)

    # Escala: symlog para manejar 0 y rango grande sin reventar el eje
    # (si prefieres lineal, comenta esta línea)
    ax.set_xscale("symlog", linthresh=0.05)

    # Plot curves
    labels = list(curves.keys())
    for i, label in enumerate(labels):
        y = curves[label]
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]

        ax.plot(
            obs_noise_list, y,
            label=label,
            color=color,
            linewidth=2.6,
            linestyle=ls,
            marker=marker,
            markersize=6.5,
            markerfacecolor=cream,
            markeredgecolor=ink,
            markeredgewidth=0.8,
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Success rate", color=ink)
    ax.set_xlabel("Observation noise std (σ)", color=ink)
    ax.set_title(
        f"Policy robustness vs observation noise  (process noise std = {PROC_NOISE_STD})",
        color=ink, pad=10
)

    # ticks legibles
    ticks = _nice_ticks_for_many(obs_noise_list)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], color=ink)

    # leyenda profesional (sin caja)
    ax.legend(frameon=False, loc="lower left", ncol=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    device = torch.device(DEVICE)

    model_abs = os.path.join(_PROJECT_ROOT, MODEL_PATH)
    if not os.path.exists(model_abs):
        raise FileNotFoundError(f"No encuentro el modelo en: {model_abs}\nRevisa MODEL_PATH.")

    model = ActorCritic(obs_dim=2, hidden=128).to(device)
    state = torch.load(model_abs, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Almacenamos curvas para el plot
    curves: Dict[str, List[float]] = {}
    curves["RAW"] = []
    curves["Kalman (matched R)"] = []
    for s in KF_ASSUMED_NOISE_LIST:
        curves[f"Kalman (assumed σ={s})"] = []

    # CSV en formato largo
    rows_csv: List[Dict[str, Any]] = []

    for o in OBS_NOISE_LIST:
        # RAW
        res_raw = eval_setting_raw(model, device=device, obs_noise_std=o)
        succ_raw = res_raw["success_rate"]
        curves["RAW"].append(succ_raw)
        rows_csv.append({
            "obs_noise_std": o,
            "proc_noise_std": PROC_NOISE_STD,
            "episodes": EPISODES,
            "deterministic": DETERMINISTIC,
            "seed": SEED,
            "method": "RAW",
            "kf_assumed_obs_noise_std": "",
            "success_rate": succ_raw,
        })

        # KF matched
        res_kf = eval_setting_kf(model, device=device, obs_noise_std=o, assumed_obs_noise_std=None)
        succ_kf = res_kf["success_rate"]
        curves["Kalman (matched R)"].append(succ_kf)
        rows_csv.append({
            "obs_noise_std": o,
            "proc_noise_std": PROC_NOISE_STD,
            "episodes": EPISODES,
            "deterministic": DETERMINISTIC,
            "seed": SEED,
            "method": "KF_matched",
            "kf_assumed_obs_noise_std": o,  # coincide con real
            "success_rate": succ_kf,
        })

        # KF mismatched list
        for s in KF_ASSUMED_NOISE_LIST:
            res_bad = eval_setting_kf(model, device=device, obs_noise_std=o, assumed_obs_noise_std=s)
            succ_bad = res_bad["success_rate"]
            curves[f"Kalman (assumed σ={s})"].append(succ_bad)
            rows_csv.append({
                "obs_noise_std": o,
                "proc_noise_std": PROC_NOISE_STD,
                "episodes": EPISODES,
                "deterministic": DETERMINISTIC,
                "seed": SEED,
                "method": "KF_mismatch",
                "kf_assumed_obs_noise_std": s,
                "success_rate": succ_bad,
            })

        # log compacto por línea (útil)
        msg = f"obs_noise={o:.4f} | RAW={succ_raw:.3f} | KFmatched={succ_kf:.3f}"
        for s in KF_ASSUMED_NOISE_LIST:
            msg += f" | KF(s={s})={curves[f'Kalman (assumed σ={s})'][-1]:.3f}"
        print(msg)

    out_csv = os.path.join(_PROJECT_ROOT, OUT_DIR, "obs_noise_sweep.csv")
    save_csv(rows_csv, out_csv)

    out_png = os.path.join(_PROJECT_ROOT, OUT_DIR, "obs_noise_sweep_success_summary.png")
    plot_success_cream_paper(OBS_NOISE_LIST, curves, out_png)

    print("\nGuardado:")
    print(" -", out_csv)
    print(" -", out_png)


if __name__ == "__main__":
    main()
