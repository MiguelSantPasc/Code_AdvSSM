#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# -------- CONFIG --------
DEVICE = "cpu"
SEED = 2026

# Grid sobre "r" = (x,y) - goal  (posición relativa al goal)
REL_LIM = 1.05
GRID_N = 240
BATCH = 16384
CMAP = "viridis"

# --- Viento fijo
WIND_EPS = 0.9
WIND_PSI = 1.5  # rad (0 => viento +x)
WIND_X = WIND_EPS * math.cos(WIND_PSI)
WIND_Y = WIND_EPS * math.sin(WIND_PSI)

# -------- Project imports --------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODEL_PATH = os.path.join(_PROJECT_ROOT, "RL", "saved_models", "AdvRL_v2_policy.pt")
OUT_DIR = os.path.join(_PROJECT_ROOT, "RL", "results")

from AdvRL_wind import ActorCritic  # type: ignore


@torch.no_grad()
def get_value(model: ActorCritic, obs_batch: torch.Tensor) -> torch.Tensor:
    """
    model(obs) -> (mu, std, v)
    We only need v (shape (N,))
    """
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.unsqueeze(0)

    mu, std, v = model(obs_batch)

    if v.ndim == 2 and v.shape[1] == 1:
        v = v.squeeze(-1)
    return v


def _line_segment_through_origin_in_box(dx: float, dy: float, L: float):
    """
    Segment of the infinite line through origin with direction (dx,dy),
    clipped to [-L, L] x [-L, L]. Returns (x0,y0,x1,y1) or None.
    """
    n = math.hypot(dx, dy)
    if n < 1e-12:
        return None

    ux, uy = dx / n, dy / n

    tmin = -float("inf")
    tmax = float("inf")

    if abs(ux) > 1e-12:
        tx1 = (-L) / ux
        tx2 = ( L) / ux
        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))

    if abs(uy) > 1e-12:
        ty1 = (-L) / uy
        ty2 = ( L) / uy
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))

    x0, y0 = tmin * ux, tmin * uy
    x1, y1 = tmax * ux, tmax * uy
    return x0, y0, x1, y1


@torch.no_grad()
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = ActorCritic(obs_dim=4, hidden=128, act_dim=2).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ------------------------------------------------------------
    # Grid en r = (x,y) - goal  (esto es lo que mostramos en el plot)
    # IMPORTANT: si el modelo fue entrenado con delta = goal - (x,y),
    # entonces delta_model = -r. Alimentamos al modelo con -r.
    # ------------------------------------------------------------
    xs = np.linspace(-REL_LIM, REL_LIM, GRID_N, dtype=np.float32)
    ys = np.linspace(-REL_LIM, REL_LIM, GRID_N, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    r = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)  # (N,2), r=(x-goal)
    delta_model = -r  # (N,2), lo que espera el modelo si entrenó con goal-pos

    # Viento fijo en todo el grid
    wind = np.array([WIND_X, WIND_Y], dtype=np.float32)
    wind_batch = np.repeat(wind[None, :], delta_model.shape[0], axis=0)

    # Observaciones que ve el modelo: [delta_x, delta_y, wind_x, wind_y]
    obs_all = np.concatenate([delta_model, wind_batch], axis=1).astype(np.float32)

    vals = np.zeros((obs_all.shape[0],), dtype=np.float32)
    for i in range(0, obs_all.shape[0], BATCH):
        ob = torch.tensor(obs_all[i:i + BATCH], dtype=torch.float32, device=device)
        v = get_value(model, ob)
        vals[i:i + BATCH] = v.detach().cpu().numpy()

    vmap = vals.reshape(GRID_N, GRID_N)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 7.0), constrained_layout=True)
    im = ax.imshow(
        vmap,
        origin="lower",
        extent=[-REL_LIM, REL_LIM, -REL_LIM, REL_LIM],
        aspect="equal",
        cmap=CMAP,
    )

    # Origin marker: r=(0,0) <=> (x,y)=goal
    ax.scatter(
        0.0, 0.0,
        s=80,
        c="red",
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        zorder=7,
        label="at goal (r=0)"
    )

    # Wind direction: mini-arrows along the wind line across the whole map
    seg = _line_segment_through_origin_in_box(WIND_X, WIND_Y, REL_LIM)
    if seg is not None:
        x0, y0, x1, y1 = seg

        # Unit direction of wind
        n = math.hypot(WIND_X, WIND_Y)
        ux, uy = WIND_X / n, WIND_Y / n

        # Where to place arrows along the segment (avoid extreme edges)
        N_ARROWS = 11
        ts = np.linspace(0.08, 0.92, N_ARROWS, dtype=np.float32)
        xs_line = x0 + ts * (x1 - x0)
        ys_line = y0 + ts * (y1 - y0)

        # Arrow length in DATA units
        ARROW_LEN = 0.14 * (2.0 * REL_LIM)
        U = np.full_like(xs_line, ux * ARROW_LEN)
        V = np.full_like(ys_line, uy * ARROW_LEN)

        ax.quiver(
            xs_line, ys_line, U, V,
            angles="xy", scale_units="xy", scale=1.0,
            color="white", alpha=0.85,
            width=0.006, headwidth=4.5, headlength=6.0, headaxislength=5.0,
            pivot="mid",
            zorder=6,
        )

    # Wind label
    ax.text(
        -0.98 * REL_LIM,
        0.93 * REL_LIM,
        f"wind = ({WIND_X:.3f}, {WIND_Y:.3f})\npsi={WIND_PSI:.2f}, eps={WIND_EPS:.2f}",
        color="white",
        fontsize=11,
        weight="bold",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=3.0),
        zorder=9
    )

    ax.set_title(
        rf"V(r | wind=({WIND_X:.3f},{WIND_Y:.3f})) with r=(x,y)-goal (model sees delta=-r)"
    )
    ax.set_xlabel(r"$r_x = x - goal_x$")
    ax.set_ylabel(r"$r_y = y - goal_y$")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("V")

    ax.legend(loc="lower right")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "v2_value_map_rpos_wind.png")
    fig.savefig(out_path, dpi=260, facecolor="white")
    print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()