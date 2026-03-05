#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# -------- CONFIG --------
DEVICE = "cpu"
SEED = 2026

# Grid over "r" = (x,y) - goal  (relative position to the goal)
REL_LIM = 1.05
GRID_N = 240
BATCH = 16384
CMAP = "viridis"

# -------- Project imports --------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODEL_PATH = os.path.join(_PROJECT_ROOT, "RL", "saved_models", "AdvRL_v2_nowind_policy.pt")
OUT_DIR = os.path.join(_PROJECT_ROOT, "RL", "results")

from AdvRL import ActorCritic  # type: ignore


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


@torch.no_grad()
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    # NOTE: no-wind policy uses obs_dim=2
    model = ActorCritic(obs_dim=2, hidden=128, act_dim=2).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ------------------------------------------------------------
    # Grid in r = (x,y) - goal  (this is what we show in the plot)
    # IMPORTANT: if the model was trained with delta = goal - (x,y),
    # then delta_model = -r. We feed the model with -r.
    # ------------------------------------------------------------
    xs = np.linspace(-REL_LIM, REL_LIM, GRID_N, dtype=np.float32)
    ys = np.linspace(-REL_LIM, REL_LIM, GRID_N, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    r = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)  # (N,2), r=(x-goal)
    delta_model = -r  # (N,2), what the model expects if trained with goal - pos

    # Observations the model sees: [delta_x, delta_y]  (NO WIND)
    obs_all = delta_model.astype(np.float32)

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

    ax.set_title(r"V(r) NO WIND | r=(x,y)-goal (model sees delta=-r)")
    ax.set_xlabel(r"$r_x = x - goal_x$")
    ax.set_ylabel(r"$r_y = y - goal_y$")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("V")

    ax.legend(loc="lower right")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "v2_value_map_rpos_nowind.png")
    fig.savefig(out_path, dpi=260, facecolor="white")
    print(f"Saved: {out_path}")

    print(
        "V stats:",
        "min=", float(vals.min()),
        "max=", float(vals.max()),
        "mean=", float(vals.mean())
    )

    plt.close(fig)


if __name__ == "__main__":
    main()