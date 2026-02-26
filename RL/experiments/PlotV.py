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

MODEL_PATH = os.path.join("RL", "saved_models", "AdvRL_policy.pt")
OUT_DIR = os.path.join("RL", "results")

DELTA_LIM = 58.0
GRID_N = 240
BATCH = 16384
CMAP = "viridis"

# -------- Project imports --------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from AdvRL import ActorCritic  # type: ignore


def unit_dir_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=-1, keepdim=True)
    return torch.where(n > eps, x / n, torch.zeros_like(x))


@torch.no_grad()
def get_value_from_forward(model: ActorCritic, obs_batch: torch.Tensor) -> torch.Tensor:
    """
    Your model(obs) returns (mu, log_std, value).
    value is out[2] with shape (N,).
    """
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.unsqueeze(0)
    out = model(obs_batch)
    v = out[2]
    if v.ndim == 2 and v.shape[1] == 1:
        v = v.squeeze(-1)
    return v


@torch.no_grad()
def main() -> None:
    device = torch.device(DEVICE)

    model_abs = os.path.join(_PROJECT_ROOT, MODEL_PATH)
    if not os.path.exists(model_abs):
        raise FileNotFoundError(f"Model not found at: {model_abs}")

    model = ActorCritic(obs_dim=2, hidden=128).to(device)
    state = torch.load(model_abs, map_location=device)
    model.load_state_dict(state)
    model.eval()

    xs = np.linspace(-DELTA_LIM, DELTA_LIM, GRID_N, dtype=np.float32)
    ys = np.linspace(-DELTA_LIM, DELTA_LIM, GRID_N, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    deltas = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)

    vals = np.zeros((deltas.shape[0],), dtype=np.float32)

    for i in range(0, deltas.shape[0], BATCH):
        d = torch.tensor(deltas[i:i + BATCH], dtype=torch.float32, device=device)
        obs = unit_dir_torch(d)  # obs(delta) = unit_dir(delta)
        v = get_value_from_forward(model, obs)
        vals[i:i + BATCH] = v.detach().cpu().numpy()

    vmap = vals.reshape(GRID_N, GRID_N)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 7.0), constrained_layout=True)
    im = ax.imshow(
        vmap,
        origin="lower",
        extent=[-DELTA_LIM, DELTA_LIM, -DELTA_LIM, DELTA_LIM],
        aspect="equal",
        cmap=CMAP,
    )
    ax.set_title(r"g($\delta$) = V(obs($\delta$)) where obs($\delta$)=unit($\delta$)")
    ax.set_xlabel(r"$\delta_x$")
    ax.set_ylabel(r"$\delta_y$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("g(delta)")

    os.makedirs(os.path.join(_PROJECT_ROOT, OUT_DIR), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, OUT_DIR, "g_value_map_delta.png")
    fig.savefig(out_path, dpi=260, facecolor="white")
    print(f"Saved: {out_path}")
    print("g(delta)=V(obs(delta)) stats:",
          "min=", float(vals.min()),
          "max=", float(vals.max()),
          "mean=", float(vals.mean()))

    plt.close(fig)


if __name__ == "__main__":
    main()