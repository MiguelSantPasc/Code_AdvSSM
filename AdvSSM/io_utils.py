"""
Small cache and filesystem helpers used by the experiment scripts.

Most scripts in this repository produce a figure together with the numerical
arrays that generated it. The convention is:

    figure.png  <->  figure.npz

The helpers below keep that convention in one place. They do not encode any
Kalman-filter mathematics directly; they support the reproducibility workflow
around the mathematical experiments by making it cheap to redraw plots without
rerunning Monte Carlo, PGD, or RL rollouts.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import numpy as np


def ensure_dir(path: str) -> str:
    """Create a directory if needed and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def data_path_for_plot(plot_path: str, suffix: str = ".npz") -> str:
    """Return a data-cache path next to a plot path."""
    root, _ = os.path.splitext(plot_path)
    return f"{root}{suffix}"


def save_npz(path: str, **arrays: Any) -> None:
    """Save arrays to a compressed NPZ file, creating parent directories."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> dict[str, Any]:
    """Load an NPZ file into a plain dictionary."""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def cached_npz(path: str, compute: Callable[[], dict[str, Any]], *, force: bool = False) -> dict[str, Any]:
    """Load a cached NPZ dictionary, or compute and save it."""
    if os.path.exists(path) and not force:
        print(f"[cache] loading data: {path}")
        return load_npz(path)

    print(f"[cache] computing data: {path}")
    data = compute()
    save_npz(path, **data)
    print(f"[cache] saved data: {path}")
    return data
