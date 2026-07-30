"""Artifact paths and NPZ cache helpers for experiment scripts.

This module keeps filesystem/cache behavior in the shared package so
experiment folders do not need local `*_utils.py` modules.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import numpy as np


def ensure_dir(path: str) -> str:
    """Create `path` if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def outputs_root_for(module_dir: str) -> str:
    """Return the canonical outputs root for a module directory."""
    return ensure_dir(os.path.join(module_dir, "outputs"))


def figures_dir_for(module_dir: str) -> str:
    """Return the canonical figure-output directory for a module."""
    return ensure_dir(os.path.join(outputs_root_for(module_dir), "figures"))


def data_dir_for(module_dir: str) -> str:
    """Return the canonical numerical-cache directory for a module."""
    return ensure_dir(os.path.join(outputs_root_for(module_dir), "data"))


def saved_models_dir_for(module_dir: str) -> str:
    """Return the canonical model-checkpoint directory for RL experiments."""
    return ensure_dir(os.path.join(outputs_root_for(module_dir), "saved_models"))


def data_path_for_plot(plot_path: str, suffix: str = ".npz") -> str:
    """Return the canonical data-cache path associated with a figure path."""
    plot_dir = os.path.dirname(plot_path)
    plot_stem = os.path.splitext(os.path.basename(plot_path))[0]

    if os.path.basename(plot_dir) == "figures" and os.path.basename(os.path.dirname(plot_dir)) == "outputs":
        target_dir = os.path.join(os.path.dirname(plot_dir), "data")
    elif os.path.basename(plot_dir) == "output":
        target_dir = os.path.join(os.path.dirname(plot_dir), "outputs", "data")
    else:
        target_dir = plot_dir

    ensure_dir(target_dir)
    return os.path.join(target_dir, f"{plot_stem}{suffix}")


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
