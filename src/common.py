"""Shared reproducibility, image and runtime helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed supported pseudo-random generators for a reproducible experiment."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        # The synthetic smoke-test path deliberately works without PyTorch.
        return


def select_torch_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to CUDA, MPS, or CPU, validating explicit requests."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("requested device must be one of auto, cpu, cuda, mps")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as error:
        if requested != "auto":
            raise RuntimeError("PyTorch is required for a non-CPU device") from error
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return "cuda"
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def as_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Convert an image to finite grayscale ``float32`` values in ``[0, 1]``.

    Integer images are scaled by their dtype range. Floating inputs in a different
    finite range are min-max normalized, preserving already normalized images.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] not in (3, 4):
            raise ValueError("3-D images must use 3 or 4 channels in the last axis")
        rgb = array[..., :3].astype(np.float32)
        array = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    if array.ndim != 2:
        raise ValueError("image must have shape (height, width) or (height, width, channels)")
    if not np.isfinite(array).all():
        raise ValueError("image contains non-finite intensities")
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        result = array.astype(np.float32) / maximum if maximum else array.astype(np.float32)
    else:
        result = array.astype(np.float32, copy=False)
        low, high = float(result.min()), float(result.max())
        if low < 0.0 or high > 1.0:
            result = np.zeros_like(result) if high == low else (result - low) / (high - low)
    return np.clip(result, 0.0, 1.0)


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stable_stem(path: str | Path) -> str:
    """Canonicalize a file stem for tolerant JSRT image/mask matching."""
    raw = Path(path).stem.lower()
    return "".join(character for character in raw if character.isalnum())
