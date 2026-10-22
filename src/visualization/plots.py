"""Publication-quality visual reporting for JSRT counterfactual experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common import ensure_directory
from src.data.superpixels import SuperpixelPartition


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("publication overlays require a 2-D grayscale image")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != gray.shape:
        raise ValueError("overlay mask and image must share a shape")
    rgb = np.repeat(np.clip(gray, 0.0, 1.0)[..., None], 3, axis=2)
    rgb[binary] = 0.42 * rgb[binary] + 0.58 * np.asarray(color, dtype=np.float32)
    return rgb


def plot_patient_panel(
    image: np.ndarray,
    gt_nodule_mask: np.ndarray,
    partition: SuperpixelPartition,
    explanation_masks: Mapping[str, np.ndarray],
    output_path: str | Path,
    image_id: str = "patient",
    dpi: int = 300,
) -> Path:
    """Save a seven-panel RX/GT/SLIC/four-occlusion explanation figure."""
    required = ("zero", "mean", "gaussian_blur", "inpaint")
    missing = [method for method in required if method not in explanation_masks]
    if missing:
        raise ValueError(f"explanation_masks lacks required methods: {missing}")
    target = Path(output_path)
    ensure_directory(target.parent)
    figure, axes = plt.subplots(1, 7, figsize=(23, 4), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("RX original")
    axes[1].imshow(_overlay(image, gt_nodule_mask, (0.0, 1.0, 0.0)))
    axes[1].set_title("Ground truth\nnódulo")
    axes[2].imshow(partition.boundary_image(image))
    axes[2].set_title(f"SLIC (K={partition.num_superpixels})")
    labels = {"zero": "Zero", "mean": "Mean", "gaussian_blur": "Gaussian blur", "inpaint": "Inpaint (Telea)"}
    colors = {
        "zero": (1.0, 0.15, 0.15),
        "mean": (1.0, 0.55, 0.0),
        "gaussian_blur": (0.25, 0.6, 1.0),
        "inpaint": (0.82, 0.25, 1.0),
    }
    for axis, method in zip(axes[3:], required, strict=True):
        axis.imshow(_overlay(image, explanation_masks[method], colors[method]))
        axis.contour(np.asarray(gt_nodule_mask, bool), levels=[0.5], colors="lime", linewidths=0.65)
        axis.set_title(labels[method])
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(f"Contrafactuales mínimos — {image_id}", fontsize=13)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_aggregate_distributions(
    results: pd.DataFrame,
    metrics: Sequence[str],
    output_path: str | Path,
    kind: str = "box",
    dpi: int = 300,
) -> Path:
    """Save box or violin plots grouped by the four occlusion mechanisms."""
    if kind not in {"box", "violin"}:
        raise ValueError("kind must be box or violin")
    required = {"occlusion_method", *metrics}
    absent = required.difference(results.columns)
    if absent:
        raise ValueError(f"results lacks required columns: {sorted(absent)}")
    target = Path(output_path)
    ensure_directory(target.parent)
    columns = min(3, len(metrics))
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 4.0 * rows), squeeze=False)
    sns.set_theme(style="whitegrid", context="paper")
    order = [method for method in ("zero", "mean", "gaussian_blur", "inpaint") if method in set(results["occlusion_method"])]
    for axis, metric in zip(axes.ravel(), metrics, strict=False):
        data = results[["occlusion_method", metric]].dropna()
        if kind == "box":
            sns.boxplot(data=data, x="occlusion_method", y=metric, order=order, ax=axis, showfliers=False)
        else:
            sns.violinplot(data=data, x="occlusion_method", y=metric, order=order, ax=axis, cut=0)
        axis.set_xlabel("Oclusión")
        axis.set_ylabel(metric.replace("_", " "))
        axis.tick_params(axis="x", rotation=22)
    for axis in axes.ravel()[len(metrics) :]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return target


def write_summary_tables(
    results: pd.DataFrame,
    metrics: Sequence[str],
    markdown_path: str | Path,
    latex_path: str | Path,
) -> pd.DataFrame:
    """Write mean±std aggregate tables in Markdown and LaTeX and return the frame."""
    required = {"occlusion_method", *metrics}
    absent = required.difference(results.columns)
    if absent:
        raise ValueError(f"results lacks required columns: {sorted(absent)}")
    summary = results.groupby("occlusion_method", sort=True)[list(metrics)].agg(["mean", "std", "median", "count"])
    markdown_target, latex_target = Path(markdown_path), Path(latex_path)
    ensure_directory(markdown_target.parent)
    ensure_directory(latex_target.parent)
    markdown_target.write_text(summary.to_markdown(floatfmt=".4f") + "\n", encoding="utf-8")
    latex_target.write_text(summary.to_latex(float_format="%.4f") + "\n", encoding="utf-8")
    return summary
