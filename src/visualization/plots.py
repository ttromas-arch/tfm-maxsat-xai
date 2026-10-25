"""Publication-quality, memory-safe reporting for JSRT counterfactual experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns

from src.common import ensure_directory
from src.data.superpixels import SuperpixelPartition
from src.occlusion.engine import OCCLUSION_METHODS, OcclusionEngine


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    gray = np.asarray(image, dtype=np.float32)
    binary = np.asarray(mask, dtype=bool)
    if gray.ndim != 2 or binary.shape != gray.shape:
        raise ValueError("overlay requires matching 2-D image and mask")
    rgb = np.repeat(np.clip(gray, 0.0, 1.0)[..., None], 3, axis=2)
    rgb[binary] = 0.42 * rgb[binary] + 0.58 * np.asarray(color, dtype=np.float32)
    return rgb


def _save_and_close(figure: plt.Figure, output_path: str | Path, dpi: int) -> Path:
    """Persist a figure and always release its Matplotlib resources."""
    target = Path(output_path)
    ensure_directory(target.parent)
    try:
        figure.savefig(target, dpi=dpi, bbox_inches="tight")
        return target
    finally:
        plt.close(figure)


def _format_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()


def plot_patient_panel(
    image: np.ndarray,
    gt_nodule_mask: np.ndarray,
    partition: SuperpixelPartition,
    explanation_masks: Mapping[str, np.ndarray],
    output_path: str | Path,
    image_id: str = "patient",
    dpi: int = 300,
) -> Path:
    """Save RX/GT/SLIC/four-explanation panel with deterministic resource cleanup."""
    required = OCCLUSION_METHODS
    missing = [method for method in required if method not in explanation_masks]
    if missing:
        raise ValueError(f"explanation_masks lacks required methods: {missing}")
    figure, axes = plt.subplots(1, 7, figsize=(23, 4), constrained_layout=True)
    try:
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        _format_axis(axes[0], "RX original")
        axes[1].imshow(_overlay(image, gt_nodule_mask, (0.0, 1.0, 0.0)))
        _format_axis(axes[1], "Ground truth\nnódulo")
        axes[2].imshow(partition.boundary_image(image))
        _format_axis(axes[2], f"SLIC (K={partition.num_superpixels})")
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
            _format_axis(axis, labels[method])
        figure.suptitle(f"Contrafactuales mínimos — {image_id}", fontsize=13)
        return _save_and_close(figure, output_path, dpi)
    except Exception:
        plt.close(figure)
        raise


def plot_occlusion_method_comparison(
    image: np.ndarray,
    partition: SuperpixelPartition,
    diagnostic_occlusion_mask: np.ndarray,
    output_path: str | Path,
    image_id: str,
    engine: OcclusionEngine | None = None,
    dpi: int = 300,
) -> Path:
    """Compare the four interventions on exactly the same diagnostically selected mask.

    This is an intervention-comparison figure, not a MaxSAT explanation: every
    method uses the same regions, making visible which artifact is induced by each
    fill policy. The mask must be a union of complete SLIC regions.
    """
    mask = np.asarray(diagnostic_occlusion_mask, dtype=bool)
    if mask.shape != partition.shape:
        raise ValueError("diagnostic_occlusion_mask must match partition shape")
    if engine is None:
        engine = OcclusionEngine(image, partition)
    selected_global = np.unique(partition.labels[mask & (partition.labels >= 0)])
    activation = np.ones(partition.num_superpixels, dtype=bool)
    activation[selected_global] = False
    figure, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    try:
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        _format_axis(axes[0], "Original")
        labels = {"zero": "Zero", "mean": "Mean", "gaussian_blur": "Gaussian blur", "inpaint": "Inpaint Telea"}
        for axis, method in zip(axes[1:], OCCLUSION_METHODS, strict=True):
            axis.imshow(engine.perturb(activation, method), cmap="gray", vmin=0, vmax=1)
            axis.contour(mask, levels=[0.5], colors="lime", linewidths=0.7)
            _format_axis(axis, labels[method])
        figure.suptitle(f"Comparativa de oclusión idéntica — {image_id}", fontsize=13)
        return _save_and_close(figure, output_path, dpi)
    except Exception:
        plt.close(figure)
        raise


def plot_ground_truth_construction(
    image: np.ndarray,
    lung_mask: np.ndarray,
    gt_nodule_mask: np.ndarray,
    output_path: str | Path,
    image_id: str,
    ground_truth_label: str = "Círculo reconstruido desde centro y diámetro",
    dpi: int = 300,
) -> Path:
    """Show source radiograph, lung anatomy and reconstructed nodule ground truth."""
    if np.asarray(lung_mask).shape != np.asarray(image).shape or np.asarray(gt_nodule_mask).shape != np.asarray(image).shape:
        raise ValueError("image, lung_mask and gt_nodule_mask must share a shape")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    try:
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        _format_axis(axes[0], "A. Radiografía original")
        axes[1].imshow(lung_mask, cmap="Blues", vmin=0, vmax=1)
        _format_axis(axes[1], "B. Máscara pulmonar")
        axes[2].imshow(_overlay(image, gt_nodule_mask, (0.0, 1.0, 0.0)))
        axes[2].contour(np.asarray(gt_nodule_mask, bool), levels=[0.5], colors="lime", linewidths=0.8)
        _format_axis(axes[2], f"C. {ground_truth_label}")
        figure.suptitle(f"Construcción del ground truth — {image_id}", fontsize=13)
        return _save_and_close(figure, output_path, dpi)
    except Exception:
        plt.close(figure)
        raise


def _superpixel_heatmap(partition: SuperpixelPartition, sensitivity: np.ndarray) -> np.ndarray:
    values = np.asarray(sensitivity, dtype=np.float64)
    if values.shape != (partition.num_superpixels,):
        raise ValueError("sensitivity must have one value per global superpixel")
    heatmap = np.full(partition.shape, np.nan, dtype=np.float64)
    valid = partition.labels >= 0
    heatmap[valid] = values[partition.labels[valid]]
    return heatmap


def plot_maxsat_explanation_process(
    image: np.ndarray,
    partition: SuperpixelPartition,
    gt_nodule_mask: np.ndarray,
    sensitivity: np.ndarray,
    explanation_mask: np.ndarray,
    output_path: str | Path,
    image_id: str,
    occlusion_method: str,
    iou_nodule: float,
    dice_nodule: float,
    nodule_coverage: float,
    dpi: int = 300,
) -> Path:
    """Save SLIC, singleton heatmap, MaxSAT mask and GT-vs-CF magnifier panel."""
    explanation = np.asarray(explanation_mask, dtype=bool)
    truth = np.asarray(gt_nodule_mask, dtype=bool)
    if explanation.shape != partition.shape or truth.shape != partition.shape:
        raise ValueError("explanation and ground truth must match partition shape")
    heatmap = _superpixel_heatmap(partition, sensitivity)
    finite = heatmap[np.isfinite(heatmap)]
    limit = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 1e-6)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    try:
        axes[0].imshow(partition.boundary_image(image))
        _format_axis(axes[0], f"1. SLIC (K={partition.num_superpixels})")
        rendered = axes[1].imshow(heatmap, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
        axes[1].contour(partition.labels >= 0, levels=[0.5], colors="black", linewidths=0.25)
        _format_axis(axes[1], "2. Sensibilidad singleton\nΔP = P original − P ocluida")
        colorbar = figure.colorbar(rendered, ax=axes[1], fraction=0.046, pad=0.04)
        colorbar.set_label("Caída de probabilidad")
        axes[2].imshow(_overlay(image, explanation, (1.0, 0.15, 0.15)))
        _format_axis(axes[2], f"3. MaxSAT final\n{occlusion_method}")
        combined = _overlay(image, explanation, (1.0, 0.15, 0.15))
        axes[3].imshow(combined)
        axes[3].contour(truth, levels=[0.5], colors="lime", linewidths=1.0)
        _format_axis(axes[3], f"4. GT verde vs CF rojo\nIoU={iou_nodule:.3f}, Dice={dice_nodule:.3f}, Cov={nodule_coverage:.3f}")
        figure.suptitle(f"Lupa del proceso Partial MaxSAT — {image_id}", fontsize=13)
        return _save_and_close(figure, output_path, dpi)
    except Exception:
        plt.close(figure)
        raise


def plot_aggregate_distributions(
    results: pd.DataFrame,
    metrics: Sequence[str],
    output_path: str | Path,
    kind: str = "box",
    dpi: int = 300,
) -> Path:
    """Save memory-safe box or violin distributions grouped by occlusion method."""
    if kind not in {"box", "violin"}:
        raise ValueError("kind must be box or violin")
    absent = {"occlusion_method", *metrics}.difference(results.columns)
    if absent:
        raise ValueError(f"results lacks required columns: {sorted(absent)}")
    columns = min(3, len(metrics))
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 4.0 * rows), squeeze=False)
    try:
        sns.set_theme(style="whitegrid", context="paper")
        order = [method for method in OCCLUSION_METHODS if method in set(results["occlusion_method"])]
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
        return _save_and_close(figure, output_path, dpi)
    except Exception:
        plt.close(figure)
        raise


def write_summary_tables(
    results: pd.DataFrame,
    metrics: Sequence[str],
    markdown_path: str | Path,
    latex_path: str | Path,
) -> pd.DataFrame:
    """Write mean/std/median/count aggregate tables in Markdown and LaTeX."""
    absent = {"occlusion_method", *metrics}.difference(results.columns)
    if absent:
        raise ValueError(f"results lacks required columns: {sorted(absent)}")
    summary = results.groupby("occlusion_method", sort=True)[list(metrics)].agg(["mean", "std", "median", "count"])
    markdown_target, latex_target = Path(markdown_path), Path(latex_path)
    ensure_directory(markdown_target.parent)
    ensure_directory(latex_target.parent)
    markdown_target.write_text(summary.to_markdown(floatfmt=".4f") + "\n", encoding="utf-8")
    latex_target.write_text(summary.to_latex(float_format="%.4f") + "\n", encoding="utf-8")
    return summary