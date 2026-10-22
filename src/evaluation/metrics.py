"""Validity, compactness, ground-truth and diversity metrics for counterfactuals."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.superpixels import SuperpixelPartition
from src.explainers.partial_maxsat import CounterfactualResult


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_mask_metrics(explanation_mask: np.ndarray, gt_nodule_mask: np.ndarray) -> dict[str, float]:
    """Compute pixel-level alignment with well-defined empty-mask behavior.

    If the ground-truth mask is empty, clinical localization metrics are undefined
    (`NaN`), rather than being reported as a misleading perfect score. IoU/Dice are
    one only when both masks are empty, following the set-equality convention.
    """
    explanation = np.asarray(explanation_mask, dtype=bool)
    truth = np.asarray(gt_nodule_mask, dtype=bool)
    if explanation.shape != truth.shape:
        raise ValueError("explanation and ground-truth masks must share a shape")
    intersection = int(np.logical_and(explanation, truth).sum())
    explanation_area = int(explanation.sum())
    truth_area = int(truth.sum())
    union = int(np.logical_or(explanation, truth).sum())
    return {
        "hit_rate": float(intersection > 0) if truth_area else float("nan"),
        "iou_nodule": 1.0 if union == 0 else float(intersection / union),
        "dice_nodule": 1.0
        if explanation_area + truth_area == 0
        else float(2 * intersection / (explanation_area + truth_area)),
        "nodule_coverage": _safe_ratio(intersection, truth_area),
        "explanation_precision": _safe_ratio(intersection, explanation_area),
        "intersection_pixels": float(intersection),
        "gt_nodule_pixels": float(truth_area),
        "explanation_pixels": float(explanation_area),
    }


def superpixel_gt_correlation(sensitivity: np.ndarray, gt_overlap: np.ndarray) -> float:
    """Return Spearman rank correlation between model sensitivity and GT overlap."""
    importance = np.asarray(sensitivity, dtype=np.float64)
    overlap = np.asarray(gt_overlap, dtype=np.float64)
    if importance.shape != overlap.shape:
        raise ValueError("sensitivity and gt_overlap must share a shape")
    finite = np.isfinite(importance) & np.isfinite(overlap)
    if finite.sum() < 2:
        return float("nan")
    left, right = importance[finite], overlap[finite]
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    correlation = spearmanr(left, right).statistic
    return float(correlation) if np.isfinite(correlation) else float("nan")


def jaccard_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Jaccard distance between two binary activation or removal vectors."""
    a, b = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Jaccard inputs must have matching shapes")
    union = int(np.logical_or(a, b).sum())
    return 0.0 if union == 0 else float(1.0 - np.logical_and(a, b).sum() / union)


def mean_diversity_jaccard(results: Sequence[CounterfactualResult]) -> float:
    """Average pairwise Jaccard distance of the sets of occluded superpixels."""
    valid = [result for result in results if result.valid]
    if len(valid) < 2:
        return float("nan")
    distances = [
        jaccard_distance(~valid[left].activation, ~valid[right].activation)
        for left in range(len(valid))
        for right in range(left + 1, len(valid))
    ]
    return float(np.mean(distances))


def evaluate_counterfactual(
    result: CounterfactualResult,
    partition: SuperpixelPartition,
    gt_nodule_mask: np.ndarray,
    image_id: str | None = None,
    solution_index: int = 0,
    diversity_jaccard: float = float("nan"),
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create one flat, CSV-ready record with every primary TFM metric."""
    if result.activation.shape != (partition.num_superpixels,):
        raise ValueError("result activation and partition do not agree")
    off_mask = partition.mask_from_activation(result.activation, off=True)
    gt = np.asarray(gt_nodule_mask, dtype=bool)
    if gt.shape != partition.shape:
        raise ValueError("gt_nodule_mask and partition must share a shape")
    pixel_metrics = binary_mask_metrics(off_mask, gt)
    total_area = off_mask.size
    lung_area = int(partition.anatomy_mask.sum())
    occluded_area = int(off_mask.sum())
    absolute_drop = result.original_probability - result.counterfactual_probability
    record: dict[str, object] = {
        "image_id": image_id,
        "solution_index": int(solution_index),
        "occlusion_method": result.occlusion_method,
        "weight_mode": result.weight_mode,
        "status": result.status,
        "validity": int(result.valid and result.counterfactual_probability < result.threshold),
        "threshold": float(result.threshold),
        "orig_prob": float(result.original_probability),
        "cf_prob": float(result.counterfactual_probability),
        "prob_drop": float(absolute_drop),
        "relative_prob_drop": _safe_ratio(absolute_drop, result.original_probability),
        "num_superpixels": int(partition.num_superpixels),
        "num_occluded_superpixels": int((~result.activation).sum()),
        "sparsity_ratio": float((~result.activation).mean()),
        "occluded_area_ratio": _safe_ratio(occluded_area, total_area),
        "occluded_lung_area_ratio": _safe_ratio(
            int(np.logical_and(off_mask, partition.anatomy_mask).sum()), lung_area
        ),
        "superpixel_gt_correlation": superpixel_gt_correlation(
            result.sensitivity, partition.gt_overlap_of_superpixel
        ),
        "solver_time_sec": float(result.solver_time_sec),
        "total_time_sec": float(result.total_time_sec),
        "oracle_calls": int(result.oracle_calls),
        "num_hard_clauses": int(result.num_hard_clauses),
        "num_soft_clauses": int(result.num_soft_clauses),
        "objective_cost": result.objective_cost,
        "diversity_jaccard": float(diversity_jaccard),
        **pixel_metrics,
    }
    if extra_fields:
        overlap = set(record).intersection(extra_fields)
        if overlap:
            raise ValueError(f"extra_fields would overwrite standard metrics: {sorted(overlap)}")
        record.update(extra_fields)
    return record


def records_to_frame(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Create a deterministic DataFrame from metric records."""
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return frame
    ordering = [column for column in ("image_id", "occlusion_method", "solution_index") if column in frame]
    return frame.sort_values(ordering, kind="stable").reset_index(drop=True) if ordering else frame
