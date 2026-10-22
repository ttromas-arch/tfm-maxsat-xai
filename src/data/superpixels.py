"""SLIC partitions and ground-truth overlap statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.segmentation import mark_boundaries, slic

from src.common import as_grayscale_float


@dataclass(frozen=True)
class SuperpixelPartition:
    """A compact SLIC labeling restricted to a valid anatomical mask.

    Pixels outside ``anatomy_mask`` carry label ``-1`` and are not decision
    variables. Labels inside the mask are contiguous in ``[0, num_superpixels)``.
    """

    labels: np.ndarray
    anatomy_mask: np.ndarray
    areas: np.ndarray
    gt_intersections: np.ndarray
    gt_overlap_of_superpixel: np.ndarray
    gt_coverage_by_superpixel: np.ndarray

    @property
    def num_superpixels(self) -> int:
        """Number of active propositional variables."""
        return int(self.areas.size)

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial image shape."""
        return int(self.labels.shape[0]), int(self.labels.shape[1])

    def validate_activation(self, activation: np.ndarray) -> np.ndarray:
        """Return a validated boolean activation vector."""
        vector = np.asarray(activation)
        if vector.shape != (self.num_superpixels,):
            raise ValueError(
                f"activation must have {self.num_superpixels} values, got {vector.shape}"
            )
        if not np.isin(vector, [0, 1, False, True]).all():
            raise ValueError("activation must be binary")
        return vector.astype(bool, copy=False)

    def mask_from_activation(self, activation: np.ndarray, off: bool = True) -> np.ndarray:
        """Return the union mask of switched-off (or switched-on) superpixels."""
        vector = self.validate_activation(activation)
        selected = ~vector if off else vector
        lookup = np.zeros(self.num_superpixels, dtype=bool)
        lookup[selected] = True
        valid = self.labels >= 0
        result = np.zeros(self.shape, dtype=bool)
        result[valid] = lookup[self.labels[valid]]
        return result

    def boundary_image(self, image: np.ndarray) -> np.ndarray:
        """Return an RGB rendering of SLIC boundaries for a publication panel."""
        gray = as_grayscale_float(image)
        displayed_labels = self.labels.copy()
        displayed_labels[~self.anatomy_mask] = self.num_superpixels
        return mark_boundaries(gray, displayed_labels, color=(1.0, 0.8, 0.0), mode="thick")


def build_slic_partition(
    image: np.ndarray,
    anatomy_mask: np.ndarray | None,
    gt_nodule_mask: np.ndarray | None,
    num_segments: int = 50,
    compactness: float = 0.15,
    sigma: float = 0.0,
    seed: int = 0,
) -> SuperpixelPartition:
    """Partition a radiograph with SLIC and compute per-region GT overlap.

    ``skimage`` SLIC is deterministic for the same inputs; ``seed`` is accepted
    and consumed only to document experiment provenance and future compatibility.
    The actual number of labels may be below the requested number for small masks.
    """
    del seed
    gray = as_grayscale_float(image)
    if num_segments < 1:
        raise ValueError("num_segments must be positive")
    if compactness <= 0:
        raise ValueError("compactness must be positive")
    if anatomy_mask is None:
        anatomy = np.ones(gray.shape, dtype=bool)
    else:
        anatomy = np.asarray(anatomy_mask, dtype=bool)
        if anatomy.shape != gray.shape:
            raise ValueError("anatomy_mask and image must have the same shape")
    if not anatomy.any():
        raise ValueError("anatomy_mask contains no active pixel")
    raw_labels = slic(
        gray,
        n_segments=num_segments,
        compactness=compactness,
        sigma=sigma,
        mask=anatomy,
        start_label=0,
        channel_axis=None,
        enforce_connectivity=True,
    )
    labels = np.full(gray.shape, -1, dtype=np.int32)
    unique = np.unique(raw_labels[anatomy])
    if unique.size == 0:
        raise RuntimeError("SLIC returned no labels inside the anatomical mask")
    for compact_label, original_label in enumerate(unique):
        labels[(raw_labels == original_label) & anatomy] = compact_label
    number = int(unique.size)
    areas = np.bincount(labels[labels >= 0], minlength=number).astype(np.int64)
    if gt_nodule_mask is None:
        ground_truth = np.zeros(gray.shape, dtype=bool)
    else:
        ground_truth = np.asarray(gt_nodule_mask, dtype=bool)
        if ground_truth.shape != gray.shape:
            raise ValueError("gt_nodule_mask and image must have the same shape")
    intersections = np.bincount(labels[(labels >= 0) & ground_truth], minlength=number).astype(
        np.int64
    )
    gt_area = int(ground_truth.sum())
    overlap = np.divide(
        intersections,
        areas,
        out=np.zeros(number, dtype=np.float64),
        where=areas > 0,
    )
    coverage = (
        intersections.astype(np.float64) / gt_area
        if gt_area
        else np.zeros(number, dtype=np.float64)
    )
    return SuperpixelPartition(
        labels=labels,
        anatomy_mask=anatomy,
        areas=areas,
        gt_intersections=intersections,
        gt_overlap_of_superpixel=overlap,
        gt_coverage_by_superpixel=coverage,
    )
