"""Comparable superpixel occlusion mechanisms for grayscale radiographs."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from src.common import as_grayscale_float
from src.data.superpixels import SuperpixelPartition

OcclusionMethod = Literal["zero", "mean", "gaussian_blur", "inpaint"]
OCCLUSION_METHODS: tuple[OcclusionMethod, ...] = ("zero", "mean", "gaussian_blur", "inpaint")


class OcclusionEngine:
    """Apply a fixed occlusion policy to any binary superpixel activation.

    The source image, mean value and strong Gaussian blur are cached once. The
    activation-specific portion is boolean indexing only, except for inpainting,
    whose Telea interpolation inherently depends on the union mask.
    """

    def __init__(
        self,
        image: np.ndarray,
        partition: SuperpixelPartition,
        mean_on_anatomy: bool = True,
        blur_sigma: float = 12.0,
        inpaint_radius: float = 5.0,
    ) -> None:
        self.image = as_grayscale_float(image)
        self.partition = partition
        if self.image.shape != partition.shape:
            raise ValueError("image and partition must have matching shapes")
        if blur_sigma <= 0 or inpaint_radius <= 0:
            raise ValueError("blur_sigma and inpaint_radius must be positive")
        reference_mask = partition.anatomy_mask if mean_on_anatomy else np.ones(self.image.shape, bool)
        self.fill_mean = float(self.image[reference_mask].mean())
        self.blur_sigma = float(blur_sigma)
        self.inpaint_radius = float(inpaint_radius)
        self.blurred_image = cv2.GaussianBlur(
            self.image, ksize=(0, 0), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma
        ).astype(np.float32)

    def off_mask(self, activation: np.ndarray) -> np.ndarray:
        """Return the anatomical pixels belonging to disabled variables."""
        return self.partition.mask_from_activation(activation, off=True)

    def perturb(self, activation: np.ndarray, method: OcclusionMethod) -> np.ndarray:
        """Generate one perturbed image under the selected intervention method."""
        if method not in OCCLUSION_METHODS:
            raise ValueError(f"unknown occlusion method {method!r}; use {OCCLUSION_METHODS}")
        mask = self.off_mask(activation)
        if not mask.any():
            return self.image.copy()
        result = self.image.copy()
        if method == "zero":
            result[mask] = 0.0
        elif method == "mean":
            result[mask] = self.fill_mean
        elif method == "gaussian_blur":
            result[mask] = self.blurred_image[mask]
        else:
            source_u8 = np.rint(self.image * 255.0).astype(np.uint8)
            mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
            inpainted = cv2.inpaint(source_u8, mask_u8, self.inpaint_radius, cv2.INPAINT_TELEA)
            result = inpainted.astype(np.float32) / 255.0
        return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

    def perturb_many(self, activations: np.ndarray, method: OcclusionMethod) -> np.ndarray:
        """Generate a stack of perturbations, validating all activation vectors."""
        matrix = np.asarray(activations)
        if matrix.ndim != 2 or matrix.shape[1] != self.partition.num_superpixels:
            raise ValueError(
                "activations must have shape (batch_size, partition.num_superpixels)"
            )
        return np.stack([self.perturb(vector, method) for vector in matrix], axis=0)
