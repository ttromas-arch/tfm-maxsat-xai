"""Classifier-oracle adapters used by the counterfactual solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from src.common import as_grayscale_float, select_torch_device


class NoduleProbabilityOracle(Protocol):
    """Minimal classifier interface accepted by the explainer."""

    def predict_probability(self, image: np.ndarray) -> float:
        """Return the probability of the Nodule pathology for one image."""

    def predict_probabilities(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return Nodule probabilities for an ordered batch of images."""


class TorchXRayVisionNoduleClassifier:
    """TorchXRayVision DenseNet adapter exposing the ``Nodule`` probability.

    Input images are normalized `[0, 1]` grayscale arrays. They are converted to the
    range `[-1024, 1024]` expected by the public TorchXRayVision model interface.
    """

    def __init__(
        self,
        device: str = "auto",
        weights: str = "densenet121-res224-all",
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            import torch
            import torchxrayvision as xrv
        except ImportError as error:
            raise RuntimeError(
                "TorchXRayVision requires torch and torchxrayvision; install requirements.txt"
            ) from error
        self._torch = torch
        self.device = select_torch_device(device)
        self.batch_size = batch_size
        try:
            self.model = xrv.models.DenseNet(weights=weights)
        except Exception as error:
            raise RuntimeError(
                f"unable to initialise TorchXRayVision weights {weights!r}; "
                "ensure network access or a populated XRV cache"
            ) from error
        self.model = self.model.to(self.device).eval()
        pathologies = list(getattr(self.model, "pathologies", []))
        try:
            self.nodule_index = pathologies.index("Nodule")
        except ValueError as error:
            raise RuntimeError(f"loaded XRV model does not expose Nodule; labels are {pathologies}") from error

    @staticmethod
    def to_xrv_range(image: np.ndarray) -> np.ndarray:
        """Convert normalized grayscale pixels to XRV Hounsfield-like range."""
        return as_grayscale_float(image) * 2048.0 - 1024.0

    def predict_probabilities(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Run deterministic inference in batches and return finite probabilities."""
        if not images:
            return np.empty(0, dtype=np.float64)
        converted = [self.to_xrv_range(image) for image in images]
        shape = converted[0].shape
        if any(image.shape != shape for image in converted):
            raise ValueError("all images in an inference batch must share a spatial shape")
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(converted), self.batch_size):
                chunk = np.stack(converted[start : start + self.batch_size]).astype(np.float32)
                tensor = self._torch.from_numpy(chunk[:, None, :, :]).to(self.device)
                scores = self.model(tensor).detach().float().cpu().numpy()[:, self.nodule_index]
                if np.any(scores < 0.0) or np.any(scores > 1.0):
                    scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -80.0, 80.0)))
                outputs.append(scores)
        result = np.concatenate(outputs).astype(np.float64)
        if not np.isfinite(result).all():
            raise RuntimeError("TorchXRayVision returned non-finite Nodule probabilities")
        return np.clip(result, 0.0, 1.0)

    def predict_probability(self, image: np.ndarray) -> float:
        """Return one Nodule probability."""
        return float(self.predict_probabilities([image])[0])


@dataclass
class SyntheticNoduleClassifier:
    """Deterministic, spatially weighted test oracle; not for clinical experiments.

    This surrogate is useful to test MaxSAT behavior when model weights or JSRT are
    unavailable. The score is a sigmoid of mean brightness in a configurable region
    of interest, so zeroing a bright synthetic lesion produces a real decision flip.
    """

    roi_mask: np.ndarray
    center: float = 0.55
    slope: float = 12.0

    def __post_init__(self) -> None:
        self.roi_mask = np.asarray(self.roi_mask, dtype=bool)
        if self.roi_mask.ndim != 2 or not self.roi_mask.any():
            raise ValueError("roi_mask must be a nonempty 2-D boolean array")
        if self.slope <= 0:
            raise ValueError("slope must be positive")

    def predict_probability(self, image: np.ndarray) -> float:
        """Score a normalized image using its ROI mean intensity."""
        normalized = as_grayscale_float(image)
        if normalized.shape != self.roi_mask.shape:
            raise ValueError("image shape does not match synthetic classifier ROI")
        logit = self.slope * (float(normalized[self.roi_mask].mean()) - self.center)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -80.0, 80.0))))

    def predict_probabilities(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Score an image sequence in order."""
        return np.asarray([self.predict_probability(image) for image in images], dtype=np.float64)