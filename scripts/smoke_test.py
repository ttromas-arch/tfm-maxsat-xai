"""Fast end-to-end validation without requiring a downloaded JSRT dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_directory, seed_everything
from src.data.superpixels import build_slic_partition
from src.evaluation.metrics import evaluate_counterfactual, mean_diversity_jaccard, records_to_frame
from src.explainers.partial_maxsat import PartialMaxSATCounterfactualExplainer
from src.models.classifier import SyntheticNoduleClassifier, TorchXRayVisionNoduleClassifier
from src.occlusion.engine import OCCLUSION_METHODS, OcclusionEngine
from src.visualization.plots import plot_patient_panel


def synthetic_radiograph(size: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic lung-shaped image and an internal bright nodule."""
    if size < 48:
        raise ValueError("synthetic image size must be at least 48 pixels")
    y, x = np.ogrid[:size, :size]
    left_lung = ((x - size * 0.34) / (size * 0.22)) ** 2 + ((y - size * 0.53) / (size * 0.36)) ** 2 <= 1
    right_lung = ((x - size * 0.66) / (size * 0.22)) ** 2 + ((y - size * 0.53) / (size * 0.36)) ** 2 <= 1
    lung = left_lung | right_lung
    lesion = (x - size * 0.65) ** 2 + (y - size * 0.50) ** 2 <= (size * 0.075) ** 2
    image = np.full((size, size), 0.08, dtype=np.float32)
    image[lung] = 0.30
    image[lesion] = 0.98
    # Add a fixed, low-amplitude anatomical texture outside the lesion.
    x_full = np.broadcast_to(x, (size, size))
    image[lung & ~lesion] += 0.025 * np.sin(x_full[lung & ~lesion] / 4.0)
    return np.clip(image, 0.0, 1.0), lung, lesion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/smoke_test")
    parser.add_argument("--exercise-torchxrayvision", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def run(output_dir: str | Path, seed: int = 7) -> dict[str, object]:
    """Execute every core method and return a JSON-serializable validation report."""
    seed_everything(seed)
    destination = ensure_directory(output_dir)
    image, lung_mask, gt_mask = synthetic_radiograph()
    partition = build_slic_partition(
        image, lung_mask, gt_mask, num_segments=24, compactness=0.12, seed=seed
    )
    if partition.num_superpixels < 4:
        raise AssertionError("SLIC synthetic partition unexpectedly has fewer than four regions")
    engine = OcclusionEngine(image, partition, blur_sigma=9.0, inpaint_radius=3.0)
    all_on = np.ones(partition.num_superpixels, dtype=bool)
    all_off = np.zeros(partition.num_superpixels, dtype=bool)
    perturbation_stats: dict[str, float] = {}
    for method in OCCLUSION_METHODS:
        perturbed = engine.perturb(all_off, method)
        if perturbed.shape != image.shape or not np.isfinite(perturbed).all():
            raise AssertionError(f"{method} did not produce a finite image of the input shape")
        perturbation_stats[method] = float(np.mean(np.abs(perturbed - image)))
    if perturbation_stats["zero"] <= 0.01:
        raise AssertionError("zero perturbation had no material effect")
    classifier = SyntheticNoduleClassifier(gt_mask, center=0.55, slope=13.0)
    if classifier.predict_probability(image) <= 0.60:
        raise AssertionError("synthetic original should be nodule-positive")
    records: list[dict[str, object]] = []
    masks: dict[str, np.ndarray] = {}
    statuses: dict[str, str] = {}
    for index, method in enumerate(OCCLUSION_METHODS):
        explainer = PartialMaxSATCounterfactualExplainer(
            partition=partition,
            occlusion_engine=engine,
            oracle=classifier,
            occlusion_method=method,
            threshold=0.60,
            weight_mode="cardinality",
            max_oracle_calls=90,
            exploration_individual=True,
            exploration_random_samples=4,
            random_seed=seed + index,
        )
        solutions = explainer.diverse_solutions(maximum_solutions=2, repetition_penalty=2)
        if not solutions:
            raise AssertionError(f"{method} yielded no counterfactual result")
        result = solutions[0]
        if not result.valid:
            raise AssertionError(f"{method} failed to find a valid synthetic counterfactual: {result.status}")
        if result.counterfactual_probability >= result.threshold:
            raise AssertionError(f"{method} did not cross its threshold")
        masks[method] = partition.mask_from_activation(result.activation)
        statuses[method] = result.status
        diversity = mean_diversity_jaccard(solutions)
        records.extend(
            evaluate_counterfactual(
                solution,
                partition,
                gt_mask,
                image_id="synthetic",
                solution_index=solution_index,
                diversity_jaccard=diversity,
            )
            for solution_index, solution in enumerate(solutions)
        )
    frame = records_to_frame(records)
    required = {"validity", "iou_nodule", "solver_time_sec", "oracle_calls", "diversity_jaccard"}
    if not required.issubset(frame.columns) or not (frame["validity"] == 1).all():
        raise AssertionError("evaluation output lacks expected valid metrics")
    csv_path = destination / "smoke_results.csv"
    frame.to_csv(csv_path, index=False)
    panel_path = plot_patient_panel(image, gt_mask, partition, masks, destination / "smoke_panel.png", "synthetic")
    report = {
        "status": "passed",
        "num_superpixels": partition.num_superpixels,
        "methods": statuses,
        "perturbation_mean_absolute_difference": perturbation_stats,
        "result_csv": str(csv_path),
        "panel": str(panel_path),
    }
    (destination / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run(args.output_dir, args.seed)
    if args.exercise_torchxrayvision:
        # Validate initialization and preprocessor separately: a synthetic image is
        # not a clinically interpretable inference target.
        image, _, _ = synthetic_radiograph(size=224)
        model = TorchXRayVisionNoduleClassifier()
        report["torchxrayvision_probability"] = model.predict_probability(image)
        Path(args.output_dir, "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
