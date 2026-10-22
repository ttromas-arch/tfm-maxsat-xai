"""Run the complete JSRT Partial-MaxSAT counterfactual benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_directory, seed_everything
from src.data.jsrt import JSRTDataset, download_nodule_coordinates
from src.data.superpixels import build_slic_partition
from src.evaluation.metrics import evaluate_counterfactual, mean_diversity_jaccard
from src.explainers.partial_maxsat import PartialMaxSATCounterfactualExplainer
from src.models.classifier import SyntheticNoduleClassifier, TorchXRayVisionNoduleClassifier
from src.occlusion.engine import OCCLUSION_METHODS, OcclusionEngine
from src.visualization.plots import plot_aggregate_distributions, plot_patient_panel, write_summary_tables


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully serializable settings for one reproducible benchmark execution."""

    jsrt_root: str
    output_dir: str
    nodule_csv: str | None
    nodule_url: str | None
    num_superpixels: int
    slic_compactness: float
    threshold: float
    threshold_mode: str
    weight_mode: str
    max_oracle_calls: int
    exploration_random_samples: int
    diverse_solutions: int
    repetition_penalty: int | None
    classifier: str
    device: str
    batch_size: int
    max_images: int | None
    seed: int


def parse_args() -> argparse.Namespace:
    """Parse all benchmark parameters from a local or Kaggle command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsrt-root", required=True, help="JSRT root containing images and lung_masks")
    parser.add_argument("--output-dir", default="outputs/jsrt_run")
    parser.add_argument("--nodule-csv", default=None)
    parser.add_argument("--nodule-url", default=None, help="Optional official coordinate CSV URL")
    parser.add_argument("--num-superpixels", type=int, default=50)
    parser.add_argument("--slic-compactness", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--threshold-mode", choices=("fixed", "relative"), default="fixed")
    parser.add_argument("--weight-mode", choices=("cardinality", "area"), default="cardinality")
    parser.add_argument("--max-oracle-calls", type=int, default=250)
    parser.add_argument("--exploration-random-samples", type=int, default=0)
    parser.add_argument("--diverse-solutions", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=int, default=None)
    parser.add_argument("--classifier", choices=("torchxrayvision", "synthetic"), default="torchxrayvision")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true", help="Discard previous incremental CSV")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def _make_classifier(args: argparse.Namespace):
    if args.classifier == "torchxrayvision":
        return TorchXRayVisionNoduleClassifier(device=args.device, batch_size=args.batch_size)
    return None


def _append_incremental(record: dict[str, object], output_csv: Path) -> None:
    frame = pd.DataFrame([record])
    frame.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)


def _completed_pairs(marker_path: Path) -> set[tuple[str, str]]:
    """Read transaction markers written only after an entire method is persisted."""
    if not marker_path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    for line_number, line in enumerate(marker_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            marker = json.loads(line)
            completed.add((str(marker["image_id"]), str(marker["occlusion_method"])))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid completion marker at {marker_path}:{line_number}") from error
    return completed


def _mark_completed(marker_path: Path, image_id: str, method: str) -> None:
    """Atomically append an idempotent end-of-method completion marker."""
    with marker_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"image_id": image_id, "occlusion_method": method}) + "\n")


def _write_failure(path: Path, image_id: str, method: str, error: Exception) -> None:
    payload = {"image_id": image_id, "occlusion_method": method, "error": repr(error)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> Path:
    """Execute benchmark and return its result CSV path."""
    seed_everything(args.seed)
    output_dir = ensure_directory(args.output_dir)
    figures_dir = ensure_directory(output_dir / "figures")
    results_csv = output_dir / "results.csv"
    failures_path = output_dir / "failures.jsonl"
    completed_path = output_dir / "completed_methods.jsonl"
    jsrt_root = Path(args.jsrt_root)
    nodule_csv = Path(args.nodule_csv) if args.nodule_csv else None
    if args.nodule_url:
        nodule_csv = download_nodule_coordinates(args.nodule_url, jsrt_root / "nodules_official.csv")
    config = ExperimentConfig(
        jsrt_root=str(jsrt_root),
        output_dir=str(output_dir),
        nodule_csv=str(nodule_csv) if nodule_csv else None,
        nodule_url=args.nodule_url,
        num_superpixels=args.num_superpixels,
        slic_compactness=args.slic_compactness,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        weight_mode=args.weight_mode,
        max_oracle_calls=args.max_oracle_calls,
        exploration_random_samples=args.exploration_random_samples,
        diverse_solutions=args.diverse_solutions,
        repetition_penalty=args.repetition_penalty,
        classifier=args.classifier,
        device=args.device,
        batch_size=args.batch_size,
        max_images=args.max_images,
        seed=args.seed,
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    dataset = JSRTDataset(jsrt_root, nodule_csv=nodule_csv)
    identifiers = dataset.image_ids(nodules_only=True)
    if not identifiers:
        raise RuntimeError("no image with both lung mask and nodule coordinate was found")
    if args.max_images is not None:
        identifiers = identifiers[: args.max_images]
    completed = set() if args.overwrite else _completed_pairs(completed_path)
    if args.overwrite and results_csv.exists():
        results_csv.unlink()
    if args.overwrite and completed_path.exists():
        completed_path.unlink()
    classifier = _make_classifier(args)
    for position, image_id in enumerate(identifiers, start=1):
        logging.info("Processing %s (%d/%d)", image_id, position, len(identifiers))
        try:
            sample = dataset.load(image_id)
            partition = build_slic_partition(
                sample.image,
                sample.lung_mask,
                sample.gt_nodule_mask,
                num_segments=args.num_superpixels,
                compactness=args.slic_compactness,
                seed=args.seed,
            )
            engine = OcclusionEngine(sample.image, partition)
        except Exception as error:
            for method in OCCLUSION_METHODS:
                _write_failure(failures_path, image_id, method, error)
            logging.exception("Unable to prepare image %s", image_id)
            continue
        panel_masks: dict[str, np.ndarray] = {}
        for method_index, method in enumerate(OCCLUSION_METHODS):
            if (image_id, method) in completed:
                logging.info("Skipping completed pair %s/%s", image_id, method)
                continue
            try:
                effective_classifier = (
                    classifier
                    if classifier is not None
                    else SyntheticNoduleClassifier(sample.gt_nodule_mask, center=0.42, slope=12.0)
                )
                explainer = PartialMaxSATCounterfactualExplainer(
                    partition=partition,
                    occlusion_engine=engine,
                    oracle=effective_classifier,
                    occlusion_method=method,
                    threshold=args.threshold,
                    threshold_mode=args.threshold_mode,
                    weight_mode=args.weight_mode,
                    max_oracle_calls=args.max_oracle_calls,
                    exploration_random_samples=args.exploration_random_samples,
                    random_seed=args.seed + method_index,
                )
                solutions = explainer.diverse_solutions(
                    maximum_solutions=args.diverse_solutions,
                    repetition_penalty=args.repetition_penalty,
                )
                if not solutions:
                    # Persist an invalid terminal trace instead of silently omitting it.
                    terminal = explainer.explain()
                    solutions = [terminal]
                diversity = mean_diversity_jaccard(solutions)
                for solution_index, solution in enumerate(solutions):
                    record = evaluate_counterfactual(
                        solution,
                        partition,
                        sample.gt_nodule_mask,
                        image_id=image_id,
                        solution_index=solution_index,
                        diversity_jaccard=diversity,
                        extra_fields={"source_path": str(sample.source_path)},
                    )
                    _append_incremental(record, results_csv)
                panel_masks[method] = partition.mask_from_activation(solutions[0].activation, off=True)
                _mark_completed(completed_path, image_id, method)
                completed.add((image_id, method))
            except Exception as error:
                _write_failure(failures_path, image_id, method, error)
                logging.exception("Failure for %s/%s", image_id, method)
        if not args.skip_figures and len(panel_masks) == len(OCCLUSION_METHODS):
            plot_patient_panel(
                sample.image,
                sample.gt_nodule_mask,
                partition,
                panel_masks,
                figures_dir / f"{image_id}_panel.png",
                image_id=image_id,
            )
    if not results_csv.exists():
        raise RuntimeError("the benchmark produced no result records; inspect failures.jsonl")
    results = pd.read_csv(results_csv)
    summary_metrics = [
        "validity",
        "num_occluded_superpixels",
        "occluded_area_ratio",
        "iou_nodule",
        "dice_nodule",
        "nodule_coverage",
        "explanation_precision",
        "solver_time_sec",
        "oracle_calls",
    ]
    write_summary_tables(results, summary_metrics, output_dir / "summary.md", output_dir / "summary.tex")
    plot_aggregate_distributions(results, summary_metrics, output_dir / "boxplots.png", kind="box")
    plot_aggregate_distributions(results, summary_metrics, output_dir / "violins.png", kind="violin")
    logging.info("Completed benchmark: %s rows in %s", len(results), results_csv)
    return results_csv


def main() -> None:
    """Configure logging and launch the command-line benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
