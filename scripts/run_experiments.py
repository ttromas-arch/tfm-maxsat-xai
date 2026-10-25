"""Run the complete JSRT Partial-MaxSAT counterfactual benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_directory, seed_everything
from src.data.jsrt import JSRTDataset, download_nodule_coordinates
from src.data.superpixels import build_slic_partition
from src.evaluation.metrics import evaluate_counterfactual, mean_diversity_jaccard
from src.explainers.partial_maxsat import PartialMaxSATCounterfactualExplainer, counterfactual_trace
from src.models.classifier import SyntheticNoduleClassifier, TorchXRayVisionNoduleClassifier
from src.occlusion.engine import OCCLUSION_METHODS, OcclusionEngine
from src.visualization.plots import (
    plot_aggregate_distributions,
    plot_ground_truth_construction,
    plot_maxsat_explanation_process,
    plot_occlusion_method_comparison,
    plot_patient_panel,
    write_summary_tables,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully serializable settings for one reproducible benchmark execution."""

    jsrt_root: str
    output_dir: str
    nodule_csv: str | None
    nodule_url: str | None
    image_dir: str | None
    lung_mask_dir: str | None
    ground_truth_representation: str
    provenance_manifest: str | None
    num_superpixels: int
    slic_compactness: float
    threshold: float
    threshold_mode: str
    weight_mode: str
    max_oracle_calls: int
    exploration_random_samples: int
    diverse_solutions: int
    repetition_penalty: int | None
    minimum_new_occluded_regions: int
    cegar_cut_mode: str
    seed_guidance_weight: float
    anti_sensitivity_penalty: float
    low_sensitivity_penalty: float
    sensitivity_normalization: str
    minimum_lung_overlap: float
    hybrid_exact_budget_fraction: float
    oracle_batch_size: int
    per_sample_timeout_sec: float | None
    save_successful_examples: int
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
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional image directory relative to --jsrt-root or absolute (e.g. cxr)",
    )
    parser.add_argument(
        "--lung-mask-dir",
        default=None,
        help="Optional lung-mask directory relative to --jsrt-root or absolute (e.g. masks)",
    )
    parser.add_argument(
        "--ground-truth-representation",
        default="circle_from_coordinate_diameter",
        help="Auditable GT label recorded in each metric row",
    )
    parser.add_argument(
        "--provenance-manifest",
        default=None,
        help="Optional JSON manifest copied into the experiment output",
    )
    parser.add_argument("--num-superpixels", type=int, default=50)
    parser.add_argument("--slic-compactness", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--threshold-mode", choices=("fixed", "relative"), default="fixed")
    parser.add_argument("--weight-mode", choices=("cardinality", "area"), default="cardinality")
    parser.add_argument("--max-oracle-calls", type=int, default=250)
    parser.add_argument("--exploration-random-samples", type=int, default=0)
    parser.add_argument("--diverse-solutions", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=int, default=None)
    parser.add_argument("--minimum-new-occluded-regions", type=int, default=1)
    parser.add_argument("--cegar-cut-mode", choices=("exact", "expansion", "hybrid"), default="exact")
    parser.add_argument("--seed-guidance-weight", type=float, default=1.0)
    parser.add_argument("--anti-sensitivity-penalty", type=float, default=4.0)
    parser.add_argument("--low-sensitivity-penalty", type=float, default=1.0)
    parser.add_argument("--sensitivity-normalization", choices=("rank", "minmax"), default="rank")
    parser.add_argument("--minimum-lung-overlap", type=float, default=0.0)
    parser.add_argument("--hybrid-exact-budget-fraction", type=float, default=0.5)
    parser.add_argument(
        "--oracle-batch-size",
        type=int,
        default=16,
        help="Number of CEGAR candidates evaluated per batched network inference",
    )
    parser.add_argument(
        "--per-sample-timeout-sec",
        type=float,
        default=None,
        help="Wall-clock deadline per image and occlusion method; None disables it",
    )
    parser.add_argument(
        "--save-successful-examples",
        type=int,
        default=0,
        metavar="N",
        help="Generate academic figures for the first N distinct valid cases with IoU > 0.10",
    )
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


def _append_trace(trace: dict[str, object], trace_path: Path) -> None:
    """Persist a JSONL trace atomically at the solution granularity."""
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")


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


def _load_traces(trace_path: Path) -> list[dict[str, object]]:
    """Load strict JSONL traces, reporting malformed line numbers for reproducibility."""
    traces: list[dict[str, object]] = []
    if not trace_path.exists():
        return traces
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed explanation trace at {trace_path}:{line_number}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"trace must be a JSON object at {trace_path}:{line_number}")
        traces.append(payload)
    return traces


def _save_successful_examples(
    dataset: JSRTDataset,
    results: pd.DataFrame,
    traces: list[dict[str, object]],
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    """Rebuild selected valid cases from traces and save the three requested figures."""
    if args.save_successful_examples <= 0:
        return 0
    if not traces:
        logging.warning("No explanation traces available; successful examples cannot be regenerated")
        return 0
    required = {"image_id", "occlusion_method", "solution_index", "validity", "iou_nodule", "cf_prob"}
    absent = required.difference(results.columns)
    if absent:
        logging.warning("Results lack example-selection columns: %s", sorted(absent))
        return 0
    trace_lookup = {
        (str(trace["image_id"]), str(trace["occlusion_method"]), int(trace["solution_index"])): trace
        for trace in traces
    }
    candidates = results[(results["validity"] == 1) & (results["iou_nodule"] > 0.10)].copy()
    candidates = candidates.sort_values(
        ["image_id", "iou_nodule", "cf_prob", "solution_index"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    selected_rows: list[pd.Series] = []
    seen_images: set[str] = set()
    for _, row in candidates.iterrows():
        image_id = str(row["image_id"])
        key = (image_id, str(row["occlusion_method"]), int(row["solution_index"]))
        if image_id not in seen_images and key in trace_lookup:
            selected_rows.append(row)
            seen_images.add(image_id)
        if len(selected_rows) >= args.save_successful_examples:
            break
    examples_root = ensure_directory(output_dir / "successful_examples")
    for row in selected_rows:
        image_id = str(row["image_id"])
        method = str(row["occlusion_method"])
        solution_index = int(row["solution_index"])
        trace = trace_lookup[(image_id, method, solution_index)]
        sample = dataset.load(image_id)
        partition = build_slic_partition(
            sample.image,
            sample.lung_mask,
            sample.gt_nodule_mask,
            num_segments=args.num_superpixels,
            compactness=args.slic_compactness,
            seed=args.seed,
        )
        activation = np.asarray(trace["activation"], dtype=bool)
        if activation.shape != (partition.num_superpixels,):
            logging.warning("Skipping stale trace for %s: SLIC cardinality differs", image_id)
            continue
        example_dir = ensure_directory(examples_root / image_id)
        engine = OcclusionEngine(sample.image, partition)
        diagnostic_labels = np.unique(partition.labels[sample.gt_nodule_mask & (partition.labels >= 0)])
        diagnostic_activation = np.ones(partition.num_superpixels, dtype=bool)
        diagnostic_activation[diagnostic_labels] = False
        diagnostic_mask = partition.mask_from_activation(diagnostic_activation, off=True)
        plot_occlusion_method_comparison(
            sample.image,
            partition,
            diagnostic_mask,
            example_dir / "occlusion_method_comparison.png",
            image_id,
            engine=engine,
        )
        plot_ground_truth_construction(
            sample.image,
            sample.lung_mask,
            sample.gt_nodule_mask,
            example_dir / "ground_truth_construction.png",
            image_id,
        )
        sensitivity = np.asarray([np.nan if value is None else value for value in trace["sensitivity"]], dtype=float)
        plot_maxsat_explanation_process(
            sample.image,
            partition,
            sample.gt_nodule_mask,
            sensitivity,
            partition.mask_from_activation(activation, off=True),
            example_dir / f"maxsat_process_{method}_solution_{solution_index}.png",
            image_id,
            method,
            float(row["iou_nodule"]),
            float(row["dice_nodule"]),
            float(row["nodule_coverage"]),
        )
    return len(selected_rows)


def run(args: argparse.Namespace) -> Path:
    """Execute benchmark and return its result CSV path."""
    seed_everything(args.seed)
    output_dir = ensure_directory(args.output_dir)
    figures_dir = ensure_directory(output_dir / "figures")
    results_csv = output_dir / "results.csv"
    traces_path = output_dir / "explanation_traces.jsonl"
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
        image_dir=args.image_dir,
        lung_mask_dir=args.lung_mask_dir,
        ground_truth_representation=args.ground_truth_representation,
        provenance_manifest=args.provenance_manifest,
        num_superpixels=args.num_superpixels,
        slic_compactness=args.slic_compactness,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        weight_mode=args.weight_mode,
        max_oracle_calls=args.max_oracle_calls,
        exploration_random_samples=args.exploration_random_samples,
        diverse_solutions=args.diverse_solutions,
        repetition_penalty=args.repetition_penalty,
        minimum_new_occluded_regions=args.minimum_new_occluded_regions,
        cegar_cut_mode=args.cegar_cut_mode,
        seed_guidance_weight=args.seed_guidance_weight,
        anti_sensitivity_penalty=args.anti_sensitivity_penalty,
        low_sensitivity_penalty=args.low_sensitivity_penalty,
        sensitivity_normalization=args.sensitivity_normalization,
        minimum_lung_overlap=args.minimum_lung_overlap,
        hybrid_exact_budget_fraction=args.hybrid_exact_budget_fraction,
        oracle_batch_size=args.oracle_batch_size,
        per_sample_timeout_sec=args.per_sample_timeout_sec,
        save_successful_examples=args.save_successful_examples,
        classifier=args.classifier,
        device=args.device,
        batch_size=args.batch_size,
        max_images=args.max_images,
        seed=args.seed,
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    if args.provenance_manifest:
        source_manifest = Path(args.provenance_manifest)
        if not source_manifest.is_file():
            raise FileNotFoundError(f"provenance manifest does not exist: {source_manifest}")
        shutil.copy2(source_manifest, output_dir / "ground_truth_provenance.json")
    dataset = JSRTDataset(
        jsrt_root,
        nodule_csv=nodule_csv,
        image_dir=args.image_dir,
        lung_mask_dir=args.lung_mask_dir,
    )
    identifiers = dataset.image_ids(nodules_only=True)
    if not identifiers:
        raise RuntimeError("no image with both lung mask and nodule coordinate was found")
    if args.max_images is not None:
        identifiers = identifiers[: args.max_images]
    completed = set() if args.overwrite else _completed_pairs(completed_path)
    if args.overwrite and results_csv.exists():
        results_csv.unlink()
    if args.overwrite and traces_path.exists():
        traces_path.unlink()
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
                    cegar_cut_mode=args.cegar_cut_mode,
                    seed_guidance_weight=args.seed_guidance_weight,
                    anti_sensitivity_penalty=args.anti_sensitivity_penalty,
                    low_sensitivity_penalty=args.low_sensitivity_penalty,
                    sensitivity_normalization=args.sensitivity_normalization,
                    minimum_lung_overlap=args.minimum_lung_overlap,
                    hybrid_exact_budget_fraction=args.hybrid_exact_budget_fraction,
                    oracle_batch_size=args.oracle_batch_size,
                    timeout_seconds=args.per_sample_timeout_sec,
                )
                method_started = time.perf_counter()
                solutions = explainer.diverse_solutions(
                    maximum_solutions=args.diverse_solutions,
                    repetition_penalty=args.repetition_penalty,
                    minimum_new_occluded_regions=args.minimum_new_occluded_regions,
                )
                if not solutions:
                    # Persist an invalid terminal trace instead of silently omitting it.
                    terminal = explainer.explain()
                    solutions = [terminal]
                method_elapsed = time.perf_counter() - method_started
                for solution in solutions:
                    logging.info(
                        "Method %s/%s done status=%s valid=%s calls=%d candidates=%d "
                        "blocks=%d time=%.2fs oracle_time=%.2fs",
                        image_id,
                        method,
                        solution.status,
                        solution.valid,
                        solution.oracle_calls,
                        solution.candidate_evaluations,
                        solution.exact_block_count,
                        method_elapsed,
                        solution.oracle_time_sec,
                    )
                diversity = mean_diversity_jaccard(solutions)
                for solution_index, solution in enumerate(solutions):
                    record = evaluate_counterfactual(
                        solution,
                        partition,
                        sample.gt_nodule_mask,
                        image_id=image_id,
                        solution_index=solution_index,
                        diversity_jaccard=diversity,
                        extra_fields={
                            "source_path": str(sample.source_path),
                            "ground_truth_representation": args.ground_truth_representation,
                        },
                    )
                    _append_incremental(record, results_csv)
                    _append_trace(counterfactual_trace(solution, image_id, solution_index), traces_path)
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
    saved_examples = _save_successful_examples(
        dataset, results, _load_traces(traces_path), args, output_dir
    )
    if args.save_successful_examples:
        logging.info("Generated academic artifacts for %d successful cases", saved_examples)
    logging.info("Completed benchmark: %s rows in %s", len(results), results_csv)
    return results_csv


def main() -> None:
    """Configure logging and launch the command-line benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
