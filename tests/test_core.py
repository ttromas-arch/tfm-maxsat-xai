"""Fast invariants for synthetic data, perturbation and MaxSAT components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from scripts.smoke_test import synthetic_radiograph
from src.data.kaggle_jsrt import prepare_kaggle_jsrt
from src.data.jsrt import JSRTDataset, load_nodule_annotations
from src.data.superpixels import SuperpixelPartition, build_slic_partition
from src.evaluation.metrics import binary_mask_metrics, evaluate_counterfactual, jaccard_distance
from src.explainers.partial_maxsat import PartialMaxSATCounterfactualExplainer, counterfactual_trace
from src.models.classifier import SyntheticNoduleClassifier
from src.occlusion.engine import OCCLUSION_METHODS, OcclusionEngine
from src.visualization.plots import (
    plot_ground_truth_construction,
    plot_maxsat_explanation_process,
    plot_occlusion_method_comparison,
)


class ActivationOracle:
    """Deterministic black-box oracle keyed by exact global activation tuples."""

    def __init__(self, image: np.ndarray, partition: SuperpixelPartition, probabilities: dict[tuple[bool, ...], float], default: float = 0.95) -> None:
        self.image = np.asarray(image, dtype=float)
        self.partition = partition
        self.probabilities = probabilities
        self.default = default
        self.calls = 0

    def _activation_from_image(self, image: np.ndarray) -> tuple[bool, ...]:
        candidate = np.asarray(image, dtype=float)
        return tuple(
            bool(np.mean(np.abs(candidate[self.partition.labels == index] - self.image[self.partition.labels == index])) < 1e-4)
            for index in range(self.partition.num_superpixels)
        )

    def predict_probability(self, image: np.ndarray) -> float:
        self.calls += 1
        return float(self.probabilities.get(self._activation_from_image(image), self.default))

    def predict_probabilities(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray([self.predict_probability(image) for image in images], dtype=float)


def _four_region_partition() -> tuple[np.ndarray, np.ndarray, np.ndarray, SuperpixelPartition, OcclusionEngine]:
    """Create a four-region anatomical partition suitable for logic-level tests."""
    image = np.full((8, 8), 0.7, dtype=np.float32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[:4, 4:] = 1
    labels[4:, :4] = 2
    labels[4:, 4:] = 3
    lung = np.ones((8, 8), dtype=bool)
    lesion = labels == 0
    areas = np.bincount(labels.ravel(), minlength=4).astype(np.int64)
    intersections = np.bincount(labels[lesion], minlength=4).astype(np.int64)
    partition = SuperpixelPartition(
        labels=labels,
        anatomy_mask=lung,
        areas=areas,
        gt_intersections=intersections,
        gt_overlap_of_superpixel=intersections / areas,
        gt_coverage_by_superpixel=intersections / intersections.sum(),
    )
    engine = OcclusionEngine(image, partition)
    return image, lung, lesion, partition, engine


def test_slic_labels_are_compact_and_covered() -> None:
    image, lung, lesion = synthetic_radiograph()
    partition = build_slic_partition(image, lung, lesion, num_segments=16)
    labels = np.unique(partition.labels[lung])
    assert np.array_equal(labels, np.arange(partition.num_superpixels))
    assert partition.areas.sum() == lung.sum()
    assert partition.gt_intersections.sum() == lesion.sum()


def test_all_occlusions_preserve_shape_and_change_disabled_pixels() -> None:
    image, lung, lesion = synthetic_radiograph()
    partition = build_slic_partition(image, lung, lesion, num_segments=16)
    engine = OcclusionEngine(image, partition)
    all_off = np.zeros(partition.num_superpixels, dtype=bool)
    for method in OCCLUSION_METHODS:
        output = engine.perturb(all_off, method)
        assert output.shape == image.shape
        assert np.isfinite(output).all()
        assert np.any(np.abs(output - image) > 1e-5)


def test_rc2_finds_a_valid_minimal_synthetic_counterfactual() -> None:
    image, lung, lesion = synthetic_radiograph()
    partition = build_slic_partition(image, lung, lesion, num_segments=20, compactness=0.12)
    engine = OcclusionEngine(image, partition, blur_sigma=9)
    classifier = SyntheticNoduleClassifier(lesion, center=0.55, slope=13)
    explainer = PartialMaxSATCounterfactualExplainer(
        partition,
        engine,
        classifier,
        "zero",
        threshold=0.60,
        max_oracle_calls=80,
        exploration_random_samples=3,
        random_seed=2,
    )
    result = explainer.explain()
    assert result.valid
    assert result.counterfactual_probability < result.threshold
    assert result.num_occluded_superpixels > 0


def test_binary_metrics_and_jaccard_edge_cases() -> None:
    empty = np.zeros((3, 3), dtype=bool)
    metrics = binary_mask_metrics(empty, empty)
    assert metrics["iou_nodule"] == 1.0
    assert metrics["dice_nodule"] == 1.0
    assert np.isnan(metrics["hit_rate"])
    assert jaccard_distance(empty, empty) == 0.0


def test_annotation_parser_normalizes_headers_delimiters_and_millimetres(tmp_path) -> None:
    table = tmp_path / "official_like.csv"
    table.write_text("Case ID;Nodule X Coordinate;Nodule Y Coordinate;Nodule Diameter (mm)\nJPCLN001;10;20;3.5\n")
    annotations = load_nodule_annotations(table, pixel_spacing_mm=0.175)
    annotation = annotations["jpcln001"]
    assert annotation.x == 10.0
    assert annotation.y == 20.0
    assert annotation.diameter == pytest.approx(20.0)


def test_dataset_loads_and_normalizes_raw_jsrt_image(tmp_path) -> None:
    root = tmp_path / "jsrt"
    (root / "images").mkdir(parents=True)
    (root / "lung_masks").mkdir()
    raw = np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8)
    raw.astype("<u2").tofile(root / "images" / "JPCLN001.IMG")
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(root / "lung_masks" / "JPCLN001.gif")
    (root / "nodules.csv").write_text("image_id,x,y,diameter\nJPCLN001,4,4,2\n")
    sample = JSRTDataset(root).load("JPCLN001")
    assert sample.image.shape == (224, 224)
    assert sample.lung_mask.all()
    assert sample.gt_nodule_mask.any()
    assert sample.image.min() == pytest.approx(0.0)
    assert sample.image.max() == pytest.approx(1.0)


def test_kaggle_layout_and_metadata_are_prepared_with_reference_coordinates(tmp_path) -> None:
    lung_root = tmp_path / "lung_input" / "content" / "jsrt"
    metadata_root = tmp_path / "metadata_input"
    cxr, masks = lung_root / "cxr", lung_root / "masks"
    cxr.mkdir(parents=True)
    masks.mkdir()
    metadata_root.mkdir()
    Image.fromarray(np.full((32, 32), 127, dtype=np.uint8)).save(cxr / "JPCLN001.png")
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(masks / "JPCLN001.png")
    (metadata_root / "jsrt_metadata.csv").write_text(
        "study_id,subtlety,size,age,gender,x,y,state,position,diagnosis\n"
        "JPCLN001.png,5,17.5,53,Male,1024,1024,malignant,l.upper lobe,lung cancer\n"
        "JPCNN001.png,,,30,Female,,,non-nodule,,\n",
        encoding="utf-8",
    )
    preparation, manifest = prepare_kaggle_jsrt(lung_root.parents[1], metadata_root, tmp_path / "prepared")
    assert preparation.annotated_nodule_count == 1
    assert preparation.matched_pair_count == 1
    assert manifest.is_file()
    sample = JSRTDataset(
        lung_root.parents[1],
        nodule_csv=preparation.nodule_csv,
        image_dir=preparation.image_dir,
        lung_mask_dir=preparation.lung_mask_dir,
    ).load("JPCLN001")
    assert sample.gt_nodule_mask[112, 112]
    assert sample.gt_nodule_mask.sum() > 0
    annotation = load_nodule_annotations(preparation.nodule_csv)["jpcln001"]
    assert annotation.diameter == pytest.approx(100.0)
    assert annotation.coordinate_width == 2048


def test_kaggle_notebook_is_valid_python_and_has_safe_publication_contract() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "kaggle_runner.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = ""
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"kaggle_runner.ipynb:cell-{index}", "exec")
            code += source
    assert "kagglehub.dataset_download" in code
    assert "THRESHOLD_MODE = 'relative'" in code
    assert "--overwrite" in code
    assert "UserSecretsClient().get_secret('GITHUB_TOKEN')" in code
    assert ".github_askpass.sh" in code
    assert "https://x-access-token:" not in code


def test_solver_freezes_regions_without_lung_overlap() -> None:
    image = np.full((8, 8), 0.7, dtype=np.float32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[:, 4:] = 1
    lung = np.zeros((8, 8), dtype=bool)
    lung[:, :4] = True
    lesion = labels == 0
    partition = SuperpixelPartition(
        labels=labels,
        anatomy_mask=lung,
        areas=np.array([32, 32], dtype=np.int64),
        gt_intersections=np.array([32, 0], dtype=np.int64),
        gt_overlap_of_superpixel=np.array([1.0, 0.0]),
        gt_coverage_by_superpixel=np.array([1.0, 0.0]),
    )
    engine = OcclusionEngine(image, partition)
    probabilities = {(True, True): 0.95, (False, True): 0.20}
    oracle = ActivationOracle(image, partition, probabilities)
    explainer = PartialMaxSATCounterfactualExplainer(
        partition, engine, oracle, "zero", threshold=0.60, max_oracle_calls=2
    )
    result = explainer.explain()
    assert result.valid
    assert result.solver_superpixel_indices.tolist() == [0]
    assert result.frozen_outside_lung_indices.tolist() == [1]
    assert result.activation.tolist() == [False, True]


def test_exact_block_preserves_non_monotonic_subset_that_expansion_excludes() -> None:
    _, _, _, partition, engine = _four_region_partition()
    oracle = ActivationOracle(np.full((8, 8), 0.7), partition, {})
    explainer = PartialMaxSATCounterfactualExplainer(
        partition, engine, oracle, "zero", threshold=0.60, max_oracle_calls=5
    )
    failed = np.array([False, False, True, True])
    smaller_alternative = np.array([False, True, True, True])
    exact = explainer._exact_block_clause(failed)
    expansion = explainer._expansion_cut_clause(failed)

    def satisfies(clause: tuple[int, ...], assignment: np.ndarray) -> bool:
        return any((literal > 0 and assignment[literal - 1]) or (literal < 0 and not assignment[-literal - 1]) for literal in clause)

    assert satisfies(exact, smaller_alternative)
    assert expansion is not None and not satisfies(expansion, smaller_alternative)


def test_exact_cegar_finds_non_monotonic_swap_after_failed_singleton() -> None:
    """Exact blocking permits a valid pair that reactivates a failed singleton.

    Singleton 0 has the best individual drop but remains positive. The valid pair
    disables 1 and 2 while retaining 0, a non-monotonic exchange that exact CEGAR
    must remain free to evaluate.
    """
    image, _, _, partition, engine = _four_region_partition()
    all_on = (True, True, True, True)
    probabilities = {all_on: 0.95}
    probabilities[(False, True, True, True)] = 0.70
    probabilities[(True, False, True, True)] = 0.95
    probabilities[(True, True, False, True)] = 0.95
    probabilities[(True, True, True, False)] = 0.95
    probabilities[(True, False, False, True)] = 0.20
    exact_oracle = ActivationOracle(image, partition, probabilities, default=0.95)
    exact = PartialMaxSATCounterfactualExplainer(
        partition,
        engine,
        exact_oracle,
        "zero",
        threshold=0.60,
        max_oracle_calls=11,
        cegar_cut_mode="exact",
    ).explain()
    assert exact.valid
    assert exact.activation.tolist() == [True, False, False, True]
    assert exact.exact_block_count > 0


@pytest.mark.parametrize("cut_mode", ("exact", "expansion", "hybrid"))
def test_all_cegar_modes_execute_on_a_valid_seed_counterfactual(cut_mode: str) -> None:
    image, _, _, partition, engine = _four_region_partition()
    all_on = (True, True, True, True)
    probabilities = {all_on: 0.95}
    for index in range(4):
        activation = list(all_on)
        activation[index] = False
        probabilities[tuple(activation)] = 0.20
    oracle = ActivationOracle(image, partition, probabilities)
    result = PartialMaxSATCounterfactualExplainer(
        partition, engine, oracle, "zero", threshold=0.60, max_oracle_calls=5, cegar_cut_mode=cut_mode
    ).explain()
    assert result.valid
    assert result.search_lane in {"exact", "expansion"}
    assert result.cegar_cut_mode == cut_mode


def test_seed_guidance_penalizes_probability_increasing_artifact_regions() -> None:
    _, _, _, partition, engine = _four_region_partition()
    oracle = ActivationOracle(np.full((8, 8), 0.7), partition, {})
    explainer = PartialMaxSATCounterfactualExplainer(
        partition,
        engine,
        oracle,
        "zero",
        threshold=0.60,
        max_oracle_calls=5,
        anti_sensitivity_penalty=4.0,
        low_sensitivity_penalty=1.0,
    )
    sensitivity = np.array([0.30, -0.20, 0.05, 0.10])
    penalties = explainer._guidance_penalties(sensitivity)
    assert penalties[1] > penalties[0]
    assert penalties[1] > penalties[3]
    weights = explainer._lexicographic_weights(penalties, np.zeros(4, dtype=np.int64))
    assert weights[1] > weights[0]
    assert explainer.base_weights.tolist() == [1, 1, 1, 1]


def test_mdsol_shares_seed_cache_and_enforces_new_regions() -> None:
    image, _, _, partition, engine = _four_region_partition()
    all_on = (True, True, True, True)
    probabilities = {all_on: 0.95}
    for index in range(4):
        activation = list(all_on)
        activation[index] = False
        probabilities[tuple(activation)] = 0.20
    oracle = ActivationOracle(image, partition, probabilities)
    explainer = PartialMaxSATCounterfactualExplainer(
        partition,
        engine,
        oracle,
        "zero",
        threshold=0.60,
        max_oracle_calls=12,
        cegar_cut_mode="exact",
    )
    results = explainer.diverse_solutions(
        maximum_solutions=2, repetition_penalty=1, minimum_new_occluded_regions=1
    )
    assert len(results) == 2
    assert all(result.valid for result in results)
    assert oracle.calls == partition.num_superpixels + 1
    assert results[-1].oracle_cache_hits > 0
    assert np.any((~results[1].activation) & results[0].activation)
    trace = counterfactual_trace(results[0], "synthetic", 0)
    assert trace["oracle_cache_misses"] == partition.num_superpixels + 1


def test_mdsol_cardinality_novelty_constraint_supports_more_than_one_new_region() -> None:
    image, _, _, partition, engine = _four_region_partition()
    all_on = (True, True, True, True)
    probabilities = {all_on: 0.95}
    for index in range(4):
        activation = list(all_on)
        activation[index] = False
        probabilities[tuple(activation)] = 0.20
    # After a singleton solution, a second valid solution needs two newly disabled regions.
    for left in range(4):
        for right in range(left + 1, 4):
            activation = list(all_on)
            activation[left] = activation[right] = False
            probabilities[tuple(activation)] = 0.20
    oracle = ActivationOracle(image, partition, probabilities)
    results = PartialMaxSATCounterfactualExplainer(
        partition, engine, oracle, "zero", threshold=0.60, max_oracle_calls=20
    ).diverse_solutions(maximum_solutions=2, repetition_penalty=1, minimum_new_occluded_regions=2)
    assert len(results) == 2
    first_off, second_off = set(results[0].occluded_indices), set(results[1].occluded_indices)
    assert len(second_off.difference(first_off)) >= 2


def test_oracle_batch_cache_deduplicates_identical_activations() -> None:
    image, _, _, partition, engine = _four_region_partition()
    all_on = np.ones(partition.num_superpixels, dtype=bool)
    one_off = all_on.copy()
    one_off[0] = False
    oracle = ActivationOracle(image, partition, {tuple(all_on): 0.95, tuple(one_off): 0.20})
    explainer = PartialMaxSATCounterfactualExplainer(
        partition, engine, oracle, "zero", threshold=0.60, max_oracle_calls=8
    )
    from src.explainers.partial_maxsat import SearchState

    state = SearchState()
    values = explainer._query_oracle_many(np.stack([one_off, one_off, all_on, one_off]), state)
    assert np.allclose(values, [0.20, 0.20, 0.95, 0.20])
    assert state.oracle_calls == 2
    assert state.oracle_cache_misses == 2
    assert state.oracle_cache_hits == 2


def test_academic_plots_close_all_figures(tmp_path) -> None:
    image, lung, lesion = synthetic_radiograph()
    partition = build_slic_partition(image, lung, lesion, num_segments=16)
    engine = OcclusionEngine(image, partition)
    activation = np.ones(partition.num_superpixels, dtype=bool)
    activation[np.unique(partition.labels[lesion])] = False
    explanation = partition.mask_from_activation(activation, off=True)
    sensitivity = np.linspace(-0.2, 0.3, partition.num_superpixels)
    output = tmp_path / "figures"
    paths = [
        plot_occlusion_method_comparison(image, partition, explanation, output / "occlusion.png", "synthetic", engine),
        plot_ground_truth_construction(image, lung, lesion, output / "ground_truth.png", "synthetic"),
        plot_maxsat_explanation_process(
            image,
            partition,
            lesion,
            sensitivity,
            explanation,
            output / "process.png",
            "synthetic",
            "zero",
            1.0,
            1.0,
            1.0,
        ),
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert plt.get_fignums() == []