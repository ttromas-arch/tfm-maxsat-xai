"""Fast invariants for synthetic data, perturbation and MaxSAT components."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scripts.smoke_test import synthetic_radiograph
from src.data.jsrt import JSRTDataset, load_nodule_annotations
from src.data.superpixels import build_slic_partition
from src.evaluation.metrics import binary_mask_metrics, jaccard_distance
from src.explainers.partial_maxsat import PartialMaxSATCounterfactualExplainer
from src.models.classifier import SyntheticNoduleClassifier
from src.occlusion.engine import OCCLUSION_METHODS, OcclusionEngine


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