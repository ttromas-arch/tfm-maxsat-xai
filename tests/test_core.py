"""Fast invariants for synthetic data, perturbation and MaxSAT components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.smoke_test import synthetic_radiograph
from src.data.kaggle_jsrt import prepare_kaggle_jsrt
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