"""Preparation of the publicly available JSRT Kaggle inputs for this benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.common import ensure_directory, stable_stem

LUNG_MASK_DATASET_REF = "abduzzami/jsrt-247-image-lung-segmentation-mask-dataset"
NODULE_METADATA_DATASET_REF = "raddar/nodules-in-chest-xrays-jsrt"
JSRT_COORDINATE_REFERENCE_SHAPE = (2048, 2048)
JSRT_PIXEL_SPACING_MM = 0.175
_IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".img"}


@dataclass(frozen=True)
class KaggleJSRTPreparation:
    """Resolved paths and provenance required for one Kaggle benchmark run."""

    lung_dataset_ref: str
    metadata_dataset_ref: str
    image_dir: str
    lung_mask_dir: str
    nodule_csv: str
    metadata_csv: str
    metadata_sha256: str
    image_count: int
    lung_mask_count: int
    matched_pair_count: int
    annotated_nodule_count: int
    coordinate_reference_width: int
    coordinate_reference_height: int
    nodule_gt_representation: str
    notes: str


def _image_count(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _find_directory(root: Path, aliases: tuple[str, ...], kind: str) -> Path:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.strip().lower() in aliases and _image_count(path) > 0
    ]
    if root.name.strip().lower() in aliases and _image_count(root) > 0:
        candidates.append(root)
    if not candidates:
        raise FileNotFoundError(f"no {kind} directory ({aliases}) was found below {root}")
    maximum = max(_image_count(path) for path in candidates)
    winners = sorted(path for path in candidates if _image_count(path) == maximum)
    if len(winners) != 1:
        raise ValueError(f"ambiguous {kind} directories under {root}: {winners}")
    return winners[0]


def discover_lung_mask_layout(lung_dataset_root: str | Path) -> tuple[Path, Path]:
    """Find the ``cxr`` and lung ``mask(s)`` folders in the Kaggle JSRT dataset."""
    root = Path(lung_dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"lung dataset root does not exist: {root}")
    images = _find_directory(root, ("cxr", "images", "image"), "radiograph")
    masks = _find_directory(root, ("mask", "masks", "lung_masks", "lung_mask"), "lung-mask")
    image_ids = {stable_stem(path) for path in images.rglob("*") if path.is_file()}
    mask_ids = {stable_stem(path) for path in masks.rglob("*") if path.is_file()}
    if not image_ids.intersection(mask_ids):
        raise ValueError(f"no matching image/mask identifiers between {images} and {masks}")
    return images, masks


def find_jsrt_metadata_csv(metadata_dataset_root: str | Path) -> Path:
    """Locate the metadata table from the complementary JSRT Kaggle dataset."""
    root = Path(metadata_dataset_root)
    if root.is_file() and root.name.lower() == "jsrt_metadata.csv":
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"metadata dataset root does not exist: {root}")
    matches = sorted(path for path in root.rglob("jsrt_metadata.csv") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one jsrt_metadata.csv below {root}; found {len(matches)}"
        )
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_nodule_csv_from_jsrt_metadata(
    metadata_csv: str | Path,
    output_csv: str | Path,
    coordinate_reference_shape: tuple[int, int] = JSRT_COORDINATE_REFERENCE_SHAPE,
    pixel_spacing_mm: float = JSRT_PIXEL_SPACING_MM,
) -> int:
    """Convert ``jsrt_metadata.csv`` positives into the canonical annotation CSV.

    The supplementary table provides centres in native 2048×2048 JSRT coordinates
    and clinical ``size`` in millimetres. It does not provide manual pixel contours;
    the GT mask is therefore a reconstructed circle, recorded in the output itself.
    """
    source, target = Path(metadata_csv), Path(output_csv)
    if min(*coordinate_reference_shape) <= 0 or pixel_spacing_mm <= 0:
        raise ValueError("coordinate reference shape and pixel spacing must be positive")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"study_id", "x", "y", "size", "state"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"metadata must include {sorted(expected)}; found {reader.fieldnames}")
        records: list[dict[str, object]] = []
        for row_number, row in enumerate(reader, start=2):
            state = str(row.get("state") or "").strip().lower()
            if state == "non-nodule":
                continue
            try:
                x, y, size_mm = float(row["x"]), float(row["y"]), float(row["size"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid positive nodule annotation at row {row_number}") from error
            if size_mm <= 0:
                raise ValueError(f"non-positive nodule size at row {row_number}")
            records.append(
                {
                    "image_id": Path(str(row["study_id"])).stem,
                    "x": x,
                    "y": y,
                    "diameter_mm": size_mm,
                    "coordinate_reference_width": coordinate_reference_shape[1],
                    "coordinate_reference_height": coordinate_reference_shape[0],
                    "has_nodule": 1,
                    "gt_representation": "circle_from_coordinate_diameter",
                    "metadata_state": state,
                }
            )
    if not records:
        raise ValueError(f"no positive nodules were found in {source}")
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def prepare_kaggle_jsrt(
    lung_dataset_root: str | Path,
    metadata_dataset_root: str | Path,
    output_dir: str | Path,
) -> tuple[KaggleJSRTPreparation, Path]:
    """Resolve Kaggle inputs, build annotations and persist a provenance manifest."""
    images, lung_masks = discover_lung_mask_layout(lung_dataset_root)
    metadata = find_jsrt_metadata_csv(metadata_dataset_root)
    output = ensure_directory(output_dir)
    nodule_csv = output / "nodules_from_jsrt_metadata.csv"
    build_nodule_csv_from_jsrt_metadata(metadata, nodule_csv)
    image_ids = {stable_stem(path) for path in images.rglob("*") if path.is_file()}
    mask_ids = {stable_stem(path) for path in lung_masks.rglob("*") if path.is_file()}
    with nodule_csv.open("r", encoding="utf-8", newline="") as handle:
        annotation_ids = {stable_stem(row["image_id"]) for row in csv.DictReader(handle)}
    matched = image_ids.intersection(mask_ids)
    matched_positive = matched.intersection(annotation_ids)
    if not matched_positive:
        raise ValueError("no positive IDs are shared by image, lung-mask and metadata sources")
    preparation = KaggleJSRTPreparation(
        lung_dataset_ref=LUNG_MASK_DATASET_REF,
        metadata_dataset_ref=NODULE_METADATA_DATASET_REF,
        image_dir=str(images.resolve()),
        lung_mask_dir=str(lung_masks.resolve()),
        nodule_csv=str(nodule_csv.resolve()),
        metadata_csv=str(metadata.resolve()),
        metadata_sha256=_sha256(metadata),
        image_count=len(image_ids),
        lung_mask_count=len(mask_ids),
        matched_pair_count=len(matched),
        annotated_nodule_count=len(matched_positive),
        coordinate_reference_width=JSRT_COORDINATE_REFERENCE_SHAPE[1],
        coordinate_reference_height=JSRT_COORDINATE_REFERENCE_SHAPE[0],
        nodule_gt_representation="circle_from_coordinate_diameter",
        notes=(
            "Nodule masks are reconstructed circles from centre coordinates and clinical "
            "diameter metadata; they are not manual pixel-level nodule segmentations."
        ),
    )
    manifest = output / "kaggle_jsrt_manifest.json"
    manifest.write_text(json.dumps(asdict(preparation), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return preparation, manifest