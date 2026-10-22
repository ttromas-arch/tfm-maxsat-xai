"""JSRT image, lung-mask and nodule-coordinate ingestion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
from PIL import Image

from src.common import as_grayscale_float, stable_stem

_IMAGE_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".img"}
_ID_COLUMNS = ("image_id", "image", "case", "case_id", "id", "filename", "file")
_X_COLUMNS = (
    "x",
    "coord_x",
    "x_coordinate",
    "nodule_x",
    "nodule_x_coordinate",
    "center_x",
    "xcoord",
)
_Y_COLUMNS = (
    "y",
    "coord_y",
    "y_coordinate",
    "nodule_y",
    "nodule_y_coordinate",
    "center_y",
    "ycoord",
)
_DIAMETER_COLUMNS = (
    "diameter",
    "diameter_px",
    "diameter_mm",
    "nodule_diameter",
    "nodule_diameter_px",
    "nodule_diameter_mm",
    "size",
)
_RADIUS_COLUMNS = ("radius", "radius_px", "radius_mm", "nodule_radius", "nodule_radius_mm")
_PRESENCE_COLUMNS = ("has_nodule", "nodule", "is_nodule", "positive")
_COORDINATE_WIDTH_COLUMNS = (
    "coordinate_width",
    "coordinate_reference_width",
    "reference_width",
    "source_width",
)
_COORDINATE_HEIGHT_COLUMNS = (
    "coordinate_height",
    "coordinate_reference_height",
    "reference_height",
    "source_height",
)


@dataclass(frozen=True)
class NoduleAnnotation:
    """Nodule centre and diameter with an optional coordinate reference image size."""

    image_id: str
    x: float
    y: float
    diameter: float
    coordinate_width: int | None = None
    coordinate_height: int | None = None


@dataclass(frozen=True)
class JSRTRadiograph:
    """One resized JSRT radiograph and its aligned medical masks."""

    image_id: str
    image: np.ndarray
    lung_mask: np.ndarray
    gt_nodule_mask: np.ndarray
    source_path: Path


def download_nodule_coordinates(url: str, destination: str | Path, timeout_sec: float = 30.0) -> Path:
    """Download a CSV coordinate table atomically enough for interrupted runs."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout_sec)
    response.raise_for_status()
    if not response.content:
        raise ValueError("downloaded coordinate file is empty")
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_bytes(response.content)
    temporary.replace(target)
    return target


def _canonical_header(header: str) -> str:
    """Normalize CSV headings such as ``Nodule Diameter (mm)`` for matching."""
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in header)
    return "_".join(part for part in normalized.split("_") if part)


def _select_column(columns: list[str], candidates: tuple[str, ...], required: bool = True) -> str | None:
    normalized = {_canonical_header(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    if required:
        raise ValueError(f"CSV requires one of {', '.join(candidates)}; found {columns}")
    return None


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "positive", "present"}


def _is_millimetre_column(column: str) -> bool:
    """Tell whether a normalized diameter/radius heading declares millimetres."""
    tokens = _canonical_header(column).split("_")
    return "mm" in tokens or "millimeter" in tokens or "millimetre" in tokens


def _dialect_for_csv(sample: str) -> csv.Dialect:
    """Infer a common delimiter while falling back safely to standard CSV."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def load_nodule_annotations(
    csv_path: str | Path,
    default_diameter_px: float = 10.0,
    pixel_spacing_mm: float = 0.175,
) -> dict[str, NoduleAnnotation]:
    """Parse flexible JSRT coordinate CSV headers into canonical annotations.

    Headings are case/punctuation-insensitive and comma, semicolon, tab and pipe
    tables are accepted. Values in a column ending in ``mm`` are converted using
    ``pixel_spacing_mm`` (JSRT's canonical spacing is 0.175 mm/pixel). Duplicate
    image rows are rejected because this pipeline evaluates one nodule mask per
    radiograph. Multi-nodule datasets should be merged upstream into a mask or
    represented by a dedicated extension of this loader.
    """
    source = Path(csv_path)
    if not source.is_file():
        raise FileNotFoundError(f"nodule CSV does not exist: {source}")
    if default_diameter_px <= 0 or pixel_spacing_mm <= 0:
        raise ValueError("default_diameter_px and pixel_spacing_mm must be positive")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        contents = handle.read()
    dialect = _dialect_for_csv(contents[:4096])
    reader = csv.DictReader(contents.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError(f"CSV has no header: {source}")
    names = list(reader.fieldnames)
    id_column = _select_column(names, _ID_COLUMNS)
    x_column = _select_column(names, _X_COLUMNS)
    y_column = _select_column(names, _Y_COLUMNS)
    diameter_column = _select_column(names, _DIAMETER_COLUMNS, required=False)
    radius_column = _select_column(names, _RADIUS_COLUMNS, required=False)
    presence_column = _select_column(names, _PRESENCE_COLUMNS, required=False)
    coordinate_width_column = _select_column(names, _COORDINATE_WIDTH_COLUMNS, required=False)
    coordinate_height_column = _select_column(names, _COORDINATE_HEIGHT_COLUMNS, required=False)
    annotations: dict[str, NoduleAnnotation] = {}
    for row_number, row in enumerate(reader, start=2):
        if presence_column and not _truthy(row.get(presence_column)):
            continue
        try:
            x, y = float(row[x_column]), float(row[y_column])
            diameter_text = str(row.get(diameter_column) or "").strip() if diameter_column else ""
            radius_text = str(row.get(radius_column) or "").strip() if radius_column else ""
            if diameter_text:
                diameter = float(diameter_text)
                if _is_millimetre_column(diameter_column or ""):
                    diameter /= pixel_spacing_mm
            elif radius_text:
                diameter = 2.0 * float(radius_text)
                if _is_millimetre_column(radius_column or ""):
                    diameter /= pixel_spacing_mm
            else:
                diameter = default_diameter_px
            coordinate_width = (
                int(float(str(row.get(coordinate_width_column) or "").strip()))
                if coordinate_width_column and str(row.get(coordinate_width_column) or "").strip()
                else None
            )
            coordinate_height = (
                int(float(str(row.get(coordinate_height_column) or "").strip()))
                if coordinate_height_column and str(row.get(coordinate_height_column) or "").strip()
                else None
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid coordinate at CSV row {row_number}") from error
        if diameter <= 0:
            raise ValueError(f"diameter must be positive at CSV row {row_number}")
        if (coordinate_width is None) != (coordinate_height is None):
            raise ValueError(
                f"coordinate reference width and height must both be set at CSV row {row_number}"
            )
        if coordinate_width is not None and min(coordinate_width, coordinate_height or 0) <= 0:
            raise ValueError(f"coordinate reference dimensions must be positive at CSV row {row_number}")
        identifier = stable_stem(str(row[id_column]))
        if not identifier:
            raise ValueError(f"empty image identifier at CSV row {row_number}")
        if identifier in annotations:
            raise ValueError(f"duplicate annotation for image identifier {identifier}")
        annotations[identifier] = NoduleAnnotation(
            identifier,
            x,
            y,
            diameter,
            coordinate_width=coordinate_width,
            coordinate_height=coordinate_height,
        )
    return annotations


def create_nodule_mask(
    annotation: NoduleAnnotation | None,
    source_shape: tuple[int, int],
    output_shape: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Rasterize a circular annotation and resize its geometry to target pixels."""
    if annotation is None:
        return np.zeros(output_shape, dtype=bool)
    source_height, source_width = source_shape
    target_height, target_width = output_shape
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("source_shape and output_shape must be positive")
    coordinate_width = annotation.coordinate_width or source_width
    coordinate_height = annotation.coordinate_height or source_height
    scale_x, scale_y = target_width / coordinate_width, target_height / coordinate_height
    center_x, center_y = annotation.x * scale_x, annotation.y * scale_y
    radius_x, radius_y = max(annotation.diameter * scale_x / 2.0, 0.5), max(
        annotation.diameter * scale_y / 2.0, 0.5
    )
    y_grid, x_grid = np.ogrid[:target_height, :target_width]
    return (((x_grid - center_x) / radius_x) ** 2 + ((y_grid - center_y) / radius_y) ** 2 <= 1.0)


def _read_raw_jsrt_image(path: Path) -> np.ndarray:
    """Read a headerless 12-bit JSRT ``.IMG`` file with automatic byte order.

    Official JSRT releases commonly use 2048×2048 unsigned 16-bit containers for
    12-bit pixels. The implementation also accepts any square raw image so private
    resized copies can be evaluated. Byte order is selected by the representation
    whose values most plausibly lie in the 12-bit interval.
    """
    byte_count = path.stat().st_size
    if byte_count == 0 or byte_count % 2:
        raise ValueError(f"raw JSRT image has an invalid byte length: {path}")
    pixels = byte_count // 2
    side = int(np.sqrt(pixels))
    if side * side != pixels:
        raise ValueError(
            f"raw JSRT image must contain a square number of uint16 pixels: {path}"
        )
    little = np.fromfile(path, dtype="<u2")
    big = little.byteswap()
    little_plausibility = int((little <= 4095).sum())
    big_plausibility = int((big <= 4095).sum())
    selected = little if little_plausibility >= big_plausibility else big
    return selected.reshape(side, side)


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".img":
        return _read_raw_jsrt_image(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _normalize_raw_intensity(image: np.ndarray) -> np.ndarray:
    """Normalize a finite RAW acquisition by observed dynamic range before resize."""
    values = np.asarray(image, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    return np.zeros_like(values) if high == low else (values - low) / (high - low)


def _resize_float(image: np.ndarray, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    array = np.asarray(image)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(array)
    resized = pil.resize((shape[1], shape[0]), resample=mode)
    return np.asarray(resized)


class JSRTDataset:
    """Discover a local/Kaggle JSRT export and yield aligned 224-pixel samples."""

    def __init__(
        self,
        root: str | Path,
        nodule_csv: str | Path | None = None,
        output_shape: tuple[int, int] = (224, 224),
        image_dir: str | Path | None = None,
        lung_mask_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.output_shape = output_shape
        if not self.root.is_dir():
            raise FileNotFoundError(f"JSRT root does not exist: {self.root}")
        candidate_csv = Path(nodule_csv) if nodule_csv else self._discover_csv()
        self.annotations = load_nodule_annotations(candidate_csv) if candidate_csv else {}
        image_root = self._resolve_data_directory(image_dir, ("cxr", "images", "image"), "image")
        mask_root = self._resolve_data_directory(
            lung_mask_dir, ("mask", "masks", "lung_masks", "lung_mask"), "lung mask"
        )
        self.images = self._discover_files(image_root, "image")
        self.lung_masks = self._discover_files(mask_root, "lung mask")

    def _discover_csv(self) -> Path | None:
        preferred = ("nodules.csv", "nodule_coordinates.csv", "nodules_official.csv")
        for filename in preferred:
            candidate = self.root / filename
            if candidate.is_file():
                return candidate
        tables = sorted(self.root.glob("*.csv"))
        return tables[0] if tables else None

    def _resolve_data_directory(
        self,
        explicit: str | Path | None,
        aliases: tuple[str, ...],
        kind: str,
    ) -> Path:
        """Resolve explicit or conventional Local/Kaggle image subdirectories.

        Kaggle's JSRT lung dataset uses ``content/jsrt/cxr`` and
        ``content/jsrt/masks``. Local exports historically use ``images`` and
        ``lung_masks``. Discovery selects the candidate with the most supported
        image files, rejecting ambiguous ties to prevent image/mask mixing.
        """
        if explicit is not None:
            candidate = Path(explicit)
            candidate = candidate if candidate.is_absolute() else self.root / candidate
            if not candidate.is_dir():
                raise FileNotFoundError(f"configured {kind} directory does not exist: {candidate}")
            return candidate
        candidates = [
            path
            for path in self.root.rglob("*")
            if path.is_dir() and path.name.strip().lower() in aliases
        ]
        if self.root.name.strip().lower() in aliases:
            candidates.append(self.root)
        scored = [
            (
                sum(1 for file in path.rglob("*") if file.suffix.lower() in _IMAGE_EXTENSIONS),
                path,
            )
            for path in candidates
        ]
        viable = [(count, path) for count, path in scored if count]
        if not viable:
            raise FileNotFoundError(
                f"no {kind} directory found below {self.root}; expected one of {aliases}"
            )
        maximum = max(count for count, _ in viable)
        winners = sorted(path for count, path in viable if count == maximum)
        if len(winners) > 1:
            raise ValueError(f"ambiguous {kind} directories beneath {self.root}: {winners}")
        return winners[0]

    def _discover_files(self, root: Path, kind: str) -> dict[str, Path]:
        files = sorted(path for path in root.rglob("*") if path.suffix.lower() in _IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"no {kind} files found beneath {root}")
        discovered: dict[str, Path] = {}
        for file in files:
            identifier = stable_stem(file)
            if identifier in discovered:
                raise ValueError(f"duplicate {kind} identifier {identifier}: {file}")
            discovered[identifier] = file
        return discovered

    def image_ids(self, nodules_only: bool = True) -> list[str]:
        """Return sorted image identifiers with a matched lung mask and optional GT."""
        identifiers = sorted(set(self.images).intersection(self.lung_masks))
        return [identifier for identifier in identifiers if identifier in self.annotations] if nodules_only else identifiers

    def load(self, image_id: str) -> JSRTRadiograph:
        """Load, normalize and spatially align one image and both masks."""
        identifier = stable_stem(image_id)
        if identifier not in self.images:
            raise KeyError(f"unknown image identifier: {image_id}")
        if identifier not in self.lung_masks:
            raise FileNotFoundError(f"no lung mask matches image identifier {identifier}")
        source_path = self.images[identifier]
        source = _read_image(source_path)
        source_shape = tuple(source.shape)
        lung = _read_image(self.lung_masks[identifier]) > 0
        if lung.shape != source.shape:
            raise ValueError(f"lung mask shape {lung.shape} does not match image shape {source.shape}")
        source_for_resize = (
            _normalize_raw_intensity(source) if source_path.suffix.lower() == ".img" else source
        )
        resized_image = as_grayscale_float(_resize_float(source_for_resize, self.output_shape))
        resized_lung = _resize_float(lung, self.output_shape, nearest=True) > 0
        if not resized_lung.any():
            raise ValueError(f"lung mask for {identifier} becomes empty after resizing")
        nodule = create_nodule_mask(self.annotations.get(identifier), source_shape, self.output_shape)
        return JSRTRadiograph(identifier, resized_image, resized_lung, nodule, source_path)

    def iter_samples(self, nodules_only: bool = True) -> Iterator[JSRTRadiograph]:
        """Yield samples in deterministic identifier order."""
        for identifier in self.image_ids(nodules_only=nodules_only):
            yield self.load(identifier)