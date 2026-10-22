"""Prepare the two public Kaggle JSRT inputs for the TFM benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.kaggle_jsrt import prepare_kaggle_jsrt


def parse_args() -> argparse.Namespace:
    """Parse mounted/downloaded Kaggle dataset roots and a writable output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lung-dataset-root", required=True)
    parser.add_argument("--metadata-dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Create the canonical annotation CSV and print its machine-readable manifest."""
    args = parse_args()
    preparation, manifest = prepare_kaggle_jsrt(
        args.lung_dataset_root, args.metadata_dataset_root, args.output_dir
    )
    print(json.dumps({"manifest": str(manifest), **preparation.__dict__}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()