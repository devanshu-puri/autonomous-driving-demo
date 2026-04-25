"""Dataset loading helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import COLOR_TO_INDEX, IMAGE_SIZE, INDEX_TO_COLOR


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    image_dir: Path
    mask_dir: Path
    metadata_path: Path
    models_dir: Path
    outputs_dir: Path
    segmentation_model_path: Path
    policy_model_path: Path
    metrics_path: Path
    predictions_path: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        data_dir = root / "data" / "comma10k_subset"
        models_dir = root / "models"
        outputs_dir = root / "outputs"
        return cls(
            root=root,
            data_dir=data_dir,
            image_dir=data_dir / "images",
            mask_dir=data_dir / "masks",
            metadata_path=data_dir / "metadata.csv",
            models_dir=models_dir,
            outputs_dir=outputs_dir,
            segmentation_model_path=models_dir / "segmentation_random_forest.joblib",
            policy_model_path=models_dir / "policy_decision_tree.joblib",
            metrics_path=outputs_dir / "metrics.json",
            predictions_path=outputs_dir / "predictions.csv",
        )


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str


def discover_records(paths: ProjectPaths) -> list[SampleRecord]:
    image_paths = sorted(paths.image_dir.glob("*.png"))
    records: list[SampleRecord] = []
    split_index = int(len(image_paths) * 0.8)

    for index, image_path in enumerate(image_paths):
        mask_path = paths.mask_dir / image_path.name
        if not mask_path.exists():
            continue
        sample_id = image_path.stem
        split = "train" if index < split_index else "test"
        records.append(
            SampleRecord(
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_path,
                split=split,
            )
        )
    return records


def save_metadata(records: list[SampleRecord], paths: ProjectPaths) -> None:
    paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "split", "image_path", "mask_path"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "split": record.split,
                    "image_path": record.image_path.name,
                    "mask_path": record.mask_path.name,
                }
            )


def load_rgb_image(path: Path, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def load_rgb_mask(path: Path, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    mask = Image.open(path).convert("RGB").resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def encode_mask(mask_rgb: np.ndarray) -> np.ndarray:
    encoded = np.full(mask_rgb.shape[:2], fill_value=-1, dtype=np.int16)
    for color, label_index in COLOR_TO_INDEX.items():
        matches = np.all(mask_rgb == np.asarray(color, dtype=np.uint8), axis=-1)
        encoded[matches] = label_index
    if np.any(encoded < 0):
        raise ValueError("Mask contains colors not defined in the class map.")
    return encoded


def decode_mask(label_map: np.ndarray) -> np.ndarray:
    mask_rgb = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    for label_index, color in INDEX_TO_COLOR.items():
        mask_rgb[label_map == label_index] = np.asarray(color, dtype=np.uint8)
    return mask_rgb
