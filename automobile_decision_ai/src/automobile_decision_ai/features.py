"""Feature engineering and explainable action logic."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .constants import (
    ACTION_DESCRIPTIONS,
    CLASS_TO_INDEX,
    SCENE_FEATURE_NAMES,
)


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    diff = maxc - minc

    hue = np.zeros_like(maxc)
    saturation = np.where(maxc == 0, 0.0, diff / np.maximum(maxc, 1e-8))
    value = maxc

    red_is_max = maxc == rgb[:, :, 0]
    green_is_max = maxc == rgb[:, :, 1]
    blue_is_max = maxc == rgb[:, :, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        hue[red_is_max] = ((rgb[:, :, 1] - rgb[:, :, 2]) / np.maximum(diff, 1e-8))[
            red_is_max
        ]
        hue[green_is_max] = (
            2.0 + (rgb[:, :, 2] - rgb[:, :, 0]) / np.maximum(diff, 1e-8)
        )[green_is_max]
        hue[blue_is_max] = (
            4.0 + (rgb[:, :, 0] - rgb[:, :, 1]) / np.maximum(diff, 1e-8)
        )[blue_is_max]

    hue = (hue / 6.0) % 1.0
    return np.stack([hue, saturation, value], axis=-1)


def build_pixel_feature_matrix(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    hsv = rgb_to_hsv(image)
    ys, xs = np.indices((height, width), dtype=np.float32)

    features = np.stack(
        [
            image[:, :, 0].astype(np.float32) / 255.0,
            image[:, :, 1].astype(np.float32) / 255.0,
            image[:, :, 2].astype(np.float32) / 255.0,
            hsv[:, :, 0],
            hsv[:, :, 1],
            hsv[:, :, 2],
            xs / float(max(width - 1, 1)),
            ys / float(max(height - 1, 1)),
        ],
        axis=-1,
    )
    return features.reshape(-1, features.shape[-1])


def sample_training_pixels(
    image: np.ndarray,
    label_map: np.ndarray,
    rng: np.random.Generator,
    per_class_samples: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    features = build_pixel_feature_matrix(image)
    labels = label_map.reshape(-1)

    sampled_features: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []

    for class_index in sorted(set(labels.tolist())):
        class_positions = np.flatnonzero(labels == class_index)
        if len(class_positions) == 0:
            continue
        sample_size = min(per_class_samples, len(class_positions))
        chosen = rng.choice(class_positions, size=sample_size, replace=False)
        sampled_features.append(features[chosen])
        sampled_labels.append(labels[chosen])

    return np.vstack(sampled_features), np.concatenate(sampled_labels)


def _region_ratios(label_map: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> dict[str, float]:
    region = label_map[y0:y1, x0:x1]
    flat = region.reshape(-1)
    counts = Counter(flat.tolist())
    total = max(len(flat), 1)

    road = counts[CLASS_TO_INDEX["road"]] / total
    lane = counts[CLASS_TO_INDEX["lane_marking"]] / total
    undrivable = counts[CLASS_TO_INDEX["undrivable"]] / total
    movable = counts[CLASS_TO_INDEX["movable"]] / total

    return {
        "road": road,
        "lane": lane,
        "drivable": road + lane,
        "undrivable": undrivable,
        "movable": movable,
    }


def compute_scene_metrics(label_map: np.ndarray) -> dict[str, float]:
    height, width = label_map.shape

    front = _region_ratios(
        label_map,
        int(width * 0.37),
        int(width * 0.63),
        int(height * 0.45),
        int(height * 0.90),
    )
    left = _region_ratios(
        label_map,
        int(width * 0.10),
        int(width * 0.37),
        int(height * 0.45),
        int(height * 0.90),
    )
    right = _region_ratios(
        label_map,
        int(width * 0.63),
        int(width * 0.90),
        int(height * 0.45),
        int(height * 0.90),
    )
    overall = _region_ratios(
        label_map,
        int(width * 0.05),
        int(width * 0.95),
        int(height * 0.35),
        int(height * 0.95),
    )

    return {
        "front_drivable_ratio": front["drivable"],
        "front_lane_ratio": front["lane"],
        "front_undrivable_ratio": front["undrivable"],
        "front_movable_ratio": front["movable"],
        "left_drivable_ratio": left["drivable"],
        "left_movable_ratio": left["movable"],
        "right_drivable_ratio": right["drivable"],
        "right_movable_ratio": right["movable"],
        "overall_movable_ratio": overall["movable"],
        "overall_undrivable_ratio": overall["undrivable"],
    }


def expert_action_from_metrics(metrics: dict[str, float]) -> str:
    if metrics["front_movable_ratio"] > 0.07 or metrics["front_undrivable_ratio"] > 0.35:
        return "STOP"

    if (
        metrics["left_movable_ratio"] > metrics["right_movable_ratio"] + 0.02
        and metrics["right_drivable_ratio"] > metrics["left_drivable_ratio"] + 0.04
    ):
        return "MOVE_RIGHT"

    if (
        metrics["right_movable_ratio"] > metrics["left_movable_ratio"] + 0.02
        and metrics["left_drivable_ratio"] > metrics["right_drivable_ratio"] + 0.04
    ):
        return "MOVE_LEFT"

    if metrics["front_movable_ratio"] > 0.025 or metrics["front_undrivable_ratio"] > 0.22:
        return "SLOW_DOWN"

    if metrics["front_lane_ratio"] < 0.003 or metrics["front_drivable_ratio"] < 0.20:
        return "CAUTION"

    return "KEEP_STRAIGHT"


def metrics_to_vector(metrics: dict[str, float]) -> list[float]:
    return [metrics[name] for name in SCENE_FEATURE_NAMES]


def build_action_explanation(action: str, metrics: dict[str, float]) -> list[str]:
    notes = [ACTION_DESCRIPTIONS[action]]

    if metrics["front_movable_ratio"] > 0.025:
        notes.append(
            f"Movable objects ahead: {metrics['front_movable_ratio']:.1%} of the front path."
        )

    if metrics["front_undrivable_ratio"] > 0.18:
        notes.append(
            f"Blocked or undrivable front area: {metrics['front_undrivable_ratio']:.1%}."
        )

    if action == "MOVE_LEFT":
        notes.append(
            f"Right side is busier ({metrics['right_movable_ratio']:.1%}) than the left ({metrics['left_movable_ratio']:.1%})."
        )

    if action == "MOVE_RIGHT":
        notes.append(
            f"Left side is busier ({metrics['left_movable_ratio']:.1%}) than the right ({metrics['right_movable_ratio']:.1%})."
        )

    if action == "CAUTION":
        notes.append(
            f"Lane visibility in front is low at {metrics['front_lane_ratio']:.1%}."
        )

    if action == "KEEP_STRAIGHT":
        notes.append(
            f"Drivable front area remains strong at {metrics['front_drivable_ratio']:.1%}."
        )

    return notes
