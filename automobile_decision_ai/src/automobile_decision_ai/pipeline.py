"""Training and inference pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .constants import ACTION_LABELS
from .dataset import (
    ProjectPaths,
    SampleRecord,
    decode_mask,
    discover_records,
    encode_mask,
    load_rgb_image,
    load_rgb_mask,
    save_metadata,
)
from .features import (
    build_action_explanation,
    compute_scene_metrics,
    expert_action_from_metrics,
    metrics_to_vector,
    sample_training_pixels,
)
from .visualization import save_demo_card


def _split_records(records: list[SampleRecord]) -> tuple[list[SampleRecord], list[SampleRecord]]:
    train_records = [record for record in records if record.split == "train"]
    test_records = [record for record in records if record.split == "test"]
    return train_records, test_records


def _train_segmentation_model(
    train_records: list[SampleRecord],
    random_state: int,
) -> RandomForestClassifier:
    rng = np.random.default_rng(random_state)
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    for record in train_records:
        image = load_rgb_image(record.image_path)
        label_map = encode_mask(load_rgb_mask(record.mask_path))
        sampled_features, sampled_labels = sample_training_pixels(
            image=image,
            label_map=label_map,
            rng=rng,
            per_class_samples=250,
        )
        feature_batches.append(sampled_features)
        label_batches.append(sampled_labels)

    features = np.vstack(feature_batches)
    labels = np.concatenate(label_batches)

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(features, labels)
    return model


def _predict_segmentation(
    segmentation_model: RandomForestClassifier,
    image: np.ndarray,
) -> np.ndarray:
    from .features import build_pixel_feature_matrix

    pixel_features = build_pixel_feature_matrix(image)
    predicted = segmentation_model.predict(pixel_features)
    return predicted.reshape(image.shape[:2])


def _train_policy_model(train_records: list[SampleRecord], random_state: int) -> DecisionTreeClassifier:
    features: list[list[float]] = []
    actions: list[str] = []

    for record in train_records:
        label_map = encode_mask(load_rgb_mask(record.mask_path))
        metrics = compute_scene_metrics(label_map)
        features.append(metrics_to_vector(metrics))
        actions.append(expert_action_from_metrics(metrics))

    model = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    model.fit(features, actions)
    return model


def train_and_save(root: Path, random_state: int = 42) -> dict[str, object]:
    paths = ProjectPaths.from_root(root)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(paths)
    save_metadata(records, paths)
    train_records, test_records = _split_records(records)

    segmentation_model = _train_segmentation_model(train_records, random_state)
    policy_model = _train_policy_model(train_records, random_state)

    joblib.dump(segmentation_model, paths.segmentation_model_path)
    joblib.dump(policy_model, paths.policy_model_path)

    pixel_truth: list[np.ndarray] = []
    pixel_pred: list[np.ndarray] = []
    expert_actions: list[str] = []
    predicted_actions: list[str] = []
    prediction_rows: list[dict[str, object]] = []

    for index, record in enumerate(test_records):
        image = load_rgb_image(record.image_path)
        gt_label_map = encode_mask(load_rgb_mask(record.mask_path))
        pred_label_map = _predict_segmentation(segmentation_model, image)

        gt_metrics = compute_scene_metrics(gt_label_map)
        pred_metrics = compute_scene_metrics(pred_label_map)

        expert_action = expert_action_from_metrics(gt_metrics)
        predicted_action = str(policy_model.predict([metrics_to_vector(pred_metrics)])[0])
        probabilities = policy_model.predict_proba([metrics_to_vector(pred_metrics)])[0]
        confidence = float(probabilities.max())

        pixel_truth.append(gt_label_map.reshape(-1))
        pixel_pred.append(pred_label_map.reshape(-1))
        expert_actions.append(expert_action)
        predicted_actions.append(predicted_action)

        prediction_rows.append(
            {
                "sample_id": record.sample_id,
                "split": record.split,
                "expert_action": expert_action,
                "predicted_action": predicted_action,
                "confidence": f"{confidence:.4f}",
                "front_drivable_ratio": f"{pred_metrics['front_drivable_ratio']:.4f}",
                "front_lane_ratio": f"{pred_metrics['front_lane_ratio']:.4f}",
                "front_undrivable_ratio": f"{pred_metrics['front_undrivable_ratio']:.4f}",
                "front_movable_ratio": f"{pred_metrics['front_movable_ratio']:.4f}",
            }
        )

        if index < 6:
            pred_mask_rgb = decode_mask(pred_label_map)
            gt_mask_rgb = decode_mask(gt_label_map)
            explanation = build_action_explanation(predicted_action, pred_metrics)
            save_demo_card(
                output_path=paths.outputs_dir / f"demo_{record.sample_id}.png",
                image=image,
                gt_mask_rgb=gt_mask_rgb,
                pred_mask_rgb=pred_mask_rgb,
                action=predicted_action,
                confidence=confidence,
                explanation=explanation,
            )

    flat_truth = np.concatenate(pixel_truth)
    flat_pred = np.concatenate(pixel_pred)
    pixel_accuracy = float(accuracy_score(flat_truth, flat_pred))
    action_accuracy = float(accuracy_score(expert_actions, predicted_actions))

    metrics = {
        "dataset": "comma10k_subset",
        "num_samples": len(records),
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "pixel_accuracy": pixel_accuracy,
        "decision_accuracy": action_accuracy,
        "action_labels": ACTION_LABELS,
        "decision_report": classification_report(
            expert_actions,
            predicted_actions,
            zero_division=0,
            output_dict=True,
        ),
    }

    with paths.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with paths.predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "split",
                "expert_action",
                "predicted_action",
                "confidence",
                "front_drivable_ratio",
                "front_lane_ratio",
                "front_undrivable_ratio",
                "front_movable_ratio",
            ),
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    return metrics


def ensure_artifacts(root: Path) -> None:
    paths = ProjectPaths.from_root(root)
    if not paths.segmentation_model_path.exists() or not paths.policy_model_path.exists():
        train_and_save(root)


def analyze_sample(root: Path, sample_id: str) -> dict[str, object]:
    ensure_artifacts(root)
    paths = ProjectPaths.from_root(root)
    records = {record.sample_id: record for record in discover_records(paths)}
    record = records[sample_id]

    segmentation_model: RandomForestClassifier = joblib.load(paths.segmentation_model_path)
    policy_model: DecisionTreeClassifier = joblib.load(paths.policy_model_path)

    image = load_rgb_image(record.image_path)
    gt_label_map = encode_mask(load_rgb_mask(record.mask_path))
    pred_label_map = _predict_segmentation(segmentation_model, image)
    pred_metrics = compute_scene_metrics(pred_label_map)

    action = str(policy_model.predict([metrics_to_vector(pred_metrics)])[0])
    probabilities = policy_model.predict_proba([metrics_to_vector(pred_metrics)])[0]
    confidence = float(probabilities.max())
    explanation = build_action_explanation(action, pred_metrics)

    output_path = paths.outputs_dir / f"demo_{record.sample_id}.png"
    save_demo_card(
        output_path=output_path,
        image=image,
        gt_mask_rgb=decode_mask(gt_label_map),
        pred_mask_rgb=decode_mask(pred_label_map),
        action=action,
        confidence=confidence,
        explanation=explanation,
    )

    return {
        "sample_id": record.sample_id,
        "split": record.split,
        "action": action,
        "confidence": confidence,
        "explanation": explanation,
        "metrics": pred_metrics,
        "image": image,
        "gt_mask_rgb": decode_mask(gt_label_map),
        "pred_mask_rgb": decode_mask(pred_label_map),
        "demo_card_path": output_path,
    }
