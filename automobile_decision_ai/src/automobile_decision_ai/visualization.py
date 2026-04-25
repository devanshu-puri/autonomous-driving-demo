"""Visualization helpers for CLI and web demos."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def overlay_mask(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = image.astype(np.float32)
    overlay = mask_rgb.astype(np.float32)
    blended = (base * (1.0 - alpha)) + (overlay * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def array_to_png_bytes(image_array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    return buffer.getvalue()


def save_demo_card(
    output_path: Path,
    image: np.ndarray,
    gt_mask_rgb: np.ndarray,
    pred_mask_rgb: np.ndarray,
    action: str,
    confidence: float,
    explanation: list[str],
) -> None:
    original = Image.fromarray(image)
    gt_overlay = Image.fromarray(overlay_mask(image, gt_mask_rgb))
    pred_overlay = Image.fromarray(overlay_mask(image, pred_mask_rgb))

    panel_width, panel_height = original.size
    text_height = 150
    canvas = Image.new(
        "RGB",
        (panel_width * 3, panel_height + text_height),
        color=(248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)

    canvas.paste(original, (0, 0))
    canvas.paste(gt_overlay, (panel_width, 0))
    canvas.paste(pred_overlay, (panel_width * 2, 0))

    draw.text((10, 10), "Original Image", fill=(20, 20, 20))
    draw.text((panel_width + 10, 10), "Ground Truth Scene", fill=(20, 20, 20))
    draw.text((panel_width * 2 + 10, 10), "AI Predicted Scene", fill=(20, 20, 20))

    text_y = panel_height + 12
    draw.text(
        (10, text_y),
        f"Driving Action: {action} | Confidence: {confidence:.1%}",
        fill=(10, 10, 10),
    )
    for index, line in enumerate(explanation, start=1):
        draw.text((10, text_y + 24 * index), f"- {line}", fill=(30, 30, 30))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
