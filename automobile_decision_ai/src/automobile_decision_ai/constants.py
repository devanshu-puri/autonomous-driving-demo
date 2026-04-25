"""Project constants."""

from __future__ import annotations

IMAGE_SIZE = (320, 240)

MASK_NAME_TO_COLOR = {
    "road": (64, 32, 32),
    "lane_marking": (255, 0, 0),
    "undrivable": (128, 128, 96),
    "movable": (0, 255, 102),
    "ego_vehicle": (204, 0, 255),
}

CLASS_NAMES = tuple(MASK_NAME_TO_COLOR.keys())
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
COLOR_TO_INDEX = {
    color: CLASS_TO_INDEX[name] for name, color in MASK_NAME_TO_COLOR.items()
}
INDEX_TO_COLOR = {
    CLASS_TO_INDEX[name]: color for name, color in MASK_NAME_TO_COLOR.items()
}

ACTION_LABELS = (
    "KEEP_STRAIGHT",
    "SLOW_DOWN",
    "STOP",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "CAUTION",
)

ACTION_DESCRIPTIONS = {
    "KEEP_STRAIGHT": "Road ahead looks sufficiently clear, so continue normally.",
    "SLOW_DOWN": "A possible obstacle or cluttered path is ahead, so reduce speed.",
    "STOP": "The front path is heavily blocked, so stopping is safest.",
    "MOVE_LEFT": "The right side looks busier than the left, so move left carefully.",
    "MOVE_RIGHT": "The left side looks busier than the right, so move right carefully.",
    "CAUTION": "Lane guidance or drivable confidence is weak, so proceed carefully.",
}

SCENE_FEATURE_NAMES = (
    "front_drivable_ratio",
    "front_lane_ratio",
    "front_undrivable_ratio",
    "front_movable_ratio",
    "left_drivable_ratio",
    "left_movable_ratio",
    "right_drivable_ratio",
    "right_movable_ratio",
    "overall_movable_ratio",
    "overall_undrivable_ratio",
)
