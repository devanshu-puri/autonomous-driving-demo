from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automobile_decision_ai.constants import CLASS_TO_INDEX
from automobile_decision_ai.dataset import decode_mask, encode_mask
from automobile_decision_ai.features import compute_scene_metrics, expert_action_from_metrics


class PipelineTests(unittest.TestCase):
    def test_mask_encode_decode_round_trip(self) -> None:
        label_map = np.array(
            [
                [CLASS_TO_INDEX["road"], CLASS_TO_INDEX["lane_marking"]],
                [CLASS_TO_INDEX["movable"], CLASS_TO_INDEX["undrivable"]],
            ],
            dtype=np.int16,
        )
        rgb_mask = decode_mask(label_map)
        restored = encode_mask(rgb_mask)
        np.testing.assert_array_equal(label_map, restored)

    def test_stop_action_when_front_is_blocked(self) -> None:
        label_map = np.full((240, 320), CLASS_TO_INDEX["road"], dtype=np.int16)
        label_map[120:230, 118:202] = CLASS_TO_INDEX["movable"]

        metrics = compute_scene_metrics(label_map)
        action = expert_action_from_metrics(metrics)

        self.assertEqual(action, "STOP")

    def test_keep_straight_or_caution_when_scene_is_clear(self) -> None:
        label_map = np.full((240, 320), CLASS_TO_INDEX["road"], dtype=np.int16)
        label_map[130:220, 155:165] = CLASS_TO_INDEX["lane_marking"]

        metrics = compute_scene_metrics(label_map)
        action = expert_action_from_metrics(metrics)

        self.assertIn(action, {"KEEP_STRAIGHT", "CAUTION"})


if __name__ == "__main__":
    unittest.main()
