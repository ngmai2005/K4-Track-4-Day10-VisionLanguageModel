"""Unit tests for the Bonus E5 GRPO IoU reward function.

Tests 5 key conditions:
1. Perfect match (1.0)
2. Corrupt / invalid JSON (0.0)
3. Wrong label (0.2)
4. Extra box / denominator penalty (0.6)
5. Both ground truth and prediction empty (1.0)
"""

import json
import pytest

from lab_helpers import compute_reward


def test_reward_perfect_match():
    gt = [{"label": "pothole", "bbox": (400.0, 400.0, 600.0, 600.0)}]
    pred_str = json.dumps([{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}])
    reward = compute_reward(pred_str, gt)
    assert reward == pytest.approx(1.0)


def test_reward_corrupt_json():
    gt = [{"label": "pothole", "bbox": (400.0, 400.0, 600.0, 600.0)}]
    reward = compute_reward("I found a pothole near the center.", gt)
    assert reward == 0.0


def test_reward_wrong_label():
    gt = [{"label": "pothole", "bbox": (400.0, 400.0, 600.0, 600.0)}]
    pred_str = json.dumps([{"label": "crack", "bbox_2d": [400, 400, 600, 600]}])
    reward = compute_reward(pred_str, gt)
    assert reward == pytest.approx(0.2)


def test_reward_extra_box_penalty():
    gt = [{"label": "pothole", "bbox": (400.0, 400.0, 600.0, 600.0)}]
    pred_str = json.dumps([
        {"label": "pothole", "bbox_2d": [400, 400, 600, 600]},
        {"label": "pothole", "bbox_2d": [10, 10, 50, 50]},
    ])
    reward = compute_reward(pred_str, gt)
    # 0.2 + 0.8 * (1.0 / 2) = 0.6
    assert reward == pytest.approx(0.6)


def test_reward_both_empty():
    gt = []
    reward = compute_reward("[]", gt)
    assert reward == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
