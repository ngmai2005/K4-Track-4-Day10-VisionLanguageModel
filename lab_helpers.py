"""Student-owned helpers for Qwen3-VL road-damage grounding.

This module deliberately does not change the provided ``scripts/`` package.
Import these functions from ``lab_day10.ipynb`` after preparing ``data/``.
"""

from __future__ import annotations

import json
from typing import Iterable


PROMPT = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)


def yolo_to_box(label: str, yolo: Iterable[float]) -> dict:
    """Convert normalized YOLO ``xc,yc,w,h`` to the rubric's integer norm1000 box."""
    xc, yc, width, height = (float(v) for v in yolo)

    def clamp(value: float) -> int:
        return int(round(max(0.0, min(1000.0, value))))

    x1 = clamp((xc - width / 2) * 1000)
    y1 = clamp((yc - height / 2) * 1000)
    x2 = clamp((xc + width / 2) * 1000)
    y2 = clamp((yc + height / 2) * 1000)
    # A valid target needs positive extent even for an annotation on an edge.
    if x2 <= x1:
        x1, x2 = min(x1, 999), min(1000, max(x1 + 1, x2))
    if y2 <= y1:
        y1, y2 = min(y1, 999), min(1000, max(y1 + 1, y2))
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects: Iterable[dict]) -> str:
    """Serialize an image target with deterministic top-to-bottom ordering."""
    boxes = [yolo_to_box(item["label"], item["yolo"]) for item in objects]
    boxes.sort(key=lambda box: (box["bbox_2d"][1], box["bbox_2d"][0]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def compute_reward(completion: str, gt_boxes: list[dict]) -> float:
    """Compute verifiable IoU reward for GRPO (Bonus E5).

    Returns:
        1.0 for perfect match or both empty, 0.0 for invalid JSON, or
        0.2 + 0.8 * (total_matched_iou / max(len(gt), len(pred), 1)).
    """
    from scripts.score import iou_xyxy, parse_output

    pred = parse_output(completion)
    if pred is None:
        return 0.0  # JSON hỏng
    if not gt_boxes and not pred:
        return 1.0  # Cả hai cùng rỗng

    total_iou = 0.0
    all_labels = {g["label"] for g in gt_boxes} | {p["label"] for p in pred}
    for lab in all_labels:
        g = [x["bbox"] for x in gt_boxes if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(
            ((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)),
            reverse=True,
        )
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < 0.5:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total_iou += v
    denom = max(len(gt_boxes), len(pred), 1)
    return 0.2 + 0.8 * (total_iou / denom)

