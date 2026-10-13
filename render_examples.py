"""Render six or more ground-truth/prediction examples for the lab report."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import read_jsonl
from scripts.score import parse_output


def gt_boxes(entry: dict) -> list[dict]:
    boxes = []
    for item in entry["objects"]:
        xc, yc, w, h = item["yolo"]
        boxes.append({"label": item["label"], "bbox": [
            (xc - w / 2) * 1000, (yc - h / 2) * 1000,
            (xc + w / 2) * 1000, (yc + h / 2) * 1000,
        ]})
    return boxes


def draw_boxes(image: Image.Image, boxes: list[dict], color: str, prefix: str) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        xy = [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]
        draw.rectangle(xy, outline=color, width=3)
        draw.text((xy[0] + 2, max(0, xy[1] - 14)), f"{prefix}:{box['label']}", fill=color)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="work/eN.jsonl")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--out", default="report/examples")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    entries = read_jsonl(root / "data/splits" / f"{args.split}.jsonl")
    predictions = {row["image_id"]: row["raw"] for row in read_jsonl(args.pred)}
    output = root / args.out
    output.mkdir(parents=True, exist_ok=True)
    # Prefer a mix of valid/invalid predictions so the report can discuss errors.
    selected = sorted(entries, key=lambda e: parse_output(predictions.get(e["image_id"], "")) is None)
    for index, entry in enumerate(selected[:args.count], 1):
        image = Image.open(root / "data" / entry["image_path"]).convert("RGB")
        draw_boxes(image, gt_boxes(entry), "#00ff00", "GT")
        parsed = parse_output(predictions.get(entry["image_id"], "")) or []
        draw_boxes(image, parsed, "#ff3030", "PRED")
        image.save(output / f"{args.split}_{index:02d}_{entry['image_id']}.png")


if __name__ == "__main__":
    main()
