"""
Utility script for running the pretrained YOLOv8 human detector on a single image.

Reference: README instructions describe using `best.pt` weights produced from the
Roboflow dataset; this script wraps that workflow with configurable thresholds
so detections can be tightened (higher `conf` / `iou`) when bounding boxes feel
loose.
"""

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def tighten_box(xyxy, shrink_ratio: float):
    """Optionally shrink the detected box to compensate loose annotations."""
    if shrink_ratio <= 0:
        return map(int, xyxy)

    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    dx = w * shrink_ratio
    dy = h * shrink_ratio
    return map(
        int,
        (
            x1 + dx,
            y1 + dy,
            x2 - dx,
            y2 - dy,
        ),
    )


def run_detection(args):
    model = YOLO(args.weights)
    results = model(
        args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save_image,
    )

    # Save optional tightly cropped outputs alongside original image.
    if args.export_crops:
        img = Image.open(args.source)
        out_dir = Path(args.crop_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, box in enumerate(results[0].boxes, start=1):
            # Only keep person class (class 0 for this model).
            if int(box.cls[0]) != 0:
                continue
            crop_coords = tighten_box(box.xyxy[0], args.shrink_ratio)
            crop = img.crop(tuple(crop_coords))
            crop.save(out_dir / f"person_{idx:02d}.jpg")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run YOLOv8 human detection on one image.")
    parser.add_argument("--weights", default="best.pt", help="Path to YOLO weights.")
    parser.add_argument(
        "--source",
        default="person4.jpg",
        help="Path to image to run inference on.",
    )
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold.")
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="Save annotated image (stored under runs/detect).",
    )
    parser.add_argument(
        "--export-crops",
        action="store_true",
        help="Export tightly cropped detections to --crop-dir.",
    )
    parser.add_argument(
        "--crop-dir",
        default="runs/salidas",
        help="Directory where cropped detections are stored.",
    )
    parser.add_argument(
        "--shrink-ratio",
        type=float,
        default=0.02,
        help="Shrink percentage (0-0.45) applied to each side for tighter boxes.",
    )
    args = parser.parse_args()
    run_detection(args)
