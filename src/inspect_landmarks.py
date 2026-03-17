from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset import FreiHandLandmarkDataset, draw_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualiser FreiHAND-landmarks over et RGB-bilde."
    )
    parser.add_argument("--index", type=int, default=0, help="Hvilket RGB-bilde som skal vises.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resizet bildestorrelse brukt i visualiseringen.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "landmark_preview_00000000.jpg",
        help="Hvor preview-bildet skal lagres.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = FreiHandLandmarkDataset(
        image_size=args.image_size,
        normalize=True,
        return_tensors=False,
    )
    sample = dataset.get_sample(args.index)

    image_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)
    preview_rgb = draw_landmarks(image_uint8, sample.landmarks_2d)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.output), preview_bgr)

    summary = dataset.summary()
    print("Dataset summary:", summary)
    print("Saved preview to:", args.output)
    print("Image path:", sample.image_path)
    print("Image index:", sample.image_index)
    print("Annotation index:", sample.annotation_index)


if __name__ == "__main__":
    main()
