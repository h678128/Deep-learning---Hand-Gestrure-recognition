from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset import DEFAULT_MAPPING_MODE, FreiHandLandmarkDataset, draw_landmarks


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
        default=None,
        help="Valgfri output-fil. Hvis ikke satt, lagres preview i en inspect-mappe.",
    )
    parser.add_argument(
        "--mapping-mode",
        choices=["grouped", "interleaved"],
        default=DEFAULT_MAPPING_MODE,
        help="Hvordan RGB-bilder kobles til annotasjoner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = FreiHandLandmarkDataset(
        image_size=args.image_size,
        normalize=True,
        return_tensors=False,
        crop_hand=False,
        mapping_mode=args.mapping_mode,
    )
    sample = dataset.get_sample(args.index)

    image_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)
    preview_rgb = draw_landmarks(
        image_uint8,
        sample.landmarks_2d,
        connections=dataset.selected_connections,
    )

    output_path = args.output or (
        Path("outputs")
        / "dataset_checks"
        / f"{dataset.num_landmarks}pts_{args.mapping_mode}"
        / f"landmark_{args.index:05d}.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), preview_bgr)

    summary = dataset.summary()
    print("Dataset summary:", summary)
    print("Saved preview to:", output_path)
    print("Image path:", sample.image_path)
    print("Image index:", sample.image_index)
    print("Annotation index:", sample.annotation_index)


if __name__ == "__main__":
    main()
