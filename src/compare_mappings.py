from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset import FreiHandLandmarkDataset, draw_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sammenlign to annotation-mappinger paa samme FreiHAND-bilde."
    )
    parser.add_argument("--index", type=int, default=25, help="Hvilket RGB-bilde som skal sjekkes.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "mapping_comparison.jpg",
    )
    return parser.parse_args()


def render_preview(image_index: int, image_size: int, mapping_mode: str) -> tuple[np.ndarray, dict[str, object]]:
    dataset = FreiHandLandmarkDataset(
        image_size=image_size,
        normalize=True,
        return_tensors=False,
        crop_hand=False,
        mapping_mode=mapping_mode,
    )
    sample = dataset.get_sample(image_index)
    image_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)
    preview = draw_landmarks(
        image_uint8,
        sample.landmarks_2d,
        connections=dataset.selected_connections,
    )
    cv2.putText(
        preview,
        f"{mapping_mode} | ann={sample.annotation_index}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview, dataset.summary()


def main() -> None:
    args = parse_args()

    grouped_preview, grouped_summary = render_preview(args.index, args.image_size, "grouped")
    interleaved_preview, interleaved_summary = render_preview(args.index, args.image_size, "interleaved")

    combined = np.concatenate([grouped_preview, interleaved_preview], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print("Saved comparison to:", args.output)
    print("Grouped summary:", grouped_summary)
    print("Interleaved summary:", interleaved_summary)


if __name__ == "__main__":
    main()
