"""Eksporter den trente heatmap-modellen til ONNX-format for nettleserbruk."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import create_heatmap_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eksporter modell til ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("modell") / "landmark_heatmap11_best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("..") / "docs" / "model.onnx",
    )
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("image_size", args.image_size))

    model = create_heatmap_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["heatmaps"],
        dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    size_mb = args.output.stat().st_size / 1e6
    print(f"Eksportert til {args.output}  ({size_mb:.1f} MB)")
    print(f"image_size brukt: {image_size}")


if __name__ == "__main__":
    main()
