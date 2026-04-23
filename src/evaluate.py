from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import (
    DEFAULT_LANDMARK_INDICES,
    FreiHandLandmarkDataset,
    decode_heatmaps,
)
from model import create_heatmap_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kjor evaluering paa et enkelt bilde og lagre preview med prediksjon."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("modell") / "landmark_heatmap11_best.pt",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Valgfri output-fil. Hvis ikke satt, lagres preview i en modellmappe.",
    )
    return parser.parse_args()


def default_output_path(checkpoint_path: Path, image_index: int) -> Path:
    model_name = checkpoint_path.stem
    return Path("outputs") / "model_runs" / model_name / "evaluate" / f"pred_{image_index:05d}.jpg"


def draw_custom_landmarks(
    image: np.ndarray,
    landmarks_xy: np.ndarray,
    connections: list[tuple[int, int]],
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
) -> np.ndarray:
    canvas = image.copy()
    for start_idx, end_idx in connections:
        start_point = tuple(np.round(landmarks_xy[start_idx]).astype(int))
        end_point = tuple(np.round(landmarks_xy[end_idx]).astype(int))
        cv2.line(canvas, start_point, end_point, line_color, 2)

    for point in landmarks_xy:
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(canvas, point_xy, 4, point_color, -1)

    return canvas


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    selected_landmark_indices = (
        checkpoint.get("selected_landmark_indices") or DEFAULT_LANDMARK_INDICES
    )
    dataset = FreiHandLandmarkDataset(
        image_size=checkpoint.get("image_size", 224),
        heatmap_size=checkpoint.get("heatmap_size", 56),
        heatmap_sigma=checkpoint.get("heatmap_sigma", 2.0),
        normalize=True,
        return_tensors=True,
        crop_hand=checkpoint.get("crop_hand", False),
        crop_padding=checkpoint.get("crop_padding", 0.25),
        selected_landmark_indices=selected_landmark_indices,
    )
    sample = dataset[args.index]

    model = create_heatmap_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_heatmaps = model(image_tensor)
        prediction = decode_heatmaps(predicted_heatmaps.cpu(), image_size=dataset.image_size)[0].numpy()

    target = sample["landmarks_2d"].cpu().numpy()
    image_rgb = (
        sample["image"].permute(1, 2, 0).cpu().numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)

    target_preview = draw_custom_landmarks(
        image_rgb,
        target,
        connections=dataset.selected_connections,
        line_color=(0, 220, 120),
        point_color=(255, 90, 90),
    )
    prediction_preview = draw_custom_landmarks(
        image_rgb,
        prediction,
        connections=dataset.selected_connections,
        line_color=(255, 200, 0),
        point_color=(50, 120, 255),
    )

    output_path = args.output if args.output is not None else default_output_path(args.checkpoint, args.index)
    combined = np.concatenate([target_preview, prediction_preview], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    pixel_error = np.linalg.norm(prediction - target, axis=1).mean()
    print("Checkpoint:", args.checkpoint)
    print("Saved preview to:", output_path)
    print("Image path:", sample["image_path"])
    print("Mean pixel error:", round(float(pixel_error), 2))


if __name__ == "__main__":
    main()
