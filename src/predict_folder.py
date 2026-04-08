from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import infer_connections, decode_heatmaps
from model import create_heatmap_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kjor landmark-modellen paa alle bilder i en mappe."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("modell") / "landmark_heatmap11_interleaved_20k.pt",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Valgfri output-mappe. Hvis ikke satt, lagres resultatene under outputs/model_runs.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    return parser.parse_args()


def default_output_dir(checkpoint_path: Path, input_dir: Path) -> Path:
    model_name = checkpoint_path.stem
    return Path("outputs") / "model_runs" / model_name / input_dir.name


def preprocess_image(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_size, image_size))
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor


def upscale_landmarks(
    landmarks_xy: np.ndarray,
    source_size: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_hw
    scaled = landmarks_xy.copy()
    scaled[:, 0] *= target_w / float(source_size)
    scaled[:, 1] *= target_h / float(source_size)
    return scaled


def heatmap_confidences(heatmaps: torch.Tensor) -> np.ndarray:
    confidences = heatmaps.view(heatmaps.shape[0], heatmaps.shape[1], -1).amax(dim=-1)
    return confidences[0].cpu().numpy()


def draw_prediction(
    image_bgr: np.ndarray,
    landmarks_xy: np.ndarray,
    connections: list[tuple[int, int]],
    confidences: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    canvas = image_bgr.copy()
    visible = confidences >= confidence_threshold

    for start_idx, end_idx in connections:
        if not (visible[start_idx] and visible[end_idx]):
            continue
        start_point = tuple(np.round(landmarks_xy[start_idx]).astype(int))
        end_point = tuple(np.round(landmarks_xy[end_idx]).astype(int))
        cv2.line(canvas, start_point, end_point, (0, 220, 120), 2)

    for index, point in enumerate(landmarks_xy):
        point_xy = tuple(np.round(point).astype(int))
        color = (0, 220, 120) if visible[index] else (0, 140, 255)
        cv2.circle(canvas, point_xy, 5, color, -1)

    cv2.putText(
        canvas,
        f"avg conf: {confidences.mean():.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = checkpoint.get("image_size", 224)
    num_landmarks = checkpoint.get("num_landmarks", 11)
    selected_landmark_indices = checkpoint.get("selected_landmark_indices")
    connections = infer_connections(selected_landmark_indices)

    model = create_heatmap_model(num_landmarks=num_landmarks).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = args.output_dir or default_output_dir(args.checkpoint, args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No image files found in {args.input_dir}")

    summary_rows: list[dict[str, object]] = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        input_tensor = preprocess_image(image_bgr, image_size).to(device)
        with torch.no_grad():
            predicted_heatmaps = model(input_tensor)
            predicted_landmarks = decode_heatmaps(
                predicted_heatmaps.cpu(),
                image_size=image_size,
            )[0].numpy()
            confidences = heatmap_confidences(predicted_heatmaps.cpu())

        predicted_landmarks = upscale_landmarks(
            predicted_landmarks,
            source_size=image_size,
            target_hw=image_bgr.shape[:2],
        )

        preview = draw_prediction(
            image_bgr,
            predicted_landmarks,
            connections,
            confidences,
            confidence_threshold=args.confidence_threshold,
        )

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), preview)
        summary_rows.append(
            {
                "image": image_path.name,
                "avg_confidence": round(float(confidences.mean()), 4),
                "min_confidence": round(float(confidences.min()), 4),
                "max_confidence": round(float(confidences.max()), 4),
            }
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "avg_confidence", "min_confidence", "max_confidence"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Checkpoint:", args.checkpoint)
    print("Input dir:", args.input_dir)
    print("Output dir:", output_dir)
    print("Saved summary to:", summary_path)
    print("Processed images:", len(summary_rows))


if __name__ == "__main__":
    main()
