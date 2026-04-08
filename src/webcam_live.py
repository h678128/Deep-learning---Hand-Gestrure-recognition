from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import decode_heatmaps, infer_connections
from model import create_heatmap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kjor live webcam-demo med den trente heatmap-modellen."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("modell") / "landmark_heatmap11_best.pt",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument(
        "--smooth-factor",
        type=float,
        default=0.8,
        help="Hoyere verdi gir roligere punkter.",
    )
    parser.add_argument(
        "--deadzone-px",
        type=float,
        default=4.0,
        help="Ignorer sma bevegelser under denne terskelen.",
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=8,
        help="Hvor lenge vi holder siste gode punkt naar confidence er lav.",
    )
    return parser.parse_args()


def preprocess_frame(frame_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size))
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


def smooth_landmarks(
    previous_landmarks: np.ndarray | None,
    predicted_landmarks: np.ndarray,
    confidences: np.ndarray,
    missed_counts: np.ndarray,
    base_smooth_factor: float,
    confidence_threshold: float,
    deadzone_px: float,
    max_missed_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if previous_landmarks is None:
        return predicted_landmarks.copy(), np.zeros_like(missed_counts)

    smoothed = previous_landmarks.copy()
    updated_missed_counts = missed_counts.copy()
    base_alpha = float(np.clip(base_smooth_factor, 0.0, 0.98))

    for index, predicted_point in enumerate(predicted_landmarks):
        confidence = float(confidences[index])

        if confidence >= confidence_threshold:
            updated_missed_counts[index] = 0
            adaptive_alpha = min(0.98, max(0.0, base_alpha + (1.0 - confidence) * 0.15))
            candidate_point = (
                adaptive_alpha * smoothed[index] + (1.0 - adaptive_alpha) * predicted_point
            )

            if np.linalg.norm(candidate_point - smoothed[index]) < deadzone_px:
                continue

            smoothed[index] = candidate_point
            continue

        updated_missed_counts[index] += 1
        if updated_missed_counts[index] > max_missed_frames:
            smoothed[index] = predicted_point

    return smoothed, updated_missed_counts


def draw_prediction(
    frame_bgr: np.ndarray,
    landmarks_xy: np.ndarray,
    connections: list[tuple[int, int]],
    confidences: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    canvas = frame_bgr.copy()

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
        "Press q to quit",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"avg conf: {confidences.mean():.2f}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
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
    num_landmarks = checkpoint.get("num_landmarks", 6)
    selected_landmark_indices = checkpoint.get("selected_landmark_indices")
    connections = infer_connections(selected_landmark_indices)

    model = create_heatmap_model(num_landmarks=num_landmarks).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera_index}.")

    smoothed_landmarks: np.ndarray | None = None
    missed_counts = np.zeros(num_landmarks, dtype=np.int32)

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            input_tensor = preprocess_frame(frame_bgr, image_size).to(device)

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
                target_hw=frame_bgr.shape[:2],
            )

            smoothed_landmarks, missed_counts = smooth_landmarks(
                previous_landmarks=smoothed_landmarks,
                predicted_landmarks=predicted_landmarks,
                confidences=confidences,
                missed_counts=missed_counts,
                base_smooth_factor=args.smooth_factor,
                confidence_threshold=args.confidence_threshold,
                deadzone_px=args.deadzone_px,
                max_missed_frames=args.max_missed_frames,
            )

            preview = draw_prediction(
                frame_bgr,
                smoothed_landmarks,
                connections,
                confidences,
                confidence_threshold=args.confidence_threshold,
            )
            cv2.imshow("Hand Landmark Live", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
