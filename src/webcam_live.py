from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import DEFAULT_LANDMARK_INDICES, decode_heatmaps, infer_connections
from gesture import GestureController, classify_gesture, GESTURE_LABELS
from model import create_heatmap_model_from_checkpoint

try:
    import mediapipe as mp
except ImportError:
    mp = None


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
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument(
        "--smooth-factor",
        type=float,
        default=0.6,
        help="Hoyere verdi gir roligere punkter.",
    )
    parser.add_argument(
        "--use-mediapipe",
        action="store_true",
        help="Bruk MediaPipe-handdeteksjon til aa croppe haanden for modellen.",
    )
    parser.add_argument(
        "--mediapipe-padding",
        type=float,
        default=0.25,
        help="Ekstra padding rundt hand-boksen fra MediaPipe.",
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


def create_mediapipe_hands_detector():
    if mp is None:
        return None

    detector_kwargs = {
        "static_image_mode": False,
        "max_num_hands": 1,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }

    hands_factory = None

    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        hands_module = getattr(solutions, "hands", None)
        if hands_module is not None:
            hands_factory = getattr(hands_module, "Hands", None)

    if hands_factory is None:
        try:
            hands_module = importlib.import_module("mediapipe.python.solutions.hands")
            hands_factory = getattr(hands_module, "Hands", None)
        except Exception:
            hands_factory = None

    if hands_factory is None:
        print("MediaPipe is installed, but the Hands API is unavailable. Falling back to full-frame inference.")
        return None

    try:
        return hands_factory(**detector_kwargs)
    except Exception as exc:
        print(f"Failed to initialize MediaPipe Hands ({exc}). Falling back to full-frame inference.")
        return None


def compute_roi_from_mediapipe(
    frame_bgr: np.ndarray,
    hands_detector,
    padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]
    xs = np.array([landmark.x * frame_w for landmark in hand_landmarks.landmark], dtype=np.float32)
    ys = np.array([landmark.y * frame_h for landmark in hand_landmarks.landmark], dtype=np.float32)

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    box_size = max(max_x - min_x, max_y - min_y)
    half_size = max(24.0, box_size * (0.5 + padding_ratio))

    x1 = max(0, int(np.floor(center_x - half_size)))
    y1 = max(0, int(np.floor(center_y - half_size)))
    x2 = min(frame_w, int(np.ceil(center_x + half_size)))
    y2 = min(frame_h, int(np.ceil(center_y + half_size)))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


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
    return canvas


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = checkpoint.get("image_size", 224)
    num_landmarks = checkpoint.get("num_landmarks", 6)
    selected_landmark_indices = checkpoint.get(
        "selected_landmark_indices",
        list(DEFAULT_LANDMARK_INDICES),
    )
    connections = infer_connections(selected_landmark_indices)

    model = create_heatmap_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    use_mediapipe = args.use_mediapipe and (mp is not None)
    hands_detector = None
    if use_mediapipe:
        hands_detector = create_mediapipe_hands_detector()

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera_index}.")

    ret, probe = capture.read()
    frame_h, frame_w = probe.shape[:2] if ret else (1080, 1920)
    controller = GestureController(frame_w=frame_w, frame_h=frame_h)

    gesture_active = False
    smoothed_landmarks: np.ndarray | None = None

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            roi = None
            if hands_detector is not None:
                roi = compute_roi_from_mediapipe(
                    frame_bgr,
                    hands_detector,
                    padding_ratio=args.mediapipe_padding,
                )

            if roi is None:
                smoothed_landmarks = None
                controller.reset()

            frame_for_model = frame_bgr
            if roi is not None:
                x1, y1, x2, y2 = roi
                frame_for_model = frame_bgr[y1:y2, x1:x2]

            input_tensor = preprocess_frame(frame_for_model, image_size).to(device)

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
                target_hw=frame_for_model.shape[:2],
            )
            if roi is not None:
                predicted_landmarks[:, 0] += x1
                predicted_landmarks[:, 1] += y1

            if smoothed_landmarks is None:
                smoothed_landmarks = predicted_landmarks
            else:
                alpha = np.clip(args.smooth_factor, 0.0, 0.99)
                smoothed_landmarks = alpha * smoothed_landmarks + (1.0 - alpha) * predicted_landmarks

            gesture = classify_gesture(smoothed_landmarks)
            if gesture_active:
                controller.process(smoothed_landmarks, gesture)

            preview = draw_prediction(
                frame_bgr,
                smoothed_landmarks,
                connections,
                confidences,
                confidence_threshold=args.confidence_threshold,
            )
            if roi is not None:
                cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 210, 80), 2)

            if gesture_active:
                status_text = "AKTIV  [space = av]"
                status_color = (0, 220, 120)
                label = GESTURE_LABELS.get(gesture, "")
                if label:
                    cv2.putText(
                        preview,
                        label,
                        (12, preview.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 220, 120),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                status_text = "PAUSE  [space = start]"
                status_color = (0, 100, 255)

            cv2.putText(
                preview,
                status_text,
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Landmark Live", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                gesture_active = not gesture_active
                controller.reset()
    finally:
        capture.release()
        if hands_detector is not None:
            hands_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
