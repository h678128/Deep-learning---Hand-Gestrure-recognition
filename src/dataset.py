from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "trene"
TRAINING_RGB_PATH = DATA_ROOT / "training" / "rgb"
TRAINING_XYZ_PATH = DATA_ROOT / "training_xyz.json"
TRAINING_K_PATH = DATA_ROOT / "training_K.json"

FULL_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
DEFAULT_LANDMARK_INDICES = (0, 4, 8, 12, 16, 20)
DEFAULT_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
]


@dataclass
class FreiHandSample:
    image: np.ndarray
    landmarks_2d: np.ndarray
    landmarks_3d: np.ndarray
    camera_matrix: np.ndarray
    image_path: Path
    image_index: int
    annotation_index: int
    crop_box: tuple[int, int, int, int]


def _load_json_array(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        return np.asarray(json.load(file), dtype=np.float32)


def infer_images_per_annotation(image_count: int, annotation_count: int) -> int:
    if annotation_count == 0:
        raise ValueError("No landmark annotations found.")

    if image_count % annotation_count != 0:
        raise ValueError(
            f"Image count ({image_count}) is not divisible by annotation count ({annotation_count})."
        )

    return image_count // annotation_count


def project_points(points_xyz: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    projected = points_xyz @ camera_matrix.T
    depth = np.clip(projected[:, 2:3], a_min=1e-6, a_max=None)
    return projected[:, :2] / depth


def resize_landmarks(
    landmarks_xy: np.ndarray,
    original_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    original_h, original_w = original_hw
    target_h, target_w = target_hw

    scaled = landmarks_xy.copy()
    scaled[:, 0] *= target_w / original_w
    scaled[:, 1] *= target_h / original_h
    return scaled


def infer_connections(num_landmarks: int) -> list[tuple[int, int]]:
    if num_landmarks == len(DEFAULT_LANDMARK_INDICES):
        return DEFAULT_CONNECTIONS
    return FULL_HAND_CONNECTIONS


def draw_landmarks(
    image: np.ndarray,
    landmarks_xy: np.ndarray,
    connections: Optional[Sequence[tuple[int, int]]] = None,
    point_radius: int = 3,
) -> np.ndarray:
    canvas = image.copy()
    active_connections = list(connections) if connections is not None else infer_connections(len(landmarks_xy))

    for start_idx, end_idx in active_connections:
        start_point = tuple(np.round(landmarks_xy[start_idx]).astype(int))
        end_point = tuple(np.round(landmarks_xy[end_idx]).astype(int))
        cv2.line(canvas, start_point, end_point, (0, 220, 120), 2)

    for point in landmarks_xy:
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(canvas, point_xy, point_radius, (255, 90, 90), -1)

    return canvas


def compute_hand_crop_box(
    landmarks_xy: np.ndarray,
    image_hw: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    image_h, image_w = image_hw
    min_xy = landmarks_xy.min(axis=0)
    max_xy = landmarks_xy.max(axis=0)

    center_x = float((min_xy[0] + max_xy[0]) / 2.0)
    center_y = float((min_xy[1] + max_xy[1]) / 2.0)
    box_size = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    half_size = max(16.0, box_size * (0.5 + padding_ratio))

    x1 = int(np.floor(center_x - half_size))
    y1 = int(np.floor(center_y - half_size))
    x2 = int(np.ceil(center_x + half_size))
    y2 = int(np.ceil(center_y + half_size))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_w, x2)
    y2 = min(image_h, y2)

    if x2 <= x1:
        x2 = min(image_w, x1 + 1)
    if y2 <= y1:
        y2 = min(image_h, y1 + 1)

    return x1, y1, x2, y2


def crop_image_and_landmarks(
    image_rgb: np.ndarray,
    landmarks_xy: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = crop_box
    cropped_image = image_rgb[y1:y2, x1:x2]
    cropped_landmarks = landmarks_xy.copy()
    cropped_landmarks[:, 0] -= x1
    cropped_landmarks[:, 1] -= y1
    return cropped_image, cropped_landmarks


def generate_heatmaps(
    landmarks_xy: np.ndarray,
    image_size: int,
    heatmap_size: int,
    sigma: float,
) -> np.ndarray:
    heatmaps = np.zeros((len(landmarks_xy), heatmap_size, heatmap_size), dtype=np.float32)
    scale = float(heatmap_size) / float(image_size)

    xs = np.arange(heatmap_size, dtype=np.float32)
    ys = np.arange(heatmap_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for index, point in enumerate(landmarks_xy):
        center_x = point[0] * scale
        center_y = point[1] * scale
        squared_distance = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
        heatmaps[index] = np.exp(-squared_distance / (2.0 * sigma**2))

    return heatmaps


def decode_heatmaps(heatmaps: torch.Tensor, image_size: int) -> torch.Tensor:
    if heatmaps.ndim != 4:
        raise ValueError("Expected heatmaps with shape [batch, keypoints, height, width].")

    batch_size, num_keypoints, height, width = heatmaps.shape
    flat_heatmaps = heatmaps.view(batch_size, num_keypoints, -1)
    flat_indices = flat_heatmaps.argmax(dim=-1)

    y_indices = torch.div(flat_indices, width, rounding_mode="floor")
    x_indices = flat_indices % width

    scale_x = float(image_size) / float(width)
    scale_y = float(image_size) / float(height)

    x_coords = (x_indices.float() + 0.5) * scale_x
    y_coords = (y_indices.float() + 0.5) * scale_y
    return torch.stack([x_coords, y_coords], dim=-1)


class FreiHandLandmarkDataset(Dataset):
    def __init__(
        self,
        image_size: int = 224,
        heatmap_size: int = 56,
        heatmap_sigma: float = 2.0,
        normalize: bool = True,
        return_tensors: bool = True,
        crop_hand: bool = False,
        crop_padding: float = 0.25,
        selected_landmark_indices: Sequence[int] = DEFAULT_LANDMARK_INDICES,
    ) -> None:
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.normalize = normalize
        self.return_tensors = return_tensors
        self.crop_hand = crop_hand
        self.crop_padding = crop_padding
        self.selected_landmark_indices = tuple(selected_landmark_indices)
        self.selected_connections = infer_connections(len(self.selected_landmark_indices))
        self.num_landmarks = len(self.selected_landmark_indices)

        self.image_paths = sorted(TRAINING_RGB_PATH.glob("*.jpg"))
        self.landmarks_3d_all = _load_json_array(TRAINING_XYZ_PATH)
        self.camera_matrices = _load_json_array(TRAINING_K_PATH)

        if len(self.landmarks_3d_all) != len(self.camera_matrices):
            raise ValueError("Landmark and camera annotation counts do not match.")

        self.images_per_annotation = infer_images_per_annotation(
            image_count=len(self.image_paths),
            annotation_count=len(self.landmarks_3d_all),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_annotation_index(self, image_index: int) -> int:
        return image_index // self.images_per_annotation

    def get_sample(self, image_index: int) -> FreiHandSample:
        image_path = self.image_paths[image_index]
        annotation_index = self.get_annotation_index(image_index)

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_hw = image_rgb.shape[:2]

        landmarks_3d = self.landmarks_3d_all[annotation_index].copy()
        camera_matrix = self.camera_matrices[annotation_index].copy()
        full_landmarks_2d = project_points(landmarks_3d, camera_matrix)
        selected_landmarks_2d = full_landmarks_2d[list(self.selected_landmark_indices)].copy()
        selected_landmarks_3d = landmarks_3d[list(self.selected_landmark_indices)].copy()

        crop_box = (0, 0, original_hw[1], original_hw[0])
        if self.crop_hand:
            crop_box = compute_hand_crop_box(full_landmarks_2d, original_hw, self.crop_padding)
            image_rgb, selected_landmarks_2d = crop_image_and_landmarks(
                image_rgb,
                selected_landmarks_2d,
                crop_box,
            )
            original_hw = image_rgb.shape[:2]

        target_hw = (self.image_size, self.image_size)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))
        selected_landmarks_2d = resize_landmarks(selected_landmarks_2d, original_hw, target_hw)

        image = image_rgb.astype(np.float32)
        if self.normalize:
            image /= 255.0

        return FreiHandSample(
            image=image,
            landmarks_2d=selected_landmarks_2d.astype(np.float32),
            landmarks_3d=selected_landmarks_3d.astype(np.float32),
            camera_matrix=camera_matrix.astype(np.float32),
            image_path=image_path,
            image_index=image_index,
            annotation_index=annotation_index,
            crop_box=crop_box,
        )

    def __getitem__(self, image_index: int) -> dict[str, object]:
        sample = self.get_sample(image_index)
        heatmaps = generate_heatmaps(
            landmarks_xy=sample.landmarks_2d,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )

        if not self.return_tensors:
            return {
                "image": sample.image,
                "landmarks_2d": sample.landmarks_2d,
                "landmarks_3d": sample.landmarks_3d,
                "camera_matrix": sample.camera_matrix,
                "heatmaps": heatmaps,
                "image_path": str(sample.image_path),
                "image_index": sample.image_index,
                "annotation_index": sample.annotation_index,
                "crop_box": sample.crop_box,
            }

        image_tensor = torch.from_numpy(sample.image).permute(2, 0, 1).float()
        landmarks_tensor = torch.from_numpy(sample.landmarks_2d).float()
        landmarks_3d_tensor = torch.from_numpy(sample.landmarks_3d).float()
        camera_matrix_tensor = torch.from_numpy(sample.camera_matrix).float()
        heatmap_tensor = torch.from_numpy(heatmaps).float()

        return {
            "image": image_tensor,
            "landmarks_2d": landmarks_tensor,
            "landmarks_3d": landmarks_3d_tensor,
            "camera_matrix": camera_matrix_tensor,
            "heatmaps": heatmap_tensor,
            "image_path": str(sample.image_path),
            "image_index": sample.image_index,
            "annotation_index": sample.annotation_index,
            "crop_box": sample.crop_box,
        }

    def summary(self) -> dict[str, object]:
        return {
            "image_count": len(self.image_paths),
            "annotation_count": len(self.landmarks_3d_all),
            "images_per_annotation": self.images_per_annotation,
            "num_landmarks": self.num_landmarks,
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size,
            "heatmap_sigma": self.heatmap_sigma,
            "crop_hand": self.crop_hand,
            "crop_padding": self.crop_padding,
            "selected_landmark_indices": list(self.selected_landmark_indices),
        }


def load_landmark_dataset(
    image_size: int = 224,
    heatmap_size: int = 56,
    heatmap_sigma: float = 2.0,
    normalize: bool = True,
    return_tensors: bool = True,
    crop_hand: bool = False,
    crop_padding: float = 0.25,
    selected_landmark_indices: Sequence[int] = DEFAULT_LANDMARK_INDICES,
) -> FreiHandLandmarkDataset:
    return FreiHandLandmarkDataset(
        image_size=image_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        normalize=normalize,
        return_tensors=return_tensors,
        crop_hand=crop_hand,
        crop_padding=crop_padding,
        selected_landmark_indices=selected_landmark_indices,
    )
