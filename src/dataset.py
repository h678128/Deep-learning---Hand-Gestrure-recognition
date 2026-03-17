from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "trene"
TRAINING_RGB_PATH = DATA_ROOT / "training" / "rgb"
TRAINING_XYZ_PATH = DATA_ROOT / "training_xyz.json"
TRAINING_K_PATH = DATA_ROOT / "training_K.json"

# Same landmark ordering as common hand-pose datasets and MediaPipe-style demos.
HAND_CONNECTIONS = [
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


@dataclass
class FreiHandSample:
    image: np.ndarray
    landmarks_2d: np.ndarray
    landmarks_3d: np.ndarray
    camera_matrix: np.ndarray
    image_path: Path
    image_index: int
    annotation_index: int


def _load_json_array(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        return np.asarray(json.load(file), dtype=np.float32)


def infer_images_per_annotation(
    image_count: int,
    annotation_count: int,
) -> int:
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


def draw_landmarks(
    image: np.ndarray,
    landmarks_xy: np.ndarray,
    point_radius: int = 3,
) -> np.ndarray:
    canvas = image.copy()

    for start_idx, end_idx in HAND_CONNECTIONS:
        start_point = tuple(np.round(landmarks_xy[start_idx]).astype(int))
        end_point = tuple(np.round(landmarks_xy[end_idx]).astype(int))
        cv2.line(canvas, start_point, end_point, (0, 220, 120), 2)

    for point in landmarks_xy:
        point_xy = tuple(np.round(point).astype(int))
        cv2.circle(canvas, point_xy, point_radius, (255, 90, 90), -1)

    return canvas


class FreiHandLandmarkDataset(Dataset):
    def __init__(
        self,
        image_size: Optional[int] = 224,
        normalize: bool = True,
        return_tensors: bool = True,
    ) -> None:
        self.image_size = image_size
        self.normalize = normalize
        self.return_tensors = return_tensors

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
        landmarks_2d = project_points(landmarks_3d, camera_matrix)

        if self.image_size is not None:
            target_hw = (self.image_size, self.image_size)
            image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))
            landmarks_2d = resize_landmarks(landmarks_2d, original_hw, target_hw)

        image = image_rgb.astype(np.float32)
        if self.normalize:
            image /= 255.0

        return FreiHandSample(
            image=image,
            landmarks_2d=landmarks_2d.astype(np.float32),
            landmarks_3d=landmarks_3d.astype(np.float32),
            camera_matrix=camera_matrix.astype(np.float32),
            image_path=image_path,
            image_index=image_index,
            annotation_index=annotation_index,
        )

    def __getitem__(self, image_index: int) -> dict[str, object]:
        sample = self.get_sample(image_index)

        if not self.return_tensors:
            return {
                "image": sample.image,
                "landmarks_2d": sample.landmarks_2d,
                "landmarks_3d": sample.landmarks_3d,
                "camera_matrix": sample.camera_matrix,
                "image_path": str(sample.image_path),
                "image_index": sample.image_index,
                "annotation_index": sample.annotation_index,
            }

        image_tensor = torch.from_numpy(sample.image).permute(2, 0, 1).float()
        landmarks_tensor = torch.from_numpy(sample.landmarks_2d).float()
        landmarks_3d_tensor = torch.from_numpy(sample.landmarks_3d).float()
        camera_matrix_tensor = torch.from_numpy(sample.camera_matrix).float()

        return {
            "image": image_tensor,
            "landmarks_2d": landmarks_tensor,
            "landmarks_3d": landmarks_3d_tensor,
            "camera_matrix": camera_matrix_tensor,
            "image_path": str(sample.image_path),
            "image_index": sample.image_index,
            "annotation_index": sample.annotation_index,
        }

    def summary(self) -> dict[str, int]:
        return {
            "image_count": len(self.image_paths),
            "annotation_count": len(self.landmarks_3d_all),
            "images_per_annotation": self.images_per_annotation,
        }


def load_landmark_dataset(
    image_size: Optional[int] = 224,
    normalize: bool = True,
    return_tensors: bool = True,
) -> FreiHandLandmarkDataset:
    return FreiHandLandmarkDataset(
        image_size=image_size,
        normalize=normalize,
        return_tensors=return_tensors,
    )
