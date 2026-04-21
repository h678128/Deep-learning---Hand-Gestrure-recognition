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
ULTRALYTICS_HAND_KEYPOINTS_ROOT = PROJECT_ROOT / "data" / "hand-keypoints"

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
LANDMARK_INDICES_6 = (0, 4, 8, 12, 16, 20)
LANDMARK_INDICES_11 = (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20)
LANDMARK_PRESETS: dict[str, tuple[int, ...]] = {
    "6": LANDMARK_INDICES_6,
    "11": LANDMARK_INDICES_11,
    "21": tuple(range(21)),
}
DEFAULT_LANDMARK_INDICES = LANDMARK_INDICES_11
DEFAULT_MAPPING_MODE = "interleaved"
SIMPLIFIED_CONNECTIONS_6 = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
]
SIMPLIFIED_CONNECTIONS_11 = [
    (0, 1),
    (1, 2),
    (0, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (0, 9),
    (9, 10),
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


@dataclass
class UltralyticsHandSample:
    image: np.ndarray
    landmarks_2d: np.ndarray
    image_path: Path
    label_index: int
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


def resolve_landmark_indices(preset: str | None = None) -> tuple[int, ...]:
    if preset is None:
        return DEFAULT_LANDMARK_INDICES
    if preset not in LANDMARK_PRESETS:
        raise ValueError(f"Unsupported landmark preset: {preset}")
    return LANDMARK_PRESETS[preset]


def infer_connections(selected_landmark_indices: Sequence[int]) -> list[tuple[int, int]]:
    selected_tuple = tuple(selected_landmark_indices)

    if selected_tuple == LANDMARK_INDICES_6:
        return SIMPLIFIED_CONNECTIONS_6
    if selected_tuple == LANDMARK_INDICES_11:
        return SIMPLIFIED_CONNECTIONS_11

    index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(selected_tuple)}
    filtered_connections: list[tuple[int, int]] = []
    for start_idx, end_idx in FULL_HAND_CONNECTIONS:
        if start_idx in index_map and end_idx in index_map:
            filtered_connections.append((index_map[start_idx], index_map[end_idx]))
    return filtered_connections


def draw_landmarks(
    image: np.ndarray,
    landmarks_xy: np.ndarray,
    connections: Optional[Sequence[tuple[int, int]]] = None,
    point_radius: int = 3,
) -> np.ndarray:
    canvas = image.copy()
    if connections is not None:
        active_connections = list(connections)
    elif len(landmarks_xy) == len(LANDMARK_INDICES_6):
        active_connections = SIMPLIFIED_CONNECTIONS_6
    elif len(landmarks_xy) == len(LANDMARK_INDICES_11):
        active_connections = SIMPLIFIED_CONNECTIONS_11
    elif len(landmarks_xy) == 21:
        active_connections = FULL_HAND_CONNECTIONS
    else:
        active_connections = []

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


def apply_geometric_augmentation(
    image_rgb: np.ndarray,
    landmarks_xy: np.ndarray,
    strength: str = "moderate",
) -> tuple[np.ndarray, np.ndarray]:
    image_h, image_w = image_rgb.shape[:2]
    center = (image_w / 2.0, image_h / 2.0)

    if strength == "strong":
        angle = float(np.random.uniform(-18.0, 18.0))
        scale = float(np.random.uniform(0.85, 1.15))
        shift_x = float(np.random.uniform(-0.08, 0.08) * image_w)
        shift_y = float(np.random.uniform(-0.08, 0.08) * image_h)
    else:
        angle = float(np.random.uniform(-12.0, 12.0))
        scale = float(np.random.uniform(0.92, 1.08))
        shift_x = float(np.random.uniform(-0.05, 0.05) * image_w)
        shift_y = float(np.random.uniform(-0.05, 0.05) * image_h)

    transform = cv2.getRotationMatrix2D(center, angle, scale)
    transform[0, 2] += shift_x
    transform[1, 2] += shift_y

    augmented_image = cv2.warpAffine(
        image_rgb,
        transform,
        (image_w, image_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    homogeneous_landmarks = np.concatenate(
        [landmarks_xy, np.ones((len(landmarks_xy), 1), dtype=np.float32)],
        axis=1,
    )
    augmented_landmarks = homogeneous_landmarks @ transform.T
    augmented_landmarks[:, 0] = np.clip(augmented_landmarks[:, 0], 0, image_w - 1)
    augmented_landmarks[:, 1] = np.clip(augmented_landmarks[:, 1], 0, image_h - 1)
    return augmented_image, augmented_landmarks.astype(np.float32)


def apply_photometric_augmentation(image_rgb: np.ndarray, strength: str = "moderate") -> np.ndarray:
    augmented = image_rgb.astype(np.float32)

    if strength == "strong":
        contrast = float(np.random.uniform(0.7, 1.3))
        brightness = float(np.random.uniform(-32.0, 32.0))
        blur_probability = 0.45
        noise_probability = 0.5
        noise_sigma_range = (5.0, 14.0)
    else:
        contrast = float(np.random.uniform(0.85, 1.15))
        brightness = float(np.random.uniform(-18.0, 18.0))
        blur_probability = 0.3
        noise_probability = 0.35
        noise_sigma_range = (3.0, 8.0)

    augmented = augmented * contrast + brightness

    if np.random.rand() < blur_probability:
        augmented = cv2.GaussianBlur(augmented, (3, 3), sigmaX=0.0)

    if np.random.rand() < noise_probability:
        noise_sigma = float(np.random.uniform(*noise_sigma_range))
        augmented += np.random.normal(0.0, noise_sigma, size=augmented.shape).astype(np.float32)

    return np.clip(augmented, 0, 255).astype(np.uint8)


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
        augment: bool = False,
        augment_strength: str = "moderate",
        selected_landmark_indices: Sequence[int] = DEFAULT_LANDMARK_INDICES,
        mapping_mode: str = DEFAULT_MAPPING_MODE,
    ) -> None:
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.normalize = normalize
        self.return_tensors = return_tensors
        self.crop_hand = crop_hand
        self.crop_padding = crop_padding
        self.augment = augment
        self.augment_strength = augment_strength
        self.selected_landmark_indices = tuple(selected_landmark_indices)
        self.selected_connections = infer_connections(self.selected_landmark_indices)
        self.num_landmarks = len(self.selected_landmark_indices)
        self.mapping_mode = mapping_mode

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
        if self.mapping_mode == "grouped":
            return image_index // self.images_per_annotation
        if self.mapping_mode == "interleaved":
            return image_index % len(self.landmarks_3d_all)
        raise ValueError(f"Unsupported mapping_mode: {self.mapping_mode}")

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

        if self.augment:
            image_rgb, selected_landmarks_2d = apply_geometric_augmentation(
                image_rgb,
                selected_landmarks_2d,
                strength=self.augment_strength,
            )
            image_rgb = apply_photometric_augmentation(
                image_rgb,
                strength=self.augment_strength,
            )

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
            "mapping_mode": self.mapping_mode,
            "num_landmarks": self.num_landmarks,
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size,
            "heatmap_sigma": self.heatmap_sigma,
            "crop_hand": self.crop_hand,
            "crop_padding": self.crop_padding,
            "augment": self.augment,
            "augment_strength": self.augment_strength,
            "selected_landmark_indices": list(self.selected_landmark_indices),
        }


class UltralyticsHandKeypointDataset(Dataset):
    def __init__(
        self,
        root: Path = ULTRALYTICS_HAND_KEYPOINTS_ROOT,
        split: str = "train",
        image_size: int = 224,
        heatmap_size: int = 56,
        heatmap_sigma: float = 2.0,
        normalize: bool = True,
        return_tensors: bool = True,
        crop_hand: bool = False,
        crop_padding: float = 0.25,
        augment: bool = False,
        augment_strength: str = "moderate",
        selected_landmark_indices: Sequence[int] = DEFAULT_LANDMARK_INDICES,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.normalize = normalize
        self.return_tensors = return_tensors
        self.crop_hand = crop_hand
        self.crop_padding = crop_padding
        self.augment = augment
        self.augment_strength = augment_strength
        self.selected_landmark_indices = tuple(selected_landmark_indices)
        self.selected_connections = infer_connections(self.selected_landmark_indices)
        self.num_landmarks = len(self.selected_landmark_indices)

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Could not find Ultralytics image directory: {self.image_dir}. "
                "Download hand-keypoints.zip and extract it under data/ first."
            )
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Could not find Ultralytics label directory: {self.label_dir}.")

        self.samples: list[tuple[Path, np.ndarray, int]] = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in image_extensions
        )

        for image_path in image_paths:
            label_path = self.label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            with label_path.open("r", encoding="utf-8") as file:
                for label_index, line in enumerate(file):
                    values = np.fromstring(line.strip(), sep=" ", dtype=np.float32)
                    if values.size < 5 + 21 * 3:
                        continue
                    keypoints = values[5 : 5 + 21 * 3].reshape(21, 3)
                    selected_visibility = keypoints[list(self.selected_landmark_indices), 2]
                    if np.all(selected_visibility <= 0):
                        continue
                    self.samples.append((image_path, values, label_index))

        if not self.samples:
            raise ValueError(f"No valid Ultralytics hand-keypoint samples found in {self.root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def get_sample(self, sample_index: int) -> UltralyticsHandSample:
        image_path, label_values, label_index = self.samples[sample_index]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_hw = image_rgb.shape[:2]
        image_h, image_w = original_hw

        keypoints = label_values[5 : 5 + 21 * 3].reshape(21, 3).copy()
        full_landmarks_2d = keypoints[:, :2]
        full_landmarks_2d[:, 0] *= image_w
        full_landmarks_2d[:, 1] *= image_h
        full_landmarks_2d[:, 0] = np.clip(full_landmarks_2d[:, 0], 0, image_w - 1)
        full_landmarks_2d[:, 1] = np.clip(full_landmarks_2d[:, 1], 0, image_h - 1)

        selected_landmarks_2d = full_landmarks_2d[list(self.selected_landmark_indices)].copy()

        crop_box = (0, 0, original_hw[1], original_hw[0])
        if self.crop_hand:
            crop_box = compute_hand_crop_box(full_landmarks_2d, original_hw, self.crop_padding)
            image_rgb, selected_landmarks_2d = crop_image_and_landmarks(
                image_rgb,
                selected_landmarks_2d,
                crop_box,
            )
            original_hw = image_rgb.shape[:2]

        if self.augment:
            image_rgb, selected_landmarks_2d = apply_geometric_augmentation(
                image_rgb,
                selected_landmarks_2d,
                strength=self.augment_strength,
            )
            image_rgb = apply_photometric_augmentation(
                image_rgb,
                strength=self.augment_strength,
            )

        target_hw = (self.image_size, self.image_size)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))
        selected_landmarks_2d = resize_landmarks(selected_landmarks_2d, original_hw, target_hw)

        image = image_rgb.astype(np.float32)
        if self.normalize:
            image /= 255.0

        return UltralyticsHandSample(
            image=image,
            landmarks_2d=selected_landmarks_2d.astype(np.float32),
            image_path=image_path,
            label_index=label_index,
            crop_box=crop_box,
        )

    def __getitem__(self, sample_index: int) -> dict[str, object]:
        sample = self.get_sample(sample_index)
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
                "heatmaps": heatmaps,
                "image_path": str(sample.image_path),
                "image_index": sample_index,
                "annotation_index": sample.label_index,
                "crop_box": sample.crop_box,
                "source": "ultralytics",
            }

        image_tensor = torch.from_numpy(sample.image).permute(2, 0, 1).float()
        landmarks_tensor = torch.from_numpy(sample.landmarks_2d).float()
        heatmap_tensor = torch.from_numpy(heatmaps).float()

        return {
            "image": image_tensor,
            "landmarks_2d": landmarks_tensor,
            "landmarks_3d": torch.zeros((self.num_landmarks, 3), dtype=torch.float32),
            "camera_matrix": torch.eye(3, dtype=torch.float32),
            "heatmaps": heatmap_tensor,
            "image_path": str(sample.image_path),
            "image_index": sample_index,
            "annotation_index": sample.label_index,
            "crop_box": sample.crop_box,
            "source": "ultralytics",
        }

    def summary(self) -> dict[str, object]:
        return {
            "dataset": "ultralytics_hand_keypoints",
            "root": str(self.root),
            "split": self.split,
            "sample_count": len(self.samples),
            "num_landmarks": self.num_landmarks,
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size,
            "heatmap_sigma": self.heatmap_sigma,
            "crop_hand": self.crop_hand,
            "crop_padding": self.crop_padding,
            "augment": self.augment,
            "augment_strength": self.augment_strength,
            "selected_landmark_indices": list(self.selected_landmark_indices),
        }


class MixedHandLandmarkDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        selected_landmark_indices: Sequence[int] = DEFAULT_LANDMARK_INDICES,
        image_size: int = 224,
        heatmap_size: int = 56,
        heatmap_sigma: float = 2.0,
        crop_hand: bool = False,
        crop_padding: float = 0.25,
        augment: bool = False,
        augment_strength: str = "moderate",
    ) -> None:
        self.datasets = list(datasets)
        self.selected_landmark_indices = tuple(selected_landmark_indices)
        self.num_landmarks = len(self.selected_landmark_indices)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.crop_hand = crop_hand
        self.crop_padding = crop_padding
        self.augment = augment
        self.augment_strength = augment_strength
        self.cumulative_sizes = np.cumsum([len(dataset) for dataset in self.datasets]).tolist()

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1]) if self.cumulative_sizes else 0

    def __getitem__(self, index: int):
        dataset_index = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        sample_index = index - previous_size
        return self.datasets[dataset_index][sample_index]

    def summary(self) -> dict[str, object]:
        return {
            "dataset": "mixed",
            "components": [
                dataset.summary() if hasattr(dataset, "summary") else {"length": len(dataset)}
                for dataset in self.datasets
            ],
            "sample_count": len(self),
            "num_landmarks": self.num_landmarks,
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size,
            "heatmap_sigma": self.heatmap_sigma,
            "crop_hand": self.crop_hand,
            "crop_padding": self.crop_padding,
            "augment": self.augment,
            "augment_strength": self.augment_strength,
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
    augment: bool = False,
    augment_strength: str = "moderate",
    selected_landmark_indices: Sequence[int] = DEFAULT_LANDMARK_INDICES,
    mapping_mode: str = DEFAULT_MAPPING_MODE,
) -> FreiHandLandmarkDataset:
    return FreiHandLandmarkDataset(
        image_size=image_size,
        heatmap_size=heatmap_size,
        heatmap_sigma=heatmap_sigma,
        normalize=normalize,
        return_tensors=return_tensors,
        crop_hand=crop_hand,
        crop_padding=crop_padding,
        augment=augment,
        augment_strength=augment_strength,
        selected_landmark_indices=selected_landmark_indices,
        mapping_mode=mapping_mode,
    )
