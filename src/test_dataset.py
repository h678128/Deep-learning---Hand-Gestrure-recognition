from pathlib import Path

import cv2
import numpy as np

from dataset import FreiHandLandmarkDataset, draw_landmarks


def main() -> None:
    dataset = FreiHandLandmarkDataset(image_size=224, normalize=True, return_tensors=False)
    sample = dataset.get_sample(0)

    image_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)
    preview_rgb = draw_landmarks(image_uint8, sample.landmarks_2d)

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_landmark_preview.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))

    print("Dataset summary:", dataset.summary())
    print("Preview saved to:", output_path)
    print("First RGB image:", sample.image_path)
    print("First annotation index:", sample.annotation_index)


if __name__ == "__main__":
    main()
