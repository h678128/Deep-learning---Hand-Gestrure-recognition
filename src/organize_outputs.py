from __future__ import annotations

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def destination_for(file_path: Path) -> Path:
    name = file_path.name.lower()

    if name.startswith("mapping_"):
        return OUTPUTS_DIR / "dataset_checks" / "mapping_checks" / file_path.name

    if "preview" in name or "landmark_" in name or "inspect" in name:
        if "crop6" in name:
            return OUTPUTS_DIR / "model_runs" / "landmark_cnn_crop6_best" / "inspect" / file_path.name
        if "_11" in name:
            return OUTPUTS_DIR / "model_runs" / "landmark_heatmap11_interleaved_20k" / "inspect" / file_path.name
        if "heatmap6" in name:
            return OUTPUTS_DIR / "model_runs" / "landmark_heatmap6_best" / "inspect" / file_path.name
        return OUTPUTS_DIR / "dataset_checks" / "legacy" / file_path.name

    if name.startswith("prediction_") or name.startswith("pred_"):
        if "_11" in name:
            model_folder = "landmark_heatmap11_interleaved_20k"
        elif "_20k" in name:
            model_folder = "landmark_heatmap6_interleaved_20k"
        elif "interleaved" in name:
            model_folder = "landmark_heatmap6_interleaved"
        else:
            model_folder = "landmark_heatmap6_best"
        return OUTPUTS_DIR / "model_runs" / model_folder / "evaluate" / file_path.name

    return OUTPUTS_DIR / "dataset_checks" / "misc" / file_path.name


def main() -> None:
    moved_files = 0

    for file_path in OUTPUTS_DIR.iterdir():
        if file_path.is_dir():
            continue

        destination = destination_for(file_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(destination))
        moved_files += 1
        print(f"Moved {file_path.name} -> {destination.relative_to(PROJECT_ROOT)}")

    print(f"Done. Moved {moved_files} files.")


if __name__ == "__main__":
    main()
