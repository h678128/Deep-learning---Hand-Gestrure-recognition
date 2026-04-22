from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import (
    DEFAULT_LANDMARK_INDICES,
    FreiHandLandmarkDataset,
    MixedHandLandmarkDataset,
    UltralyticsHandKeypointDataset,
    decode_heatmaps,
)
from model import create_heatmap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train en heatmap-modell for hand-keypoints."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--heatmap-size", type=int, default=56)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        default="none",
        help="Valgfri learning-rate scheduler. Standard endrer ikke eksisterende trening.",
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Begrens antall CPU-traader PyTorch bruker. Nyttig hvis PC-en blir treg under trening.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Brukes for rask testing. Lar deg trene paa et mindre delsett.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("modell") / "landmark_heatmap11_best.pt",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Fortsett trening fra en eksisterende checkpoint.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Bruk moderat dataaugmentasjon paa treningssettet.",
    )
    parser.add_argument(
        "--augment-strength",
        choices=["moderate", "strong"],
        default="moderate",
        help="Styrke paa augmentasjon naar --augment er aktiv.",
    )
    parser.add_argument(
        "--crop-hand",
        action="store_true",
        help="Crop rundt haanden basert paa fasit-landmarks.",
    )
    parser.add_argument(
        "--data-source",
        choices=["freihand", "ultralytics", "combined"],
        default="freihand",
        help="Velg om treningen bruker FreiHAND, Ultralytics hand-keypoints, eller begge.",
    )
    parser.add_argument(
        "--ultralytics-root",
        type=Path,
        default=Path("data") / "hand-keypoints",
        help="Rotmappe for Ultralytics hand-keypoints datasettet.",
    )
    return parser.parse_args()


class DatasetSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def summary(self) -> dict[str, object]:
        base_summary = self.dataset.summary() if hasattr(self.dataset, "summary") else {"length": len(self.dataset)}
        return {
            "dataset": "subset",
            "subset_size": len(self.indices),
            "base": base_summary,
        }


def get_freihand_image_indices_for_annotations(
    dataset: FreiHandLandmarkDataset,
    annotation_indices: list[int],
) -> list[int]:
    annotation_count = len(dataset.landmarks_3d_all)

    if dataset.mapping_mode == "interleaved":
        return [
            annotation_index + view_index * annotation_count
            for annotation_index in annotation_indices
            for view_index in range(dataset.images_per_annotation)
        ]

    if dataset.mapping_mode == "grouped":
        return [
            image_index
            for annotation_index in annotation_indices
            for image_index in range(
                annotation_index * dataset.images_per_annotation,
                (annotation_index + 1) * dataset.images_per_annotation,
            )
        ]

    annotation_set = set(annotation_indices)
    return [
        image_index
        for image_index in range(len(dataset))
        if dataset.get_annotation_index(image_index) in annotation_set
    ]


def split_freihand_image_indices(
    dataset: FreiHandLandmarkDataset,
    val_ratio: float,
    seed: int,
    max_samples: int | None = None,
) -> tuple[list[int], list[int]]:
    annotation_count = len(dataset.landmarks_3d_all)

    val_annotation_count = max(1, int(annotation_count * val_ratio))
    train_annotation_count = annotation_count - val_annotation_count

    if train_annotation_count <= 0:
        raise ValueError("Validation split is too large for the selected dataset size.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_annotations = torch.randperm(annotation_count, generator=generator).tolist()
    train_annotations = shuffled_annotations[:train_annotation_count]
    val_annotations = shuffled_annotations[train_annotation_count:]

    train_indices = get_freihand_image_indices_for_annotations(dataset, train_annotations)
    val_indices = get_freihand_image_indices_for_annotations(dataset, val_annotations)

    if max_samples is not None:
        selected_image_count = min(max_samples, len(train_indices) + len(val_indices))
        val_image_count = max(1, int(selected_image_count * val_ratio))
        train_image_count = selected_image_count - val_image_count

        if train_image_count <= 0:
            raise ValueError("Validation split is too large for the selected dataset size.")

        train_indices = train_indices[: min(train_image_count, len(train_indices))]
        val_indices = val_indices[: min(val_image_count, len(val_indices))]

    return train_indices, val_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    if isinstance(train_dataset, FreiHandLandmarkDataset) and isinstance(
        val_dataset,
        FreiHandLandmarkDataset,
    ):
        train_indices, val_indices = split_freihand_image_indices(
            dataset=train_dataset,
            val_ratio=val_ratio,
            seed=seed,
            max_samples=max_samples,
        )
    else:
        total_indices = list(range(len(train_dataset)))

        if max_samples is not None:
            total_indices = total_indices[:max_samples]

        total_size = len(total_indices)
        val_size = max(1, int(total_size * val_ratio))
        train_size = total_size - val_size

        if train_size <= 0:
            raise ValueError("Validation split is too large for the selected dataset size.")

        generator = torch.Generator().manual_seed(seed)
        shuffled_indices = torch.randperm(total_size, generator=generator).tolist()
        train_indices = [total_indices[index] for index in shuffled_indices[:train_size]]
        val_indices = [total_indices[index] for index in shuffled_indices[train_size:]]

    train_subset = DatasetSubset(train_dataset, train_indices)
    val_subset = DatasetSubset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def create_native_split_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    if max_samples is not None:
        train_size = max(1, int(max_samples * 0.9))
        val_size = max(1, max_samples - train_size)
        train_generator = torch.Generator().manual_seed(seed)
        val_generator = torch.Generator().manual_seed(seed + 1)
        train_indices = torch.randperm(len(train_dataset), generator=train_generator).tolist()
        val_indices = torch.randperm(len(val_dataset), generator=val_generator).tolist()
        train_indices = train_indices[: min(train_size, len(train_indices))]
        val_indices = val_indices[: min(val_size, len(val_indices))]
        train_dataset = DatasetSubset(train_dataset, train_indices)
        val_dataset = DatasetSubset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_freihand_datasets(args: argparse.Namespace) -> tuple[FreiHandLandmarkDataset, FreiHandLandmarkDataset]:
    train_dataset = FreiHandLandmarkDataset(
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        normalize=True,
        return_tensors=True,
        crop_hand=args.crop_hand,
        augment=args.augment,
        augment_strength=args.augment_strength,
        selected_landmark_indices=DEFAULT_LANDMARK_INDICES,
    )
    val_dataset = FreiHandLandmarkDataset(
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        normalize=True,
        return_tensors=True,
        crop_hand=args.crop_hand,
        augment=False,
        augment_strength=args.augment_strength,
        selected_landmark_indices=DEFAULT_LANDMARK_INDICES,
    )
    return train_dataset, val_dataset


def build_ultralytics_datasets(args: argparse.Namespace) -> tuple[UltralyticsHandKeypointDataset, UltralyticsHandKeypointDataset]:
    train_dataset = UltralyticsHandKeypointDataset(
        root=args.ultralytics_root,
        split="train",
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        normalize=True,
        return_tensors=True,
        crop_hand=args.crop_hand,
        augment=args.augment,
        augment_strength=args.augment_strength,
        selected_landmark_indices=DEFAULT_LANDMARK_INDICES,
    )
    val_dataset = UltralyticsHandKeypointDataset(
        root=args.ultralytics_root,
        split="val",
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        normalize=True,
        return_tensors=True,
        crop_hand=args.crop_hand,
        augment=False,
        augment_strength=args.augment_strength,
        selected_landmark_indices=DEFAULT_LANDMARK_INDICES,
    )
    return train_dataset, val_dataset


def split_dataset_indices(
    dataset_size: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    total_indices = list(range(dataset_size))
    val_size = max(1, int(dataset_size * val_ratio))
    train_size = dataset_size - val_size

    if train_size <= 0:
        raise ValueError("Validation split is too large for the selected dataset size.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = [total_indices[index] for index in shuffled_indices[:train_size]]
    val_indices = [total_indices[index] for index in shuffled_indices[train_size:]]
    return train_indices, val_indices


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset, bool]:
    if args.data_source == "freihand":
        train_dataset, val_dataset = build_freihand_datasets(args)
        return train_dataset, val_dataset, False

    if args.data_source == "ultralytics":
        train_dataset, val_dataset = build_ultralytics_datasets(args)
        return train_dataset, val_dataset, True

    freihand_train_full, freihand_val_full = build_freihand_datasets(args)
    ultralytics_train, ultralytics_val = build_ultralytics_datasets(args)
    freihand_train_indices, freihand_val_indices = split_freihand_image_indices(
        dataset=freihand_train_full,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    freihand_train = DatasetSubset(freihand_train_full, freihand_train_indices)
    freihand_val = DatasetSubset(freihand_val_full, freihand_val_indices)
    train_dataset = MixedHandLandmarkDataset(
        [freihand_train, ultralytics_train],
        selected_landmark_indices=DEFAULT_LANDMARK_INDICES,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        crop_hand=args.crop_hand,
        augment=args.augment,
        augment_strength=args.augment_strength,
    )
    val_dataset = MixedHandLandmarkDataset(
        [freihand_val, ultralytics_val],
        selected_landmark_indices=DEFAULT_LANDMARK_INDICES,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        crop_hand=args.crop_hand,
        augment=False,
        augment_strength=args.augment_strength,
    )
    return train_dataset, val_dataset, True


def mean_pixel_error(
    predicted_heatmaps: torch.Tensor,
    target_landmarks: torch.Tensor,
    image_size: int,
) -> float:
    predicted_landmarks = decode_heatmaps(predicted_heatmaps, image_size=image_size)
    distances = torch.linalg.norm(predicted_landmarks - target_landmarks, dim=-1)
    return distances.mean().item()


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    image_size: int,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_error = 0.0
    total_batches = 0

    context_manager = torch.enable_grad if is_training else torch.no_grad

    with context_manager():
        for batch in dataloader:
            images = batch["image"].to(device)
            target_heatmaps = batch["heatmaps"].to(device)
            target_landmarks = batch["landmarks_2d"].to(device)

            predicted_heatmaps = model(images)
            loss = criterion(predicted_heatmaps, target_heatmaps)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_error += mean_pixel_error(
                predicted_heatmaps.detach(),
                target_landmarks.detach(),
                image_size=image_size,
            )
            total_batches += 1

    return total_loss / total_batches, total_error / total_batches


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
    dataset: Dataset,
    best_val_loss: float,
    best_epoch: int,
    completed_epochs: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "image_size": args.image_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "num_landmarks": dataset.num_landmarks,
            "selected_landmark_indices": list(dataset.selected_landmark_indices),
            "crop_hand": dataset.crop_hand,
            "crop_padding": dataset.crop_padding,
            "data_source": args.data_source,
            "ultralytics_root": str(args.ultralytics_root),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
        },
        checkpoint_path,
    )

    metadata_path = checkpoint_path.with_suffix(".json")
    metadata = {
        "checkpoint": str(checkpoint_path),
        "image_size": args.image_size,
        "heatmap_size": args.heatmap_size,
        "heatmap_sigma": args.heatmap_sigma,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "min_learning_rate": args.min_learning_rate,
        "weight_decay": args.weight_decay,
        "selected_landmark_indices": list(dataset.selected_landmark_indices),
        "crop_hand": dataset.crop_hand,
        "augment": dataset.augment,
        "augment_strength": dataset.augment_strength,
        "data_source": args.data_source,
        "ultralytics_root": str(args.ultralytics_root),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def latest_checkpoint_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_latest{checkpoint_path.suffix}")


def load_resume_checkpoint(
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dataset: Dataset,
    device: torch.device,
) -> tuple[float, int, int]:
    checkpoint = torch.load(resume_path, map_location=device)

    checkpoint_indices = tuple(checkpoint.get("selected_landmark_indices", []))
    if checkpoint_indices != dataset.selected_landmark_indices:
        raise ValueError(
            "Checkpoint uses different landmark indices than the current dataset setup."
        )

    checkpoint_num_landmarks = checkpoint.get("num_landmarks")
    if checkpoint_num_landmarks != dataset.num_landmarks:
        raise ValueError("Checkpoint uses a different number of landmarks.")

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    completed_epochs = int(checkpoint.get("completed_epochs", best_epoch))
    return best_val_loss, best_epoch, completed_epochs


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, uses_native_split = build_datasets(args)
    if uses_native_split:
        train_loader, val_loader = create_native_split_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    else:
        train_loader, val_loader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

    model = create_heatmap_model(num_landmarks=train_dataset.num_landmarks).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_learning_rate,
        )

    best_val_loss = float("inf")
    best_epoch = 0
    completed_epochs = 0

    if args.resume_from is not None:
        best_val_loss, best_epoch, completed_epochs = load_resume_checkpoint(
            resume_path=args.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset=train_dataset,
            device=device,
        )

    print("Device:", device)
    if args.torch_threads is not None:
        print("Torch CPU threads:", torch.get_num_threads())
    print("Train dataset summary:", train_dataset.summary())
    print("Validation dataset summary:", val_dataset.summary())
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    if args.resume_from is not None:
        print("Resume checkpoint:", args.resume_from)
        print("Previously completed epochs:", completed_epochs)
        print("Best validation loss so far:", round(best_val_loss, 5))

    for epoch_offset in range(1, args.epochs + 1):
        epoch = completed_epochs + epoch_offset
        train_loss, train_error = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            image_size=args.image_size,
        )
        val_loss, val_error = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            image_size=args.image_size,
        )

        print(
            f"Epoch {epoch:02d}/{completed_epochs + args.epochs} | "
            f"train_loss={train_loss:.5f} | train_pixel_error={train_error:.2f} | "
            f"val_loss={val_loss:.5f} | val_pixel_error={val_error:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                scheduler,
                args,
                train_dataset,
                best_val_loss,
                best_epoch,
                completed_epochs=epoch,
            )
            print(f"Saved best checkpoint to {args.checkpoint}")

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
            latest_checkpoint_path(args.checkpoint),
            model,
            optimizer,
            scheduler,
            args,
            train_dataset,
            best_val_loss,
            best_epoch,
            completed_epochs=epoch,
        )


if __name__ == "__main__":
    main()
