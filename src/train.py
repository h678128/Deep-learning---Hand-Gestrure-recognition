from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import DEFAULT_LANDMARK_INDICES, FreiHandLandmarkDataset, decode_heatmaps
from model import create_heatmap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train en heatmap-modell for 6 hand-keypoints."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--heatmap-size", type=int, default=56)
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Brukes for rask testing. Lar deg trene paa et mindre delsett.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("modell") / "landmark_heatmap6_best.pt",
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    dataset: FreiHandLandmarkDataset,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    total_indices = list(range(len(dataset)))

    if max_samples is not None:
        total_indices = total_indices[:max_samples]

    subset_dataset: Dataset
    if max_samples is None:
        subset_dataset = dataset
    else:
        subset_dataset = DatasetSubset(dataset, total_indices)

    total_size = len(subset_dataset)
    val_size = max(1, int(total_size * val_ratio))
    train_size = total_size - val_size

    if train_size <= 0:
        raise ValueError("Validation split is too large for the selected dataset size.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        subset_dataset,
        [train_size, val_size],
        generator=generator,
    )

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
    args: argparse.Namespace,
    dataset: FreiHandLandmarkDataset,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": args.image_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "num_landmarks": dataset.num_landmarks,
            "selected_landmark_indices": list(dataset.selected_landmark_indices),
            "crop_hand": dataset.crop_hand,
            "crop_padding": dataset.crop_padding,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
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
        "weight_decay": args.weight_decay,
        "selected_landmark_indices": list(dataset.selected_landmark_indices),
        "crop_hand": dataset.crop_hand,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FreiHandLandmarkDataset(
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        heatmap_sigma=args.heatmap_sigma,
        normalize=True,
        return_tensors=True,
        crop_hand=False,
        selected_landmark_indices=DEFAULT_LANDMARK_INDICES,
    )
    train_loader, val_loader = create_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    model = create_heatmap_model(num_landmarks=dataset.num_landmarks).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0

    print("Device:", device)
    print("Dataset summary:", dataset.summary())
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    for epoch in range(1, args.epochs + 1):
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
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | train_pixel_error={train_error:.2f} | "
            f"val_loss={val_loss:.5f} | val_pixel_error={val_error:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(args.checkpoint, model, args, dataset, best_val_loss, best_epoch)
            print(f"Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
