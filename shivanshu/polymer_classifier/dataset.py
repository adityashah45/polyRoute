"""
Dataset and DataLoader utilities for Polymer Classification.
Includes tailored augmentation pipelines, class distribution analysis,
and class-balanced samplers to solve extreme dataset skew.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from .config import ClassifierConfig, DEFAULT_CLASSES


def get_transforms(
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation transforms tailored for plastic items.
    Augmentation incorporates affine distortion, color jitter, and random erasing
    to generalize across varying transparency, creases, and lighting.
    """
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), shear=6),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value="random"),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def get_class_counts(dataset: datasets.ImageFolder, num_classes: int) -> List[int]:
    """Computes exact sample counts for each class index in the dataset."""
    counts = [0] * num_classes
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def build_balanced_sampler(
    dataset: datasets.ImageFolder,
    class_counts: List[int],
    mode: str = "sqrt_frequency",
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Constructs a WeightedRandomSampler to balance batch composition.
    'sqrt_frequency': weights proportional to 1 / sqrt(N_c), striking the ideal balance
    between minority exposure and avoiding over-repetition of rare samples.
    """
    num_classes = len(class_counts)
    if mode == "sqrt_frequency":
        class_weights = [1.0 / np.sqrt(max(1, c)) for c in class_counts]
    elif mode == "inverse_frequency":
        class_weights = [1.0 / max(1, c) for c in class_counts]
    else:
        class_weights = [1.0] * num_classes

    # Assign each sample its corresponding class weight
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    if num_samples is None:
        num_samples = len(dataset)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )
    return sampler


def create_dataloaders(
    config: ClassifierConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], List[int], List[str]]:
    """
    Creates PyTorch DataLoaders for train, valid, and test splits.
    Ensures consistent class ordering and applies class-balanced sampling on train split.
    """
    train_transform, eval_transform = get_transforms(image_size=config.image_size)

    train_dir = os.path.join(config.data_dir, "train")
    val_dir = os.path.join(config.data_dir, "valid")
    test_dir = os.path.join(config.data_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training dataset directory not found at: {train_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    classes = train_dataset.classes
    num_classes = len(classes)

    class_counts = get_class_counts(train_dataset, num_classes)

    # Sampler configuration
    if config.sampler in ("sqrt_frequency", "inverse_frequency"):
        target_samples = config.samples_per_epoch if config.samples_per_epoch > 0 else len(train_dataset)
        sampler = build_balanced_sampler(
            train_dataset,
            class_counts,
            mode=config.sampler,
            num_samples=target_samples,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )

    val_loader = None
    if os.path.exists(val_dir):
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    test_loader = None
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    return train_loader, val_loader, test_loader, class_counts, classes
