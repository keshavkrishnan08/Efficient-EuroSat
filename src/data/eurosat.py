"""
EuroSAT dataset loading, splitting, and dataloader construction.

Provides convenience functions to download the EuroSAT dataset
(Sentinel-2 satellite imagery, 10 land use classes) via torchvision,
apply appropriate transforms, split into train/val/test partitions,
and build DataLoaders with optional weighted sampling for class-imbalance
mitigation.
"""

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.datasets import EuroSAT

from .transforms import get_train_transform, get_test_transform
from .class_weights import compute_class_weights, compute_sample_weights

# All 10 EuroSAT land use class names
EUROSAT_CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


class _TransformDataset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a per-split transform.

    This avoids setting ``transform`` on the shared parent dataset
    (which would affect all splits equally) by intercepting items
    returned by the underlying subset and applying the desired
    transform pipeline.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_eurosat_datasets(root="./data", img_size=224, val_split=0.1, test_split=0.1, seed=42):
    """Download (if needed) and return EuroSAT train, val, and test datasets.

    EuroSAT does not have a predefined train/test split, so we split
    the full dataset into train/val/test using the given ratios.

    Args:
        root (str): Root directory for data storage. Default './data'.
        img_size (int): Spatial resolution for transforms. Default 224.
        val_split (float): Fraction for validation. Default 0.1.
        test_split (float): Fraction for test. Default 0.1.
        seed (int): Random seed for reproducible splits. Default 42.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_transform = get_train_transform(img_size=img_size)
    test_transform = get_test_transform(img_size=img_size)

    # Download the full dataset with transform=None so each split can
    # apply its own pipeline.
    full_dataset = EuroSAT(root=root, download=True, transform=None)

    # Compute split sizes
    total = len(full_dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap each subset with the appropriate transform
    train_dataset = _TransformDataset(train_subset, transform=train_transform)
    val_dataset = _TransformDataset(val_subset, transform=test_transform)
    test_dataset = _TransformDataset(test_subset, transform=test_transform)

    return train_dataset, val_dataset, test_dataset


def get_eurosat_dataloaders(
    root="./data",
    batch_size=64,
    img_size=224,
    num_workers=4,
    val_split=0.1,
    test_split=0.1,
    use_weighted_sampling=True,
    seed=42,
):
    """Build train, validation, and test DataLoaders for EuroSAT.

    Args:
        root (str): Root directory for EuroSAT data. Default './data'.
        batch_size (int): Mini-batch size. Default 64.
        img_size (int): Image resolution. Default 224.
        num_workers (int): Data loading workers. Default 4.
        val_split (float): Validation fraction. Default 0.1.
        test_split (float): Test fraction. Default 0.1.
        use_weighted_sampling (bool): Apply weighted sampling. Default True.
        seed (int): Random seed for split. Default 42.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    train_dataset, val_dataset, test_dataset = get_eurosat_datasets(
        root=root, img_size=img_size, val_split=val_split,
        test_split=test_split, seed=seed
    )

    # Compute class weights on the training subset
    class_weights = compute_class_weights(train_dataset)

    if use_weighted_sampling:
        sample_weights = compute_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


def get_class_names():
    """Return the list of all 10 EuroSAT class names.

    Returns:
        list[str]: Human-readable names for classes 0 through 9.
    """
    return list(EUROSAT_CLASS_NAMES)
