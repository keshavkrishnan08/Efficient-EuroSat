"""
Unified multi-dataset data loading for EfficientEuroSAT experiments.

Provides a single dispatcher interface that returns the same 4-tuple
(train_loader, val_loader, test_loader, class_weights) regardless of
which dataset is requested, enabling seamless multi-dataset evaluation.

Supported datasets:
    - eurosat:  EuroSAT Sentinel-2 satellite imagery (10 classes, ~27K images)
    - cifar100: CIFAR-100 natural images (100 classes, 60K images)
    - resisc45: NWPU-RESISC45 remote sensing (45 classes, 31.5K images)
"""

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets as tv_datasets

from .transforms import get_train_transform, get_test_transform
from .class_weights import compute_class_weights, compute_sample_weights
from .eurosat import (
    get_eurosat_dataloaders,
    get_class_names as _eurosat_class_names,
    _TransformDataset,
)


# ======================================================================
# Dataset Registry
# ======================================================================

DATASET_REGISTRY = {
    "eurosat": {
        "num_classes": 10,
        "mean": [0.3444, 0.3803, 0.4078],
        "std": [0.2027, 0.1365, 0.1155],
        "default_img_size": 224,
        "class_names": [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
            "Industrial", "Pasture", "PermanentCrop", "Residential",
            "River", "SeaLake",
        ],
    },
    "cifar100": {
        "num_classes": 100,
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
        "default_img_size": 224,
        "class_names": None,  # populated lazily from torchvision
    },
    "resisc45": {
        "num_classes": 45,
        "mean": [0.3680, 0.3810, 0.3436],
        "std": [0.1454, 0.1356, 0.1320],
        "default_img_size": 224,
        "class_names": None,  # populated lazily from directory names
    },
}


def get_dataset_info(dataset_name: str) -> dict:
    """Return metadata dict for the given dataset.

    Keys: num_classes, mean, std, default_img_size, class_names.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}"
        )
    info = dict(DATASET_REGISTRY[dataset_name])

    # Lazily populate CIFAR-100 class names
    if dataset_name == "cifar100" and info["class_names"] is None:
        try:
            ds = tv_datasets.CIFAR100(root="./data", train=True, download=False)
            info["class_names"] = ds.classes
        except Exception:
            info["class_names"] = [f"class_{i}" for i in range(100)]

    return info


def get_class_names(dataset_name: str = "eurosat") -> list:
    """Return human-readable class names for the given dataset."""
    if dataset_name == "eurosat":
        return _eurosat_class_names()
    return get_dataset_info(dataset_name).get("class_names") or []


# ======================================================================
# CIFAR-100 Loader
# ======================================================================

def _get_cifar100_dataloaders(
    root="./data",
    batch_size=64,
    img_size=224,
    num_workers=4,
    val_split=0.1,
    use_weighted_sampling=True,
    seed=42,
):
    """Build CIFAR-100 dataloaders.

    CIFAR-100 has a predefined train/test split. We further split the
    official train set into train+val.
    """
    train_transform = get_train_transform(img_size=img_size, dataset_name="cifar100")
    test_transform = get_test_transform(img_size=img_size, dataset_name="cifar100")

    # Download CIFAR-100
    train_full = tv_datasets.CIFAR100(root=root, train=True, download=True, transform=None)
    test_dataset_raw = tv_datasets.CIFAR100(root=root, train=False, download=True, transform=None)

    # Split train into train + val
    total = len(train_full)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_full, [train_size, val_size], generator=generator
    )

    train_dataset = _TransformDataset(train_subset, transform=train_transform)
    val_dataset = _TransformDataset(val_subset, transform=test_transform)
    test_dataset = _TransformDataset(
        torch.utils.data.Subset(test_dataset_raw, range(len(test_dataset_raw))),
        transform=test_transform,
    )

    class_weights = compute_class_weights(train_dataset)

    if use_weighted_sampling:
        sample_weights = compute_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


# ======================================================================
# RESISC45 Loader
# ======================================================================

def _get_resisc45_dataloaders(
    root="./data",
    batch_size=64,
    img_size=224,
    num_workers=4,
    val_split=0.1,
    test_split=0.1,
    use_weighted_sampling=True,
    seed=42,
):
    """Build RESISC45 dataloaders.

    RESISC45 has no predefined split, so we do 80/10/10 random split.
    """
    from .resisc45 import ensure_resisc45

    resisc_root = ensure_resisc45(root)

    train_transform = get_train_transform(img_size=img_size, dataset_name="resisc45")
    test_transform = get_test_transform(img_size=img_size, dataset_name="resisc45")

    full_dataset = tv_datasets.ImageFolder(root=resisc_root, transform=None)

    # Update class names in registry
    DATASET_REGISTRY["resisc45"]["class_names"] = full_dataset.classes

    # Split
    total = len(full_dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = _TransformDataset(train_subset, transform=train_transform)
    val_dataset = _TransformDataset(val_subset, transform=test_transform)
    test_dataset = _TransformDataset(test_subset, transform=test_transform)

    class_weights = compute_class_weights(train_dataset)

    if use_weighted_sampling:
        sample_weights = compute_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights


# ======================================================================
# Unified Dispatcher
# ======================================================================

def get_dataloaders(
    dataset_name: str = "eurosat",
    root: str = "./data",
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.1,
    use_weighted_sampling: bool = True,
    seed: int = 42,
):
    """Unified dataloader factory for all supported datasets.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    if dataset_name == "eurosat":
        return get_eurosat_dataloaders(
            root=root, batch_size=batch_size, img_size=img_size,
            num_workers=num_workers, val_split=val_split, test_split=test_split,
            use_weighted_sampling=use_weighted_sampling, seed=seed,
        )
    elif dataset_name == "cifar100":
        return _get_cifar100_dataloaders(
            root=root, batch_size=batch_size, img_size=img_size,
            num_workers=num_workers, val_split=val_split,
            use_weighted_sampling=use_weighted_sampling, seed=seed,
        )
    elif dataset_name == "resisc45":
        return _get_resisc45_dataloaders(
            root=root, batch_size=batch_size, img_size=img_size,
            num_workers=num_workers, val_split=val_split, test_split=test_split,
            use_weighted_sampling=use_weighted_sampling, seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}"
        )
