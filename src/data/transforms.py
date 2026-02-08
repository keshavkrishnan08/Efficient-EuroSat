"""
Data augmentation transforms for multi-dataset training.

Provides dataset-specific training and test transforms with appropriate
normalization statistics and augmentation strategies:
- Satellite imagery (eurosat, resisc45): aggressive geometric augmentation
  (vertical flips, 90-degree rotations) since orientation is arbitrary.
- Natural images (cifar100): standard horizontal flip + color jitter only.
"""

from torchvision import transforms

# Per-dataset normalization statistics
NORM_STATS = {
    "eurosat": {
        "mean": [0.3444, 0.3803, 0.4078],
        "std": [0.2027, 0.1365, 0.1155],
    },
    "cifar100": {
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
    },
    "resisc45": {
        "mean": [0.3680, 0.3810, 0.3436],
        "std": [0.1454, 0.1356, 0.1320],
    },
}

# Backward compatibility
EUROSAT_MEAN = NORM_STATS["eurosat"]["mean"]
EUROSAT_STD = NORM_STATS["eurosat"]["std"]

# Satellite datasets that benefit from rotation-invariant augmentation
_SATELLITE_DATASETS = {"eurosat", "resisc45"}


def _get_norm(dataset_name: str):
    """Return (mean, std) for the given dataset, defaulting to ImageNet."""
    stats = NORM_STATS.get(dataset_name, NORM_STATS["eurosat"])
    return stats["mean"], stats["std"]


def get_train_transform(img_size=224, dataset_name="eurosat"):
    """Build the training augmentation pipeline.

    Satellite images get aggressive geometric augmentation (vertical flip,
    90-degree rotation). Natural images get horizontal flip + color jitter.

    Args:
        img_size (int): Final spatial dimension. Default 224.
        dataset_name (str): Dataset identifier. Default 'eurosat'.

    Returns:
        transforms.Compose: Composed training transform pipeline.
    """
    mean, std = _get_norm(dataset_name)

    if dataset_name in _SATELLITE_DATASETS:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # Natural images (CIFAR-100, etc.)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=img_size // 8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_test_transform(img_size=224, dataset_name="eurosat"):
    """Build the test/validation transform pipeline.

    Args:
        img_size (int): Final spatial dimension. Default 224.
        dataset_name (str): Dataset identifier. Default 'eurosat'.

    Returns:
        transforms.Compose: Composed test transform pipeline.
    """
    mean, std = _get_norm(dataset_name)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_simple_transform(img_size=224, dataset_name="eurosat"):
    """Build a minimal transform pipeline for debugging and visualization.

    Args:
        img_size (int): Final spatial dimension. Default 224.
        dataset_name (str): Dataset identifier. Default 'eurosat'.

    Returns:
        transforms.Compose: Composed simple transform pipeline.
    """
    mean, std = _get_norm(dataset_name)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
