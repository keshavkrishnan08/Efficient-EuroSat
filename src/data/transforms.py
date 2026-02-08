"""
Data augmentation transforms for EuroSAT satellite land use classification.

Provides training, testing, and simple transforms with EuroSAT-specific
normalization statistics. Training transforms include augmentation suited
for overhead satellite imagery where orientation is arbitrary.
"""

from torchvision import transforms

# EuroSAT normalization stats (computed over the full dataset)
EUROSAT_MEAN = [0.3444, 0.3803, 0.4078]
EUROSAT_STD = [0.2027, 0.1365, 0.1155]


def get_train_transform(img_size=224):
    """Build the training augmentation pipeline for EuroSAT.

    Satellite images can appear in any orientation, so we apply
    aggressive geometric augmentation (horizontal/vertical flips,
    90-degree rotations) alongside moderate photometric jitter.

    Args:
        img_size (int): Final spatial dimension. Default 224.

    Returns:
        transforms.Compose: Composed training transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
    ])


def get_test_transform(img_size=224):
    """Build the test/validation transform pipeline for EuroSAT.

    Applies only deterministic preprocessing (resize and normalization)
    so that evaluation results are reproducible across runs.

    Args:
        img_size (int): Final spatial dimension. Default 224.

    Returns:
        transforms.Compose: Composed test transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
    ])


def get_simple_transform(img_size=224):
    """Build a minimal transform pipeline for debugging and visualization.

    Args:
        img_size (int): Final spatial dimension. Default 224.

    Returns:
        transforms.Compose: Composed simple transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
    ])
