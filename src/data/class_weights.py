"""
Class and sample weight computation for imbalanced EuroSAT dataset.

EuroSAT contains 10 land use classes derived from Sentinel-2 satellite
imagery, with per-class sample counts that are not perfectly balanced.
This module provides utilities to compute inverse-frequency class weights
and per-sample weights that can be used with weighted loss functions
(e.g., CrossEntropyLoss) and WeightedRandomSampler respectively to
mitigate any class imbalance.
"""

import torch
from collections import Counter


def _extract_labels(dataset):
    """Extract labels without loading/transforming images.

    Tries to access targets directly from the underlying dataset structure
    (Subset or ImageFolder) to avoid the cost of loading every image.
    Falls back to full iteration if the structure is not recognised.
    """
    # _TransformDataset wrapping a Subset
    subset = getattr(dataset, 'subset', None)
    if subset is not None:
        inner = getattr(subset, 'dataset', None)
        indices = getattr(subset, 'indices', None)
        if inner is not None and indices is not None:
            targets = getattr(inner, 'targets', None)
            if targets is not None:
                return [int(targets[i]) for i in indices]

    # Direct Subset
    inner = getattr(dataset, 'dataset', None)
    indices = getattr(dataset, 'indices', None)
    if inner is not None and indices is not None:
        targets = getattr(inner, 'targets', None)
        if targets is not None:
            return [int(targets[i]) for i in indices]

    # Fallback: iterate (slow but correct)
    labels = []
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        labels.append(int(label))
    return labels


def compute_class_weights(dataset):
    """
    Compute inverse-frequency class weights from a dataset.

    Extracts labels efficiently without loading images, then assigns each
    class a weight proportional to the inverse of its frequency. The
    weights are normalized so that they sum to the total number of classes.

    Args:
        dataset: A PyTorch dataset (or Subset) where each element is a
            (image, label) tuple. The label must be an integer class index.

    Returns:
        torch.Tensor: A 1-D float tensor of shape (num_classes,) containing
            the normalized class weights. Weights sum to num_classes.
    """
    labels = _extract_labels(dataset)
    label_counts = Counter(labels)

    num_classes = len(label_counts)
    total_samples = sum(label_counts.values())

    # Compute raw inverse-frequency weights
    raw_weights = torch.zeros(num_classes, dtype=torch.float32)
    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 1)  # guard against missing classes
        raw_weights[class_idx] = total_samples / (num_classes * count)

    # Normalize so weights sum to num_classes
    weight_sum = raw_weights.sum()
    normalized_weights = raw_weights * (num_classes / weight_sum)

    return normalized_weights


def compute_class_rarity(dataset, num_classes=10):
    """Compute normalised class rarity scores for epistemic loss.

    Rarity = 1 - frequency, normalised to ``[0, 1]``.

    Parameters
    ----------
    dataset : Dataset
        Training dataset.
    num_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Rarity tensor of shape ``(num_classes,)``.
    """
    labels = _extract_labels(dataset)
    label_counts = Counter(labels)
    total = sum(label_counts.values())

    freq = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        freq[c] = label_counts.get(c, 0) / total

    rarity = 1.0 - freq
    # Normalise to [0, 1]
    rarity = (rarity - rarity.min()) / (rarity.max() - rarity.min() + 1e-8)
    return rarity


def compute_sample_weights(dataset):
    """
    Compute per-sample weights for use with WeightedRandomSampler.

    Each sample receives the inverse-frequency weight of its class, so that
    the sampler draws from under-represented classes more often.

    Args:
        dataset: A PyTorch dataset (or Subset) where each element is a
            (image, label) tuple. The label must be an integer class index.

    Returns:
        torch.Tensor: A 1-D float tensor of shape (len(dataset),) where
            entry i holds the sampling weight for the i-th example.
    """
    labels = _extract_labels(dataset)
    class_weights = compute_class_weights(dataset)

    sample_weights = torch.zeros(len(labels), dtype=torch.float32)
    for idx, label in enumerate(labels):
        sample_weights[idx] = class_weights[label]

    return sample_weights
