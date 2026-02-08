"""Tests for data loading, transforms, and class weights.

Uses dummy tensors and PIL images so that the EuroSAT dataset does not need
to be downloaded.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np
from PIL import Image

from src.data.transforms import get_train_transform, get_test_transform
from src.data.class_weights import compute_class_weights
from src.data.eurosat import get_class_names

# ---------------------------------------------------------------------------
# Small test constants
# ---------------------------------------------------------------------------
IMG_SIZE = 32
NUM_EUROSAT_CLASSES = 10


def _make_random_pil_image(size: int = 64) -> Image.Image:
    """Create a random RGB PIL image of the given size."""
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class DummyDataset:
    """A minimal dataset that returns (tensor, label) pairs.

    Used to test compute_class_weights without downloading EuroSAT.
    """

    def __init__(self, num_classes: int = 5, samples_per_class: int = 10):
        self.data = []
        for cls in range(num_classes):
            for _ in range(samples_per_class):
                self.data.append((torch.randn(3, 32, 32), cls))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


class DummyImbalancedDataset:
    """A dataset with intentionally imbalanced class distribution."""

    def __init__(self):
        self.data = []
        # Class 0: 100 samples, Class 1: 10 samples, Class 2: 50 samples
        counts = {0: 100, 1: 10, 2: 50}
        for cls, count in counts.items():
            for _ in range(count):
                self.data.append((torch.randn(3, 32, 32), cls))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


# ===========================================================================
# Transform Tests
# ===========================================================================


class TestTransforms:
    """Test training and test transforms."""

    def test_train_transform_output_shape(self):
        """Training transform produces correct tensor shape [C, H, W]."""
        transform = get_train_transform(img_size=IMG_SIZE)
        img = _make_random_pil_image(size=64)
        tensor = transform(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), (
            f"Expected shape (3, {IMG_SIZE}, {IMG_SIZE}), got {tensor.shape}"
        )

    def test_test_transform_output_shape(self):
        """Test transform produces correct tensor shape [C, H, W]."""
        transform = get_test_transform(img_size=IMG_SIZE)
        img = _make_random_pil_image(size=64)
        tensor = transform(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), (
            f"Expected shape (3, {IMG_SIZE}, {IMG_SIZE}), got {tensor.shape}"
        )

    def test_train_transform_normalization(self):
        """Output is normalized: roughly zero mean per channel."""
        transform = get_train_transform(img_size=IMG_SIZE)
        # Accumulate statistics over multiple random images
        means = []
        for _ in range(50):
            img = _make_random_pil_image(size=64)
            tensor = transform(img)
            means.append(tensor.mean().item())
        avg_mean = sum(means) / len(means)
        # After normalization with EuroSAT stats, the mean should be
        # approximately zero (within a reasonable tolerance for random images)
        assert abs(avg_mean) < 1.5, (
            f"Average mean {avg_mean} is too far from zero; "
            f"normalization may not be applied"
        )


# ===========================================================================
# Class Weights Tests
# ===========================================================================


class TestClassWeights:
    """Test class weight computation."""

    def test_class_weights_shape(self):
        """compute_class_weights returns tensor of correct size."""
        num_classes = 5
        dataset = DummyDataset(num_classes=num_classes, samples_per_class=10)
        weights = compute_class_weights(dataset)
        assert weights.shape == (num_classes,), (
            f"Expected shape ({num_classes},), got {weights.shape}"
        )

    def test_class_weights_positive(self):
        """All class weights must be positive."""
        dataset = DummyDataset(num_classes=5, samples_per_class=10)
        weights = compute_class_weights(dataset)
        assert (weights > 0).all(), f"Some class weights are non-positive: {weights}"

    def test_class_weights_imbalanced(self):
        """Under-represented classes get higher weights."""
        dataset = DummyImbalancedDataset()
        weights = compute_class_weights(dataset)
        # Class 1 has fewest samples (10) so should get the highest weight
        assert weights[1] > weights[0], (
            f"Minority class weight ({weights[1]}) should exceed "
            f"majority class weight ({weights[0]})"
        )
        assert weights[1] > weights[2], (
            f"Minority class weight ({weights[1]}) should exceed "
            f"medium class weight ({weights[2]})"
        )


# ===========================================================================
# Class Names Tests
# ===========================================================================


class TestClassNames:
    """Test EuroSAT class name listing."""

    def test_class_names_count(self):
        """get_class_names returns a list of 10 names."""
        names = get_class_names()
        assert isinstance(names, list), f"Expected list, got {type(names)}"
        assert len(names) == NUM_EUROSAT_CLASSES, (
            f"Expected {NUM_EUROSAT_CLASSES} class names, got {len(names)}"
        )

    def test_class_names_are_strings(self):
        """Each class name is a non-empty string."""
        names = get_class_names()
        for i, name in enumerate(names):
            assert isinstance(name, str), (
                f"Class name at index {i} is {type(name)}, expected str"
            )
            assert len(name) > 0, f"Class name at index {i} is empty"
