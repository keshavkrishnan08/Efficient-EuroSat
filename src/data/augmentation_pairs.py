"""Generate augmentation pairs for aleatoric consistency training."""

import torch
import torchvision.transforms.functional as TF
import random


def generate_augmentation_pair(images: torch.Tensor) -> torch.Tensor:
    """Apply random augmentations to create a paired view of the batch.

    Applies a random combination of flips, rotation, and color jitter
    to each image independently.  The augmentations preserve semantic
    content (same class) but change low-level appearance.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images ``(B, C, H, W)``, already normalised.

    Returns
    -------
    torch.Tensor
        Augmented copy, same shape ``(B, C, H, W)``.
    """
    augmented = images.clone()
    B = images.shape[0]

    for i in range(B):
        img = augmented[i]

        # Random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)

        # Random vertical flip
        if random.random() > 0.5:
            img = TF.vflip(img)

        # Random rotation (small angle, satellite images are orientation-invariant)
        angle = random.uniform(-30, 30)
        img = TF.rotate(img.unsqueeze(0), angle).squeeze(0)

        # Mild color jitter (brightness and contrast only)
        brightness = random.uniform(0.85, 1.15)
        contrast = random.uniform(0.85, 1.15)
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)

        augmented[i] = img

    return augmented
