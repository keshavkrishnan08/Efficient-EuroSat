"""Gaussian blur at multiple severity levels for aleatoric temperature training."""

import torch
import torch.nn.functional as F


# (kernel_size, sigma) for each blur level
BLUR_LEVELS = {
    0: (0, 0.0),     # no blur
    1: (3, 0.5),
    2: (5, 1.0),
    3: (7, 1.5),
    4: (9, 2.0),
}


def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return kernel / kernel.sum()


def apply_gaussian_blur(
    images: torch.Tensor,
    level: int,
) -> torch.Tensor:
    """Apply Gaussian blur at the specified severity level.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images ``(B, C, H, W)``.
    level : int
        Blur level in ``{0, 1, 2, 3, 4}``.  Level 0 returns the
        input unchanged.

    Returns
    -------
    torch.Tensor
        Blurred images, same shape as input.
    """
    if level == 0:
        return images

    kernel_size, sigma = BLUR_LEVELS[level]
    kernel = _gaussian_kernel_2d(kernel_size, sigma).to(images.device, images.dtype)

    # Shape kernel for depthwise convolution: (C, 1, K, K)
    C = images.shape[1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

    pad = kernel_size // 2
    return F.conv2d(images, kernel, padding=pad, groups=C)


def apply_all_blur_levels(
    images: torch.Tensor,
) -> tuple:
    """Apply all blur levels to a batch and return stacked results.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images ``(B, C, H, W)``.

    Returns
    -------
    tuple of (blurred_images, blur_level_tensor)
        blurred_images: ``(B * 5, C, H, W)`` — all blur levels concatenated.
        blur_level_tensor: ``(B * 5,)`` — corresponding blur level per image.
    """
    all_blurred = []
    all_levels = []
    B = images.shape[0]

    for level in range(5):
        blurred = apply_gaussian_blur(images, level)
        all_blurred.append(blurred)
        all_levels.append(torch.full((B,), level, dtype=torch.float32, device=images.device))

    return torch.cat(all_blurred, dim=0), torch.cat(all_levels, dim=0)
