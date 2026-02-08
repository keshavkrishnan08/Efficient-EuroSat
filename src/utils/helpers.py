"""Convenience re-exports and small utility helpers.

This module exists so that scripts can do a single import::

    from src.utils.helpers import set_seed, get_device, count_parameters

without needing to know which sub-module each function lives in.
"""

from __future__ import annotations

import torch.nn as nn

from .seed import set_seed, get_device

__all__ = ["set_seed", "get_device", "count_parameters"]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to inspect.

    Returns
    -------
    tuple[int, int]
        ``(total_parameters, trainable_parameters)``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
