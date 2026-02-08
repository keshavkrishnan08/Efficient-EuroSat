"""
Checkpoint Utilities for EfficientEuroSAT.

Provides save/load functionality for full training state (model, optimizer,
scheduler, epoch, metrics) and standalone model weights, enabling seamless
training resumption and model deployment.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    path: str
) -> None:
    """
    Save a complete training checkpoint for later resumption.

    Persists the full training state including model weights, optimizer
    state (momentum buffers, adaptive learning rates), scheduler state,
    current epoch, and recorded metrics.

    Args:
        model: The model whose weights to save.
        optimizer: The optimizer whose state to save (includes per-parameter
            states like momentum and adaptive learning rate terms).
        scheduler: Optional learning rate scheduler. If None, scheduler
            state is saved as None.
        epoch: Current epoch number (0-indexed).
        metrics: Dictionary of metric values at checkpoint time (e.g.,
            {'val_loss': 0.23, 'val_acc': 0.91}).
        path: File path where the checkpoint will be saved. Parent
            directories are created automatically if they do not exist.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': (
            scheduler.state_dict() if scheduler is not None else None
        ),
        'metrics': metrics,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None
) -> Dict:
    """
    Load a training checkpoint and restore model/optimizer/scheduler states.

    Restores the full training state from a previously saved checkpoint.
    The model weights are always loaded; optimizer and scheduler states
    are only restored if the corresponding objects are provided.

    Args:
        path: Path to the checkpoint file.
        model: Model instance to load weights into. Must have the same
            architecture as the model that was saved.
        optimizer: Optional optimizer to restore state into. If None,
            optimizer state from the checkpoint is ignored.
        scheduler: Optional scheduler to restore state into. If None,
            scheduler state from the checkpoint is ignored.

    Returns:
        dict: Dictionary containing the saved metrics and epoch:
            {
                'epoch': int,
                'metrics': dict
            }

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Restore model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore scheduler state if provided
    if (
        scheduler is not None
        and 'scheduler_state_dict' in checkpoint
        and checkpoint['scheduler_state_dict'] is not None
    ):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
    }


def save_model(model: nn.Module, path: str) -> None:
    """
    Save only the model weights (state dict) without optimizer state.

    This produces a smaller file than a full checkpoint and is suitable
    for model deployment, sharing, or inference-only use cases.

    Args:
        model: The model whose weights to save.
        path: File path where the model weights will be saved. Parent
            directories are created automatically if they do not exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, strict: bool = True) -> nn.Module:
    """
    Load model weights from a saved state dict file.

    Args:
        model: Model instance to load weights into. Must have a compatible
            architecture with the saved weights.
        path: Path to the saved model weights file.
        strict: If True (default), requires an exact match between the
            model's parameter names and the saved state dict keys. Set to
            False when loading partial weights or when the model has been
            modified (e.g., added classification head).

    Returns:
        nn.Module: The model with loaded weights.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        RuntimeError: If strict=True and there is a mismatch between
            model parameters and saved state dict keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found: {path}")

    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=strict)

    return model
