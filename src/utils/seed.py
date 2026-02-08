"""
Seed and Device Utilities for EfficientEuroSAT.

Provides reproducibility controls and automatic device selection
for consistent experimental results across runs.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across all sources of randomness for full reproducibility.

    Configures deterministic behavior for PyTorch, NumPy, Python's random module,
    and CUDA backends. This ensures that experiments produce identical results
    across runs given the same seed value.

    Args:
        seed: Integer seed value. Default is 42.

    Note:
        Setting deterministic mode and disabling cudnn.benchmark may reduce
        training throughput by 10-20%, but guarantees bitwise reproducibility.
        For final benchmarking runs where speed matters more than exact
        reproducibility, consider re-enabling benchmark mode after seeding.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic algorithms in cuDNN
    torch.backends.cudnn.deterministic = True

    # Disable cuDNN auto-tuner that finds the fastest convolution algorithms
    # (auto-tuner introduces non-determinism)
    torch.backends.cudnn.benchmark = False

    # Control hash-based randomness in Python
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """
    Automatically select the best available compute device.

    Selection priority:
        1. CUDA (NVIDIA GPU) - preferred for training and inference
        2. MPS (Apple Silicon GPU) - fallback for macOS with M-series chips
        3. CPU - universal fallback

    Returns:
        torch.device: The selected device object ready for tensor placement.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
