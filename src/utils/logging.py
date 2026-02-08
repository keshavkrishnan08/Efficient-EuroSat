"""
Logging Utilities for EfficientEuroSAT.

Provides structured logging with console and file output, metric formatting,
and model summary reporting for experiment tracking.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Create and configure a named logger with console and optional file output.

    Sets up a logger with a standardized format that includes timestamps,
    logger name, log level, and the message. Prevents duplicate handlers
    when the same logger name is requested multiple times.

    Args:
        name: Logger name, typically the module or experiment name.
        log_file: Optional path to a log file. If provided, all log messages
            are also written to this file.
        level: Logging level (e.g., logging.INFO, logging.DEBUG). Default
            is logging.INFO.

    Returns:
        logging.Logger: Configured logger instance ready for use.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler - always present
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - optional
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_metrics(
    logger: logging.Logger,
    metrics_dict: Dict[str, float],
    prefix: str = ''
) -> None:
    """
    Log a dictionary of metrics in a clean, aligned format.

    Formats each metric on its own line with consistent alignment for
    easy reading in both console output and log files.

    Args:
        logger: Logger instance to write to.
        metrics_dict: Dictionary mapping metric names to their values.
            Values are formatted to 6 decimal places for floats.
        prefix: Optional prefix string prepended to the log header
            (e.g., 'Train', 'Val', 'Test').

    Example output:
        [Train Metrics]
          loss          : 0.234500
          accuracy      : 0.912300
          learning_rate : 0.000100
    """
    header = f"[{prefix} Metrics]" if prefix else "[Metrics]"
    logger.info(header)

    if not metrics_dict:
        logger.info("  (no metrics to report)")
        return

    # Find the longest key for alignment
    max_key_len = max(len(str(k)) for k in metrics_dict)

    for key, value in metrics_dict.items():
        if isinstance(value, float):
            formatted_value = f"{value:.6f}"
        else:
            formatted_value = str(value)
        logger.info(f"  {str(key):<{max_key_len}} : {formatted_value}")


def log_model_summary(logger: logging.Logger, model: nn.Module) -> None:
    """
    Log a comprehensive summary of the model architecture.

    Reports total parameter count, trainable vs frozen parameter counts,
    estimated model size in MB, number of top-level layers, and a
    breakdown of parameters per top-level module.

    Args:
        logger: Logger instance to write to.
        model: PyTorch model to summarize.
    """
    logger.info("=" * 60)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 60)

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    logger.info(f"  Total parameters     : {total_params:,}")
    logger.info(f"  Trainable parameters : {trainable_params:,}")
    logger.info(f"  Frozen parameters    : {frozen_params:,}")

    # Model size estimate (assuming float32)
    param_size_mb = total_params * 4 / (1024 ** 2)
    logger.info(f"  Estimated size (FP32): {param_size_mb:.2f} MB")

    # Count top-level modules (children)
    children = list(model.named_children())
    logger.info(f"  Top-level layers     : {len(children)}")

    # Per-module parameter breakdown
    if children:
        logger.info("")
        logger.info("  Per-module breakdown:")
        max_name_len = max(len(name) for name, _ in children) if children else 0
        for name, module in children:
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            pct = (module_params / total_params * 100) if total_params > 0 else 0
            logger.info(
                f"    {name:<{max_name_len}} : {module_params:>12,} params "
                f"({module_trainable:,} trainable, {pct:.1f}%)"
            )

    logger.info("=" * 60)
