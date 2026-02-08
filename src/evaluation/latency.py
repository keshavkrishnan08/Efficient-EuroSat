"""
Latency and throughput benchmarking for EfficientEuroSAT models.

Provides precise latency measurement using CUDA events (GPU) or
``time.perf_counter`` (CPU), with proper warmup and synchronization.
Also includes utilities for comparing multiple models, benchmarking
with real data and early exit, and estimating FLOPs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_logits(
    output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> torch.Tensor:
    """Extract classification logits from model output.

    The model may return a plain tensor or a tuple where the first
    element is the logits tensor.

    Parameters
    ----------
    output : torch.Tensor or tuple of torch.Tensor
        Raw model output.

    Returns
    -------
    torch.Tensor
        Logits tensor.
    """
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _is_cuda(device: str) -> bool:
    """Check whether *device* refers to a CUDA device."""
    return "cuda" in str(device).lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def benchmark_latency(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cuda",
    num_warmup: int = 50,
    num_runs: int = 200,
) -> Dict[str, float]:
    """Benchmark single-sample inference latency.

    Performs ``num_warmup`` untimed forward passes followed by
    ``num_runs`` timed passes.  On CUDA devices, timing uses
    ``torch.cuda.Event`` for sub-millisecond accuracy; on CPU,
    ``time.perf_counter`` is used instead.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark.
    input_size : tuple of int, optional
        Shape of the synthetic input tensor, typically
        ``(batch, channels, height, width)``.  Default is
        ``(1, 3, 224, 224)``.
    device : str, optional
        Device on which to run the benchmark.  Default is ``'cuda'``.
    num_warmup : int, optional
        Number of warmup iterations (not timed).  Default is ``50``.
    num_runs : int, optional
        Number of timed iterations.  Default is ``200``.

    Returns
    -------
    dict[str, float]
        Dictionary with the following keys (all values in milliseconds):

        - ``'mean_ms'``: Mean latency.
        - ``'std_ms'``: Standard deviation of latency.
        - ``'min_ms'``: Minimum latency.
        - ``'max_ms'``: Maximum latency.
        - ``'p50_ms'``: Median (50th percentile) latency.
        - ``'p95_ms'``: 95th percentile latency.
        - ``'p99_ms'``: 99th percentile latency.
    """
    model.eval()
    model.to(device)

    use_cuda = _is_cuda(device)
    dummy_input = torch.randn(*input_size, device=device)

    # ---- Warmup phase ----
    with torch.no_grad():
        for _ in range(num_warmup):
            _extract_logits(model(dummy_input))

    if use_cuda:
        torch.cuda.synchronize()

    # ---- Timed phase ----
    timings_ms: List[float] = []

    with torch.no_grad():
        for _ in range(num_runs):
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _extract_logits(model(dummy_input))
                end_event.record()

                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                start = time.perf_counter()
                _extract_logits(model(dummy_input))
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000.0

            timings_ms.append(elapsed_ms)

    arr = np.array(timings_ms)

    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def benchmark_with_early_exit(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Benchmark latency on real data with early exit enabled.

    Unlike :func:`benchmark_latency`, this function feeds actual
    samples from the dataloader so that the model's early exit
    mechanism can trigger on inputs of varying difficulty.  It records
    per-sample latency as well as the exit layer distribution.

    Parameters
    ----------
    model : nn.Module
        Model with early exit support.  The model is expected to
        optionally expose an ``exit_layer`` attribute or return
        ``(logits, exit_info)`` tuples.
    dataloader : DataLoader
        Data loader yielding ``(images, labels)`` batches.
    device : str, optional
        Device to run inference on.  Default is ``'cuda'``.

    Returns
    -------
    dict
        Dictionary containing:

        - ``'mean_ms'`` (float): Mean per-sample latency.
        - ``'std_ms'`` (float): Standard deviation of per-sample latency.
        - ``'min_ms'`` (float): Minimum per-sample latency.
        - ``'max_ms'`` (float): Maximum per-sample latency.
        - ``'p50_ms'`` (float): Median per-sample latency.
        - ``'p95_ms'`` (float): 95th percentile per-sample latency.
        - ``'p99_ms'`` (float): 99th percentile per-sample latency.
        - ``'total_samples'`` (int): Total samples processed.
        - ``'exit_layer_distribution'`` (dict): Count of samples exiting
          at each layer, or ``{}`` if the model does not report exit
          layers.
    """
    model.eval()
    model.to(device)

    use_cuda = _is_cuda(device)
    sample_timings_ms: List[float] = []
    exit_layers: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            batch_size = images.size(0)

            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                output = model(images)
                end_event.record()

                torch.cuda.synchronize()
                batch_ms = start_event.elapsed_time(end_event)
            else:
                start = time.perf_counter()
                output = model(images)
                end = time.perf_counter()
                batch_ms = (end - start) * 1000.0

            # Per-sample latency (approximate)
            per_sample_ms = batch_ms / batch_size
            sample_timings_ms.extend([per_sample_ms] * batch_size)

            # Attempt to capture exit layer information
            if isinstance(output, (tuple, list)) and len(output) > 1:
                exit_info = output[1]
                if isinstance(exit_info, torch.Tensor):
                    # If exit_info is a per-sample exit layer tensor
                    exit_layers.extend(exit_info.cpu().tolist())
                elif isinstance(exit_info, (int, float)):
                    exit_layers.extend([int(exit_info)] * batch_size)

            # Also check model attributes for exit layer tracking
            if hasattr(model, "last_exit_layer"):
                layer_val = model.last_exit_layer
                if isinstance(layer_val, torch.Tensor):
                    exit_layers.extend(layer_val.cpu().tolist())
                elif isinstance(layer_val, (int, float)):
                    exit_layers.extend([int(layer_val)] * batch_size)

    arr = np.array(sample_timings_ms)

    # Build exit layer distribution
    exit_distribution: Dict[int, int] = {}
    if exit_layers:
        for layer in exit_layers:
            layer_int = int(layer)
            exit_distribution[layer_int] = (
                exit_distribution.get(layer_int, 0) + 1
            )

    result: Dict[str, Any] = {
        "mean_ms": float(np.mean(arr)) if len(arr) > 0 else 0.0,
        "std_ms": float(np.std(arr)) if len(arr) > 0 else 0.0,
        "min_ms": float(np.min(arr)) if len(arr) > 0 else 0.0,
        "max_ms": float(np.max(arr)) if len(arr) > 0 else 0.0,
        "p50_ms": float(np.percentile(arr, 50)) if len(arr) > 0 else 0.0,
        "p95_ms": float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if len(arr) > 0 else 0.0,
        "total_samples": len(sample_timings_ms),
        "exit_layer_distribution": exit_distribution,
    }

    return result


def compare_latency(
    models_dict: Dict[str, nn.Module],
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """Compare inference latency across multiple models.

    Runs :func:`benchmark_latency` on each model in the dictionary and
    returns a nested dictionary of timing statistics keyed by model
    name.

    Parameters
    ----------
    models_dict : dict[str, nn.Module]
        Dictionary mapping descriptive model names to model instances.
    input_size : tuple of int, optional
        Shape of the synthetic input tensor.  Default is
        ``(1, 3, 224, 224)``.
    device : str, optional
        Device on which to run the benchmarks.  Default is ``'cuda'``.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dictionary: ``{model_name: latency_stats}``, where
        ``latency_stats`` has the same keys as the output of
        :func:`benchmark_latency`.
    """
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models_dict.items():
        results[name] = benchmark_latency(
            model=model,
            input_size=input_size,
            device=device,
        )

    return results


def estimate_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
) -> Dict[str, Any]:
    """Estimate the floating-point operations (FLOPs) for a single forward pass.

    Attempts to use the ``thop`` library (``torchprofile`` alternative)
    for accurate counting.  If ``thop`` is not installed, falls back to
    a simple parameter-based heuristic: ``FLOPs ~ 2 * MACs ~ 2 * params``
    (a very rough approximation suitable for comparing relative model
    sizes).

    Parameters
    ----------
    model : nn.Module
        Model to profile.
    input_size : tuple of int, optional
        Shape of the synthetic input tensor.  Default is
        ``(1, 3, 224, 224)``.

    Returns
    -------
    dict
        Dictionary containing:

        - ``'total_flops'`` (int or float): Estimated total FLOPs.
        - ``'total_params'`` (int): Total number of parameters.
        - ``'trainable_params'`` (int): Number of trainable parameters.
        - ``'method'`` (str): Either ``'thop'`` or ``'parameter_estimate'``
          indicating how FLOPs were computed.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # Attempt to use thop for accurate profiling
    try:
        from thop import profile as thop_profile

        dummy_input = torch.randn(*input_size)
        model_copy = model.cpu().eval()
        flops, params = thop_profile(model_copy, inputs=(dummy_input,), verbose=False)

        return {
            "total_flops": int(flops),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "method": "thop",
        }
    except ImportError:
        pass

    # Fallback: rough estimate based on parameter count
    # For linear layers: FLOPs ~ 2 * in_features * out_features per sample
    # This is a coarse heuristic useful only for relative comparisons.
    estimated_flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            estimated_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            # FLOPs = 2 * Cin * Cout * Kh * Kw * Hout * Wout
            # We estimate Hout * Wout from input_size (very rough)
            k_h, k_w = module.kernel_size
            c_in = module.in_channels
            c_out = module.out_channels
            groups = module.groups
            # Rough spatial estimate based on input dimensions
            h_out = input_size[2] if len(input_size) > 2 else 1
            w_out = input_size[3] if len(input_size) > 3 else 1
            estimated_flops += (
                2 * (c_in // groups) * c_out * k_h * k_w * h_out * w_out
            )

    # If no structured layers found, fall back to 2 * params
    if estimated_flops == 0:
        estimated_flops = 2 * total_params

    return {
        "total_flops": int(estimated_flops),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "method": "parameter_estimate",
    }
