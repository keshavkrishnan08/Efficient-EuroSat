"""
Visualization Utilities for EfficientEuroSAT.

Provides tools for visualizing attention maps, learned temperature parameters,
adaptive dropout rates, and residual connection weights. These visualizations
are essential for understanding and interpreting the behavior of the
Temperature-Scaled Representations model.
"""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


def visualize_attention(
    model: nn.Module,
    image: torch.Tensor,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Extract and visualize attention maps overlaid on the input image.

    Uses forward hooks to capture attention weights from the specified
    transformer layer. Displays the attention from the [CLS] token to
    all spatial patches, reshaped into a 2D grid and upsampled to match
    the original image dimensions.

    Args:
        model: The EfficientEuroSAT model (must have a .blocks attribute
            containing transformer blocks, each with an .attn sub-module
            that stores attention weights in .attn_weights).
        image: Input image tensor of shape (C, H, W) or (1, C, H, W).
            Should be normalized according to model preprocessing.
        layer_idx: Which transformer layer to visualize. If None,
            defaults to the last layer.
        head_idx: Which attention head to visualize. If None, attention
            is averaged across all heads.
        save_path: Optional file path to save the figure. If None,
            the plot is displayed interactively.
    """
    model.eval()
    device = next(model.parameters()).device

    # Ensure image has batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    # Storage for captured attention weights
    captured_attn = {}

    def _make_hook(name):
        def hook_fn(module, input, output):
            # Expect the attention module to store weights as .attn_weights
            # or return them as part of output
            if hasattr(module, 'attn_weights'):
                captured_attn[name] = module.attn_weights.detach().cpu()
        return hook_fn

    # Determine which layers to hook
    blocks = list(model.blocks) if hasattr(model, 'blocks') else []
    if not blocks:
        # Try alternate attribute names
        for attr in ['encoder_blocks', 'layers', 'encoder']:
            if hasattr(model, attr):
                blocks = list(getattr(model, attr))
                break

    if not blocks:
        raise AttributeError(
            "Model must have a 'blocks', 'encoder_blocks', 'layers', or "
            "'encoder' attribute containing transformer blocks."
        )

    # Register hooks
    hooks = []
    for i, block in enumerate(blocks):
        attn_module = block.attn if hasattr(block, 'attn') else block
        hook = attn_module.register_forward_hook(_make_hook(f"block_{i}"))
        hooks.append(hook)

    # Forward pass to capture attention
    with torch.no_grad():
        _ = model(image)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Select the target layer
    num_layers = len(blocks)
    if layer_idx is None:
        layer_idx = num_layers - 1
    target_key = f"block_{layer_idx}"

    if target_key not in captured_attn:
        raise RuntimeError(
            f"No attention weights captured for layer {layer_idx}. "
            f"Ensure the attention module stores weights in .attn_weights "
            f"during forward pass."
        )

    # attn shape: (batch, num_heads, seq_len, seq_len)
    attn = captured_attn[target_key]

    if head_idx is not None:
        # Select specific head: (batch, seq_len, seq_len)
        attn = attn[:, head_idx, :, :]
        title_suffix = f"Layer {layer_idx}, Head {head_idx}"
    else:
        # Average across heads: (batch, seq_len, seq_len)
        attn = attn.mean(dim=1)
        title_suffix = f"Layer {layer_idx}, All Heads (avg)"

    # Extract [CLS] token attention to all patches (exclude CLS-to-CLS)
    # attn shape: (batch, seq_len, seq_len), CLS is token 0
    cls_attn = attn[0, 0, 1:]  # (num_patches,)

    # Reshape to 2D grid (assume square patch grid, e.g. 14x14 = 196 patches)
    num_patches = cls_attn.shape[0]
    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        # Handle non-square cases by finding closest factors
        grid_size = int(np.ceil(num_patches ** 0.5))
        # Pad if necessary
        padded = torch.zeros(grid_size * grid_size)
        padded[:num_patches] = cls_attn
        cls_attn = padded

    attn_map = cls_attn.reshape(grid_size, grid_size).numpy()

    # Upsample attention map to image size
    img_h, img_w = image.shape[-2], image.shape[-1]
    attn_map_upsampled = np.array(
        F.interpolate(
            torch.tensor(attn_map).unsqueeze(0).unsqueeze(0).float(),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
    )

    # Prepare the original image for display
    img_display = image[0].cpu()
    # Denormalize with ImageNet stats as default
    img_display = denormalize_image(
        img_display,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_display)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Attention map
    im = axes[1].imshow(attn_map_upsampled, cmap='jet')
    axes[1].set_title(f"Attention Map\n{title_suffix}")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(attn_map_upsampled, cmap='jet', alpha=0.5)
    axes[2].set_title(f"Attention Overlay\n{title_suffix}")
    axes[2].axis('off')

    plt.suptitle("EfficientEuroSAT Attention Visualization", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_learned_temperatures(
    model: nn.Module,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the learned temperature (tau) values as a heatmap.

    Displays a layers-by-heads heatmap showing how the temperature scaling
    parameter has been learned across the model. Higher tau values produce
    softer attention distributions; lower values produce sharper ones.

    Args:
        model: The EfficientEuroSAT model. Each transformer block must have
            an attention module with a learnable 'tau' or 'temperature'
            parameter.
        save_path: Optional file path to save the figure. If None,
            the plot is displayed interactively.
    """
    tau_values = []

    blocks = _get_blocks(model)
    for block in blocks:
        attn = block.attn if hasattr(block, 'attn') else block
        # Look for temperature parameter under common names
        tau = None
        for param_name in ['tau', 'temperature', 'temp']:
            if hasattr(attn, param_name):
                tau = getattr(attn, param_name).detach().cpu()
                break

        if tau is not None:
            # tau may be scalar, per-head, or other shape
            if tau.dim() == 0:
                tau = tau.unsqueeze(0)
            tau_values.append(tau.flatten().numpy())

    if not tau_values:
        raise AttributeError(
            "No learned temperature parameters found in model. "
            "Ensure attention modules have a 'tau' or 'temperature' attribute."
        )

    # Build matrix: (num_layers, num_heads)
    num_heads = max(len(row) for row in tau_values)
    tau_matrix = np.zeros((len(tau_values), num_heads))
    for i, row in enumerate(tau_values):
        tau_matrix[i, :len(row)] = row

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(4, len(tau_values) * 0.5 + 1)))
    sns.heatmap(
        tau_matrix,
        annot=True,
        fmt=".3f",
        cmap='YlOrRd',
        xticklabels=[f"Head {i}" for i in range(num_heads)],
        yticklabels=[f"Layer {i}" for i in range(len(tau_values))],
        ax=ax,
        cbar_kws={'label': 'Temperature (tau)'}
    )
    ax.set_title("Learned Temperature Parameters (tau)\nper Layer and Head")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Transformer Layer")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_learned_dropouts(
    model: nn.Module,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the learned dropout rates as a heatmap.

    Displays a layers-by-heads heatmap showing how adaptive dropout rates
    have been learned. Higher dropout values indicate the model has learned
    to regularize certain heads more aggressively, potentially because they
    contribute less to the final representation.

    Args:
        model: The EfficientEuroSAT model. Each transformer block must have
            an attention module with a learnable 'drop_rate' or
            'dropout_rate' parameter.
        save_path: Optional file path to save the figure. If None,
            the plot is displayed interactively.
    """
    dropout_values = []

    blocks = _get_blocks(model)
    for block in blocks:
        attn = block.attn if hasattr(block, 'attn') else block
        # Look for dropout parameter under common names
        drop = None
        for param_name in ['drop_rate', 'dropout_rate', 'learned_dropout',
                           'adaptive_dropout']:
            if hasattr(attn, param_name):
                drop = getattr(attn, param_name).detach().cpu()
                break

        if drop is not None:
            if drop.dim() == 0:
                drop = drop.unsqueeze(0)
            # Apply sigmoid to get values in [0, 1] if stored as logits
            drop_val = drop.flatten().numpy()
            dropout_values.append(drop_val)

    if not dropout_values:
        raise AttributeError(
            "No learned dropout parameters found in model. "
            "Ensure attention modules have a 'drop_rate' or "
            "'dropout_rate' attribute."
        )

    # Build matrix: (num_layers, num_heads)
    num_heads = max(len(row) for row in dropout_values)
    drop_matrix = np.zeros((len(dropout_values), num_heads))
    for i, row in enumerate(dropout_values):
        drop_matrix[i, :len(row)] = row

    # Plot heatmap
    fig, ax = plt.subplots(
        figsize=(10, max(4, len(dropout_values) * 0.5 + 1))
    )
    sns.heatmap(
        drop_matrix,
        annot=True,
        fmt=".3f",
        cmap='Blues',
        vmin=0.0,
        vmax=1.0,
        xticklabels=[f"Head {i}" for i in range(num_heads)],
        yticklabels=[f"Layer {i}" for i in range(len(dropout_values))],
        ax=ax,
        cbar_kws={'label': 'Dropout Rate'}
    )
    ax.set_title("Learned Dropout Rates\nper Layer and Head")
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Transformer Layer")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_residual_weights(
    model: nn.Module,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the learned residual connection weights (alpha) as a bar chart.

    Shows one bar per transformer layer, indicating how much each layer's
    output contributes relative to the skip connection. An alpha of 1.0
    means the residual branch contributes equally with the skip; values
    below 1.0 indicate the model has learned to dampen that layer's
    contribution.

    Args:
        model: The EfficientEuroSAT model. Each transformer block must have
            a learnable 'alpha', 'residual_weight', or 'layer_scale'
            parameter.
        save_path: Optional file path to save the figure. If None,
            the plot is displayed interactively.
    """
    alpha_values = []
    layer_names = []

    blocks = _get_blocks(model)
    for i, block in enumerate(blocks):
        alpha = None
        for param_name in ['alpha', 'residual_weight', 'layer_scale',
                           'res_weight', 'gamma']:
            if hasattr(block, param_name):
                alpha = getattr(block, param_name).detach().cpu()
                break

        if alpha is not None:
            # Handle scalar or vector alpha
            if alpha.dim() == 0:
                alpha_values.append(alpha.item())
            else:
                alpha_values.append(alpha.mean().item())
            layer_names.append(f"Layer {i}")

    if not alpha_values:
        raise AttributeError(
            "No learned residual weight parameters found in model. "
            "Ensure transformer blocks have an 'alpha', 'residual_weight', "
            "or 'layer_scale' attribute."
        )

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(alpha_values) * 0.8), 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(alpha_values)))
    bars = ax.bar(layer_names, alpha_values, color=colors, edgecolor='black',
                  linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, alpha_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    ax.set_title("Learned Residual Connection Weights (alpha)\nper Layer")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Alpha Value")
    ax.set_ylim(0, max(alpha_values) * 1.15 if alpha_values else 1.0)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='alpha=1.0')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_all_learned_params(
    model: nn.Module,
    save_dir: Optional[str] = None
) -> None:
    """
    Generate all learned parameter visualizations in one call.

    Produces temperature heatmap, dropout heatmap, and residual weight
    bar chart. Each visualization is saved as a separate file when
    save_dir is provided, or displayed interactively otherwise.

    Args:
        model: The EfficientEuroSAT model with learned parameters.
        save_dir: Optional directory path where all figures will be saved.
            Files are named descriptively (e.g., 'temperatures.png').
            If None, plots are displayed interactively one at a time.
    """
    visualization_funcs = [
        (visualize_learned_temperatures, "temperatures.png"),
        (visualize_learned_dropouts, "dropout_rates.png"),
        (visualize_residual_weights, "residual_weights.png"),
    ]

    for viz_func, filename in visualization_funcs:
        try:
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, filename)
                viz_func(model, save_path=path)
            else:
                viz_func(model)
        except AttributeError as e:
            # Skip visualizations for parameters that don't exist in the model
            print(f"Skipping {filename}: {e}")


def denormalize_image(
    tensor: torch.Tensor,
    mean: List[float],
    std: List[float]
) -> np.ndarray:
    """
    Convert a normalized image tensor back to a displayable NumPy array.

    Reverses the standard normalization (x_norm = (x - mean) / std) and
    clips values to [0, 1] for proper display with matplotlib.

    Args:
        tensor: Normalized image tensor of shape (C, H, W).
        mean: Per-channel mean values used during normalization
            (e.g., [0.485, 0.456, 0.406] for ImageNet).
        std: Per-channel std values used during normalization
            (e.g., [0.229, 0.224, 0.225] for ImageNet).

    Returns:
        np.ndarray: Denormalized image as a (H, W, C) array with values
            clipped to [0, 1], ready for plt.imshow().
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    # Reverse normalization: x = x_norm * std + mean
    img = tensor.cpu().clone() * std + mean

    # Clip to valid range and convert to (H, W, C)
    img = torch.clamp(img, 0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()

    return img


def _get_blocks(model: nn.Module) -> list:
    """
    Extract transformer blocks from the model under various attribute names.

    Args:
        model: The model to extract blocks from.

    Returns:
        list: List of transformer block modules.

    Raises:
        AttributeError: If no blocks attribute can be found.
    """
    for attr in ['blocks', 'encoder_blocks', 'layers', 'encoder']:
        if hasattr(model, attr):
            return list(getattr(model, attr))

    raise AttributeError(
        "Model must have a 'blocks', 'encoder_blocks', 'layers', or "
        "'encoder' attribute containing transformer blocks."
    )
