#!/usr/bin/env python3
"""
Attention visualization script for EfficientEuroSAT.

Generates visualizations of attention patterns and learned parameters
from a trained EfficientEuroSAT model, including:
- Multi-head attention maps overlaid on input images
- Attention rollout across layers
- Learned temperature, dropout, and residual parameter heatmaps
- Per-layer attention entropy analysis

Usage:
    python visualize_attention.py --checkpoint ./checkpoints/best.pth
    python visualize_attention.py --checkpoint ./checkpoints/best.pth --num_samples 20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from src.data.eurosat import get_eurosat_dataloaders, EUROSAT_CLASS_NAMES
from src.models.efficient_vit import EfficientEuroSATViT, create_efficient_eurosat_tiny
from src.models.baseline import BaselineViT
from src.utils.helpers import set_seed, get_device

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. No plots will be generated.")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize attention maps and learned parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--save_dir', type=str, default='./attention_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of sample images to visualize')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'efficient_eurosat')

    if model_type == 'efficient_eurosat':
        model = EfficientEuroSATViT(
            img_size=model_config.get('img_size', 224),
            num_classes=model_config.get('num_classes', 10),
            use_learned_temp=model_config.get('use_learned_temp', True),
            use_early_exit=model_config.get('use_early_exit', True),
            use_learned_dropout=model_config.get('use_learned_dropout', True),
            use_learned_residual=model_config.get('use_learned_residual', True),
            use_temp_annealing=model_config.get('use_temp_annealing', True),
            use_decomposition=model_config.get('use_decomposition', False),
        )
    else:
        model = BaselineViT(
            img_size=model_config.get('img_size', 224),
            num_classes=model_config.get('num_classes', 10),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, model_type, model_config


def get_attention_maps(model, images, device):
    """Extract attention maps from the model."""
    images = images.to(device)
    model.eval()

    attention_maps = []

    # Register hooks to capture attention weights
    hooks = []
    captured_attentions = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Attention modules typically output (attn_output, attn_weights)
            # or store attention weights as an attribute
            if isinstance(output, tuple) and len(output) > 1:
                captured_attentions.append(
                    (layer_idx, output[1].detach().cpu())
                )
            elif hasattr(module, 'attention_weights'):
                captured_attentions.append(
                    (layer_idx, module.attention_weights.detach().cpu())
                )
        return hook_fn

    # Find attention modules and register hooks
    for name, module in model.named_modules():
        if 'attn' in name.lower() and hasattr(module, 'forward'):
            layer_idx = len(hooks)
            hook = module.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        output = model(images)

    # If model has a dedicated method, use that instead
    if hasattr(model, 'get_attention_maps'):
        with torch.no_grad():
            attention_maps = model.get_attention_maps(images)
    elif captured_attentions:
        captured_attentions.sort(key=lambda x: x[0])
        attention_maps = [attn for _, attn in captured_attentions]

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return attention_maps, output


def compute_attention_rollout(attention_maps):
    """Compute attention rollout across all layers."""
    if not attention_maps:
        return None

    # attention_maps: list of [batch, heads, seq_len, seq_len]
    result = None
    for attn in attention_maps:
        if attn.dim() == 4:
            # Average across heads
            attn_avg = attn.mean(dim=1)  # [batch, seq_len, seq_len]
        else:
            attn_avg = attn

        # Add identity (residual connection)
        identity = torch.eye(attn_avg.size(-1)).unsqueeze(0)
        identity = identity.expand_as(attn_avg)
        attn_with_residual = (attn_avg + identity) / 2

        # Normalize rows
        attn_with_residual = attn_with_residual / attn_with_residual.sum(
            dim=-1, keepdim=True
        )

        if result is None:
            result = attn_with_residual
        else:
            result = torch.bmm(attn_with_residual, result)

    return result


def denormalize_image(image, mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)):
    """Denormalize an image tensor for display."""
    img = image.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def plot_attention_per_sample(image, attention_maps, label, pred,
                              class_names, save_path, img_size=224):
    """Plot attention maps for a single sample across all layers."""
    if not HAS_MATPLOTLIB:
        return

    img = denormalize_image(image)
    num_layers = len(attention_maps)
    cols = min(4, num_layers + 1)
    rows = (num_layers + cols) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    # Original image
    axes[0, 0].imshow(img)
    class_name = class_names[label] if label < len(class_names) else f'Class {label}'
    pred_name = class_names[pred] if pred < len(class_names) else f'Class {pred}'
    correct = "OK" if label == pred else "WRONG"
    axes[0, 0].set_title(f'Input\nTrue: {class_name}\nPred: {pred_name} ({correct})',
                         fontsize=8)
    axes[0, 0].axis('off')

    # Attention maps per layer
    for i, attn in enumerate(attention_maps):
        row = (i + 1) // cols
        col = (i + 1) % cols

        if attn.dim() >= 3:
            # Average across heads if multi-head
            if attn.dim() == 3:
                attn_map = attn.mean(dim=0)  # [seq_len, seq_len]
            else:
                attn_map = attn

            # Extract CLS token attention to patch tokens
            if attn_map.size(0) > 1:
                cls_attn = attn_map[0, 1:]  # CLS attending to patches
            else:
                cls_attn = attn_map[0]

            # Reshape to spatial grid
            num_patches = cls_attn.size(0)
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size == num_patches:
                attn_spatial = cls_attn.reshape(grid_size, grid_size).numpy()
            else:
                attn_spatial = cls_attn.numpy().reshape(-1)[:grid_size * grid_size]
                attn_spatial = attn_spatial.reshape(grid_size, grid_size)

            axes[row, col].imshow(img)
            axes[row, col].imshow(
                np.array(
                    plt.cm.jet(
                        np.interp(
                            np.kron(
                                attn_spatial,
                                np.ones((img_size // grid_size,
                                         img_size // grid_size))
                            ),
                            [attn_spatial.min(), attn_spatial.max()],
                            [0, 1]
                        )
                    )
                ),
                alpha=0.5
            )
            axes[row, col].set_title(f'Layer {i + 1}', fontsize=9)
        axes[row, col].axis('off')

    # Hide unused subplots
    for idx in range(num_layers + 1, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_rollout_grid(images, rollout_maps, labels, preds,
                                class_names, save_path, img_size=224):
    """Plot attention rollout for multiple samples in a grid."""
    if not HAS_MATPLOTLIB:
        return

    num_samples = len(images)
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))

    if num_samples == 1:
        axes = axes[:, np.newaxis]

    for i in range(num_samples):
        img = denormalize_image(images[i])

        # Original image
        axes[0, i].imshow(img)
        class_name = class_names[int(labels[i])] if int(labels[i]) < len(class_names) else f'{labels[i]}'
        axes[0, i].set_title(class_name, fontsize=8)
        axes[0, i].axis('off')

        # Rollout attention
        if rollout_maps is not None and rollout_maps.size(0) > i:
            rollout = rollout_maps[i]
            if rollout.dim() >= 2:
                cls_attn = rollout[0, 1:]  # CLS to patches
            else:
                cls_attn = rollout[1:]

            num_patches = cls_attn.size(0)
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size == num_patches:
                attn_spatial = cls_attn.reshape(grid_size, grid_size).numpy()
                # Upsample to image size
                attn_upsampled = np.kron(
                    attn_spatial,
                    np.ones((img_size // grid_size, img_size // grid_size))
                )
                attn_norm = (attn_upsampled - attn_upsampled.min()) / \
                            (attn_upsampled.max() - attn_upsampled.min() + 1e-8)

                axes[1, i].imshow(img)
                axes[1, i].imshow(attn_norm, cmap='jet', alpha=0.5)
            else:
                axes[1, i].imshow(img)

        correct = "OK" if int(labels[i]) == int(preds[i]) else "X"
        pred_name = class_names[int(preds[i])] if int(preds[i]) < len(class_names) else f'{preds[i]}'
        axes[1, i].set_title(f'Pred: {pred_name} ({correct})', fontsize=8)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Input', fontsize=10)
    axes[1, 0].set_ylabel('Attention Rollout', fontsize=10)

    plt.suptitle('Attention Rollout Visualization', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learned_parameter_heatmaps(model, save_dir):
    """Generate heatmaps for all learned parameters in the model."""
    if not HAS_MATPLOTLIB:
        return

    param_groups = {}

    for name, param in model.named_parameters():
        if param.requires_grad and param.numel() < 10000:
            # Collect small learned parameters (temperatures, scales, etc.)
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in
                   ['temp', 'tau', 'dropout', 'drop_rate', 'residual',
                    'alpha', 'scale', 'gate']):
                data = param.detach().cpu().numpy().flatten()
                category = 'temperature' if 'temp' in name_lower or 'tau' in name_lower \
                    else 'dropout' if 'drop' in name_lower \
                    else 'residual' if 'resid' in name_lower or 'alpha' in name_lower \
                    else 'other'
                if category not in param_groups:
                    param_groups[category] = []
                param_groups[category].append((name, data))

    if not param_groups:
        print("  No learned parameters found to visualize.")
        return

    # Create a figure for each category
    for category, params in param_groups.items():
        num_params = len(params)
        fig, axes = plt.subplots(
            1, num_params, figsize=(4 * num_params, 4), squeeze=False
        )

        for i, (name, data) in enumerate(params):
            ax = axes[0, i]

            if len(data) > 1:
                # Try to reshape into a 2D grid
                n = len(data)
                side = int(np.ceil(np.sqrt(n)))
                padded = np.zeros(side * side)
                padded[:n] = data
                grid = padded.reshape(side, side)

                im = ax.imshow(grid, cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.bar([0], data, color='steelblue')
                ax.set_xlim(-0.5, 0.5)

            short_name = name.split('.')[-1] if '.' in name else name
            ax.set_title(f'{short_name}\n({name})', fontsize=7, wrap=True)

        plt.suptitle(f'Learned {category.title()} Parameters', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'learned_{category}_params.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    # Summary plot: all parameters in one figure
    fig, ax = plt.subplots(figsize=(12, 6))

    all_names = []
    all_means = []
    all_stds = []
    colors = []
    color_map = {
        'temperature': '#2196F3',
        'dropout': '#4CAF50',
        'residual': '#FF9800',
        'other': '#9C27B0',
    }

    for category, params in param_groups.items():
        for name, data in params:
            short_name = name.replace('transformer.', '').replace('blocks.', 'B')
            all_names.append(short_name)
            all_means.append(np.mean(data))
            all_stds.append(np.std(data))
            colors.append(color_map.get(category, '#757575'))

    x = np.arange(len(all_names))
    ax.bar(x, all_means, yerr=all_stds, capsize=3, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Parameter Value')
    ax.set_title('All Learned Parameters Summary')
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=cat.title())
        for cat, color in color_map.items()
        if cat in param_groups
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_learned_params_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_attention_entropy(attention_maps):
    """Compute the entropy of attention distributions per layer."""
    entropies = []
    for attn in attention_maps:
        if attn.dim() == 4:
            # [batch, heads, seq, seq]
            # Compute entropy per head per sample
            attn_flat = attn.clamp(min=1e-8)
            entropy = -(attn_flat * attn_flat.log()).sum(dim=-1)  # [B, H, S]
            avg_entropy = entropy.mean().item()
        elif attn.dim() == 3:
            attn_flat = attn.clamp(min=1e-8)
            entropy = -(attn_flat * attn_flat.log()).sum(dim=-1)
            avg_entropy = entropy.mean().item()
        else:
            avg_entropy = 0.0
        entropies.append(avg_entropy)
    return entropies


def plot_attention_entropy(entropies, save_path):
    """Plot attention entropy per layer."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    layers = list(range(1, len(entropies) + 1))
    ax.plot(layers, entropies, 'o-', color='steelblue', linewidth=2,
            markersize=8)
    ax.fill_between(layers, entropies, alpha=0.2, color='steelblue')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Attention Entropy')
    ax.set_title('Attention Entropy Across Layers')
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 70)
    print("EfficientEuroSAT Attention Visualization")
    print("=" * 70)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Device:      {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    model, model_type, model_config = load_model(args.checkpoint, device)
    print(f"Model type: {model_type}")

    # Load test data
    print("\nLoading test data...")
    _, _, test_loader, _ = get_eurosat_dataloaders(
        root=args.data_root,
        img_size=model_config.get('img_size', args.img_size),
        batch_size=args.num_samples,
        num_workers=2,
    )

    # Get sample images
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:args.num_samples]
    sample_labels = sample_labels[:args.num_samples]
    print(f"Selected {len(sample_images)} samples")

    # Get attention maps
    print("\nExtracting attention maps...")
    attention_maps, outputs = get_attention_maps(model, sample_images, device)

    if isinstance(outputs, dict):
        logits = outputs['logits']
    else:
        logits = outputs
    preds = logits.argmax(dim=-1).cpu().numpy()

    if attention_maps:
        print(f"  Captured {len(attention_maps)} attention layers")

        # 1. Per-sample attention visualization
        print("\nGenerating per-sample attention visualizations...")
        sample_dir = os.path.join(args.save_dir, 'per_sample')
        os.makedirs(sample_dir, exist_ok=True)

        for i in range(len(sample_images)):
            sample_attns = [
                attn[i] if attn.dim() >= 3 else attn
                for attn in attention_maps
            ]
            save_path = os.path.join(sample_dir, f'sample_{i:03d}_attention.png')
            plot_attention_per_sample(
                sample_images[i].cpu(),
                sample_attns,
                int(sample_labels[i]),
                int(preds[i]),
                EUROSAT_CLASS_NAMES,
                save_path,
                img_size=model_config.get('img_size', args.img_size),
            )
        print(f"  Saved {len(sample_images)} per-sample plots to: {sample_dir}")

        # 2. Attention rollout
        print("\nComputing attention rollout...")
        rollout = compute_attention_rollout(attention_maps)
        if rollout is not None:
            rollout_path = os.path.join(args.save_dir, 'attention_rollout.png')
            plot_attention_rollout_grid(
                sample_images[:min(8, len(sample_images))].cpu(),
                rollout[:min(8, len(sample_images))],
                sample_labels[:min(8, len(sample_images))].numpy(),
                preds[:min(8, len(sample_images))],
                EUROSAT_CLASS_NAMES,
                rollout_path,
                img_size=model_config.get('img_size', args.img_size),
            )
            print(f"  Saved: {rollout_path}")

        # 3. Attention entropy analysis
        print("\nComputing attention entropy...")
        entropies = compute_attention_entropy(attention_maps)
        entropy_path = os.path.join(args.save_dir, 'attention_entropy.png')
        plot_attention_entropy(entropies, entropy_path)
        print(f"  Saved: {entropy_path}")
        for i, ent in enumerate(entropies):
            print(f"    Layer {i + 1}: entropy = {ent:.4f}")
    else:
        print("  No attention maps captured (model may not expose attention weights)")

    # 4. Learned parameter heatmaps
    if model_type == 'efficient_eurosat':
        print("\nGenerating learned parameter visualizations...")
        params_dir = os.path.join(args.save_dir, 'learned_params')
        os.makedirs(params_dir, exist_ok=True)
        plot_learned_parameter_heatmaps(model, params_dir)

    # Save metadata
    metadata = {
        'checkpoint': args.checkpoint,
        'model_type': model_type,
        'num_samples': len(sample_images),
        'num_attention_layers': len(attention_maps),
        'predictions': preds.tolist(),
        'labels': sample_labels.numpy().tolist(),
        'accuracy': float(np.mean(preds == sample_labels.numpy())),
    }
    if attention_maps:
        metadata['attention_entropies'] = entropies

    metadata_path = os.path.join(args.save_dir, 'visualization_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll visualizations saved to: {args.save_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
