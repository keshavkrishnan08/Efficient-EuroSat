"""Pretrained weight loading for ViT-Tiny from timm (ImageNet-1K).

Downloads and maps timm's `vit_tiny_patch16_224` weights to our custom
EfficientEuroSATViT and BaselineViT architectures. Only backbone weights
are transferred — the classification head is re-initialized for EuroSAT
(10 classes), and all UCAT modification parameters start from scratch.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_timm_state_dict() -> Dict[str, torch.Tensor]:
    """Download ViT-Tiny pretrained weights via timm."""
    import timm
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    return model.state_dict()


def load_pretrained_efficient(model: nn.Module) -> None:
    """Load ImageNet-pretrained ViT-Tiny weights into EfficientEuroSATViT.

    Our model's naming matches timm almost 1:1 for the shared backbone:
        patch_embed.proj, cls_token, pos_embed, blocks.{i}.norm1/2,
        blocks.{i}.attn.qkv, blocks.{i}.attn.proj, blocks.{i}.mlp.fc1/fc2,
        norm.

    The classification head (``head``) is skipped (different num_classes).
    All UCAT modification parameters (temperatures, dropout gates, residual
    weights, exit controllers, temp_predictor) are left at their random init.

    Parameters
    ----------
    model : nn.Module
        An ``EfficientEuroSATViT`` instance.
    """
    timm_sd = _get_timm_state_dict()
    model_sd = model.state_dict()

    loaded, skipped = 0, 0
    new_sd = {}

    for key, param in timm_sd.items():
        # Skip classification head — we have 10 classes, not 1000
        if key.startswith("head."):
            skipped += 1
            continue
        # Skip timm-specific keys not in our model
        if key not in model_sd:
            skipped += 1
            continue
        # Skip shape mismatches (shouldn't happen for Tiny but be safe)
        if param.shape != model_sd[key].shape:
            logger.warning(
                "Shape mismatch for %s: timm %s vs model %s — skipping",
                key, param.shape, model_sd[key].shape,
            )
            skipped += 1
            continue
        new_sd[key] = param
        loaded += 1

    model.load_state_dict(new_sd, strict=False)
    total_model = len(model_sd)
    logger.info(
        "Pretrained: loaded %d/%d params from timm ViT-Tiny (skipped %d timm keys, "
        "%d model keys untouched)",
        loaded, total_model, skipped, total_model - loaded,
    )
    print(
        f"  Pretrained: loaded {loaded}/{total_model} backbone params from "
        f"ImageNet ViT-Tiny ({total_model - loaded} modification params init from scratch)"
    )


def load_pretrained_baseline(model: nn.Module) -> None:
    """Load ImageNet-pretrained ViT-Tiny weights into BaselineViT.

    The baseline has minor naming differences from timm:
        - ``patch_embed`` is a raw Conv2d, timm uses ``patch_embed.proj``
        - MLP uses ``nn.Sequential`` with indexed keys (``mlp.0``, ``mlp.3``),
          timm uses named keys (``mlp.fc1``, ``mlp.fc2``)

    Parameters
    ----------
    model : nn.Module
        A ``BaselineViT`` instance.
    """
    timm_sd = _get_timm_state_dict()
    model_sd = model.state_dict()

    # Build explicit key mapping: timm_key -> baseline_key
    key_map: Dict[str, str] = {}

    # Patch embed: timm uses patch_embed.proj, baseline uses patch_embed directly
    key_map["patch_embed.proj.weight"] = "patch_embed.weight"
    key_map["patch_embed.proj.bias"] = "patch_embed.bias"

    # CLS token, pos_embed, pos_drop (same naming)
    key_map["cls_token"] = "cls_token"
    key_map["pos_embed"] = "pos_embed"

    # Final norm (same naming)
    key_map["norm.weight"] = "norm.weight"
    key_map["norm.bias"] = "norm.bias"

    # Blocks
    num_layers = getattr(model, "num_layers", 12)
    for i in range(num_layers):
        # Norm layers
        key_map[f"blocks.{i}.norm1.weight"] = f"blocks.{i}.norm1.weight"
        key_map[f"blocks.{i}.norm1.bias"] = f"blocks.{i}.norm1.bias"
        key_map[f"blocks.{i}.norm2.weight"] = f"blocks.{i}.norm2.weight"
        key_map[f"blocks.{i}.norm2.bias"] = f"blocks.{i}.norm2.bias"

        # Attention QKV and projection
        key_map[f"blocks.{i}.attn.qkv.weight"] = f"blocks.{i}.attn.qkv.weight"
        key_map[f"blocks.{i}.attn.qkv.bias"] = f"blocks.{i}.attn.qkv.bias"
        key_map[f"blocks.{i}.attn.proj.weight"] = f"blocks.{i}.attn.proj.weight"
        key_map[f"blocks.{i}.attn.proj.bias"] = f"blocks.{i}.attn.proj.bias"

        # MLP: timm uses fc1/fc2, baseline uses Sequential indices 0/3
        key_map[f"blocks.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.0.weight"
        key_map[f"blocks.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.0.bias"
        key_map[f"blocks.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.3.weight"
        key_map[f"blocks.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.3.bias"

    loaded, skipped = 0, 0
    new_sd = {}

    for timm_key, param in timm_sd.items():
        if timm_key.startswith("head."):
            skipped += 1
            continue
        target_key = key_map.get(timm_key)
        if target_key is None or target_key not in model_sd:
            skipped += 1
            continue
        if param.shape != model_sd[target_key].shape:
            logger.warning(
                "Shape mismatch for %s -> %s: %s vs %s — skipping",
                timm_key, target_key, param.shape, model_sd[target_key].shape,
            )
            skipped += 1
            continue
        new_sd[target_key] = param
        loaded += 1

    model.load_state_dict(new_sd, strict=False)
    total_model = len(model_sd)
    print(
        f"  Pretrained: loaded {loaded}/{total_model} backbone params from "
        f"ImageNet ViT-Tiny"
    )
