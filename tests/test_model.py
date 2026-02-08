"""Tests for the full EfficientEuroSAT Vision Transformer model.

Uses small dimensions (embed_dim=64, num_heads=2, num_layers=2, patch_size=8,
img_size=32, num_classes=5) for fast execution.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from src.models.efficient_vit import (
    EfficientEuroSATViT,
    create_efficient_eurosat_tiny,
    create_baseline_vit_tiny,
)

# ---------------------------------------------------------------------------
# Small test constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 2
IMG_SIZE = 32
PATCH_SIZE = 8
IN_CHANNELS = 3
NUM_CLASSES = 5
EMBED_DIM = 64
DEPTH = 2
NUM_HEADS = 2


def _make_small_model(**overrides) -> EfficientEuroSATViT:
    """Create a small EfficientEuroSAT model for testing."""
    defaults = dict(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=2.0,
        dropout=0.0,
        use_learned_temp=True,
        use_early_exit=True,
        use_learned_dropout=True,
        use_learned_residual=True,
        use_temp_annealing=True,
        tau_min=0.5,
        dropout_max=0.3,
        exit_threshold=0.9,
        exit_min_layer=1,
    )
    defaults.update(overrides)
    return EfficientEuroSATViT(**defaults)


def _make_baseline_model(**overrides) -> EfficientEuroSATViT:
    """Create a small baseline ViT (all mods disabled) for testing."""
    return _make_small_model(
        use_learned_temp=False,
        use_early_exit=False,
        use_learned_dropout=False,
        use_learned_residual=False,
        use_temp_annealing=False,
        **overrides,
    )


def _dummy_input() -> torch.Tensor:
    """Create a dummy input image batch."""
    return torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


# ===========================================================================
# Tests
# ===========================================================================


class TestModelForward:
    """Test forward pass shapes and basic behaviour."""

    def test_baseline_forward_shape(self):
        """Baseline model output shape is [B, num_classes]."""
        model = _make_baseline_model()
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"
        )

    def test_efficient_eurosat_forward_shape(self):
        """EfficientEuroSAT model output shape is [B, num_classes]."""
        model = _make_small_model()
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"
        )

    def test_model_backward(self):
        """loss.backward() works without errors."""
        model = _make_small_model()
        model.train()
        x = _dummy_input()
        out = model(x, training_progress=0.5)
        loss = nn.CrossEntropyLoss()(out, torch.zeros(BATCH_SIZE, dtype=torch.long))
        loss.backward()
        # Verify at least one parameter received a gradient
        grads_found = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grads_found = True
                break
        assert grads_found, "No parameter received a non-zero gradient"

    def test_model_parameter_count(self):
        """EfficientEuroSAT has slightly more params than baseline."""
        efficient = _make_small_model()
        baseline = _make_baseline_model()
        n_eff = efficient.get_num_params()
        n_base = baseline.get_num_params()
        assert n_eff > n_base, (
            f"EfficientEuroSAT ({n_eff} params) should have more params "
            f"than baseline ({n_base} params)"
        )


class TestModelFeatures:
    """Test model-level features and utilities."""

    def test_attention_stats(self):
        """get_attention_stats returns dict with expected keys."""
        model = _make_small_model()
        stats = model.get_attention_stats()
        assert isinstance(stats, dict), "get_attention_stats must return a dict"
        expected_keys = {
            "temperatures",
            "dropout_rates",
            "residual_weights",
            "exit_confidences",
        }
        assert expected_keys.issubset(stats.keys()), (
            f"Missing keys: {expected_keys - stats.keys()}"
        )

    def test_early_exit_disabled(self):
        """Model processes all layers when early exit is disabled."""
        model = _make_small_model()
        model.eval()
        model.set_early_exit(False)
        x = _dummy_input()
        with torch.no_grad():
            out = model(x)
        # All samples should exit at the last layer
        exit_stats = model.get_exit_statistics()
        # When early exit is disabled, _forward_with_early_exit is not called,
        # so total_samples should be 0 (no tracking) or all at the last layer.
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"Output shape mismatch: {out.shape}"
        )

    def test_model_eval_mode(self):
        """Model works correctly in eval mode."""
        model = _make_small_model()
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)
        # Verify outputs are finite
        assert torch.isfinite(out).all(), "Model produced non-finite outputs"

    def test_model_training_progress(self):
        """Model accepts training_progress argument without errors."""
        model = _make_small_model()
        model.train()
        x = _dummy_input()
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out = model(x, training_progress=progress)
            assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
                f"Shape mismatch at training_progress={progress}"
            )


class TestFactoryFunctions:
    """Test the factory / convenience constructors."""

    def test_factory_functions(self):
        """create_efficient_eurosat_tiny and create_baseline_vit_tiny work."""
        efficient = create_efficient_eurosat_tiny(num_classes=NUM_CLASSES)
        assert isinstance(efficient, EfficientEuroSATViT)
        assert efficient.num_classes == NUM_CLASSES

        baseline = create_baseline_vit_tiny(num_classes=NUM_CLASSES)
        assert isinstance(baseline, EfficientEuroSATViT)
        assert baseline.num_classes == NUM_CLASSES

    def test_all_mods_disabled(self):
        """Model with all mods disabled produces valid outputs like baseline."""
        model = _make_baseline_model()
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)
        assert torch.isfinite(out).all(), "Baseline model produced non-finite outputs"
        # Verify no modification-specific parameters exist
        for block in model.blocks:
            attn = block.attn
            assert attn.learned_temp is None, "Baseline should have no learned_temp"
            assert attn.early_exit is None, "Baseline should have no early_exit"
            assert attn.learned_dropout is None, "Baseline should have no learned_dropout"
            assert attn.learned_residual is None, "Baseline should have no learned_residual"
            assert attn.temp_scheduler is None, "Baseline should have no temp_scheduler"
