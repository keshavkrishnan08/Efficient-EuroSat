#!/usr/bin/env python3
"""
ONNX export script for EfficientEuroSAT.

Exports a trained EfficientEuroSAT model to ONNX format for deployment.
Early exit is disabled during export since ONNX does not support
dynamic computation graphs with conditional execution.

Usage:
    python export_onnx.py --checkpoint ./checkpoints/best.pth
    python export_onnx.py --checkpoint ./checkpoints/best.pth --output_path ./model.onnx
    python export_onnx.py --checkpoint ./checkpoints/best.pth --opset_version 14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import torch

from src.models.efficient_vit import EfficientEuroSATViT, create_efficient_eurosat_tiny
from src.models.baseline import BaselineViT
from src.utils.helpers import count_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export EfficientEuroSAT model to ONNX format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output ONNX file path (auto-generated if not set)')
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size (from checkpoint if not set)')
    parser.add_argument('--dynamic_batch', action='store_true',
                        help='Enable dynamic batch size in ONNX model')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model (requires onnxsim)')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate exported ONNX model')
    parser.add_argument('--no_validate', action='store_true',
                        help='Skip ONNX model validation')
    return parser.parse_args()


def load_model(checkpoint_path):
    """Load model from checkpoint for export."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'efficient_eurosat')

    if model_type == 'efficient_eurosat':
        # Disable early exit for ONNX export (dynamic control flow not supported)
        model = EfficientEuroSATViT(
            img_size=model_config.get('img_size', 224),
            num_classes=model_config.get('num_classes', 10),
            use_learned_temp=model_config.get('use_learned_temp', True),
            use_early_exit=False,  # Force disabled for export
            use_learned_dropout=model_config.get('use_learned_dropout', True),
            use_learned_residual=model_config.get('use_learned_residual', True),
            use_temp_annealing=model_config.get('use_temp_annealing', True),
            tau_min=model_config.get('tau_min', 0.1),
            dropout_max=model_config.get('dropout_max', 0.3),
        )

        if model_config.get('use_early_exit', False):
            print("NOTE: Early exit has been disabled for ONNX export.")
            print("      ONNX does not support dynamic control flow.")
    elif model_type == 'baseline':
        model = BaselineViT(
            img_size=model_config.get('img_size', 224),
            num_classes=model_config.get('num_classes', 10),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights (strict=False to handle missing early exit params)
    missing, unexpected = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    if missing:
        print(f"  Missing keys (expected for early exit removal): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    return model, model_type, model_config


class ONNXWrapper(torch.nn.Module):
    """Wrapper to ensure clean ONNX-compatible output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # EfficientEuroSAT may return a dict; extract logits
        if isinstance(output, dict):
            return output['logits']
        return output


def export_to_onnx(model, img_size, output_path, opset_version, dynamic_batch):
    """Export the model to ONNX format."""
    # Wrap model for clean output
    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()

    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    # Verify forward pass works
    print("  Verifying forward pass...")
    with torch.no_grad():
        output = wrapped_model(dummy_input)
    print(f"  Output shape: {output.shape}")

    # Set up dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    # Export
    print(f"  Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")

    return file_size_mb


def validate_onnx_model(onnx_path, img_size, pytorch_model):
    """Validate the exported ONNX model."""
    try:
        import onnx
    except ImportError:
        print("  WARNING: onnx package not installed. Skipping validation.")
        print("  Install with: pip install onnx")
        return False

    print("\n  Validating ONNX model...")

    # Check model is well-formed
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("  ONNX model structure validation: PASSED")
    except Exception as e:
        print(f"  ONNX model structure validation: FAILED ({e})")
        return False

    # Print model info
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  Producer: {onnx_model.producer_name}")

    # Print input/output info
    graph = onnx_model.graph
    print(f"  Inputs:")
    for inp in graph.input:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"    {inp.name}: {shape}")

    print(f"  Outputs:")
    for out in graph.output:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"    {out.name}: {shape}")

    # Numerical validation with ONNX Runtime
    try:
        import onnxruntime as ort
    except ImportError:
        print("  WARNING: onnxruntime not installed. Skipping numerical validation.")
        print("  Install with: pip install onnxruntime")
        return True

    print("\n  Running numerical validation with ONNX Runtime...")

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)

    # Generate test inputs
    test_inputs = [
        torch.randn(1, 3, img_size, img_size),
        torch.randn(4, 3, img_size, img_size),
    ]

    wrapped_model = ONNXWrapper(pytorch_model)
    wrapped_model.eval()

    all_close = True
    for i, test_input in enumerate(test_inputs):
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = wrapped_model(test_input).numpy()

        # ONNX Runtime inference
        ort_inputs = {session.get_inputs()[0].name: test_input.numpy()}
        ort_output = session.run(None, ort_inputs)[0]

        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - ort_output))
        mean_diff = np.mean(np.abs(pytorch_output - ort_output))
        is_close = np.allclose(pytorch_output, ort_output, atol=1e-5, rtol=1e-4)

        status = "PASSED" if is_close else "FAILED"
        print(f"  Test {i + 1} (batch={test_input.shape[0]}): {status} "
              f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")

        if not is_close:
            all_close = False

    # Check ONNX Runtime predictions match
    pytorch_preds = np.argmax(pytorch_output, axis=1)
    ort_preds = np.argmax(ort_output, axis=1)
    preds_match = np.all(pytorch_preds == ort_preds)
    print(f"  Prediction consistency: {'PASSED' if preds_match else 'FAILED'}")

    return all_close


def simplify_onnx(onnx_path):
    """Simplify the ONNX model using onnx-simplifier."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("  WARNING: onnxsim not installed. Skipping simplification.")
        print("  Install with: pip install onnxsim")
        return

    print("\n  Simplifying ONNX model...")
    onnx_model = onnx.load(onnx_path)

    original_size = os.path.getsize(onnx_path)

    simplified_model, check = simplify(onnx_model)

    if check:
        # Save simplified model (overwrite)
        onnx.save(simplified_model, onnx_path)
        new_size = os.path.getsize(onnx_path)
        reduction = (1 - new_size / original_size) * 100
        print(f"  Simplified model saved ({reduction:.1f}% size reduction)")
        print(f"  Original: {original_size / 1024 / 1024:.2f} MB")
        print(f"  Simplified: {new_size / 1024 / 1024:.2f} MB")
    else:
        print("  Simplification check failed. Keeping original model.")


def main():
    args = parse_args()

    print("=" * 70)
    print("EfficientEuroSAT ONNX Export")
    print("=" * 70)
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Opset version: {args.opset_version}")

    # Load model
    print("\nLoading model...")
    model, model_type, model_config = load_model(args.checkpoint)
    total_params, trainable_params = count_parameters(model)
    print(f"  Model type: {model_type}")
    print(f"  Parameters: {total_params:,}")

    # Determine image size
    img_size = args.img_size or model_config.get('img_size', 224)
    print(f"  Image size: {img_size}")

    # Determine output path
    if args.output_path is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_path = os.path.join(
            checkpoint_dir, f'{checkpoint_name}.onnx'
        )

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    # Export
    print("\nExporting model...")
    file_size_mb = export_to_onnx(
        model, img_size, args.output_path,
        args.opset_version, args.dynamic_batch
    )

    # Simplify if requested
    if args.simplify:
        simplify_onnx(args.output_path)
        file_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)

    # Validate if requested
    validation_passed = None
    if args.validate and not args.no_validate:
        validation_passed = validate_onnx_model(
            args.output_path, img_size, model
        )

    # Save export metadata
    metadata = {
        'checkpoint': args.checkpoint,
        'output_path': args.output_path,
        'model_type': model_type,
        'model_config': {
            k: v for k, v in model_config.items() if not callable(v)
        },
        'img_size': img_size,
        'opset_version': args.opset_version,
        'dynamic_batch': args.dynamic_batch,
        'file_size_mb': file_size_mb,
        'total_parameters': total_params,
        'early_exit_disabled': True,
        'validation_passed': validation_passed,
        'input_shape': [1, 3, img_size, img_size],
        'output_shape': [1, model_config.get('num_classes', 10)],
    }

    metadata_path = args.output_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"  Model:              {model_type}")
    print(f"  Parameters:         {total_params:,}")
    print(f"  Input shape:        [N, 3, {img_size}, {img_size}]")
    print(f"  Output shape:       [N, {model_config.get('num_classes', 10)}]")
    print(f"  ONNX file:          {args.output_path}")
    print(f"  File size:          {file_size_mb:.2f} MB")
    print(f"  Opset version:      {args.opset_version}")
    print(f"  Dynamic batch:      {args.dynamic_batch}")
    print(f"  Early exit:         Disabled (not supported in ONNX)")
    if validation_passed is not None:
        status = "PASSED" if validation_passed else "FAILED"
        print(f"  Validation:         {status}")
    print(f"  Metadata:           {metadata_path}")
    print("=" * 70)

    # Usage example
    print("\nUsage example (Python):")
    print("  import onnxruntime as ort")
    print("  import numpy as np")
    print(f"  session = ort.InferenceSession('{args.output_path}')")
    print(f"  input_data = np.random.randn(1, 3, {img_size}, {img_size}).astype(np.float32)")
    print("  result = session.run(None, {'input': input_data})")
    print("  predictions = np.argmax(result[0], axis=1)")


if __name__ == '__main__':
    main()
