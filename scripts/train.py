#!/usr/bin/env python3
"""
Main training script for EfficientEuroSAT.

Trains either the EfficientEuroSAT model (with learned modifications) or
a baseline ViT model on the EuroSAT satellite land use classification dataset.

Usage:
    python train.py --model efficient_eurosat --epochs 100 --batch_size 64
    python train.py --model baseline --epochs 100
    python train.py --model efficient_eurosat --no_learned_temp --no_early_exit
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
import datetime
import numpy as np
import torch

from src.data.eurosat import get_eurosat_dataloaders
from src.models.efficient_vit import EfficientEuroSATViT, create_efficient_eurosat_tiny, create_baseline_vit_tiny
from src.models.baseline import BaselineViT
from src.training.trainer import EuroSATTrainer
from src.utils.helpers import set_seed, get_device, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train EfficientEuroSAT or baseline model on EuroSAT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument('--model', type=str, default='efficient_eurosat',
                        choices=['efficient_eurosat', 'baseline'],
                        help='Model architecture to train')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Load ImageNet-pretrained ViT-Tiny weights (default: True)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Disable pretrained weight loading (train from scratch)')
    parser.add_argument('--mixup_alpha', type=float, default=0.8,
                        help='MixUp alpha (0 to disable)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix alpha (0 to disable)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay for optimizer')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')

    # Modification toggles (store_true to DISABLE)
    parser.add_argument('--no_learned_temp', action='store_true',
                        help='Disable learned temperature scaling')
    parser.add_argument('--no_early_exit', action='store_true',
                        help='Disable early exit mechanism')
    parser.add_argument('--no_learned_dropout', action='store_true',
                        help='Disable learned dropout rates')
    parser.add_argument('--no_learned_residual', action='store_true',
                        help='Disable learned residual scaling')
    parser.add_argument('--no_temp_annealing', action='store_true',
                        help='Disable temperature annealing schedule')

    # Modification hyperparameters
    parser.add_argument('--tau_min', type=float, default=0.1,
                        help='Minimum temperature value')
    parser.add_argument('--dropout_max', type=float, default=0.3,
                        help='Maximum learned dropout rate')
    parser.add_argument('--exit_threshold', type=float, default=0.9,
                        help='Confidence threshold for early exit')
    parser.add_argument('--exit_min_layer', type=int, default=4,
                        help='Minimum layer before early exit is allowed')

    # Experiment settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--wandb_project', type=str, default='efficient_eurosat',
                        help='Weights & Biases project name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--lambda_ucat', type=float, default=0.1,
                        help='Weight for UCAT loss (0 to disable)')

    # Decomposition settings
    parser.add_argument('--use_decomposition', action='store_true',
                        help='Enable aleatoric/epistemic temperature decomposition')
    parser.add_argument('--lambda_aleatoric', type=float, default=0.05,
                        help='Weight for aleatoric consistency/blur loss')
    parser.add_argument('--lambda_epistemic', type=float, default=0.05,
                        help='Weight for epistemic rarity/decay loss')
    parser.add_argument('--blur_loss_frequency', type=int, default=10,
                        help='Apply blur correlation loss every N batches')

    return parser.parse_args()


def generate_experiment_name(args):
    """Generate a descriptive experiment name from the configuration."""
    parts = [args.model]

    if args.model == 'efficient_eurosat':
        mods = []
        if not args.no_learned_temp:
            mods.append('temp')
        if not args.no_early_exit:
            mods.append('exit')
        if not args.no_learned_dropout:
            mods.append('drop')
        if not args.no_learned_residual:
            mods.append('resid')
        if not args.no_temp_annealing:
            mods.append('anneal')

        if len(mods) == 5:
            parts.append('all_mods')
        elif len(mods) == 0:
            parts.append('no_mods')
        else:
            parts.append('_'.join(mods))

    parts.append(f'ep{args.epochs}')
    parts.append(f'bs{args.batch_size}')
    parts.append(f'lr{args.lr}')
    parts.append(f'seed{args.seed}')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parts.append(timestamp)

    return '-'.join(parts)


def build_model(args, device):
    """Build the model based on command-line arguments."""
    num_classes = 10  # EuroSAT has 10 classes
    use_pretrained = args.pretrained and not args.no_pretrained

    if args.model == 'efficient_eurosat':
        model = EfficientEuroSATViT(
            img_size=args.img_size,
            num_classes=num_classes,
            use_learned_temp=not args.no_learned_temp,
            use_early_exit=not args.no_early_exit,
            use_learned_dropout=not args.no_learned_dropout,
            use_learned_residual=not args.no_learned_residual,
            use_temp_annealing=not args.no_temp_annealing,
            tau_min=args.tau_min,
            dropout_max=args.dropout_max,
            exit_threshold=args.exit_threshold,
            exit_min_layer=args.exit_min_layer,
            use_decomposition=args.use_decomposition,
        )
        if use_pretrained:
            from src.models.pretrained import load_pretrained_efficient
            load_pretrained_efficient(model)
    elif args.model == 'baseline':
        model = BaselineViT(
            img_size=args.img_size,
            num_classes=num_classes,
        )
        if use_pretrained:
            from src.models.pretrained import load_pretrained_baseline
            load_pretrained_baseline(model)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(device)
    return model


def main():
    args = parse_args()

    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = generate_experiment_name(args)

    print("=" * 70)
    print("EfficientEuroSAT Training Script")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Model:      {args.model}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Seed:       {args.seed}")
    print()

    # Set seed and get device
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Load EuroSAT data
    print("\nLoading EuroSAT dataset...")
    train_loader, val_loader, test_loader, class_weights = get_eurosat_dataloaders(
        root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Create model
    print(f"\nBuilding {args.model} model...")
    model = build_model(args, device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    if args.model == 'efficient_eurosat':
        print("\n  Modifications enabled:")
        print(f"    Learned Temperature:  {not args.no_learned_temp}")
        print(f"    Early Exit:           {not args.no_early_exit}")
        print(f"    Learned Dropout:      {not args.no_learned_dropout}")
        print(f"    Learned Residual:     {not args.no_learned_residual}")
        print(f"    Temp Annealing:       {not args.no_temp_annealing}")
        print(f"    Decomposition:        {args.use_decomposition}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Compute class rarity for epistemic loss
    class_rarity = None
    if args.use_decomposition:
        from src.data.class_weights import compute_class_rarity
        class_rarity = compute_class_rarity(train_loader.dataset)

    # Create trainer
    use_amp = device.type == 'cuda'  # AMP only beneficial on CUDA
    trainer = EuroSATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        class_weights=class_weights,
        device=device,
        save_dir=args.save_dir,
        use_amp=use_amp,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.experiment_name,
        lambda_ucat=args.lambda_ucat,
        use_decomposition=args.use_decomposition,
        lambda_aleatoric=args.lambda_aleatoric,
        lambda_epistemic=args.lambda_epistemic,
        blur_loss_frequency=args.blur_loss_frequency,
        class_rarity=class_rarity,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    train_start = time.time()
    train_results = trainer.train(num_epochs=args.epochs)
    train_duration = time.time() - train_start
    print(f"\nTraining completed in {train_duration / 60:.1f} minutes")

    # Test evaluation
    print("\n" + "=" * 70)
    print("Running test evaluation...")
    print("=" * 70)
    test_results = trainer.test()

    # Compile all results
    results = {
        'experiment_name': args.experiment_name,
        'model': args.model,
        'args': vars(args),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'train_duration_seconds': train_duration,
        'training': train_results,
        'test': test_results,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    # Save results to JSON
    results_dir = os.path.join(os.path.dirname(args.save_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'{args.experiment_name}.json')

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert_for_json)
    print(f"\nResults saved to: {results_path}")

    # Print final metrics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Model:              {args.model}")
    print(f"  Test Accuracy:      {test_results.get('test_acc', 0) * 100:.2f}%")
    print(f"  Test Top-5 Acc:     {test_results.get('test_top5_acc', 0) * 100:.2f}%")
    print(f"  Best Val Accuracy:  {train_results.get('best_val_acc', 0) * 100:.2f}%")
    print(f"  Total Parameters:   {total_params:,}")
    print(f"  Training Time:      {train_duration / 60:.1f} minutes")

    if args.model == 'efficient_eurosat' and not args.no_early_exit:
        avg_exit = test_results.get('avg_exit_layer', None)
        if avg_exit is not None:
            print(f"  Avg Exit Layer:     {avg_exit:.2f}")
            exit_ratio = test_results.get('early_exit_ratio', None)
            if exit_ratio is not None:
                print(f"  Early Exit Ratio:   {exit_ratio * 100:.1f}%")

    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
