#!/bin/bash
# EfficientEuroSAT: Run All Experiments
# Usage: bash scripts/run_all_experiments.sh [--quick]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse args
QUICK=false
if [ "$1" == "--quick" ]; then
    QUICK=true
    EPOCHS=10
    echo "Quick mode: 10 epochs per run"
else
    EPOCHS=100
    echo "Full mode: 100 epochs per run"
fi

echo "================================"
echo "EfficientEuroSAT Experiment Suite"
echo "================================"

# Step 1: Run unit tests
echo "[1/6] Running unit tests..."
python -m pytest tests/ -v --tb=short
echo "Tests passed!"

# Step 2: Train baseline
echo "[2/6] Training baseline..."
python scripts/train.py --model baseline --epochs $EPOCHS --seed 42 --no_wandb --experiment_name baseline_s42
python scripts/train.py --model baseline --epochs $EPOCHS --seed 123 --no_wandb --experiment_name baseline_s123
python scripts/train.py --model baseline --epochs $EPOCHS --seed 456 --no_wandb --experiment_name baseline_s456

# Step 3: Train full EfficientEuroSAT
echo "[3/6] Training EfficientEuroSAT (all modifications)..."
python scripts/train.py --model efficient_eurosat --epochs $EPOCHS --seed 42 --no_wandb --experiment_name efficient_eurosat_s42
python scripts/train.py --model efficient_eurosat --epochs $EPOCHS --seed 123 --no_wandb --experiment_name efficient_eurosat_s123
python scripts/train.py --model efficient_eurosat --epochs $EPOCHS --seed 456 --no_wandb --experiment_name efficient_eurosat_s456

# Step 4: Run full ablation study
echo "[4/6] Running ablation study..."
python scripts/ablation_study.py --epochs $EPOCHS --no_wandb

# Step 5: Benchmark latency
echo "[5/6] Benchmarking latency..."
python scripts/benchmark_latency.py --save_dir results/

# Step 6: Generate visualizations
echo "[6/6] Generating visualizations..."
# Only if a checkpoint exists
if ls checkpoints/efficient_eurosat*.pt 1>/dev/null 2>&1; then
    CKPT=$(ls -t checkpoints/efficient_eurosat*.pt | head -1)
    python scripts/visualize_attention.py --checkpoint "$CKPT" --save_dir figures/
    python scripts/analyze_early_exit.py --checkpoint "$CKPT" --save_dir figures/
fi

echo "================================"
echo "All experiments complete!"
echo "Results saved to: results/"
echo "Figures saved to: figures/"
echo "================================"
