#!/bin/bash
# ==============================================================================
# EfficientEuroSAT — Autonomous IEEE TNNLS Pipeline
# ==============================================================================
#
# Enhanced UCAT: Input-Dependent Temperature + Aleatoric/Epistemic Decomposition
#
# Runs EVERYTHING autonomously in one command:
#   - Environment setup & smoke tests
#   - All training experiments (baseline + ablations + decomposition)
#   - Self-tuning: if benchmarks not met, adjusts hyperparams and retrains
#   - All evaluation & analysis scripts
#   - Publication-quality figure generation
#
# Usage:
#   cd efficient_eurosat && bash run.sh
#
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "######################################################################"
echo "#                                                                    #"
echo "#    EfficientEuroSAT — Autonomous IEEE TNNLS Pipeline               #"
echo "#    Enhanced UCAT with Aleatoric/Epistemic Decomposition            #"
echo "#                                                                    #"
echo "######################################################################"
echo ""

# ------------------------------------------------------------------
# 0. Environment Setup
# ------------------------------------------------------------------
echo "=================================================================="
echo " PHASE 0: Environment Setup"
echo "=================================================================="

echo "[0.1] Installing dependencies..."
pip install -q torch torchvision numpy matplotlib seaborn pyyaml tqdm \
    scikit-learn pandas Pillow 2>&1 | tail -3
echo "  Done."

echo ""
echo "[0.2] Detecting compute device..."
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  CUDA GPU: {name} ({mem:.1f} GB)')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  Apple MPS (Metal)')
else:
    print('  CPU only (training will be slow)')
print(f'  PyTorch: {torch.__version__}')
"

echo ""
echo "[0.3] Creating output directories..."
mkdir -p checkpoints results figures analysis_results

echo ""
echo "[0.4] Verifying model builds (including decomposition)..."
python3 -u -c "
import sys; sys.path.insert(0, '.')
from src.models.efficient_vit import EfficientEuroSATViT
from src.models.baseline import BaselineViT
from src.data.eurosat import get_eurosat_dataloaders
import torch

# Verify baseline model
b = BaselineViT()
x = torch.randn(2, 3, 224, 224)
out_b = b(x)
print(f'  Baseline OK: {sum(p.numel() for p in b.parameters()):,} params, output {out_b.shape}')

# Verify efficient model (no decomposition)
m = EfficientEuroSATViT()
out = m(x, training_progress=0.5)
out_t, temps = m(x, training_progress=0.5, return_temperatures=True)
print(f'  EfficientViT OK: {sum(p.numel() for p in m.parameters()):,} params, output {out.shape}')

# Verify efficient model WITH decomposition
md = EfficientEuroSATViT(use_decomposition=True)
out_d = md(x, training_progress=0.5)
result = md(x, training_progress=0.5, return_temperatures=True)
logits, tau_total, tau_a, tau_e = result
print(f'  Decomposed OK: {sum(p.numel() for p in md.parameters()):,} params')
print(f'    logits: {logits.shape}, tau_a: {tau_a.shape}, tau_e: {tau_e.shape}')
assert tau_a.shape == (2,), f'Expected tau_a shape (2,), got {tau_a.shape}'
print(f'    tau_a values: {tau_a.tolist()}')
print(f'    tau_e value:  {tau_e.item():.4f}')

# Verify data loading
train_l, val_l, test_l, cw = get_eurosat_dataloaders(root='./data', batch_size=4, num_workers=0)
print(f'  Data OK: {len(train_l.dataset)} train, {len(val_l.dataset)} val, {len(test_l.dataset)} test')

# Verify loss functions
from src.training.losses import DecomposedCombinedLoss
criterion = DecomposedCombinedLoss()
dummy_logits = torch.randn(4, 10)
dummy_labels = torch.randint(0, 10, (4,))
dummy_temps = torch.ones(4)
dummy_tau_a = torch.ones(4)
dummy_tau_aug = torch.ones(4) * 1.1
dummy_tau_e = torch.tensor(0.8)
total, l_task, l_ucat, l_a, l_e = criterion(
    dummy_logits, dummy_labels, dummy_temps,
    dummy_tau_a, dummy_tau_aug, dummy_tau_e, 0.5
)
print(f'  DecomposedLoss OK: total={total.item():.4f}')

print('  All checks passed.')
" 2>&1
echo ""

# ------------------------------------------------------------------
# 1. Autonomous Training + Evaluation + Figures
# ------------------------------------------------------------------
echo "=================================================================="
echo " PHASES 1-3: Autonomous Training, Evaluation & Figures"
echo "=================================================================="
echo ""
echo " The autonomous orchestrator will now:"
echo "   1. Train baseline (control)"
echo "   2. Train main decomposed model (with self-tuning)"
echo "   3. Train all ablation experiments"
echo "   4. Train decomposition ablation variants"
echo "   5. Run all evaluation & analysis scripts"
echo "   6. Generate publication-quality figures"
echo ""
echo " Self-tuning: if benchmarks are not met, hyperparameters"
echo " will be adjusted and the experiment retrained (up to 3 retries)."
echo ""

python3 -u scripts/autonomous_orchestrator.py \
    --epochs 100 \
    --batch_size 64 \
    --data_root ./data

echo ""
echo "######################################################################"
echo "#                                                                    #"
echo "#                    ALL EXPERIMENTS COMPLETE                         #"
echo "#                                                                    #"
echo "######################################################################"
echo ""
echo " Outputs:"
echo "   Checkpoints:    ./checkpoints/<experiment_name>/best_model_val_acc.pt"
echo "   Training logs:  ./results/<experiment_name>.json"
echo "   Analysis:       ./analysis_results/"
echo "     accuracy/     - Per-experiment test accuracy"
echo "     decomposition/- tau_a vs tau_e analysis (Figs A,B,E)"
echo "     dynamics/     - Temperature training dynamics (Fig D)"
echo "     ucat/         - Temperature-entropy correlation"
echo "     ood/          - OOD detection EuroSAT vs DTD"
echo "     calibration/  - ECE baseline vs UCAT"
echo "     robustness/   - Corruption robustness"
echo "     latency/      - Inference speed benchmarks"
echo "   Figures:        ./figures/ (publication-ready, 300 DPI)"
echo "   Summary:        ./analysis_results/pipeline_summary.json"
echo ""
echo "######################################################################"
