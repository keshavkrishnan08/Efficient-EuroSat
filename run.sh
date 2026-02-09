#!/bin/bash
# ==============================================================================
# EfficientEuroSAT — Autonomous IEEE TNNLS Pipeline
# ==============================================================================
#
# Enhanced UCAT: Input-Dependent Temperature + Aleatoric/Epistemic Decomposition
# Multi-Dataset (EuroSAT, CIFAR-100, RESISC45) | Multi-Architecture (Tiny, Small)
# 35 experiments | 3 seeds | Uncertainty baselines | Statistical significance
#
# Runs EVERYTHING autonomously in one command:
#   - Environment setup & smoke tests
#   - All 35 training experiments (baselines, ablations, decomposition,
#     multi-dataset, multi-architecture, multi-seed)
#   - Self-tuning: if benchmarks not met, adjusts hyperparams and retrains
#   - All evaluation & analysis scripts (per-dataset)
#   - Uncertainty method comparison (MC Dropout, Deep Ensemble, Post-hoc Temp)
#   - Statistical significance tests (paired t-tests, LaTeX tables)
#   - Theoretical analysis (gradient independence, convergence)
#   - Publication-quality figure generation (12 figures, IEEE TNNLS format)
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
echo "#    Multi-Dataset | Multi-Architecture | 35 Experiments             #"
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
pip install -q torch torchvision timm numpy matplotlib seaborn pyyaml tqdm \
    scikit-learn pandas Pillow scipy 2>&1 | tail -3
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
echo "[0.4] Verifying model builds (all architectures and datasets)..."
python3 -u -c "
import sys; sys.path.insert(0, '.')
from src.models.efficient_vit import EfficientEuroSATViT
from src.models.baseline import BaselineViT, create_baseline_vit_small
from src.data.datasets import get_dataloaders, get_dataset_info
import torch

# Verify baseline model (ViT-Tiny)
b = BaselineViT()
x = torch.randn(2, 3, 224, 224)
out_b = b(x)
print(f'  Baseline Tiny OK: {sum(p.numel() for p in b.parameters()):,} params, output {out_b.shape}')

# Verify baseline model (ViT-Small)
bs = create_baseline_vit_small(num_classes=10)
out_bs = bs(x)
print(f'  Baseline Small OK: {sum(p.numel() for p in bs.parameters()):,} params, output {out_bs.shape}')

# Verify efficient model (no decomposition)
m = EfficientEuroSATViT()
out = m(x, training_progress=0.5)
out_t, temps = m(x, training_progress=0.5, return_temperatures=True)
print(f'  EfficientViT Tiny OK: {sum(p.numel() for p in m.parameters()):,} params, output {out.shape}')

# Verify efficient model ViT-Small
ms = EfficientEuroSATViT(embed_dim=384, depth=12, num_heads=6)
out_ms = ms(x, training_progress=0.5)
print(f'  EfficientViT Small OK: {sum(p.numel() for p in ms.parameters()):,} params, output {out_ms.shape}')

# Verify efficient model WITH decomposition
md = EfficientEuroSATViT(use_decomposition=True)
out_d = md(x, training_progress=0.5)
result = md(x, training_progress=0.5, return_temperatures=True)
logits, tau_total, tau_a, tau_e = result
print(f'  Decomposed OK: {sum(p.numel() for p in md.parameters()):,} params')
print(f'    logits: {logits.shape}, tau_a: {tau_a.shape}, tau_e: {tau_e.shape}')

# Verify dataset info for all datasets
for ds in ['eurosat', 'cifar100', 'resisc45']:
    info = get_dataset_info(ds)
    print(f'  Dataset {ds}: {info[\"num_classes\"]} classes')

# Verify EuroSAT data loading
train_l, val_l, test_l, cw = get_dataloaders(dataset_name='eurosat', root='./data', batch_size=4, num_workers=0)
print(f'  EuroSAT data OK: {len(train_l.dataset)} train, {len(val_l.dataset)} val, {len(test_l.dataset)} test')

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
echo " PHASES 1-5: Autonomous Training, Evaluation & Figures"
echo "=================================================================="
echo ""
echo " The autonomous orchestrator will now:"
echo "   1. Train EuroSAT baselines (3 seeds for Deep Ensemble)"
echo "   2. Train main decomposed model (with self-tuning)"
echo "   3. Train all ablation experiments (seed 42 + seeds 123/456)"
echo "   4. Train decomposition ablation variants"
echo "   5. Train CIFAR-100 experiments (baseline + combined + decomp)"
echo "   6. Train RESISC45 experiments (baseline + combined + decomp)"
echo "   7. Train ViT-Small experiments (baseline + combined + decomp)"
echo "   8. Run per-dataset evaluation & analysis scripts"
echo "   9. Run uncertainty method comparison (UCAT vs MC Dropout vs Ensemble vs Post-hoc)"
echo "  10. Run statistical significance tests (paired t-tests + LaTeX tables)"
echo "  11. Run theoretical analysis (gradient independence + convergence)"
echo "  12. Generate publication-quality figures (12 figures, 300 DPI)"
echo ""
echo " Total: 35 experiments across 3 datasets, 2 architectures, 3 seeds"
echo ""
echo " Self-tuning: if benchmarks are not met, hyperparameters"
echo " will be adjusted and the experiment retrained (up to 3 retries)."
echo ""

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo " Using: EPOCHS=$EPOCHS, BATCH_SIZE=$BATCH_SIZE"
echo " (Override with: EPOCHS=200 BATCH_SIZE=32 bash run.sh)"
echo ""

python3 -u scripts/autonomous_orchestrator.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
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
echo "     accuracy/     - Per-experiment test accuracy (all datasets)"
echo "     decomposition/- tau_a vs tau_e analysis"
echo "     dynamics/     - Temperature training dynamics"
echo "     ucat/         - Temperature-entropy correlation"
echo "     ood/          - OOD detection"
echo "     calibration/  - ECE baseline vs UCAT"
echo "     robustness/   - Corruption robustness"
echo "     latency/      - Inference speed benchmarks"
echo "     uncertainty_*/ - Uncertainty method comparisons per dataset"
echo "     significance/ - Statistical significance tests + LaTeX tables"
echo "     theoretical/  - Gradient independence + convergence analysis"
echo "   Figures:        ./figures/ (publication-ready, 300 DPI, 12 figures)"
echo "   Summary:        ./analysis_results/pipeline_summary.json"
echo ""
echo " Experiment matrix:"
echo "   EuroSAT ViT-Tiny:  26 runs (full ablation, 3 seeds)"
echo "   EuroSAT ViT-Small:  3 runs (baseline + combined + decomp)"
echo "   CIFAR-100 ViT-Tiny:  3 runs (baseline + combined + decomp)"
echo "   RESISC45 ViT-Tiny:  3 runs (baseline + combined + decomp)"
echo "   Total: 35 runs"
echo ""
echo "######################################################################"
