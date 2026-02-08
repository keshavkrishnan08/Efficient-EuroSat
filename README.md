# EfficientEuroSAT: Uncertainty-Calibrated Attention Temperatures for Vision Transformers

Enhanced UCAT (Uncertainty-Calibrated Attention Temperatures) with aleatoric/epistemic temperature decomposition for satellite land use classification.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/keshavkrishnan08/Efficient-EuroSat.git
cd Efficient-EuroSat

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run everything autonomously
bash run.sh
```

That's it. `run.sh` handles everything from data download to final figures.

## What This Project Does

This project implements 5 learned attention modifications for Vision Transformers:

1. **Learned Temperature Scaling (UCAT)** — per-head, input-dependent temperature parameters
2. **Early Exit** — confidence-based early termination at intermediate layers
3. **Learned Dropout** — per-layer learned dropout rates via sigmoid gating
4. **Learned Residual Scaling** — adaptive residual connection weights
5. **Temperature Annealing** — cosine annealing schedule for temperature evolution

Plus an **aleatoric/epistemic temperature decomposition** that separates attention temperature into:
- **Aleatoric (input-dependent)**: responds to image-level noise/blur
- **Epistemic (class-dependent)**: responds to training data rarity

## What Happens When You Run `run.sh`

### Phase 0: Setup & Smoke Tests
- Installs dependencies (including scipy for statistical tests)
- Auto-detects GPU (CUDA or MPS)
- Verifies all models build correctly (ViT-Tiny + ViT-Small, baseline + efficient + decomposed)
- Verifies all 3 datasets are accessible (EuroSAT, CIFAR-100, RESISC45)
- Verifies loss functions compute without errors

### Phase 1-5: Training (35 Experiments via Autonomous Orchestrator)

| Phase | Experiments | Description |
|-------|------------|-------------|
| 1 | 3 | EuroSAT baselines (seeds 42, 123, 456 — needed for Deep Ensemble) |
| 2 | 1 | Main decomposed model with self-tuning retries |
| 3 | 12 | EuroSAT ablations (each mod individually + combined + no-UCAT + decomp variants) |
| 3b | 11 | 3-seed coverage for all ablations (seeds 123, 456) |
| 4 | 6 | Multi-dataset: CIFAR-100 + RESISC45 (baseline + combined + decomp each) |
| 5 | 3 | Multi-architecture: ViT-Small on EuroSAT (baseline + combined + decomp) |
| **Total** | **35** | |

### Self-Tuning
If benchmark targets aren't met, the orchestrator automatically:
- Diagnoses the issue (low accuracy, overfitting, high tau correlation, etc.)
- Adjusts hyperparameters (learning rate, dropout, lambda weights, etc.)
- Retrains (up to 3 retries per experiment)

### Evaluation & Analysis
After training, the pipeline runs:
- Per-dataset accuracy evaluation and per-class breakdown
- UCAT temperature-entropy correlation analysis
- OOD detection via temperature
- Calibration comparison (baseline vs UCAT)
- Robustness under corruptions
- Decomposition validation (tau_a vs tau_e independence)
- Blur response analysis
- Training dynamics
- Latency benchmarks
- **Uncertainty method comparison**: UCAT vs MC Dropout vs Deep Ensemble vs Post-hoc Temperature Scaling
- **Statistical significance tests**: paired t-tests across 3 seeds with LaTeX table output
- **Theoretical analysis**: gradient independence verification, temperature bounds, convergence metrics
- **12 publication-quality figures** (IEEE TNNLS format, 300 DPI)

## Datasets

| Dataset | Classes | Images | Type | Source |
|---------|---------|--------|------|--------|
| EuroSAT | 10 | 27,000 | Satellite land use | `torchvision.datasets.EuroSAT` |
| CIFAR-100 | 100 | 60,000 | Natural images | `torchvision.datasets.CIFAR100` |
| RESISC45 | 45 | 31,500 | Remote sensing scenes | Auto-downloaded from GCS |

All datasets are automatically downloaded on first run.

## Architectures

| Architecture | Params | Embed Dim | Layers | Heads |
|-------------|--------|-----------|--------|-------|
| ViT-Tiny | ~5.7M | 192 | 12 | 3 |
| ViT-Small | ~22M | 384 | 12 | 6 |

Both architectures use ImageNet-pretrained weights from `timm`.

## GPU Requirements

| Setup | VRAM Needed | Estimated Time (35 runs) |
|-------|-------------|--------------------------|
| Single GPU (A100/H100) | ~4-8 GB | ~20 hours |
| Single GPU (RTX 3090/4090) | ~4-8 GB | ~30-40 hours |
| Single GPU (T4/V100) | ~4-8 GB | ~40-60 hours |
| Apple MPS (M1/M2/M3) | ~8 GB | ~60-80 hours |

ViT-Tiny uses ~4 GB VRAM, ViT-Small uses ~8 GB. Training time depends on the number of retry cycles triggered by the self-tuning orchestrator.

## Running on Common GPU Platforms

### SLURM Cluster
```bash
#!/bin/bash
#SBATCH --job-name=eurosat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=eurosat_%j.log

module load cuda python
pip install -r requirements.txt
bash run.sh
```

### Google Colab
```python
!git clone https://github.com/keshavkrishnan08/Efficient-EuroSat.git
%cd Efficient-EuroSat
!pip install -r requirements.txt
!bash run.sh
```

### Lambda / Vast.ai / RunPod
```bash
git clone https://github.com/keshavkrishnan08/Efficient-EuroSat.git
cd Efficient-EuroSat
pip install -r requirements.txt
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

## Running Individual Experiments

```bash
# Run a specific experiment by number
python scripts/run_experiments.py --run 1 2 7

# Resume (skip completed experiments)
python scripts/run_experiments.py --resume

# Custom epochs/batch size
python scripts/run_experiments.py --epochs 50 --batch_size 32

# Train a single model directly
python scripts/train.py --model efficient_eurosat --dataset eurosat --arch tiny --epochs 200
python scripts/train.py --model baseline --dataset cifar100 --arch small --epochs 200
```

## Running Individual Analysis Scripts

All analysis scripts accept a `--dataset` parameter:

```bash
# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint ./checkpoints/decomp_with_losses/best_model_val_acc.pt --dataset eurosat

# Calibration analysis
python scripts/analyze_calibration.py \
    --checkpoint_ucat ./checkpoints/decomp_with_losses/best_model_val_acc.pt \
    --checkpoint_baseline ./checkpoints/baseline/best_model_val_acc.pt \
    --dataset eurosat

# OOD detection
python scripts/analyze_ood.py --checkpoint ./checkpoints/decomp_with_losses/best_model_val_acc.pt --dataset eurosat

# Statistical significance
python scripts/statistical_significance.py --results_dir ./results

# Uncertainty method comparison
python scripts/compare_uncertainty_methods.py \
    --ucat_checkpoint ./checkpoints/decomp_with_losses/best_model_val_acc.pt \
    --baseline_checkpoints ./checkpoints/baseline/best_model_val_acc.pt \
    --dataset eurosat

# Generate all figures
python scripts/generate_figures.py --results_dir ./analysis_results --output_dir ./figures
```

## Output Structure

After completion, results are organized as:

```
checkpoints/              # Trained model weights (35 experiments)
  baseline/
  baseline_s123/
  baseline_s456/
  decomp_with_losses/
  all_combined_s42/
  cifar100_baseline/
  resisc45_decomp/
  eurosat_baseline_small/
  ...
results/                  # Training metrics JSON per experiment
analysis_results/         # Evaluation outputs
  accuracy/               - Per-experiment test accuracy (all datasets)
  ucat/                   - Temperature-entropy correlation
  ood/                    - OOD detection
  calibration/            - ECE baseline vs UCAT
  robustness/             - Corruption robustness
  decomposition/          - tau_a vs tau_e analysis
  dynamics/               - Training dynamics
  latency/                - Inference speed benchmarks
  uncertainty_eurosat/    - Uncertainty method comparison (EuroSAT)
  uncertainty_cifar100/   - Uncertainty method comparison (CIFAR-100)
  uncertainty_resisc45/   - Uncertainty method comparison (RESISC45)
  significance/           - Statistical significance tests + LaTeX tables
  theoretical/            - Gradient independence + convergence analysis
figures/                  # Publication-ready PNG figures (300 DPI)
  fig2_tau_entropy_scatter.png
  fig3_correct_vs_wrong.png
  fig4_ood_detection.png
  fig5_calibration_comparison.png
  fig6_robustness.png
  fig7_ablation_accuracy.png
  fig8_decomposition_scatter.png
  fig9_blur_response.png
  fig10_cross_dataset_accuracy.png
  fig11_uncertainty_comparison.png
  fig12_architecture_comparison.png
pipeline_summary.json     # Full pipeline report
```

## Experiment Matrix

| Dataset | Architecture | Experiments | Seeds | Training Runs |
|---------|-------------|-------------|-------|---------------|
| EuroSAT | ViT-Tiny | Full ablation (10 configs) | 42, 123, 456 | 26 |
| EuroSAT | ViT-Small | baseline + combined + decomp | 42 | 3 |
| CIFAR-100 | ViT-Tiny | baseline + combined + decomp | 42 | 3 |
| RESISC45 | ViT-Tiny | baseline + combined + decomp | 42 | 3 |
| **Total** | | | | **35 runs** |

## Troubleshooting

- **Out of memory**: Reduce batch size: `python scripts/autonomous_orchestrator.py --batch_size 32`
- **Data download fails**: Manually download EuroSAT to `./data/eurosat/`, CIFAR-100 downloads automatically, RESISC45 to `./data/resisc45/`
- **Script hangs on data loading**: Set `num_workers=0` in the orchestrator if running in a container
- **Want to skip training and just evaluate**: Place checkpoint files in `./checkpoints/<experiment_name>/best_model_val_acc.pt` and run `python scripts/autonomous_orchestrator.py --resume`
- **Run only EuroSAT experiments**: `python scripts/run_experiments.py --run 1 2 3 4 5 6 7 8 9 10 11 12 13`
- **Run only multi-dataset experiments**: `python scripts/run_experiments.py --run 27 28 29 30 31 32`
