# EfficientEuroSAT

**Uncertainty-calibrated attention temperatures (UCAT) for Vision Transformers on satellite imagery.
Each attention head learns how sharply to look, so that the temperature itself could serve as an
uncertainty signal at no extra inference cost.**

A Vision Transformer divides every attention logit by the same constant. A blurry Sentinel-2 tile or
an ambiguous land-use patch gets attention exactly as sharp as a clean, easy one. UCAT makes the
temperature a learned quantity and adds a loss that ties it to the model's own prediction entropy:
soft attention when unsure, sharp attention when confident. If that works, the temperature is a free
by-product of the forward pass that can flag miscalibrated or out-of-distribution inputs, and it can
be compared directly with MC Dropout, deep ensembles and post-hoc temperature scaling, which all cost
extra passes or extra models.

```
EuroSAT / CIFAR-100 / RESISC45 images (224 x 224)
    ->  ViT-Tiny or ViT-Small backbone, ImageNet weights from timm
    ->  attention logits / tau,  tau = tau_a (per image, per head) + tau_e (per head)
    ->  task loss + UCAT loss (tau follows prediction entropy) + blur and consistency losses on tau_a
    ->  35 training runs, then calibration, OOD, corruption, latency and baseline comparisons
```

Sharp attention when sure, soft attention when not.

## The core idea

1. **Temperature as a signal.** Each head divides its scaled attention logits by a learned
   temperature, parameterised as `softplus(raw) + 0.1`, so it can sharpen or soften but never reach
   zero.
2. **Two parts.** `tau_e` is a learned value per head, shared by every input, annealed from 1.5x to
   1.0x over training. `tau_a` comes from a small MLP on the CLS token and is meant to vary per
   image. The decomposed model uses `tau = tau_a + tau_e`; the plain UCAT model uses `tau_e` alone.
3. **Supervision on the temperature.** The UCAT loss is the negative Pearson correlation between
   the mean temperature and the prediction entropy across a batch. A blur loss, applied every tenth
   batch, asks `tau_a` to rise across five Gaussian blur levels (none to sigma 2.0), and a
   consistency loss asks `tau_a` to agree across two augmentations of the same image.
4. **Four other modifications, ablated one at a time.** Early exit when attention is peaked, learned
   per-head attention dropout (rate capped at 0.3), a learned residual mix
   `alpha * attn + (1 - alpha) * x`, and the temperature annealing schedule. Each runs alone, all run
   together, and all run without the UCAT loss.

## What it does

No trained checkpoints, result files or figures are committed, so this section lists what the
pipeline is built to measure rather than measured numbers.

| Question | How the pipeline answers it | Script |
|---|---|---|
| Does tau track uncertainty? | temperature vs prediction entropy (Pearson r), tau on correct vs wrong predictions | `analyze_ucat.py` |
| Is the model calibrated? | ECE and MCE with reliability diagrams, UCAT vs baseline | `analyze_calibration.py` |
| Does tau flag foreign inputs? | AUROC, AUPR and FPR@95 with temperature as the score, EuroSAT vs DTD textures | `analyze_ood.py` |
| Does it survive corruption? | accuracy and mean tau under blur, color jitter, Gaussian noise, brightness and contrast shifts | `analyze_robustness.py` |
| Are tau_a and tau_e separate? | Pearson, Spearman and distance correlation, blur response curves | `analyze_decomposition.py` |
| Is it better than the usual tools? | MC Dropout, a deep ensemble of the three EuroSAT baselines, post-hoc temperature scaling, ECE over 15 bins | `compare_uncertainty_methods.py` |
| Are differences real? | paired t-tests over seeds 42, 123, 456, Bonferroni correction, Cohen's d with bootstrap CI, LaTeX tables | `statistical_significance.py` |
| What does it cost? | latency with and without early exit on GPU and CPU (50 warmup, 200 timed runs), ONNX export | `benchmark_latency.py`, `export_onnx.py` |

The 35 runs: 26 on EuroSAT with ViT-Tiny (baselines, the decomposed model, every ablation, three
seeds for most configs), plus baseline, all-modifications and decomposed runs on CIFAR-100,
RESISC45, and EuroSAT with ViT-Small, one seed each.

## How it works

**Data.** EuroSAT (10 classes, split 80/10/10 with seed 42 because it has no official split),
CIFAR-100 (100 classes) and NWPU-RESISC45 (45 classes, downloaded by `src/data/resisc45.py`). All
are resized to 224 x 224. Training loaders use inverse-frequency weighted sampling by default.

**Model.** A ViT with 16 x 16 patches. ViT-Tiny: 192-dim embeddings, 12 layers, 3 heads, about
5.7M parameters. ViT-Small: 384-dim, 12 layers, 6 heads, about 22M. Backbone weights come from timm's
ImageNet checkpoints; the head and every UCAT parameter start from scratch.

**Training.** AdamW (learning rate 5e-4, weight decay 0.05), cosine schedule with linear warmup over
the first tenth of training, label smoothing 0.1, MixUp 0.8 and CutMix 1.0 (via timm), gradient
clipping at 1.0, mixed precision on CUDA. Loss weights are 0.1 for UCAT and 0.05 each for the
aleatoric and epistemic terms. These come from the command-line defaults in `scripts/train.py`.

**Orchestration.** `scripts/autonomous_orchestrator.py` runs all 35 experiments in order, skips any
with an existing checkpoint under `--resume`, then runs every analysis script per dataset and
writes `pipeline_summary.json`. If the main decomposed model's test accuracy lands below the
baseline's, or its validation-to-test gap exceeds 0.15, the orchestrator retrains it, up to three
attempts.

**Early exit.** At inference, after the first four layers, a sample leaves the network when the mean
of its heads' maximum attention weights reaches 0.9, and the classifier reads its CLS token there.
Early exit is off during training and in the ONNX export.

## Run it

```bash
pip install -r requirements.txt
bash run.sh                                   # setup checks, all 35 runs, every analysis (EPOCHS=50 by default)
EPOCHS=200 BATCH_SIZE=32 bash run.sh          # longer schedule, smaller batches
```

Single pieces:

```bash
python scripts/run_experiments.py --run 1 2 7         # chosen experiments by number
python scripts/run_experiments.py --resume            # skip finished ones
python scripts/train.py --model efficient_eurosat --dataset eurosat --arch tiny --epochs 200
python scripts/analyze_calibration.py \
    --checkpoint_ucat ./checkpoints/decomp_with_losses/best_model_val_acc.pt \
    --checkpoint_baseline ./checkpoints/baseline/best_model_val_acc.pt --dataset eurosat
python scripts/statistical_significance.py --results_dir ./results
pytest tests/
```

Datasets download on first use. `GPU_RUN_GUIDE.md` has SLURM and Colab recipes.

## Layout

```
configs/                 reference YAML for training, ablations, deployment (not loaded by the code)
src/
  models/                ViT with modified attention, temperature predictor, baselines, timm weight loading
  training/              trainer, UCAT and decomposition losses, schedulers, callbacks
  data/                  EuroSAT, CIFAR-100, RESISC45 loaders, blur levels, augmentation pairs, class weights
  evaluation/            calibration, OOD, corruption tests, early exit stats, latency, MC Dropout, ensembles
  utils/                 checkpoints, logging, seeding, plotting
scripts/                 orchestrator, train, evaluate, one analysis script per question, figures, ONNX export
tests/                   attention modifications, data, early exit, model shapes
run.sh                   one-command pipeline
```

## Honesty

* Nothing has been measured in this repo. There are no checkpoints, result JSONs or figures, and
  the old README's run-time estimates have no logs behind them.
* `tau_a` does not depend on the image yet. The temperature MLP reads the CLS token before the
  first transformer block, where it is the same learned vector for every input (plus dropout noise
  during training). It needs a later CLS token or pooled patch features to see the image.
* As a result, every correlation loss currently sees a temperature that is constant across the
  batch. `tau_e` is shared by all inputs by design, and `tau_a` is constant because of the point
  above, so the UCAT, blur and class-rarity terms have no image-specific signal to learn from.
* The retry loop reads the test set. The orchestrator retrains the main model when its test accuracy
  falls below the baseline's, which is model selection on test data. The retry also multiplies only
  flags that were passed, and `--lr` is not, so an accuracy retry reruns the same config.
* Early exit reuses the final classifier on intermediate layers. `EarlyExitLoss` is defined but never
  used in training, and the exit trigger is attention peakedness, not class confidence.
* Seeds are thin. Three seeds exist only for 8 of the 10 EuroSAT ViT-Tiny configs; CIFAR-100,
  RESISC45 and ViT-Small get one run each, so cross-dataset differences cannot be tested for
  significance. Paired t-tests on three seeds have little power.
* `run.sh` defaults to 50 epochs for a free Colab T4, while `train.py` defaults to 200. The YAML
  files in `configs/` are not read by any script; the settings that run are the command-line
  defaults in `scripts/train.py` and the flags in the orchestrator.

MIT licensed.
