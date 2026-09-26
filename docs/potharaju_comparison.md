# Comparison against Potharaju et al. 2025

**Potharaju, S. P., et al. "Enhanced X-ray Image Classification for Pneumonia Detection Using Deep Learning Based CBAM and SE Mechanisms." *Intelligence-Based Medicine*, 2025.** (`potharaju2025enhanced` in `references.bib`)

This paper reports much higher numbers than this project on what turns out to be the same underlying dataset (the Kermany pediatric chest X-ray pneumonia set, via the `paultimothymooney/chest-xray-pneumonia` Kaggle mirror — confirmed from their cited data source). Rather than assume their architecture is simply better, this investigation ran two controlled experiments, following the brief: **use their dataset split with our method, and use our dataset split with their method**, to isolate which of the two — split or architecture — actually explains the gap.

## Why their reported numbers were worth questioning before replicating

Before running anything, their paper showed several internal red flags:
- **Table 3 is mathematically inconsistent**: CBAM+CNN reports Precision 98.4%, Recall 98.3%, but F1 94.5% — with those precision/recall values, F1 should also be ≈98.3%, not 94.5%. This is an error in their own published table.
- **No confidence intervals, no seeds, no released code.** This project has repeatedly shown (the CBAM 3→5 seed extension, the label-smoothing sweep, seed 2024's non-determinism divergence) that single, unreplicated numbers in this exact literature are unreliable.
- **Their "baseline CNN" architecture is unspecified** (no depth, no mention of ImageNet pretraining), unlike this project's DenseNet-121 backbone.
- Their reported split (5,216 train / 160 val / 480 test, from their Results section) totals 5,856 — the *entire* Kermany pool — and their paper never mentions patient-level grouping, despite pneumonia-class filenames in this dataset encoding a patient ID (`personXXX_...`), with multiple images per patient.

## Experiment A: our method, their split

`data/scripts/prepare_dataset_potharaju_split.py` reproduces their reported split sizes as closely as their paper allows: a plain, patient-blind, stratified-by-class split at the image level (matching what their paper describes — no patient grouping mentioned).

**Leakage check, run automatically by the script:** 272 pneumonia patient IDs appear in both train/val and test under this split, meaning **304 of 480 test images (63%) belong to a patient also seen during training.**

Training the project's existing DenseNet-121 + CBAM model (unchanged code, seed 42) on this split instead of our patient-level split:

| | Our patient-level split | Potharaju-style split (patient-blind) |
|---|---:|---:|
| Test set size | 624 | 480 |
| Accuracy | 86.38% | **96.88%** |
| AUROC | 0.9604 | **0.9969** |
| F1 | 0.9015 | **0.9784** |

Same model, same code, same seed — the entire jump comes from the split. This is strong evidence that a large part of the gap between this project's numbers and Potharaju et al.'s reported 98.6% accuracy is a split-leakage artifact of an undisclosed splitting methodology, not a real advantage of their architecture. This cannot be stated as certain proof of what their actual pipeline did, since they published no code — only that their paper's description is consistent with a split that leaks this badly.

## Experiment B: their method, our split

Potharaju et al. evaluate SE (Squeeze-and-Excitation, Hu et al. 2018) as a separate attention mechanism alongside CBAM. Unlike their unspecified "baseline CNN," SE is precisely defined in the literature, so it was added to this project's existing DenseNet-121 backbone as a proper, swappable alternative to CBAM (`src/models/attention.py`'s `SEBlock`, `use_se` flag in `ChestXrayVisionModel`) — reusing all of this project's existing rigor (bootstrap CIs, 5-seed evaluation) rather than adopting their weaker methodology.

**Note on faithfulness:** their paper describes SE only in general terms (no reduction ratio, no exact layer placement). `SEBlock` is a standard-formulation SE block (global average pool → two FC layers → sigmoid → channel rescale), not a byte-for-byte reconstruction of their unpublished code.

Evaluated across the same 5 seeds as the CBAM comparison (42, 123, 2024, 7, 2025), on our patient-level split:

| Model | AUROC (mean ± SD) | Localization (mean ± SD) |
|---|---:|---:|
| No attention | 0.9595 ± 0.0184 | 0.439 ± 0.026 |
| CBAM | 0.9559 ± 0.0079 | 0.468 ± 0.086 |
| **SE** | **0.9632 ± 0.0089** | **0.479 ± 0.049** |

| Comparison | AUROC diff | AUROC p | Localization diff | Localization p |
|---|---:|---:|---:|---:|
| SE vs. no attention | +0.0037 | 0.72 | +0.0397 | 0.16 |
| SE vs. CBAM | +0.0073 | 0.099 | +0.0106 | 0.84 |

SE has the best mean AUROC and best mean localization of the three attention configurations tested, and neither difference reaches statistical significance at 5 seeds — though the SE-vs-CBAM AUROC comparison (p=0.099) is closer to conventional significance than any CBAM-vs-no-attention comparison in this project ever was. Read plainly: **SE shows a mild, real-looking, but not-yet-confirmed improvement over both alternatives.** More seeds would be needed to say anything stronger.

## Overall conclusion

The professor's assignment asked for both directions of this comparison, and they point the same way: **the split, not the architecture, explains most of Potharaju et al.'s reported advantage.** Their one clearly-specified architectural idea (SE), tested properly on this project's honest split with full statistical rigor, gives a modest, unconfirmed improvement — nowhere near the scale of their reported numbers, but a legitimate, worthwhile addition to this project's attention-mechanism comparison. The internal inconsistency in their own results table, combined with the leakage risk in their described split, means their headline number should not be taken at face value without their actual code and a proper patient-level evaluation.

## Reproducing this

```bash
# Experiment A: our method, their split
python data/scripts/prepare_dataset_potharaju_split.py
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline_potharaju_split.yaml
python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_potharaju_split/best_model.pth --train-config configs/vision_baseline_potharaju_split.yaml

# Experiment B: their method (SE), our split — repeat for seeds 42, 123, 2024, 7, 2025
python src/train.py --data-config configs/data.yaml --train-config configs/vision_se.yaml --seed 42
python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_se/best_model.pth --train-config configs/vision_se.yaml
python src/explain/measure_lung_localization.py --checkpoint checkpoints/vision_se/best_model.pth --train-config configs/vision_se.yaml
```
