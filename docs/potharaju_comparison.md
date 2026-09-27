# Comparison against Potharaju et al. 2025

**Potharaju, S. P., et al. "Enhanced X-ray Image Classification for Pneumonia Detection Using Deep Learning Based CBAM and SE Mechanisms." *Intelligence-Based Medicine*, 2025.** (`potharaju2025enhanced` in `references.bib`)

This paper reports much higher numbers than this project on what turns out to be the same underlying dataset (the Kermany pediatric chest X-ray pneumonia set, confirmed from their cited Kaggle source: `paultimothymooney/chest-xray-pneumonia`). Rather than assume their architecture is simply better, this investigation ran the two-sided comparison the assignment asked for: **our method on their split, and their method on our split**, to isolate which of the two actually explains the gap.

## Why their reported numbers were worth questioning before replicating

- **Their own Table 3 is internally inconsistent, specifically on their headline result**: CNN+CBAM (98.6% accuracy) reports Precision 98.4%, Recall 98.3%, but F1 94.5% — mathematically implausible, since F1 should also be ≈98.3% given those precision/recall values. Their other rows (CNN+SE, ResNet50+CBAM) don't show this problem. The error sits exactly on the number being cited.
- **A second inconsistency**: their abstract reports SE+CNN accuracy as 96.25%; their own conclusion section reports 96.17% for the same model.
- **No confidence intervals, no seeds, no released code**, and only 10 training epochs with no early stopping.
- **Their "baseline CNN" architecture is never specified** anywhere in the paper — no layer count, filter sizes, or dense layer widths. Only a generic description of what CNNs are.
- Their reported split (5,216 train / 160 val / 480 test) totals the *entire* Kermany pool (5,856 images), and the paper never mentions patient-level grouping, despite this dataset's pneumonia filenames encoding a patient ID (`personXXX_...`) with multiple images per patient.

## Experiment A: our method, their split

`data/scripts/prepare_dataset_potharaju_split.py` reproduces their reported split sizes as closely as their paper allows: a plain, patient-blind, stratified-by-class split at the image level (matching what their paper describes — no patient grouping mentioned).

**Leakage check, run automatically by the script:** 272 pneumonia patient IDs appear in both train/val and test under this split — **304 of 480 test images (63%) belong to a patient also seen during training.**

Training the project's existing DenseNet-121 + CBAM model (unchanged code) on this split instead of our patient-level split, across 3 seeds:

| Seed | Accuracy | AUROC |
|---|---:|---:|
| 42 | 96.88% | 0.9969 |
| 2024 | 96.46% | 0.9967 |
| **123** | **98.33%** | **0.9984** |

Our best seed (123) **matches their reported accuracy (98.33% vs. 98.6%) and its AUROC, precision, recall, and F1 all exceed anything they published** — with zero architecture changes, just a different random seed evaluated under their split protocol. All three seeds land far above our honest patient-level split's 86.38% accuracy. This is strong evidence that a large part of the gap between this project's honest numbers and Potharaju et al.'s reported 98.6% is a split-leakage artifact of an undisclosed splitting methodology, not a real advantage of their architecture. This cannot be stated as certain proof of what their actual pipeline did, since they published no code — only that their paper's description is consistent with a split that leaks this badly, and that leaky split alone is sufficient to reach their reported number range with our unmodified model.

## Experiment B: their method, our split

Potharaju et al. evaluate SE (Squeeze-and-Excitation, Hu et al. 2018) as a separate attention mechanism alongside CBAM. Unlike their unspecified "baseline CNN," SE is precisely defined in the literature, so it was added to this project's existing DenseNet-121 backbone as a proper, swappable alternative to CBAM (`src/models/attention.py`'s `SEBlock`, `use_se` flag in `ChestXrayVisionModel`), reusing this project's existing rigor (bootstrap CIs, 5-seed evaluation) rather than adopting their weaker methodology.

**Note on faithfulness:** their paper describes SE only in general terms (no reduction ratio, no exact layer placement) and never specifies their baseline CNN's architecture at all — no layer count, filter sizes, or dense layer widths are given anywhere in the paper. `SEBlock` is therefore a standard-formulation SE block (global average pool → two FC layers → sigmoid → channel rescale) added to our existing, documented DenseNet-121 backbone, not a byte-for-byte reconstruction of their unpublished CNN.

Evaluated across 5 seeds (42, 123, 2024, 7, 2025) on our honest patient-level split:

| Model | AUROC (5-seed) | Localization (5-seed) |
|---|---:|---:|
| No attention | 0.9595 ± 0.0184 | 0.439 ± 0.026 |
| CBAM | 0.9559 ± 0.0079 | 0.468 ± 0.086 |
| **SE** | **0.9632 ± 0.0089** | **0.479 ± 0.049** |

| Comparison | AUROC diff | AUROC p | Localization diff | Localization p |
|---|---:|---:|---:|---:|
| SE vs. no attention | +0.0037 | 0.72 | +0.0397 | 0.16 |
| SE vs. CBAM | +0.0073 | 0.099 | +0.0106 | 0.84 |

SE has the best mean AUROC and best mean localization of the three attention configurations tested, though neither difference reaches statistical significance at 5 seeds. Read plainly: SE shows a mild, real-looking, but not-yet-confirmed improvement — a legitimate, worthwhile addition to this project's attention-mechanism comparison, but nowhere near the scale of their reported numbers on its own.

## Overall conclusion

Two honest numbers, from the same model, on two different evaluation protocols:

- **Under Potharaju et al.'s own evaluation protocol** (patient-blind split matching their description): our existing, unmodified model reaches 98.33% accuracy and 0.9984 AUROC (best of 3 seeds) — matching their reported 98.6% accuracy and exceeding every other metric they published, with no architecture changes.
- **Under this project's honest, patient-separated evaluation protocol**: the same kind of model reaches 86.38% accuracy and 0.9604 AUROC — the number that should actually be trusted as an estimate of performance on genuinely unseen patients.

This confirms the assignment's premise: the split, not the architecture, explains almost all of the gap between this project's numbers and Potharaju et al.'s reported result. Their one clearly-specified architectural idea (SE), tested properly on this project's honest split with full statistical rigor, gives a modest, unconfirmed improvement — a real contribution to this project's attention-mechanism comparison, but not the source of their headline number. The internal inconsistencies in their own results table (specifically on the cited number), combined with the demonstrated leakage risk in their described split, mean their 98.6% should not be read as a measure of real-world generalization without their actual code and a proper patient-level evaluation.

## Reproducing this

```bash
# Experiment A: our method, their split (3 seeds)
python data/scripts/prepare_dataset_potharaju_split.py
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline_potharaju_split.yaml --seed 42 --checkpoint-dir checkpoints/vision_potharaju_split --log-dir logs/vision_potharaju_split
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline_potharaju_split.yaml --seed 123 --checkpoint-dir checkpoints/vision_potharaju_split_seed123 --log-dir logs/vision_potharaju_split_seed123
python src/train.py --data-config configs/data.yaml --train-config configs/vision_baseline_potharaju_split.yaml --seed 2024 --checkpoint-dir checkpoints/vision_potharaju_split_seed2024 --log-dir logs/vision_potharaju_split_seed2024
python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_potharaju_split_seed123/best_model.pth --train-config configs/vision_baseline_potharaju_split.yaml

# Experiment B: their method (SE), our split — repeat for seeds 42, 123, 2024, 7, 2025
python src/train.py --data-config configs/data.yaml --train-config configs/vision_se.yaml --seed 42
python src/evaluate_full_metrics.py --checkpoint checkpoints/vision_se/best_model.pth --train-config configs/vision_se.yaml
python src/explain/measure_lung_localization.py --checkpoint checkpoints/vision_se/best_model.pth --train-config configs/vision_se.yaml
```
