# Model card: dedup vision model (DenseNet-121 + CBAM)

> **Research prototype. Not a medical device and not validated for clinical use.** Do not use it to make or support medical decisions.

This card describes the checkpoint served by the live demo and hosted on Hugging Face Hub as `best_model.pth`. Source code, split scripts, and all experiment reports are in the GitHub repository (`mkshitizreddy-ctrl/multimodal-xai-diagnostic`).

## Model

- **Architecture:** DenseNet-121 (ImageNet-initialized) with CBAM attention, dropout 0.2, and one output logit for `Pneumonia` (sigmoid probability; decision threshold 0.5).
- **Input:** a grayscale chest X-ray resized to 224 × 224 (aspect ratio is not preserved), replicated to 3 channels, normalized with mean 0.5 and std 0.25.
- **Vision only:** this checkpoint takes no tabular input. The synthetic-feature fusion models are separate and not part of this card.
- **Selected checkpoint:** epoch 9 as recorded in the checkpoint, chosen by validation macro AUROC.

## Training data and procedure

- **Data:** the public pediatric Chest X-Ray Pneumonia dataset (5,856 records). 32 exact duplicate image contents were removed, leaving 5,824 images, split into 3,727 train, 932 validation, and 1,165 test images. SHA-256 image hashes and known pneumonia patient IDs showed no overlap across splits. Patient identity for Normal images could not be verified, so complete patient-level separation is **not** claimed.
- **Optimization:** AdamW, learning rate 1e-4, weight decay 1e-5, cosine schedule, batch size 32, up to 15 epochs with early stopping (patience 4), seed 42. Training augmentation: horizontal flip (p = 0.5) and rotation up to 5 degrees.
- **Reproduce:** see "Reproducing the latest experiment" in the repository README (derived from the scripts; not yet tested end to end).

## Test results (1,165 images: 849 Pneumonia, 316 Normal)

AUROC **0.997782**. At threshold 0.5:

| Metric | Value | Wilson 95% CI |
|---|---|---|
| Accuracy | 0.9803 (1142/1165) | [0.9705, 0.9868] |
| Sensitivity (recall) | 0.9859 (837/849) | [0.9755, 0.9919] |
| Specificity | 0.9652 (305/316) | [0.9388, 0.9805] |
| Precision | 0.9870 (837/848) | [0.9769, 0.9927] |

Confusion counts: TP 837, FN 12, TN 305, FP 11. These were computed from the per-image probabilities in `docs/gradcam_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision/gradcam_testset_measurements.csv`; the AUROC recomputed from the same file matches the reported value. The test set is 73% Pneumonia, so accuracy partly reflects prevalence. Calibration was not assessed for this checkpoint.

## Explainability behavior

- Grad-CAM produced entirely zero maps for 268 of 1,165 test images (23.0%): 266 of 316 Normal and 2 of 849 Pneumonia. All were entirely negative before the standard ReLU. All 268 zero-map images have a predicted pneumonia probability of 0.134 or lower (mean 0.004). The model has a single Pneumonia output and Grad-CAM explains that output, so a blank map means no region supports a Pneumonia prediction, which is expected for images scored as clearly Normal. The 2 Pneumonia-labelled zero-map images are false negatives (probabilities 0.004 and 0.070). A zero map carries no spatial information; it does not show that the model ignores the image. The demo reports this instead of drawing a heatmap.
- Among the 897 nonzero maps, the mean fraction of positive Grad-CAM activation inside segmented lungs was 0.499 (SD 0.152). This is an overlap proxy, not a measure of explanation correctness.
- In a patch-deletion test, Grad-CAM-ranked deletion reduced the Pneumonia logit more than random deletion at 5–20% deletion, with a shrinking advantage at 30% and a reversal at 40%. It measures sensitivity to one intervention and does not establish faithfulness or clinical validity.

## Intended use and limitations

- **Intended for** research, teaching, and demonstrating explainability methods.
- **Not for** diagnosis, triage, screening, or any clinical workflow.
- Single public pediatric dataset; no external validation, so performance on other hospitals, devices, age groups, or disease prevalences is unknown.
- AUROC is near ceiling, so small differences are hard to interpret.
- Explanations are uneven across classes: most Normal images get empty Grad-CAM maps.

## Loading and integrity

The checkpoint is a PyTorch pickle (`torch.load(..., weights_only=False)`). Only load files whose SHA-256 matches the value below; do not load pickles from unknown sources.

```
SHA-256: b622f8c91b43f39b4e89b34b680eaeba9ab65c99591dde0dd91d35436027abd7
```

```python
import torch
from src.models.vision_encoder import ChestXrayVisionModel  # from the GitHub repo

ckpt = torch.load("best_model.pth", map_location="cpu", weights_only=False)
model = ChestXrayVisionModel(num_classes=len(ckpt["classes"]), pretrained=False,
                             use_cbam=ckpt.get("use_cbam", False))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()  # input: [B, 3, 224, 224]; output: [B, 1] logit for "Pneumonia"
```

The dashboard (`dashboard/app.py`) compares the loaded file's SHA-256 with this value and shows "sha256 verified" or "UNVERIFIED" in its header.

## Citation and contact

M. Kshitiz Reddy, M.Tech (AI), Bennett University. See `docs/ethics_statement.md` and `docs/references.bib`.
