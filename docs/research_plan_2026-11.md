# Research plan: do the explanations and the classifier rely on lung content?

Submission deadline: 2026-11-19. Plan written 2026-11-01. Status of each step is updated as it completes.

## Question

The frozen vision model reaches AUROC 0.9978 on the patient-disjoint, deduplicated test set, but only
about half of its Grad-CAM mass falls inside the segmented lungs (mean 0.499, SD 0.152, 897 nonzero maps).
This project asks:

1. **Dependence.** Does classification performance depend on lung content, or can it be carried by
   non-lung regions and acquisition cues?
2. **Explanations.** Does the fraction of Grad-CAM mass inside the lungs track that dependence?
3. **Remedy.** Does lung-aware training reduce reliance on non-lung regions without losing in-distribution
   performance?

Scope limits: one dataset (paediatric Kermany), one deduplicated split, automatic lung masks from a
segmenter trained on adult radiographs, and no clinical claims.

## Steps

| Step | Window | Deliverable | Depends on |
|---|---|---|---|
| 1. Frozen-model audit | Nov 1-3 | `src/explain/shortcut_audit.py` results: AUROC under lung-only, background-only, displaced-mask controls, border blanking; mask QC image | none |
| 2. Mask QC and interpretation | Nov 3-5 | Decision on whether the masks are trustworthy; written reading of step 1 | step 1 |
| 3. Explanation link | Nov 5-7 | Per-image lung-energy fraction (`docs/lung_localization_testset_rho0.csv`) versus audit scores (Spearman, with CIs) | step 1 |
| 4. Remedy (conditional) | Nov 7-12 | Lung-aware training variants, 3 seeds each, same split; only if step 1 shows reliance. Otherwise robustness checks (blur, contrast) and calibration | steps 1-3 |
| 5. Write-up | Nov 12-17 | Paper/report section, figures, README and model card updates | steps 1-4 |
| 6. Freeze and check | Nov 18 | Final CI, README claims checked against results, release tag | step 5 |
| Buffer | Nov 19 | Submission day | |

## Decision rule for step 4

- If background-only AUROC stays well above the displaced-mask control, run lung-aware training.
- If lung-only AUROC is close to the original and background-only is near 0.5, report that the model uses
  lung content and spend step 4 on robustness and calibration instead.
- If the masks fail visual QC, fix or replace the segmentation before interpreting anything.

## Reporting rules

- Every result is paired with its displaced-mask control and a bootstrap CI.
- Image-level bootstrap intervals ignore clustering of images within a patient; state this.
- Negative and null results are reported as such.
