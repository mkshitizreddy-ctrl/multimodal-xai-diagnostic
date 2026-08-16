# Ethics & Limitations Statement

## ⚠️ Synthetic clinical data — read this before citing any results

The current pipeline uses the [Kaggle Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia),
which ships **images only** — no EHR/vitals data. To keep this project
genuinely multimodal (rather than reverting to a vision-only classifier),
`data/scripts/prepare_pneumonia_dataset.py` **generates synthetic** age,
gender, temperature, and SpO2 values for every image.

These are **not real patient measurements.** They are simulated with
clinically plausible distributions correlated with the Pneumonia label
(fever and lower oxygen saturation more likely when Pneumonia is present)
specifically so the fusion ablation study demonstrates something real about
the *architecture's* ability to exploit correlated tabular signal — not to
fabricate clinical findings. Any AUROC improvement from fusion in this
version of the project reflects the model successfully learning the
synthetic correlation that was deliberately built in, **not** a genuine
clinical discovery. This must be stated explicitly in any presentation,
report, or interview discussion of this project's results.

(An earlier version of this project used the NIH Chest X-ray14 dataset,
which does include some real patient metadata — age, gender, view position
— but not vitals like temperature/SpO2. See `configs/data_nih_legacy.yaml`.)

## Not for clinical use

This is a research/portfolio prototype built for demonstrating multimodal
ML and explainability techniques. It is **not validated, certified, or
approved for clinical diagnostic use**, and predictions should never be
used as a substitute for a qualified radiologist's or physician's
judgment.

## Dataset limitations

- **Source population**: the Chest X-ray Pneumonia dataset was collected
  from pediatric patients (ages 1–5) at Guangzhou Women and Children's
  Medical Center. A model trained on it should not be assumed to
  generalize to adult patients, other hospitals, or other imaging
  equipment.
- **Patient grouping for splits**: pneumonia-positive filenames encode a
  person identifier that this pipeline groups by to avoid leaking a
  patient's images across train/val. Normal-class filenames do not expose
  an equivalent identifier, so each normal image is conservatively treated
  as its own patient for splitting purposes — a known limitation of this
  specific dataset's file naming, not a data leakage bug.
- **Small dataset**: ~5,800 total images is small by deep learning
  standards, which increases variance in reported metrics compared to a
  dataset the size of full NIH Chest X-ray14.

## Explainability caveats

- **Grad-CAM** highlights regions that influenced the model's output, but
  a highlighted region is not proof of correct clinical reasoning — a
  model can be "right for the wrong reasons" (e.g. keying off an
  annotation artifact rather than the actual pathology).
- **The occlusion-based counterfactual module** is an intentionally
  simplified stand-in for generative counterfactual methods (e.g.
  conditional VAEs or diffusion-based counterfactuals). It shows how the
  model's confidence changes when the highlighted region is removed, which
  is a useful sanity check, but it is not a claim about what the anatomy
  would actually look like without the pathology.

### Observed evidence of possible shortcut learning

Manual inspection of Grad-CAM outputs across several confidently-predicted
test images (`docs/gradcam_examples/`) found inconsistent localization
quality:

- Some examples (`example_535`, `example_272`) show a well-concentrated
  activation peak over central chest tissue — the expected pattern for a
  model attending to genuine pathology.
- Others (`example_55`, `example_269`, `example_406`) show diffuse
  activation spread across nearly the entire image, including areas well
  outside the lung fields — weak evidence the model is attending to
  anything specific. `example_406` in particular renders as almost fully
  color-saturated with little of the underlying X-ray visible; this was
  checked and confirmed to reflect a genuinely broad, near-maximal
  heatmap rather than a rendering bug — `show_cam_on_image` uses a fixed
  50/50 blend for every image, and other examples render with the
  underlying X-ray clearly visible, so the saturation is a property of
  this specific image's activation pattern, not the overlay code.
- One example (`example_269`) shows visible burned-in image annotations
  (an "R" laterality marker and an acquisition timestamp) with warm-colored
  activation near that region — a concrete, observed instance of the
  general shortcut-learning risk described above, not just a theoretical
  concern. This is a known risk with this specific dataset: Normal and
  Pneumonia images were not necessarily captured with identical equipment,
  cropping, or annotation conventions, so a model can learn to exploit
  those incidental differences rather than the underlying pathology while
  still achieving a high AUROC.

This is disclosed here specifically *because* it was found using this
project's own explainability tooling — it's presented as evidence the
tools are doing their job (surfacing exactly this kind of risk for
inspection), not as something to hide. A systematic audit (e.g. checking
localization quality across the full test set, or applying lung
segmentation as a preprocessing step to constrain where activation can
occur) would be a natural next step before trusting this model's
attention patterns further.

**Mitigation implemented and verified:** `src/explain/lung_segmentation.py` uses a
pretrained chest X-ray segmentation model
([torchxrayvision](https://github.com/mlmed/torchxrayvision)) to constrain
Grad-CAM activation to the segmented lung fields, available via
`restrict_to_lungs=True` on `ChestXrayExplainer.explain()` and
`OcclusionCounterfactualExplainer.generate()`, or `--restrict-to-lungs` on
the example-generation scripts. Re-running the two worst offenders above
with this enabled confirms it works as intended: `example_406`'s
near-total color saturation is now visibly clipped to the lung silhouette,
and `example_269`'s activation no longer sits on the burned-in "R"
marker/timestamp — see the before/after comparison in the main
[`README.md`](../README.md#a-real-finding-manual-audit-caught-shortcut-learning).
This constrains the *explanation* to anatomically valid regions — it does
not fix the underlying shortcut-learning risk in the trained model itself
(the model may still be using non-anatomical cues internally), but it does
prevent the explanation from misleadingly appearing valid when it isn't.
Not enabled by default, to keep the unconstrained output available for
exactly this kind of audit.

## Bias considerations

Public chest X-ray datasets have documented demographic imbalances (e.g.
skew in age, sex, and imaging equipment across sites). No fairness/bias
audit across demographic subgroups has been performed on this model as
part of this project; this would be a required step before any real-world
consideration.

## Intended use

This repository is intended for:
- Educational and portfolio purposes
- Demonstrating a multimodal fusion + explainability pipeline
- A starting point for further research (e.g. with radiologist-verified
  datasets, real EHR data, and formal clinical validation studies)

It is **not** intended for:
- Direct patient care or diagnosis
- Deployment in any clinical setting without extensive additional
  validation, regulatory clearance, radiologist oversight, and replacement
  of all synthetic data with real, ethically-sourced clinical data
