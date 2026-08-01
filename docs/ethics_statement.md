# Ethics & Limitations Statement

## Not for clinical use

This is a research/portfolio prototype built for demonstrating multimodal
ML and explainability techniques. It is **not validated, certified, or
approved for clinical diagnostic use**, and predictions should never be
used as a substitute for a qualified radiologist's or physician's
judgment.

## Dataset limitations

- **NIH Chest X-ray14**'s labels were extracted from radiology reports
  using NLP (not verified by radiologists for every image), so a portion
  of labels carry noise. Published estimates put label accuracy at roughly
  90%+ for most classes, but this is not ground truth in the way a
  radiologist-adjudicated dataset would be.
- The dataset's patient population, imaging equipment, and clinical
  protocols reflect a specific set of US hospitals at a specific point in
  time. A model trained on it may not generalize to other populations,
  scanners, or imaging protocols without further validation.
- Some pathologies (e.g. Hernia) have very few positive examples, so
  per-class performance for rare findings should be interpreted with wide
  uncertainty.

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
  datasets and formal clinical validation studies)

It is **not** intended for:
- Direct patient care or diagnosis
- Deployment in any clinical setting without extensive additional
  validation, regulatory clearance, and radiologist oversight
