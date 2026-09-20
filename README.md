# 🩺 Explainable Multimodal Diagnostic Support System

> **An explainable multimodal deep-learning system for pediatric pneumonia detection from chest X-rays and clinical metadata, with Grad-CAM, CBAM attention, counterfactual explanations, lung-localization analysis, and attention-consistency training.**

[![Tests](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions/workflows/tests.yml/badge.svg)](https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic/actions)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 1. Project Overview

Medical image classification models can achieve strong predictive performance while providing limited insight into **why** a prediction was made.

This project develops an **explainable multimodal diagnostic support system** for pediatric pneumonia detection using:

* Chest X-ray images
* Clinical/tabular features
* DenseNet-121 image representation
* CBAM channel + spatial attention
* Grad-CAM visual explanations
* Occlusion-based counterfactual explanations
* Lung-field localization analysis
* Attention-consistency training
* Multimodal image + tabular fusion
* Statistical evaluation across multiple random seeds

The project therefore evaluates not only **whether the model predicts pneumonia**, but also investigates **where the model is focusing and how attention changes under different training strategies**.

> ⚠️ **Important:** The tabular clinical features used for multimodal fusion are **synthetically generated** because the source chest X-ray dataset does not provide real EHR/vital-sign metadata. They are used to demonstrate and evaluate the multimodal architecture and must not be interpreted as real clinical findings.

---

# 2. Research Questions

The project investigates several related questions:

1. Can a DenseNet-121 model accurately classify pneumonia from chest X-rays?
2. Does adding tabular information improve the image-only model?
3. Can Grad-CAM reveal whether the model focuses on clinically relevant image regions?
4. Does CBAM improve the localization of model attention?
5. Does the effect of CBAM remain consistent when evaluated across multiple random seeds?
6. Does CBAM behave similarly in vision-only and multimodal models?
7. Can attention-consistency training explicitly encourage attention to remain within lung fields?
8. What trade-off exists between predictive performance and localization quality?
9. How does the attention-consistency loss weight affect this trade-off?

---

# 3. Main Contributions

### Vision model

* DenseNet-121 chest X-ray classifier.
* Patient-level dataset splitting.
* Binary pneumonia classification.
* Full evaluation using Accuracy, Precision, Recall, F1, AUROC and AUPR.

### Multimodal fusion

* Combines image representation with tabular features.
* Provides an experimental comparison between image-only and multimodal learning.
* Supports explainability through a dedicated fusion wrapper.

### Explainability

* Grad-CAM visualization.
* Occlusion-based counterfactual explanations.
* Lung-field localization measurement.
* Attention-map analysis.

### Attention modelling

* CBAM channel + spatial attention.
* Multi-seed CBAM evaluation.
* Fusion-model CBAM evaluation.
* Attention-consistency loss using lung segmentation masks.
* Weight-sensitivity analysis.

### Engineering and reproducibility

* Configurable training pipeline.
* Automated evaluation scripts.
* Cached lung masks for attention-consistency training.
* Automated test suite.
* 84 tests passing in the development environment.

---

# 4. System Architecture

```text
                    ┌──────────────────────┐
                    │   Chest X-ray Image  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     DenseNet-121     │
                    │    Vision Encoder    │
                    └──────────┬───────────┘
                               │
                         CBAM Attention
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Representation │
                    └──────────┬───────────┘
                               │
                               │
                               │       ┌──────────────────────┐
                               │       │ Synthetic Clinical   │
                               │       │     Features         │
                               │       └──────────┬───────────┘
                               │                  │
                               │                  ▼
                               │       ┌──────────────────────┐
                               │       │   Tabular Encoder    │
                               │       └──────────┬───────────┘
                               │                  │
                               └─────────┬────────┘
                                         ▼
                              ┌──────────────────────┐
                              │    Fusion Network    │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ Pneumonia Prediction │
                              └──────────────────────┘

       ┌─────────────────────────────────────────────────────┐
       │                  Explainability                     │
       ├─────────────────────────────────────────────────────┤
       │ Grad-CAM                                             │
       │ CBAM Spatial Attention                               │
       │ Lung Localization                                    │
       │ Occlusion Counterfactuals                            │
       │ Attention-Consistency Analysis                       │
       └─────────────────────────────────────────────────────┘
```

Architecture documentation is available in:

`docs/architecture.md`

---

# 5. Dataset

The project uses the **Chest X-ray Pneumonia** dataset containing:

* **5,856 pediatric chest X-ray images**
* Age range: approximately 1–5 years
* Classes:

  * Normal
  * Pneumonia
* Source: Guangzhou Women and Children's Medical Center

The project originally considered the NIH Chest X-ray14 dataset but moved to the pediatric pneumonia dataset because of local storage and computational constraints.

### Important data limitation

The original dataset contains chest X-ray images but does not provide the real patient vitals required for multimodal fusion.

Therefore, the following tabular features are **synthetically generated**:

* Age
* Gender
* Temperature
* SpO₂

The synthetic features contain clinically plausible correlations with the pneumonia label and are used specifically to demonstrate multimodal fusion.

This means the fusion result should be interpreted as an **architecture-level experiment**, not as evidence that real clinical metadata improves pneumonia diagnosis.

See:

`docs/ethics_statement.md`

---

# 6. Data Splitting

The project uses patient-level splitting where possible.

The resulting experimental dataset contains approximately:

* **4,434 training images**
* **798 validation images**
* **624 test images**

Pneumonia filenames provide patient identifiers that are used for grouping.

Normal-class images do not provide an equivalent identifier and are conservatively treated as individually unique patients.

The original test split supplied by the dataset is preserved.

---

# 7. Technology Stack

* Python 3.11+
* PyTorch
* Torchvision
* DenseNet-121
* CBAM
* scikit-learn
* Pandas
* NumPy
* Grad-CAM
* Streamlit
* OpenCV/PIL
* pytest

---

# 8. Repository Structure

```text
multimodal-xai-diagnostic/
│
├── data/
│   └── scripts/
│       ├── prepare_pneumonia_dataset.py
│       └── precompute_lung_masks.py
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── lung_mask_dataset.py
│   │
│   ├── models/
│   │   ├── vision_encoder.py
│   │   ├── fusion.py
│   │   ├── attention.py
│   │   └── attention_consistency_loss.py
│   │
│   ├── explain/
│   │   ├── gradcam.py
│   │   ├── counterfactual.py
│   │   ├── lung_segmentation.py
│   │   └── measure_lung_localization.py
│   │
│   ├── train.py
│   ├── train_fusion.py
│   ├── train_attention_consistency.py
│   ├── evaluate.py
│   ├── evaluate_fusion.py
│   └── evaluate_full_metrics.py
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   ├── 01_vision_baseline_results.ipynb
│   └── 02_fusion_ablation_results.ipynb
│
├── configs/
│
├── tests/
│
├── docs/
│   ├── architecture.md
│   ├── ethics_statement.md
│   ├── augmentation_ablation.md
│   ├── deployment.md
│   ├── paper_notes.md
│   ├── references.bib
│   └── research_paper_*.tex
│
├── CHANGELOG.md
└── README.md
```

---

# 9. Baseline Vision Model

The primary vision model uses **DenseNet-121** for chest X-ray classification.

Training uses:

* Binary classification
* BCEWithLogitsLoss
* AdamW optimizer
* Cosine learning-rate schedule
* Early stopping
* Patient-level train/validation splitting

The model can operate with CBAM enabled or disabled through configuration.

---

# 10. Multimodal Fusion

The multimodal model combines:

```text
Chest X-ray
     │
     ▼
DenseNet-121
     │
     ▼
Image Features
     │
     ├──────────────┐
                    │
Clinical Features ──► Tabular Encoder
                    │
                    ▼
             Fusion Representation
                    │
                    ▼
             Pneumonia Prediction
```

The fusion model allows the project to investigate whether additional tabular information can complement image-based representations.

Because the tabular variables are synthetic, this experiment demonstrates **multimodal architecture behaviour**, not clinical benefit.

---

# 11. Explainability

## Grad-CAM

Grad-CAM is used to identify image regions contributing to the model's prediction.

Example generation:

```bash
python src/explain/generate_examples.py \
    --checkpoint checkpoints/vision_baseline/best_model.pth
```

The project found that some predictions produced anatomically plausible heatmaps while others showed activation outside the lung region.

This observation motivated the localization experiments.

---

# 12. Shortcut-Learning Investigation

Manual inspection of Grad-CAM outputs identified cases where activation extended beyond the lungs.

Some examples showed activation around:

* shoulders
* image borders
* annotations
* burned-in markers
* timestamps

This is important because a high classification score does not automatically imply that the model learned medically meaningful visual features.

The project therefore treats explainability as an experimental object rather than assuming that a heatmap is automatically trustworthy.

---

# 13. Lung-Restricted Explanations

A pretrained chest X-ray segmentation model is used to identify lung fields.

Grad-CAM explanations can then be restricted to the lung region:

```bash
python src/explain/generate_examples.py \
    --checkpoint checkpoints/vision_baseline/best_model.pth \
    --restrict-to-lungs \
    --output-dir docs/gradcam_examples_lung_restricted
```

This constrains the **visual explanation** to anatomically relevant regions.

It does not prove that the underlying classifier itself learned only from those regions.

That distinction is intentionally preserved in the project.

---

# 14. Counterfactual Explanations

The project also implements occlusion-based counterfactual explanations.

The highest-activation region identified by Grad-CAM is masked/inpainted and the model is evaluated again.

Conceptually:

```text
Original X-ray
      │
      ▼
Grad-CAM
      │
      ▼
Important region
      │
      ▼
Mask / Occlude region
      │
      ▼
Run model again
      │
      ▼
Compare confidence
```

A large confidence reduction indicates that the highlighted region was important to the prediction.

In an evaluation of six borderline predictions:

* **1/6 predictions flipped completely**
* **4 of the remaining 5 showed substantial confidence reductions**

This is supportive evidence that the highlighted regions can influence predictions, although counterfactual explanations are not equivalent to causal explanations.

---

# 15. CBAM Attention

The project adds **Convolutional Block Attention Module (CBAM)** to DenseNet-121.

CBAM contains:

1. Channel attention
2. Spatial attention

The spatial component is particularly relevant to this project because localization quality is one of the research questions.

CBAM is configurable using:

```yaml
use_cbam: true
```

The implementation is shared by the vision and fusion architectures.

---

# 16. Full Evaluation Results

The current full-metrics evaluation reports:

| Model / Experiment     |   Accuracy | Precision | Recall |         F1 |      AUROC |       AUPR |
| ---------------------- | ---------: | --------: | -----: | ---------: | ---------: | ---------: |
| Vision baseline        | **86.38%** |    0.8224 | 0.9974 | **0.9015** | **0.9604** | **0.9628** |
| Vision + rotation/zoom | **73.88%** |    0.7052 | 1.0000 |     0.8271 | **0.9651** | **0.9730** |
| Multimodal fusion      | **87.02%** |    0.8280 | 1.0000 | **0.9059** | **0.9899** | **0.9921** |

These values are taken from:

* `docs/vision_full_metrics.csv`
* `docs/vision_rotation_zoom_metrics.csv`
* `docs/fusion_full_metrics.csv`

The evaluation also stores bootstrap confidence intervals for the reported metrics.

---

# 17. Vision Baseline

The evaluated vision baseline achieved:

* Accuracy: **0.8638**
* Precision: **0.8224**
* Recall: **0.9974**
* F1: **0.9015**
* AUROC: **0.9604**
* AUPR: **0.9628**

95% bootstrap confidence intervals:

| Metric    | 95% CI          |
| --------- | --------------- |
| Accuracy  | 0.8365 – 0.8895 |
| Precision | 0.7876 – 0.8566 |
| Recall    | 0.9920 – 1.0000 |
| F1        | 0.8800 – 0.9217 |
| AUROC     | 0.9412 – 0.9757 |
| AUPR      | 0.9377 – 0.9825 |

---

# 18. Multimodal Fusion Results

The fusion model achieved:

* Accuracy: **0.8702**
* Precision: **0.8280**
* Recall: **1.0000**
* F1: **0.9059**
* AUROC: **0.9899**
* AUPR: **0.9921**

95% bootstrap confidence intervals:

| Metric    | 95% CI          |
| --------- | --------------- |
| Accuracy  | 0.8429 – 0.8959 |
| Precision | 0.7927 – 0.8627 |
| Recall    | 1.0000 – 1.0000 |
| F1        | 0.8843 – 0.9263 |
| AUROC     | 0.9810 – 0.9961 |
| AUPR      | 0.9839 – 0.9978 |

The fusion experiment demonstrates that the multimodal architecture can exploit the available tabular signal.

However, because the tabular features are synthetic, this result should **not** be interpreted as evidence that real patient vitals improve clinical diagnosis.

---

# 19. Augmentation Ablation

A training-only augmentation experiment added:

* Horizontal flipping
* Random rotation
* Random magnification / zoom

Validation and test images remained unaugmented.

The comparison was:

| Model                         |   Accuracy |         F1 |      AUROC |       AUPR |
| ----------------------------- | ---------: | ---------: | ---------: | ---------: |
| Existing CBAM vision baseline |     0.8638 |     0.9015 |     0.9604 |     0.9628 |
| Rotation + zoom               | **0.7388** | **0.8271** | **0.9651** | **0.9730** |

The augmentation run produced **higher AUROC/AUPR but substantially lower accuracy and F1**.

Therefore, it is not described as a simple performance improvement.

This experiment demonstrates why multiple evaluation metrics are important when analysing model changes.

Full methodology:

`docs/augmentation_ablation.md`

---

# 20. CBAM Multi-Seed Experiment

CBAM was evaluated across three random seeds:

* 42
* 123
* 2024

### Test AUROC

| Seed          |             No CBAM |                CBAM |           Difference |
| ------------- | ------------------: | ------------------: | -------------------: |
| 42            |              0.9592 |              0.9608 |              +0.0016 |
| 123           |              0.9695 |              0.9445 |              −0.0250 |
| 2024          |              0.9736 |              0.9604 |              −0.0132 |
| **Mean ± SD** | **0.9674 ± 0.0074** | **0.9552 ± 0.0093** | **−0.0122 ± 0.0139** |

Exploratory one-sample test on the three seed-level differences:

**p = 0.25**

### Lung localization

Mean Grad-CAM lung-energy fraction difference:

**+0.060 ± 0.076**

with:

**p = 0.31**

The three seeds did not produce a statistically significant localization effect.

Two seeds showed improvement, while one seed showed a decline.

---

# 21. Statistical Correction

An earlier single-seed experiment produced a much smaller p-value.

That analysis was subsequently identified as **pseudo-replication** because individual images from the same trained model were treated as independent experimental replicates.

The correct replication unit for the training experiment is the **training run / random seed**, not each image.

The project therefore repeated the CBAM comparison using three independent seeds.

This correction is intentionally documented rather than hiding the earlier result.

> **Research lesson:** statistical significance must be evaluated at the correct experimental unit.

---

# 22. CBAM on the Fusion Model

CBAM was also evaluated on the multimodal fusion architecture.

### Lung localization

| Seed          | No CBAM |  CBAM |          Difference |
| ------------- | ------: | ----: | ------------------: |
| 42            |   0.451 | 0.516 |              +0.065 |
| 123           |   0.489 | 0.393 |              −0.096 |
| 2024          |   0.444 | 0.476 |              +0.032 |
| **Mean ± SD** |       — |     — | **+0.0003 ± 0.085** |

Exploratory test:

**p = 0.995**

### Test AUROC

| Seed          |             No CBAM |                CBAM |           Difference |
| ------------- | ------------------: | ------------------: | -------------------: |
| 42            |              0.9935 |              0.9921 |              −0.0014 |
| 123           |              0.9795 |              0.9915 |              +0.0120 |
| 2024          |              0.9816 |              0.9899 |              +0.0083 |
| **Mean ± SD** | **0.9849 ± 0.0076** | **0.9912 ± 0.0011** | **+0.0063 ± 0.0071** |

Exploratory test:

**p = 0.26**

The fusion experiment therefore does not reproduce the same localization trend observed in the vision-only model.

This suggests that the effect of CBAM is dependent on the surrounding architecture and should not automatically be generalized from one model type to another.

---

# 23. Attention-Consistency Training

The project extends CBAM with an **attention-consistency loss**.

Instead of only measuring whether attention falls inside the lungs after training, the model is explicitly trained to encourage its attention toward the segmented lung field.

The loss is defined using:

```text
1 − lung_energy_fraction
```

Lung masks are precomputed at the spatial resolution of the CBAM attention map.

This avoids running the segmentation model during every training batch.

---

# 24. Attention-Consistency Results

The experiment was repeated across three seeds.

### Lung localization

| Seed          |  CBAM | Attention Consistency |         Difference |
| ------------- | ----: | --------------------: | -----------------: |
| 42            | 0.515 |                 0.641 |             +0.126 |
| 123           | 0.424 |                 0.616 |             +0.192 |
| 2024          | 0.473 |                 0.557 |             +0.084 |
| **Mean ± SD** |     — |                     — | **+0.134 ± 0.055** |

Exploratory test:

**p = 0.051**

All three seeds showed improved localization.

### Test AUROC

| Seed          |                CBAM | Attention Consistency |         Difference |
| ------------- | ------------------: | --------------------: | -----------------: |
| 42            |              0.9608 |                0.9167 |            −0.0441 |
| 123           |              0.9445 |                0.9414 |            −0.0031 |
| 2024          |              0.9604 |                0.9296 |            −0.0308 |
| **Mean ± SD** | **0.9552 ± 0.0093** |   **0.9292 ± 0.0124** | **−0.026 ± 0.021** |

Exploratory test:

**p = 0.164**

The experiment therefore shows a consistent localization improvement accompanied by a reduction in predictive AUROC.

Because only three seeds were used, these statistical results should be treated as exploratory rather than definitive.

---

# 25. Attention-Consistency Weight Sweep

A three-seed sweep was performed over different attention-consistency weights.

|          Weight |      Test AUROC |  Localization |
| --------------: | --------------: | ------------: |
| 0.0 — CBAM only | 0.9552 ± 0.0093 | 0.462 ± 0.062 |
|            0.05 | 0.9356 ± 0.0119 | 0.550 ± 0.022 |
|            0.10 | 0.9292 ± 0.0124 | 0.593 ± 0.048 |
|            0.20 | 0.9100 ± 0.0380 | 0.611 ± 0.028 |

The sweep shows a clear trade-off:

```text
Higher attention-consistency weight
             │
             ├──► Better lung localization
             │
             └──► Lower predictive AUROC
```

The localization improvement shows diminishing returns as the weight increases, while the higher weights introduce greater variation in AUROC.

This experiment demonstrates that explainability constraints can affect predictive performance and therefore need to be treated as a tunable design decision rather than a free improvement.

---

# 26. Failed / Corrected Experiment

An earlier single-run experiment at attention-consistency weight `0.03` produced:

* AUROC: **0.8933**
* Suspicious validation score: **1.0000**

After the subsequent multi-seed weight sweep, this result did not fit the observed trend.

It was therefore treated as an **unreplicated outlier** rather than presented as evidence supporting the method.

This is another example of why the project uses repeated experiments rather than selecting a single favourable run.

---

# 27. What the Experiments Show

The experiments provide several distinct findings:

### Predictive performance

The multimodal fusion model produced the highest AUROC among the main evaluated configurations:

**AUROC = 0.9899**

### Explainability

Grad-CAM revealed that high predictive performance does not guarantee anatomically appropriate localization.

### CBAM

CBAM produced inconsistent effects across random seeds and across vision vs. fusion architectures.

### Attention consistency

Explicitly training attention toward lung regions produced more consistent localization improvements across the three evaluated seeds.

### Trade-off

Increasing the attention-consistency loss improves localization but reduces predictive AUROC.

### Statistical methodology

The project identified and corrected an earlier pseudo-replication problem, changing the interpretation of the initial CBAM result.

---

# 28. Reproducibility

## Environment

The project was tested using:

```text
Python 3.11.15
pytest 9.1.1
```

### Test result

```text
84 passed, 6 warnings
```

The warnings originate from dependency code and do not represent test failures.

---

## Installation

```bash
git clone https://github.com/mkshitizreddy-ctrl/multimodal-xai-diagnostic.git
cd multimodal-xai-diagnostic

conda create -n ai_env python=3.11
conda activate ai_env

pip install -r requirements.txt
```

---

# 29. Dataset Preparation

The dataset preparation script handles downloading and preprocessing:

```bash
python data/scripts/prepare_pneumonia_dataset.py
```

The script prepares the dataset, creates the experimental splits, and generates the synthetic tabular features required for the fusion experiment.

---

# 30. Training

### Vision baseline

```bash
python src/train.py \
    --data-config configs/data.yaml \
    --train-config configs/vision_baseline.yaml
```

### Fusion model

```bash
python src/train_fusion.py \
    --data-config configs/data.yaml \
    --train-config configs/fusion.yaml
```

### Attention consistency

First precompute lung masks:

```bash
python data/scripts/precompute_lung_masks.py \
    --train-config configs/vision_attention_consistency.yaml
```

Then train:

```bash
python src/train_attention_consistency.py \
    --train-config configs/vision_attention_consistency.yaml
```

---

# 31. Evaluation

Standard evaluation:

```bash
python src/evaluate.py \
    --checkpoint checkpoints/vision_baseline/best_model.pth
```

Fusion evaluation:

```bash
python src/evaluate_fusion.py \
    --checkpoint checkpoints/fusion/best_model.pth
```

Full metrics with bootstrap confidence intervals:

```bash
python src/evaluate_full_metrics.py \
    --checkpoint checkpoints/vision_baseline/best_model.pth \
    --output-csv docs/vision_full_metrics.csv
```

---

# 32. Lung Localization

Localization can be measured using:

```bash
python src/explain/measure_lung_localization.py \
    --checkpoint checkpoints/vision_baseline/best_model.pth \
    --output-csv docs/localization_cbam.csv
```

The metric used is the fraction of Grad-CAM heatmap energy located within the segmented lung field.

---

# 33. Dashboard

The project includes a Streamlit dashboard for interactive demonstration.

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard supports:

* X-ray input
* Pneumonia prediction
* Prediction probability
* Grad-CAM visualization
* Counterfactual visualization
* Model/checkpoint information

A deployed demonstration is also documented in:

`docs/deployment.md`

---

# 34. Testing

The repository contains unit and integration tests covering:

* Dataset handling
* Vision model
* Fusion model
* CBAM attention
* Attention-consistency loss
* Lung-mask dataset
* Fusion explainability
* Localization measurement
* Evaluation pipeline
* Dashboard utilities

Current local test result:

```text
84 passed, 6 warnings
```

The project also includes GitHub Actions for automated testing.

---

# 35. Research Documentation

The repository contains supporting research documentation:

```text
docs/
├── architecture.md
├── ethics_statement.md
├── augmentation_ablation.md
├── paper_notes.md
├── deployment.md
├── references.bib
├── research_paper_main.tex
└── research_paper_revisions.tex
```

`paper_notes.md` documents the literature review that motivated the CBAM and localization work.

The research-paper files contain the formal academic version of the methodology, experiments, limitations, and results.

---

# 36. Limitations

This project has several important limitations.

### Synthetic clinical metadata

The tabular fusion features are synthetic and therefore cannot establish real-world clinical benefit.

### Dataset size

The dataset is relatively modest compared with large-scale medical imaging datasets.

### Limited replication

Several statistical comparisons use only three random seeds.

Therefore, reported p-values should be considered exploratory.

### Localization metric

Lung-energy fraction measures whether attention lies inside a segmented lung region.

It does not prove that the model is focusing on the correct pathological lesion.

### Grad-CAM limitations

Grad-CAM is an interpretation method rather than a causal explanation.

### Counterfactual limitations

Occlusion-based counterfactuals show sensitivity to image-region removal, but they do not establish clinical causality.

### Clinical deployment

The system is a research/portfolio prototype and has **not been clinically validated**.

It must not be used as a medical diagnostic device.

---

# 37. Research Integrity

A central design principle of this project is to report both positive and negative findings.

Examples include:

* CBAM did not consistently improve predictive performance.
* CBAM localization effects varied across seeds.
* CBAM behaved differently in vision and fusion models.
* Attention consistency improved localization while reducing AUROC.
* An earlier pseudo-replicated statistical result was corrected.
* An unreplicated attention-consistency result was subsequently treated as an outlier.

The purpose is to evaluate the methods rather than select only favourable results.

---

# 38. Project Status

### Completed

* [x] Dataset preparation pipeline
* [x] Patient-level splitting
* [x] DenseNet-121 vision model
* [x] Multimodal image + tabular fusion
* [x] Grad-CAM
* [x] Counterfactual explanations
* [x] Lung segmentation
* [x] Lung-localization metric
* [x] CBAM attention
* [x] Multi-seed CBAM evaluation
* [x] Fusion CBAM evaluation
* [x] Attention-consistency loss
* [x] Three-seed attention-consistency experiment
* [x] Attention-consistency weight sweep
* [x] Augmentation ablation
* [x] Full evaluation metrics
* [x] Bootstrap confidence intervals
* [x] Streamlit dashboard
* [x] Automated test suite
* [x] Research paper documentation
* [x] Statistical correction of earlier pseudo-replication

---

# 39. Future Work

Potential extensions include:

* Larger multi-center datasets
* Real clinical/EHR metadata
* More independent training seeds
* External validation
* Pathology-level localization annotations
* Quantitative evaluation of counterfactual explanations
* Calibration analysis
* Additional multimodal fusion strategies
* Comparison with alternative attention mechanisms
* Prospective clinical validation

---

# 40. Conclusion

This project developed an explainable multimodal pneumonia-classification pipeline that goes beyond measuring classification accuracy.

The system combines:

**Chest X-ray → DenseNet-121 → CBAM → Multimodal Fusion → Prediction**

with:

**Grad-CAM → Counterfactuals → Lung Localization → Attention Consistency**

The experiments show that the multimodal model can achieve strong predictive performance, while the explainability experiments demonstrate that predictive performance and localization quality are separate objectives.

Most importantly, the project explicitly evaluates the trade-off between **prediction quality and explanation localization**, including negative results and statistical corrections rather than reporting only the most favourable experiment.

---

## License

MIT License.

See [`LICENSE`](LICENSE).

---

## Citation

If this repository is used in academic work, please cite the project and the associated research documentation in `docs/references.bib`.

---

**Author:** M. Kshitiz Reddy
**Program:** M.Tech Artificial Intelligence
**Institution:** Bennett University
