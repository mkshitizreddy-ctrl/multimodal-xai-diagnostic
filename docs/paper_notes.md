# Paper notes — CBAM addition

Prof asked (lab session, 21 Aug) to read at least 5 papers related to the
project and actually put an idea from them into the code before next
Friday's lab. Notes below are what I read this week, roughly in the order
I found them.

Started from a general search on attention mechanisms + pneumonia CXR
since that's the most direct way to improve the Grad-CAM explainability
side of the project (see `docs/ethics_statement.md` — I already found one
Grad-CAM localization problem manually, curious if attention helps here
for real).

---

### 1. Dey, "CBAM-Enhanced DenseNet121 for Multi-Class Chest X-Ray Classification with Grad-CAM Explainability" (2026)
https://arxiv.org/abs/2604.12305

Almost exactly our setup — DenseNet121 backbone + CBAM + Grad-CAM, except
theirs is a 3-way split (normal / bacterial pneumonia / viral pneumonia)
instead of our binary normal/pneumonia. Reports per-class AUC in the
0.92–0.96 range with CBAM added, and — more relevant to us — the paper
notes the Grad-CAM maps land on plausible lung regions per class rather
than scattering. Also has a useful negative result buried in there:
EfficientNetB3 underperformed even their own plain CNN baseline, which is
a good reminder that a fancier backbone isn't automatically a better one.
This is the closest paper to our exact stack, so it's the main one I
followed for how to wire CBAM in.

### 2. "Enhanced X-ray image classification for pneumonia detection using deep learning based CBAM and SE mechanisms" — ScienceDirect (2025)
https://www.sciencedirect.com/science/article/pii/S2666521225001036

Broader survey-style paper, compares CBAM against SE (Squeeze-and-Excite)
across a few CXR tasks. Useful for deciding *which* attention module to
use — SE only does channel attention, CBAM adds the spatial term on top.
For our case the spatial part is the whole point (we want the heatmap to
stop activating outside the lungs), so this is basically the argument for
picking CBAM over SE. Also cites a separate result combining ResNet152 +
DenseNet121 + ResNet18 features with attention for pneumonia, reporting
accuracy above 96% with high recall — didn't fully read that sub-paper,
just noted the reference.

### 3. "An Enhanced Deep Learning Framework for Pneumonia Detection in Chest X-rays" — SN Computer Science / Springer (2025)
https://link.springer.com/article/10.1007/s42979-025-04017-x

DenseNet-121 + CBAM again, this time compared against heavier ensemble
models. Their headline point is that the CBAM version matched or beat
more complicated ensembles while using far fewer parameters — good
argument to cite for *why* CBAM specifically (cheap, ~2M extra params on
our backbone) rather than something heavier like a second model to
ensemble with. Full text is paywalled past the abstract, only skimmed
what's publicly visible.

### 4. Shahi & Bagale, "Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations" (2025)
https://arxiv.org/pdf/2511.00456v1

Not about CBAM, but directly about the localization problem — using
Grad-CAM as a cheap stand-in for pixel-level annotation, benchmarking
across 7 backbones (ResNet, DenseNet121, EfficientNet, MobileNet, ViT) on
the Kermany CXR dataset (basically our dataset's source). Patient-level
splits called out explicitly to avoid leakage, which matches what
`data/scripts/prepare_pneumonia_dataset.py` already does. This is the
paper that pushed me toward actually *measuring* localization quality
instead of eyeballing heatmaps — hence
`src/explain/measure_lung_localization.py`.

### 5. "Explainable Deep Learning in Medical Imaging: Brain Tumor and Pneumonia Detection" (2025)
https://arxiv.org/html/2510.21823

DenseNet121 vs ResNet50 with Grad-CAM, on the same Kaggle pneumonia
dataset (5,863 images — matches ours almost exactly, 5,856). Reports
DenseNet121 beating ResNet50 on accuracy (89.1% vs 84.4% for the
pneumonia task), and — the actually useful part — says DenseNet121's
Grad-CAM stayed on core pathological regions more consistently, while
ResNet50 sometimes drifted to peripheral/non-pathological areas. Good
independent confirmation that our backbone choice was reasonable, and
another data point for "attention/architecture choice affects heatmap
quality, not just accuracy."

### 6. "Pneumonia Image Classification Using DenseNet Architecture" — MDPI (2024)
https://www.mdpi.com/2078-2489/15/10/611

Compares DenseNet121/169/201 on the same 5,856-image Kaggle set, plain
accuracy comparison (92% normal / 97% pneumonia for the best variant, no
attention mechanism involved). No attention mechanism, so not directly
about the idea I'm implementing — reading list padding, but genuinely
useful as an accuracy sanity-check for our own no-CBAM baseline numbers,
since we're on the exact same dataset with the exact same base
architecture.

---

## What I'm actually doing with this

Papers #1–3 all converge on the same idea: CBAM after the DenseNet
backbone, before the classifier head. Cheap (~2M extra params), and
because of #2's channel-vs-spatial argument, CBAM specifically (not SE)
is the right pick given our project already has an explainability
problem to solve (the shortcut-learning finding), not just an accuracy
one.

Papers #4 and #5 are why I'm not just adding CBAM and calling it done —
both point at *measuring* localization instead of assuming it improved,
which is what `src/explain/measure_lung_localization.py` does: percentage
of Grad-CAM heatmap energy that falls inside vs. outside the segmented
lung field, with and without CBAM.

Implementation: `src/models/attention.py`, wired into both
`src/models/vision_encoder.py` and `src/models/fusion.py` behind a
`use_cbam` config flag (see `docs/architecture.md#attention-module` for
the full writeup). Still need to actually retrain both configs
(`use_cbam: false` vs `true`) and run the localization comparison before
Friday — that's the next step, code side is done.
