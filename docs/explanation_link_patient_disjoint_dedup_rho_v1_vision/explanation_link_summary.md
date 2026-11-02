# Grad-CAM lung fraction versus audit logits

Pneumonia images with a nonzero Grad-CAM map: n=847. Partial Spearman correlation of the
lung-energy fraction with each logit, controlling for lung-mask area; 95% image-level bootstrap CI.

| Logit | Partial rho [95% CI] |
|---|---|
| original | -0.415 [-0.476, -0.348] |
| lung_only | -0.025 [-0.095, +0.046] |
| background_only | -0.634 [-0.677, -0.582] |
| displaced_lung_only | -0.282 [-0.343, -0.211] |
| displaced_background_only | -0.472 [-0.529, -0.409] |
| contrast_true (lung_only - background_only) | +0.536 [+0.479, +0.585] |
| contrast_displaced (matched control) | +0.333 [+0.266, +0.393] |
| difference (true contrast - displaced contrast) | +0.203 [+0.129, +0.281] |

Enrichment (lung fraction / mask area): median 1.26, 97.0% of maps above 1.
