# Related work notes (2026-11-01)

Purpose: positioning for the shortcut audit (`src/explain/shortcut_audit.py`) and the research plan
(`docs/research_plan_2026-11.md`). Each entry below was read on 2026-11-01 from the linked page, and the
summary states only what that page showed. The search was limited (four queries, top results), so this is
a starting point, not a systematic review, and it does not establish novelty.

## Read and verified

| Work | What it shows | Relevance here |
|---|---|---|
| DeGrave, Janizek, Lee. *AI for radiographic COVID-19 detection selects shortcuts over signal.* medRxiv, Sept 2020 (v2 Oct 2020; later peer reviewed). [PMC7523163](https://pmc.ncbi.nlm.nih.gov/articles/PMC7523163) | DenseNet-121 models lost about half their performance on external data. Saliency maps often highlighted laterality markers, text, image borders, diaphragm and cardiac silhouette. GAN counterfactuals changed markers and border opacity; swapping laterality markers shifted output more than random patch swaps. Patient sex and projection were predictable across datasets. No lung masking. | Same architecture family and the same concern. Their random-patch swap controls motivate our displaced-mask controls. |
| Ben Ahmed et al. *Deep Learning Models May Spuriously Classify Covid-19 from X-ray Images Based on Confounders.* arXiv:2102.04300 (Feb 2021). [arXiv](https://arxiv.org/pdf/2102.04300) | ResNet50 reached 0.997 AUC on held-out data from seen sources but AUC 0.38 and 0.63 on unseen sources in two splits. Lungs were cropped during preprocessing for all images, there was no lungs-removed or background-only test, and the Grad-CAM analysis was called inconclusive. | Shows near-perfect in-source AUROC can hide confounding. They had no second source for the paediatric Kermany data, and neither do we. |
| Teixeira et al. *Impact of lung segmentation on the diagnosis and explanation of COVID-19 in chest X-ray images.* arXiv:2009.09780 (2020). [arXiv](https://ar5iv.labs.arxiv.org/html/2009.09780) | Training on lung-segmented images lowered macro F1 (best 0.88 vs 0.92 unsegmented; COVID-19 F1 0.83 vs 0.94; Wilcoxon p = 0.019), but Grad-CAM and LIME focused on the lungs, whereas full-image models often focused outside them (e.g. burned-in annotations). Classifying the data source worked better on full images (F1 0.93) than on segmented ones (0.78). | The closest prior to our "does the model need the lungs" question and to the remedy step. They trained on masked images; we start from a frozen model with controlled interventions, on paediatric Kermany. Their source-classification result suggests background carries dataset-source information, so we should expect background-only AUROC above 0.5. |
| Ridwan. *XAI-Guided Analysis of Residual Networks for Interpretable Pneumonia Detection in Paediatric Chest X-rays.* arXiv:2507.18647 (July 2025 by arXiv ID). [arXiv](https://arxiv.org/pdf/2507.18647) | Kermany dataset, ResNet-50, Grad-CAM and BayesGrad-CAM. Says Grad-CAM highlights pulmonary regions but reports no systematic inside-versus-outside-lung measure, runs no shortcut tests, and has not validated maps against expert annotation. Randomization sanity checks proposed as future work. | Same dataset. Confirms the gap our lung-fraction and intervention analysis addresses. |
| Faruqui and Dubey. *Explanation-Aware Learning for Enhanced Interpretability in Biomedical Imaging.* arXiv:2605.10054 (May 2026 by arXiv ID). [arXiv](https://arxiv.org/pdf/2605.10054) | Adds a loss to the classification loss that penalizes Grad-CAM mass outside radiologist bounding boxes (VinDr-CXR, DenseNet-121, seven binary tasks). Accuracy stayed broadly comparable; larger weights concentrated saliency. | Closest to our remedy step. Differences: they use bounding boxes on adult data, and we would use lung masks on paediatric data with a shortcut audit as motivation. |

## Cited from memory, not re-verified this session (verify before citing)

- Zech et al., 2018, *PLoS Medicine*: pneumonia CNNs and hospital-system confounding.
- Ross, Hughes, Doshi-Velez, 2017, IJCAI: "Right for the right reasons" (penalizing input gradients outside the relevant region).
- Adebayo et al., 2018, NeurIPS: sanity checks for saliency maps (model and label randomization).

## What this implies for the plan

- Position the work as an audit and controlled replication on the deduplicated paediatric Kermany split,
  linking intervention results to a per-image explanation metric. Do not claim a new method.
- Predictions to test: background-only AUROC above 0.5; lung-only close to the original; per-image lung-energy
  fraction related to background reliance.
- The adult-trained lung segmenter is a validity risk on these images. Report the mask QC.
- Randomization sanity checks (Adebayo et al.) are a natural extension if time allows.
