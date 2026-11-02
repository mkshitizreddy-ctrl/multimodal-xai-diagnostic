# CNN versus acquisition-baseline discordance

Within-class Spearman correlation between the CNN logit and each baseline's score (test split).

| Baseline | Pneumonia (n) | rho | Normal (n) | rho |
|---|---|---|---|---|
| file_size_only | 849 | +0.169 | 316 | +0.267 |
| size_meta+intensity | 849 | +0.278 | 316 | +0.355 |

File-size baseline at the validation Youden threshold (0.71): accuracy 0.953, 55 errors (FN 20, FP 35).

| Subset | n | Pneumonia | Normal | CNN AUROC [95% CI] | CNN error rate |
|---|---|---|---|---|---|
| baseline right | 1110 | 829 | 281 | 0.9993 [0.9986, 0.9998] | 0.011 |
| baseline wrong | 55 | 20 | 35 | 0.9229 [0.8450, 0.9780] | 0.200 |

CNN errors at logit > 0: 23 (FN 12, FP 11); 11 of them are also file-size baseline errors.
Median file size of CNN false positives: 121.2 KB (all Normal: 494.5 KB).
Median file size of CNN false negatives: 137.9 KB (all Pneumonia: 71.1 KB).
