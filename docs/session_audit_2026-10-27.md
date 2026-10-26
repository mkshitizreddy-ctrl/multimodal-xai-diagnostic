# Research session audit — 2026-10-27

This note records the final repository review for the research session captured in the referenced “GitHub Project” conversation. It supplements the experiment-specific report at `docs/patient_disjoint_dedup_rho_experiment_report_v1.md`; it does not replace that report or claim new model results.

## Confirmed in the local checkout

- The repository was on `master`, tracking `origin/master`, with no tracked modifications or staged changes at the start of the audit. The only untracked files other than the explicitly excluded `README.local-backup.md` were `benchmark_latency.py` and `benchmark_latency_v2.py`.
- The existing patient-disjoint, deduplicated rho=0 experiment report and its Grad-CAM, raw-map, and lung-localization artifacts already preserve the reported results and limitations. In particular, the test set has 1,165 images, 268 zero-valued Grad-CAM maps, and 897 nonzero maps. The report states that the zero maps arose from entirely negative raw maps before Grad-CAM ReLU; it also describes the conditional lung-localization statistic and the limits of the patch-deletion and counterfactual analyses.
- The localization CSV (`docs/lung_localization_testset_rho0.csv`), Grad-CAM diagnostic CSV (`docs/gradcam_testset_diagnostic_patient_disjoint_dedup_rho_v1_vision/gradcam_testset_measurements.csv`), and processed test split were joined by index. All 1,165 rows matched one-to-one, and every localization image filename matched the filename at the corresponding test-split row (0 mismatches).
- On those joined rows, the largest absolute difference between the localization and diagnostic pneumonia probabilities was `1.192092896618035e-07`; all 268 undefined lung-energy fractions corresponded to zero maps; and both probability columns had 317 values below 0.5. Both index columns were unique.
- The recent chat also contained an earlier check reporting that the same filename alignment passed with 0 mismatches. This was independently repeated against the local files during this audit.

## Preserved benchmark utilities and unresolved provenance

`benchmark_latency.py` and `benchmark_latency_v2.py` are retained as GPU latency measurement utilities for the local rho=0 vision and fusion checkpoints. They use synthetic inputs, warm up both models, collect repeated CUDA-event timings, and print GPU snapshots when `nvidia-smi` is available. The first enables cuDNN benchmarking before measurement; the second disables it. No benchmark output, hardware/runtime record, or conversation evidence confirming that either script was executed was present in the audited materials. Therefore this repository records these as utilities only and makes no latency claim.

Checkpoint files and training logs are excluded from Git under the repository's existing policy. The current audit materials do not establish which local checkpoint bytes, if any, produced every older result, nor do they independently authenticate checkpoint provenance. No model was retrained and no research output was regenerated or modified for this audit.

## Remote synchronization status

After refreshing `origin` from GitHub, the local branch and remote `master` were both at commit `6aad2b74100a957a09d66ca8daf94cf68ff22145`, with zero commits ahead or behind before this audit commit. A successful push is still required to publish the changes recorded here.

## Next steps

1. Push this audit commit to `origin/master` and verify that the remote points to the new commit.
2. If latency results are later collected, record the GPU model, PyTorch/CUDA/cuDNN versions, checkpoint identity, and complete timing output alongside the run; treat the current scripts as measurement tools, not results.
3. Continue to describe Grad-CAM overlap and deletion findings as model- and intervention-specific proxies, not clinical validation or causal evidence.
