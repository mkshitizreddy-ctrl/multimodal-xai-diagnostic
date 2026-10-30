# GPU latency benchmark notes (2026-10-30)

**Status: informal measurement, not a reported result.** No latency claim is made anywhere in this
repository, and these numbers do not support one. The notes exist so the runs are reproducible and
their limits are recorded.

## Setup

- Script: `benchmark_latency.py` (commit `c486c9b`), run once with each `--cudnn-benchmark` /
  `--no-cudnn-benchmark` setting, one after the other.
- Hardware / software: NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.9.1+cu128.
- Models: the rho = 0 patient-disjoint, deduplicated vision and fusion checkpoints. Vision has
  7,087,139 parameters; fusion has 7,374,435 (+287,296, about 4%).
- Input: random tensors, batch size 1 (`[1, 3, 224, 224]` image, `[1, 4]` tabular), `torch.inference_mode()`.
  Latency does not depend on the weight values or the image content, so no real data is involved.
- Protocol: 40 warm-up iterations, then 15 rounds of 100 iterations each; the order of vision and
  fusion alternates between rounds. Reported unit is ms/image (batch size 1).
- The script records the GPU name and `nvidia-smi` snapshots (temperature, utilization, memory,
  power) before and after timing. It does not record checkpoint hashes.

## Results

| Setting | Vision median | Vision mean | Fusion median | Fusion mean | Fusion vs vision (median) |
|---|---|---|---|---|---|
| `--cudnn-benchmark` | 16.424 | 15.799 | 17.078 | 16.804 | +3.98% |
| `--no-cudnn-benchmark` | 18.173 | 18.418 | 18.043 | 18.269 | -0.72% |

All values are ms/image.

Per-round values (ms/image):

`--cudnn-benchmark`
- Vision: 12.89, 13.216, 12.739, 12.695, 14.152, 17.909, 18.408, 17.006, 16.163, 16.424, 17.859, 16.493, 16.175, 16.764, 18.097
- Fusion: 13.263, 13.748, 13.242, 12.712, 24.42, 18.511, 16.779, 17.078, 17.195, 17.666, 17.733, 16.612, 17.816, 18.263, 17.02
- GPU before / after: 41.25 C, 27 W / 48.86 C, 18.16 W

`--no-cudnn-benchmark`
- Vision: 21.561, 16.973, 17.738, 18.623, 18.174, 18.038, 17.924, 18.173, 19.365, 18.408, 18.974, 17.579, 17.579, 17.41, 19.747
- Fusion: 17.679, 19.443, 19.327, 19.531, 16.982, 18.174, 17.776, 18.043, 17.989, 18.676, 17.773, 19.055, 17.911, 18.638, 17.044
- GPU before / after: 47.25 C, 26.67 W / 52.70 C, 18.12 W

## Interpretation and limits

- **No reliable fusion-vs-vision difference.** The sign of the median change flips between the two
  runs (+3.98% vs -0.72%), so the data show no consistent latency cost for the tabular branch.
- **The `--cudnn-benchmark` run drifted.** Rounds 1-4 sit at about 12.7-13.7 ms, then latency steps up
  to about 16-18 ms from round 5 (plus one 24.42 ms fusion outlier). Round-to-round coefficient of
  variation is about 13% (vision) and 17% (fusion), against about 6% and 4% without the flag. Reported
  power draw fell from 27 W to 18 W over the run, which is consistent with a laptop GPU clock or
  power-state change, but the cause was not established.
- **The two settings are not a clean comparison.** The runs were sequential with different starting
  temperatures (41 C vs 47 C) and a drifting state in the first. No conclusion about whether
  cuDNN benchmark mode helps or hurts should be drawn.
- Single machine, single run per setting, batch size 1, a laptop GPU with variable clocks. Not
  evidence of deployment latency on any other hardware (including the CPU-only Hugging Face Space).

A defensible latency number would need repeated independent runs, a fixed clock or power state, and
the checkpoint hashes logged alongside the output.
