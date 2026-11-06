"""Evaluate baseline and lung-aware checkpoints with the region probe and aggregate over seeds.

Runs region_probe.py once per checkpoint (skipping runs whose summary already exists unless --rerun),
then writes docs/lung_aware_eval/lung_aware_summary.{csv,md}. Descriptive only: three seeds, one dataset.

Usage:
    python src/explain/lung_aware_eval.py
    python src/explain/lung_aware_eval.py --aggregate-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explain.lung_aware_eval_utils import KEY_CONDITIONS, aggregate_seeds, summary_markdown

SEEDS = (42, 123, 2024)
BASELINE_S42 = "checkpoints/vision_patient_disjoint_dedup_rho_v1_0"
GROUPS = {
    "baseline": lambda s: BASELINE_S42 if s == 42 else f"checkpoints/vision_baseline_s{s}",
    "random_background": lambda s: f"checkpoints/vision_lung_random_background_s{s}",
    "thorax_only": lambda s: f"checkpoints/vision_lung_thorax_only_s{s}",
}


def main():
    parser = argparse.ArgumentParser(description="Region probe over baseline and lung-aware checkpoints.")
    parser.add_argument("--output-dir", default="docs/lung_aware_eval")
    parser.add_argument("--train-config", default="configs/vision_patient_disjoint_dedup_rho_v1_0.yaml")
    parser.add_argument(
        "--masks-path", default="docs/shortcut_audit_patient_disjoint_dedup_rho_v1_vision/lung_masks.npz"
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    summaries: dict[str, dict[int, pd.DataFrame]] = {}
    for group, ckpt_dir in GROUPS.items():
        summaries[group] = {}
        for seed in SEEDS:
            run_dir = out_root / f"{group}_s{seed}"
            summary_path = run_dir / "region_probe_summary.csv"
            checkpoint = Path(ckpt_dir(seed)) / "best_model.pth"
            if not summary_path.exists() or args.rerun:
                if args.aggregate_only:
                    raise FileNotFoundError(f"{summary_path} missing; run without --aggregate-only")
                if not checkpoint.exists():
                    raise FileNotFoundError(f"{checkpoint} not found")
                print(f"=== {group} seed {seed} ===")
                subprocess.run(
                    [
                        sys.executable,
                        "src/explain/region_probe.py",
                        "--checkpoint", str(checkpoint),
                        "--train-config", args.train_config,
                        "--masks-path", args.masks_path,
                        "--output-dir", str(run_dir),
                    ],
                    check=True,
                )  # fmt: skip
            summaries[group][seed] = pd.read_csv(summary_path)

    aggregate = aggregate_seeds(summaries)
    out_root.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(out_root / "lung_aware_summary.csv", index=False)
    markdown = summary_markdown(aggregate, KEY_CONDITIONS)
    (out_root / "lung_aware_summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Wrote results to {out_root}")


if __name__ == "__main__":
    main()
