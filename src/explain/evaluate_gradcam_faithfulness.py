
"""Evaluate Grad-CAM patch deletion against a random-patch baseline.

This is a model-sensitivity experiment, not evidence of clinical causality,
clinical validity, or clinically correct localization.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ChestXrayDataset
from src.explain.gradcam import ChestXrayExplainer
from src.models.vision_encoder import ChestXrayVisionModel


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_output(model, image, device):
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
    return float(logits.reshape(-1)[0].item())


def grid_bounds(length, grid_size):
    """Return patch boundaries that cover the dimension exactly."""
    return np.linspace(0, length, grid_size + 1, dtype=int)


def patch_scores(heatmap, grid_size):
    """Average Grad-CAM activation within each grid patch."""
    height, width = heatmap.shape
    ys = grid_bounds(height, grid_size)
    xs = grid_bounds(width, grid_size)
    scores = np.empty((grid_size, grid_size), dtype=np.float64)

    for r in range(grid_size):
        for c in range(grid_size):
            patch = heatmap[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            scores[r, c] = patch.mean()

    return scores.reshape(-1), ys, xs


def delete_patches(image, patch_indices, ys, xs, grid_size):
    """Replace selected patches with normalized gray (zero)."""
    result = image.clone()
    for index in patch_indices:
        r, c = divmod(int(index), grid_size)
        result[:, ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = 0.0
    return result


def bootstrap_mean_ci(values, rng, replicates=2000):
    """Percentile bootstrap CI for the mean of paired per-image values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, np.nan, np.nan

    boot_means = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[b] = sample.mean()

    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return estimate, float(lower), float(upper)


def fraction_key(fraction):
    return f"{fraction:.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare Grad-CAM patch deletion with random patch deletion."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Vision checkpoint to evaluate. Required: it must match --test-csv's split "
        "(e.g. checkpoints/vision_patient_disjoint_dedup_rho_v1_0/best_model.pth).",
    )
    parser.add_argument(
        "--test-csv",
        default="data/processed_patient_disjoint_dedup_rho_v1_0/test.csv",
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/vision_baseline.yaml")
    parser.add_argument(
        "--output-dir",
        default="docs/gradcam_faithfulness_patient_disjoint_dedup_rho_v1_vision",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 evaluates the complete test set.",
    )
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.40],
    )
    args = parser.parse_args()

    if args.limit < 0 or args.random_repeats < 1:
        parser.error("--limit must be >= 0 and --random-repeats must be positive")
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if any(not 0 < f < 1 for f in args.fractions):
        parser.error("Every deletion fraction must be between 0 and 1")
    if sorted(set(args.fractions)) != args.fractions:
        parser.error("--fractions must be unique and increasing")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    bootstrap_rng = np.random.default_rng(args.seed + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.train_config)

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    classes = checkpoint["classes"]

    if len(classes) != 1:
        raise ValueError(
            "Expected a one-logit binary Pneumonia model; "
            f"checkpoint classes={classes!r}"
        )

    model = ChestXrayVisionModel(
        num_classes=len(classes),
        pretrained=False,
        use_cbam=checkpoint.get("use_cbam", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = ChestXrayDataset(
        csv_path=args.test_csv,
        image_dir=train_cfg["data"]["image_dir"],
        classes=classes,
        tabular_features=data_cfg["tabular_features"],
        image_size=train_cfg["data"]["image_size"],
        train=False,
    )

    n = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))
    explainer = ChestXrayExplainer(model, device=str(device))
    rows = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Evaluating {n}/{len(dataset)} images")
    print(f"Random patch masks per fraction: {args.random_repeats}")
    print(f"Patch grid: {args.grid_size} x {args.grid_size}")
    print(f"Bootstrap replicates: {args.bootstrap_replicates}")

    for i in range(n):
        image, _tabular, labels = dataset[i]
        original_logit = model_output(model, image, device)
        original_prob = float(
            torch.sigmoid(torch.tensor(original_logit)).item()
        )

        _, heatmap = explainer.explain(image, class_idx=0)
        heatmap = np.asarray(heatmap, dtype=np.float64)

        if heatmap.ndim != 2:
            raise ValueError(
                f"Expected a 2-D Grad-CAM map at test index {i}; "
                f"got shape {heatmap.shape}"
            )
        if not np.isfinite(heatmap).all():
            raise ValueError(f"Non-finite Grad-CAM values at test index {i}")

        height, width = image.shape[-2:]
        if heatmap.shape != (height, width):
            raise ValueError(
                f"Heatmap/image size mismatch at index {i}: "
                f"{heatmap.shape} versus {(height, width)}"
            )

        scores, ys, xs = patch_scores(heatmap, args.grid_size)
        zero_map = float(heatmap.max()) <= 0.0
        ranked_patches = np.argsort(-scores, kind="stable")

        record = {
            "test_index": i,
            "image": dataset.df.iloc[i]["Image Index"],
            "true_label": float(labels[0].item()),
            "original_logit": original_logit,
            "original_probability": original_prob,
            "zero_map": zero_map,
            "heatmap_max": float(heatmap.max()),
            "grid_size": args.grid_size,
        }

        total_patches = args.grid_size * args.grid_size

        for fraction in args.fractions:
            # The actual fraction is quantized to a whole number of patches.
            k = max(1, min(
                total_patches - 1,
                int(round(fraction * total_patches)),
            ))
            actual_fraction = k / total_patches
            key = fraction_key(fraction)

            record[f"requested_fraction_{key}"] = fraction
            record[f"actual_patch_fraction_{key}"] = actual_fraction
            record[f"patch_count_{key}"] = k

            if not zero_map:
                cam_indices = ranked_patches[:k]
                cam_image = delete_patches(
                    image, cam_indices, ys, xs, args.grid_size
                )
                cam_logit = model_output(model, cam_image, device)
                cam_prob = float(
                    torch.sigmoid(torch.tensor(cam_logit)).item()
                )
                record[f"cam_logit_drop_{key}"] = (
                    original_logit - cam_logit
                )
                record[f"cam_probability_drop_{key}"] = (
                    original_prob - cam_prob
                )
            else:
                record[f"cam_logit_drop_{key}"] = np.nan
                record[f"cam_probability_drop_{key}"] = np.nan

            random_logit_drops = []
            random_probability_drops = []

            for _ in range(args.random_repeats):
                random_indices = rng.choice(
                    total_patches, size=k, replace=False
                )
                random_image = delete_patches(
                    image, random_indices, ys, xs, args.grid_size
                )
                random_logit = model_output(model, random_image, device)
                random_prob = float(
                    torch.sigmoid(torch.tensor(random_logit)).item()
                )
                random_logit_drops.append(original_logit - random_logit)
                random_probability_drops.append(original_prob - random_prob)

            record[f"random_logit_drop_mean_{key}"] = float(
                np.mean(random_logit_drops)
            )
            record[f"random_logit_drop_sd_{key}"] = float(
                np.std(random_logit_drops, ddof=1)
                if len(random_logit_drops) > 1 else 0.0
            )
            record[f"random_probability_drop_mean_{key}"] = float(
                np.mean(random_probability_drops)
            )
            record[f"random_probability_drop_sd_{key}"] = float(
                np.std(random_probability_drops, ddof=1)
                if len(random_probability_drops) > 1 else 0.0
            )

        rows.append(record)
        print(
            f"[{i + 1}/{n}] {record['image']} | "
            f"label={record['true_label']:.0f} | "
            f"prob={original_prob:.4f} | zero_map={zero_map}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements = pd.DataFrame(rows)

    csv_path = output_dir / "faithfulness_measurements.csv"
    measurements.to_csv(csv_path, index=False)

    nonzero = measurements.loc[~measurements["zero_map"]].copy()
    zero = measurements.loc[measurements["zero_map"]].copy()

    summary = [
        "# Grad-CAM patch-deletion experiment",
        "",
        f"- Test CSV: `{args.test_csv}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Images evaluated: {len(measurements)}",
        f"- Nonzero Grad-CAM maps: {len(nonzero)}",
        f"- Zero Grad-CAM maps: {len(zero)}",
        f"- Random masks per image/fraction: {args.random_repeats}",
        f"- Patch grid: {args.grid_size} x {args.grid_size}",
        f"- Bootstrap replicates: {args.bootstrap_replicates}",
        f"- Random seed: {args.seed}",
        "",
        "Each selected patch is replaced by zero in normalized input space, "
        "corresponding to mid-gray. The selected patch fraction is quantized "
        "to the nearest whole number of grid patches.",
        "",
        "Grad-CAM and random-patch comparisons use the same nonzero-map images. "
        "Zero-map cases have no Grad-CAM ranking and are excluded from the "
        "primary paired comparison.",
        "",
        "Random patches are spatially coherent individually but may be "
        "scattered across the image. This is a patch-level random baseline, "
        "not a matched contiguous-region baseline.",
        "",
        "## Primary result: mean logit decrease",
        "",
        "Positive paired difference means Grad-CAM deletion caused a larger "
        "logit decrease than the per-image mean random-patch deletion.",
        "",
        "| Requested fraction | Actual patch fraction | n paired | "
        "Grad-CAM mean drop | Random mean drop (same images) | "
        "Paired difference | Bootstrap 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for fraction in args.fractions:
        key = fraction_key(fraction)
        cam_col = f"cam_logit_drop_{key}"
        random_col = f"random_logit_drop_mean_{key}"
        actual_col = f"actual_patch_fraction_{key}"

        cam_values = nonzero[cam_col].to_numpy(dtype=float)
        random_values = nonzero[random_col].to_numpy(dtype=float)
        paired_difference = cam_values - random_values

        estimate, lower, upper = bootstrap_mean_ci(
            paired_difference, bootstrap_rng, args.bootstrap_replicates
        )

        cam_mean = float(np.nanmean(cam_values)) if len(cam_values) else np.nan
        random_mean = (
            float(np.nanmean(random_values)) if len(random_values) else np.nan
        )
        actual = (
            float(nonzero[actual_col].mean()) if len(nonzero) else np.nan
        )

        summary.append(
            f"| {fraction:.0%} | {actual:.2%} | {len(paired_difference)} | "
            f"{cam_mean:.6f} | {random_mean:.6f} | {estimate:.6f} | "
            f"[{lower:.6f}, {upper:.6f}] |"
        )

    summary += [
        "",
        "## Zero-map subgroup",
        "",
        f"- Zero-map images: {len(zero)} of {len(measurements)} "
        f"({len(zero) / len(measurements):.2%}).",
        "- Grad-CAM deletion metrics are undefined for these images; random "
        "patch deletion results are still recorded in the CSV.",
        "",
        "## Class counts",
        "",
        "| True label | Images | Zero maps |",
        "|---:|---:|---:|",
    ]

    for label in sorted(measurements["true_label"].dropna().unique()):
        group = measurements[measurements["true_label"] == label]
        summary.append(
            f"| {label:.0f} | {len(group)} | {int(group['zero_map'].sum())} |"
        )

    summary += [
        "",
        "## Interpretation cautions",
        "",
        "- Deletion can create out-of-distribution inputs; results measure "
        "sensitivity to this specific intervention.",
        "- Grid patches are not lung-aware regions, and this experiment does "
        "not establish clinically correct localization or causal reasoning.",
        "- Bootstrap intervals describe uncertainty in the mean paired "
        "difference across evaluated images; random-repeat SD is not a "
        "population confidence interval.",
        "- This experiment must not be interpreted as clinical validation.",
        "",
    ]

    summary_path = output_dir / "faithfulness_summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print(f"\nSaved measurements: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
