
from pathlib import Path

import yaml

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]

BASE_DIR = Path("configs")
DATA_PREFIX = "data/processed_patient_disjoint_dedup_rho_v1"

# Each generated configuration must have a unique output filename.
CONFIGS = {
    "fusion": BASE_DIR / "fusion.yaml",
    "vision": BASE_DIR / "vision_baseline.yaml",
}


def load_yaml(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing base configuration: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    for key in ("data", "checkpoint", "logging", "train", "model"):
        if key not in config:
            raise ValueError(f"{path} is missing section: {key}")

    return config


def main():
    bases = {
        name: load_yaml(path)
        for name, path in CONFIGS.items()
    }

    outputs = []

    for rho in RHOS:
        suffix = f"patient_disjoint_dedup_rho_v1_{rho:g}"
        data_dir = f"{DATA_PREFIX}_{rho:g}"

        for model_type, base in bases.items():
            cfg = yaml.safe_load(yaml.safe_dump(base))

            cfg["data"]["train_csv"] = f"{data_dir}/train.csv"
            cfg["data"]["val_csv"] = f"{data_dir}/val.csv"
            cfg["data"]["test_csv"] = f"{data_dir}/test.csv"

            cfg["checkpoint"]["dir"] = f"checkpoints/{model_type}_{suffix}"
            cfg["logging"]["log_dir"] = f"logs/{model_type}_{suffix}"

            output = BASE_DIR / f"{model_type}_{suffix}.yaml"
            outputs.append((output, rho, model_type, cfg))

    # Check everything before writing any files.
    existing = [str(path) for path, *_ in outputs if path.exists()]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing configurations:\n"
            + "\n".join(existing)
        )

    missing_data = []

    for rho in RHOS:
        data_dir = Path(f"{DATA_PREFIX}_{rho:g}")
        for split in ("train", "val", "test"):
            csv_path = data_dir / f"{split}.csv"
            if not csv_path.is_file():
                missing_data.append(str(csv_path))

    if missing_data:
        raise FileNotFoundError(
            "Required rho CSVs are missing:\n" + "\n".join(missing_data)
        )

    for output, rho, model_type, cfg in outputs:
        with output.open("x", encoding="utf-8") as file:
            file.write(
                f"# Deduplicated patient-grouped synthetic experiment, "
                f"rho={rho:g}, model={model_type}\n"
            )
            yaml.safe_dump(cfg, file, sort_keys=False)

        print(f"Created {output}")

    print(f"\nCreated {len(outputs)} configurations.")
    print("Existing files were not overwritten.")


if __name__ == "__main__":
    main()
