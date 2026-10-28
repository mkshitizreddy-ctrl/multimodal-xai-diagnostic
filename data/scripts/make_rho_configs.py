"""Writes configs/fusion_rho{X}.yaml from configs/fusion.yaml, changing only the data
paths and checkpoint/log dirs. Seed and per-seed dirs are passed on the command line."""
import yaml

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]

with open("configs/fusion.yaml") as f:
    base = yaml.safe_load(f)

for rho in RHOS:
    cfg = yaml.safe_load(yaml.safe_dump(base))  # deep copy
    d = f"data/processed_rho{rho:g}"
    cfg["data"]["train_csv"] = f"{d}/train.csv"
    cfg["data"]["val_csv"] = f"{d}/val.csv"
    cfg["data"]["test_csv"] = f"{d}/test.csv"
    cfg["checkpoint"]["dir"] = f"checkpoints/fusion_rho{rho:g}"
    cfg["logging"]["log_dir"] = f"logs/fusion_rho{rho:g}"
    path = f"configs/fusion_rho{rho:g}.yaml"
    with open(path, "w") as f:
        f.write(
            f"# Fusion with synthetic-vitals signal strength rho={rho:g} "
            "(see data/scripts/make_vitals_rho_variants.py)\n"
        )
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote", path)