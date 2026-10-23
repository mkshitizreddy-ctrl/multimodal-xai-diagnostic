
from pathlib import Path
import yaml

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]

with open("configs/fusion.yaml", encoding="utf-8") as f:
    base = yaml.safe_load(f)

for rho in RHOS:
    cfg = yaml.safe_load(yaml.safe_dump(base))
    data_dir = f"data/processed_rho_v2_{rho:g}"

    cfg["data"]["train_csv"] = f"{data_dir}/train.csv"
    cfg["data"]["val_csv"] = f"{data_dir}/val.csv"
    cfg["data"]["test_csv"] = f"{data_dir}/test.csv"

    cfg["checkpoint"]["dir"] = f"checkpoints/fusion_rho_v2_{rho:g}"
    cfg["logging"]["log_dir"] = f"logs/fusion_rho_v2_{rho:g}"

    output = Path(f"configs/fusion_rho_v2_{rho:g}.yaml")
    with output.open("w", encoding="utf-8") as f:
        f.write(f"# Corrected synthetic-signal experiment, rho={rho:g}\n")
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Created {output}")
