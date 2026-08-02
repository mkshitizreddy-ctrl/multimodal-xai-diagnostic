"""
PyTorch Dataset for the NIH Chest X-ray14 multimodal pipeline.

Each sample returns:
    image:   FloatTensor [3, H, W]   (grayscale X-ray replicated to 3 channels)
    tabular: FloatTensor [num_tabular_features]
    labels:  FloatTensor [num_classes]  (multi-hot)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# pandas >= 3.0 defaults to pyarrow-backed string columns in read_csv(), which
# has been observed to cause a silent access-violation crash in pyarrow's
# arrow.dll on some Windows setups (no Python traceback — the process just
# exits). Disabling this reverts to pandas' legacy object-dtype strings,
# sidestepping pyarrow entirely for CSV loading. Wrapped in try/except since
# this option doesn't exist on older pandas versions.
try:
    pd.options.future.infer_string = False
except AttributeError:
    pass


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        classes: list[str],
        tabular_features: list[str],
        image_size: int = 224,
        train: bool = False,
        tabular_stats: dict | None = None,
    ):
        """
        Args:
            tabular_stats: previously-fitted normalization stats (from
                get_tabular_stats() on another split — normally the train
                split). When provided, this dataset REUSES those stats
                instead of fitting its own. This matters: fitting
                separately per split means age normalization and the
                categorical vocabulary (e.g. which integer "Male" maps to)
                could differ between train/val/test, silently corrupting
                results. Always fit once on train and pass the result to
                val/test — see src/train.py's build_dataloaders() and
                src/evaluate.py for the reference usage pattern.
        """
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.tabular_features = tabular_features
        self.train = train

        if tabular_stats is not None:
            self._load_tabular_stats(tabular_stats)
        else:
            self._fit_tabular_normalizers()

        self.transform = self._build_transform(image_size, train)

    def _fit_tabular_normalizers(self) -> None:
        self.tabular_means = {}
        self.tabular_stds = {}
        self.tabular_vocab = {}
        for col in self.tabular_features:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.tabular_means[col] = float(self.df[col].mean())
                self.tabular_stds[col] = float(self.df[col].std() or 1.0)
            else:
                # Categorical -> map to a stable integer vocabulary
                categories = sorted(self.df[col].astype(str).unique())
                self.tabular_vocab[col] = {c: i for i, c in enumerate(categories)}

    def _load_tabular_stats(self, stats: dict) -> None:
        self.tabular_means = dict(stats["tabular_means"])
        self.tabular_stds = dict(stats["tabular_stds"])
        self.tabular_vocab = {k: dict(v) for k, v in stats["tabular_vocab"].items()}

    def get_tabular_stats(self) -> dict:
        """Returns this dataset's fitted (or reused) normalization stats,
        for passing into another split's constructor via tabular_stats=."""
        return {
            "tabular_means": dict(self.tabular_means),
            "tabular_stds": dict(self.tabular_stds),
            "tabular_vocab": {k: dict(v) for k, v in self.tabular_vocab.items()},
        }

    @staticmethod
    def _build_transform(image_size: int, train: bool) -> transforms.Compose:
        aug = []
        if train:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *aug,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.25] * 3),
            ]
        )

    def _encode_tabular(self, row: pd.Series) -> torch.Tensor:
        values = []
        for col in self.tabular_features:
            if col in self.tabular_means:
                val = (row[col] - self.tabular_means[col]) / self.tabular_stds[col]
            else:
                vocab = self.tabular_vocab[col]
                val = vocab.get(str(row[col]), 0)
            values.append(float(val))
        return torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = self.image_dir / row["Image Index"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        tabular = self._encode_tabular(row)

        labels = torch.tensor(
            row[self.classes].values.astype(np.float32), dtype=torch.float32
        )

        return image, tabular, labels
