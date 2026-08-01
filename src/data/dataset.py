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


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        classes: list[str],
        tabular_features: list[str],
        image_size: int = 224,
        train: bool = False,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.tabular_features = tabular_features
        self.train = train

        # Precompute normalization stats for tabular features (fit on this split)
        self._fit_tabular_normalizers()

        self.transform = self._build_transform(image_size, train)

    def _fit_tabular_normalizers(self) -> None:
        self.tabular_means = {}
        self.tabular_stds = {}
        for col in self.tabular_features:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.tabular_means[col] = self.df[col].mean()
                self.tabular_stds[col] = self.df[col].std() or 1.0
            else:
                # Categorical -> map to a stable integer vocabulary
                categories = sorted(self.df[col].astype(str).unique())
                self.tabular_vocab = getattr(self, "tabular_vocab", {})
                self.tabular_vocab[col] = {c: i for i, c in enumerate(categories)}

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
