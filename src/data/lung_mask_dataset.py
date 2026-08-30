"""
Wraps ChestXrayDataset to also return a precomputed lung mask alongside
image/tabular/labels — used only by src/train_attention_consistency.py.

Deliberately a separate wrapper class rather than modifying
ChestXrayDataset itself: every other script in this project (train.py,
train_fusion.py, evaluate.py, evaluate_fusion.py, the dashboard, every
explain/*.py script, and 50+ existing tests) unpacks
`image, tabular, labels = dataset[i]` — changing that to a 4-tuple
everywhere it's constructed would be a much larger, riskier change for a
single experimental training variant. This wrapper touches none of that.
"""

import pickle

import torch

from src.data.dataset import ChestXrayDataset


class LungMaskAugmentedDataset(ChestXrayDataset):
    def __init__(self, *args, lung_masks_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        with open(lung_masks_path, "rb") as f:
            self._lung_masks = pickle.load(f)

        missing = [
            fname for fname in self.df["Image Index"] if fname not in self._lung_masks
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} images in this dataset have no precomputed lung mask "
                f"in {lung_masks_path} (e.g. {missing[0]}). Re-run "
                "data/scripts/precompute_lung_masks.py, or check that --splits "
                "covered this CSV's split."
            )

    def __getitem__(self, idx: int):
        image, tabular, labels = super().__getitem__(idx)
        filename = self.df.iloc[idx]["Image Index"]
        lung_mask = torch.from_numpy(self._lung_masks[filename])
        return image, tabular, labels, lung_mask
