"""
Tabular encoder: a small MLP that embeds patient metadata (age, gender,
view position) into a fixed-size vector for fusion with the vision branch.
"""

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, num_features: int, embedding_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
