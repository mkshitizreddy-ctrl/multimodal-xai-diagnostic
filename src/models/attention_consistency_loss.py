"""
Attention-consistency loss: penalizes CBAM's spatial attention map for
putting weight outside the segmented lung field, instead of only measuring
this after training the way src/explain/measure_lung_localization.py does.

Deliberately defined as (1 - the same lung-energy-fraction quantity that
measure_lung_localization.py reports) — training directly optimizes the
exact metric this project already uses to evaluate localization quality,
rather than a different proxy loss that happens to point in a similar
direction. See docs/paper_notes.md (paper #4, Shahi & Bagale) for the
literature basis for treating localization as something to train toward,
not just audit after the fact.
"""

import torch


def attention_consistency_loss(
    attention_map: torch.Tensor, lung_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Args:
        attention_map: CBAM's spatial attention map, shape [B, 1, H, W],
            values in [0, 1] (sigmoid output — see
            CBAM.get_spatial_attention_map in src/models/attention.py).
        lung_mask: precomputed binary lung segmentation mask, shape
            [B, H, W] or [B, 1, H, W], same spatial resolution as
            attention_map (both are 7x7 for a 224x224 input with the
            current DenseNet-121 backbone — see
            data/scripts/precompute_lung_masks.py).
        eps: avoids division by zero for the rare degenerate case of a
            near-all-zero attention map (mirrors the same guard in
            src/explain/measure_lung_localization.py's
            lung_energy_fraction()).

    Returns:
        Scalar loss, mean over the batch. 0.0 = every sample's attention
        falls entirely inside its lung mask (perfect localization by this
        measure); 1.0 = attention falls entirely outside.
    """
    if lung_mask.dim() == 3:
        lung_mask = lung_mask.unsqueeze(1)

    if attention_map.shape != lung_mask.shape:
        raise ValueError(
            f"attention_map shape {tuple(attention_map.shape)} doesn't match "
            f"lung_mask shape {tuple(lung_mask.shape)} — precomputed masks "
            "must be generated at the same resolution as the model's "
            "attention map (see data/scripts/precompute_lung_masks.py)."
        )

    total_per_sample = attention_map.sum(dim=(1, 2, 3))
    inside_per_sample = (attention_map * lung_mask).sum(dim=(1, 2, 3))
    lung_fraction_per_sample = inside_per_sample / (total_per_sample + eps)

    return 1.0 - lung_fraction_per_sample.mean()
