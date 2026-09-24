"""Deterministic seeding for training runs.

torch.manual_seed() alone does not make a training run reproducible in this
codebase. Two gaps found while investigating why seed 2024's
attention-consistency rerun diverged substantially from its original run
(localization 0.537 vs. 0.642 - a gap larger than the effect size being
measured):

1. DataLoader workers (num_workers > 0 in every config here) are separate
   processes. torchvision's classic transforms - RandomHorizontalFlip,
   RandomRotation, RandomAffine, all used in this project's training
   augmentation - draw from Python's built-in `random` module internally,
   not torch's RNG. Nothing here ever seeded `random` in worker processes,
   so augmentation was effectively uncontrolled by --seed.
2. cuDNN's default convolution algorithms are non-deterministic on GPU for
   performance reasons, independent of any RNG seeding.

set_seed() fixes both: seeds Python's random, NumPy, and torch (CPU + all
CUDA devices), and pins cuDNN to its deterministic algorithms.
seeded_worker_init_fn, passed to DataLoader as worker_init_fn, reseeds each
worker process's random/NumPy/torch state from the same base seed.

Note: torch.use_deterministic_algorithms(True) is NOT enabled here - it
raises at runtime for any operation lacking a deterministic implementation,
and confirming every op used across all three training scripts (including
CBAM's custom modules) has one is out of scope for this fix. cuDNN
determinism plus fixed RNG seeding closes the two gaps actually identified
above; use_deterministic_algorithms would be the next thing to try if a
gap remains after this.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG this codebase's training pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic (slower) convolution algorithms instead of cuDNN's
    # default performance-tuned, non-deterministic ones.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeded_worker_init_fn(worker_id: int) -> None:
    """Reseed a DataLoader worker process's random/NumPy/torch state.

    Pass as DataLoader(..., worker_init_fn=seeded_worker_init_fn). Without
    this, each worker process's Python `random` state - what
    torchvision's classic RandomHorizontalFlip/RandomRotation/RandomAffine
    actually draw from - is whatever the OS/interpreter gave it at process
    startup, unrelated to set_seed() in the main process.

    torch.initial_seed() inside a worker returns a value PyTorch already
    derives deterministically from the main process's seed plus worker_id,
    so reusing it here (rather than inventing a separate scheme) keeps
    torch's own per-worker RNG and this function's seeding in agreement.
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
