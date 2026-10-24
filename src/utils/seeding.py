"""Tekrarlanabilirlik için seed sabitleme (K8)."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Python `random`, NumPy ve PyTorch (CPU + CUDA) seed'lerini sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
