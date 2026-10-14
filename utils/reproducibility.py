"""
utils/reproducibility.py
========================
Shared helpers for reproducible runs.

All entry-point scripts expose:
  --seed INT          seed for python / numpy / torch (default 42)
  --deterministic     force deterministic cuDNN kernels (slower)
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every RNG used in this project.

    Args:
        seed          : base seed. DataLoaders derive per-split seeds from it
                        (train = seed, val = seed + 1, test = seed + 2).
        deterministic : if True, use deterministic cuDNN / torch algorithms.
                        Results become bit-reproducible on the same hardware
                        and software stack at the cost of speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def add_seed_args(parser, default: int = 42):
    """Add --seed and --deterministic to an argparse parser."""
    parser.add_argument("--seed", type=int, default=default,
                        help=f"Random seed (default: {default})")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic cuDNN/torch kernels (slower)")
    return parser
