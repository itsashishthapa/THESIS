"""
Utility functions placeholder.
Add shared helpers for data loading, preprocessing, metrics, and plotting here.
Keep notebooks thin and import from this module.
"""

from __future__ import annotations


def set_seed(seed: int) -> None:
    """Set common library seeds (NumPy only for now). Extend as needed."""
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
