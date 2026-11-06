from __future__ import annotations
import numpy as np


def validate_masks(train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray,
                   num_nodes: int) -> None:
    masks = [np.asarray(m, dtype=bool) for m in (train_mask, val_mask, test_mask)]
    if any(m.ndim != 1 or len(m) != num_nodes for m in masks):
        raise ValueError("Each mask must be a 1D array with one entry per node.")
    if np.any(masks[0] & masks[1]) or np.any(masks[0] & masks[2]) or np.any(masks[1] & masks[2]):
        raise ValueError("Train, validation, and test masks must be mutually exclusive.")
    if not any(mask.any() for mask in masks):
        raise ValueError("At least one split must contain nodes.")
