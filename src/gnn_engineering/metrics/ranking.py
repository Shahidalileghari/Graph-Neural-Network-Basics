from __future__ import annotations
import numpy as np


def hits_at_k(ranks, k: int) -> float:
    ranks = np.asarray(ranks, dtype=float)
    if k < 1 or ranks.size == 0 or np.any(ranks < 1):
        raise ValueError("k must be >= 1 and ranks must be non-empty positive integers.")
    return float(np.mean(ranks <= k))


def mean_reciprocal_rank(ranks) -> float:
    ranks = np.asarray(ranks, dtype=float)
    if ranks.size == 0 or np.any(ranks < 1):
        raise ValueError("ranks must be non-empty and >= 1.")
    return float(np.mean(1.0 / ranks))
