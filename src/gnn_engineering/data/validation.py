from __future__ import annotations
import numpy as np


def validate_node_features(x: np.ndarray) -> None:
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("Node features must be a 2D NumPy array [num_nodes, num_features].")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Node feature dimensions must be non-empty.")
    if not np.isfinite(x).all():
        raise ValueError("Node features contain NaN or infinite values.")


def validate_edge_index(edge_index: np.ndarray, num_nodes: int) -> None:
    if not isinstance(edge_index, np.ndarray) or edge_index.ndim != 2:
        raise ValueError("edge_index must be a 2D NumPy array.")
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_index.size == 0:
        return
    if not np.issubdtype(edge_index.dtype, np.integer):
        raise ValueError("edge_index must contain integer node indices.")
    if edge_index.min() < 0 or edge_index.max() >= num_nodes:
        raise ValueError("edge_index contains an out-of-range node index.")
