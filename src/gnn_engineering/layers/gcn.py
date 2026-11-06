from __future__ import annotations
import numpy as np


def gcn_aggregate(x: np.ndarray, edge_index: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """Apply symmetric-normalized neighborhood aggregation (without learnable weights).

    Args:
        x: Node features with shape [N, F].
        edge_index: Directed source/destination pairs with shape [2, E].
        add_self_loops: Add one self-loop per node before normalization.

    Returns:
        Aggregated node features with shape [N, F].

    Notes:
        For an undirected graph, provide both (u, v) and (v, u). Duplicate edges
        are treated as repeated adjacency entries; callers should deduplicate if
        their graph semantics require a simple graph.
    """
    x = np.asarray(x, dtype=np.float64)
    edge_index = np.asarray(edge_index)
    if x.ndim != 2:
        raise ValueError("x must have shape [num_nodes, num_features].")
    n = x.shape[0]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_index.size and (not np.issubdtype(edge_index.dtype, np.integer)
                            or edge_index.min() < 0 or edge_index.max() >= n):
        raise ValueError("edge_index contains invalid node indices.")

    adjacency = np.zeros((n, n), dtype=np.float64)
    if edge_index.size:
        src, dst = edge_index
        adjacency[dst, src] += 1.0
    if add_self_loops:
        adjacency += np.eye(n, dtype=np.float64)

    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = degree[nonzero] ** -0.5
    normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return normalized @ x
