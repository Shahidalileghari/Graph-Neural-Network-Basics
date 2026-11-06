from __future__ import annotations
import numpy as np


def dot_product_scores(embeddings: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings)
    edge_index = np.asarray(edge_index)
    if embeddings.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected embeddings [N, D] and edge_index [2, E].")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(embeddings)):
        raise ValueError("Edge index out of range.")
    return np.sum(embeddings[edge_index[0]] * embeddings[edge_index[1]], axis=1)


def sample_negative_edges(num_nodes: int, positive_edges: np.ndarray, num_samples: int,
                          seed: int = 42, undirected: bool = True,
                          max_attempt_factor: int = 20) -> np.ndarray:
    """Sample non-edges by rejection sampling for small/medium sparse graphs.

    This is a teaching utility, not an optimized sampler for dense or huge graphs.
    Self-pairs are excluded. For undirected graphs, (u,v) and (v,u) are equivalent.
    """
    if num_nodes < 2 or num_samples < 0:
        raise ValueError("num_nodes must be >= 2 and num_samples >= 0.")
    pos = np.asarray(positive_edges)
    if pos.ndim != 2 or pos.shape[0] != 2:
        raise ValueError("positive_edges must have shape [2, E].")
    if pos.size and (pos.min() < 0 or pos.max() >= num_nodes):
        raise ValueError("Positive edge index out of range.")
    def key(u, v):
        return (min(u, v), max(u, v)) if undirected else (u, v)
    existing = {key(int(u), int(v)) for u, v in pos.T if u != v}
    capacity = num_nodes * (num_nodes - 1) // 2 if undirected else num_nodes * (num_nodes - 1)
    if len(existing) + num_samples > capacity:
        raise ValueError("Requested more negative edges than available non-edges.")
    rng = np.random.default_rng(seed)
    result, chosen = [], set()
    attempts = 0
    limit = max(100, num_samples * max_attempt_factor)
    while len(result) < num_samples and attempts < limit:
        u, v = rng.integers(0, num_nodes, size=2)
        attempts += 1
        if u == v:
            continue
        pair = key(int(u), int(v))
        if pair in existing or pair in chosen:
            continue
        chosen.add(pair)
        result.append(pair)
    if len(result) != num_samples:
        remaining = [key(u, v) for u in range(num_nodes) for v in range(num_nodes)
                     if u != v and (not undirected or u < v)]
        remaining = list(dict.fromkeys(p for p in remaining if p not in existing and p not in chosen))
        if len(remaining) < num_samples - len(result):
            raise ValueError("Not enough non-edges available.")
        result.extend(remaining[:num_samples - len(result)])
    return np.asarray(result, dtype=np.int64).T.reshape(2, num_samples)
