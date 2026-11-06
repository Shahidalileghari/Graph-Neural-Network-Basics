# Architecture

## Data flow

```text
Raw graph / dataset
      |
      v
Validation -----> split protocol (node masks or edge split)
      |                         |
      v                         v
Node features + edges ------> GNN encoder
                                  |
                         node embeddings
                           /         \
                 classifier         decoder
                     |                 |
              node logits       edge scores
                     |                 |
              classification      link ranking
                     \                 /
                      evaluation + experiment record
```

## Main abstractions

- **Graph representation:** `x` stores node features; `edge_index` stores source/destination index pairs; labels and masks define supervision.
- **Layer:** `gcn_aggregate` makes normalized neighborhood aggregation explicit and inspectable.
- **Encoder:** optional PyTorch GCN / MLP baselines.
- **Task utilities:** dot-product edge decoder and negative-edge sampling.
- **Metrics:** classification and ranking metrics are intentionally small, transparent implementations.
- **Experiment config:** YAML records dataset protocol, model hyperparameters, seed, and training settings.

## Complexity notes

The NumPy teaching aggregation materializes a dense `N × N` adjacency matrix and is therefore suitable only for small graphs. Sparse GNN libraries such as PyG are preferable for realistic graphs. The sample PyG example uses sparse edge-index operations.
