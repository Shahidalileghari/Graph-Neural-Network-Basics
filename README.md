# Graph Neural Networks Engineering

A practical, engineering-focused repository for learning and building Graph Neural Network (GNN) systems: graph data modeling, message passing, node/graph prediction, link prediction, evaluation, experiment configuration, and reproducible workflows.

> **Scope:** an educational engineering lab and reusable baseline toolkit—not a production graph platform. The core examples use PyTorch and PyTorch Geometric (PyG). A small pure-PyTorch message-passing implementation is included to make the mechanics inspectable.

## What you'll build

- Understand graph tensors: node features `X`, edge index `edge_index`, labels, masks, and graph-level batches.
- Implement a Graph Convolutional Network (GCN) layer from first principles.
- Train a node-classification baseline on a citation-style dataset (Cora through PyG).
- Compare a simple MLP baseline with a GCN.
- Build a link-prediction workflow with negative sampling and ranking metrics.
- Evaluate models with accuracy, macro-F1, ROC-AUC, average precision, and Hits@K.
- Avoid common graph ML mistakes: leakage across edges, incorrect splits, transductive/inductive confusion, and misleading random negatives.
- Run tests and CI; configure experiments with YAML.

## Repository map

```text
src/gnn_engineering/
  data/          # graph validation and split helpers
  layers/        # transparent GCN message passing
  models/        # GCN and MLP baselines
  tasks/         # node classification and link prediction utilities
  metrics/       # classification and ranking metrics
  training/      # reproducibility and training helpers
  cli.py         # command-line entry point
configs/         # experiment configurations
examples/        # runnable PyG examples
tests/           # unit tests for dependency-light core
docs/            # architecture, evaluation, pitfalls, roadmap
```

## Quick start

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

For PyTorch and PyG, install versions compatible with your platform and CUDA setup. Start with the official PyTorch selector and PyG installation instructions; see [docs/SETUP.md](docs/SETUP.md). The pure-PyTorch layer and its tests do not require PyG.

## Run the core demo

```bash
gnn-lab inspect-config --config configs/cora_gcn.yaml
```

Optional PyG example (after installing compatible PyTorch/PyG dependencies):

```bash
python examples/cora_node_classification.py
```

Optional link-prediction example:

```bash
python examples/synthetic_link_prediction.py
```

## Core concepts

### Message passing

A GNN updates each node representation by aggregating information from its neighbors:

\[
h_v^{(k+1)} = \sigma\left(W^{(k)} \cdot \operatorname{AGG}
\{h_u^{(k)} : u \in \mathcal{N}(v)\}\right)
\]

The included GCN layer uses symmetric degree normalization with self-loops:

\[
H' = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}HW
\]

where \(\hat{A}=A+I\). `edge_index` follows PyG's `[2, num_edges]` convention.

### Learning tasks

| Task | Input | Prediction |
|---|---|---|
| Node classification | One graph + node features | Class per node |
| Graph classification | A collection of graphs | Class per graph |
| Link prediction | Node pairs | Edge existence / score |
| Node regression | Graph + node features | Numeric value per node |

This repository provides working foundations for node classification and link prediction. Graph classification is documented as an extension exercise.

## Engineering principles

1. Split data according to the deployment question; do not assume random edge splits are always valid.
2. Keep validation/test labels out of model selection and early stopping.
3. For link prediction, ensure positive edges are excluded from message-passing adjacency when the protocol requires it.
4. Report class balance and compare against simple baselines.
5. Fix random seeds, record package versions, and preserve configuration with each run.
6. Treat benchmark scores as protocol-dependent—not universal model quality.

## Testing status

The dependency-light core is designed for unit testing with pytest. Optional PyG examples require separately installed compatible dependencies and dataset downloads. Do not interpret a successful import or config inspection as proof that training has been executed.
