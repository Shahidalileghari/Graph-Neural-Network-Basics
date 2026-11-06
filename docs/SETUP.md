# Setup and dependency strategy

The package's validation, metrics, negative sampler, and NumPy GCN aggregation use NumPy and PyYAML. Install the optional PyTorch extra for model definitions:

```bash
pip install -e ".[torch]"
```

For PyTorch Geometric examples, install PyTorch for your CPU/CUDA environment first, then follow the official PyG installation matrix for the installed PyTorch version. Finally install:

```bash
pip install -e ".[pyg]"
```

The `pyg` extra declares common Python dependencies, but platform-specific PyTorch wheels may require a separate installation command. Avoid blindly copying CUDA wheel commands between machines.

## CPU-first workflow

- Start with the unit tests and synthetic examples.
- Use Cora for a small, classic transductive node-classification benchmark.
- Keep batch sizes and graph sizes small while learning.
- GPU is optional for the included examples.
