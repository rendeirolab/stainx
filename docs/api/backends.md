# Backends

StainX is Torch-only. Choose a backend explicitly or let auto-selection pick one.

## Available backends

- **torch**: PyTorch ops on CPU, CUDA, or MPS
- **torch_cuda**: Compiled CUDA extension (`stainx_cuda_torch`) when built with nvcc

## Auto-selection

1. Non-CUDA device → `torch`
2. CUDA device + extension available → `torch_cuda`
3. Otherwise → `torch`

## Explicit selection

```python
from stainx import Reinhard

normalizer = Reinhard(device="cpu", backend="torch")
normalizer = Reinhard(device="cuda", backend="torch_cuda")
```

## When to use torch_cuda

Prefer `torch_cuda` for Reinhard and Histogram Matching when the extension builds successfully (real CUDA kernels). Macenko's CUDA path uses an ATen parity implementation and is typically close to Torch speed (~1×).
