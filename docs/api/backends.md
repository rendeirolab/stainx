# Backends

StainX is Torch-only. Choose a backend explicitly or let auto-selection pick one.

## Available backends

- **torch**: PyTorch ops on CPU, CUDA, or MPS
- **torch_cuda**: Compiled CUDA extension (`stainx_cuda_torch`) when the extension
  was built successfully (requires a CUDA-capable GPU visible to PyTorch **and**
  `nvcc` at build time)

## Auto-selection

1. Non-CUDA device → `torch`
2. CUDA device + extension loaded (`CUDA_AVAILABLE`) + `torch.cuda.is_available()` → `torch_cuda`
3. Otherwise → `torch`

## Explicit selection

```python
from stainx import Reinhard

normalizer = Reinhard(device="cpu", backend="torch")
normalizer = Reinhard(device="cuda", backend="torch_cuda")
```

## When to use torch_cuda

Prefer `torch_cuda` when the extension builds successfully: Reinhard, Histogram Matching,
and Macenko all ship real CUDA kernels. Fit still runs on the Torch path (Macenko HE / maxC
on CPU for torchstain parity); CUDA accelerates **transform**.

Macenko defaults to `precision="stable"` (fp64 cov / analytic eigh); use
`precision="fast"` for lower latency when a small MAE trade-off is acceptable
(`"fast"` requires `backend="torch_cuda"`).
