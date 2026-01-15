# Backends

StainX supports multiple backends for different performance characteristics and device compatibility.

## Available Backends

- **PyTorch**: Default fallback backend, works on CPU, CUDA, and MPS devices
- **CUDA**: Optimized CUDA kernels, requires CUDA Toolkit and CUDA device

## Backend Selection

Backends are automatically selected based on device availability:

1. If device is CUDA and CUDA backend is available → use CUDA backend
2. Otherwise → use PyTorch backend

```python
from stainx import Reinhard

# Automatic selection (CUDA backend if available, else PyTorch)
normalizer = Reinhard(device="cuda")

# Force PyTorch backend (works on CPU, CUDA, and MPS)
normalizer = Reinhard(device="cuda", backend="torch")

# Force CUDA backend (only works on CUDA devices)
normalizer = Reinhard(device="cuda", backend="cuda")
```

## Device Support

### PyTorch Backend
- ✅ CPU
- ✅ CUDA (NVIDIA GPUs)
- ✅ MPS (Apple Silicon)

### CUDA Backend
- ❌ CPU (not supported)
- ✅ CUDA (NVIDIA GPUs only)
- ❌ MPS (not supported)

## Performance

The CUDA backend provides significant performance improvements for batch processing:

- **Reinhard**: 5.3-5.4x speedup compared to PyTorch backend
- **Macenko**: 4.6-7.3x speedup compared to PyTorch backend
- **HistogramMatching**: PyTorch backend is typically faster for this method
- Best performance with larger batch sizes (64-128 images)
- Optimized memory access patterns

**Example performance** (NVIDIA RTX A6000, batch 32, 256×256):
- Reinhard: CUDA 0.72ms vs PyTorch 3.87ms (5.4x speedup)
- Macenko: CUDA 12.51ms vs PyTorch 57.02ms (4.6x speedup)

See the [Benchmarks](../benchmarks.md) page for detailed performance comparisons.

## Check Backend Availability

```python
from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

if CUDA_AVAILABLE:
    print("CUDA backend is available!")
else:
    print("CUDA backend is not available. Using PyTorch backend.")
```

## When to Use Which Backend

**Use CUDA backend when:**
- You have a CUDA-capable GPU
- Processing large batches of images
- Maximum performance is required

**Use PyTorch backend when:**
- Running on CPU or MPS
- CUDA backend is not available
- You need maximum compatibility
- Working with small batches where overhead matters less
