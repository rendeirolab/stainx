# Backends

StainX supports multiple backends for different performance characteristics and device compatibility.

## Available Backends

- **torch**: PyTorch backend, works on CPU, CUDA, and MPS devices
- **torch_cuda**: Optimized PyTorch CUDA extension, requires CUDA Toolkit and CUDA device
- **cupy**: CuPy backend, requires CUDA and CuPy
- **cupy_cuda**: CuPy CUDA backend, requires CUDA and CuPy

## Backend Selection

Backends are automatically selected based on device availability:

1. For non-CUDA devices → use `torch` backend
2. For CUDA devices → try `torch_cuda`, then `cupy_cuda`, then `cupy`, fallback to `torch`

```python
from stainx import Reinhard

# Automatic selection
normalizer = Reinhard(device="cuda")

# Force specific backend
normalizer = Reinhard(device="cuda", backend="torch")
normalizer = Reinhard(device="cuda", backend="torch_cuda")
normalizer = Reinhard(device="cuda", backend="cupy")
normalizer = Reinhard(device="cuda", backend="cupy_cuda")
```

## Device Support

### torch Backend
- ✅ CPU
- ✅ CUDA (NVIDIA GPUs)
- ✅ MPS (Apple Silicon)

### torch_cuda Backend
- ❌ CPU (not supported)
- ✅ CUDA (NVIDIA GPUs only)
- ❌ MPS (not supported)

### cupy Backend
- ❌ CPU (not supported)
- ✅ CUDA (NVIDIA GPUs only)
- ❌ MPS (not supported)

### cupy_cuda Backend
- ❌ CPU (not supported)
- ✅ CUDA (NVIDIA GPUs only)
- ❌ MPS (not supported)

## Performance

CUDA backends help most where custom kernels remain:

- **Reinhard**: ~5.6–5.8× speedup with `torch_cuda` vs torch (custom kernels)
- **Macenko**: ~1.0–1.1× with `torch_cuda` vs torch (ATen parity path for torchstain match)
- **HistogramMatching**: torch backend is typically faster for this method
- Best performance with larger batch sizes (64-128 images)

**Example performance** (NVIDIA RTX A6000, batch 32, 256×256):
- Reinhard: ~42,300 vs ~7,400 img/s (~5.7×)
- Macenko: ~347 vs ~320 img/s (~1.09×)

See the [Benchmarks](../benchmarks.md) page for detailed performance comparisons.

## Check Backend Availability

```python
from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE as TORCH_CUDA_AVAILABLE
from stainx.backends.cupy_cuda_backend import CUDA_AVAILABLE as CUPY_CUDA_AVAILABLE

if TORCH_CUDA_AVAILABLE:
    print("torch_cuda backend is available!")
if CUPY_CUDA_AVAILABLE:
    print("cupy_cuda backend is available!")
```

## When to Use Which Backend

**Use torch_cuda or cupy_cuda when:**
- You have a CUDA-capable GPU
- Processing large batches of images
- Maximum performance is required

**Use torch backend when:**
- Running on CPU or MPS
- CUDA backends are not available
- You need maximum compatibility
- Working with small batches where overhead matters less
