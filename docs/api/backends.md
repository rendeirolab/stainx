# Backends

## Available Backends

- **PyTorch**: Default fallback, works on CPU and GPU
- **CUDA**: Optimized kernels, requires CUDA Toolkit

## Selection

Automatically selects CUDA if available and device is CUDA, else PyTorch.

```python
# Automatic
normalizer = Reinhard(device="cuda")

# Force backend
normalizer = Reinhard(device="cuda", backend="pytorch")
```

## Performance

CUDA backend provides 2-10x speedup for batch processing.

## Check Availability

```python
from stainx.backends.cuda_backend import CUDA_AVAILABLE
print(f"CUDA backend: {CUDA_AVAILABLE}")
```
