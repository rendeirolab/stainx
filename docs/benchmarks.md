# Performance Benchmarks

StainX provides significant performance improvements, especially when processing batches of images. This page demonstrates how to benchmark performance and compare different backends and configurations.

## Simple Performance Benchmark

Measure the throughput of a single normalization method:

```python
import torch
import time
from stainx import Reinhard

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
height, width = 256, 256

# Generate test images
reference_image = (torch.rand(1, 3, height, width, device=device) * 255).round().to(torch.uint8)
source_images = (torch.rand(batch_size, 3, height, width, device=device) * 255).round().to(torch.uint8)

# Create normalizer
normalizer = Reinhard(device=device)
normalizer.fit(reference_image)

# Benchmark transform
if device == "cuda":
    torch.cuda.synchronize()

start_time = time.time()
normalized = normalizer.transform(source_images)
if device == "cuda":
    torch.cuda.synchronize()
elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

print(f"Processed {batch_size} images in {elapsed_time:.3f} ms")
print(f"Throughput: {batch_size * 1000 / elapsed_time:.2f} images/second")
```

## Comparing Backends

Compare the performance of CUDA and PyTorch backends:

```python
import torch
import time
from stainx import Reinhard

device = "cuda"
batch_size = 64
images = torch.randn(batch_size, 3, 512, 512, device=device)
reference = torch.randn(1, 3, 512, 512, device=device)

# CUDA backend (optimized kernels)
normalizer_cuda = Reinhard(device=device, backend="cuda")
normalizer_cuda.fit(reference)

torch.cuda.synchronize()
start = time.time()
result_cuda = normalizer_cuda.transform(images)
torch.cuda.synchronize()
time_cuda = (time.time() - start) * 1000

# PyTorch backend (fallback)
normalizer_torch = Reinhard(device=device, backend="torch")
normalizer_pytorch.fit(reference)

torch.cuda.synchronize()
start = time.time()
result_pytorch = normalizer_pytorch.transform(images)
torch.cuda.synchronize()
time_pytorch = (time.time() - start) * 1000

speedup = time_pytorch / time_cuda
print(f"CUDA backend: {time_cuda:.3f} ms")
print(f"PyTorch backend: {time_pytorch:.3f} ms")
print(f"Speedup: {speedup:.2f}x")
```

## Batch Size Impact

Analyze how batch size affects throughput:

```python
import torch
import time
from stainx import Macenko

device = "cuda"
reference = torch.randn(1, 3, 512, 512, device=device)
normalizer = Macenko(device=device)
normalizer.fit(reference)

batch_sizes = [1, 8, 16, 32, 64, 128]
results = []

for batch_size in batch_sizes:
    images = torch.randn(batch_size, 3, 512, 512, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    normalized = normalizer.transform(images)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    
    throughput = batch_size * 1000 / elapsed
    results.append((batch_size, elapsed, throughput))
    print(f"Batch size {batch_size:3d}: {elapsed:6.2f} ms ({throughput:6.2f} img/s)")
```

## Comparing All Normalizers

Benchmark all three normalization methods:

```python
import torch
import time
from stainx import Reinhard, Macenko, HistogramMatching

device = "cuda"
batch_size = 32
reference = torch.randn(1, 3, 512, 512, device=device)
images = torch.randn(batch_size, 3, 512, 512, device=device)

normalizers = {
    "Reinhard": Reinhard(device=device),
    "Macenko": Macenko(device=device),
    "HistogramMatching": HistogramMatching(device=device, channel_axis=1)
}

results = {}

for name, normalizer in normalizers.items():
    normalizer.fit(reference)
    
    torch.cuda.synchronize()
    start = time.time()
    normalized = normalizer.transform(images)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    
    throughput = batch_size * 1000 / elapsed
    results[name] = (elapsed, throughput)
    print(f"{name:20s}: {elapsed:6.2f} ms ({throughput:6.2f} img/s)")
```

## Device Comparison

Compare performance across different devices (CPU, CUDA, MPS):

```python
import torch
import time
from stainx import Reinhard

batch_size = 16
reference = torch.randn(1, 3, 256, 256)
images = torch.randn(batch_size, 3, 256, 256)

devices = []
if torch.cuda.is_available():
    devices.append("cuda")
if torch.backends.mps.is_available():
    devices.append("mps")
devices.append("cpu")

results = {}

for device in devices:
    ref_device = reference.to(device)
    img_device = images.to(device)
    
    normalizer = Reinhard(device=device)
    normalizer.fit(ref_device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    
    start = time.time()
    normalized = normalizer.transform(img_device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    
    elapsed = (time.time() - start) * 1000
    throughput = batch_size * 1000 / elapsed
    results[device] = (elapsed, throughput)
    print(f"{device.upper():6s}: {elapsed:6.2f} ms ({throughput:6.2f} img/s)")
```

## Expected Performance

Based on benchmarks run on NVIDIA RTX A6000:

### Backend Speedup (CUDA vs PyTorch)

- **Reinhard**: 5.3-5.4x faster with CUDA backend
  - 256×256 images, batch 32: CUDA 0.72ms vs PyTorch 3.87ms
  - 512×512 images, batch 64: CUDA 5.33ms vs PyTorch 28.40ms
- **Macenko**: 4.6-7.3x faster with CUDA backend
  - 256×256 images, batch 32: CUDA 12.51ms vs PyTorch 57.02ms
  - 512×512 images, batch 64: CUDA 39.19ms vs PyTorch 286.96ms

### Batch Size Impact

Throughput increases significantly with batch size (Reinhard, 256×256 images, CUDA backend):

- Batch 1: ~5,500 images/second
- Batch 8: ~31,000 images/second
- Batch 16: ~38,100 images/second
- Batch 32: ~44,100 images/second
- Batch 64: ~46,600 images/second
- Batch 128: ~46,500 images/second

**Optimal batch size**: 64-128 images provides best throughput for most use cases.

### Method Performance (CUDA backend, batch 32, 256×256)

- **Reinhard**: ~0.75ms (~42,600 images/second)
- **HistogramMatching**: ~8.36ms (~3,800 images/second)
- **Macenko**: ~16.5ms (~1,900 images/second)

### Recommendations

For best performance:
- Use CUDA backend when available (especially for Reinhard and Macenko)
- Process images in batches of 64-128 images
- Use appropriate image sizes for your use case
- Reinhard is fastest, followed by HistogramMatching, then Macenko

