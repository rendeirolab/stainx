# Performance Benchmarks

StainX is designed for batch throughput. This page shows how to measure performance
and points at the repository harnesses that produce comparable numbers.

## Repository harnesses (preferred)

| Script | What it measures |
|--------|------------------|
| `benchmarks/benchmark_stainx_backend.py` | `torch_cuda` vs `torch` grid (batch × size); writes `benchmarks/logs/stainx_backend_benchmark_*.log` |
| `benchmarks/pareto_time_mae.py` | Cross-package MAE vs throughput (needs `uv sync --group benchmark --python 3.11`) |
| `benchmarks/run_stainx.py` | Quick single-method microbench |

Example:

```bash
# editable install with CUDA extension when gates are met
make install-dev

python benchmarks/benchmark_stainx_backend.py --method reinhard --batch-size 32 --image-size 256
```

Logs under `benchmarks/logs/` are gitignored — regenerate on your hardware before
citing absolute img/s figures.

## Simple Performance Benchmark

Illustrative single-shot timing (no warmup). Prefer the harness above for published numbers.

```python
import torch
import time
from stainx import Reinhard

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
height, width = 256, 256

reference_image = (torch.rand(1, 3, height, width, device=device) * 255).round().to(torch.uint8)
source_images = (torch.rand(batch_size, 3, height, width, device=device) * 255).round().to(torch.uint8)

normalizer = Reinhard(device=device)
normalizer.fit(reference_image)

if device == "cuda":
    torch.cuda.synchronize()

start_time = time.time()
normalized = normalizer.transform(source_images)
if device == "cuda":
    torch.cuda.synchronize()
elapsed_time = (time.time() - start_time) * 1000

print(f"Processed {batch_size} images in {elapsed_time:.3f} ms")
print(f"Throughput: {batch_size * 1000 / elapsed_time:.2f} images/second")
```

## Comparing Backends

```python
import torch
import time
from stainx import Reinhard

device = "cuda"
batch_size = 64
# uint8 / [0, 255] — avoid torch.randn (negative values break Macenko OD math)
images = (torch.rand(batch_size, 3, 512, 512, device=device) * 255).round().to(torch.uint8)
reference = (torch.rand(1, 3, 512, 512, device=device) * 255).round().to(torch.uint8)

normalizer_torch_cuda = Reinhard(device=device, backend="torch_cuda")
normalizer_torch_cuda.fit(reference)

torch.cuda.synchronize()
start = time.time()
result_torch_cuda = normalizer_torch_cuda.transform(images)
torch.cuda.synchronize()
time_torch_cuda = (time.time() - start) * 1000

normalizer_torch = Reinhard(device=device, backend="torch")
normalizer_torch.fit(reference)

torch.cuda.synchronize()
start = time.time()
result_torch = normalizer_torch.transform(images)
torch.cuda.synchronize()
time_torch = (time.time() - start) * 1000

speedup = time_torch / time_torch_cuda
print(f"torch_cuda backend: {time_torch_cuda:.3f} ms")
print(f"torch backend: {time_torch:.3f} ms")
print(f"Speedup: {speedup:.2f}x")
```

## Batch Size Impact

```python
import torch
import time
from stainx import Macenko

device = "cuda"
reference = (torch.rand(1, 3, 512, 512, device=device) * 255).round().to(torch.uint8)
normalizer = Macenko(device=device)
normalizer.fit(reference)

batch_sizes = [1, 8, 16, 32, 64, 128]
results = []

for batch_size in batch_sizes:
    images = (torch.rand(batch_size, 3, 512, 512, device=device) * 255).round().to(torch.uint8)

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

```python
import torch
import time
from stainx import Reinhard, Macenko, HistogramMatching

device = "cuda"
batch_size = 32
reference = (torch.rand(1, 3, 512, 512, device=device) * 255).round().to(torch.uint8)
images = (torch.rand(batch_size, 3, 512, 512, device=device) * 255).round().to(torch.uint8)

normalizers = {
    "Reinhard": Reinhard(device=device),
    "Macenko": Macenko(device=device),
    "HistogramMatching": HistogramMatching(device=device, channel_axis=1),
}

for name, normalizer in normalizers.items():
    normalizer.fit(reference)

    torch.cuda.synchronize()
    start = time.time()
    normalized = normalizer.transform(images)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000

    throughput = batch_size * 1000 / elapsed
    print(f"{name:20s}: {elapsed:6.2f} ms ({throughput:6.2f} img/s)")
```

## Device Comparison

```python
import torch
import time
from stainx import Reinhard

batch_size = 16
reference = (torch.rand(1, 3, 256, 256) * 255).round().to(torch.uint8)
images = (torch.rand(batch_size, 3, 256, 256) * 255).round().to(torch.uint8)

devices = []
if torch.cuda.is_available():
    devices.append("cuda")
if torch.backends.mps.is_available():
    devices.append("mps")
devices.append("cpu")

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
    print(f"{device.upper():6s}: {elapsed:6.2f} ms ({throughput:6.2f} img/s)")
```

## Historical numbers (RTX A6000)

The figures below were measured on an NVIDIA RTX A6000 during the 0.1.x torch_cuda
work. They are **not** checked into `benchmarks/logs/` (that directory is gitignored).
Re-run `benchmark_stainx_backend.py` before treating them as current.

### Backend Speedup (torch_cuda vs torch)

- **Reinhard**: ~5.6–5.8× faster with torch_cuda
  - 256×256, batch 32: ~42,300 vs ~7,400 img/s (~5.7×)
  - 512×512, batch 64: ~11,400 vs ~2,000 img/s (~5.8×)
- **Macenko**: ~5–9× vs the previous ATen/CPU-offload path (not vs `backend="torch"` on the same night)
  - Example: 555 → 5177 img/s at 64×150²; 86 → 476 img/s at 32×512²
  - Default `precision="stable"`; `precision="fast"` trades some MAE for latency

### Batch Size Impact (Reinhard, 256×256, CUDA)

- Batch 1: ~5,500 img/s → Batch 64–128: ~46,500–46,600 img/s

### Method Performance (torch_cuda, batch 32, 256×256) — historical

- **Reinhard**: ~0.76 ms (~42,300 img/s)
- **HistogramMatching**: ~8.36 ms (~3,800 img/s)
- **Macenko**: depends on tile size / batch / `precision`

### Recommendations

- Use `torch_cuda` for all three methods when the extension builds
- Prefer Macenko `precision="stable"` for torchstain parity; `"fast"` when latency matters more
- Process batches of 64–128 when memory allows
- Reinhard is typically fastest among the three for equal tile size
