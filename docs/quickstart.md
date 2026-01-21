# Quick Start

This guide will help you get started with StainX in just a few minutes.

## Basic Usage

All normalizers follow a scikit-learn-like interface with `fit()` and `transform()` methods:

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching

# Prepare your images as torch.Tensor
# Shape: (N, C, H, W) where N is batch size, C is channels, H/W are height/width
reference = torch.randn(1, 3, 512, 512)  # Reference image for normalization
images = torch.randn(10, 3, 512, 512)    # Images to normalize

# Reinhard normalization
normalizer = Reinhard(device="cuda")  # or "cpu", "mps"
normalizer.fit(reference)
normalized = normalizer.transform(images)

# Macenko normalization
normalizer = Macenko(device="cuda")
normalizer.fit(reference)
normalized = normalizer.transform(images)

# Histogram Matching
normalizer = HistogramMatching(device="cuda", channel_axis=1)
normalizer.fit(reference)
normalized = normalizer.transform(images)
```

## Image Formats

StainX supports two image formats:

- **Channels-first**: `(N, C, H, W)` - Default format, recommended
- **Channels-last**: `(N, H, W, C)` - Use `channel_axis=-1` for HistogramMatching

```python
# Channels-first (default)
images = torch.randn(10, 3, 512, 512)  # (N, C, H, W)
normalizer = Reinhard(device="cuda")
normalized = normalizer.transform(images)

# Channels-last
images = torch.randn(10, 512, 512, 3)  # (N, H, W, C)
normalizer = HistogramMatching(device="cuda", channel_axis=-1)
normalized = normalizer.transform(images)
```

## Device Selection

StainX automatically selects the best available device:

```python
import torch
from stainx import Reinhard

# Automatic device selection (CUDA > MPS > CPU)
normalizer = Reinhard(device=None)  # Auto-selects best device

# Explicit device selection
normalizer = Reinhard(device="cuda")   # NVIDIA GPU
normalizer = Reinhard(device="mps")    # Apple Silicon
normalizer = Reinhard(device="cpu")    # CPU fallback
```

## Backend Selection

StainX automatically selects the best backend, but you can force a specific one:

```python
# Automatic backend selection
normalizer = Reinhard(device="cuda")

# Force specific backend
normalizer = Reinhard(device="cuda", backend="torch")
normalizer = Reinhard(device="cuda", backend="torch_cuda")
normalizer = Reinhard(device="cuda", backend="cupy")
normalizer = Reinhard(device="cuda", backend="cupy_cuda")
```

## Batch Processing

One of StainX's key features is efficient batch processing:

```python
# Process a single image
single_image = torch.randn(1, 3, 512, 512)
normalized = normalizer.transform(single_image)

# Process a batch of images (much more efficient!)
batch = torch.randn(32, 3, 512, 512)
normalized = normalizer.transform(batch)  # Processes all 32 images at once
```

Batch processing provides significant performance improvements, especially with the CUDA backend. See the [Benchmarks](benchmarks.md) page for performance comparisons.

## Complete Example

Here's a complete example that demonstrates the typical workflow:

```python
import torch
from stainx import Reinhard

# 1. Load or prepare your images
# In practice, you'd load from files, but for this example we'll use random data
reference_image = torch.randn(1, 3, 512, 512)  # Your reference/template image
source_images = torch.randn(20, 3, 512, 512)  # Images to normalize

# 2. Create and configure the normalizer
normalizer = Reinhard(device="cuda")  # Use GPU if available

# 3. Fit the normalizer to the reference image
normalizer.fit(reference_image)

# 4. Transform the source images
normalized_images = normalizer.transform(source_images)

print(f"Normalized {source_images.shape[0]} images")
print(f"Output shape: {normalized_images.shape}")
```

## Next Steps

- Check out [Examples](examples.md) for more detailed usage patterns
- See [Benchmarks](benchmarks.md) for performance tips
- Read the [API Reference](api/index.md) for complete documentation
