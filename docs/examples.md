# Examples

This page provides practical examples for common use cases with StainX.

## Basic Usage

The simplest workflow: fit on a reference image, then transform source images.

```python
import torch
from stainx import Reinhard

# Prepare images
reference = torch.randn(1, 3, 512, 512)  # Reference/template image
images = torch.randn(10, 3, 512, 512)    # Images to normalize

# Create normalizer and fit
normalizer = Reinhard(device="cuda")
normalizer.fit(reference)

# Transform images
normalized = normalizer.transform(images)
```

## All Normalizers

StainX provides three normalization algorithms:

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching

reference = torch.randn(1, 3, 512, 512)
images = torch.randn(10, 3, 512, 512)

# Reinhard normalization
reinhard = Reinhard(device="cuda")
reinhard.fit(reference)
normalized_reinhard = reinhard.transform(images)

# Macenko normalization
macenko = Macenko(device="cuda")
macenko.fit(reference)
normalized_macenko = macenko.transform(images)

# Histogram Matching
histogram = HistogramMatching(device="cuda", channel_axis=1)
histogram.fit(reference)
normalized_histogram = histogram.transform(images)
```

## Fit and Transform in One Step

Use `fit_transform()` for convenience:

```python
normalizer = Reinhard(device="cuda")
normalized = normalizer.fit_transform(images)  # Fits and transforms in one call
```

## Batch Processing

Process multiple images efficiently in a single batch:

```python
import torch
from stainx import Reinhard

# Small batch
small_batch = torch.randn(8, 3, 512, 512)

# Large batch (more efficient)
large_batch = torch.randn(128, 3, 512, 512)

normalizer = Reinhard(device="cuda")
normalizer.fit(torch.randn(1, 3, 512, 512))

# Process entire batch at once
normalized = normalizer.transform(large_batch)
print(f"Processed {large_batch.shape[0]} images")
```

## Channels-Last Format

Support for channels-last format (useful when working with certain image loaders):

```python
import torch
from stainx import HistogramMatching

# Images in (N, H, W, C) format
images = torch.randn(10, 512, 512, 3)

# Use channel_axis=-1 for channels-last
normalizer = HistogramMatching(device="cuda", channel_axis=-1)
normalizer.fit(images[:1])  # Fit on first image
normalized = normalizer.transform(images)
```

## Working with Real Images

Example of loading and processing real images:

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from stainx import Reinhard

# Load reference image
reference_img = Image.open("reference.png")
reference_tensor = transforms.ToTensor()(reference_img).unsqueeze(0)  # (1, 3, H, W)

# Load source images
source_images = []
for path in ["img1.png", "img2.png", "img3.png"]:
    img = Image.open(path)
    tensor = transforms.ToTensor()(img)
    source_images.append(tensor)

# Stack into batch
source_batch = torch.stack(source_images)  # (3, 3, H, W)

# Normalize
normalizer = Reinhard(device="cuda")
normalizer.fit(reference_tensor)
normalized_batch = normalizer.transform(source_batch)

# Convert back to images if needed
for i, normalized in enumerate(normalized_batch):
    img = transforms.ToPILImage()(normalized)
    img.save(f"normalized_{i}.png")
```

## Device Selection

Examples for different devices:

```python
import torch
from stainx import Reinhard

reference = torch.randn(1, 3, 512, 512)
images = torch.randn(10, 3, 512, 512)

# CPU
normalizer_cpu = Reinhard(device="cpu")
normalizer_cpu.fit(reference)
normalized_cpu = normalizer_cpu.transform(images)

# CUDA (NVIDIA GPU)
if torch.cuda.is_available():
    normalizer_cuda = Reinhard(device="cuda")
    normalizer_cuda.fit(reference.to("cuda"))
    normalized_cuda = normalizer_cuda.transform(images.to("cuda"))

# MPS (Apple Silicon)
if torch.backends.mps.is_available():
    normalizer_mps = Reinhard(device="mps")
    normalizer_mps.fit(reference.to("mps"))
    normalized_mps = normalizer_mps.transform(images.to("mps"))
```

## Backend Selection

Force a specific backend:

```python
from stainx import Reinhard

reference = torch.randn(1, 3, 512, 512, device="cuda")
images = torch.randn(10, 3, 512, 512, device="cuda")

# Use optimized CUDA kernels (if available)
normalizer_cuda = Reinhard(device="cuda", backend="cuda")
normalizer_cuda.fit(reference)
normalized_cuda = normalizer_cuda.transform(images)

# Force PyTorch backend (works everywhere)
normalizer_pytorch = Reinhard(device="cuda", backend="pytorch")
normalizer_pytorch.fit(reference)
normalized_pytorch = normalizer_pytorch.transform(images)
```

## Processing Different Image Sizes

Handle images of different sizes in a batch:

```python
import torch
from stainx import Reinhard

# Images must be the same size in a batch
# If you have different sizes, process them separately or resize first

reference = torch.randn(1, 3, 512, 512)
small_images = torch.randn(5, 3, 256, 256)
large_images = torch.randn(5, 3, 1024, 1024)

normalizer = Reinhard(device="cuda")

# Process small images
normalizer.fit(reference)
normalized_small = normalizer.transform(small_images)

# Process large images
normalized_large = normalizer.transform(large_images)
```

## Preserving Data Types

StainX preserves input data types:

```python
import torch
from stainx import Reinhard

# uint8 input
reference_uint8 = (torch.rand(1, 3, 512, 512) * 255).round().to(torch.uint8)
images_uint8 = (torch.rand(10, 3, 512, 512) * 255).round().to(torch.uint8)

normalizer = Reinhard(device="cuda")
normalizer.fit(reference_uint8)
normalized = normalizer.transform(images_uint8)

print(f"Input dtype: {images_uint8.dtype}")
print(f"Output dtype: {normalized.dtype}")  # Should match input
```
