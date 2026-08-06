# Examples

This page provides practical examples for common use cases with StainX.

For training pipelines see [Training](training.md). Use float tensors in
`[0, 1]` or `uint8` in `[0, 255]`. Prefer `torch.rand` over `torch.randn`
(Macenko optical-density math does not accept negative pixels).

Repository snippets:

- `examples/torch_transform_example.py` — DataLoader + `StainNormalizerTransform`
- `examples/simple_example.py` — CLI fit/transform demo on real images under `examples/data/`

## Basic Usage

The simplest workflow: fit on a reference image, then transform source images.

```python
import torch
from stainx import Reinhard

reference = torch.rand(1, 3, 512, 512)  # float [0, 1], NCHW
images = torch.rand(10, 3, 512, 512)

normalizer = Reinhard(device="cuda")
normalizer.fit(reference)
normalized = normalizer.transform(images)
```

## All Normalizers

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching

reference = torch.rand(1, 3, 512, 512)
images = torch.rand(10, 3, 512, 512)

reinhard = Reinhard(device="cuda")
reinhard.fit(reference)
normalized_reinhard = reinhard.transform(images)  # float stays ~[0, 1]

# Bare Macenko defaults normalize_to_0_1=False → output ~[0, 255]
macenko = Macenko(device="cuda")
macenko.fit(reference)
normalized_macenko = macenko.transform(images)

# For float [0, 1] training pipelines prefer StainNormalizerTransform
# (defaults normalize_to_0_1=True for Macenko) or Macenko(normalize_to_0_1=True)

histogram = HistogramMatching(device="cuda", channel_axis=1)
histogram.fit(reference)
normalized_histogram = histogram.transform(images)
```

Macenko and Reinhard require **NCHW** (`C=3`). Only HistogramMatching supports NHWC
via `channel_axis=-1` / `3`.

## Fit and Transform in One Step

```python
normalizer = Reinhard(device="cuda")
normalized = normalizer.fit_transform(images)  # Fits and transforms in one call
```

## Batch Processing

```python
import torch
from stainx import Reinhard

small_batch = torch.rand(8, 3, 512, 512)
large_batch = torch.rand(128, 3, 512, 512)

normalizer = Reinhard(device="cuda")
normalizer.fit(torch.rand(1, 3, 512, 512))
normalized = normalizer.transform(large_batch)
print(f"Processed {large_batch.shape[0]} images")
```

## Channels-Last Format (HistogramMatching only)

```python
import torch
from stainx import HistogramMatching

images = torch.rand(10, 512, 512, 3)  # (N, H, W, C)

normalizer = HistogramMatching(device="cuda", channel_axis=-1)
normalizer.fit(images[:1])
normalized = normalizer.transform(images)
```

Passing NHWC into Macenko or Reinhard raises.

## Working with Real Images

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from stainx import Reinhard

reference_img = Image.open("reference.png")
reference_tensor = transforms.ToTensor()(reference_img).unsqueeze(0)  # (1, 3, H, W)

source_images = []
for path in ["img1.png", "img2.png", "img3.png"]:
    img = Image.open(path)
    source_images.append(transforms.ToTensor()(img))

source_batch = torch.stack(source_images)  # (3, 3, H, W)

normalizer = Reinhard(device="cuda")
normalizer.fit(reference_tensor)
normalized_batch = normalizer.transform(source_batch)

for i, normalized in enumerate(normalized_batch):
    transforms.ToPILImage()(normalized).save(f"normalized_{i}.png")
```

Or run `python examples/simple_example.py reinhard` with images under `examples/data/`.

## Device Selection

Bare normalizers: `device=None` auto-picks CUDA > MPS > CPU.
`StainNormalizerTransform(device=None)` keeps each batch on its **input** device.

```python
import torch
from stainx import Reinhard

reference = torch.rand(1, 3, 512, 512)
images = torch.rand(10, 3, 512, 512)

normalizer_cpu = Reinhard(device="cpu")
normalizer_cpu.fit(reference)
normalized_cpu = normalizer_cpu.transform(images)

if torch.cuda.is_available():
    normalizer_cuda = Reinhard(device="cuda")
    normalizer_cuda.fit(reference.to("cuda"))
    normalized_cuda = normalizer_cuda.transform(images.to("cuda"))

if torch.backends.mps.is_available():
    normalizer_mps = Reinhard(device="mps")
    normalizer_mps.fit(reference.to("mps"))
    normalized_mps = normalizer_mps.transform(images.to("mps"))
```

## Backend Selection

Valid ids: `"torch"` and `"torch_cuda"` only.

```python
from stainx import Reinhard

reference = torch.rand(1, 3, 512, 512, device="cuda")
images = torch.rand(10, 3, 512, 512, device="cuda")

normalizer_torch_cuda = Reinhard(device="cuda", backend="torch_cuda")
normalizer_torch_cuda.fit(reference)
normalized_torch_cuda = normalizer_torch_cuda.transform(images)

normalizer_torch = Reinhard(device="cuda", backend="torch")
normalizer_torch.fit(reference)
normalized_torch = normalizer_torch.transform(images)
```

## Processing Different Image Sizes

Images in one batch must share `H×W`. Process different sizes separately (or resize first).

```python
import torch
from stainx import Reinhard

reference = torch.rand(1, 3, 512, 512)
small_images = torch.rand(5, 3, 256, 256)
large_images = torch.rand(5, 3, 1024, 1024)

normalizer = Reinhard(device="cuda")
normalizer.fit(reference)
normalized_small = normalizer.transform(small_images)
normalized_large = normalizer.transform(large_images)
```

## Data types and value range

Reinhard and HistogramMatching preserve input dtype (uint8 stays uint8; float stays float in `[0, 1]`).

Bare `Macenko` defaults `normalize_to_0_1=False`, so float/`uint8` outputs are ~`[0, 255]`.
`StainNormalizerTransform(method="macenko")` defaults `normalize_to_0_1=True` (float `[0, 1]`).

```python
import torch
from stainx import Reinhard, Macenko

reference_uint8 = (torch.rand(1, 3, 512, 512) * 255).round().to(torch.uint8)
images_uint8 = (torch.rand(10, 3, 512, 512) * 255).round().to(torch.uint8)

reinhard = Reinhard(device="cuda")
reinhard.fit(reference_uint8)
out_r = reinhard.transform(images_uint8)
print(out_r.dtype)  # torch.uint8

macenko = Macenko(device="cuda")  # normalize_to_0_1=False
macenko.fit(reference_uint8)
out_m = macenko.transform(images_uint8)
print(out_m.dtype, float(out_m.amax()))  # uint8-ish / ~255 range

macenko_01 = Macenko(device="cuda", normalize_to_0_1=True)
macenko_01.fit(reference_uint8)
out_01 = macenko_01.transform(images_uint8)
print(out_01.dtype, float(out_01.amax()))  # float, <= 1
```
