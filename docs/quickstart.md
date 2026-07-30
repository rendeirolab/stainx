# Quick Start

## Basic Usage

All normalizers follow a scikit-learn-like interface with `fit()` and `transform()`:

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching, StainNormalizerTransform

reference = torch.randn(1, 3, 512, 512)
images = torch.randn(10, 3, 512, 512)

normalizer = Reinhard(device="cuda")  # or "cpu", "mps"
normalizer.fit(reference)
normalized = normalizer.transform(images)

# Training-friendly transform (fit once — preferred for supervised training)
transform = StainNormalizerTransform(method="macenko", mode="reference", reference=reference, device="cuda")
batch_out = transform(images)
```

## Image Formats

- **Channels-first**: `(N, C, H, W)` — default, recommended
- **Channels-last**: `(N, H, W, C)` — use `channel_axis=-1` for HistogramMatching

## Device Selection

```python
normalizer = Reinhard(device=None)     # CUDA > MPS > CPU
normalizer = Reinhard(device="cuda")
normalizer = Reinhard(device="mps")
normalizer = Reinhard(device="cpu")
```

## Backend Selection

```python
normalizer = Reinhard(device="cuda")  # auto: torch_cuda if built, else torch
normalizer = Reinhard(device="cuda", backend="torch")
normalizer = Reinhard(device="cuda", backend="torch_cuda")
```

## Transform modes

| Mode | When to use |
|------|-------------|
| `reference` | Fit once on a fixed reference — default for training |
| `batch` | Fit every forward — exploratory / domain-shift checks; usually unsafe for reproducible supervised training |

See [Training](training.md).
