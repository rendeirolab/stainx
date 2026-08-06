# Quick Start

## Basic Usage

All normalizers follow a scikit-learn-like interface with `fit()` and `transform()`.
Use **non-negative** float tensors in `[0, 1]` (or `uint8` in `[0, 255]`). Prefer
`torch.rand` over `torch.randn` — Macenko optical-density math rejects negative pixels.

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching, StainNormalizerTransform

reference = torch.rand(1, 3, 512, 512)  # float [0, 1], NCHW
images = torch.rand(10, 3, 512, 512)

normalizer = Reinhard(device="cuda")  # or "cpu", "mps"
normalizer.fit(reference)
normalized = normalizer.transform(images)

# Training-friendly transform (fit once — preferred for supervised training)
# Macenko via the transform defaults normalize_to_0_1=True → float [0, 1] output
transform = StainNormalizerTransform(
    method="macenko",
    mode="reference",
    reference=reference,
    device="cuda",
)
batch_out = transform(images)
```

## Image Formats

| Method | Layout |
|--------|--------|
| Macenko / Reinhard | **NCHW** `(N, 3, H, W)` only |
| HistogramMatching | NCHW (`channel_axis=1`) or NHWC (`channel_axis=-1` / `3`) |

## Device Selection

Bare normalizers and the training transform differ when `device=None`:

```python
from stainx import Reinhard, StainNormalizerTransform

# Bare normalizer: None → CUDA > MPS > CPU
normalizer = Reinhard(device=None)
normalizer = Reinhard(device="cuda")
normalizer = Reinhard(device="mps")
normalizer = Reinhard(device="cpu")

# Transform: None keeps each batch on its input device
transform = StainNormalizerTransform(
    method="reinhard",
    mode="reference",
    reference=torch.rand(1, 3, 64, 64),
    device=None,
)
```

## Backend Selection

Valid backend ids are `"torch"` and `"torch_cuda"` only:

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

See [Training](training.md) for dtype / range rules and checkpointing.
