<div align="center">

<h1>StainX</h1>
<img src="https://raw.githubusercontent.com/rendeirolab/stainx/refs/heads/main/assets/staix-logo-256.png"/>

![CI](https://github.com/rendeirolab/stainx/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
</div>


Torch-first stain normalization for histopathology images with batch processing, training transforms, and optional CUDA kernels.

## Features

- **Multiple algorithms**: Histogram Matching, Reinhard, and Macenko normalization
- **Torch backends**: `torch` (CPU / CUDA / MPS) and optional `torch_cuda` compiled kernels
- **Training-ready**: `StainNormalizerTransform` for DataLoader / torchvision pipelines

## Installation

### Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- CUDA Toolkit + nvcc (optional; builds `stainx_cuda_torch` when available)

**Supported platforms**

| Platform | Support |
|----------|---------|
| Linux + CUDA | Primary (Torch + optional CUDA extension) |
| Linux CPU | Primary (Torch backend) |
| macOS (MPS / CPU) | Best-effort Torch path (no CUDA extension) |
| Windows | Best-effort Torch path (CUDA extension not guaranteed) |

### Install from PyPI

```bash
pip install stainx
```

### Install from source (recommended: Makefile)

```bash
git clone https://github.com/rendeirolab/stainx.git
cd stainx
make install          # editable + best-effort CUDA build
# or
make install-dev      # + test/docs tooling
```

Plain pip also works:

```bash
pip install .
# CUDA extension builds automatically when CUDA/nvcc are present; otherwise Torch-only.
```

## Quick Start

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching, StainNormalizerTransform

reference_image = torch.randn(1, 3, 512, 512)
source_images = torch.randn(10, 3, 512, 512)

normalizer = Reinhard(device="cuda")  # or "cpu" / "mps"
normalizer.fit(reference_image)
normalized = normalizer.transform(source_images)

# Training transform (fit once on a reference — preferred for supervised training)
transform = StainNormalizerTransform(
    method="macenko",
    mode="reference",
    reference=reference_image,
    device="cuda",
)
batch_out = transform(source_images)
```

### Modes

| Mode | Behavior | When to use |
|------|----------|-------------|
| `reference` | Fit once on a fixed reference, then transform | Default for training |
| `batch` | Fit on the current batch every forward | Exploratory / domain-shift checks; usually unsafe for reproducible supervised training |

## API

- `fit(reference_images)` / `transform(images)` / `fit_transform(images)`
- `StainNormalizerTransform` — `nn.Module` for pipelines
- Backends: `"torch"` (default) or `"torch_cuda"` when the extension is built

## Documentation

See the [documentation site](https://stainx.readthedocs.io/) for installation details, training usage, and examples.

## License

GPL-3.0-or-later
