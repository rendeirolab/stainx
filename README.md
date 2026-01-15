<div align="center">

<h1>StainX</h1>
<img src="https://raw.githubusercontent.com/rendeirolab/stainx/refs/heads/main/assets/staix-logo-256.png"/>

![CI](https://github.com/rendeirolab/stainx/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
</div>


Enhanced stain normalization for histopathology images with batch processing support. Optimized for CPU, GPU (CUDA), and MPS (Apple Silicon) devices.

## Features

- **Multiple algorithms**: Histogram Matching, Reinhard, and Macenko normalization
- **Automatic backend selection**: PyTorch (CPU/GPU/MPS) or optimized CUDA kernels
- **Batch processing**: Enhanced normalization through efficient batch processing of multiple images
- **Flexible device support**: CPU, CUDA, MPS (Apple Silicon)

## Installation

### Requirements

- Python >=3.11
- PyTorch >=2.0.0
- CUDA (optional, for GPU acceleration)

### Install from PyPI

```bash
pip install stainx
```

### Install from source

```bash
git clone https://github.com/rendeirolab/stainx.git
cd stainx
pip install .
```

CUDA extensions will be automatically built if CUDA is available.

## Quick Start

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching

# Load your images as torch.Tensor (N, C, H, W) or (N, H, W, C)
reference_image = torch.randn(1, 3, 512, 512)  # Reference image
source_images = torch.randn(10, 3, 512, 512)  # Images to normalize

# Reinhard normalization
normalizer = Reinhard(device="cuda")  # or "cpu"
normalizer.fit(reference_image)
normalized = normalizer.transform(source_images)

# Macenko normalization
normalizer = Macenko(device="cuda")
normalizer.fit(reference_image)
normalized = normalizer.transform(source_images)

# Histogram Matching
normalizer = HistogramMatching(device="cuda", channel_axis=1)
normalizer.fit(reference_image)
normalized = normalizer.transform(source_images)
```

## API

All normalizers follow a scikit-learn-like interface:

- `fit(reference_images)`: Compute normalization parameters from reference image(s)
- `transform(images)`: Apply normalization to images
- `fit_transform(images)`: Fit and transform in one step

### Available Normalizers

- `Reinhard`: Reinhard color normalization
- `Macenko`: Macenko stain separation and normalization
- `HistogramMatching`: Histogram matching normalization

### Backend Selection

Backends are automatically selected based on device availability:
- **CUDA**: Used when CUDA is available and device is set to CUDA
- **PyTorch**: Fallback backend, works on CPU and GPU

You can explicitly specify a backend:

```python
normalizer = Reinhard(device="cuda", backend="torch")  # Force Torch backend
```

## Requirements

- Python >=3.11
- PyTorch >=2.0.0
- CUDA Toolkit (optional, for CUDA backend)

## License

This project is licensed under the GNU General Public License v3 (GPL-3.0-or-later).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to StainX.

## Links

- **GitHub**: https://github.com/rendeirolab/stainx
- **Issues**: https://github.com/rendeirolab/stainx/issues

