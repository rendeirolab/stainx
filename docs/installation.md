# Installation

## Requirements

- Python >=3.11
- PyTorch >=2.0.0
- CuPy >=12.0.0 (cupy-cuda12x>=12.0.0 for non-ARM64, cupy>=12.0.0 for ARM64)
- CUDA Toolkit (optional, for CUDA backend acceleration)
- Apple Silicon with macOS (optional, for MPS acceleration)

## Install from PyPI

```bash
pip install stainx
```

## Install from Source

```bash
git clone https://github.com/rendeirolab/stainx.git
cd stainx
pip install .
```

CUDA extensions will be automatically built if CUDA is available.

## Verify Installation

After installation, verify that StainX is working correctly:

```python
import torch
from stainx import Reinhard

# Test basic functionality
reference = torch.randn(1, 3, 256, 256)
images = torch.randn(4, 3, 256, 256)

normalizer = Reinhard(device="cpu")
normalizer.fit(reference)
normalized = normalizer.transform(images)

print(f"Normalized {images.shape[0]} images successfully!")
print(f"Output shape: {normalized.shape}")
```

## Check Backend Availability

To check if CUDA backends are available:

```python
from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE as TORCH_CUDA_AVAILABLE
from stainx.backends.cupy_cuda_backend import CUDA_AVAILABLE as CUPY_CUDA_AVAILABLE

if TORCH_CUDA_AVAILABLE:
    print("Torch CUDA backend is available!")
if CUPY_CUDA_AVAILABLE:
    print("CuPy CUDA backend is available!")
```

## Development Installation

For development, install with development dependencies:

```bash
pip install -e ".[dev]"
# or
make install-dev
```

This includes testing, linting, and documentation tools.
