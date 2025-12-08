# Installation

## Requirements

- Python >=3.11
- PyTorch >=2.0.0
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

## Check CUDA Backend Availability

To check if the CUDA backend is available:

```python
from stainx.backends.cuda_backend import CUDA_AVAILABLE

if CUDA_AVAILABLE:
    print("CUDA backend is available!")
else:
    print("CUDA backend is not available. Using PyTorch backend.")
```

## Development Installation

For development, install with development dependencies:

```bash
pip install -e ".[dev]"
# or
make install-dev
```

This includes testing, linting, and documentation tools.
