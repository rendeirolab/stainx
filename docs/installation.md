# Installation

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- CUDA Toolkit + nvcc (optional, for the `torch_cuda` extension)

## Platform support

| Platform | Support |
|----------|---------|
| Linux + CUDA | Primary — Torch + optional CUDA extension |
| Linux CPU | Primary — Torch backend |
| macOS (MPS / CPU) | Best-effort Torch path (no CUDA extension) |
| Windows | Best-effort Torch path (CUDA extension not guaranteed) |

## Install from PyPI

```bash
pip install stainx
```

## Install from source (recommended)

```bash
git clone https://github.com/rendeirolab/stainx.git
cd stainx
make install          # editable + best-effort CUDA build
make install-dev      # + test/docs tooling
```

The Makefile never fails the install if CUDA/nvcc is missing — you continue with the Torch backend only.

Equivalent pip flow:

```bash
pip install .
# or editable
pip install -e .
python setup.py build_ext --inplace  # optional; skipped safely without CUDA
```

## Verify installation

```python
import torch
from stainx import Reinhard, StainNormalizerTransform

reference = torch.randn(1, 3, 256, 256)
images = torch.randn(4, 3, 256, 256)

normalizer = Reinhard(device="cpu")
normalizer.fit(reference)
normalized = normalizer.transform(images)

transform = StainNormalizerTransform(method="reinhard", mode="reference", reference=reference, device="cpu")
assert transform(images).shape == images.shape
print("OK", normalized.shape)
```

## Check CUDA extension availability

```python
from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

print("torch_cuda available:", CUDA_AVAILABLE)
```

## Development installation

```bash
make install-dev
# or
uv sync --group dev
pip install -e .
```
