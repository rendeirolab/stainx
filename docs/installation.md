# Installation

## Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- Optional CUDA extension (`torch_cuda` / `stainx_cuda_torch`): a CUDA-capable GPU
  visible to PyTorch at **build** time (`torch.cuda.is_available()`) **and** `nvcc`
  on `PATH` (or under `CUDA_HOME`)

## Platform support

| Platform | Support | CI coverage |
|----------|---------|-------------|
| Linux + CUDA | Primary — Torch + optional CUDA extension | GPU job (Python 3.12) |
| Linux CPU | Primary — Torch backend | Python 3.11–3.13 |
| Windows | Torch path (CUDA extension not guaranteed) | Python 3.11–3.12 (CPU) |
| macOS (MPS / CPU) | Best-effort Torch path (no CUDA extension) | Not in CI |

## Install from PyPI

```bash
pip install stainx
```

PyPI ships an **sdist** (source distribution), not prebuilt CUDA wheels. A plain
`pip install` gives the Torch backends. The `torch_cuda` extension compiles on
your machine only when the CUDA build gates above are met.

## Install from source (recommended)

```bash
git clone https://github.com/rendeirolab/stainx.git
cd stainx
make install          # editable + best-effort CUDA build
make install-dev      # + test/docs tooling
```

The Makefile never fails the install if CUDA/`nvcc` is missing — you continue with
the Torch backend only.

Equivalent pip flow:

```bash
pip install .
# or editable
pip install -e .
python setup.py build_ext --inplace  # optional; skipped when CUDA gates fail
```

Note: bare `pip install .` can still fail if CUDA gates pass but the compile fails.
Prefer `make install` for best-effort builds.

## Verify installation

```python
import torch
from stainx import Reinhard, StainNormalizerTransform

reference = torch.rand(1, 3, 256, 256)  # float in [0, 1], NCHW
images = torch.rand(4, 3, 256, 256)

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
python setup.py build_ext --inplace  # optional best-effort CUDA build
```
