# Installation

## Requirements

- Python >=3.12
- PyTorch >=2.0.0
- CUDA Toolkit (optional, for GPU acceleration)

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

CUDA extensions build automatically if CUDA is available.

## Development

```bash
pip install -e ".[dev]"
# or
make install-dev
```
