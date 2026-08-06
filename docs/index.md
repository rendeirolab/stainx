# StainX

<p align="center">
  <img src="https://raw.githubusercontent.com/rendeirolab/stainx/refs/heads/main/assets/StainX-logo.svg" alt="StainX Logo" width="256"/>
  <br/>
  <a href="https://github.com/rendeirolab/stainx/actions/workflows/ci.yml">
    <img src="https://github.com/rendeirolab/stainx/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"/>
</p>

## Overview

**StainX** is a Torch-first stain normalization library for histopathology images,
designed for efficient **batch** processing on CPU, CUDA, and MPS, with an optional
compiled `torch_cuda` extension.

### Key Advantages

- **Batch Processing**: Process multiple images simultaneously, maximizing GPU throughput
- **Multi-Device Support**: CPU, CUDA (NVIDIA), and MPS (Apple Silicon)
- **Multiple Algorithms**: Histogram Matching, Reinhard, and Macenko
- **Automatic Backend Selection**: Chooses between PyTorch ops and compiled CUDA kernels when available

### Why Batch Processing Matters

Batch processing is crucial for histopathology workflows where you often need to normalize
hundreds or thousands of images:

- **Better GPU Utilization**: Parallel processing across the batch
- **Reduced Overhead**: Fewer kernel launches than per-image loops
- **Higher Throughput**: Batch sizes of 64–128 typically outperform single-image transforms
  (see [Benchmarks](benchmarks.md); regenerate numbers on your hardware)

## Quick Example

Use float tensors in `[0, 1]` (or `uint8`). Prefer `torch.rand` — Macenko does not
accept negative pixels from `torch.randn`.

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching

reference_image = torch.rand(1, 3, 512, 512)  # NCHW, float [0, 1]
source_images = torch.rand(10, 3, 512, 512)

normalizer = Reinhard(device="cuda")  # or "cpu", "mps"
normalizer.fit(reference_image)
normalized = normalizer.transform(source_images)
```

## Performance

On an NVIDIA RTX A6000 during 0.1.x development, `torch_cuda` Reinhard was about
**~5.7×** the pure Torch backend at mid batch sizes, with throughput climbing into
the tens of thousands of images/second for 256² tiles. Those figures are historical —
re-run `benchmarks/benchmark_stainx_backend.py` and see [Benchmarks](benchmarks.md).

## Installation

```bash
pip install stainx
# from source: make install
```

Requires Python >= 3.11 and PyTorch >= 2.0.0. PyPI ships an **sdist** (Torch backends
out of the box). The optional `torch_cuda` extension compiles locally when a CUDA GPU
is visible to PyTorch **and** `nvcc` is available. See [Installation](installation.md).

## Features

- **Multiple algorithms**: Histogram Matching, Reinhard, and Macenko
- **Torch backends**: `torch` and optional `torch_cuda`
- **Training transforms**: `StainNormalizerTransform` for DataLoader pipelines
- **Batch processing**: Efficient multi-image normalization
- **Flexible device support**: CPU, CUDA, MPS (Apple Silicon)

## Documentation

- [Installation Guide](installation.md)
- [Quick Start Guide](quickstart.md)
- [Training](training.md)
- [Notebook](examples/visualize_normalization.ipynb) — Visual before/after stain normalization
- [Benchmarks](benchmarks.md)
- [Correctness Report](correctness_report.md)
- [API Reference](api/index.md)
- [Changelog](changelog.md)

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/rendeirolab/stainx/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the GNU General Public License v3 (GPL-3.0-or-later).

## Links

- **GitHub**: https://github.com/rendeirolab/stainx
- **Issues**: https://github.com/rendeirolab/stainx/issues
- **Documentation**: https://stainx.readthedocs.io/
