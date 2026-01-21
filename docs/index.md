# StainX

<p align="center">
  <img src="https://raw.githubusercontent.com/rendeirolab/stainx/refs/heads/main/assets/staix-logo-256.png" alt="StainX Logo" width="256"/>
  <br/>
  <a href="https://github.com/rendeirolab/stainx/actions/workflows/ci.yml">
    <img src="https://github.com/rendeirolab/stainx/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"/>
</p>

## Overview

**StainX** is an enhanced stain normalization library for histopathology images that provides significant performance improvements through optimized batch processing. Unlike other frameworks that process images individually, StainX is designed from the ground up to handle batches of images efficiently, resulting in better GPU utilization and faster processing times.

### Key Advantages

- **Batch Processing**: Process multiple images simultaneously, maximizing GPU throughput and reducing overhead
- **Multi-Device Support**: Seamlessly works on CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon)
- **Multiple Algorithms**: Supports Histogram Matching, Reinhard, and Macenko normalization methods
- **Automatic Backend Selection**: Intelligently chooses between optimized CUDA kernels and PyTorch backends

### Why Batch Processing Matters

Batch processing is crucial for histopathology workflows where you often need to normalize hundreds or thousands of images. By processing images in batches:

- **Better GPU Utilization**: Parallel processing across the entire batch maximizes GPU compute resources
- **Reduced Overhead**: Single kernel launches for entire batches instead of per-image launches
- **Faster Processing**: Up to 5-7x speedup with CUDA backend compared to PyTorch backend for batch processing
- **Memory Efficiency**: Optimized memory access patterns for batched operations
- **Higher Throughput**: Batch processing achieves 40,000+ images/second (vs ~5,500 for single images)

## Quick Example

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching

# Load your images as torch.Tensor (N, C, H, W) or (N, H, W, C)
reference_image = torch.randn(1, 3, 512, 512)  # Reference image
source_images = torch.randn(10, 3, 512, 512)    # Batch of images to normalize

# Reinhard normalization
normalizer = Reinhard(device="cuda")  # or "cpu", "mps"
normalizer.fit(reference_image)
normalized = normalizer.transform(source_images)  # Process entire batch at once
```

## Performance

StainX provides significant performance improvements, especially when processing batches of images. Based on benchmarks on NVIDIA RTX A6000:

- **CUDA Backend Speedup**: 5.3-5.4x for Reinhard, 4.6-7.3x for Macenko
- **Batch Processing Throughput**: Up to 46,600 images/second (vs ~5,500 for single images)
- **Optimal Batch Size**: 64-128 images provides best performance

See the [Benchmarks](benchmarks.md) page for detailed performance benchmarks and code examples.

## Installation

```bash
pip install stainx
```

CUDA extensions will be automatically built if CUDA is available. Requires PyTorch >=2.0.0 and CuPy >=12.0.0.

## Features

- **Multiple algorithms**: Histogram Matching, Reinhard, and Macenko normalization
- **Automatic backend selection**: torch, torch_cuda, cupy, or cupy_cuda backends
- **Batch processing**: Enhanced normalization through efficient batch processing of multiple images
- **Flexible device support**: CPU, CUDA, MPS (Apple Silicon)

## Documentation

- [Quick Start Guide](quickstart.md) - Get started in minutes
- [Installation Guide](installation.md) - Detailed installation instructions
- [Examples](examples.md) - Usage examples and patterns
- [Benchmarks](benchmarks.md) - Performance benchmarks and comparisons
- [API Reference](api/index.md) - Complete API documentation

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/rendeirolab/stainx/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the GNU General Public License v3 (GPL-3.0-or-later).

## Links

- **GitHub**: https://github.com/rendeirolab/stainx
- **Issues**: https://github.com/rendeirolab/stainx/issues
- **Documentation**: https://stainx.readthedocs.io/
