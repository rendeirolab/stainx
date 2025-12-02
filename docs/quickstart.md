# Quick Start

## Basic Usage

```python
import torch
from stainx import Reinhard, Macenko, HistogramMatching

reference = torch.randn(1, 3, 512, 512)
images = torch.randn(10, 3, 512, 512)

# Reinhard
normalizer = Reinhard(device="cuda")
normalizer.fit(reference)
normalized = normalizer.transform(images)

# Macenko
normalizer = Macenko(device="cuda")
normalizer.fit(reference)
normalized = normalizer.transform(images)

# Histogram Matching
normalizer = HistogramMatching(device="cuda", channel_axis=1)
normalizer.fit(reference)
normalized = normalizer.transform(images)
```

## Image Formats

- **Channels-first**: `(N, C, H, W)` - Default
- **Channels-last**: `(N, H, W, C)` - Use `channel_axis=-1` for HistogramMatching

## Device & Backend

```python
# Automatic selection
normalizer = Reinhard(device="cuda")

# Force backend
normalizer = Reinhard(device="cuda", backend="pytorch")
```

## Batch Processing

```python
batch = torch.randn(32, 3, 512, 512)
normalized = normalizer.transform(batch)
```
