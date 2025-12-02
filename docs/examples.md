# Examples

## Basic Usage

```python
import torch
from stainx import Reinhard

reference = torch.randn(1, 3, 512, 512)
images = torch.randn(10, 3, 512, 512)

normalizer = Reinhard(device="cuda")
normalizer.fit(reference)
normalized = normalizer.transform(images)
```

## All Normalizers

```python
from stainx import Reinhard, Macenko, HistogramMatching

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

## Channels-Last Format

```python
images = torch.randn(10, 512, 512, 3)  # (N, H, W, C)
normalizer = HistogramMatching(device="cuda", channel_axis=-1)
normalizer.fit(images[:1])
normalized = normalizer.transform(images)
```

## Batch Processing

```python
batch = torch.randn(32, 3, 512, 512)
normalized = normalizer.transform(batch)
```
