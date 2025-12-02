# StainX

GPU-accelerated stain normalization for histopathology images.

## Quick Example

```python
import torch
from stainx import Reinhard

reference = torch.randn(1, 3, 512, 512)
images = torch.randn(10, 3, 512, 512)

normalizer = Reinhard(device="cuda")
normalizer.fit(reference)
normalized = normalizer.transform(images)
```

## Installation

```bash
pip install stainx
```

## Documentation

- [Quick Start](quickstart.md)
- [API Reference](api/index.md)
- [Examples](examples.md)
- [Contributing](https://github.com/rendeirolab/stainx/blob/main/CONTRIBUTING.md)
