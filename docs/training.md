# Training pipelines

StainX is a foundation for fast stain normalization. Plug `StainNormalizerTransform` (or the low-level normalizers) into your own DataLoader / training loop.

## `StainNormalizerTransform`

```python
import torch
from stainx import StainNormalizerTransform

reference = torch.rand(1, 3, 224, 224)
transform = StainNormalizerTransform(
    method="macenko",          # or "reinhard", "histogram_matching"
    mode="reference",          # preferred for supervised training
    reference=reference,
    device="cuda",
)

batch = torch.rand(8, 3, 224, 224)
out = transform(batch)
```

### Modes

| Mode | Fit | Use for |
|------|-----|---------|
| `reference` | Once on a fixed reference | Default training / evaluation |
| `batch` | Every forward on the batch (or `batch_ref_index`) | Exploratory / domain-shift visualization |

`batch` mode changes statistics every step and is usually **unsafe** for reproducible supervised training unless that is intentional.

See `examples/torch_transform_example.py` for a torchvision / DataLoader wiring example.
