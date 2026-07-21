# Training pipelines

StainX is a foundation for fast stain normalization. Plug `StainNormalizerTransform` (or the low-level normalizers) into your own DataLoader / training loop.

## `StainNormalizerTransform`

```python
import torch
from stainx import StainNormalizerTransform

reference = torch.rand(1, 3, 224, 224)  # float in [0, 1]
transform = StainNormalizerTransform(
    method="macenko",
    mode="reference",          # preferred for supervised training
    reference=reference,
    device="cuda",
    normalize_to_0_1=True,     # required before torchvision ImageNet Normalize
)

batch = torch.rand(8, 3, 224, 224)
out = transform(batch)         # still in [0, 1]
```

### Value range (Macenko)

| Input | Flag | Output |
|-------|------|--------|
| `uint8` / float in ~`[0, 255]` | default | ~`[0, 255]` |
| float in `[0, 1]` (e.g. `ToDtype(..., scale=True)`) | `normalize_to_0_1=True` | `[0, 1]` |

If you pass float `[0, 1]` without the flag, the transform auto-scales Macenko output back to `[0, 1]` so ImageNet `Normalize` stays correct. Prefer setting `normalize_to_0_1=True` explicitly in training pipelines.

Reinhard already preserves a `[0, 1]` float convention when inputs are unit-scaled.

### Modes

| Mode | Fit | Use for |
|------|-----|---------|
| `reference` | Once on a fixed reference | Default training / evaluation |
| `batch` | Every forward on the batch (or `batch_ref_index`) | Exploratory / domain-shift visualization |

`batch` mode **re-fits inside `forward`**, mutates normalizer state, and changes statistics every step. It is usually **unsafe** for reproducible supervised training and a poor fit under `DataLoader` workers unless that behavior is intentional.

### Checkpointing

Fitted stain parameters are **not** stored in `state_dict()` (they live on the inner normalizer, not as buffers). After loading a training checkpoint, call `fit_reference(...)` again.

See `examples/torch_transform_example.py` for a torchvision / DataLoader wiring example.
