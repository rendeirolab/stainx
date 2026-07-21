# Training pipelines

StainX is a foundation for fast stain normalization. Plug `StainNormalizerTransform` (or the low-level normalizers) into your own DataLoader / training loop.

## `StainNormalizerTransform`

```python
import torch
from stainx import StainNormalizerTransform

reference = torch.rand(1, 3, 224, 224)  # float in [0, 1], NCHW
transform = StainNormalizerTransform(
    method="macenko",
    mode="reference",          # preferred for supervised training
    reference=reference,
    device="cuda",             # optional; default keeps the input device
    normalize_to_0_1=True,     # required before torchvision ImageNet Normalize
)

batch = torch.rand(8, 3, 224, 224)
out = transform(batch)         # still in [0, 1]
```

### Layout

| Method | Layout |
|--------|--------|
| Macenko / Reinhard | **NCHW** with `C=3` only (`channel_axis` must stay default `1`) |
| Histogram matching | NCHW (`channel_axis=1`) or NHWC (`channel_axis=-1` / `3`) |

Passing NHWC into Macenko/Reinhard raises — those backends would otherwise treat height as channels.

### Value range (Macenko)

| Input | Flag | Output |
|-------|------|--------|
| `uint8` / float in ~`[0, 255]` | default | ~`[0, 255]` |
| float in `[0, 1]` (e.g. `ToDtype(..., scale=True)`) | **`normalize_to_0_1=True`** | `[0, 1]` |

Always set `normalize_to_0_1=True` for unit-float training pipelines. There is no `amax()`-based auto-scale (it breaks after color jitter and syncs CUDA every step).

### Modes

| Mode | Fit | Use for |
|------|-----|---------|
| `reference` | Once on a fixed reference | Default training / evaluation |
| `batch` | Every forward on the batch (or `batch_ref_index`) | Exploratory / domain-shift visualization |

`batch` mode **re-fits inside `forward`**, mutates normalizer state, and changes statistics every step. It is usually **unsafe** for reproducible supervised training and a poor fit under `DataLoader` workers unless that behavior is intentional.

### Device

Default `device=None` keeps batches on the **input** device and syncs the inner
normalizer (and auto backend selection) to that device on first fit/forward.
Pass `device="cuda"` to always move data onto the GPU inside the transform.
`backend="torch_cuda"` with `device=None` requires CUDA input tensors.

### Checkpointing

Fitted stain parameters are **not** stored in `state_dict()` (they live on the inner normalizer, not as buffers). After loading a training checkpoint, call `fit_reference(...)` again.

See `examples/torch_transform_example.py` for a torchvision / DataLoader wiring example.
