# HistogramMatching

```python
class HistogramMatching(StainNormalizerBase)
```

## Constructor

```python
HistogramMatching(
    device: str | torch.device | None = None,
    backend: str | None = None,
    channel_axis: int = 1
)
```

**Parameters:**
- `device`: Device to use. Default: None (auto-detect)
- `backend`: Backend ("pytorch" or "cuda"). Default: None (auto-select)
- `channel_axis`: Channel axis. Use `1` for `(N, C, H, W)` or `-1` for `(N, H, W, C)`. Default: 1

## Methods

- `fit(reference_images)` - Compute reference histogram
- `transform(images)` - Apply histogram matching

## Example

```python
from stainx import HistogramMatching

normalizer = HistogramMatching(device="cuda", channel_axis=1)
normalizer.fit(reference)
normalized = normalizer.transform(images)
```
