# Reinhard

```python
class Reinhard(StainNormalizerBase)
```

## Constructor

```python
Reinhard(
    device: str | torch.device | None = None,
    backend: str | None = None
)
```

**Parameters:**
- `device`: Device to use. Default: None (auto-detect)
- `backend`: Backend ("pytorch" or "cuda"). Default: None (auto-select)

## Methods

- `fit(reference_images)` - Compute reference mean/std in LAB color space
- `transform(images)` - Apply Reinhard normalization

## Example

```python
from stainx import Reinhard

normalizer = Reinhard(device="cuda")
normalizer.fit(reference)
normalized = normalizer.transform(images)
```
