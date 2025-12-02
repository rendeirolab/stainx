# Macenko

```python
class Macenko(StainNormalizerBase)
```

## Constructor

```python
Macenko(
    device: str | torch.device | None = None,
    backend: str | None = None
)
```

**Parameters:**
- `device`: Device to use. Default: None (auto-detect)
- `backend`: Backend ("pytorch" or "cuda"). Default: None (auto-select)

## Methods

- `fit(reference_images)` - Compute stain matrix and target max concentration
- `transform(images)` - Apply Macenko normalization

## Example

```python
from stainx import Macenko

normalizer = Macenko(device="cuda")
normalizer.fit(reference)
normalized = normalizer.transform(images)
```
