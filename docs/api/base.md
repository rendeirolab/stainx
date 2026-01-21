# StainNormalizerBase

```python
class StainNormalizerBase(ABC)
```

Base class for all normalizers. Abstract class - use `Reinhard`, `Macenko`, or `HistogramMatching`.

## Constructor

```python
StainNormalizerBase(device: str | torch.device | None = None)
```

## Abstract Methods

- `fit(images) -> StainNormalizerBase`
- `transform(images) -> torch.Tensor`

## Methods

- `fit_transform(images) -> torch.Tensor` - Fit and transform in one step

## Properties

- `device` - Device used for computation
- `_is_fitted` - Whether normalizer has been fitted
