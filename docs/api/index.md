# API Reference

## Normalizers

- [`HistogramMatching`](histogram_matching.md)
- [`Reinhard`](reinhard.md)
- [`Macenko`](macenko.md)

## Base Classes

- [`StainNormalizerBase`](base.md)

## Backends

- [Backend Overview](backends.md)

## Common Interface

All normalizers implement:

- `fit(reference_images)` - Compute normalization parameters
- `transform(images)` - Apply normalization
- `fit_transform(images)` - Fit and transform in one step

**Parameters:**
- `device` (str | torch.device | None): Device (`"cpu"`, `"cuda"`, `"mps"`, or `torch.device`)
- `backend` (str | None): Backend (`"torch"` or `"torch_cuda"`). Auto-selects if None

## Training

- `StainNormalizerTransform` — `from stainx import StainNormalizerTransform`
