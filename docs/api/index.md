# API Reference

## Normalizers

- [`HistogramMatching`](histogram_matching.md)
- [`Reinhard`](reinhard.md)
- [`Macenko`](macenko.md)

## Training

- [`StainNormalizerTransform`](transform.md)

## Base Classes

- [`StainNormalizerBase`](base.md)

## Backends

- [Backend Overview](backends.md)

## Common Interface

All normalizers implement:

- `fit(images)` — Compute normalization parameters
- `transform(images)` — Apply normalization
- `fit_transform(images)` — Fit and transform in one step

**Constructor parameters (concrete normalizers):**

- `device` (`str | torch.device | None`): Device (`"cpu"`, `"cuda"`, `"mps"`, or `torch.device`).
  Default `None` auto-selects CUDA > MPS > CPU.
- `backend` (`str | None`): `"torch"` or `"torch_cuda"`. Auto-selects if `None`.

Macenko also accepts `normalize_to_0_1` and `precision`. HistogramMatching also accepts
`channel_axis`. See the class pages for details.

## Version

```python
import stainx
print(stainx.__version__)
```
