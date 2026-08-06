# StainNormalizerTransform

`nn.Module` wrapper for DataLoader / torchvision pipelines. Prefer
`mode="reference"` for supervised training.

Key differences from bare normalizers:

- `device=None` keeps the **input** device (bare normalizers auto-pick CUDA > MPS > CPU)
- `method="macenko"` defaults `normalize_to_0_1=True` (bare `Macenko` defaults `False`)
- Fitted stain parameters are **not** in `state_dict()` — call `fit_reference` after loading a checkpoint

::: stainx.StainNormalizerTransform
    options:
      docstring_style: google
      show_root_heading: false
      members:
        - __init__
        - fit_reference
        - forward
