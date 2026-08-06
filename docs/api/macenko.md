# Macenko

Macenko stain-matrix normalization. Low-level `normalize_to_0_1` defaults to
`False` (output ~`[0, 255]`). Prefer
[`StainNormalizerTransform`](transform.md) for training pipelines — that path
defaults Macenko to `[0, 1]`.

`precision="fast"` requires `backend="torch_cuda"`. Layout is **NCHW** (`C=3`) only.
Float inputs are always treated as `[0, 1]` (no `max()>1` heuristic).

::: stainx.Macenko
    options:
      docstring_style: numpy
      show_root_heading: false
      members:
        - __init__
        - fit
        - transform
        - fit_transform
