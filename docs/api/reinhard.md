# Reinhard

LAB mean/std stain normalization. Layout is **NCHW** (`C=3`) only.
Float inputs are always treated as `[0, 1]`.

::: stainx.Reinhard
    options:
      docstring_style: numpy
      show_root_heading: false
      members:
        - __init__
        - fit
        - transform
        - fit_transform
