# HistogramMatching

Per-channel histogram matching. Supports NCHW (`channel_axis=1` or `-3`) and
NHWC (`channel_axis=-1` or `3`). Float inputs are always treated as `[0, 1]`.

::: stainx.HistogramMatching
    options:
      docstring_style: numpy
      show_root_heading: false
      members:
        - __init__
        - fit
        - transform
        - fit_transform
