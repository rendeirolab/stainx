"""
StainX: GPU-accelerated stain normalization for histopathology images.
"""

__version__ = "0.1.0"

# Import main normalizer classes
from .normalizers import HistogramMatching, Macenko, Reinhard, Vahadane

# Import base class for extension
from .base import StainNormalizerBase

__all__ = [
    "__version__",
    "HistogramMatching",
    "Reinhard",
    "Macenko", 
    "Vahadane",
    "StainNormalizerBase",
]
