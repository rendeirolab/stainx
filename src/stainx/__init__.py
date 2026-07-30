from importlib.metadata import PackageNotFoundError, version

from stainx.base import StainNormalizerBase
from stainx.normalizers import HistogramMatching, Macenko, Reinhard
from stainx.transforms import StainNormalizerTransform

__all__ = ["HistogramMatching", "Macenko", "Reinhard", "StainNormalizerBase", "StainNormalizerTransform", "__version__"]


def _get_version() -> str:
    try:
        return version("stainx")
    except PackageNotFoundError:
        return "0.1.3"


__version__ = _get_version()
