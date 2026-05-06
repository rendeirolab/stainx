from importlib.metadata import version

from stainx.base import StainNormalizerBase
from stainx.normalizers import HistogramMatching, Macenko, Reinhard

__all__ = ["HistogramMatching", "Macenko", "Reinhard", "StainNormalizerBase", "__version__"]


def _get_version():
    """Get version from package metadata or pyproject.toml."""
    # Get version from installed package metadata first
    return version("stainx")


__version__ = _get_version()
