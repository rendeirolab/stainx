from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from stainx.base import StainNormalizerBase
from stainx.normalizers import HistogramMatching, Macenko, Reinhard


def _get_version():
    """Get version from package metadata or pyproject.toml."""
    import tomllib

    # Get version from installed package metadata first
    return version("stainx")


__version__ = _get_version()

__all__ = ["HistogramMatching", "Macenko", "Reinhard", "StainNormalizerBase", "__version__"]
