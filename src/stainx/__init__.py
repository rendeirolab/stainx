from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from stainx.base import StainNormalizerBase
from stainx.normalizers import HistogramMatching, Macenko, Reinhard


def _get_version():
    """Get version from package metadata or pyproject.toml."""
    import tomllib

    # Try to get version from installed package metadata first
    try:
        return version("stainx")
    except PackageNotFoundError:
        # Fall back to reading from pyproject.toml (for development/source installs)
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                return data["project"]["version"]
        raise RuntimeError("Could not find version in package metadata or pyproject.toml") from None


__version__ = _get_version()

__all__ = ["HistogramMatching", "Macenko", "Reinhard", "StainNormalizerBase", "__version__"]
