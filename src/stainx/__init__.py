import re
from pathlib import Path

from stainx.base import StainNormalizerBase
from stainx.normalizers import HistogramMatching, Macenko, Reinhard


def _get_version():
    """Read version from pyproject.toml (single source of truth)."""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path) as f:
        for line in f:
            if line.strip().startswith("version ="):
                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', line)
                if match:
                    return match.group(1)
    raise RuntimeError("Could not find version in pyproject.toml")


__version__ = _get_version()

__all__ = ["HistogramMatching", "Macenko", "Reinhard", "StainNormalizerBase", "__version__"]
