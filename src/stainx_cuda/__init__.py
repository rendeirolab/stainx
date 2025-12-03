# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


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

# Set PyTorch library path for runtime linking
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = torch_lib_path
elif torch_lib_path not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"

# Import the compiled CUDA extension if available
FUNCTIONS_AVAILABLE = False
try:
    from .stainx_cuda import histogram_matching, macenko, reinhard

    if all(callable(f) for f in [histogram_matching, macenko, reinhard]):
        FUNCTIONS_AVAILABLE = True
        __all__ = ["FUNCTIONS_AVAILABLE", "histogram_matching", "macenko", "reinhard"]
    else:
        __all__ = ["FUNCTIONS_AVAILABLE"]
except ImportError:
    __all__ = ["FUNCTIONS_AVAILABLE"]
except Exception as e:
    if "cannot open shared object file" in str(e) or "libc10" in str(e) or "libtorch" in str(e):
        raise RuntimeError(f"Failed to load CUDA extension due to missing libraries. Please ensure LD_LIBRARY_PATH includes PyTorch's lib directory: {torch_lib_path}") from e
    __all__ = ["FUNCTIONS_AVAILABLE"]
