# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version

import torch


def _get_version():
    """Get version from package metadata or pyproject.toml."""
    try:
        return version("stainx")
    except PackageNotFoundError:
        return "0.1.0"


__version__ = _get_version()

# Set PyTorch library path for runtime linking
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = torch_lib_path
elif torch_lib_path not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"

# Import the compiled CUDA extension if available and loadable
FUNCTIONS_AVAILABLE = False
histogram_matching = None
macenko = None
macenko_fast = None
reinhard = None

if importlib.util.find_spec(f"{__name__}.stainx_cuda_torch") is not None:
    try:
        from .stainx_cuda_torch import histogram_matching, macenko, macenko_fast, reinhard

        if all(callable(f) for f in [histogram_matching, macenko, macenko_fast, reinhard]):
            FUNCTIONS_AVAILABLE = True
    except Exception:
        # Stale/incompatible .so (ABI mismatch) — fall back to Torch backend
        histogram_matching = None
        macenko = None
        macenko_fast = None
        reinhard = None
        FUNCTIONS_AVAILABLE = False

__all__ = ["FUNCTIONS_AVAILABLE", "histogram_matching", "macenko", "macenko_fast", "reinhard"]
