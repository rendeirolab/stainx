# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import os

__version__ = "0.1.0"

# Set PyTorch library path for runtime linking
# This ensures the extension can find PyTorch's shared libraries at runtime
import torch

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = torch_lib_path
elif torch_lib_path not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"

# Import the compiled CUDA extension if available
try:
    from .stainx_cuda import histogram_matching, reinhard, macenko

    # Expose functions at package level for easy import
    __all__ = ["histogram_matching", "reinhard", "macenko"]
except ImportError:
    # CUDA extension not available - this is expected if CUDA extension wasn't built
    __all__ = []
except Exception as e:
    # Other errors - show error if in debug mode
    if os.environ.get("STAINX_DEBUG_CUDA"):
        import traceback

        traceback.print_exc()
    # Re-raise if it's a library loading issue (not just missing extension)
    if "cannot open shared object file" in str(e) or "libc10" in str(e) or "libtorch" in str(e):
        raise RuntimeError(
            f"Failed to load CUDA extension due to missing libraries. "
            f"Please ensure LD_LIBRARY_PATH includes PyTorch's lib directory: {torch_lib_path if 'torch_lib_path' in locals() else 'unknown'}"
        ) from e
    __all__ = []
