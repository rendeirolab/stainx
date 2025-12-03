# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def _get_version():
    """Get version from package metadata or pyproject.toml."""
    # Try to get version from installed package metadata first
    try:
        return version("stainx")
    except PackageNotFoundError:
        # Fall back to reading from pyproject.toml (for development/source installs)
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path) as f:
                for line in f:
                    if line.strip().startswith("version ="):
                        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', line)
                        if match:
                            return match.group(1)
        raise RuntimeError("Could not find version in package metadata or pyproject.toml") from None


__version__ = _get_version()

# Set PyTorch library path for runtime linking
# This ensures the extension can find PyTorch's shared libraries at runtime

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = torch_lib_path
elif torch_lib_path not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"

# Import the compiled CUDA extension if available
# Track whether functions are actually available (not just if package imports)
FUNCTIONS_AVAILABLE = False
histogram_matching = None
macenko = None
reinhard = None

print("=" * 80)
print("DEBUG: stainx_cuda/__init__.py: Starting CUDA extension import")
print(f"DEBUG: Package location: {Path(__file__).parent}")
print(f"DEBUG: Looking for stainx_cuda module in: {Path(__file__).parent}")

# Check if compiled extension file exists
extension_files = list(Path(__file__).parent.glob("stainx_cuda*.so"))
print(f"DEBUG: Found extension files: {[str(f) for f in extension_files]}")

try:
    print("DEBUG: Attempting to import from .stainx_cuda...")
    from .stainx_cuda import histogram_matching, macenko, reinhard

    print("DEBUG: Successfully imported from .stainx_cuda")

    # Verify functions are actually available
    print(f"DEBUG: histogram_matching callable: {callable(histogram_matching)}")
    print(f"DEBUG: macenko callable: {callable(macenko)}")
    print(f"DEBUG: reinhard callable: {callable(reinhard)}")

    if all(callable(f) for f in [histogram_matching, macenko, reinhard]):
        FUNCTIONS_AVAILABLE = True
        print("DEBUG: All CUDA functions are available!")
        # Expose functions at package level for easy import
        __all__ = ["FUNCTIONS_AVAILABLE", "histogram_matching", "macenko", "reinhard"]
    else:
        print("DEBUG: Some CUDA functions are not callable")
        __all__ = ["FUNCTIONS_AVAILABLE"]
except ImportError as e:
    # CUDA extension not available - this is expected if CUDA extension wasn't built
    print(f"DEBUG: ImportError when importing CUDA extension: {e}")
    print("DEBUG: CUDA extension was not built or is not available")
    __all__ = ["FUNCTIONS_AVAILABLE"]
except Exception as e:
    # Other errors - show error if in debug mode
    print(f"DEBUG: Exception when importing CUDA extension: {type(e).__name__}: {e}")
    if os.environ.get("STAINX_DEBUG_CUDA"):
        import traceback

        traceback.print_exc()
    # Re-raise if it's a library loading issue (not just missing extension)
    if "cannot open shared object file" in str(e) or "libc10" in str(e) or "libtorch" in str(e):
        raise RuntimeError(f"Failed to load CUDA extension due to missing libraries. Please ensure LD_LIBRARY_PATH includes PyTorch's lib directory: {torch_lib_path if 'torch_lib_path' in locals() else 'unknown'}") from e
    __all__ = ["FUNCTIONS_AVAILABLE"]

print(f"DEBUG: FUNCTIONS_AVAILABLE = {FUNCTIONS_AVAILABLE}")
print(f"DEBUG: __all__ = {__all__}")
print("=" * 80)
