# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
__version__ = "0.1.0"

# Import the compiled CUDA extension if available
try:
    import importlib.util
    import sys
    from pathlib import Path

    # Find and load the compiled .so file
    # Check multiple possible locations and patterns
    parent_dir = Path(__file__).parent
    so_file = None
    
    # Try different patterns for the .so file
    patterns = [
        "stainx_cuda*.so",
        "*_compiled*.so",
        "*.so",  # Fallback: any .so file in the directory
    ]
    
    for pattern in patterns:
        matches = list(parent_dir.glob(pattern))
        if matches:
            so_file = matches[0]
            break
    
    # Also check parent directory (where --inplace might put it)
    if not so_file:
        parent_parent = parent_dir.parent
        for pattern in ["stainx_cuda*.so", "*_compiled*.so"]:
            matches = list(parent_parent.glob(pattern))
            if matches:
                so_file = matches[0]
                break

    if so_file:
        spec = importlib.util.spec_from_file_location("_compiled", so_file)
        if spec and spec.loader:
            _compiled = importlib.util.module_from_spec(spec)
            # Register in sys.modules to avoid reload issues
            sys.modules["stainx_cuda._compiled"] = _compiled
            spec.loader.exec_module(_compiled)
            # Import only the CUDA functions (not other attributes)
            for attr in ["histogram_matching", "reinhard", "macenko"]:
                if hasattr(_compiled, attr):
                    globals()[attr] = getattr(_compiled, attr)
        else:
            # Failed to create spec
            import warnings
            warnings.warn(f"Failed to create module spec from {so_file}")
    else:
        # No .so file found - this is expected if CUDA extension wasn't built
        pass
except Exception as e:
    # CUDA extension not available - silently fail unless in debug mode
    import os
    if os.environ.get("STAINX_DEBUG_CUDA"):
        import traceback
        traceback.print_exc()
    pass

__all__ = []
