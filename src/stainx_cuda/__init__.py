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
    so_file = next(Path(__file__).parent.glob("stainx_cuda*.so"), None)

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
except Exception:
    # CUDA extension not available - silently fail
    pass

__all__ = []
