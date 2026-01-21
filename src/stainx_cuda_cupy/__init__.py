# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import cupy as cp


def _get_version():
    """Get version from package metadata or pyproject.toml."""
    from importlib.metadata import version

    # Get version from installed package metadata first
    return version("stainx")


__version__ = _get_version()

# Import the compiled CUDA extension if available
FUNCTIONS_AVAILABLE = False
try:
    from . import stainx_cuda_cupy

    if all(hasattr(stainx_cuda_cupy, f) for f in ["histogram_matching", "macenko", "reinhard"]):
        FUNCTIONS_AVAILABLE = True

        # Create wrapper functions that handle CuPy array conversion
        def _create_cupy_array_from_ptr(ptr, shape, dtype_str):
            """Create a CuPy array from a device pointer."""
            # Convert dtype string to CuPy dtype
            dtype_map = {"float32": cp.float32, "uint8": cp.uint8}
            dtype = dtype_map.get(dtype_str, cp.float32)

            # Create memory pointer
            memptr = cp.cuda.UnownedMemory(ptr, shape[0] * shape[1] * shape[2] * shape[3] * cp.dtype(dtype).itemsize, None)
            memptr_obj = cp.cuda.MemoryPointer(memptr, 0)

            # Create array
            return cp.ndarray(shape, dtype, memptr_obj)

        def histogram_matching(input_images, reference_histogram):
            """Histogram matching with CuPy arrays."""
            result_dict = stainx_cuda_cupy.histogram_matching(input_images, reference_histogram)
            return _create_cupy_array_from_ptr(result_dict["ptr"], result_dict["shape"], result_dict["dtype"])

        def reinhard(input_images, reference_mean, reference_std):
            """Reinhard normalization with CuPy arrays."""
            result_dict = stainx_cuda_cupy.reinhard(input_images, reference_mean, reference_std)
            return _create_cupy_array_from_ptr(result_dict["ptr"], result_dict["shape"], result_dict["dtype"])

        def macenko(input_images, stain_matrix, target_max_conc):
            """Macenko normalization with CuPy arrays."""
            result_dict = stainx_cuda_cupy.macenko(input_images, stain_matrix, target_max_conc)
            return _create_cupy_array_from_ptr(result_dict["ptr"], result_dict["shape"], result_dict["dtype"])

        __all__ = ["FUNCTIONS_AVAILABLE", "histogram_matching", "macenko", "reinhard"]
    else:
        __all__ = ["FUNCTIONS_AVAILABLE"]
except ImportError:
    __all__ = ["FUNCTIONS_AVAILABLE"]
