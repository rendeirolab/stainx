// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * CuPy Macenko binding stub.
 *
 * Shipping backend="cupy_cuda" uses MacenkoCupy (Python), not this extension.
 * The old pure-CUDA Macenko kernels were removed; do not reintroduce a divergent
 * histogram/SVD path without torchstain parity.
 */

#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

py::dict macenko_cuda(py::object /*input_images_obj*/, py::object /*stain_matrix_obj*/, py::object /*target_max_conc_obj*/) {
    throw std::runtime_error(
        "stainx_cuda_cupy.macenko is not implemented. Use backend='cupy' or backend='cupy_cuda' "
        "(MacenkoCupy Python path), or backend='torch_cuda' for the Torch ATen Macenko path.");
}
