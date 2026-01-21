// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * CuPy C++ extension bindings for CUDA backend.
 *
 * This file provides Python bindings for CUDA implementations using CuPy's array interface.
 * This is the CuPy-specific interface (stainx_cuda_cupy).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Forward declarations from CUDA files
pybind11::dict histogram_matching_cuda(pybind11::object input_images, pybind11::object reference_histogram);
pybind11::dict reinhard_cuda(pybind11::object input_images, pybind11::object reference_mean, pybind11::object reference_std);
pybind11::dict macenko_cuda(pybind11::object input_images, pybind11::object stain_matrix, pybind11::object target_max_conc);

PYBIND11_MODULE(stainx_cuda_cupy, m) {
    m.doc() = "StainX CUDA backend for GPU-accelerated stain normalization (CuPy interface)";

    // Bind CUDA implementations
    m.def("histogram_matching", &histogram_matching_cuda, "Histogram matching CUDA");
    m.def("reinhard", &reinhard_cuda, "Reinhard normalization CUDA");
    m.def("macenko", &macenko_cuda, "Macenko normalization CUDA");
}
