// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * PyTorch C++ extension bindings for CUDA backend.
 * 
 * This file provides Python bindings for CUDA implementations.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Forward declarations from CUDA files
torch::Tensor histogram_matching_cuda(
    torch::Tensor input_images,
    torch::Tensor reference_histogram
);

// TODO: Add forward declarations for other methods when implemented
// torch::Tensor reinhard_cuda(...);
// torch::Tensor macenko_cuda(...);

PYBIND11_MODULE(stainx_cuda, m) {
    m.doc() = "StainX CUDA backend for GPU-accelerated stain normalization";
    
    // Bind CUDA implementations
    m.def("histogram_matching", &histogram_matching_cuda, "Histogram matching CUDA");
    
    // TODO: Add bindings for other CUDA implementations when ready
    // m.def("reinhard", &reinhard_cuda, "Reinhard normalization CUDA");
    // m.def("macenko", &macenko_cuda, "Macenko normalization CUDA");
}





