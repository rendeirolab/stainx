// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * PyTorch C++ extension bindings for CUDA backend.
 *
 * This file provides Python bindings for CUDA implementations using PyTorch's tensor interface.
 * This is the PyTorch-specific interface (stainx_cuda_torch).
 */

#include <torch/extension.h>

#include <pybind11/pybind11.h>

// Forward declarations from CUDA files
torch::Tensor histogram_matching_cuda(torch::Tensor input_images, torch::Tensor reference_histogram);

torch::Tensor reinhard_cuda(torch::Tensor input_images, torch::Tensor reference_mean, torch::Tensor reference_std);

torch::Tensor macenko_cuda(torch::Tensor input_images, torch::Tensor stain_matrix, torch::Tensor target_max_conc);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "StainX CUDA backend for GPU-accelerated stain normalization (PyTorch interface)";

    // Bind CUDA implementations
    m.def("histogram_matching", &histogram_matching_cuda, "Histogram matching CUDA");
    m.def("reinhard", &reinhard_cuda, "Reinhard normalization CUDA");
    m.def("macenko", &macenko_cuda, "Macenko normalization CUDA");
}
