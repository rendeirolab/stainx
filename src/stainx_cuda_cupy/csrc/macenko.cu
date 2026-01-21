// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * CuPy wrapper for Macenko normalization CUDA kernels.
 *
 * This file provides CuPy array interfaces that call the pure CUDA kernels
 * from the main csrc/ directory.
 */

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <vector>

// Include the pure CUDA kernels from the main csrc directory
#include "csrc/macenko.cu"

#define THREADS_PER_BLOCK 256

namespace py = pybind11;

// Helper functions (same as histogram_matching.cu)
void* get_cupy_ptr(py::object arr) {
    auto intf = arr.attr("__cuda_array_interface__");
    uintptr_t ptr = intf["data"].cast<py::tuple>()[0].cast<uintptr_t>();
    return reinterpret_cast<void*>(ptr);
}

std::vector<int> get_cupy_shape(py::object arr) {
    auto intf = arr.attr("__cuda_array_interface__");
    std::vector<ssize_t> shape_py = intf["shape"].cast<std::vector<ssize_t>>();
    std::vector<int> shape(shape_py.begin(), shape_py.end());
    return shape;
}

std::string get_cupy_dtype(py::object arr) {
    auto intf = arr.attr("__cuda_array_interface__");
    return intf["typestr"].cast<std::string>();
}

py::dict macenko_cuda(py::object input_images_obj, py::object stain_matrix_obj, py::object target_max_conc_obj) {
    // Extract device pointers and shapes
    void* input_ptr = get_cupy_ptr(input_images_obj);
    void* stain_matrix_ptr = get_cupy_ptr(stain_matrix_obj);
    void* target_max_conc_ptr = get_cupy_ptr(target_max_conc_obj);
    
    std::vector<int> input_shape = get_cupy_shape(input_images_obj);

    // Check inputs
    if (input_shape.size() != 4 || input_shape[1] != 3) {
        throw std::runtime_error("input_images must be 4D (N, 3, H, W)");
    }

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaSuccess;

    int N = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];

    // Allocate output array
    size_t output_size = N * C * H * W * sizeof(float);
    void* output_mem = nullptr;
    err = cudaMalloc(&output_mem, output_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error allocating output array: " + std::string(cudaGetErrorString(err)));
    }
    float* output_ptr = static_cast<float*>(output_mem);

    // Macenko implementation would go here (simplified for now)
    // Full implementation would match PyTorch version
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(output_mem);
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    // Return output info
    py::dict result;
    result["ptr"] = reinterpret_cast<uintptr_t>(output_mem);
    result["shape"] = py::make_tuple(N, C, H, W);
    result["dtype"] = "float32";
    result["needs_free"] = true;
    return result;
}
