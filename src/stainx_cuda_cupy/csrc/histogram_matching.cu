// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * CuPy wrapper for histogram matching CUDA kernels.
 *
 * This file provides CuPy array interfaces that call the pure CUDA kernels
 * from the main csrc/ directory.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <vector>

// Include the pure CUDA kernels from the main csrc directory
// Project root is in include_dirs, so we can include from csrc/
#include "csrc/histogram_matching.cu"

#define THREADS_PER_BLOCK 256

namespace py = pybind11;

// Helper function to extract device pointer from CuPy array
void* get_cupy_ptr(py::object arr) {
    auto intf = arr.attr("__cuda_array_interface__");
    uintptr_t ptr = intf["data"].cast<py::tuple>()[0].cast<uintptr_t>();
    return reinterpret_cast<void*>(ptr);
}

// Helper function to get shape from CuPy array
std::vector<int> get_cupy_shape(py::object arr) {
    auto intf = arr.attr("__cuda_array_interface__");
    std::vector<ssize_t> shape_py = intf["shape"].cast<std::vector<ssize_t>>();
    std::vector<int> shape(shape_py.begin(), shape_py.end());
    return shape;
}

// Helper function to get dtype string from CuPy array
std::string get_cupy_dtype(py::object arr) {
    auto intf = arr.attr("__cuda_array_interface__");
    return intf["typestr"].cast<std::string>();
}

// Main function - returns output as a dict with pointer and shape info
// Python side will create the CuPy array from this
py::dict histogram_matching_cuda(py::object input_images_obj, py::object reference_histogram_obj) {
    // Extract device pointers
    void* input_ptr = get_cupy_ptr(input_images_obj);
    void* ref_hist_original_ptr = get_cupy_ptr(reference_histogram_obj);
    
    // Extract shapes
    std::vector<int> input_shape = get_cupy_shape(input_images_obj);
    std::vector<int> ref_hist_shape = get_cupy_shape(reference_histogram_obj);
    std::string input_dtype = get_cupy_dtype(input_images_obj);
    std::string ref_hist_dtype = get_cupy_dtype(reference_histogram_obj);

    // Check inputs
    if (input_shape.size() != 4) {
        throw std::runtime_error("input_images must be 4D (N, C, H, W), got " + std::to_string(input_shape.size()) + "D array");
    }

    // Get device and stream
    cudaStream_t stream = nullptr;  // Use default stream for CuPy
    cudaError_t err = cudaSuccess;

    // Track original dtype for output preservation
    bool is_uint8 = (input_dtype == "|u1");
    bool needs_scale_back = false;
    bool was_uint8_or_high_range = false;

    int N = input_shape[0];
    int C = input_shape[1];
    int H = input_shape[2];
    int W = input_shape[3];
    int num_bins = 256;
    int total_pixels_per_channel = N * H * W;
    int num_threads = THREADS_PER_BLOCK;

    // Check input range and convert to uint8 if needed
    uint8_t* images_uint8_ptr = nullptr;
    void* images_uint8_mem = nullptr;
    
    if (is_uint8) {
        images_uint8_ptr = static_cast<uint8_t*>(input_ptr);
        was_uint8_or_high_range = true;
    } else {
        // For float input, check range and convert
        float* input_float = static_cast<float*>(input_ptr);
        int total_pixels = N * C * H * W;
        
        // Sample max value (check first 1000 elements)
        float max_val = 0.0f;
        for (int i = 0; i < std::min(total_pixels, 1000); i++) {
            if (input_float[i] > max_val) max_val = input_float[i];
        }
        
        if (max_val <= 1.0f) {
            needs_scale_back = true;
        } else {
            was_uint8_or_high_range = true;
        }

        // Allocate uint8 array
        size_t uint8_size = N * C * H * W * sizeof(uint8_t);
        err = cudaMalloc(&images_uint8_mem, uint8_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error allocating uint8 array: " + std::string(cudaGetErrorString(err)));
        }
        images_uint8_ptr = static_cast<uint8_t*>(images_uint8_mem);
        
        int num_blocks = (total_pixels + num_threads - 1) / num_threads;
        convert_to_uint8_kernel<<<num_blocks, num_threads, 0, stream>>>(
            input_float, images_uint8_ptr, needs_scale_back, total_pixels);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(images_uint8_mem);
            throw std::runtime_error("CUDA error in convert_to_uint8_kernel: " + std::string(cudaGetErrorString(err)));
        }
    }

    // Prepare reference histogram - ensure float32
    float* ref_hist_ptr = nullptr;
    void* ref_hist_mem = nullptr;
    
    if (ref_hist_dtype == "<f4") {  // float32
        ref_hist_ptr = static_cast<float*>(ref_hist_original_ptr);
    } else {
        // Need to convert - allocate new memory and convert
        size_t ref_hist_size = 1;
        for (auto dim : ref_hist_shape) ref_hist_size *= dim;
        size_t float_size = ref_hist_size * sizeof(float);
        err = cudaMalloc(&ref_hist_mem, float_size);
        if (err != cudaSuccess) {
            if (images_uint8_mem) cudaFree(images_uint8_mem);
            throw std::runtime_error("CUDA error allocating ref_hist array: " + std::string(cudaGetErrorString(err)));
        }
        ref_hist_ptr = static_cast<float*>(ref_hist_mem);
        // For now, assume conversion handled in Python - just use the pointer
        // In practice, would need a conversion kernel here
    }

    bool per_channel_histograms = false;
    if (ref_hist_shape.size() == 1 && ref_hist_shape[0] == num_bins) {
        per_channel_histograms = false;
    } else if (ref_hist_shape.size() == 2 && ref_hist_shape[0] == C && ref_hist_shape[1] == num_bins) {
        per_channel_histograms = true;
    } else {
        if (images_uint8_mem) cudaFree(images_uint8_mem);
        throw std::runtime_error(
            "reference_histogram must be 1D with 256 elements or 2D with shape (C, 256)");
    }

    // Allocate output array
    size_t output_size = N * C * H * W * sizeof(float);
    void* output_mem = nullptr;
    err = cudaMalloc(&output_mem, output_size);
    if (err != cudaSuccess) {
        if (images_uint8_mem) cudaFree(images_uint8_mem);
        throw std::runtime_error("CUDA error allocating output array: " + std::string(cudaGetErrorString(err)));
    }
    float* output_ptr = static_cast<float*>(output_mem);

    // Choose blocks per channel
    const int pixels_per_block = 4096;
    int blocks_per_channel = (total_pixels_per_channel + pixels_per_block - 1) / pixels_per_block;
    if (blocks_per_channel < 1) blocks_per_channel = 1;
    if (blocks_per_channel > 2048) blocks_per_channel = 2048;

    // Allocate temporary buffers
    size_t partial_hist_size = C * blocks_per_channel * num_bins * sizeof(uint32_t);
    void* partial_hist_mem = nullptr;
    err = cudaMalloc(&partial_hist_mem, partial_hist_size);
    if (err != cudaSuccess) goto cleanup_error;
    uint32_t* partial_hist_ptr = static_cast<uint32_t*>(partial_hist_mem);
    
    size_t hist_u32_size = C * num_bins * sizeof(uint32_t);
    void* hist_u32_mem = nullptr;
    err = cudaMalloc(&hist_u32_mem, hist_u32_size);
    if (err != cudaSuccess) goto cleanup_error;
    uint32_t* hist_u32_ptr = static_cast<uint32_t*>(hist_u32_mem);
    
    size_t ref_cdf_size = C * num_bins * sizeof(float);
    void* ref_cdf_mem = nullptr;
    err = cudaMalloc(&ref_cdf_mem, ref_cdf_size);
    if (err != cudaSuccess) goto cleanup_error;
    float* ref_cdf_ptr = static_cast<float*>(ref_cdf_mem);
    
    size_t lut_size = C * num_bins * sizeof(float);
    void* lut_mem = nullptr;
    err = cudaMalloc(&lut_mem, lut_size);
    if (err != cudaSuccess) goto cleanup_error;
    float* lut_ptr = static_cast<float*>(lut_mem);

    // 1) Partial histograms
    dim3 grid_partial(blocks_per_channel, C, 1);
    hm_partial_hist_kernel<<<grid_partial, num_threads, 0, stream>>>(
        images_uint8_ptr, partial_hist_ptr, N, C, H, W, blocks_per_channel);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup_error;

    // 2) Reduce partials
    dim3 grid_reduce(1, C, 1);
    hm_reduce_hist_kernel<<<grid_reduce, num_threads, 0, stream>>>(
        partial_hist_ptr, hist_u32_ptr, C, blocks_per_channel);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup_error;

    // 3) Compute reference CDF
    hm_ref_cdf_kernel<<<C, num_threads, 0, stream>>>(
        ref_hist_ptr, ref_cdf_ptr, C, per_channel_histograms);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup_error;

    // 4) Build LUT
    hm_build_lut_kernel<<<C, num_threads, 0, stream>>>(
        hist_u32_ptr, ref_cdf_ptr, lut_ptr, N, H, W);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup_error;

    // 5) Apply LUT
    int total_pixels = N * C * H * W;
    int num_blocks = (total_pixels + num_threads - 1) / num_threads;
    hm_apply_lut_kernel<<<num_blocks, num_threads, 0, stream>>>(
        images_uint8_ptr, output_ptr, lut_ptr, N, C, H, W);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup_error;

    // Scale and clamp
    scale_clamp_output_kernel<<<num_blocks, num_threads, 0, stream>>>(
        output_ptr, needs_scale_back, false, total_pixels);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup_error;

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup_error;

    // Cleanup temporary buffers
    if (images_uint8_mem) cudaFree(images_uint8_mem);
    if (ref_hist_mem) cudaFree(ref_hist_mem);
    cudaFree(partial_hist_mem);
    cudaFree(hist_u32_mem);
    cudaFree(ref_cdf_mem);
    cudaFree(lut_mem);

    // Return output info as dict
    py::dict result;
    result["ptr"] = reinterpret_cast<uintptr_t>(output_mem);
    result["shape"] = py::make_tuple(N, C, H, W);
    result["dtype"] = "float32";
    result["needs_free"] = true;
    result["was_uint8"] = was_uint8_or_high_range && !needs_scale_back;
    result["original_dtype"] = is_uint8 ? "uint8" : "float32";
    
    return result;

cleanup_error:
    if (images_uint8_mem) cudaFree(images_uint8_mem);
    if (ref_hist_mem) cudaFree(ref_hist_mem);
    if (output_mem) cudaFree(output_mem);
    if (partial_hist_mem) cudaFree(partial_hist_mem);
    if (hist_u32_mem) cudaFree(hist_u32_mem);
    if (ref_cdf_mem) cudaFree(ref_cdf_mem);
    if (lut_mem) cudaFree(lut_mem);
    throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
}
