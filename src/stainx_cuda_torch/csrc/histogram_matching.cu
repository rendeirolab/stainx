// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * PyTorch wrapper for histogram matching CUDA kernels.
 *
 * This file provides PyTorch tensor interfaces that call the pure CUDA kernels
 * from the main csrc/ directory.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

// Include the pure CUDA kernels from the main csrc directory
// Project root is in include_dirs, so we can include from csrc/
#include "csrc/histogram_matching.cu"

#define THREADS_PER_BLOCK 256

torch::Tensor histogram_matching_cuda(torch::Tensor input_images, torch::Tensor reference_histogram) {
    // Check inputs
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(reference_histogram.is_cuda(), "reference_histogram must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W), got ", input_images.dim(), "D tensor with shape ", input_images.sizes());

    // Check that tensors are on the same device
    TORCH_CHECK(input_images.device() == reference_histogram.device(),
                "input_images and reference_histogram must be on the same device. "
                "input_images device: ",
                input_images.device(),
                ", reference_histogram device: ",
                reference_histogram.device());

    // Get device and stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Track original dtype for output preservation
    torch::ScalarType original_dtype = input_images.scalar_type();
    bool needs_scale_back            = false;
    bool was_uint8_or_high_range     = false;

    int N                        = input_images.size(0);
    int C                        = input_images.size(1);
    int H                        = input_images.size(2);
    int W                        = input_images.size(3);
    int num_bins                 = 256;
    int total_pixels_per_channel = N * H * W;
    int num_threads              = THREADS_PER_BLOCK;
    cudaError_t err              = cudaSuccess;  // Declare once at function level

    // Check input range and convert to uint8 if needed
    torch::Tensor images_uint8;
    if (input_images.dtype() == torch::kUInt8) {
        images_uint8            = input_images.contiguous();
        was_uint8_or_high_range = true;
    } else {
        // Check if input is in [0, 1] range using PyTorch's max operation (safer than CUB)
        // This avoids potential memory access issues with CUB DeviceReduce
        float max_val = input_images.max().item<float>();

        // Determine if we need to scale
        if (max_val <= 1.0f) {
            needs_scale_back = true;
        } else {
            was_uint8_or_high_range = true;
        }

        // Convert to uint8 using CUDA kernel
        // Ensure input is contiguous for kernel access
        torch::Tensor input_flat = input_images.contiguous().view(-1);
        images_uint8             = torch::empty({N, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8).device(input_images.device()));
        int total_pixels         = N * C * H * W;
        int num_blocks           = (total_pixels + num_threads - 1) / num_threads;

        convert_to_uint8_kernel<<<num_blocks, num_threads, 0, stream>>>(input_flat.data_ptr<float>(), images_uint8.data_ptr<uint8_t>(), needs_scale_back, total_pixels);
        err = cudaGetLastError();
        if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in convert_to_uint8_kernel: ", cudaGetErrorString(err)); }
    }

    // Prepare reference histogram(s) - convert to float32 and make contiguous
    torch::Tensor ref_hist = reference_histogram.contiguous();
    if (ref_hist.dtype() != torch::kFloat32) {
        // Convert to float32 while preserving device
        ref_hist = ref_hist.to(torch::kFloat32).contiguous();
        // Ensure it's still on CUDA after conversion
        TORCH_CHECK(ref_hist.is_cuda(), "reference_histogram must remain on CUDA device after dtype conversion");
    }

    bool per_channel_histograms = false;
    if (ref_hist.dim() == 1 && ref_hist.size(0) == num_bins) {
        // Single histogram for all channels
        per_channel_histograms = false;
    } else if (ref_hist.dim() == 2 && ref_hist.size(0) == C && ref_hist.size(1) == num_bins) {
        // Per-channel histograms: (C, 256)
        per_channel_histograms = true;
    } else {
        TORCH_CHECK(false,
                    "reference_histogram must be 1D with 256 elements or 2D with shape (C, 256), "
                    "where C is the number of channels in input_images. "
                    "Got reference_histogram with shape ",
                    ref_hist.sizes(),
                    " but input_images has ",
                    C,
                    " channels (shape: ",
                    input_images.sizes(),
                    ")");
    }

    // Allocate output tensor
    torch::Tensor output = torch::empty({N, C, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(input_images.device()));

    // Choose blocks per channel based on workload (kernel uses pixels_per_block=4096).
    // Cap to avoid excessive temporary memory for very large images.
    const int pixels_per_block = 4096;
    int blocks_per_channel = (total_pixels_per_channel + pixels_per_block - 1) / pixels_per_block;
    if (blocks_per_channel < 1) blocks_per_channel = 1;
    if (blocks_per_channel > 2048) blocks_per_channel = 2048;

    // Temporary buffers
    // partial_hist: [C, blocks_per_channel, 256] int32 (used as uint32)
    torch::Tensor partial_hist = torch::empty({C, blocks_per_channel, num_bins}, torch::TensorOptions().dtype(torch::kInt32).device(input_images.device()));
    // hist_u32: [C, 256] int32 (used as uint32)
    torch::Tensor hist_u32 = torch::empty({C, num_bins}, torch::TensorOptions().dtype(torch::kInt32).device(input_images.device()));
    // ref_cdf: [C, 256] float32
    torch::Tensor ref_cdf = torch::empty({C, num_bins}, torch::TensorOptions().dtype(torch::kFloat32).device(input_images.device()));
    // lut: [C, 256] float32
    torch::Tensor lut = torch::empty({C, num_bins}, torch::TensorOptions().dtype(torch::kFloat32).device(input_images.device()));

    // 1) Partial histograms (many blocks per channel)
    dim3 grid_partial(blocks_per_channel, C, 1);
    hm_partial_hist_kernel<<<grid_partial, num_threads, 0, stream>>>(images_uint8.data_ptr<uint8_t>(),
                                                                     reinterpret_cast<uint32_t*>(partial_hist.data_ptr<int32_t>()),
                                                                     N,
                                                                     C,
                                                                     H,
                                                                     W,
                                                                     blocks_per_channel);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in hm_partial_hist_kernel: ", cudaGetErrorString(err)); }

    // 2) Reduce partials -> final histogram per channel
    dim3 grid_reduce(1, C, 1);
    hm_reduce_hist_kernel<<<grid_reduce, num_threads, 0, stream>>>(reinterpret_cast<const uint32_t*>(partial_hist.data_ptr<int32_t>()),
                                                                   reinterpret_cast<uint32_t*>(hist_u32.data_ptr<int32_t>()),
                                                                   C,
                                                                   blocks_per_channel);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in hm_reduce_hist_kernel: ", cudaGetErrorString(err)); }

    // 3) Compute reference CDF per channel (single hist is broadcast)
    hm_ref_cdf_kernel<<<C, num_threads, 0, stream>>>(ref_hist.data_ptr<float>(), ref_cdf.data_ptr<float>(), C, per_channel_histograms);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in hm_ref_cdf_kernel: ", cudaGetErrorString(err)); }

    // 4) Build LUT per channel from source hist + reference CDF
    hm_build_lut_kernel<<<C, num_threads, 0, stream>>>(reinterpret_cast<const uint32_t*>(hist_u32.data_ptr<int32_t>()),
                                                       ref_cdf.data_ptr<float>(),
                                                       lut.data_ptr<float>(),
                                                       N,
                                                       H,
                                                       W);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in hm_build_lut_kernel: ", cudaGetErrorString(err)); }

    // 5) Apply LUT across all pixels
    int total_pixels = N * C * H * W;
    int num_blocks   = (total_pixels + num_threads - 1) / num_threads;
    hm_apply_lut_kernel<<<num_blocks, num_threads, 0, stream>>>(images_uint8.data_ptr<uint8_t>(),
                                                                output.data_ptr<float>(),
                                                                lut.data_ptr<float>(),
                                                                N,
                                                                C,
                                                                H,
                                                                W);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in hm_apply_lut_kernel: ", cudaGetErrorString(err)); }

    // Scale and clamp output
    scale_clamp_output_kernel<<<num_blocks, num_threads, 0, stream>>>(output.data_ptr<float>(), needs_scale_back, false, total_pixels);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in scale_clamp_output_kernel: ", cudaGetErrorString(err)); }

    // Preserve original dtype (matching PyTorch backend logic)
    if (was_uint8_or_high_range && !needs_scale_back) {
        // Output is in [0, 255] range, convert to original dtype
        if (original_dtype == torch::kUInt8) { output = output.to(torch::kUInt8); }
    }

    return output;
}
