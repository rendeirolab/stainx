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

    // Pre-compute reference CDFs on GPU using CUB
    torch::Tensor ref_cdf;
    if (!per_channel_histograms) {
        // Single reference histogram - normalize and compute CDF on GPU using CUB
        ref_cdf = torch::empty({num_bins}, torch::TensorOptions().dtype(torch::kFloat32).device(input_images.device()));

        // Compute sum using CUB DeviceReduce::Sum
        void* d_temp_storage      = nullptr;
        size_t temp_storage_bytes = 0;
        float ref_sum             = 0.0f;

        // Determine temporary storage requirements for sum
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, ref_hist.data_ptr<float>(), &ref_sum, num_bins, stream);

        // Allocate temporary storage
        err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
        if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error allocating temporary storage for reference histogram sum: ", cudaGetErrorString(err)); }

        // Run sum reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, ref_hist.data_ptr<float>(), &ref_sum, num_bins, stream);

        // Synchronize to ensure sum is computed (DeviceReduce writes to host pointer)
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            cudaFree(d_temp_storage);
            TORCH_CHECK(false, "CUDA error synchronizing stream after reference histogram sum: ", cudaGetErrorString(err));
        }

        // Normalize histogram (copy first to avoid modifying original)
        torch::Tensor ref_hist_norm = torch::empty_like(ref_hist);
        err                         = cudaMemcpyAsync(ref_hist_norm.data_ptr<float>(), ref_hist.data_ptr<float>(), num_bins * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            cudaFree(d_temp_storage);
            TORCH_CHECK(false, "CUDA error copying reference histogram: ", cudaGetErrorString(err));
        }
        int num_threads = THREADS_PER_BLOCK;
        int num_blocks  = (num_bins + num_threads - 1) / num_threads;
        normalize_histogram_kernel<<<num_blocks, num_threads, 0, stream>>>(ref_hist_norm.data_ptr<float>(), ref_sum, num_bins);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_temp_storage);
            TORCH_CHECK(false, "CUDA error in normalize_histogram_kernel: ", cudaGetErrorString(err));
        }

        // Compute CDF using CUB DeviceScan::InclusiveSum
        // Determine temporary storage requirements for scan
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ref_hist_norm.data_ptr<float>(), ref_cdf.data_ptr<float>(), num_bins, stream);

        // Reallocate if needed (scan might need more storage)
        err = cudaFree(d_temp_storage);
        if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error freeing temporary storage before scan: ", cudaGetErrorString(err)); }
        if (temp_storage_bytes > 0) {
            err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
            if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error allocating temporary storage for scan: ", cudaGetErrorString(err)); }
        }

        // Run inclusive scan (prefix sum)
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ref_hist_norm.data_ptr<float>(), ref_cdf.data_ptr<float>(), num_bins, stream);

        // Free temporary storage
        if (d_temp_storage != nullptr) {
            err = cudaFree(d_temp_storage);
            if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error freeing temporary storage after scan: ", cudaGetErrorString(err)); }
        }
    }

    // Allocate output tensor
    torch::Tensor output = torch::empty({N, C, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(input_images.device()));

    // Pre-allocate per-channel buffers for parallel processing
    // Each channel needs: source_hist (num_bins), source_cdf (num_bins), ref_cdf_channel (num_bins, for per-channel case), lookup_table (num_bins)
    // Total: C * 4 * num_bins

    // Per-channel buffers: (C, 4*num_bins) - for hist, cdf, ref_cdf (temp), lookup_table
    torch::Tensor channel_buffers = torch::empty({C, 4 * num_bins}, torch::TensorOptions().dtype(torch::kFloat32).device(input_images.device()));

    // Prepare reference data pointer
    const float* ref_data_ptr;
    if (!per_channel_histograms) {
        // Single histogram - use pre-computed CDF
        ref_data_ptr = ref_cdf.data_ptr<float>();
    } else {
        // Per-channel histograms - pass the histogram data
        ref_data_ptr = ref_hist.data_ptr<float>();
    }

    // Launch parallel kernel: one block per channel
    // Minimal shared memory needed (just for sum broadcast)
    size_t shared_mem_size = sizeof(float);  // For ref_sum_shared

    int threads_per_block = THREADS_PER_BLOCK;

    process_channel_kernel<<<C, threads_per_block, shared_mem_size, stream>>>(images_uint8.data_ptr<uint8_t>(), output.data_ptr<float>(), ref_data_ptr, channel_buffers.data_ptr<float>(), N, C, H, W, num_bins, total_pixels_per_channel, per_channel_histograms);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false,
                    "CUDA error in process_channel_kernel: ",
                    cudaGetErrorString(err),
                    ". This may indicate invalid input shapes, out-of-bounds memory access, or device mismatch. "
                    "Input shape: (",
                    N,
                    ", ",
                    C,
                    ", ",
                    H,
                    ", ",
                    W,
                    "), "
                    "Reference histogram shape: ",
                    reference_histogram.sizes(),
                    ", per_channel_histograms: ",
                    per_channel_histograms);
    }

    // Scale and clamp output
    int total_pixels = N * C * H * W;
    int num_blocks   = (total_pixels + num_threads - 1) / num_threads;
    scale_clamp_output_kernel<<<num_blocks, num_threads, 0, stream>>>(output.data_ptr<float>(), needs_scale_back, false, total_pixels);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in scale_clamp_output_kernel: ", cudaGetErrorString(err)); }

    // Synchronize before final dtype conversion
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing stream: ", cudaGetErrorString(err)); }

    // Preserve original dtype (matching PyTorch backend logic)
    if (was_uint8_or_high_range && !needs_scale_back) {
        // Output is in [0, 255] range, convert to original dtype
        if (original_dtype == torch::kUInt8) { output = output.to(torch::kUInt8); }
    }

    return output;
}
