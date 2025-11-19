// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Histogram matching CUDA kernel implementation.
 *
 * This file contains CUDA kernels for histogram matching stain normalization.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Kernel to convert input to uint8 (handles float input, scaling from [0,1] to [0,255])
__global__ void convert_to_uint8_kernel(const float* input, uint8_t* output, bool scale_from_01, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        float val = input[idx];
        if (scale_from_01) { val = val * 255.0f; }
        val         = fmaxf(0.0f, fminf(255.0f, val));
        output[idx] = (uint8_t) val;
    }
}

// Kernel to compute histogram using atomic operations
__global__ void compute_histogram_kernel(const uint8_t* input, float* histogram, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        uint8_t value = input[idx];
        atomicAdd(&histogram[value], 1.0f);
    }
}

// Kernel to normalize histogram (divide by sum)
__global__ void normalize_histogram_kernel(float* histogram, float sum, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bins) { histogram[idx] = histogram[idx] / (sum + 1e-8f); }
}

// Kernel to build lookup table from source and reference CDFs
// This matches the PyTorch implementation using searchsorted-like logic
__global__ void build_lookup_table_kernel(const float* source_cdf, const float* ref_cdf, float* lookup_table, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bins) {
        float source_quantile = source_cdf[idx];
        float ref_min         = ref_cdf[0];
        float ref_max         = ref_cdf[num_bins - 1];

        // Binary search to find position (similar to searchsorted)
        int left  = 0;
        int right = num_bins;
        int mid;

        while (left < right) {
            mid = (left + right) / 2;
            if (ref_cdf[mid] < source_quantile) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Clamp indices
        int indices = max(1, min(left, num_bins - 1));

        // Get quantiles for interpolation
        float quantile_left  = ref_cdf[indices - 1];
        float quantile_right = ref_cdf[indices];
        float quantile_diff  = quantile_right - quantile_left;

        // Compute alpha for interpolation
        float alpha = 0.0f;
        if (quantile_diff > 1e-10f) { alpha = (source_quantile - quantile_left) / quantile_diff; }

        // Interpolate values (ref_values is just 0..255, so we use indices directly)
        float matched_value = (indices - 1) + alpha;

        // Handle edge cases (matching PyTorch backend: apply below_min first, then above_max)
        bool below_min = (source_quantile <= ref_min);
        bool above_max = (source_quantile >= ref_max);

        // Apply edge case handling in same order as PyTorch: below_min first, then above_max
        if (below_min) { matched_value = 0.0f; }
        if (above_max) { matched_value = (float) (num_bins - 1); }

        lookup_table[idx] = fmaxf(0.0f, fminf(255.0f, matched_value));
    }
}

// Kernel to apply lookup table to input image
__global__ void apply_lookup_table_kernel(const uint8_t* input, const float* lookup_table, float* output, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        uint8_t pixel_value = input[idx];
        output[idx]         = lookup_table[pixel_value];
    }
}

// Kernel to zero out a buffer
__global__ void zero_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { data[idx] = 0.0f; }
}

// Kernel to scale and clamp output
__global__ void scale_clamp_output_kernel(float* output, bool scale_to_01, bool convert_to_uint8, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        float val = output[idx];
        if (scale_to_01) {
            val = val / 255.0f;
            val = fmaxf(0.0f, fminf(1.0f, val));
        } else {
            val = fmaxf(0.0f, fminf(255.0f, val));
        }
        output[idx] = val;
    }
}

// Fused kernel to process a single channel (one block per channel)
// Each block processes one channel completely: histogram -> normalize -> CDF -> lookup -> apply
__global__ void process_channel_kernel(const uint8_t* images_uint8,  // Input images (N*C*H*W, channels interleaved)
                                       float* output,                // Output images (N*C*H*W, channels interleaved)
                                       const float* ref_hist,        // Reference histogram(s): (C, 256) or (256,)
                                       float* channel_buffers,       // Per-channel buffers: (C, 4*num_bins) for hist, cdf, ref_cdf, lookup
                                       int N,
                                       int C,
                                       int H,
                                       int W,                         // Image dimensions
                                       int num_bins,                  // Number of histogram bins (256)
                                       int total_pixels_per_channel,  // N * H * W
                                       bool per_channel_histograms    // Whether ref_hist is per-channel
) {
    // Each block processes one channel
    int channel_idx = blockIdx.x;
    if (channel_idx >= C) return;

    // In (N, C, H, W) format, channels are interleaved across images
    // For channel c, data is at: n * (C*H*W) + c * (H*W) + h * W + w
    // So we need to stride by C*H*W between images, and start at c*H*W
    int pixels_per_image      = H * W;
    int stride_between_images = C * pixels_per_image;            // C * H * W
    int channel_base_offset   = channel_idx * pixels_per_image;  // c * H * W

    // Get pointers to this channel's buffers
    int buffer_offset      = channel_idx * 4 * num_bins;
    float* source_hist     = channel_buffers + buffer_offset;
    float* source_cdf      = channel_buffers + buffer_offset + num_bins;
    float* ref_cdf_channel = channel_buffers + buffer_offset + 2 * num_bins;  // Temporary for per-channel case
    float* lookup_table    = channel_buffers + buffer_offset + 3 * num_bins;

    // Get reference CDF pointer
    const float* ref_cdf_ptr;
    if (per_channel_histograms) {
        // Per-channel reference histogram - need to compute CDF
        // Use source_hist temporarily for ref_hist_channel (we'll overwrite it later)
        float* ref_hist_channel = source_hist;  // Temporary, will be overwritten

        // Copy reference histogram for this channel
        const float* ref_hist_src = ref_hist + channel_idx * num_bins;
        for (int i = threadIdx.x; i < num_bins; i += blockDim.x) { ref_hist_channel[i] = ref_hist_src[i]; }
        __syncthreads();

        // Compute sum of reference histogram (sequential for correctness with 256 elements)
        // For small arrays, sequential is fast and guaranteed correct
        __shared__ float ref_sum_shared;
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int i = 0; i < num_bins; i++) { sum += ref_hist_channel[i]; }
            ref_sum_shared = sum;
        }
        __syncthreads();
        float ref_sum = ref_sum_shared;

        // Normalize reference histogram
        float inv_sum = 1.0f / (ref_sum + 1e-8f);
        for (int i = threadIdx.x; i < num_bins; i += blockDim.x) { ref_hist_channel[i] *= inv_sum; }
        __syncthreads();

        // Compute reference CDF (prefix sum - sequential for correctness)
        // For 256 elements, sequential is fast and guaranteed correct
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int i = 0; i < num_bins; i++) {
                sum += ref_hist_channel[i];
                ref_cdf_channel[i] = sum;
            }
        }
        __syncthreads();
        ref_cdf_ptr = ref_cdf_channel;

        // Now zero out source histogram (ref_hist_channel was using this space)
    } else {
        // Single reference histogram - already computed, just get pointer
        ref_cdf_ptr = ref_hist;  // In this case, ref_hist is actually the pre-computed CDF
    }

    // Zero out source histogram
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) { source_hist[i] = 0.0f; }
    __syncthreads();

    // Compute histogram using atomic operations
    // Access channel data with proper striding for (N, C, H, W) layout
    int num_pixels = total_pixels_per_channel;
    for (int pixel_idx = threadIdx.x; pixel_idx < num_pixels; pixel_idx += blockDim.x) {
        // Calculate which image and pixel within that image
        int n              = pixel_idx / pixels_per_image;  // Image index
        int pixel_in_image = pixel_idx % pixels_per_image;  // Pixel index within image
        // Calculate actual memory offset: n * (C*H*W) + c * (H*W) + pixel_in_image
        int mem_offset = n * stride_between_images + channel_base_offset + pixel_in_image;
        uint8_t value  = images_uint8[mem_offset];
        atomicAdd(&source_hist[value], 1.0f);
    }
    __syncthreads();

    // Normalize source histogram
    float total_sum = (float) total_pixels_per_channel;
    float inv_total = 1.0f / (total_sum + 1e-8f);
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) { source_hist[i] *= inv_total; }
    __syncthreads();

    // Compute source CDF (prefix sum - sequential for correctness)
    // For 256 elements, sequential is fast and guaranteed correct
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < num_bins; i++) {
            sum += source_hist[i];
            source_cdf[i] = sum;
        }
    }
    __syncthreads();

    // Build lookup table
    for (int idx = threadIdx.x; idx < num_bins; idx += blockDim.x) {
        float source_quantile = source_cdf[idx];
        float ref_min         = ref_cdf_ptr[0];
        float ref_max         = ref_cdf_ptr[num_bins - 1];

        // Binary search to find position
        int left  = 0;
        int right = num_bins;
        int mid;

        while (left < right) {
            mid = (left + right) / 2;
            if (ref_cdf_ptr[mid] < source_quantile) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Clamp indices
        int indices = max(1, min(left, num_bins - 1));

        // Get quantiles for interpolation
        float quantile_left  = ref_cdf_ptr[indices - 1];
        float quantile_right = ref_cdf_ptr[indices];
        float quantile_diff  = quantile_right - quantile_left;

        // Compute alpha for interpolation
        float alpha = 0.0f;
        if (quantile_diff > 1e-10f) { alpha = (source_quantile - quantile_left) / quantile_diff; }

        // Interpolate values
        float matched_value = (indices - 1) + alpha;

        // Handle edge cases
        bool below_min = (source_quantile <= ref_min);
        bool above_max = (source_quantile >= ref_max);

        if (below_min) { matched_value = 0.0f; }
        if (above_max) { matched_value = (float) (num_bins - 1); }

        lookup_table[idx] = fmaxf(0.0f, fminf(255.0f, matched_value));
    }
    __syncthreads();

    // Apply lookup table to output
    // Access channel data with proper striding for (N, C, H, W) layout
    for (int pixel_idx = threadIdx.x; pixel_idx < num_pixels; pixel_idx += blockDim.x) {
        // Calculate which image and pixel within that image
        int n              = pixel_idx / pixels_per_image;  // Image index
        int pixel_in_image = pixel_idx % pixels_per_image;  // Pixel index within image
        // Calculate actual memory offset: n * (C*H*W) + c * (H*W) + pixel_in_image
        int mem_offset      = n * stride_between_images + channel_base_offset + pixel_in_image;
        uint8_t pixel_value = images_uint8[mem_offset];
        output[mem_offset]  = lookup_table[pixel_value];
    }
}

torch::Tensor histogram_matching_cuda(torch::Tensor input_images, torch::Tensor reference_histogram) {
    // Check inputs
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(reference_histogram.is_cuda(), "reference_histogram must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W)");
    // Note: We allow any number of channels at dim 1 to match PyTorch backend behavior
    // when processing corrupted formats from prepare_for_normalizer

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

    // Check input range and convert to uint8 if needed
    torch::Tensor images_uint8;
    if (input_images.dtype() == torch::kUInt8) {
        images_uint8            = input_images.contiguous();
        was_uint8_or_high_range = true;
    } else {
        // Check if input is in [0, 1] range using CUB max reduction
        torch::Tensor input_flat = input_images.contiguous().view(-1);
        int num_elements         = input_flat.numel();

        // Use CUB DeviceReduce::Max to find max value
        void* d_temp_storage      = nullptr;
        size_t temp_storage_bytes = 0;
        float max_val             = 0.0f;

        // Determine temporary storage requirements
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, input_flat.data_ptr<float>(), &max_val, num_elements, stream);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Run max reduction
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, input_flat.data_ptr<float>(), &max_val, num_elements, stream);

        // Synchronize to ensure max is computed (DeviceReduce writes to host pointer)
        cudaStreamSynchronize(stream);

        // Free temporary storage
        if (d_temp_storage != nullptr) { cudaFree(d_temp_storage); }

        // Determine if we need to scale
        if (max_val <= 1.0f) {
            needs_scale_back = true;
        } else {
            was_uint8_or_high_range = true;
        }

        // Convert to uint8 using CUDA kernel
        images_uint8     = torch::empty({N, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8).device(input_images.device()));
        int total_pixels = N * C * H * W;
        int num_blocks   = (total_pixels + num_threads - 1) / num_threads;

        convert_to_uint8_kernel<<<num_blocks, num_threads, 0, stream>>>(input_flat.data_ptr<float>(), images_uint8.data_ptr<uint8_t>(), needs_scale_back, total_pixels);
    }

    // Prepare reference histogram(s) - convert to float32 and make contiguous
    torch::Tensor ref_hist = reference_histogram.contiguous();
    if (ref_hist.dtype() != torch::kFloat32) {
        // Need to convert, but we'll do it on CPU for simplicity (small tensor)
        ref_hist = ref_hist.to(torch::kFloat32).contiguous();
    }

    bool per_channel_histograms = false;
    if (ref_hist.dim() == 1 && ref_hist.size(0) == num_bins) {
        // Single histogram for all channels
        per_channel_histograms = false;
    } else if (ref_hist.dim() == 2 && ref_hist.size(0) == C && ref_hist.size(1) == num_bins) {
        // Per-channel histograms: (C, 256)
        per_channel_histograms = true;
    } else {
        TORCH_CHECK(false, "reference_histogram must be 1D with 256 elements or 2D with shape (C, 256)");
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
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Run sum reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, ref_hist.data_ptr<float>(), &ref_sum, num_bins, stream);

        // Synchronize to ensure sum is computed (DeviceReduce writes to host pointer)
        cudaStreamSynchronize(stream);

        // Normalize histogram (copy first to avoid modifying original)
        torch::Tensor ref_hist_norm = torch::empty_like(ref_hist);
        cudaMemcpyAsync(ref_hist_norm.data_ptr<float>(), ref_hist.data_ptr<float>(), num_bins * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        int num_threads = THREADS_PER_BLOCK;
        int num_blocks  = (num_bins + num_threads - 1) / num_threads;
        normalize_histogram_kernel<<<num_blocks, num_threads, 0, stream>>>(ref_hist_norm.data_ptr<float>(), ref_sum, num_bins);

        // Compute CDF using CUB DeviceScan::InclusiveSum
        // Determine temporary storage requirements for scan
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ref_hist_norm.data_ptr<float>(), ref_cdf.data_ptr<float>(), num_bins, stream);

        // Reallocate if needed (scan might need more storage)
        cudaFree(d_temp_storage);
        if (temp_storage_bytes > 0) { cudaMalloc(&d_temp_storage, temp_storage_bytes); }

        // Run inclusive scan (prefix sum)
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ref_hist_norm.data_ptr<float>(), ref_cdf.data_ptr<float>(), num_bins, stream);

        // Free temporary storage
        if (d_temp_storage != nullptr) { cudaFree(d_temp_storage); }
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

    // Scale and clamp output
    int total_pixels = N * C * H * W;
    int num_blocks   = (total_pixels + num_threads - 1) / num_threads;
    scale_clamp_output_kernel<<<num_blocks, num_threads, 0, stream>>>(output.data_ptr<float>(), needs_scale_back, false, total_pixels);

    // Synchronize before final dtype conversion
    cudaStreamSynchronize(stream);

    // Preserve original dtype (matching PyTorch backend logic)
    if (was_uint8_or_high_range && !needs_scale_back) {
        // Output is in [0, 255] range, convert to original dtype
        if (original_dtype == torch::kUInt8) { output = output.to(torch::kUInt8); }
    }

    return output;
}
