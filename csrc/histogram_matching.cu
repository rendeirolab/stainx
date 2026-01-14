// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Histogram matching CUDA kernel implementation.
 *
 * This file contains pure CUDA kernels for histogram matching stain normalization.
 * These kernels have no PyTorch dependencies and can be used by any CUDA interface.
 */

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
