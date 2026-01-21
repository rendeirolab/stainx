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

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
// Histogram matching parameters
#define HM_NUM_BINS 256

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

// Build partial histograms for each channel.
// Grid: dim3(blocks_per_channel, C, 1)
// partial_hist layout: [C, blocks_per_channel, 256] (uint32)
__global__ void hm_partial_hist_kernel(const uint8_t* images_uint8,
                                      uint32_t* partial_hist,
                                      int N,
                                      int C,
                                      int H,
                                      int W,
                                      int blocks_per_channel) {
    const int channel_idx = blockIdx.y;
    const int block_in_channel = blockIdx.x;

    const int pixels_per_image = H * W;
    const int stride_between_images = C * pixels_per_image;
    const int channel_base_offset = channel_idx * pixels_per_image;
    const int total_pixels_per_channel = N * pixels_per_image;

    __shared__ uint32_t hist_s[HM_NUM_BINS];
    for (int i = threadIdx.x; i < HM_NUM_BINS; i += blockDim.x) {
        hist_s[i] = 0;
    }
    __syncthreads();

    // Fixed tile size per block to control partial count and shared contention.
    // Each block covers a contiguous range of pixel indices within this channel.
    const int pixels_per_block = 4096;  // tuneable (16 pixels/thread at 256 threads)
    const int tile_start = block_in_channel * pixels_per_block;
    const int tile_end = min(tile_start + pixels_per_block, total_pixels_per_channel);

    for (int pixel_idx = tile_start + threadIdx.x; pixel_idx < tile_end; pixel_idx += blockDim.x) {
        const int n = pixel_idx / pixels_per_image;
        const int pixel_in_image = pixel_idx - n * pixels_per_image;
        const int mem_offset = n * stride_between_images + channel_base_offset + pixel_in_image;
        const uint8_t v = images_uint8[mem_offset];
        atomicAdd(&hist_s[v], 1u);
    }
    __syncthreads();

    // Store shared histogram into global partial buffer.
    // Indexing: (((channel * blocks_per_channel) + block) * 256 + bin)
    const int base = (channel_idx * blocks_per_channel + block_in_channel) * HM_NUM_BINS;
    for (int i = threadIdx.x; i < HM_NUM_BINS; i += blockDim.x) {
        partial_hist[base + i] = hist_s[i];
    }
}

// Reduce partial histograms into final per-channel histogram.
// Grid: dim3(1, C, 1), block: 256 threads
// hist_out layout: [C, 256] (uint32)
__global__ void hm_reduce_hist_kernel(const uint32_t* partial_hist, uint32_t* hist_out, int C, int blocks_per_channel) {
    const int channel_idx = blockIdx.y;
    const int bin = threadIdx.x;
    if (bin >= HM_NUM_BINS) return;

    uint32_t sum = 0;
    const int base_channel = channel_idx * blocks_per_channel * HM_NUM_BINS;
    for (int b = 0; b < blocks_per_channel; ++b) {
        sum += partial_hist[base_channel + b * HM_NUM_BINS + bin];
    }
    hist_out[channel_idx * HM_NUM_BINS + bin] = sum;
}

// Compute per-channel reference CDF from reference histogram.
// Supports:
// - per_channel_histograms=false: ref_hist is [256]
// - per_channel_histograms=true:  ref_hist is [C, 256]
// Output: ref_cdf [C, 256]
// Grid: dim3(C, 1, 1), block: 256 threads
__global__ void hm_ref_cdf_kernel(const float* ref_hist, float* ref_cdf, int C, bool per_channel_histograms) {
    const int channel_idx = blockIdx.x;
    const int bin = threadIdx.x;

    // Shared buffer for the normalized histogram (float) and cdf (float).
    __shared__ float hist_s[HM_NUM_BINS];
    __shared__ float cdf_s[HM_NUM_BINS];
    __shared__ float sum_s;

    // Load histogram (or broadcast single histogram)
    if (bin < HM_NUM_BINS) {
        const int src_base = per_channel_histograms ? (channel_idx * HM_NUM_BINS) : 0;
        hist_s[bin] = ref_hist[src_base + bin];
    }
    __syncthreads();

    if (bin == 0) {
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < HM_NUM_BINS; ++i) s += hist_s[i];
        sum_s = s;
    }
    __syncthreads();

    if (bin < HM_NUM_BINS) {
        hist_s[bin] = hist_s[bin] / (sum_s + 1e-8f);
    }
    __syncthreads();

    if (bin == 0) {
        float run = 0.0f;
        #pragma unroll
        for (int i = 0; i < HM_NUM_BINS; ++i) {
            run += hist_s[i];
            cdf_s[i] = run;
        }
    }
    __syncthreads();

    if (bin < HM_NUM_BINS) {
        ref_cdf[channel_idx * HM_NUM_BINS + bin] = cdf_s[bin];
    }
}

// Build LUT for each channel based on source histogram and reference CDF.
// Inputs:
// - hist_u32: [C, 256] source histogram counts
// - ref_cdf:  [C, 256] reference CDF
// Output:
// - lut:      [C, 256] float mapping for each uint8 value
// Grid: dim3(C, 1, 1), block: 256 threads
__global__ void hm_build_lut_kernel(const uint32_t* hist_u32, const float* ref_cdf, float* lut, int N, int H, int W) {
    const int channel_idx = blockIdx.x;
    const int bin = threadIdx.x;

    __shared__ float source_cdf_s[HM_NUM_BINS];
    __shared__ float ref_cdf_s[HM_NUM_BINS];
    __shared__ float ref_min_s;
    __shared__ float ref_max_s;

    // Load ref CDF to shared
    if (bin < HM_NUM_BINS) {
        ref_cdf_s[bin] = ref_cdf[channel_idx * HM_NUM_BINS + bin];
    }
    __syncthreads();

    if (bin == 0) {
        ref_min_s = ref_cdf_s[0];
        ref_max_s = ref_cdf_s[HM_NUM_BINS - 1];
    }
    __syncthreads();

    // Compute source CDF (sequential in one thread; 256 bins only)
    if (bin == 0) {
        const float inv_total = 1.0f / (float)(N * H * W + 1e-8f);
        float run = 0.0f;
        #pragma unroll
        for (int i = 0; i < HM_NUM_BINS; ++i) {
            run += (float)hist_u32[channel_idx * HM_NUM_BINS + i] * inv_total;
            source_cdf_s[i] = run;
        }
    }
    __syncthreads();

    // Build LUT in parallel across bins (binary search in ref CDF)
    if (bin < HM_NUM_BINS) {
        const float source_q = source_cdf_s[bin];
        const float ref_min = ref_min_s;
        const float ref_max = ref_max_s;

        int left = 0;
        int right = HM_NUM_BINS;
        while (left < right) {
            const int mid = (left + right) >> 1;
            if (ref_cdf_s[mid] < source_q) left = mid + 1;
            else right = mid;
        }

        int idx = max(1, min(left, HM_NUM_BINS - 1));
        const float ql = ref_cdf_s[idx - 1];
        const float qr = ref_cdf_s[idx];
        const float diff = qr - ql;
        float alpha = 0.0f;
        if (diff > 1e-10f) alpha = (source_q - ql) / diff;
        float matched = (float)(idx - 1) + alpha;

        if (source_q <= ref_min) matched = 0.0f;
        if (source_q >= ref_max) matched = (float)(HM_NUM_BINS - 1);

        lut[channel_idx * HM_NUM_BINS + bin] = fmaxf(0.0f, fminf(255.0f, matched));
    }
}

// Apply LUT to all pixels (NCHW contiguous).
// Grid: 1D over total pixels (N*C*H*W)
__global__ void hm_apply_lut_kernel(const uint8_t* images_uint8, float* output, const float* lut, int N, int C, int H, int W) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (idx >= total) return;

    const int pixels_per_image = H * W;
    const int pixels_per_channel = pixels_per_image;
    const int idx_in_image = idx % (C * pixels_per_channel);
    const int channel = idx_in_image / pixels_per_channel;

    const uint8_t v = images_uint8[idx];
    output[idx] = lut[channel * HM_NUM_BINS + (int)v];
}
