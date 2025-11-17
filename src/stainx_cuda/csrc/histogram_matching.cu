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

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Kernel to compute histogram using atomic operations
__global__ void compute_histogram_kernel(const uint8_t* input, float* histogram, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        uint8_t value = input[idx];
        atomicAdd(&histogram[value], 1.0f);
    }
}

// Kernel to build lookup table from source and reference CDFs
// This matches the PyTorch implementation using searchsorted-like logic
__global__ void build_lookup_table_kernel(const float* source_cdf, const float* ref_cdf, const float* ref_values, float* lookup_table, int num_bins) {
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

        // Interpolate values
        float matched_value = ref_values[indices - 1] + alpha * (ref_values[indices] - ref_values[indices - 1]);

        // Handle edge cases (matching PyTorch backend: apply below_min first, then above_max)
        bool below_min = (source_quantile <= ref_min);
        bool above_max = (source_quantile >= ref_max);

        // Apply edge case handling in same order as PyTorch: below_min first, then above_max
        if (below_min) { matched_value = ref_values[0]; }
        if (above_max) { matched_value = ref_values[num_bins - 1]; }

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

torch::Tensor histogram_matching_cuda(torch::Tensor input_images, torch::Tensor reference_histogram) {
    // Check inputs
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(reference_histogram.is_cuda(), "reference_histogram must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W)");
    // Note: We allow any number of channels at dim 1 to match PyTorch backend behavior
    // when processing corrupted formats from prepare_for_normalizer

    // Get device and stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Track original dtype and range for output preservation
    torch::ScalarType original_dtype = input_images.scalar_type();
    bool was_uint8_or_high_range     = (original_dtype == torch::kUInt8) || (input_images.max().item<float>() > 1.0f);
    bool needs_scale_back            = false;

    // Ensure input is uint8 and channels-first format (N, C, H, W)
    torch::Tensor images_uint8;
    if (input_images.dtype() == torch::kUInt8) {
        images_uint8 = input_images.contiguous();
    } else {
        // Check if input is in [0, 1] range
        if (input_images.max().item<float>() <= 1.0f) {
            // Input is in [0, 1] range, scale to [0, 255] for processing
            images_uint8     = (input_images * 255.0f).clamp(0.0f, 255.0f).to(torch::kUInt8).contiguous();
            needs_scale_back = true;
        } else {
            // Input is already in [0, 255] range
            images_uint8 = input_images.clamp(0.0f, 255.0f).to(torch::kUInt8).contiguous();
        }
    }

    int N        = images_uint8.size(0);
    int C        = images_uint8.size(1);
    int H        = images_uint8.size(2);
    int W        = images_uint8.size(3);
    int num_bins = 256;

    // Prepare reference histogram(s)
    // Accept either: (256,) single histogram or (C, 256) per-channel histograms
    torch::Tensor ref_hist      = reference_histogram.contiguous().to(torch::kFloat32);
    bool per_channel_histograms = false;
    torch::Tensor ref_cdf;

    if (ref_hist.dim() == 1 && ref_hist.size(0) == num_bins) {
        // Single histogram for all channels
        float ref_sum               = ref_hist.sum().item<float>();
        torch::Tensor ref_hist_norm = ref_hist / (ref_sum + 1e-8f);
        ref_cdf                     = ref_hist_norm.cumsum(0);
    } else if (ref_hist.dim() == 2 && ref_hist.size(0) == C && ref_hist.size(1) == num_bins) {
        // Per-channel histograms: (C, 256)
        per_channel_histograms = true;
        // Will compute per-channel CDFs in the loop
    } else {
        TORCH_CHECK(false, "reference_histogram must be 1D with 256 elements or 2D with shape (C, 256)");
    }

    // Allocate output tensor
    torch::Tensor output = torch::empty_like(images_uint8, torch::kFloat32);

    // Process each channel
    for (int c = 0; c < C; c++) {
        // Extract channel
        torch::Tensor channel      = images_uint8.select(1, c).contiguous();  // (N, H, W)
        torch::Tensor channel_flat = channel.reshape(-1);                     // (N*H*W,)

        // Allocate histogram
        torch::Tensor source_hist = torch::zeros({num_bins}, torch::TensorOptions().dtype(torch::kFloat32).device(images_uint8.device()));

        // Compute histogram
        int num_threads = THREADS_PER_BLOCK;
        int num_blocks  = (channel_flat.numel() + num_threads - 1) / num_threads;

        compute_histogram_kernel<<<num_blocks, num_threads, 0, stream>>>(channel_flat.data_ptr<uint8_t>(), source_hist.data_ptr<float>(), channel_flat.numel());

        // Normalize histogram
        float total_pixels             = channel_flat.numel();
        torch::Tensor source_hist_norm = source_hist / (total_pixels + 1e-8f);

        // Compute CDF using cumsum
        torch::Tensor source_cdf = source_hist_norm.cumsum(0);

        // Get reference CDF for this channel
        torch::Tensor ref_cdf_channel;
        if (per_channel_histograms) {
            // Use per-channel histogram
            torch::Tensor ref_hist_channel = ref_hist[c];  // (256,)
            float ref_sum                  = ref_hist_channel.sum().item<float>();
            torch::Tensor ref_hist_norm    = ref_hist_channel / (ref_sum + 1e-8f);
            ref_cdf_channel                = ref_hist_norm.cumsum(0);
        } else {
            // Use single histogram for all channels
            ref_cdf_channel = ref_cdf;
        }

        // Build lookup table
        // Create ref_values tensor (0 to 255)
        torch::Tensor ref_values = torch::arange(num_bins, torch::TensorOptions().dtype(torch::kFloat32).device(images_uint8.device()));

        torch::Tensor lookup_table = torch::empty({num_bins}, torch::TensorOptions().dtype(torch::kFloat32).device(images_uint8.device()));

        num_blocks = (num_bins + num_threads - 1) / num_threads;
        build_lookup_table_kernel<<<num_blocks, num_threads, 0, stream>>>(source_cdf.data_ptr<float>(), ref_cdf_channel.data_ptr<float>(), ref_values.data_ptr<float>(), lookup_table.data_ptr<float>(), num_bins);

        // Apply lookup table
        torch::Tensor channel_output = output.select(1, c).reshape(-1);  // (N*H*W,)
        num_blocks                   = (channel_flat.numel() + num_threads - 1) / num_threads;

        apply_lookup_table_kernel<<<num_blocks, num_threads, 0, stream>>>(channel_flat.data_ptr<uint8_t>(), lookup_table.data_ptr<float>(), channel_output.data_ptr<float>(), channel_flat.numel());
    }

    // Reshape output back to (N, C, H, W)
    output = output.view({N, C, H, W});

    // Scale back to [0, 1] if input was in [0, 1] range (matching PyTorch backend logic)
    if (needs_scale_back) { output = output / 255.0f; }

    // Clamp output to appropriate range
    output = output.clamp(0.0f, needs_scale_back ? 1.0f : 255.0f);

    // Preserve original dtype (matching PyTorch backend logic)
    if (was_uint8_or_high_range && !needs_scale_back) {
        // Output is in [0, 255] range, convert to original dtype
        if (original_dtype == torch::kUInt8) { output = output.to(torch::kUInt8); }
    }

    return output;
}
