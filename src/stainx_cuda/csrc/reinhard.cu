// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Reinhard normalization CUDA kernel implementation.
 *
 * This file contains CUDA kernels for Reinhard stain normalization.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

// Constants for color space conversion
#define RGB_TO_XYZ_00 0.412453f
#define RGB_TO_XYZ_01 0.357580f
#define RGB_TO_XYZ_02 0.180423f
#define RGB_TO_XYZ_10 0.212671f
#define RGB_TO_XYZ_11 0.715160f
#define RGB_TO_XYZ_12 0.072169f
#define RGB_TO_XYZ_20 0.019334f
#define RGB_TO_XYZ_21 0.119193f
#define RGB_TO_XYZ_22 0.950227f

#define XYZ_TO_RGB_00 3.2404542f
#define XYZ_TO_RGB_01 -1.5371385f
#define XYZ_TO_RGB_02 -0.4985314f
#define XYZ_TO_RGB_10 -0.9692660f
#define XYZ_TO_RGB_11 1.8760108f
#define XYZ_TO_RGB_12 0.0415560f
#define XYZ_TO_RGB_20 0.0556434f
#define XYZ_TO_RGB_21 -0.2040259f
#define XYZ_TO_RGB_22 1.0572252f

#define D65_X 0.95047f
#define D65_Y 1.0f
#define D65_Z 1.08883f

// Kernel for RGB to LAB conversion
__global__ void rgb_to_lab_kernel(const float* rgb, float* lab, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int base_idx = idx * 3;

        // Get RGB values (already normalized to [0, 1])
        float r = rgb[base_idx];
        float g = rgb[base_idx + 1];
        float b = rgb[base_idx + 2];

        // Gamma correction: sRGB to linear RGB
        float linear_r = (r > 0.04045f) ? powf((r + 0.055f) / 1.055f, 2.4f) : r / 12.92f;
        float linear_g = (g > 0.04045f) ? powf((g + 0.055f) / 1.055f, 2.4f) : g / 12.92f;
        float linear_b = (b > 0.04045f) ? powf((b + 0.055f) / 1.055f, 2.4f) : b / 12.92f;

        // RGB to XYZ conversion
        float x = RGB_TO_XYZ_00 * linear_r + RGB_TO_XYZ_01 * linear_g + RGB_TO_XYZ_02 * linear_b;
        float y = RGB_TO_XYZ_10 * linear_r + RGB_TO_XYZ_11 * linear_g + RGB_TO_XYZ_12 * linear_b;
        float z = RGB_TO_XYZ_20 * linear_r + RGB_TO_XYZ_21 * linear_g + RGB_TO_XYZ_22 * linear_b;

        // Normalize by D65 white point
        float x_norm = x / D65_X;
        float y_norm = y / D65_Y;
        float z_norm = z / D65_Z;

        // XYZ to LAB conversion
        float fx = (x_norm > 0.008856f) ? cbrtf(x_norm) : (7.787f * x_norm + 16.0f / 116.0f);
        float fy = (y_norm > 0.008856f) ? cbrtf(y_norm) : (7.787f * y_norm + 16.0f / 116.0f);
        float fz = (z_norm > 0.008856f) ? cbrtf(z_norm) : (7.787f * z_norm + 16.0f / 116.0f);

        float L     = (116.0f * fy - 16.0f) * 2.55f;
        float a     = 500.0f * (fx - fy) + 128.0f;
        float b_val = 200.0f * (fy - fz) + 128.0f;

        lab[base_idx]     = L;
        lab[base_idx + 1] = a;
        lab[base_idx + 2] = b_val;
    }
}

// Kernel for LAB to RGB conversion
__global__ void lab_to_rgb_kernel(const float* lab, float* rgb, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int base_idx = idx * 3;

        // Get LAB values
        float L     = lab[base_idx] / 2.55f;
        float a     = lab[base_idx + 1] - 128.0f;
        float b_val = lab[base_idx + 2] - 128.0f;

        // LAB to XYZ conversion
        float fy = (L + 16.0f) / 116.0f;
        float fx = a / 500.0f + fy;
        float fz = fy - b_val / 200.0f;

        float x_norm = (fx > 0.2068966f) ? fx * fx * fx : (fx - 16.0f / 116.0f) / 7.787f;
        float y_norm = (fy > 0.2068966f) ? fy * fy * fy : (fy - 16.0f / 116.0f) / 7.787f;
        float z_norm = (fz > 0.2068966f) ? fz * fz * fz : (fz - 16.0f / 116.0f) / 7.787f;

        // Denormalize by D65 white point
        float x = x_norm * D65_X;
        float y = y_norm * D65_Y;
        float z = z_norm * D65_Z;

        // XYZ to RGB conversion
        float linear_r = XYZ_TO_RGB_00 * x + XYZ_TO_RGB_01 * y + XYZ_TO_RGB_02 * z;
        float linear_g = XYZ_TO_RGB_10 * x + XYZ_TO_RGB_11 * y + XYZ_TO_RGB_12 * z;
        float linear_b = XYZ_TO_RGB_20 * x + XYZ_TO_RGB_21 * y + XYZ_TO_RGB_22 * z;

        // Gamma correction: linear RGB to sRGB
        float r = (linear_r > 0.0031308f) ? (1.055f * powf(linear_r, 1.0f / 2.4f) - 0.055f) : (12.92f * linear_r);
        float g = (linear_g > 0.0031308f) ? (1.055f * powf(linear_g, 1.0f / 2.4f) - 0.055f) : (12.92f * linear_g);
        float b = (linear_b > 0.0031308f) ? (1.055f * powf(linear_b, 1.0f / 2.4f) - 0.055f) : (12.92f * linear_b);

        // Clamp to [0, 1]
        rgb[base_idx]     = fmaxf(0.0f, fminf(1.0f, r));
        rgb[base_idx + 1] = fmaxf(0.0f, fminf(1.0f, g));
        rgb[base_idx + 2] = fmaxf(0.0f, fminf(1.0f, b));
    }
}

// Kernel for statistics matching in LAB space
__global__ void statistics_matching_kernel(const float* lab_input, float* lab_output, const float* ref_mean, const float* ref_std, const float* src_mean, const float* src_std, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int base_idx = idx * 3;

        for (int c = 0; c < 3; c++) {
            float src_val            = lab_input[base_idx + c];
            float normalized         = (src_val - src_mean[c]) / (src_std[c] + 1e-8f);
            lab_output[base_idx + c] = normalized * ref_std[c] + ref_mean[c];
        }
    }
}

// Kernel to compute mean and std using reduction
__global__ void compute_mean_std_kernel(const float* lab, float* mean, float* std, int num_pixels, int channel) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    float sum = 0.0f;
    if (idx < num_pixels) { sum = lab[idx * 3 + channel]; }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (idx + s) < num_pixels) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) { mean[blockIdx.x] = sdata[0]; }
}

torch::Tensor reinhard_cuda(torch::Tensor input_images, torch::Tensor reference_mean, torch::Tensor reference_std) {
    // Check inputs
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(reference_mean.is_cuda(), "reference_mean must be a CUDA tensor");
    TORCH_CHECK(reference_std.is_cuda(), "reference_std must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W)");
    TORCH_CHECK(input_images.size(1) == 3, "input_images must have 3 channels");

    // Get device and stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Normalize input to [0, 1] float
    torch::Tensor images_float;
    if (input_images.dtype() == torch::kUInt8 || input_images.max().item<float>() > 1.0f) {
        images_float = input_images.to(torch::kFloat32) / 255.0f;
    } else {
        images_float = input_images.to(torch::kFloat32);
    }

    int N          = images_float.size(0);
    int C          = images_float.size(1);
    int H          = images_float.size(2);
    int W          = images_float.size(3);
    int num_pixels = N * H * W;

    // Reshape to (N*H*W, 3) for kernel processing
    torch::Tensor rgb_flat = images_float.permute(at::IntArrayRef({0, 2, 3, 1})).contiguous().view({-1, 3});

    // Convert RGB to LAB
    torch::Tensor lab_flat = torch::empty_like(rgb_flat);

    int num_threads = THREADS_PER_BLOCK;
    int num_blocks  = (num_pixels + num_threads - 1) / num_threads;

    rgb_to_lab_kernel<<<num_blocks, num_threads, 0, stream>>>(rgb_flat.data_ptr<float>(), lab_flat.data_ptr<float>(), num_pixels);

    // Compute source mean and std in LAB space
    torch::Tensor lab_reshaped = lab_flat.view({N, H, W, 3}).permute(at::IntArrayRef({0, 3, 1, 2}));  // (N, 3, H, W)

    // Compute mean and std using PyTorch operations (more efficient than custom kernel for this)
    torch::Tensor src_mean = lab_reshaped.mean({0, 2, 3});  // (3,)
    torch::Tensor src_std  = lab_reshaped.std({0, 2, 3});   // (3,)

    // Prepare reference mean and std
    torch::Tensor ref_mean = reference_mean.to(torch::kFloat32).contiguous();
    torch::Tensor ref_std  = reference_std.to(torch::kFloat32).contiguous();

    if (ref_mean.dim() == 1 && ref_mean.size(0) == 3) {
        // Already in correct format
    } else {
        ref_mean = ref_mean.view({3});
        ref_std  = ref_std.view({3});
    }

    // Normalize LAB values to match reference statistics
    torch::Tensor lab_normalized = torch::empty_like(lab_flat);

    statistics_matching_kernel<<<num_blocks, num_threads, 0, stream>>>(lab_flat.data_ptr<float>(), lab_normalized.data_ptr<float>(), ref_mean.data_ptr<float>(), ref_std.data_ptr<float>(), src_mean.data_ptr<float>(), src_std.data_ptr<float>(), num_pixels);

    // Convert LAB back to RGB
    torch::Tensor rgb_normalized = torch::empty_like(lab_normalized);

    lab_to_rgb_kernel<<<num_blocks, num_threads, 0, stream>>>(lab_normalized.data_ptr<float>(), rgb_normalized.data_ptr<float>(), num_pixels);

    // Reshape back to (N, C, H, W)
    torch::Tensor output = rgb_normalized.view({N, H, W, 3}).permute(at::IntArrayRef({0, 3, 1, 2}));

    // Preserve original dtype
    if (input_images.dtype() == torch::kUInt8 || input_images.max().item<float>() > 1.0f) {
        output = (output * 255.0f).clamp(0.0f, 255.0f);
        if (input_images.dtype() == torch::kUInt8) { output = output.to(torch::kUInt8); }
    }

    return output;
}
