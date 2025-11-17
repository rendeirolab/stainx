// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Macenko normalization CUDA kernel implementation.
 *
 * This file contains CUDA kernels for Macenko stain normalization using SVD.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#include <cuda_runtime.h>
#include <limits>
#include <math.h>

#define THREADS_PER_BLOCK 256

// Kernel to convert RGB to optical density
__global__ void rgb_to_od_kernel(const float* rgb, float* od, float Io, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int base_idx = idx * 3;

        float r = rgb[base_idx];
        float g = rgb[base_idx + 1];
        float b = rgb[base_idx + 2];

        // Convert to OD: OD = -log((RGB + 1) / Io)
        od[base_idx]     = -logf((r + 1.0f) / Io);
        od[base_idx + 1] = -logf((g + 1.0f) / Io);
        od[base_idx + 2] = -logf((b + 1.0f) / Io);
    }
}

// Kernel to filter pixels by minimum OD threshold
__global__ void filter_od_kernel(const float* od, float* od_filtered, int* mask, float beta, int num_pixels, int* filtered_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pixels) {
        int base_idx = idx * 3;

        // Find minimum OD across channels
        float min_od = fminf(fminf(od[base_idx], od[base_idx + 1]), od[base_idx + 2]);

        if (min_od >= beta) {
            // Use atomic to get unique index for filtered pixels
            int pos = atomicAdd(filtered_count, 1);
            if (pos < num_pixels) {
                mask[pos]                = idx;
                od_filtered[pos * 3]     = od[base_idx];
                od_filtered[pos * 3 + 1] = od[base_idx + 1];
                od_filtered[pos * 3 + 2] = od[base_idx + 2];
            }
        }
    }
}

// Kernel to compute atan2 for phi calculation
__global__ void compute_phi_kernel(const float* That, float* phi, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        float x  = That[idx * 2];
        float y  = That[idx * 2 + 1];
        phi[idx] = atan2f(y, x);
    }
}

// Kernel to convert OD back to RGB
__global__ void od_to_rgb_kernel(const float* od, float* rgb, float Io, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int base_idx = idx * 3;

        float od_r = od[base_idx];
        float od_g = od[base_idx + 1];
        float od_b = od[base_idx + 2];

        // Convert back to RGB: RGB = Io * exp(-OD)
        float r = Io * expf(-od_r);
        float g = Io * expf(-od_g);
        float b = Io * expf(-od_b);

        // Clamp to [0, 255]
        rgb[base_idx]     = fmaxf(0.0f, fminf(255.0f, r));
        rgb[base_idx + 1] = fmaxf(0.0f, fminf(255.0f, g));
        rgb[base_idx + 2] = fmaxf(0.0f, fminf(255.0f, b));
    }
}

// Helper function to compute percentile (using PyTorch operations)
float compute_percentile_cuda(torch::Tensor t, float q) {
    int k       = 1 + (int) round(0.01f * q * (t.numel() - 1));
    auto result = t.view(-1).kthvalue(k);
    return std::get<0>(result).item<float>();
}

torch::Tensor macenko_cuda(torch::Tensor input_images, torch::Tensor stain_matrix, torch::Tensor target_max_conc) {
    // Check inputs
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(stain_matrix.is_cuda(), "stain_matrix must be a CUDA tensor");
    TORCH_CHECK(target_max_conc.is_cuda(), "target_max_conc must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W)");
    TORCH_CHECK(input_images.size(1) == 3, "input_images must have 3 channels");
    TORCH_CHECK(stain_matrix.size(0) == 3 && stain_matrix.size(1) == 2, "stain_matrix must have shape (3, 2)");

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
    int num_pixels = H * W;

    // Constants
    float Io    = 240.0f;
    float beta  = 0.15f;
    float alpha = 1.0f;

    // Flatten target_max_conc
    torch::Tensor target_max_conc_flat = target_max_conc.flatten().to(torch::kFloat32);
    TORCH_CHECK(target_max_conc_flat.size(0) == 2, "target_max_conc must have 2 elements");

    // Pre-allocate output tensor
    torch::Tensor output = torch::empty({N, C, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device()));

    // Process each image
    for (int n = 0; n < N; n++) {
        // Extract single image
        torch::Tensor image = images_float[n];  // (3, H, W)

        // Convert RGB to OD
        // Scale to [0, 255] range first (matching PyTorch backend logic)
        torch::Tensor rgb_scaled   = image * 255.0f;  // (3, H, W)
        torch::Tensor rgb_permuted = rgb_scaled.permute(at::IntArrayRef({1, 2, 0}));
        torch::Tensor rgb_flat     = rgb_permuted.reshape({num_pixels, 3});  // (H*W, 3)
        torch::Tensor od_flat      = torch::empty_like(rgb_flat);

        int num_threads = THREADS_PER_BLOCK;
        int num_blocks  = (num_pixels + num_threads - 1) / num_threads;

        rgb_to_od_kernel<<<num_blocks, num_threads, 0, stream>>>(rgb_flat.data_ptr<float>(), od_flat.data_ptr<float>(), Io, num_pixels);

        // Filter by minimum OD
        torch::Tensor od_reshaped_3d   = od_flat.reshape({H, W, 3});
        torch::Tensor od_reshaped      = od_reshaped_3d.permute(at::IntArrayRef({2, 0, 1}));  // (3, H, W)
        auto min_result                = od_reshaped.min(0);
        torch::Tensor od_min           = std::get<0>(min_result);  // (H, W)
        torch::Tensor mask             = od_min >= beta;
        torch::Tensor od_permuted      = od_reshaped.permute(at::IntArrayRef({1, 2, 0}));
        torch::Tensor od_flat_reshaped = od_permuted.reshape({num_pixels, 3});  // (H*W, 3)
        torch::Tensor mask_flat        = mask.flatten();
        torch::Tensor indices          = torch::nonzero(mask_flat).squeeze(1);
        torch::Tensor od_filtered      = od_flat_reshaped.index_select(0, indices);  // (num_filtered, 3)

        // Safety check: if too few pixels, use all pixels
        if (od_filtered.size(0) < 3) { od_filtered = od_flat; }

        // Compute covariance matrix using PyTorch (more efficient)
        torch::Tensor od_filtered_T = od_filtered.t();              // (3, num_filtered)
        torch::Tensor od_mean       = od_filtered_T.mean(1, true);  // (3, 1)
        torch::Tensor od_centered   = od_filtered_T - od_mean;      // (3, num_filtered)
        int num_filtered            = od_filtered.size(0);
        torch::Tensor cov           = torch::matmul(od_centered, od_centered.t()) / (num_filtered - 1.0f);  // (3, 3)

        // Eigenvalue decomposition using PyTorch
        auto eig_result       = at::linalg_eigh(cov);
        torch::Tensor eigvecs = std::get<1>(eig_result);                                                  // (3, 3)
        eigvecs               = eigvecs.index({torch::indexing::Slice(), torch::indexing::Slice(1, 3)});  // (3, 2)

        // Compute That and phi
        torch::Tensor That = torch::matmul(od_filtered, eigvecs);                                                                 // (num_filtered, 2)
        torch::Tensor phi  = torch::atan2(That.index({torch::indexing::Slice(), 1}), That.index({torch::indexing::Slice(), 0}));  // (num_filtered,)

        // Compute percentiles
        float min_phi = compute_percentile_cuda(phi, alpha);
        float max_phi = compute_percentile_cuda(phi, 100.0f - alpha);

        // Compute stain vectors
        torch::Tensor cos_min = torch::cos(torch::tensor(min_phi, torch::TensorOptions().device(images_float.device())));
        torch::Tensor sin_min = torch::sin(torch::tensor(min_phi, torch::TensorOptions().device(images_float.device())));
        torch::Tensor cos_max = torch::cos(torch::tensor(max_phi, torch::TensorOptions().device(images_float.device())));
        torch::Tensor sin_max = torch::sin(torch::tensor(max_phi, torch::TensorOptions().device(images_float.device())));

        torch::Tensor angle_min = torch::stack({cos_min, sin_min});  // (2,)
        torch::Tensor angle_max = torch::stack({cos_max, sin_max});  // (2,)

        torch::Tensor vMin = torch::matmul(eigvecs, angle_min).unsqueeze(1);  // (3, 1)
        torch::Tensor vMax = torch::matmul(eigvecs, angle_max).unsqueeze(1);  // (3, 1)

        torch::Tensor HE_source;
        if (vMin[0].item<float>() > vMax[0].item<float>()) {
            HE_source = torch::cat({vMin, vMax}, 1);  // (3, 2)
        } else {
            HE_source = torch::cat({vMax, vMin}, 1);  // (3, 2)
        }

        // Reshape OD for concentration computation
        torch::Tensor od_all = od_reshaped.reshape({3, -1});  // (3, H*W)

        // Compute concentrations using least squares or pseudoinverse
        bool use_fallback = false;
        if (od_all.size(1) > 1000000) { use_fallback = true; }

        torch::Tensor concentrations;
        if (!use_fallback && HE_source.size(0) >= HE_source.size(1)) {
            try {
                torch::Tensor cond_num = at::linalg_cond(HE_source);
                if (cond_num.item<float>() > 10.0f) { use_fallback = true; }
            } catch (...) { use_fallback = true; }
        }

        if (use_fallback) {
            torch::Tensor HE_pinv = at::linalg_pinv(HE_source);
            concentrations        = torch::matmul(HE_pinv, od_all);  // (2, H*W)
        } else {
            try {
                auto lstsq_result = at::linalg_lstsq(HE_source, od_all);
                concentrations    = std::get<0>(lstsq_result);  // (2, H*W)
            } catch (...) {
                torch::Tensor HE_pinv = at::linalg_pinv(HE_source);
                concentrations        = torch::matmul(HE_pinv, od_all);
            }
        }

        // Compute max concentrations
        float max_conc_0       = compute_percentile_cuda(concentrations[0], 99.0f);
        float max_conc_1       = compute_percentile_cuda(concentrations[1], 99.0f);
        torch::Tensor max_conc = torch::tensor({max_conc_0, max_conc_1}, torch::TensorOptions().device(images_float.device()));

        // Normalize concentrations
        torch::Tensor norm_factor         = target_max_conc_flat / max_conc;            // (2,)
        torch::Tensor concentrations_norm = concentrations * norm_factor.unsqueeze(1);  // (2, H*W)

        // Reconstruct OD using reference stain matrix
        torch::Tensor od_recon = torch::matmul(stain_matrix, concentrations_norm);  // (3, H*W)

        // Clamp negative OD values to 0 (OD must be >= 0)
        od_recon = torch::clamp(od_recon, 0.0f, std::numeric_limits<float>::max());

        // Convert OD back to RGB
        // Kernel expects interleaved format (H*W, 3), so transpose od_recon
        torch::Tensor od_recon_T = od_recon.t();  // (H*W, 3)
        torch::Tensor rgb_recon  = torch::empty_like(od_recon_T);

        od_to_rgb_kernel<<<num_blocks, num_threads, 0, stream>>>(od_recon_T.data_ptr<float>(), rgb_recon.data_ptr<float>(), Io, H * W);

        // Transpose back to (3, H*W) and reshape to (3, H, W)
        torch::Tensor rgb_recon_T = rgb_recon.t();  // (3, H*W)

        output[n] = rgb_recon_T.view({3, H, W});
    }

    // Preserve original dtype (matching PyTorch backend's preserve_dtype logic)
    // Result is in [0, 255] range (result_in_0_255_range=True)
    output = output.clamp(0.0f, 255.0f);
    // Always convert to original dtype (matching PyTorch backend)
    torch::ScalarType original_dtype = input_images.scalar_type();
    output                           = output.to(original_dtype);

    return output;
}
