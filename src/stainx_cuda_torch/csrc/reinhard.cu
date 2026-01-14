// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * PyTorch wrapper for Reinhard normalization CUDA kernels.
 *
 * This file provides PyTorch tensor interfaces that call the pure CUDA kernels
 * from the main csrc/ directory.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

// Include the pure CUDA kernels from the main csrc directory
// Project root is in include_dirs, so we can include from csrc/
#include "csrc/reinhard.cu"

#define THREADS_PER_BLOCK 256

torch::Tensor reinhard_cuda(torch::Tensor input_images, torch::Tensor reference_mean, torch::Tensor reference_std) {
    // Check inputs
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(reference_mean.is_cuda(), "reference_mean must be a CUDA tensor");
    TORCH_CHECK(reference_std.is_cuda(), "reference_std must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W), got ", input_images.dim(), "D tensor with shape ", input_images.sizes());
    TORCH_CHECK(input_images.size(1) == 3, "input_images must have 3 channels, got ", input_images.size(1), " channels");

    // Check that tensors are on the same device
    TORCH_CHECK(input_images.device() == reference_mean.device(),
                "input_images and reference_mean must be on the same device. "
                "input_images device: ",
                input_images.device(),
                ", reference_mean device: ",
                reference_mean.device());
    TORCH_CHECK(input_images.device() == reference_std.device(),
                "input_images and reference_std must be on the same device. "
                "input_images device: ",
                input_images.device(),
                ", reference_std device: ",
                reference_std.device());

    // Get device and stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err     = cudaSuccess;  // Declare once at function level

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
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in rgb_to_lab_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }

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
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in statistics_matching_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }

    // Convert LAB back to RGB
    torch::Tensor rgb_normalized = torch::empty_like(lab_normalized);

    lab_to_rgb_kernel<<<num_blocks, num_threads, 0, stream>>>(lab_normalized.data_ptr<float>(), rgb_normalized.data_ptr<float>(), num_pixels);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in lab_to_rgb_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }

    // Synchronize stream before final operations
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing stream: ", cudaGetErrorString(err)); }

    // Reshape back to (N, C, H, W)
    torch::Tensor output = rgb_normalized.view({N, H, W, 3}).permute(at::IntArrayRef({0, 3, 1, 2}));

    // Preserve original dtype
    if (input_images.dtype() == torch::kUInt8 || input_images.max().item<float>() > 1.0f) {
        output = (output * 255.0f).clamp(0.0f, 255.0f);
        if (input_images.dtype() == torch::kUInt8) { output = output.to(torch::kUInt8); }
    }

    return output;
}
