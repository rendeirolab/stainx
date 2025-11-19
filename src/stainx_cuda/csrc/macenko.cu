// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Macenko normalization CUDA kernel implementation.
 *
 * This file contains CUDA kernels for Macenko stain normalization using SVD.
 * All computations are performed directly in CUDA using cuBLAS and cuSOLVER.
 * Uses batched cuBLAS operations for parallel processing.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Kernel to count filtered pixels for all images and store per-image counts
__global__ void count_filtered_pixels_all_images_kernel(const float* min_od_batched, int* num_filtered_array, float beta, int num_pixels, int N) {
    int n = blockIdx.x;  // One block per image
    if (n >= N) return;

    const float* min_od = min_od_batched + n * num_pixels;

    __shared__ int block_count;
    if (threadIdx.x == 0) { block_count = 0; }
    __syncthreads();

    // Count filtered pixels
    int local_count = 0;
    for (int i = threadIdx.x; i < num_pixels; i += blockDim.x) {
        if (min_od[i] >= beta) { local_count++; }
    }

    atomicAdd(&block_count, local_count);
    __syncthreads();

    // Store count for this image (use all pixels if count < 3)
    if (threadIdx.x == 0) {
        int final_count       = (block_count < 3) ? num_pixels : block_count;
        num_filtered_array[n] = final_count;
    }
}

// Kernel to find maximum value in an array
__global__ void find_max_kernel(const int* array, int* max_val, int N) {
    __shared__ int shared_max[THREADS_PER_BLOCK / WARP_SIZE];

    int tid       = threadIdx.x;
    int local_max = 0;

    // Each thread finds local max
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) { local_max = max(local_max, array[i]); }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        int other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = max(local_max, other);
    }

    // First thread in each warp writes to shared memory
    int warp_id = tid / WARP_SIZE;
    if (tid % WARP_SIZE == 0) { shared_max[warp_id] = local_max; }
    __syncthreads();

    // Final reduction in first warp
    if (tid < THREADS_PER_BLOCK / WARP_SIZE) {
        local_max = shared_max[tid];
    } else {
        local_max = 0;
    }

    if (warp_id == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            int other = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = max(local_max, other);
        }

        if (tid == 0) { atomicMax(max_val, local_max); }
    }
}

// Kernel to convert RGB to optical density - batched version
__global__ void rgb_to_od_kernel_batched(const float* rgb, float* od, float Io, int num_pixels, int N) {
    int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * num_pixels;
    if (idx < total_pixels) {
        int n         = idx / num_pixels;
        int pixel_idx = idx % num_pixels;
        int base_idx  = n * num_pixels * 3 + pixel_idx * 3;

        float r = rgb[base_idx];
        float g = rgb[base_idx + 1];
        float b = rgb[base_idx + 2];

        // Convert to OD: OD = -log((RGB + 1) / Io)
        od[base_idx]     = -logf((r + 1.0f) / Io);
        od[base_idx + 1] = -logf((g + 1.0f) / Io);
        od[base_idx + 2] = -logf((b + 1.0f) / Io);
    }
}

// Kernel to compute minimum OD per pixel and create mask - batched version
__global__ void compute_min_od_mask_kernel_batched(const float* od, float* min_od, int num_pixels, int N) {
    int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * num_pixels;
    if (idx < total_pixels) {
        int n                              = idx / num_pixels;
        int pixel_idx                      = idx % num_pixels;
        int base_idx                       = n * num_pixels * 3 + pixel_idx * 3;
        min_od[n * num_pixels + pixel_idx] = fminf(fminf(od[base_idx], od[base_idx + 1]), od[base_idx + 2]);
    }
}

// Batched kernel to compact filtered pixels for all images
// Processes all images, each image writes to its allocated space (padded to max_num_filtered)
__global__ void compact_filtered_batched_kernel(const float* od_flat_batched, const float* min_od_batched, float* od_filtered_batched, const int* num_filtered_array, float beta, int num_pixels, int max_num_filtered, int N) {
    int n = blockIdx.x;  // One block per image
    if (n >= N) return;

    const float* od_flat = od_flat_batched + n * num_pixels * 3;
    const float* min_od  = min_od_batched + n * num_pixels;
    float* od_filtered   = od_filtered_batched + n * max_num_filtered * 3;
    int num_filtered     = num_filtered_array[n];
    bool use_all_pixels  = (num_filtered == num_pixels);

    __shared__ int write_pos;
    if (threadIdx.x == 0) { write_pos = 0; }
    __syncthreads();

    // Each thread processes pixels
    for (int i = threadIdx.x; i < num_pixels; i += blockDim.x) {
        bool passes_filter = use_all_pixels || (min_od[i] >= beta);

        if (passes_filter) {
            int pos = atomicAdd(&write_pos, 1);
            if (pos < max_num_filtered) {
                od_filtered[pos * 3 + 0] = od_flat[i * 3 + 0];
                od_filtered[pos * 3 + 1] = od_flat[i * 3 + 1];
                od_filtered[pos * 3 + 2] = od_flat[i * 3 + 2];
            }
        }
    }
    __syncthreads();

    // Pad remaining entries with zeros
    for (int i = write_pos + threadIdx.x; i < max_num_filtered; i += blockDim.x) {
        od_filtered[i * 3 + 0] = 0.0f;
        od_filtered[i * 3 + 1] = 0.0f;
        od_filtered[i * 3 + 2] = 0.0f;
    }
}

// Kernel to clamp values
__global__ void clamp_kernel(float* data, float min_val, float max_val, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) { data[idx] = fmaxf(min_val, fminf(max_val, data[idx])); }
}

// Kernel to scale and convert image format (3, H, W) -> (H*W, 3) - batched version
__global__ void reshape_and_scale_kernel_batched(const float* input, float* output, int H, int W, int N) {
    int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels   = H * W;
    int total_pixels = N * num_pixels;
    if (idx < total_pixels) {
        int n         = idx / num_pixels;
        int pixel_idx = idx % num_pixels;
        int h         = pixel_idx / W;
        int w         = pixel_idx % W;
        int base_in   = n * 3 * H * W;
        int base_out  = n * num_pixels * 3;
        // Input is (N, 3, H, W), output is (N, H*W, 3)
        output[base_out + pixel_idx * 3]     = input[base_in + 0 * H * W + h * W + w] * 255.0f;
        output[base_out + pixel_idx * 3 + 1] = input[base_in + 1 * H * W + h * W + w] * 255.0f;
        output[base_out + pixel_idx * 3 + 2] = input[base_in + 2 * H * W + h * W + w] * 255.0f;
    }
}

// Kernel to reshape output back (H*W, 3) -> (3, H, W) - batched version
__global__ void reshape_output_kernel_batched(const float* input, float* output, int H, int W, int N) {
    int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels   = H * W;
    int total_pixels = N * num_pixels;
    if (idx < total_pixels) {
        int n         = idx / num_pixels;
        int pixel_idx = idx % num_pixels;
        int h         = pixel_idx / W;
        int w         = pixel_idx % W;
        int base_in   = n * num_pixels * 3;
        int base_out  = n * 3 * H * W;
        // Input is (N, H*W, 3), output is (N, 3, H, W)
        output[base_out + 0 * H * W + h * W + w] = input[base_in + pixel_idx * 3];
        output[base_out + 1 * H * W + h * W + w] = input[base_in + pixel_idx * 3 + 1];
        output[base_out + 2 * H * W + h * W + w] = input[base_in + pixel_idx * 3 + 2];
    }
}

// Batched kernel to compute covariance matrices for all images
__global__ void compute_covariance_batched_kernel(const float* od_filtered_batched, float* cov_batched, const int* num_filtered_array, int max_num_filtered, int N) {
    int n = blockIdx.x;  // One block per image
    if (n >= N) return;

    int num_filtered         = num_filtered_array[n];
    const float* od_filtered = od_filtered_batched + n * max_num_filtered * 3;
    float* cov               = cov_batched + n * 9;

    // Compute mean using shared memory
    __shared__ float mean[3];
    __shared__ float sums[3];

    if (threadIdx.x < 3) { sums[threadIdx.x] = 0.0f; }
    __syncthreads();

    // Parallel sum for mean
    for (int i = threadIdx.x; i < num_filtered; i += blockDim.x) {
        atomicAdd(&sums[0], od_filtered[i * 3 + 0]);
        atomicAdd(&sums[1], od_filtered[i * 3 + 1]);
        atomicAdd(&sums[2], od_filtered[i * 3 + 2]);
    }
    __syncthreads();

    if (threadIdx.x < 3) { mean[threadIdx.x] = sums[threadIdx.x] / num_filtered; }
    __syncthreads();

    // Compute covariance matrix elements
    __shared__ float cov_sums[9];
    if (threadIdx.x < 9) { cov_sums[threadIdx.x] = 0.0f; }
    __syncthreads();

    // Each thread accumulates covariance contributions
    for (int k = threadIdx.x; k < num_filtered; k += blockDim.x) {
        float diff[3];
        diff[0] = od_filtered[k * 3 + 0] - mean[0];
        diff[1] = od_filtered[k * 3 + 1] - mean[1];
        diff[2] = od_filtered[k * 3 + 2] - mean[2];

        // Accumulate all 9 covariance elements
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) { atomicAdd(&cov_sums[i * 3 + j], diff[i] * diff[j]); }
        }
    }
    __syncthreads();

    // Scale and write final covariance
    float scale = 1.0f / (num_filtered - 1.0f);
    if (threadIdx.x < 9) { cov[threadIdx.x] = cov_sums[threadIdx.x] * scale; }
}

// Batched kernel to extract last 2 eigenvectors from 3x3 eigenvector matrices
__global__ void extract_eigvecs_2d_batched_kernel(const float* eigvecs_batched, float* eigvecs_2d_batched, int N) {
    int n = blockIdx.x;  // One block per image
    if (n >= N) return;

    const float* eigvecs = eigvecs_batched + n * 9;     // 3x3 column-major
    float* eigvecs_2d    = eigvecs_2d_batched + n * 6;  // 3x2 column-major

    // Copy columns 1 and 2 (last 2 eigenvectors)
    // In column-major: column 0 is at indices 0,1,2; column 1 at 3,4,5; column 2 at 6,7,8
    if (threadIdx.x < 6) {
        int col = threadIdx.x / 3;  // 0 or 1
        int row = threadIdx.x % 3;  // 0, 1, or 2
        // Source column is col+1 (skip first column)
        eigvecs_2d[col * 3 + row] = eigvecs[(col + 1) * 3 + row];
    }
}

// Batched kernel to compute That = od_filtered * eigvecs_2d for all images
__global__ void compute_That_batched_kernel(const float* od_filtered_batched, const float* eigvecs_2d_batched, float* That_batched, const int* num_filtered_array, int max_num_filtered, int N) {
    int n = blockIdx.x;  // One block per image
    if (n >= N) return;

    int num_filtered         = num_filtered_array[n];
    const float* od_filtered = od_filtered_batched + n * max_num_filtered * 3;
    const float* eigvecs_2d  = eigvecs_2d_batched + n * 6;  // 3x2 matrix
    float* That              = That_batched + n * max_num_filtered * 2;

    // Each thread computes elements of That
    for (int idx = threadIdx.x; idx < num_filtered * 2; idx += blockDim.x) {
        int row = idx / 2;
        int col = idx % 2;

        float sum = 0.0f;
        for (int k = 0; k < 3; k++) {
            float od_val  = od_filtered[row * 3 + k];
            float eig_val = eigvecs_2d[col * 3 + k];  // column-major
            sum += od_val * eig_val;
        }
        That[row * 2 + col] = sum;
    }
}

// Batched kernel to compute phi for all images
__global__ void compute_phi_batched_kernel(const float* That_batched, float* phi_batched, const int* num_filtered_array, int max_num_filtered, int N) {
    int n = blockIdx.x;  // One block per image
    if (n >= N) return;

    int num_filtered  = num_filtered_array[n];
    const float* That = That_batched + n * max_num_filtered * 2;
    float* phi        = phi_batched + n * max_num_filtered;

    for (int i = threadIdx.x; i < num_filtered; i += blockDim.x) {
        float x = That[i * 2];
        float y = That[i * 2 + 1];
        phi[i]  = atan2f(y, x);
    }
}

// Batched kernel to compute stain vectors from phi percentiles
__global__ void compute_stain_vectors_batched_kernel(const float* eigvecs_2d_batched, const float* min_phi_array, const float* max_phi_array, float* HE_source_batched, int N) {
    int n = blockIdx.x;  // One block per image
    if (n >= N) return;

    const float* eigvecs_2d = eigvecs_2d_batched + n * 6;  // 3x2
    float* HE_source        = HE_source_batched + n * 6;   // 3x2

    float min_phi = min_phi_array[n];
    float max_phi = max_phi_array[n];

    // Compute angle vectors
    float angle_min[2] = {cosf(min_phi), sinf(min_phi)};
    float angle_max[2] = {cosf(max_phi), sinf(max_phi)};

    // Compute vMin = eigvecs_2d * angle_min (3x2 * 2x1 = 3x1)
    __shared__ float vMin[3];
    __shared__ float vMax[3];

    if (threadIdx.x < 3) {
        float sum_min = 0.0f;
        float sum_max = 0.0f;
        for (int k = 0; k < 2; k++) {
            float eig_val = eigvecs_2d[k * 3 + threadIdx.x];  // column-major
            sum_min += eig_val * angle_min[k];
            sum_max += eig_val * angle_max[k];
        }
        vMin[threadIdx.x] = sum_min;
        vMax[threadIdx.x] = sum_max;
    }
    __syncthreads();

    // Assemble HE_source based on which is larger
    // HE_source is 3x2 in column-major format for cuSOLVER
    if (threadIdx.x == 0) {
        if (vMin[0] > vMax[0]) {
            // HE_source = [vMin, vMax] - column 0 is vMin, column 1 is vMax
            // In column-major: col0 = indices 0,1,2; col1 = indices 3,4,5
            for (int i = 0; i < 3; i++) {
                HE_source[0 * 3 + i] = vMin[i];  // column 0
                HE_source[1 * 3 + i] = vMax[i];  // column 1
            }
        } else {
            // HE_source = [vMax, vMin]
            for (int i = 0; i < 3; i++) {
                HE_source[0 * 3 + i] = vMax[i];  // column 0
                HE_source[1 * 3 + i] = vMin[i];  // column 1
            }
        }
    }
}

// Batched kernel to reshape OD for all images: (N, H*W, 3) -> (N, 3, H*W)
__global__ void reshape_od_to_3xN_batched_kernel(const float* od_flat_batched, float* od_all_batched, int num_pixels, int N) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_pixels;
    if (idx < total) {
        int n         = idx / num_pixels;
        int pixel_idx = idx % num_pixels;
        int base_in   = n * num_pixels * 3;
        int base_out  = n * 3 * num_pixels;

        od_all_batched[base_out + 0 * num_pixels + pixel_idx] = od_flat_batched[base_in + pixel_idx * 3 + 0];
        od_all_batched[base_out + 1 * num_pixels + pixel_idx] = od_flat_batched[base_in + pixel_idx * 3 + 1];
        od_all_batched[base_out + 2 * num_pixels + pixel_idx] = od_flat_batched[base_in + pixel_idx * 3 + 2];
    }
}

// Batched kernel to compute concentrations using pseudoinverse for all images
// Handles: concentrations = V * S^-1 * U^T * od_all
__global__ void compute_concentrations_batched_kernel(const float* U_batched, const float* S_batched, const float* VT_batched, const float* od_all_batched, float* concentrations_batched, int num_pixels, int N) {
    int n = blockIdx.x;  // One block per image
    if (n >= N) return;

    const float* U        = U_batched + n * 9;  // 3x3
    const float* S        = S_batched + n * 2;
    const float* VT       = VT_batched + n * 4;  // 2x2
    const float* od_all   = od_all_batched + n * 3 * num_pixels;
    float* concentrations = concentrations_batched + n * 2 * num_pixels;

    float threshold = 1e-6f;

    // Compute S_inv in shared memory
    __shared__ float S_inv[2];
    if (threadIdx.x < 2) { S_inv[threadIdx.x] = (S[threadIdx.x] > threshold) ? (1.0f / S[threadIdx.x]) : 0.0f; }
    __syncthreads();

    // Compute U^T * od_all -> UT_od (3 x num_pixels)
    // Then scale first 2 rows by S_inv -> scaled_ut (2 x num_pixels)
    // Then multiply by V -> concentrations (2 x num_pixels)

    // Process in chunks to fit in shared memory
    extern __shared__ float shared_buffer[];

    for (int col_start = 0; col_start < num_pixels; col_start += blockDim.x) {
        int col = col_start + threadIdx.x;

        if (col < num_pixels) {
            // Compute UT_od for this column
            float UT_od[3];
            for (int row = 0; row < 3; row++) {
                float sum = 0.0f;
                for (int k = 0; k < 3; k++) {
                    float u_val  = U[row * 3 + k];  // column-major
                    float od_val = od_all[k * num_pixels + col];
                    sum += u_val * od_val;
                }
                UT_od[row] = sum;
            }

            // Scale first 2 rows by S_inv
            float scaled_ut[2];
            scaled_ut[0] = UT_od[0] * S_inv[0];
            scaled_ut[1] = UT_od[1] * S_inv[1];

            // Multiply by V (VT^T)
            for (int row = 0; row < 2; row++) {
                float sum = 0.0f;
                for (int k = 0; k < 2; k++) {
                    float v_val = VT[row * 2 + k];  // VT[k][row] = V[row][k]
                    sum += v_val * scaled_ut[k];
                }
                concentrations[row * num_pixels + col] = sum;
            }
        }
    }
}

// Batched kernel to normalize concentrations for all images
__global__ void normalize_concentrations_batched_kernel(float* concentrations_batched, const float* max_conc_array, const float* target_max_conc, int num_pixels, int N) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 2 * num_pixels;
    if (idx < total) {
        int n         = idx / (2 * num_pixels);
        int remainder = idx % (2 * num_pixels);
        int channel   = remainder / num_pixels;
        int pixel     = remainder % num_pixels;

        float max_conc    = max_conc_array[n * 2 + channel];
        float target      = target_max_conc[channel];
        float norm_factor = target / max_conc;

        concentrations_batched[n * 2 * num_pixels + channel * num_pixels + pixel] *= norm_factor;
    }
}

// Batched kernel to reconstruct OD for all images
__global__ void compute_od_recon_batched_kernel(const float* stain_matrix, const float* concentrations_batched, float* od_recon_batched, int num_pixels, int N) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 3 * num_pixels;
    if (idx < total) {
        int n         = idx / (3 * num_pixels);
        int remainder = idx % (3 * num_pixels);
        int row       = remainder / num_pixels;
        int col       = remainder % num_pixels;

        const float* concentrations = concentrations_batched + n * 2 * num_pixels;

        float sum = 0.0f;
        for (int k = 0; k < 2; k++) {
            float stain_val = stain_matrix[row * 2 + k];
            float conc_val  = concentrations[k * num_pixels + col];
            sum += stain_val * conc_val;
        }

        od_recon_batched[n * 3 * num_pixels + row * num_pixels + col] = fmaxf(0.0f, sum);
    }
}

// Batched kernel to transpose OD from (N, 3, H*W) to (N, H*W, 3) and convert to RGB
__global__ void od_to_rgb_transpose_batched_kernel(const float* od_recon_batched, float* rgb_batched, float Io, int num_pixels, int N) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_pixels;
    if (idx < total) {
        int n         = idx / num_pixels;
        int pixel_idx = idx % num_pixels;

        int base_in  = n * 3 * num_pixels;
        int base_out = n * num_pixels * 3;

        float od_r = od_recon_batched[base_in + 0 * num_pixels + pixel_idx];
        float od_g = od_recon_batched[base_in + 1 * num_pixels + pixel_idx];
        float od_b = od_recon_batched[base_in + 2 * num_pixels + pixel_idx];

        // Convert to RGB
        float r = Io * expf(-od_r);
        float g = Io * expf(-od_g);
        float b = Io * expf(-od_b);

        rgb_batched[base_out + pixel_idx * 3 + 0] = fmaxf(0.0f, fminf(255.0f, r));
        rgb_batched[base_out + pixel_idx * 3 + 1] = fmaxf(0.0f, fminf(255.0f, g));
        rgb_batched[base_out + pixel_idx * 3 + 2] = fmaxf(0.0f, fminf(255.0f, b));
    }
}

// Helper function to compute percentile using sorting (CPU fallback)
float compute_percentile_cuda(const float* data, int n, float q) {
    // Calculate index
    int k = 1 + (int) round(0.01f * q * (n - 1));
    if (k < 1) k = 1;
    if (k > n) k = n;

    // Use a simple approach: copy to host, sort, get percentile
    // For better performance, use CUB::DeviceSelect or Thrust
    std::vector<float> host_data(n);
    cudaMemcpy(host_data.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::sort(host_data.begin(), host_data.end());
    float result = host_data[k - 1];

    return result;
}

// Helper to get cuSOLVER handle
cusolverDnHandle_t get_cusolver_handle() {
    static cusolverDnHandle_t handle = nullptr;
    if (handle == nullptr) { cusolverDnCreate(&handle); }
    return handle;
}

torch::Tensor macenko_cuda(torch::Tensor input_images, torch::Tensor stain_matrix, torch::Tensor target_max_conc) {
    // Check inputs
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(stain_matrix.is_cuda(), "stain_matrix must be a CUDA tensor");
    TORCH_CHECK(target_max_conc.is_cuda(), "target_max_conc must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W)");
    TORCH_CHECK(input_images.size(1) == 3, "input_images must have 3 channels");
    TORCH_CHECK(stain_matrix.size(0) == 3 && stain_matrix.size(1) == 2, "stain_matrix must have shape (3, 2)");

    // Get cuSOLVER handle
    cusolverDnHandle_t cusolver_handle = get_cusolver_handle();

    // Normalize input to [0, 1] float - minimal PyTorch operation for type conversion
    torch::Tensor images_float;
    if (input_images.dtype() == torch::kUInt8) {
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

    // Get target_max_conc as device pointer
    torch::Tensor target_max_conc_flat = target_max_conc.flatten().to(torch::kFloat32).contiguous();
    TORCH_CHECK(target_max_conc_flat.size(0) == 2, "target_max_conc must have 2 elements");
    const float* target_max_conc_ptr = target_max_conc_flat.data_ptr<float>();

    // Get stain_matrix as device pointer (ensure contiguous)
    torch::Tensor stain_matrix_contig = stain_matrix.contiguous();
    const float* stain_matrix_ptr     = stain_matrix_contig.data_ptr<float>();

    // Pre-allocate output tensor
    torch::Tensor output = torch::empty({N, C, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device()));

    // Thread configuration
    int num_threads = THREADS_PER_BLOCK;

    // Batch process initial operations across all images
    // Allocate batched buffers
    torch::Tensor rgb_flat_batched = torch::empty({N, num_pixels, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor od_flat_batched  = torch::empty({N, num_pixels, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor min_od_batched   = torch::empty({N, num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    // Reshape and scale all images: (N, 3, H, W) -> (N, H*W, 3) and scale to [0, 255]
    int total_pixels       = N * num_pixels;
    int num_blocks_batched = (total_pixels + num_threads - 1) / num_threads;
    reshape_and_scale_kernel_batched<<<num_blocks_batched, num_threads>>>(images_float.data_ptr<float>(), rgb_flat_batched.data_ptr<float>(), H, W, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in reshape_and_scale_kernel_batched: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Convert RGB to OD for all images
    rgb_to_od_kernel_batched<<<num_blocks_batched, num_threads>>>(rgb_flat_batched.data_ptr<float>(), od_flat_batched.data_ptr<float>(), Io, num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in rgb_to_od_kernel_batched: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Compute minimum OD for all images
    compute_min_od_mask_kernel_batched<<<num_blocks_batched, num_threads>>>(od_flat_batched.data_ptr<float>(), min_od_batched.data_ptr<float>(), num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_min_od_mask_kernel_batched: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Step 1: Pre-compute maximum filtered count across all images
    torch::Tensor num_filtered_array = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(images_float.device())).contiguous();

    // Count filtered pixels for each image
    count_filtered_pixels_all_images_kernel<<<N, num_threads>>>(min_od_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), beta, num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in count_filtered_pixels_all_images_kernel: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Find maximum filtered count
    torch::Tensor max_filtered_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(images_float.device()));
    find_max_kernel<<<1, num_threads>>>(num_filtered_array.data_ptr<int>(), max_filtered_tensor.data_ptr<int>(), N);
    cudaDeviceSynchronize();

    int max_num_filtered;
    cudaMemcpy(&max_num_filtered, max_filtered_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);

    // Step 2: Allocate padded buffers for batched processing
    torch::Tensor od_filtered_batched = torch::zeros({N, max_num_filtered, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    // Step 2: Compact filtered pixels for all images
    compact_filtered_batched_kernel<<<N, num_threads>>>(od_flat_batched.data_ptr<float>(), min_od_batched.data_ptr<float>(), od_filtered_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), beta, num_pixels, max_num_filtered, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compact_filtered_batched_kernel: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Step 3: Compute covariance matrices for all images in parallel
    torch::Tensor cov_batched = torch::empty({N, 9}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    compute_covariance_batched_kernel<<<N, num_threads>>>(od_filtered_batched.data_ptr<float>(), cov_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), max_num_filtered, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_covariance_batched_kernel: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Step 4: Eigenvalue decomposition for all images using cuSOLVER batched API
    // Prepare pointers for batched eigendecomposition
    torch::Tensor eigvecs_batched = torch::empty({N, 9}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor eigvals_batched = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    // Copy cov to eigvecs (will be overwritten with eigenvectors)
    cudaMemcpy(eigvecs_batched.data_ptr<float>(), cov_batched.data_ptr<float>(), N * 9 * sizeof(float), cudaMemcpyDeviceToDevice);

    // Prepare array of pointers for eigendecomposition
    std::vector<float*> eigvecs_ptrs(N);
    std::vector<float*> eigvals_ptrs(N);
    for (int i = 0; i < N; i++) {
        eigvecs_ptrs[i] = eigvecs_batched.data_ptr<float>() + i * 9;
        eigvals_ptrs[i] = eigvals_batched.data_ptr<float>() + i * 3;
    }

    // Compute workspace size
    int lwork_syevd               = 0;
    cusolverStatus_t status_batch = cusolverDnSsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 3, eigvecs_batched.data_ptr<float>(), 3, eigvals_batched.data_ptr<float>(), &lwork_syevd);
    if (status_batch != CUSOLVER_STATUS_SUCCESS) { TORCH_CHECK(false, "cusolverDnSsyevd_bufferSize failed"); }

    torch::Tensor workspace_syevd = torch::empty({lwork_syevd * N}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor devInfo_syevd   = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(images_float.device())).contiguous();

    // Run eigendecomposition for all images
    for (int n = 0; n < N; n++) {
        status_batch = cusolverDnSsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 3, eigvecs_ptrs[n], 3, eigvals_ptrs[n], workspace_syevd.data_ptr<float>() + n * lwork_syevd, lwork_syevd, devInfo_syevd.data_ptr<int>() + n);
        if (status_batch != CUSOLVER_STATUS_SUCCESS) { TORCH_CHECK(false, "cusolverDnSsyevd failed for image ", n); }
    }
    cudaDeviceSynchronize();

    // Extract last 2 eigenvectors for all images
    torch::Tensor eigvecs_2d_batched = torch::empty({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    extract_eigvecs_2d_batched_kernel<<<N, num_threads>>>(eigvecs_batched.data_ptr<float>(), eigvecs_2d_batched.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    // Step 5: Compute That and phi for all images
    torch::Tensor That_batched = torch::empty({N, max_num_filtered, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor phi_batched  = torch::empty({N, max_num_filtered}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    compute_That_batched_kernel<<<N, num_threads>>>(od_filtered_batched.data_ptr<float>(), eigvecs_2d_batched.data_ptr<float>(), That_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), max_num_filtered, N);
    cudaDeviceSynchronize();

    compute_phi_batched_kernel<<<N, num_threads>>>(That_batched.data_ptr<float>(), phi_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), max_num_filtered, N);
    cudaDeviceSynchronize();

    // Compute percentiles for all images (still requires CPU for simplicity)
    torch::Tensor min_phi_array = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor max_phi_array = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    std::vector<int> num_filtered_host(N);
    cudaMemcpy(num_filtered_host.data(), num_filtered_array.data_ptr<int>(), N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int n = 0; n < N; n++) {
        const float* phi_ptr = phi_batched.data_ptr<float>() + n * max_num_filtered;
        min_phi_array[n]     = compute_percentile_cuda(phi_ptr, num_filtered_host[n], alpha);
        max_phi_array[n]     = compute_percentile_cuda(phi_ptr, num_filtered_host[n], 100.0f - alpha);
    }

    // Copy percentiles to device
    torch::Tensor min_phi_device = min_phi_array.to(images_float.device());
    torch::Tensor max_phi_device = max_phi_array.to(images_float.device());

    // Step 6: Compute stain vectors for all images
    torch::Tensor HE_source_batched = torch::empty({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    compute_stain_vectors_batched_kernel<<<N, num_threads>>>(eigvecs_2d_batched.data_ptr<float>(), min_phi_device.data_ptr<float>(), max_phi_device.data_ptr<float>(), HE_source_batched.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    // Step 7: SVD for HE_source matrices (batched)
    torch::Tensor U_batched  = torch::empty({N, 9}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor S_batched  = torch::empty({N, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor VT_batched = torch::empty({N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    // Compute workspace size for SVD
    int lwork_svd               = 0;
    cusolverStatus_t status_svd = cusolverDnSgesvd_bufferSize(cusolver_handle, 3, 2, &lwork_svd);
    if (status_svd != CUSOLVER_STATUS_SUCCESS) { TORCH_CHECK(false, "cusolverDnSgesvd_bufferSize failed"); }

    torch::Tensor workspace_svd = torch::empty({lwork_svd * N}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor devInfo_svd   = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(images_float.device())).contiguous();

    // Run SVD for each HE_source
    for (int n = 0; n < N; n++) {
        float* HE_ptr = HE_source_batched.data_ptr<float>() + n * 6;
        float* U_ptr  = U_batched.data_ptr<float>() + n * 9;
        float* S_ptr  = S_batched.data_ptr<float>() + n * 2;
        float* VT_ptr = VT_batched.data_ptr<float>() + n * 4;

        status_svd = cusolverDnSgesvd(cusolver_handle, 'A', 'A', 3, 2, HE_ptr, 3, S_ptr, U_ptr, 3, VT_ptr, 2, workspace_svd.data_ptr<float>() + n * lwork_svd, lwork_svd, nullptr, devInfo_svd.data_ptr<int>() + n);
        if (status_svd != CUSOLVER_STATUS_SUCCESS) { TORCH_CHECK(false, "cusolverDnSgesvd failed for image ", n); }
    }
    cudaDeviceSynchronize();

    // Reshape OD for all images: (N, H*W, 3) -> (N, 3, H*W)
    torch::Tensor od_all_batched = torch::empty({N, 3, num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    int num_blocks_reshape       = (N * num_pixels + num_threads - 1) / num_threads;
    reshape_od_to_3xN_batched_kernel<<<num_blocks_reshape, num_threads>>>(od_flat_batched.data_ptr<float>(), od_all_batched.data_ptr<float>(), num_pixels, N);
    cudaDeviceSynchronize();

    // Step 7: Compute concentrations for all images
    torch::Tensor concentrations_batched = torch::empty({N, 2, num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    size_t shared_mem_size = 0;  // Can allocate if needed
    compute_concentrations_batched_kernel<<<N, num_threads, shared_mem_size>>>(U_batched.data_ptr<float>(), S_batched.data_ptr<float>(), VT_batched.data_ptr<float>(), od_all_batched.data_ptr<float>(), concentrations_batched.data_ptr<float>(), num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_concentrations_batched_kernel: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Compute max concentrations (99th percentile) for all images
    torch::Tensor max_conc_array = torch::empty({N, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    for (int n = 0; n < N; n++) {
        const float* conc_ptr = concentrations_batched.data_ptr<float>() + n * 2 * num_pixels;
        float max_conc_0      = compute_percentile_cuda(conc_ptr, num_pixels, 99.0f);
        float max_conc_1      = compute_percentile_cuda(conc_ptr + num_pixels, num_pixels, 99.0f);

        // Avoid division by zero
        max_conc_array[n][0] = std::max(max_conc_0, 1.0f);
        max_conc_array[n][1] = std::max(max_conc_1, 1.0f);
    }

    // Copy to device
    torch::Tensor max_conc_device = max_conc_array.to(images_float.device());

    // Step 8: Normalize concentrations for all images
    int total_conc_elements = N * 2 * num_pixels;
    int num_blocks_norm     = (total_conc_elements + num_threads - 1) / num_threads;
    normalize_concentrations_batched_kernel<<<num_blocks_norm, num_threads>>>(concentrations_batched.data_ptr<float>(), max_conc_device.data_ptr<float>(), target_max_conc_ptr, num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in normalize_concentrations_batched_kernel: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Reconstruct OD for all images
    torch::Tensor od_recon_batched = torch::empty({N, 3, num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    int total_od_elements          = N * 3 * num_pixels;
    int num_blocks_recon           = (total_od_elements + num_threads - 1) / num_threads;
    compute_od_recon_batched_kernel<<<num_blocks_recon, num_threads>>>(stain_matrix_ptr, concentrations_batched.data_ptr<float>(), od_recon_batched.data_ptr<float>(), num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_od_recon_batched_kernel: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Convert OD back to RGB and reshape to output format
    torch::Tensor rgb_batched = torch::empty({N, num_pixels, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    int num_blocks_rgb        = (N * num_pixels + num_threads - 1) / num_threads;
    od_to_rgb_transpose_batched_kernel<<<num_blocks_rgb, num_threads>>>(od_recon_batched.data_ptr<float>(), rgb_batched.data_ptr<float>(), Io, num_pixels, N);
    cudaDeviceSynchronize();

    // Reshape to final output format: (N, H*W, 3) -> (N, 3, H, W)
    reshape_output_kernel_batched<<<num_blocks_batched, num_threads>>>(rgb_batched.data_ptr<float>(), output.data_ptr<float>(), H, W, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in reshape_output_kernel_batched: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Final clamp to [0, 255]
    int total_output_elements = N * C * H * W;
    int num_blocks_output     = (total_output_elements + num_threads - 1) / num_threads;
    clamp_kernel<<<num_blocks_output, num_threads>>>(output.data_ptr<float>(), 0.0f, 255.0f, total_output_elements);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in final clamp_kernel: ", cudaGetErrorString(err)); }
    cudaDeviceSynchronize();

    // Convert to original dtype
    torch::ScalarType original_dtype = input_images.scalar_type();
    output                           = output.to(original_dtype);

    return output;
}
