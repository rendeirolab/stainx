// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Macenko normalization CUDA kernel implementation.
 *
 * This file contains pure CUDA kernels for Macenko stain normalization using SVD.
 * All computations are performed directly in CUDA using cuBLAS and cuSOLVER.
 * Uses batched cuBLAS operations for parallel processing.
 * These kernels have no PyTorch dependencies and can be used by any CUDA interface.
 */

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

// GPU kernel to compute min/max for normalization
__global__ void compute_minmax_kernel(const float* data, int n, float* min_val, float* max_val) {
    __shared__ float shared_min[THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float shared_max[THREADS_PER_BLOCK / WARP_SIZE];

    int tid         = threadIdx.x;
    float local_min = INFINITY;
    float local_max = -INFINITY;

    // Each thread finds local min/max
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float val = data[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_min = __shfl_down_sync(0xffffffff, local_min, offset);
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        local_min       = fminf(local_min, other_min);
        local_max       = fmaxf(local_max, other_max);
    }

    // First thread in each warp writes to shared memory
    int warp_id = tid / WARP_SIZE;
    if (tid % WARP_SIZE == 0) {
        shared_min[warp_id] = local_min;
        shared_max[warp_id] = local_max;
    }
    __syncthreads();

    // Final reduction in first warp
    if (tid < THREADS_PER_BLOCK / WARP_SIZE) {
        local_min = shared_min[tid];
        local_max = shared_max[tid];
    } else {
        local_min = INFINITY;
        local_max = -INFINITY;
    }

    if (warp_id == 0) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_min = __shfl_down_sync(0xffffffff, local_min, offset);
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            local_min       = fminf(local_min, other_min);
            local_max       = fmaxf(local_max, other_max);
        }

        if (tid == 0) {
            atomicMin((int*) min_val, __float_as_int(local_min));
            atomicMax((int*) max_val, __float_as_int(local_max));
        }
    }
}

// GPU kernel to build histogram
__global__ void build_histogram_kernel(const float* data, int n, float min_val, float max_val, int* histogram, int num_bins) {
    __shared__ int shared_hist[1024];  // Support up to 1024 bins

    // Initialize shared histogram
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) { shared_hist[i] = 0; }
    __syncthreads();

    // Build histogram in shared memory
    float range = max_val - min_val;
    if (range < 1e-10f) range = 1e-10f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = data[i];
        int bin   = (int) ((val - min_val) / range * num_bins);
        if (bin < 0) bin = 0;
        if (bin >= num_bins) bin = num_bins - 1;
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();

    // Write shared histogram to global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) { atomicAdd(&histogram[i], shared_hist[i]); }
}

// GPU kernel to find percentile from histogram
__global__ void find_percentile_from_histogram_kernel(const int* histogram, int num_bins, float min_val, float max_val, int n, float percentile, float* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int target_count = (int) (percentile / 100.0f * (n - 1) + 1);
        int cumsum       = 0;
        int bin          = 0;

        for (int i = 0; i < num_bins; i++) {
            cumsum += histogram[i];
            if (cumsum >= target_count) {
                bin = i;
                break;
            }
        }

        // Interpolate within bin
        float range     = max_val - min_val;
        float bin_width = range / num_bins;
        *result         = min_val + (bin + 0.5f) * bin_width;
    }
}

// GPU-based percentile computation using histogram method
void compute_percentile_gpu(const float* data_device, int n, float percentile, float* result_device, cudaStream_t stream = 0) {
    const int num_bins    = 1024;
    const int num_threads = THREADS_PER_BLOCK;
    const int num_blocks  = min(256, (n + num_threads - 1) / num_threads);

    // Allocate temporary buffers
    float *min_val_device, *max_val_device;
    int* histogram_device;

    cudaMalloc(&min_val_device, sizeof(float));
    cudaMalloc(&max_val_device, sizeof(float));
    cudaMalloc(&histogram_device, num_bins * sizeof(int));

    // Initialize min/max
    float init_min = INFINITY;
    float init_max = -INFINITY;
    cudaMemcpy(min_val_device, &init_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(max_val_device, &init_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(histogram_device, 0, num_bins * sizeof(int));

    // Compute min/max
    compute_minmax_kernel<<<num_blocks, num_threads, 0, stream>>>(data_device, n, min_val_device, max_val_device);

    // Build histogram
    float min_val_host, max_val_host;
    cudaMemcpy(&min_val_host, min_val_device, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_val_host, max_val_device, sizeof(float), cudaMemcpyDeviceToHost);

    build_histogram_kernel<<<num_blocks, num_threads, 0, stream>>>(data_device, n, min_val_host, max_val_host, histogram_device, num_bins);

    // Find percentile
    find_percentile_from_histogram_kernel<<<1, 1, 0, stream>>>(histogram_device, num_bins, min_val_host, max_val_host, n, percentile, result_device);

    // Cleanup
    cudaFree(min_val_device);
    cudaFree(max_val_device);
    cudaFree(histogram_device);
}

// Batched GPU percentile computation for multiple arrays
__global__ void compute_percentiles_batched_kernel(const float* data_batched, const int* counts, int max_count, float percentile, float* results, int N) {
    int n = blockIdx.x;  // One block per batch
    if (n >= N) return;

    const int num_bins = 256;  // Reduced for per-block computation
    __shared__ int histogram[256];
    __shared__ float min_val;
    __shared__ float max_val;

    const float* data = data_batched + n * max_count;
    int count         = counts[n];

    // Initialize shared memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) { histogram[i] = 0; }

    if (threadIdx.x == 0) {
        min_val = INFINITY;
        max_val = -INFINITY;
    }
    __syncthreads();

    // Compute min/max in parallel
    float local_min = INFINITY;
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        float val = data[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    // Reduce to single min/max
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        float other_min = __shfl_down_sync(0xffffffff, local_min, offset);
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        local_min       = fminf(local_min, other_min);
        local_max       = fmaxf(local_max, other_max);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        atomicMin((int*) &min_val, __float_as_int(local_min));
        atomicMax((int*) &max_val, __float_as_int(local_max));
    }
    __syncthreads();

    // Build histogram
    float range = max_val - min_val;
    if (range < 1e-10f) range = 1e-10f;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        float val = data[i];
        int bin   = (int) ((val - min_val) / range * num_bins);
        if (bin < 0) bin = 0;
        if (bin >= num_bins) bin = num_bins - 1;
        atomicAdd(&histogram[bin], 1);
    }
    __syncthreads();

    // Find percentile (single thread)
    if (threadIdx.x == 0) {
        int target_count = (int) (percentile / 100.0f * (count - 1) + 1);
        int cumsum       = 0;
        int bin          = 0;

        for (int i = 0; i < num_bins; i++) {
            cumsum += histogram[i];
            if (cumsum >= target_count) {
                bin = i;
                break;
            }
        }

        // Interpolate within bin
        float bin_width = range / num_bins;
        results[n]      = min_val + (bin + 0.5f) * bin_width;
    }
}

// Helper to get cuSOLVER handle
cusolverDnHandle_t get_cusolver_handle() {
    static cusolverDnHandle_t handle = nullptr;
    if (handle == nullptr) { cusolverDnCreate(&handle); }
    return handle;
}

