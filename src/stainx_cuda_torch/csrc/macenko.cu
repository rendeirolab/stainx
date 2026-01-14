// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * PyTorch wrapper for Macenko normalization CUDA kernels.
 *
 * This file provides PyTorch tensor interfaces that call the pure CUDA kernels
 * from the main csrc/ directory.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

// Include the pure CUDA kernels from the main csrc directory
// Project root is in include_dirs, so we can include from csrc/
#include "csrc/macenko.cu"

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

torch::Tensor macenko_cuda(torch::Tensor input_images, torch::Tensor stain_matrix, torch::Tensor target_max_conc) {
    // Check inputs
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(stain_matrix.is_cuda(), "stain_matrix must be a CUDA tensor");
    TORCH_CHECK(target_max_conc.is_cuda(), "target_max_conc must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W), got ", input_images.dim(), "D tensor with shape ", input_images.sizes());
    TORCH_CHECK(input_images.size(1) == 3, "input_images must have 3 channels, got ", input_images.size(1), " channels");
    TORCH_CHECK(stain_matrix.size(0) == 3 && stain_matrix.size(1) == 2, "stain_matrix must have shape (3, 2), got shape ", stain_matrix.sizes());

    // Check that tensors are on the same device
    TORCH_CHECK(input_images.device() == stain_matrix.device(),
                "input_images and stain_matrix must be on the same device. "
                "input_images device: ",
                input_images.device(),
                ", stain_matrix device: ",
                stain_matrix.device());
    TORCH_CHECK(input_images.device() == target_max_conc.device(),
                "input_images and target_max_conc must be on the same device. "
                "input_images device: ",
                input_images.device(),
                ", target_max_conc device: ",
                target_max_conc.device());

    // Get cuSOLVER handle
    cusolverDnHandle_t cusolver_handle = get_cusolver_handle();
    cudaError_t err                    = cudaSuccess;  // Declare once at function level

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
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in reshape_and_scale_kernel_batched: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after reshape_and_scale_kernel_batched: ", cudaGetErrorString(err)); }

    // Convert RGB to OD for all images
    rgb_to_od_kernel_batched<<<num_blocks_batched, num_threads>>>(rgb_flat_batched.data_ptr<float>(), od_flat_batched.data_ptr<float>(), Io, num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in rgb_to_od_kernel_batched: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after rgb_to_od_kernel_batched: ", cudaGetErrorString(err)); }

    // Compute minimum OD for all images
    compute_min_od_mask_kernel_batched<<<num_blocks_batched, num_threads>>>(od_flat_batched.data_ptr<float>(), min_od_batched.data_ptr<float>(), num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_min_od_mask_kernel_batched: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_min_od_mask_kernel_batched: ", cudaGetErrorString(err)); }

    // Step 1: Pre-compute maximum filtered count across all images
    torch::Tensor num_filtered_array = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(images_float.device())).contiguous();

    // Count filtered pixels for each image
    count_filtered_pixels_all_images_kernel<<<N, num_threads>>>(min_od_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), beta, num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in count_filtered_pixels_all_images_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after count_filtered_pixels_all_images_kernel: ", cudaGetErrorString(err)); }

    // Find maximum filtered count
    torch::Tensor max_filtered_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(images_float.device()));
    find_max_kernel<<<1, num_threads>>>(num_filtered_array.data_ptr<int>(), max_filtered_tensor.data_ptr<int>(), N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in find_max_kernel: ", cudaGetErrorString(err)); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after find_max_kernel: ", cudaGetErrorString(err)); }

    int max_num_filtered;
    err = cudaMemcpy(&max_num_filtered, max_filtered_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error copying max_num_filtered from device to host: ", cudaGetErrorString(err)); }

    // Step 2: Allocate padded buffers for batched processing
    torch::Tensor od_filtered_batched = torch::zeros({N, max_num_filtered, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    // Step 2: Compact filtered pixels for all images
    compact_filtered_batched_kernel<<<N, num_threads>>>(od_flat_batched.data_ptr<float>(), min_od_batched.data_ptr<float>(), od_filtered_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), beta, num_pixels, max_num_filtered, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compact_filtered_batched_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compact_filtered_batched_kernel: ", cudaGetErrorString(err)); }

    // Step 3: Compute covariance matrices for all images in parallel
    torch::Tensor cov_batched = torch::empty({N, 9}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    compute_covariance_batched_kernel<<<N, num_threads>>>(od_filtered_batched.data_ptr<float>(), cov_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), max_num_filtered, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_covariance_batched_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_covariance_batched_kernel: ", cudaGetErrorString(err)); }

    // Step 4: Eigenvalue decomposition for all images using cuSOLVER with streams for parallelization
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

    // Run eigendecomposition for all images using CUDA streams for parallelization
    const int num_streams = std::min(N, 16);  // Use up to 16 streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) { cudaStreamCreate(&streams[i]); }

    for (int n = 0; n < N; n++) {
        int stream_id = n % num_streams;
        cusolverDnSetStream(cusolver_handle, streams[stream_id]);
        status_batch = cusolverDnSsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 3, eigvecs_ptrs[n], 3, eigvals_ptrs[n], workspace_syevd.data_ptr<float>() + n * lwork_syevd, lwork_syevd, devInfo_syevd.data_ptr<int>() + n);
        if (status_batch != CUSOLVER_STATUS_SUCCESS) { TORCH_CHECK(false, "cusolverDnSsyevd failed for image ", n); }
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        err = cudaStreamSynchronize(streams[i]);
        if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing stream ", i, " after eigendecomposition: ", cudaGetErrorString(err)); }
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after eigendecomposition: ", cudaGetErrorString(err)); }

    // Cleanup streams
    for (int i = 0; i < num_streams; i++) { cudaStreamDestroy(streams[i]); }

    // Extract last 2 eigenvectors for all images
    torch::Tensor eigvecs_2d_batched = torch::empty({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    extract_eigvecs_2d_batched_kernel<<<N, num_threads>>>(eigvecs_batched.data_ptr<float>(), eigvecs_2d_batched.data_ptr<float>(), N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in extract_eigvecs_2d_batched_kernel: ", cudaGetErrorString(err)); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after extract_eigvecs_2d_batched_kernel: ", cudaGetErrorString(err)); }

    // Step 5: Compute That and phi for all images
    torch::Tensor That_batched = torch::empty({N, max_num_filtered, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor phi_batched  = torch::empty({N, max_num_filtered}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    compute_That_batched_kernel<<<N, num_threads>>>(od_filtered_batched.data_ptr<float>(), eigvecs_2d_batched.data_ptr<float>(), That_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), max_num_filtered, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_That_batched_kernel: ", cudaGetErrorString(err)); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_That_batched_kernel: ", cudaGetErrorString(err)); }

    compute_phi_batched_kernel<<<N, num_threads>>>(That_batched.data_ptr<float>(), phi_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), max_num_filtered, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_phi_batched_kernel: ", cudaGetErrorString(err)); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_phi_batched_kernel: ", cudaGetErrorString(err)); }

    // Compute percentiles for all images using GPU
    torch::Tensor min_phi_device = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device()));
    torch::Tensor max_phi_device = torch::empty({N}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device()));

    // Launch batched percentile kernels
    compute_percentiles_batched_kernel<<<N, num_threads>>>(phi_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), max_num_filtered, alpha, min_phi_device.data_ptr<float>(), N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_percentiles_batched_kernel (min_phi): ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }

    compute_percentiles_batched_kernel<<<N, num_threads>>>(phi_batched.data_ptr<float>(), num_filtered_array.data_ptr<int>(), max_num_filtered, 100.0f - alpha, max_phi_device.data_ptr<float>(), N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_percentiles_batched_kernel (max_phi): ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_percentiles_batched_kernel: ", cudaGetErrorString(err)); }

    // Step 6: Compute stain vectors for all images
    torch::Tensor HE_source_batched = torch::empty({N, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    compute_stain_vectors_batched_kernel<<<N, num_threads>>>(eigvecs_2d_batched.data_ptr<float>(), min_phi_device.data_ptr<float>(), max_phi_device.data_ptr<float>(), HE_source_batched.data_ptr<float>(), N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_stain_vectors_batched_kernel: ", cudaGetErrorString(err)); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_stain_vectors_batched_kernel: ", cudaGetErrorString(err)); }

    // Step 7: SVD for HE_source matrices
    torch::Tensor U_batched  = torch::empty({N, 9}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor S_batched  = torch::empty({N, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor VT_batched = torch::empty({N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    // Compute workspace size for SVD
    int lwork_svd               = 0;
    cusolverStatus_t status_svd = cusolverDnSgesvd_bufferSize(cusolver_handle, 3, 2, &lwork_svd);
    if (status_svd != CUSOLVER_STATUS_SUCCESS) { TORCH_CHECK(false, "cusolverDnSgesvd_bufferSize failed"); }

    torch::Tensor workspace_svd = torch::empty({lwork_svd * N}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    torch::Tensor devInfo_svd   = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(images_float.device())).contiguous();

    // Run SVD for each HE_source using CUDA streams for parallelization (similar to eigendecomposition)
    const int num_streams_svd = std::min(N, 16);  // Use up to 16 streams
    std::vector<cudaStream_t> streams_svd(num_streams_svd);
    for (int i = 0; i < num_streams_svd; i++) { cudaStreamCreate(&streams_svd[i]); }

    for (int n = 0; n < N; n++) {
        int stream_id = n % num_streams_svd;
        cusolverDnSetStream(cusolver_handle, streams_svd[stream_id]);
        float* HE_ptr = HE_source_batched.data_ptr<float>() + n * 6;
        float* U_ptr  = U_batched.data_ptr<float>() + n * 9;
        float* S_ptr  = S_batched.data_ptr<float>() + n * 2;
        float* VT_ptr = VT_batched.data_ptr<float>() + n * 4;

        status_svd = cusolverDnSgesvd(cusolver_handle, 'A', 'A', 3, 2, HE_ptr, 3, S_ptr, U_ptr, 3, VT_ptr, 2, workspace_svd.data_ptr<float>() + n * lwork_svd, lwork_svd, nullptr, devInfo_svd.data_ptr<int>() + n);
        if (status_svd != CUSOLVER_STATUS_SUCCESS) { TORCH_CHECK(false, "cusolverDnSgesvd failed for image ", n); }
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams_svd; i++) {
        err = cudaStreamSynchronize(streams_svd[i]);
        if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing stream ", i, " after SVD: ", cudaGetErrorString(err)); }
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after SVD: ", cudaGetErrorString(err)); }

    // Cleanup streams
    for (int i = 0; i < num_streams_svd; i++) { cudaStreamDestroy(streams_svd[i]); }

    // Reshape OD for all images: (N, H*W, 3) -> (N, 3, H*W)
    torch::Tensor od_all_batched = torch::empty({N, 3, num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    int num_blocks_reshape       = (N * num_pixels + num_threads - 1) / num_threads;
    reshape_od_to_3xN_batched_kernel<<<num_blocks_reshape, num_threads>>>(od_flat_batched.data_ptr<float>(), od_all_batched.data_ptr<float>(), num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in reshape_od_to_3xN_batched_kernel: ", cudaGetErrorString(err)); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after reshape_od_to_3xN_batched_kernel: ", cudaGetErrorString(err)); }

    // Step 7: Compute concentrations for all images
    torch::Tensor concentrations_batched = torch::empty({N, 2, num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();

    size_t shared_mem_size = 0;  // Can allocate if needed
    compute_concentrations_batched_kernel<<<N, num_threads, shared_mem_size>>>(U_batched.data_ptr<float>(), S_batched.data_ptr<float>(), VT_batched.data_ptr<float>(), od_all_batched.data_ptr<float>(), concentrations_batched.data_ptr<float>(), num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_concentrations_batched_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_concentrations_batched_kernel: ", cudaGetErrorString(err)); }

    // Compute max concentrations (99th percentile) for all images using GPU
    torch::Tensor max_conc_device  = torch::empty({N, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device()));
    torch::Tensor counts_per_pixel = torch::full({N}, num_pixels, torch::TensorOptions().dtype(torch::kInt32).device(images_float.device()));

    // Compute percentile for channel 0
    compute_percentiles_batched_kernel<<<N, num_threads>>>(concentrations_batched.data_ptr<float>(), counts_per_pixel.data_ptr<int>(), num_pixels, 99.0f, max_conc_device.data_ptr<float>(), N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_percentiles_batched_kernel (conc channel 0): ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }

    // Compute percentile for channel 1 (offset by num_pixels in concentrations_batched)
    compute_percentiles_batched_kernel<<<N, num_threads>>>(concentrations_batched.data_ptr<float>() + num_pixels, counts_per_pixel.data_ptr<int>(), num_pixels, 99.0f, max_conc_device.data_ptr<float>() + N, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_percentiles_batched_kernel (conc channel 1): ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_percentiles_batched_kernel (concentrations): ", cudaGetErrorString(err)); }

    // Clamp to avoid division by zero
    clamp_kernel<<<(N * 2 + num_threads - 1) / num_threads, num_threads>>>(max_conc_device.data_ptr<float>(), 1.0f, 1e10f, N * 2);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in clamp_kernel: ", cudaGetErrorString(err)); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after clamp_kernel: ", cudaGetErrorString(err)); }

    // Step 8: Normalize concentrations for all images
    int total_conc_elements = N * 2 * num_pixels;
    int num_blocks_norm     = (total_conc_elements + num_threads - 1) / num_threads;
    normalize_concentrations_batched_kernel<<<num_blocks_norm, num_threads>>>(concentrations_batched.data_ptr<float>(), max_conc_device.data_ptr<float>(), target_max_conc_ptr, num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in normalize_concentrations_batched_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after normalize_concentrations_batched_kernel: ", cudaGetErrorString(err)); }

    // Reconstruct OD for all images
    torch::Tensor od_recon_batched = torch::empty({N, 3, num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    int total_od_elements          = N * 3 * num_pixels;
    int num_blocks_recon           = (total_od_elements + num_threads - 1) / num_threads;
    compute_od_recon_batched_kernel<<<num_blocks_recon, num_threads>>>(stain_matrix_ptr, concentrations_batched.data_ptr<float>(), od_recon_batched.data_ptr<float>(), num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in compute_od_recon_batched_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after compute_od_recon_batched_kernel: ", cudaGetErrorString(err)); }

    // Convert OD back to RGB and reshape to output format
    torch::Tensor rgb_batched = torch::empty({N, num_pixels, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(images_float.device())).contiguous();
    int num_blocks_rgb        = (N * num_pixels + num_threads - 1) / num_threads;
    od_to_rgb_transpose_batched_kernel<<<num_blocks_rgb, num_threads>>>(od_recon_batched.data_ptr<float>(), rgb_batched.data_ptr<float>(), Io, num_pixels, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in od_to_rgb_transpose_batched_kernel: ", cudaGetErrorString(err)); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after od_to_rgb_transpose_batched_kernel: ", cudaGetErrorString(err)); }

    // Reshape to final output format: (N, H*W, 3) -> (N, 3, H, W)
    reshape_output_kernel_batched<<<num_blocks_batched, num_threads>>>(rgb_batched.data_ptr<float>(), output.data_ptr<float>(), H, W, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in reshape_output_kernel_batched: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after reshape_output_kernel_batched: ", cudaGetErrorString(err)); }

    // Final clamp to [0, 255]
    int total_output_elements = N * C * H * W;
    int num_blocks_output     = (total_output_elements + num_threads - 1) / num_threads;
    clamp_kernel<<<num_blocks_output, num_threads>>>(output.data_ptr<float>(), 0.0f, 255.0f, total_output_elements);
    err = cudaGetLastError();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error in final clamp_kernel: ", cudaGetErrorString(err), ". Input shape: (", N, ", ", C, ", ", H, ", ", W, ")"); }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { TORCH_CHECK(false, "CUDA error synchronizing device after final clamp_kernel: ", cudaGetErrorString(err)); }

    // Convert to original dtype
    torch::ScalarType original_dtype = input_images.scalar_type();
    output                           = output.to(original_dtype);

    return output;
}
