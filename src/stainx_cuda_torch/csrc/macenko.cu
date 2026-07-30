// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Macenko normalization for CUDA tensors — fully on-GPU, two precision modes.
 *
 * Earlier versions offloaded the 3x3 OD covariance, its eigendecomposition and
 * the concentration least-squares to the CPU: CUDA float32 covariance reductions
 * diverge on near-degenerate spectra and flip the stain plane, so parity with
 * torchstain was only reachable via LAPACK on the host. That host round-trip
 * forces a full device sync per image and dominates the runtime.
 *
 * This implementation provides two selectable modes:
 *
 *   Stable (default) — fp64 cov + fp64 analytic 3x3 eigh for the stain-plane
 *   solve (numerically robust against flips on near-degenerate spectra), with
 *   everything else in fp32.
 *
 *   Fast — fp32 cov + fp32 eigh for the plane solve, fp16 for large pixel
 *   tensors (projection bmm, phi sort, reconstruct matmul), and fp32 for the
 *   2x2 concentration solve.  No fp64 at all; ~1.2–1.3× faster on RTX A6000
 *   while staying within the same MAE tolerance vs torchstain on synthetic H&E.
 *
 * Both modes share the pure templated per-image reduction kernel
 * (macenko_cov_kernel<T> from csrc/macenko.cu) with warp-level tree reduction,
 * the same templated closed-form 3x3 eigensolver (analytic_eigh_sym3<T>),
 * and batched ATen ops for the downstream pipeline.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cstdint>
#include <limits>

// Pure CUDA kernels (cov + analytic 3x3 eigh) — no PyTorch deps
#include "csrc/macenko.cu"

using namespace torch::indexing;

namespace {

// Nearest-rank percentile (matches torchstain: k = 1 + round(.01*q*(n-1)),
// 0-based index = round(.01*q*(n-1))) over sorted rows, per image.
// Works with any dtype for sorted_asc and counts (index arithmetic is always
// done in fp64 internally, so counts dtype does not need to match sorted_asc).
torch::Tensor gather_percentile(const torch::Tensor& sorted_asc, const torch::Tensor& counts, double q) {
    // counts and sorted_asc may be fp32 or fp64; do arithmetic in float64 for
    // the index computation to match torchstain's exact formula regardless of
    // the tensor dtype.
    auto idx = torch::round(0.01 * q * (counts.to(torch::kFloat64) - 1.0)).to(torch::kLong).clamp_min(0).unsqueeze(1);
    return sorted_asc.gather(1, idx);  // (N, 1)
}

}  // namespace

// ---------------------------------------------------------------------------
// Shared Macenko pipeline with two precision modes:
//   Stable — fp64 cov + eigh, fp32 downstream.
//   Fast   — fp32 cov + eigh, fp16 large pixels, fp32 2×2 solve.
// ---------------------------------------------------------------------------
static torch::Tensor macenko_cuda_impl(torch::Tensor input_images, torch::Tensor stain_matrix, torch::Tensor target_max_conc, bool fast) {
    TORCH_CHECK(input_images.is_cuda(), "input_images must be a CUDA tensor");
    TORCH_CHECK(stain_matrix.is_cuda(), "stain_matrix must be a CUDA tensor");
    TORCH_CHECK(target_max_conc.is_cuda(), "target_max_conc must be a CUDA tensor");
    TORCH_CHECK(input_images.dim() == 4, "input_images must be 4D (N, C, H, W), got ", input_images.dim(), "D");
    TORCH_CHECK(input_images.size(1) == 3, "input_images must have 3 channels, got ", input_images.size(1));
    TORCH_CHECK(stain_matrix.size(0) == 3 && stain_matrix.size(1) == 2, "stain_matrix must have shape (3, 2)");
    TORCH_CHECK(input_images.device() == stain_matrix.device(), "input_images and stain_matrix device mismatch");
    TORCH_CHECK(input_images.device() == target_max_conc.device(), "input_images and target_max_conc device mismatch");

    // uint8 is [0,255]; float is assumed already [0,1] (no max()>1 heuristic / sync).
    torch::Tensor images_float;
    if (input_images.dtype() == torch::kUInt8) {
        images_float = input_images.to(torch::kFloat32) / 255.0f;
    } else {
        images_float = input_images.to(torch::kFloat32);
    }

    const int64_t N = images_float.size(0);
    const int64_t H = images_float.size(2);
    const int64_t W = images_float.size(3);
    const int64_t P = H * W;

    constexpr double beta  = 0.15;
    constexpr double alpha = 1.0;
    constexpr float Io_f   = 240.0f;
    constexpr float beta_f = 0.15f;

    auto f32 = images_float.options().dtype(torch::kFloat32);

    // ---- fp32 optical density (keeps all intermediates small/fast) ----------
    // OD in fp32 for downstream: projection, phi, concentrations, reconstruct.
    auto od_all = -torch::log((images_float * 255.0f + 1.0f) / Io_f).reshape({N, 3, P});
    auto od_pix = od_all.permute({0, 2, 1}).contiguous();  // (N, P, 3) fp32

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Pre-compute OD mask from fp32 od_pix (before any fp16 conversion).
    auto od_min_f32 = std::get<0>(od_pix.min(/*dim=*/2));       // (N, P) fp32
    auto mask       = od_min_f32 >= beta_f;                     // (N, P) bool
    auto cnt        = mask.sum(/*dim=*/1).to(torch::kFloat32);  // (N,) fp32
    auto use_all    = cnt < 3.0f;                               // (N,) bool
    auto eff_mask   = mask.logical_or(use_all.unsqueeze(1));    // (N, P) bool
    auto cnt_eff    = torch::where(use_all, torch::full_like(cnt, static_cast<float>(P)), cnt);

    torch::Tensor eigvecs;  // (N, 3, 2) — always fp32 output
    torch::Tensor min_phi, max_phi, HE;
    torch::Tensor C0, C1, maxC0, maxC1, od_recon;

    if (fast) {
        // =====================================================================
        // FAST PATH: fp32 cov/eigh + fp16 large pixels + fp32 2×2 solve.
        // No fp64 anywhere in this path.
        // =====================================================================

        // ---- fp32 covariance + analytic eigh --------------------------------
        eigvecs = torch::empty({N, 3, 2}, f32);
        macenko_cov_kernel<float><<<static_cast<unsigned int>(N), MACENKO_THREADS, 0, stream>>>(od_pix.data_ptr<float>(), P, static_cast<float>(beta), eigvecs.data_ptr<float>());
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA error in macenko_cov_kernel<float>: ", cudaGetErrorString(err));

        // --- Projection onto the stain plane (fp16 bmm → Tensor Cores) -------
        auto od_pix_f16  = od_pix.to(torch::kFloat16);           // (N, P, 3)
        auto eigvecs_f16 = eigvecs.to(torch::kFloat16);          // (N, 3, 2)
        auto That_f16    = torch::bmm(od_pix_f16, eigvecs_f16);  // (N, P, 2) fp16

        // --- phi, mask, percentiles (fp16) -----------------------------------
        auto phi_f16 = torch::atan2(That_f16.index({Slice(), Slice(), 1}), That_f16.index({Slice(), Slice(), 0}));  // (N, P) fp16

        const float INF_F16 = 65504.0f;  // fp16 max representable
        auto phi_masked_f16 = torch::where(eff_mask, phi_f16, torch::full_like(phi_f16, INF_F16));
        auto phi_sorted_f16 = std::get<0>(torch::sort(phi_masked_f16, /*dim=*/1, /*descending=*/false));
        // gather_percentile does fp64 index math; gather works on any dtype.
        min_phi = gather_percentile(phi_sorted_f16, cnt_eff, alpha).to(torch::kFloat32);          // (N, 1) fp32
        max_phi = gather_percentile(phi_sorted_f16, cnt_eff, 100.0 - alpha).to(torch::kFloat32);  // (N, 1) fp32

        // --- Extreme stain vectors (small fp32 bmm — not worth fp16) ---------
        auto angle_min = torch::cat({torch::cos(min_phi), torch::sin(min_phi)}, /*dim=*/1).unsqueeze(2);
        auto angle_max = torch::cat({torch::cos(max_phi), torch::sin(max_phi)}, /*dim=*/1).unsqueeze(2);
        auto vMin      = torch::bmm(eigvecs, angle_min);
        auto vMax      = torch::bmm(eigvecs, angle_max);

        // --- H/E ordering heuristic ------------------------------------------
        auto he_first_min = (vMin.index({Slice(), 0, 0}) > vMax.index({Slice(), 0, 0})).view({N, 1, 1});
        auto HE_min_first = torch::cat({vMin, vMax}, /*dim=*/2);
        auto HE_max_first = torch::cat({vMax, vMin}, /*dim=*/2);
        HE                = torch::where(he_first_min, HE_min_first, HE_max_first);  // (N, 3, 2) fp32

        // --- concentrations: fp32 rhs (sensitive) + fp32 2×2 solve -----------
        // rhs → C → percentile is numerically sensitive; keep fp32 here.
        auto HEt = HE.transpose(1, 2);       // (N, 2, 3) fp32
        auto A2  = torch::bmm(HEt, HE);      // (N, 2, 2) fp32 — tiny
        auto rhs = torch::bmm(HEt, od_all);  // (N, 2, P) fp32

        auto a     = A2.index({Slice(), 0, 0});
        auto b     = A2.index({Slice(), 0, 1});
        auto c_    = A2.index({Slice(), 1, 1});
        auto det   = (a * c_ - b * b).view({N, 1, 1});
        auto inv00 = (c_ / det.view({N})).view({N, 1});  // fp32
        auto inv01 = (-b / det.view({N})).view({N, 1});  // fp32
        auto inv11 = (a / det.view({N})).view({N, 1});   // fp32
        auto rhs0  = rhs.index({Slice(), 0, Slice()});   // (N, P) fp32
        auto rhs1  = rhs.index({Slice(), 1, Slice()});   // (N, P) fp32
        C0         = inv00 * rhs0 + inv01 * rhs1;        // (N, P) fp32
        C1         = inv01 * rhs0 + inv11 * rhs1;        // (N, P) fp32

        // --- 99th percentile of concentrations (fp16 sort) --------------------
        auto cnt_all_f32   = torch::full({N}, static_cast<float>(P), f32);
        auto C0_f16        = C0.to(torch::kFloat16);
        auto C1_f16        = C1.to(torch::kFloat16);
        auto C0_sorted_f16 = std::get<0>(torch::sort(C0_f16, /*dim=*/1));
        auto C1_sorted_f16 = std::get<0>(torch::sort(C1_f16, /*dim=*/1));
        maxC0              = gather_percentile(C0_sorted_f16, cnt_all_f32, 99.0).to(torch::kFloat32);  // (N, 1) fp32
        maxC1              = gather_percentile(C1_sorted_f16, cnt_all_f32, 99.0).to(torch::kFloat32);  // (N, 1) fp32

        // --- normalise and reconstruct (fp16 matmul → safe) -------------------
        auto tmc = target_max_conc.flatten().to(torch::kFloat32);
        TORCH_CHECK(tmc.size(0) == 2, "target_max_conc must have 2 elements");
        auto scale0        = (tmc.index({0}) / maxC0);                                 // (N, 1) fp32
        auto scale1        = (tmc.index({1}) / maxC1);                                 // (N, 1) fp32
        auto C0_scaled_f16 = (C0_f16 * scale0.to(torch::kFloat16));                    // (N, P) fp16
        auto C1_scaled_f16 = (C1_f16 * scale1.to(torch::kFloat16));                    // (N, P) fp16
        auto Cn_f16        = torch::stack({C0_scaled_f16, C1_scaled_f16}, /*dim=*/1);  // (N, 2, P) fp16
        auto stain_f16     = stain_matrix.to(torch::kFloat16);                         // (3, 2) fp16
        od_recon           = torch::matmul(stain_f16, Cn_f16).to(torch::kFloat32);     // (N, 3, P) fp32

    } else {
        // =====================================================================
        // STABLE PATH: fp64 cov + fp64 eigh, fp32 downstream.
        // =====================================================================
        auto f64 = f32.dtype(torch::kFloat64);

        // ---- fp64 covariance + analytic eigh --------------------------------
        auto od_pix_f64  = od_pix.to(torch::kFloat64);  // (N, P, 3) fp64
        auto eigvecs_f64 = torch::empty({N, 3, 2}, f64);
        macenko_cov_kernel<double><<<static_cast<unsigned int>(N), MACENKO_THREADS, 0, stream>>>(od_pix_f64.data_ptr<double>(), P, beta, eigvecs_f64.data_ptr<double>());
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA error in macenko_cov_kernel<double>: ", cudaGetErrorString(err));
        eigvecs = eigvecs_f64.to(torch::kFloat32);  // (N, 3, 2) fp32

        // Projection onto the stain plane.
        auto That  = torch::bmm(od_pix, eigvecs);  // (N, P, 2) fp32
        auto phi_s = torch::atan2(That.index({Slice(), Slice(), 1}), That.index({Slice(), Slice(), 0}));

        // Percentiles of phi (fp32 sort).
        const float INF = std::numeric_limits<float>::infinity();
        auto phi_masked = torch::where(eff_mask, phi_s, torch::full_like(phi_s, INF));
        auto phi_sorted = std::get<0>(torch::sort(phi_masked, /*dim=*/1, /*descending=*/false));
        min_phi         = gather_percentile(phi_sorted, cnt_eff, alpha);
        max_phi         = gather_percentile(phi_sorted, cnt_eff, 100.0 - alpha);

        // Extreme stain vectors from angular bounds.
        auto angle_min = torch::cat({torch::cos(min_phi), torch::sin(min_phi)}, /*dim=*/1).unsqueeze(2);
        auto angle_max = torch::cat({torch::cos(max_phi), torch::sin(max_phi)}, /*dim=*/1).unsqueeze(2);
        auto vMin      = torch::bmm(eigvecs, angle_min);
        auto vMax      = torch::bmm(eigvecs, angle_max);

        // H/E ordering heuristic.
        auto he_first_min = (vMin.index({Slice(), 0, 0}) > vMax.index({Slice(), 0, 0})).view({N, 1, 1});
        auto HE_min_first = torch::cat({vMin, vMax}, /*dim=*/2);
        auto HE_max_first = torch::cat({vMax, vMin}, /*dim=*/2);
        HE                = torch::where(he_first_min, HE_min_first, HE_max_first);

        // Concentrations: 2×2 normal equations.
        auto HEt   = HE.transpose(1, 2);
        auto A2    = torch::bmm(HEt, HE);
        auto rhs   = torch::bmm(HEt, od_all);
        auto a     = A2.index({Slice(), 0, 0});
        auto b     = A2.index({Slice(), 0, 1});
        auto c_    = A2.index({Slice(), 1, 1});
        auto det   = (a * c_ - b * b).view({N, 1, 1});
        auto inv00 = (c_ / det.view({N})).view({N, 1});
        auto inv01 = (-b / det.view({N})).view({N, 1});
        auto inv11 = (a / det.view({N})).view({N, 1});
        auto rhs0  = rhs.index({Slice(), 0, Slice()});
        auto rhs1  = rhs.index({Slice(), 1, Slice()});
        C0         = inv00 * rhs0 + inv01 * rhs1;
        C1         = inv01 * rhs0 + inv11 * rhs1;

        // 99th percentile of each concentration.
        auto cnt_all   = torch::full({N}, static_cast<float>(P), f32);
        auto C0_sorted = std::get<0>(torch::sort(C0, /*dim=*/1));
        auto C1_sorted = std::get<0>(torch::sort(C1, /*dim=*/1));
        maxC0          = gather_percentile(C0_sorted, cnt_all, 99.0);
        maxC1          = gather_percentile(C1_sorted, cnt_all, 99.0);

        // Normalise and reconstruct.
        auto tmc = target_max_conc.flatten().to(torch::kFloat32);
        TORCH_CHECK(tmc.size(0) == 2, "target_max_conc must have 2 elements");
        auto scale0 = (tmc.index({0}) / maxC0);
        auto scale1 = (tmc.index({1}) / maxC1);
        auto Cn     = torch::stack({C0 * scale0, C1 * scale1}, /*dim=*/1);
        auto stain  = stain_matrix.to(torch::kFloat32);
        od_recon    = torch::matmul(stain, Cn);
    }

    // ---- final clamp + reshape (fp32, shared by both paths) -----------------
    auto rgb = torch::clamp(Io_f * torch::exp(-od_recon), 0.0f, 255.0f).reshape({N, 3, H, W});
    return rgb.to(input_images.scalar_type());
}

torch::Tensor macenko_cuda(torch::Tensor input_images, torch::Tensor stain_matrix, torch::Tensor target_max_conc) {
    return macenko_cuda_impl(input_images, stain_matrix, target_max_conc, /*fast=*/false);
}

torch::Tensor macenko_cuda_fast(torch::Tensor input_images, torch::Tensor stain_matrix, torch::Tensor target_max_conc) {
    return macenko_cuda_impl(input_images, stain_matrix, target_max_conc, /*fast=*/true);
}
