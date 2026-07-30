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
 * Both modes share the same templated per-image reduction kernel
 * (macenko_cov_kernel<T>) with warp-level tree reduction (no atomicAdd),
 * the same templated closed-form 3x3 eigensolver (analytic_eigh_sym3<T>),
 * and the same batched ATen ops for the downstream pipeline.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cstdint>

using namespace torch::indexing;

namespace {

constexpr int MACENKO_THREADS = 256;
constexpr int NUM_WARPS       = MACENKO_THREADS / 32;  // 8

// ---------------------------------------------------------------------------
// Closed-form eigendecomposition of a symmetric 3x3 matrix.
// Templated on scalar type T (float or double).
// Returns the two leading eigenvectors (columns for the middle and largest
// eigenvalue, i.e. torch.linalg.eigh(...)[:, [1, 2]]) in `evecs2` as
// evecs2[row][0] = middle, evecs2[row][1] = largest.
// ---------------------------------------------------------------------------
template <typename T>
__device__ inline void cross3(const T a[3], const T b[3], T out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename T>
__device__ inline void normalize3(T v[3]) {
    const T n   = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const T inv = n > T(1e-30) ? T(1.0) / n : T(0.0);
    v[0] *= inv;
    v[1] *= inv;
    v[2] *= inv;
}

// Eigenvector of a symmetric matrix `M` (already A - lambda*I) as the most robust
// cross product of its rows (largest-magnitude null-space direction).
template <typename T>
__device__ inline void eigvec_from_shifted(const T M[3][3], T out[3]) {
    T r0[3] = {M[0][0], M[0][1], M[0][2]};
    T r1[3] = {M[1][0], M[1][1], M[1][2]};
    T r2[3] = {M[2][0], M[2][1], M[2][2]};
    T c0[3], c1[3], c2[3];
    cross3(r0, r1, c0);
    cross3(r0, r2, c1);
    cross3(r1, r2, c2);
    const T n0    = c0[0] * c0[0] + c0[1] * c0[1] + c0[2] * c0[2];
    const T n1    = c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2];
    const T n2    = c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2];
    const T* best = c0;
    T bestn       = n0;
    if (n1 > bestn) {
        best  = c1;
        bestn = n1;
    }
    if (n2 > bestn) {
        best  = c2;
        bestn = n2;
    }
    out[0] = best[0];
    out[1] = best[1];
    out[2] = best[2];
    normalize3(out);
}

template <typename T>
__device__ void analytic_eigh_sym3(const T A[3][3], T evecs2[3][2]) {
    const T p1 = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
    const T q  = (A[0][0] + A[1][1] + A[2][2]) / T(3.0);

    T e_asc[3];  // eigenvalues ascending
    if (p1 <= T(1e-30)) {
        // Already diagonal (or effectively zero off-diagonal):
        // eigenvalues are the diagonal entries, not the trace mean.
        e_asc[0] = A[0][0];
        e_asc[1] = A[1][1];
        e_asc[2] = A[2][2];
        // Insertion-sort ascending.
        for (int i = 1; i < 3; ++i) {
            const T key = e_asc[i];
            int j       = i - 1;
            while (j >= 0 && e_asc[j] > key) {
                e_asc[j + 1] = e_asc[j];
                --j;
            }
            e_asc[j + 1] = key;
        }
    } else {
        const T p2 = (A[0][0] - q) * (A[0][0] - q) + (A[1][1] - q) * (A[1][1] - q) + (A[2][2] - q) * (A[2][2] - q) + T(2.0) * p1;
        const T p  = sqrt(p2 / T(6.0));
        // B = (A - q I) / p ; r = det(B) / 2
        T B[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) B[i][j] = (A[i][j] - (i == j ? q : T(0.0))) / p;
        const T detB = B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1]) - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0]) + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]);
        T r          = detB / T(2.0);
        if (r < T(-1.0)) r = T(-1.0);
        if (r > T(1.0)) r = T(1.0);
        const T phi   = acos(r) / T(3.0);
        const T e_max = q + T(2.0) * p * cos(phi);
        const T e_min = q + T(2.0) * p * cos(phi + T(2.0) * T(M_PI) / T(3.0));
        const T e_mid = T(3.0) * q - e_max - e_min;
        e_asc[0]      = e_min;
        e_asc[1]      = e_mid;
        e_asc[2]      = e_max;
    }

    // Eigenvectors for the two largest eigenvalues (indices 1 and 2 ascending).
    for (int col = 0; col < 2; ++col) {
        const T lam = e_asc[col + 1];
        T M[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M[i][j] = A[i][j] - (i == j ? lam : T(0.0));
        T v[3];
        eigvec_from_shifted(M, v);
        evecs2[0][col] = v[0];
        evecs2[1][col] = v[1];
        evecs2[2][col] = v[2];
    }
}

// ---------------------------------------------------------------------------
// Per-image filtered covariance + eigenvectors.
// Templated on scalar type T (float for fast path, double for stable).
// One block per image. `od_pix` is (N, P, 3) in type T (per-pixel OD vectors).
//
// Reduction: each thread accumulates 10 partial sums in registers over its
// pixel stride, then a warp-level tree reduce (__shfl_down_sync) collapses
// each warp into lane 0, which writes to shared memory.  A final single-warp
// pass sums across the 8 warps.
//
// Pixels with min(OD) >= beta are kept; if fewer than 3 survive, all pixels
// are used (matches the torchstain / Torch-backend fallback). Outputs the two
// leading eigenvectors of the covariance as `eigvecs` (N, 3, 2) in type T.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void macenko_cov_kernel(const T* __restrict__ od_pix, int64_t P, T beta, T* __restrict__ eigvecs) {
    constexpr int NW      = NUM_WARPS;  // 8
    constexpr int N_ACCUM = 10;

    const int n    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const T* img   = od_pix + static_cast<int64_t>(n) * P * 3;

    // [cnt, s0, s1, s2, xx, xy, xz, yy, yz, zz]
    T lm[N_ACCUM] = {0};
    T la[N_ACCUM] = {0};

    // --- thread-local accumulation over pixel stride ---
    for (int64_t p = tid; p < P; p += MACENKO_THREADS) {
        const T x0 = img[p * 3 + 0];
        const T x1 = img[p * 3 + 1];
        const T x2 = img[p * 3 + 2];
        // Accumulate for all-pixel fallback
        la[0] += T(1.0);
        la[1] += x0;
        la[2] += x1;
        la[3] += x2;
        la[4] += x0 * x0;
        la[5] += x0 * x1;
        la[6] += x0 * x2;
        la[7] += x1 * x1;
        la[8] += x1 * x2;
        la[9] += x2 * x2;
        // Masked accumulation
        const T mn = fmin(x0, fmin(x1, x2));
        if (mn >= beta) {
            lm[0] += T(1.0);
            lm[1] += x0;
            lm[2] += x1;
            lm[3] += x2;
            lm[4] += x0 * x0;
            lm[5] += x0 * x1;
            lm[6] += x0 * x2;
            lm[7] += x1 * x1;
            lm[8] += x1 * x2;
            lm[9] += x2 * x2;
        }
    }

    // --- warp-level tree reduction (no atomics) ---
    // Shared memory: [NW][N_ACCUM] for masked + all, interleaved as
    //   sm[w * N_ACCUM + k], sa[w * N_ACCUM + k]
    __shared__ T sm[NW * N_ACCUM];
    __shared__ T sa[NW * N_ACCUM];

#pragma unroll
    for (int k = 0; k < N_ACCUM; ++k) {
        T vm = lm[k];
        T va = la[k];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            vm += __shfl_down_sync(0xffffffff, vm, offset);
            va += __shfl_down_sync(0xffffffff, va, offset);
        }
        if (lane == 0) {
            sm[warp * N_ACCUM + k] = vm;
            sa[warp * N_ACCUM + k] = va;
        }
    }
    __syncthreads();

    // --- cross-warp reduction (first 10 threads of block) ---
    if (tid < N_ACCUM) {
        T sum_m = T(0.0), sum_a = T(0.0);
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            sum_m += sm[w * N_ACCUM + tid];
            sum_a += sa[w * N_ACCUM + tid];
        }
        sm[tid] = sum_m;
        sa[tid] = sum_a;
    }
    __syncthreads();

    // --- thread 0: covariance + eigh ---
    if (tid == 0) {
        // Choose masked accumulators unless too few pixels survived the OD filter.
        const T* acc = (sm[0] >= T(3.0)) ? sm : sa;
        const T cnt  = acc[0];
        T cov[3][3];
        if (cnt > T(1.0)) {
            const T inv   = T(1.0) / cnt;
            const T mean0 = acc[1] * inv, mean1 = acc[2] * inv, mean2 = acc[3] * inv;
            const T denom = cnt - T(1.0);
            cov[0][0]     = (acc[4] - acc[1] * mean0) / denom;
            cov[0][1]     = (acc[5] - acc[1] * mean1) / denom;
            cov[0][2]     = (acc[6] - acc[1] * mean2) / denom;
            cov[1][1]     = (acc[7] - acc[2] * mean1) / denom;
            cov[1][2]     = (acc[8] - acc[2] * mean2) / denom;
            cov[2][2]     = (acc[9] - acc[3] * mean2) / denom;
            cov[1][0]     = cov[0][1];
            cov[2][0]     = cov[0][2];
            cov[2][1]     = cov[1][2];
        } else {
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j) cov[i][j] = T(0.0);
        }
        T evecs2[3][2];
        analytic_eigh_sym3(cov, evecs2);
        T* out = eigvecs + static_cast<int64_t>(n) * 6;
        out[0] = evecs2[0][0];
        out[1] = evecs2[0][1];
        out[2] = evecs2[1][0];
        out[3] = evecs2[1][1];
        out[4] = evecs2[2][0];
        out[5] = evecs2[2][1];
    }
}

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
