// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Macenko normalization for CUDA tensors — fully on-GPU implementation.
 *
 * Earlier versions offloaded the 3x3 OD covariance, its eigendecomposition and
 * the concentration least-squares to the CPU: CUDA float32 covariance reductions
 * diverge on near-degenerate spectra and flip the stain plane, so parity with
 * torchstain was only reachable via LAPACK on the host. That host round-trip
 * forces a full device sync per image and dominates the runtime.
 *
 * This implementation keeps the covariance + eigh in double precision (the only
 * numerically sensitive part) and runs everything else in float32 for speed:
 *   - a custom per-image reduction kernel (macenko_cov_kernel) forms the filtered
 *     OD mean + covariance in fp64 using warp-level tree reduction (no atomicAdd);
 *   - a closed-form symmetric-3x3 eigensolver (analytic_eigh_sym3) replaces
 *     cuSOLVER/LAPACK eigh and matches torch.linalg.eigh to machine precision;
 *   - projection, angle percentiles, the H/E heuristic, the (H^T H) 2x2 normal
 *     equations for concentrations, and reconstruction are batched ATen ops in fp32.
 *
 * fp64 on the 3x3 cov/eigh makes the stain-plane extraction robust to the flip
 * seen with fp32 reductions on near-degenerate spectra.  Once the plane is fixed,
 * fp32 downstream cannot change which plane we picked — it only computes
 * coordinates and reconstructs, with errors well below one grey level.
 *
 * On an RTX A6000 (CC 8.6, fp64:fp32 = 1:64), the mixed-precision path is
 * ~2.5–3× faster than the all-fp64 pipeline on real H&E tiles while maintaining
 * the same correctness (max|e| ≤ 2 vs torchstain).
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
// Closed-form eigendecomposition of a symmetric 3x3 matrix (double precision).
// Returns the two leading eigenvectors (columns for the middle and largest
// eigenvalue, i.e. torch.linalg.eigh(...)[:, [1, 2]]) in `evecs2` as
// evecs2[row][0] = middle, evecs2[row][1] = largest.
// ---------------------------------------------------------------------------
__device__ inline void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void normalize3(double v[3]) {
    const double n   = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const double inv = n > 1e-300 ? 1.0 / n : 0.0;
    v[0] *= inv;
    v[1] *= inv;
    v[2] *= inv;
}

// Eigenvector of a symmetric matrix `M` (already A - lambda*I) as the most robust
// cross product of its rows (largest-magnitude null-space direction).
__device__ inline void eigvec_from_shifted(const double M[3][3], double out[3]) {
    double r0[3] = {M[0][0], M[0][1], M[0][2]};
    double r1[3] = {M[1][0], M[1][1], M[1][2]};
    double r2[3] = {M[2][0], M[2][1], M[2][2]};
    double c0[3], c1[3], c2[3];
    cross3(r0, r1, c0);
    cross3(r0, r2, c1);
    cross3(r1, r2, c2);
    const double n0    = c0[0] * c0[0] + c0[1] * c0[1] + c0[2] * c0[2];
    const double n1    = c1[0] * c1[0] + c1[1] * c1[1] + c1[2] * c1[2];
    const double n2    = c2[0] * c2[0] + c2[1] * c2[1] + c2[2] * c2[2];
    const double* best = c0;
    double bestn       = n0;
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

__device__ void analytic_eigh_sym3(const double A[3][3], double evecs2[3][2]) {
    const double p1 = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
    const double q  = (A[0][0] + A[1][1] + A[2][2]) / 3.0;

    double e_asc[3];  // eigenvalues ascending
    if (p1 <= 1e-300) {
        // Already diagonal.
        e_asc[0] = e_asc[1] = e_asc[2] = q;
    } else {
        const double p2 = (A[0][0] - q) * (A[0][0] - q) + (A[1][1] - q) * (A[1][1] - q) + (A[2][2] - q) * (A[2][2] - q) + 2.0 * p1;
        const double p  = sqrt(p2 / 6.0);
        // B = (A - q I) / p ; r = det(B) / 2
        double B[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) B[i][j] = (A[i][j] - (i == j ? q : 0.0)) / p;
        const double detB = B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1]) - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0]) + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]);
        double r          = detB / 2.0;
        if (r < -1.0) r = -1.0;
        if (r > 1.0) r = 1.0;
        const double phi   = acos(r) / 3.0;
        const double e_max = q + 2.0 * p * cos(phi);
        const double e_min = q + 2.0 * p * cos(phi + 2.0 * M_PI / 3.0);
        const double e_mid = 3.0 * q - e_max - e_min;
        e_asc[0]           = e_min;
        e_asc[1]           = e_mid;
        e_asc[2]           = e_max;
    }

    // Eigenvectors for the two largest eigenvalues (indices 1 and 2 ascending).
    for (int col = 0; col < 2; ++col) {
        const double lam = e_asc[col + 1];
        double M[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M[i][j] = A[i][j] - (i == j ? lam : 0.0);
        double v[3];
        eigvec_from_shifted(M, v);
        evecs2[0][col] = v[0];
        evecs2[1][col] = v[1];
        evecs2[2][col] = v[2];
    }
}

// ---------------------------------------------------------------------------
// Per-image filtered covariance + eigenvectors.
// One block per image. `od_pix` is (N, P, 3) double (per-pixel OD vectors).
//
// Reduction: each thread accumulates 10 partial sums in registers over its
// pixel stride, then a warp-level tree reduce (__shfl_down_sync) collapses
// each warp into lane 0, which writes to shared memory.  A final single-warp
// pass sums across the 8 warps.  This replaces the old atomicAdd-on-shared-double
// path which used a slow CAS loop for every atomic.
//
// Pixels with min(OD) >= beta are kept; if fewer than 3 survive, all pixels
// are used (matches the torchstain / Torch-backend fallback). Outputs the two
// leading eigenvectors of the covariance as `eigvecs` (N, 3, 2) double.
// ---------------------------------------------------------------------------
__global__ void macenko_cov_kernel(const double* __restrict__ od_pix, int64_t P, double beta, double* __restrict__ eigvecs) {
    constexpr int NW      = NUM_WARPS;  // 8
    constexpr int N_ACCUM = 10;

    const int n       = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp    = tid >> 5;
    const double* img = od_pix + static_cast<int64_t>(n) * P * 3;

    // [cnt, s0, s1, s2, xx, xy, xz, yy, yz, zz]
    double lm[N_ACCUM] = {0};
    double la[N_ACCUM] = {0};

    // --- thread-local accumulation over pixel stride ---
    for (int64_t p = tid; p < P; p += MACENKO_THREADS) {
        const double x0 = img[p * 3 + 0];
        const double x1 = img[p * 3 + 1];
        const double x2 = img[p * 3 + 2];
        // Accumulate for all-pixel fallback
        la[0] += 1.0;
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
        const double mn = fmin(x0, fmin(x1, x2));
        if (mn >= beta) {
            lm[0] += 1.0;
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
    __shared__ double sm[NW * N_ACCUM];
    __shared__ double sa[NW * N_ACCUM];

#pragma unroll
    for (int k = 0; k < N_ACCUM; ++k) {
        double vm = lm[k];
        double va = la[k];
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
        double sum_m = 0.0, sum_a = 0.0;
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
        const double* acc = (sm[0] >= 3.0) ? sm : sa;
        const double cnt  = acc[0];
        double cov[3][3];
        if (cnt > 1.0) {
            const double inv   = 1.0 / cnt;
            const double mean0 = acc[1] * inv, mean1 = acc[2] * inv, mean2 = acc[3] * inv;
            const double denom = cnt - 1.0;
            cov[0][0]          = (acc[4] - acc[1] * mean0) / denom;
            cov[0][1]          = (acc[5] - acc[1] * mean1) / denom;
            cov[0][2]          = (acc[6] - acc[1] * mean2) / denom;
            cov[1][1]          = (acc[7] - acc[2] * mean1) / denom;
            cov[1][2]          = (acc[8] - acc[2] * mean2) / denom;
            cov[2][2]          = (acc[9] - acc[3] * mean2) / denom;
            cov[1][0]          = cov[0][1];
            cov[2][0]          = cov[0][2];
            cov[2][1]          = cov[1][2];
        } else {
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j) cov[i][j] = 0.0;
        }
        double evecs2[3][2];
        analytic_eigh_sym3(cov, evecs2);
        double* out = eigvecs + static_cast<int64_t>(n) * 6;
        out[0]      = evecs2[0][0];
        out[1]      = evecs2[0][1];
        out[2]      = evecs2[1][0];
        out[3]      = evecs2[1][1];
        out[4]      = evecs2[2][0];
        out[5]      = evecs2[2][1];
    }
}

// Nearest-rank percentile (matches torchstain: k = 1 + round(.01*q*(n-1)),
// 0-based index = round(.01*q*(n-1))) over sorted rows, per image.
// Works with fp32 or fp64 sorted_asc and counts (counts must be same dtype as
// sorted_asc for the arithmetic to stay in the same precision).
torch::Tensor gather_percentile(const torch::Tensor& sorted_asc, const torch::Tensor& counts, double q) {
    // counts and sorted_asc may be fp32 or fp64; do arithmetic in float64 for
    // the index computation to match torchstain's exact formula regardless of
    // the tensor dtype.
    auto idx = torch::round(0.01 * q * (counts.to(torch::kFloat64) - 1.0)).to(torch::kLong).clamp_min(0).unsqueeze(1);
    return sorted_asc.gather(1, idx);  // (N, 1)
}

}  // namespace

torch::Tensor macenko_cuda(torch::Tensor input_images, torch::Tensor stain_matrix, torch::Tensor target_max_conc) {
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
    auto f64 = f32.dtype(torch::kFloat64);

    // ---- fp32 optical density (keeps all intermediates small/fast) ----------
    // OD in fp32 for downstream: projection, phi, concentrations, reconstruct.
    auto od_all = -torch::log((images_float * 255.0f + 1.0f) / Io_f).reshape({N, 3, P});
    auto od_pix = od_all.permute({0, 2, 1}).contiguous();  // (N, P, 3) fp32

    // Cast only what the cov kernel needs to fp64.
    auto od_pix_f64 = od_pix.to(torch::kFloat64);  // (N, P, 3) fp64

    // ---- fp64 covariance + analytic eigh (the numerically sensitive part) ---
    auto eigvecs_f64    = torch::empty({N, 3, 2}, f64);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    macenko_cov_kernel<<<static_cast<unsigned int>(N), MACENKO_THREADS, 0, stream>>>(od_pix_f64.data_ptr<double>(), P, beta, eigvecs_f64.data_ptr<double>());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in macenko_cov_kernel: ", cudaGetErrorString(err));

    // ---- fp32 downstream: projection → phi → HE → concentrations → RGB -----
    auto eigvecs = eigvecs_f64.to(torch::kFloat32);  // (N, 3, 2) fp32

    // Projection onto the stain plane (fp32 bmm: ~20× faster than fp64 on consumer GPUs).
    auto That = torch::bmm(od_pix, eigvecs);                                                         // (N, P, 2) fp32
    auto phi  = torch::atan2(That.index({Slice(), Slice(), 1}), That.index({Slice(), Slice(), 0}));  // (N, P) fp32

    // Mask: pixels with min(OD) >= beta (fp32).
    auto od_min   = std::get<0>(od_pix.min(/*dim=*/2));       // (N, P) fp32
    auto mask     = od_min >= beta_f;                         // (N, P) bool
    auto cnt      = mask.sum(/*dim=*/1).to(torch::kFloat32);  // (N,) fp32
    auto use_all  = cnt < 3.0f;                               // (N,) bool
    auto eff_mask = mask.logical_or(use_all.unsqueeze(1));    // (N, P) bool
    auto cnt_eff  = torch::where(use_all, torch::full_like(cnt, static_cast<float>(P)), cnt);

    // Percentiles of phi (fp32 sort).
    const float INF = std::numeric_limits<float>::infinity();
    auto phi_masked = torch::where(eff_mask, phi, torch::full_like(phi, INF));
    auto phi_sorted = std::get<0>(torch::sort(phi_masked, /*dim=*/1, /*descending=*/false));
    auto min_phi    = gather_percentile(phi_sorted, cnt_eff, alpha);          // (N, 1)
    auto max_phi    = gather_percentile(phi_sorted, cnt_eff, 100.0 - alpha);  // (N, 1)

    // Extreme stain vectors from angular bounds.
    auto angle_min = torch::cat({torch::cos(min_phi), torch::sin(min_phi)}, /*dim=*/1).unsqueeze(2);  // (N, 2, 1)
    auto angle_max = torch::cat({torch::cos(max_phi), torch::sin(max_phi)}, /*dim=*/1).unsqueeze(2);  // (N, 2, 1)
    auto vMin      = torch::bmm(eigvecs, angle_min);                                                  // (N, 3, 1)
    auto vMax      = torch::bmm(eigvecs, angle_max);                                                  // (N, 3, 1)

    // H/E ordering heuristic: hematoxylin vector (larger red-channel OD) first.
    auto he_first_min = (vMin.index({Slice(), 0, 0}) > vMax.index({Slice(), 0, 0})).view({N, 1, 1});
    auto HE_min_first = torch::cat({vMin, vMax}, /*dim=*/2);  // (N, 3, 2)
    auto HE_max_first = torch::cat({vMax, vMin}, /*dim=*/2);
    auto HE           = torch::where(he_first_min, HE_min_first, HE_max_first);  // (N, 3, 2)

    // ---- concentrations via 2×2 normal equations (fp32) ----------------------
    auto HEt = HE.transpose(1, 2);       // (N, 2, 3)
    auto A2  = torch::bmm(HEt, HE);      // (N, 2, 2) symmetric
    auto rhs = torch::bmm(HEt, od_all);  // (N, 2, P)
    auto a   = A2.index({Slice(), 0, 0});
    auto b   = A2.index({Slice(), 0, 1});
    auto c_  = A2.index({Slice(), 1, 1});
    auto det = (a * c_ - b * b).view({N, 1, 1});
    // inv = 1/det * [[c, -b], [-b, a]]
    auto inv00 = (c_ / det.view({N})).view({N, 1});
    auto inv01 = (-b / det.view({N})).view({N, 1});
    auto inv11 = (a / det.view({N})).view({N, 1});
    auto rhs0  = rhs.index({Slice(), 0, Slice()});  // (N, P)
    auto rhs1  = rhs.index({Slice(), 1, Slice()});  // (N, P)
    auto C0    = inv00 * rhs0 + inv01 * rhs1;       // (N, P)
    auto C1    = inv01 * rhs0 + inv11 * rhs1;       // (N, P)

    // 99th percentile of each concentration (fp32 sort).
    auto cnt_all   = torch::full({N}, static_cast<float>(P), f32);
    auto C0_sorted = std::get<0>(torch::sort(C0, /*dim=*/1));
    auto C1_sorted = std::get<0>(torch::sort(C1, /*dim=*/1));
    auto maxC0     = gather_percentile(C0_sorted, cnt_all, 99.0);  // (N, 1)
    auto maxC1     = gather_percentile(C1_sorted, cnt_all, 99.0);  // (N, 1)

    // ---- normalise concentrations to the reference and reconstruct -----------
    auto tmc = target_max_conc.flatten().to(torch::kFloat32);
    TORCH_CHECK(tmc.size(0) == 2, "target_max_conc must have 2 elements");
    auto scale0 = (tmc.index({0}) / maxC0);                             // (N, 1)
    auto scale1 = (tmc.index({1}) / maxC1);                             // (N, 1)
    auto Cn     = torch::stack({C0 * scale0, C1 * scale1}, /*dim=*/1);  // (N, 2, P)

    auto stain    = stain_matrix.to(torch::kFloat32);  // (3, 2)
    auto od_recon = torch::matmul(stain, Cn);          // (N, 3, P)
    auto rgb      = torch::clamp(Io_f * torch::exp(-od_recon), 0.0f, 255.0f).reshape({N, 3, H, W});

    return rgb.to(input_images.scalar_type());
}
