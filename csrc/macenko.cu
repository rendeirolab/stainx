// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Macenko pure CUDA kernels: filtered OD covariance + analytic 3x3 symmetric eigh.
 *
 * No PyTorch dependencies — usable by any CUDA interface. The Torch wrapper in
 * src/stainx_cuda_torch/csrc/macenko.cu includes this file and owns the ATen
 * downstream pipeline (projection, percentiles, concentration, reconstruct).
 */

#include <cstdint>
#include <cuda_runtime.h>
#include <math.h>

#define MACENKO_THREADS 256
#define MACENKO_NUM_WARPS (MACENKO_THREADS / 32)  // 8

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
    constexpr int NW      = MACENKO_NUM_WARPS;  // 8
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
