// Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
// All rights reserved.
//
// This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
// See the LICENSE file for details.

/*
 * Macenko normalization for CUDA tensors.
 *
 * Numerically mirrors MacenkoTorch / torchstain: CPU eigh, nearest-percentile
 * (kthvalue), and linalg.lstsq. Bulk OD/RGB math stays on the tensor device.
 */

#include <torch/extension.h>

#include <cmath>
#include <tuple>

using namespace torch::indexing;

namespace {

float percentile_nearest(const torch::Tensor& t, double q) {
    auto flat       = t.reshape({-1}).contiguous();
    const int64_t n = flat.numel();
    TORCH_CHECK(n > 0, "percentile_nearest: empty tensor");
    int64_t k = 1 + static_cast<int64_t>(std::llround(0.01 * q * static_cast<double>(n - 1)));
    if (k < 1) k = 1;
    if (k > n) k = n;
    return std::get<0>(flat.kthvalue(k)).item<float>();
}

torch::Tensor process_single_image(const torch::Tensor& od,  // (3, H, W) float, on CUDA
                                   const torch::Tensor& stain_matrix,
                                   const torch::Tensor& target_max_conc,
                                   float beta,
                                   float alpha,
                                   float Io,
                                   int64_t H,
                                   int64_t W) {
    auto od_reshaped = od.permute({1, 2, 0}).reshape({-1, 3});
    auto od_min      = std::get<0>(od_reshaped.min(/*dim=*/1));
    auto mask        = od_min >= beta;
    auto od_filtered = od_reshaped.index({mask});
    if (od_filtered.size(0) < 3) { od_filtered = od_reshaped; }

    auto od_filtered_T       = od_filtered.transpose(0, 1);
    auto od_mean             = od_filtered_T.mean(/*dim=*/1, /*keepdim=*/true);
    auto od_centered         = od_filtered_T - od_mean;
    const int64_t num_pixels = od_filtered.size(0);
    torch::Tensor cov;
    if (num_pixels > 1) {
        cov = torch::matmul(od_centered, od_centered.transpose(0, 1)) / static_cast<float>(num_pixels - 1);
    } else {
        cov = torch::zeros({3, 3}, od_centered.options());
    }

    // CPU eigh for torchstain parity (unstable 2D plane on CUDA syevd)
    auto eigh_out = torch::linalg_eigh(cov.cpu());
    auto eigvecs  = std::get<1>(eigh_out).to(od.device()).index({Slice(), Slice(1, 3)});

    auto That = torch::matmul(od_filtered, eigvecs);
    auto phi  = torch::atan2(That.index({Slice(), 1}), That.index({Slice(), 0}));

    const float min_phi = percentile_nearest(phi, alpha);
    const float max_phi = percentile_nearest(phi, 100.0 - alpha);

    auto min_phi_t = torch::tensor(min_phi, od.options());
    auto max_phi_t = torch::tensor(max_phi, od.options());
    auto angle_min = torch::stack({torch::cos(min_phi_t), torch::sin(min_phi_t)});
    auto angle_max = torch::stack({torch::cos(max_phi_t), torch::sin(max_phi_t)});

    auto vMin      = torch::matmul(eigvecs, angle_min).unsqueeze(1);
    auto vMax      = torch::matmul(eigvecs, angle_max).unsqueeze(1);
    auto HE_source = torch::where(vMin.index({0}) > vMax.index({0}), torch::cat({vMin, vMax}, /*dim=*/1), torch::cat({vMax, vMin}, /*dim=*/1));

    auto od_all = od.reshape({3, -1});
    // CPU lstsq for torchstain / MacenkoTorch parity
    auto concentrations = std::get<0>(torch::linalg_lstsq(HE_source.cpu(), od_all.cpu(), /*rcond=*/c10::nullopt, /*driver=*/c10::nullopt)).to(od.device());

    const float max_conc_0 = percentile_nearest(concentrations.index({0}), 99.0);
    const float max_conc_1 = percentile_nearest(concentrations.index({1}), 99.0);
    auto max_conc          = torch::tensor({max_conc_0, max_conc_1}, od.options());

    auto norm_factor         = target_max_conc / max_conc;
    auto concentrations_norm = concentrations * norm_factor.unsqueeze(-1);
    auto od_recon            = torch::matmul(stain_matrix, concentrations_norm);
    auto rgb_recon           = torch::clamp(Io * torch::exp(-od_recon), 0.0f, 255.0f);
    return rgb_recon.reshape({3, H, W});
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

    torch::Tensor images_float;
    if (input_images.dtype() == torch::kUInt8) {
        images_float = input_images.to(torch::kFloat32) / 255.0f;
    } else {
        images_float = input_images.to(torch::kFloat32);
        if (images_float.max().item<float>() > 1.0f) { images_float = images_float / 255.0f; }
    }

    const int64_t N = images_float.size(0);
    const int64_t H = images_float.size(2);
    const int64_t W = images_float.size(3);

    constexpr float Io    = 240.0f;
    constexpr float beta  = 0.15f;
    constexpr float alpha = 1.0f;

    auto stain = stain_matrix.to(torch::kFloat32).contiguous();
    auto max_c = target_max_conc.flatten().to(torch::kFloat32).contiguous();
    TORCH_CHECK(max_c.size(0) == 2, "target_max_conc must have 2 elements");

    auto od_all_images = -torch::log((images_float * 255.0f + 1.0f) / Io);
    auto output        = torch::empty({N, 3, H, W}, images_float.options());

    for (int64_t n = 0; n < N; n++) { output.index_put_({n}, process_single_image(od_all_images.index({n}), stain, max_c, beta, alpha, Io, H, W)); }

    return output.to(input_images.scalar_type());
}
