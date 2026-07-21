# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""``backend="torch_cuda"`` focused CUDA checks: baselines + direct torch parity.

Full size matrix vs baselines lives in ``tests/torch_interface/``.
"""

from __future__ import annotations

import pytest
import torch
from skimage.exposure import match_histograms
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE as TORCH_CUDA_AVAILABLE
from stainx.utils import ChannelFormatConverter

RTOL = 0.0
ATOL = 1.0
MACENKO_ATOL = 2.0
# Direct torch↔cuda: allow one grey level on uint8 outputs (rounding), plus float noise.
PARITY_RTOL = 0.0
PARITY_ATOL = 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="stainx torch_cuda extension is not available")
class TestTorchCudaVsBaselines:
    @pytest.fixture
    def cuda_device(self):
        return torch.device("cuda")

    @pytest.fixture
    def reference_image(self, cuda_device):
        torch.manual_seed(42)
        return (torch.rand(1, 3, 256, 256, device=cuda_device) * 255).round().to(torch.uint8)

    @pytest.fixture
    def source_image(self, cuda_device):
        torch.manual_seed(123)
        return (torch.rand(1, 3, 256, 256, device=cuda_device) * 255).round().to(torch.uint8)

    def test_reinhard_vs_torchstain(self, reference_image, source_image, cuda_device):
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image.squeeze(0).cpu()
        baseline = TorchReinhardNormalizer()
        baseline.fit(ref_chw)
        baseline_tensor = baseline.normalize(src_chw).permute(2, 0, 1).float()

        normalizer = Reinhard(device=cuda_device, backend="torch_cuda")
        normalizer.fit(reference_image)
        result = normalizer.transform(source_image).squeeze(0).cpu().float()

        assert torch.allclose(result, baseline_tensor, rtol=RTOL, atol=ATOL)

    def test_macenko_vs_torchstain_fit_and_transform_on_cuda(self, reference_image, source_image, cuda_device):
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image.squeeze(0).cpu()
        baseline = TorchMacenkoNormalizer()
        baseline.fit(ref_chw)
        baseline_rgb, _, _ = baseline.normalize(src_chw, stains=True)
        baseline_tensor = baseline_rgb.permute(2, 0, 1).float()

        normalizer = Macenko(device=cuda_device, backend="torch_cuda")
        normalizer.fit(reference_image)
        result = normalizer.transform(source_image).squeeze(0).cpu().float()

        assert torch.allclose(normalizer._stain_matrix.cpu().float(), baseline.HERef.float(), rtol=1e-4, atol=1e-5)
        assert torch.allclose(result, baseline_tensor, rtol=RTOL, atol=MACENKO_ATOL)
        if baseline_tensor.max().item() > 240.0:
            assert result.max().item() > 240.0, "torch_cuda Macenko incorrectly capped at Io"

    def test_histogram_matching_vs_skimage(self, reference_image, source_image, cuda_device):
        converter = ChannelFormatConverter(channel_axis=1)
        skimage_chw = torch.from_numpy(match_histograms(converter.to_hwc(source_image, squeeze_batch=True), converter.to_hwc(reference_image, squeeze_batch=True), channel_axis=-1)).float().permute(2, 0, 1)

        normalizer = HistogramMatching(device=cuda_device, backend="torch_cuda", channel_axis=1)
        normalizer.fit(reference_image)
        result = normalizer.transform(source_image).squeeze(0).cpu().float()

        assert torch.allclose(result, skimage_chw, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="stainx torch_cuda extension is not available")
class TestTorchCudaVsTorchParity:
    """Direct backend parity — catches CUDA drift that baseline atol can hide."""

    @pytest.fixture
    def images(self):
        torch.manual_seed(7)
        ref = (torch.rand(1, 3, 128, 128) * 255).round().to(torch.uint8)
        src = (torch.rand(2, 3, 128, 128) * 255).round().to(torch.uint8)
        return ref, src

    def test_reinhard_fit_transform_parity(self, images):
        ref, src = images
        torch_n = Reinhard(device="cpu", backend="torch")
        cuda_n = Reinhard(device="cuda", backend="torch_cuda")
        torch_n.fit(ref)
        cuda_n.fit(ref.cuda())
        out_t = torch_n.transform(src)
        out_c = cuda_n.transform(src.cuda()).cpu()
        assert torch.allclose(out_t.float(), out_c.float(), rtol=PARITY_RTOL, atol=PARITY_ATOL)

    def test_macenko_fit_transform_parity(self, images):
        ref, src = images
        torch_n = Macenko(device="cpu", backend="torch")
        cuda_n = Macenko(device="cuda", backend="torch_cuda")
        torch_n.fit(ref)
        cuda_n.fit(ref.cuda())
        assert torch.allclose(torch_n._stain_matrix.float(), cuda_n._stain_matrix.cpu().float(), rtol=1e-5, atol=1e-5)
        assert torch.allclose(torch_n._target_max_conc.float().flatten(), cuda_n._target_max_conc.cpu().float().flatten(), rtol=1e-4, atol=1e-4)
        out_t = torch_n.transform(src)
        out_c = cuda_n.transform(src.cuda()).cpu()
        assert torch.allclose(out_t.float(), out_c.float(), rtol=PARITY_RTOL, atol=PARITY_ATOL)

    def test_histogram_matching_fit_transform_parity(self, images):
        ref, src = images
        torch_n = HistogramMatching(device="cpu", backend="torch", channel_axis=1)
        cuda_n = HistogramMatching(device="cuda", backend="torch_cuda", channel_axis=1)
        torch_n.fit(ref)
        cuda_n.fit(ref.cuda())
        out_t = torch_n.transform(src)
        out_c = cuda_n.transform(src.cuda()).cpu()
        assert torch.allclose(out_t.float(), out_c.float(), rtol=PARITY_RTOL, atol=PARITY_ATOL)
