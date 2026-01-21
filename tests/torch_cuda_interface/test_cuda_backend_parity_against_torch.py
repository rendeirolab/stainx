# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""Parity tests: **Torch CUDA extension backend** vs **Torch backend**.

What this file tests
--------------------
On a CUDA-capable system, compare:
- `backend="cuda"` (custom CUDA extension via `stainx_cuda_torch`)
- `backend="torch"` (Torch ops)

for the same inputs and assert outputs match within tolerance.
"""

import pytest
import torch

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter


def compute_relative_absolute_error_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    assert x.shape == y.shape, "Tensors must have the same shape"
    assert x.dtype == y.dtype, "Tensors must have the same dtype"
    assert x.device == y.device, "Tensors must be on the same device"

    epsilon = 1e-16
    norm_y = torch.norm(y, p=2) + epsilon
    rel_abs_error = torch.norm((x - y).abs(), p=2) / norm_y
    return rel_abs_error.item()


@pytest.mark.cuda
class TestCUDABackendComparisonTorch:
    @pytest.fixture
    def cuda_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return torch.device("cuda")

    @pytest.fixture
    def reference_image(self, cuda_device):
        torch.manual_seed(42)
        return (torch.rand(1, 3, 256, 256, device=cuda_device) * 255).round().to(torch.uint8)

    @pytest.fixture
    def source_image_torch(self, cuda_device):
        torch.manual_seed(123)
        return (torch.rand(1, 3, 256, 256, device=cuda_device) * 255).round().to(torch.uint8)

    def test_reinhard_cuda_vs_torch(self, reference_image, source_image_torch, cuda_device):
        """Test CUDA backend vs Torch backend for Reinhard normalization."""
        # Torch backend
        normalizer_torch = Reinhard(device=cuda_device, backend="torch")
        normalizer_torch.fit(reference_image)
        result_torch = normalizer_torch.transform(source_image_torch)
        result_torch_cpu = result_torch.squeeze(0).cpu().float()

        # CUDA backend
        normalizer_cuda = Reinhard(device=cuda_device, backend="cuda")
        normalizer_cuda.fit(reference_image)
        result_cuda = normalizer_cuda.transform(source_image_torch)
        result_cuda_cpu = result_cuda.squeeze(0).cpu().float()

        rel_abs_error = compute_relative_absolute_error_torch(result_cuda_cpu, result_torch_cpu)
        assert rel_abs_error < 0.01, f"CUDA vs Torch Reinhard relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"

    def test_macenko_cuda_vs_torch(self, reference_image, source_image_torch, cuda_device):
        """Test CUDA backend vs Torch backend for Macenko normalization."""
        # Torch backend
        normalizer_torch = Macenko(device=cuda_device, backend="torch")
        normalizer_torch.fit(reference_image)
        result_torch = normalizer_torch.transform(source_image_torch)
        result_torch_cpu = result_torch.squeeze(0).cpu().float()

        # CUDA backend
        normalizer_cuda = Macenko(device=cuda_device, backend="cuda")
        normalizer_cuda.fit(reference_image)
        result_cuda = normalizer_cuda.transform(source_image_torch)
        result_cuda_cpu = result_cuda.squeeze(0).cpu().float()

        rel_abs_error = compute_relative_absolute_error_torch(result_cuda_cpu, result_torch_cpu)
        assert rel_abs_error < 0.01, f"CUDA vs Torch Macenko relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"

    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_cuda_vs_torch(self, reference_image, source_image_torch, cuda_device, channel_axis):
        """Test CUDA backend vs Torch backend for histogram matching."""
        converter = ChannelFormatConverter(channel_axis=channel_axis)

        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image_torch)

        # Torch backend
        normalizer_torch = HistogramMatching(device=cuda_device, backend="torch", channel_axis=channel_axis)
        normalizer_torch.fit(ref_input)
        result_torch = normalizer_torch.transform(src_input)
        result_torch_chw = converter.to_chw(result_torch, squeeze_batch=True)

        # CUDA backend
        normalizer_cuda = HistogramMatching(device=cuda_device, backend="cuda", channel_axis=channel_axis)
        normalizer_cuda.fit(ref_input)
        result_cuda = normalizer_cuda.transform(src_input)
        result_cuda_chw = converter.to_chw(result_cuda, squeeze_batch=True)

        rel_abs_error = compute_relative_absolute_error_torch(result_cuda_chw, result_torch_chw)
        assert rel_abs_error < 0.01, f"CUDA vs Torch histogram matching relative absolute error too large: {rel_abs_error:.6f}, expected <0.01 (channel_axis={channel_axis})"


if __name__ == "__main__":
    pytest.main()
