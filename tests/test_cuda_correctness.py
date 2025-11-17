# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter


def compute_relative_absolute_error(x: torch.Tensor, y: torch.Tensor) -> float:
    assert x.shape == y.shape, "Tensors must have the same shape"
    assert x.dtype == y.dtype, "Tensors must have the same dtype"
    assert x.device == y.device, "Tensors must be on the same device"

    epsilon = 1e-16
    norm_y = torch.norm(y, p=2) + epsilon
    rel_abs_error = torch.norm((x - y).abs(), p=2) / norm_y
    return rel_abs_error.item()


@pytest.mark.cuda
class TestCUDABackendComparison:
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

    def test_reinhard_cuda_vs_pytorch(self, reference_image, source_image_torch, cuda_device):
        """Test CUDA backend vs PyTorch backend for Reinhard normalization."""
        # PyTorch backend
        normalizer_pytorch = Reinhard(device=cuda_device, backend="pytorch")
        normalizer_pytorch.fit(reference_image)
        result_pytorch = normalizer_pytorch.transform(source_image_torch)
        result_pytorch_cpu = result_pytorch.squeeze(0).cpu().float()

        # CUDA backend
        try:
            normalizer_cuda = Reinhard(device=cuda_device, backend="cuda")
            normalizer_cuda.fit(reference_image)
            result_cuda = normalizer_cuda.transform(source_image_torch)
            result_cuda_cpu = result_cuda.squeeze(0).cpu().float()

            rel_abs_error = compute_relative_absolute_error(result_cuda_cpu, result_pytorch_cpu)

            assert rel_abs_error < 0.01, f"CUDA vs PyTorch Reinhard relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"
        except (ImportError, NotImplementedError) as e:
            pytest.skip(f"CUDA backend not available: {e}")

    def test_macenko_cuda_vs_pytorch(self, reference_image, source_image_torch, cuda_device):
        """Test CUDA backend vs PyTorch backend for Macenko normalization."""
        # PyTorch backend
        normalizer_pytorch = Macenko(device=cuda_device, backend="pytorch")
        normalizer_pytorch.fit(reference_image)
        result_pytorch = normalizer_pytorch.transform(source_image_torch)
        result_pytorch_cpu = result_pytorch.squeeze(0).cpu().float()

        # CUDA backend
        try:
            normalizer_cuda = Macenko(device=cuda_device, backend="cuda")
            normalizer_cuda.fit(reference_image)
            result_cuda = normalizer_cuda.transform(source_image_torch)
            result_cuda_cpu = result_cuda.squeeze(0).cpu().float()

            rel_abs_error = compute_relative_absolute_error(result_cuda_cpu, result_pytorch_cpu)

            assert rel_abs_error < 0.01, f"CUDA vs PyTorch Macenko relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"
        except (ImportError, NotImplementedError) as e:
            pytest.skip(f"CUDA backend not available: {e}")

    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_cuda_vs_pytorch(self, reference_image, source_image_torch, cuda_device, channel_axis):
        """Test CUDA backend vs PyTorch backend for histogram matching."""
        # Create converter
        converter = ChannelFormatConverter(channel_axis=channel_axis)

        # Prepare inputs
        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image_torch)

        # PyTorch backend
        normalizer_pytorch = HistogramMatching(device=cuda_device, backend="pytorch", channel_axis=channel_axis)
        normalizer_pytorch.fit(ref_input)
        result_pytorch = normalizer_pytorch.transform(src_input)
        result_pytorch_chw = converter.to_chw(result_pytorch, squeeze_batch=True)

        # CUDA backend
        try:
            normalizer_cuda = HistogramMatching(device=cuda_device, backend="cuda", channel_axis=channel_axis)
            normalizer_cuda.fit(ref_input)
            result_cuda = normalizer_cuda.transform(src_input)
            result_cuda_chw = converter.to_chw(result_cuda, squeeze_batch=True)

            rel_abs_error = compute_relative_absolute_error(result_cuda_chw, result_pytorch_chw)

            assert rel_abs_error < 0.01, f"CUDA vs PyTorch histogram matching relative absolute error too large: {rel_abs_error:.6f}, expected <0.01 (channel_axis={channel_axis})"
        except (ImportError, NotImplementedError) as e:
            pytest.skip(f"CUDA backend not available: {e}")


if __name__ == "__main__":
    pytest.main()
