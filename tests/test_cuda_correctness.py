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
from skimage.exposure import match_histograms
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

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

    def test_reinhard_cuda_vs_torchstain(self, reference_image, source_image_torch, cuda_device):
        """Test CUDA backend vs torchstain reference for Reinhard normalization."""
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image_torch.squeeze(0).cpu()

        torchstain_normalizer = TorchReinhardNormalizer()
        torchstain_normalizer.fit(ref_chw)
        torchstain_result = torchstain_normalizer.normalize(src_chw)
        torchstain_tensor = torchstain_result.permute(2, 0, 1).float()

        try:
            normalizer_cuda = Reinhard(device=cuda_device, backend="cuda")
            normalizer_cuda.fit(reference_image)
            result_cuda = normalizer_cuda.transform(source_image_torch)
            result_cuda_cpu = result_cuda.squeeze(0).cpu().float()

            rel_abs_error = compute_relative_absolute_error(result_cuda_cpu, torchstain_tensor)

            assert rel_abs_error < 0.01, f"CUDA vs torchstain Reinhard relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"
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

    def test_macenko_cuda_vs_torchstain(self, reference_image, source_image_torch, cuda_device):
        """Test CUDA backend vs torchstain reference for Macenko normalization."""
        print("\n" + "=" * 80)
        print("DEBUG: test_macenko_cuda_vs_torchstain")
        print("=" * 80)

        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image_torch.squeeze(0).cpu()

        print(f"DEBUG: reference_image shape: {reference_image.shape}, dtype: {reference_image.dtype}, range: [{reference_image.min().item():.2f}, {reference_image.max().item():.2f}]")
        print(f"DEBUG: source_image_torch shape: {source_image_torch.shape}, dtype: {source_image_torch.dtype}, range: [{source_image_torch.min().item():.2f}, {source_image_torch.max().item():.2f}]")
        print(f"DEBUG: ref_chw shape: {ref_chw.shape}, dtype: {ref_chw.dtype}")
        print(f"DEBUG: src_chw shape: {src_chw.shape}, dtype: {src_chw.dtype}")

        torchstain_normalizer = TorchMacenkoNormalizer()
        torchstain_normalizer.fit(ref_chw)
        torchstain_result, _, _ = torchstain_normalizer.normalize(src_chw, stains=True)
        torchstain_tensor = torchstain_result.permute(2, 0, 1).float()

        print(f"DEBUG: torchstain_result shape: {torchstain_result.shape}, dtype: {torchstain_result.dtype}, range: [{torchstain_result.min().item():.2f}, {torchstain_result.max().item():.2f}]")
        print(f"DEBUG: torchstain_tensor shape: {torchstain_tensor.shape}, dtype: {torchstain_tensor.dtype}, range: [{torchstain_tensor.min().item():.2f}, {torchstain_tensor.max().item():.2f}]")

        try:
            normalizer_cuda = Macenko(device=cuda_device, backend="cuda")
            normalizer_cuda.fit(reference_image)

            print("DEBUG: CUDA normalizer fitted")
            print(f"DEBUG: CUDA stain_matrix shape: {normalizer_cuda._stain_matrix.shape if hasattr(normalizer_cuda, '_stain_matrix') else 'N/A'}")
            print(f"DEBUG: CUDA target_max_conc: {normalizer_cuda._target_max_conc if hasattr(normalizer_cuda, '_target_max_conc') else 'N/A'}")

            result_cuda = normalizer_cuda.transform(source_image_torch)
            result_cuda_cpu = result_cuda.squeeze(0).cpu().float()

            print(f"DEBUG: result_cuda shape: {result_cuda.shape}, dtype: {result_cuda.dtype}, range: [{result_cuda.min().item():.2f}, {result_cuda.max().item():.2f}]")
            print(f"DEBUG: result_cuda_cpu shape: {result_cuda_cpu.shape}, dtype: {result_cuda_cpu.dtype}, range: [{result_cuda_cpu.min().item():.2f}, {result_cuda_cpu.max().item():.2f}]")

            rel_abs_error = compute_relative_absolute_error(result_cuda_cpu, torchstain_tensor)

            print(f"DEBUG: Relative absolute error: {rel_abs_error:.6f}")
            print(f"DEBUG: torchstain_tensor sample (first 5x5):\n{torchstain_tensor[0, :5, :5]}")
            print(f"DEBUG: result_cuda_cpu sample (first 5x5):\n{result_cuda_cpu[0, :5, :5]}")
            print(f"DEBUG: Difference sample (first 5x5):\n{(result_cuda_cpu[0, :5, :5] - torchstain_tensor[0, :5, :5]).abs()}")

            assert rel_abs_error < 0.01, f"CUDA vs torchstain Macenko relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"
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

    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_cuda_vs_skimage(self, reference_image, source_image_torch, cuda_device, channel_axis):
        """Test CUDA backend vs skimage reference for histogram matching."""
        # Create converter for skimage (always uses channels-first for input)
        converter_skimage = ChannelFormatConverter(channel_axis=1)

        # Convert to numpy HWC format for skimage (which always uses channels-last)
        ref_np_uint8 = converter_skimage.to_hwc(reference_image.cpu(), squeeze_batch=True)
        src_np_uint8 = converter_skimage.to_hwc(source_image_torch.cpu(), squeeze_batch=True)

        # skimage always uses channels-last format
        skimage_result = match_histograms(src_np_uint8, ref_np_uint8, channel_axis=-1)
        skimage_tensor = torch.from_numpy(skimage_result).float()  # (H, W, C)

        # Create converter for our normalizer
        converter = ChannelFormatConverter(channel_axis=channel_axis)

        # Prepare inputs in the format expected by our normalizer
        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image_torch)

        # Test CUDA implementation
        try:
            normalizer_cuda = HistogramMatching(device=cuda_device, backend="cuda", channel_axis=channel_axis)
            normalizer_cuda.fit(ref_input)
            result_cuda = normalizer_cuda.transform(src_input)

            # Convert both results to CHW format for comparison
            result_cuda_chw = converter.to_chw(result_cuda, squeeze_batch=True)
            skimage_tensor_chw = skimage_tensor.permute(2, 0, 1)  # (C, H, W)

            rel_abs_error = compute_relative_absolute_error(result_cuda_chw, skimage_tensor_chw)

            # Use a more lenient threshold (5%) since implementations use different algorithms
            assert rel_abs_error < 0.05, f"CUDA vs skimage histogram matching relative absolute error too large: {rel_abs_error:.6f}, expected <0.05 (channel_axis={channel_axis})"
        except (ImportError, NotImplementedError) as e:
            pytest.skip(f"CUDA backend not available: {e}")


if __name__ == "__main__":
    pytest.main()
