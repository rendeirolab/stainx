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
    """
    Compute relative absolute error between two tensors.

    Relative absolute error: ||x - y|| / ||y|| where ||y|| is the L2 norm (Frobenius norm for tensors)
    Returns a single scalar value representing the relative absolute error.

    Parameters
    ----------
    x : torch.Tensor
        Result x tensor
    y : torch.Tensor
        Result y tensor

    Returns
    -------
    float
        Relative absolute error: ||x - y|| / ||y||
    """
    assert x.shape == y.shape, "Tensors must have the same shape"
    assert x.dtype == y.dtype, "Tensors must have the same dtype"
    assert x.device == y.device, "Tensors must be on the same device"

    epsilon = 1e-16
    norm_y = torch.norm(y, p=2) + epsilon
    rel_abs_error = torch.norm((x - y).abs(), p=2) / norm_y
    return rel_abs_error.item()


class TestTorchstainComparison:
    """Test that our implementations match torchstain results."""

    @pytest.fixture
    def reference_image(self, device):
        """Create a reference image for fitting."""
        torch.manual_seed(42)
        return (torch.rand(1, 3, 256, 256, device=device) * 255).round().to(torch.uint8)

    @pytest.fixture
    def source_image_torch(self, device):
        """Create a source image for transformation."""
        torch.manual_seed(123)
        return (torch.rand(1, 3, 256, 256, device=device) * 255).round().to(torch.uint8)

    def test_reinhard_comparison(self, reference_image, source_image_torch, device):
        """Test that our Reinhard implementation matches torchstain."""
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image_torch.squeeze(0).cpu()

        torchstain_normalizer = TorchReinhardNormalizer()
        torchstain_normalizer.fit(ref_chw)
        torchstain_result = torchstain_normalizer.normalize(src_chw)
        torchstain_tensor = torchstain_result.permute(2, 0, 1).float()

        normalizer = Reinhard(device=device)
        normalizer.fit(reference_image)
        result = normalizer.transform(source_image_torch)
        result_cpu = result.squeeze(0).cpu().float()

        rel_abs_error = compute_relative_absolute_error(result_cpu, torchstain_tensor)

        assert rel_abs_error < 0.01, (
            f"Relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"
        )

    def test_macenko_comparison(self, reference_image, source_image_torch, device):  # noqa: ARG002
        """Test that our Macenko implementation matches torchstain."""
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image_torch.squeeze(0).cpu()

        torchstain_normalizer = TorchMacenkoNormalizer()
        torchstain_normalizer.fit(ref_chw)
        torchstain_result, _, _ = torchstain_normalizer.normalize(src_chw, stains=True)
        torchstain_tensor = torchstain_result.permute(2, 0, 1).float()

        normalizer = Macenko(device=torch.device("cpu"))
        normalizer.fit(reference_image.cpu())
        result = normalizer.transform(source_image_torch.cpu())
        result_cpu = result.squeeze(0).cpu().float()

        rel_abs_error = compute_relative_absolute_error(result_cpu, torchstain_tensor)

        assert rel_abs_error < 0.01, (
            f"Macenko relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"
        )

    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_comparison(
        self, reference_image, source_image_torch, device, channel_axis
    ):
        """Test that our HistogramMatching implementation produces similar results to skimage.match_histograms."""
        # Create converter for skimage (always uses channels-first for input)
        converter_skimage = ChannelFormatConverter(channel_axis=1)

        # Convert to numpy HWC format for skimage (which always uses channels-last)
        ref_np_uint8 = converter_skimage.to_hwc(reference_image, squeeze_batch=True)
        src_np_uint8 = converter_skimage.to_hwc(source_image_torch, squeeze_batch=True)

        # skimage always uses channels-last format
        skimage_result = match_histograms(src_np_uint8, ref_np_uint8, channel_axis=-1)
        skimage_tensor = torch.from_numpy(skimage_result).float()  # (H, W, C)

        # Create converter for our normalizer
        converter = ChannelFormatConverter(channel_axis=channel_axis)

        # Prepare inputs in the format expected by our normalizer
        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image_torch)

        # Test our implementation
        normalizer = HistogramMatching(device=device, channel_axis=channel_axis)
        normalizer.fit(ref_input)
        result = normalizer.transform(src_input)

        # Convert both results to CHW format for comparison
        result_tensor_chw = converter.to_chw(result, squeeze_batch=True)
        skimage_tensor_chw = skimage_tensor.permute(2, 0, 1)  # (C, H, W)

        rel_abs_error = compute_relative_absolute_error(
            result_tensor_chw, skimage_tensor_chw
        )

        # Use a more lenient threshold (5%) since implementations use different algorithms
        assert rel_abs_error < 0.05, (
            f"Histogram matching relative absolute error too large: {rel_abs_error:.6f}, expected <0.05 (channel_axis={channel_axis})"
        )


if __name__ == "__main__":
    pytest.main()
