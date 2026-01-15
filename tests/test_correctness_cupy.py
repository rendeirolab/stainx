# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cupy as cp
import numpy as np
import pytest
import torch
from skimage.exposure import match_histograms
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter


def compute_relative_absolute_error_cupy(x: cp.ndarray, y: cp.ndarray) -> float:
    assert x.shape == y.shape, "Arrays must have the same shape"
    assert x.dtype == y.dtype, "Arrays must have the same dtype"
    assert x.device == y.device, "Arrays must be on the same device"

    epsilon = 1e-16
    # Flatten arrays before computing norm
    y_flat = y.flatten()
    x_flat = x.flatten()
    norm_y = cp.linalg.norm(y_flat, ord=2) + epsilon
    rel_abs_error = cp.linalg.norm(cp.abs(x_flat - y_flat), ord=2) / norm_y
    return float(rel_abs_error)


class TestTorchstainComparisonCupy:
    @pytest.fixture
    def reference_image(self, device_cupy):
        cp.random.seed(42)
        with cp.cuda.Device(device_cupy):
            return (cp.random.rand(1, 3, 256, 256) * 255).round().astype(cp.uint8)

    @pytest.fixture
    def source_image_cupy(self, device_cupy):
        cp.random.seed(123)
        with cp.cuda.Device(device_cupy):
            return (cp.random.rand(1, 3, 256, 256) * 255).round().astype(cp.uint8)

    def test_reinhard_comparison(self, reference_image, source_image_cupy, device_cupy):
        # Convert to numpy for torchstain
        ref_chw = cp.asnumpy(reference_image.squeeze(0))
        src_chw = cp.asnumpy(source_image_cupy.squeeze(0))

        # Use torchstain as reference (it uses PyTorch internally)
        # torchstain expects torch tensors, so convert numpy to torch (torchstain imports torch internally)
        torchstain_normalizer = TorchReinhardNormalizer()
        # Convert numpy to torch tensor (torchstain's internal API requires torch tensors)
        # We use a local import to avoid top-level torch dependency in this CuPy test file
        ref_torch = torch.from_numpy(ref_chw)
        src_torch = torch.from_numpy(src_chw)
        torchstain_normalizer.fit(ref_torch)
        torchstain_result = torchstain_normalizer.normalize(src_torch)
        # Convert result back to numpy
        torchstain_np = torchstain_result.cpu().numpy()
        # Convert HWC to CHW format
        torchstain_chw = np.transpose(torchstain_np, (2, 0, 1)).astype(np.float32) if torchstain_np.ndim == 3 and torchstain_np.shape[2] == 3 else torchstain_np.astype(np.float32)

        normalizer = Reinhard(device=device_cupy, backend="cupy")
        normalizer.fit(reference_image)
        result = normalizer.transform(source_image_cupy)
        result = cp.asarray(result)  # Ensure it's a CuPy array
        result_cpu = cp.asnumpy(result.squeeze(0)).astype(np.float32)
        # Convert to CHW if needed
        result_chw = result_cpu if result_cpu.ndim == 3 and result_cpu.shape[0] == 3 else np.transpose(result_cpu, (2, 0, 1)) if result_cpu.ndim == 3 else result_cpu

        rel_abs_error = compute_relative_absolute_error_cupy(cp.asarray(result_chw), cp.asarray(torchstain_chw))

        assert rel_abs_error < 0.01, f"Relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"

    def test_macenko_comparison(self, reference_image, source_image_cupy, device_cupy):
        # Convert to numpy for torchstain (reference implementation uses PyTorch internally)
        ref_chw = cp.asnumpy(reference_image.squeeze(0))
        src_chw = cp.asnumpy(source_image_cupy.squeeze(0))

        # Use torchstain as reference (it uses PyTorch internally)
        # torchstain expects torch tensors, so convert numpy to torch (torchstain imports torch internally)
        torchstain_normalizer = TorchMacenkoNormalizer()
        # Convert numpy to torch tensor (torchstain's internal API requires torch tensors)
        # We use a local import to avoid top-level torch dependency in this CuPy test file
        ref_torch = torch.from_numpy(ref_chw)
        src_torch = torch.from_numpy(src_chw)
        torchstain_normalizer.fit(ref_torch)
        torchstain_result, _, _ = torchstain_normalizer.normalize(src_torch, stains=True)
        # Convert result back to numpy
        torchstain_np = torchstain_result.cpu().numpy()
        # Convert HWC to CHW format
        torchstain_chw = np.transpose(torchstain_np, (2, 0, 1)).astype(np.float32) if torchstain_np.ndim == 3 and torchstain_np.shape[2] == 3 else torchstain_np.astype(np.float32)

        normalizer = Macenko(device=device_cupy, backend="cupy")
        normalizer.fit(reference_image)
        result = normalizer.transform(source_image_cupy)
        result = cp.asarray(result)  # Ensure it's a CuPy array
        result_cpu = cp.asnumpy(result.squeeze(0)).astype(np.float32)
        # Convert to CHW if needed
        result_chw = result_cpu if result_cpu.ndim == 3 and result_cpu.shape[0] == 3 else np.transpose(result_cpu, (2, 0, 1)) if result_cpu.ndim == 3 else result_cpu

        rel_abs_error = compute_relative_absolute_error_cupy(cp.asarray(result_chw), cp.asarray(torchstain_chw))

        assert rel_abs_error < 0.1, f"Macenko relative absolute error too large: {rel_abs_error:.6f}, expected <0.1"

    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_comparison(self, reference_image, source_image_cupy, device_cupy, channel_axis):
        # Create converter for skimage (always uses channels-first for input)
        converter_skimage = ChannelFormatConverter(channel_axis=1)

        # Convert to numpy HWC format for skimage (which always uses channels-last)
        # Use .get() to explicitly convert CuPy arrays to NumPy
        ref_np_uint8 = converter_skimage.to_hwc(reference_image.get() if isinstance(reference_image, cp.ndarray) else reference_image, squeeze_batch=True)
        src_np_uint8 = converter_skimage.to_hwc(source_image_cupy.get() if isinstance(source_image_cupy, cp.ndarray) else source_image_cupy, squeeze_batch=True)

        # skimage always uses channels-last format
        skimage_result = match_histograms(src_np_uint8, ref_np_uint8, channel_axis=-1)
        skimage_tensor = cp.asarray(skimage_result).astype(cp.float32)  # (H, W, C)

        # Create converter for our normalizer
        converter = ChannelFormatConverter(channel_axis=channel_axis)

        # Prepare inputs in the format expected by our normalizer
        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image_cupy)

        # Test our implementation
        normalizer = HistogramMatching(device=device_cupy, backend="cupy", channel_axis=channel_axis)
        normalizer.fit(ref_input)
        result = normalizer.transform(src_input)

        # Convert both results to CHW format for comparison
        result_tensor_chw = converter.to_chw(result, squeeze_batch=True, return_torch=False)
        # Ensure result_tensor_chw is a CuPy array
        if isinstance(result_tensor_chw, np.ndarray):
            result_tensor_chw = cp.asarray(result_tensor_chw)
        skimage_tensor_chw = cp.transpose(skimage_tensor, (2, 0, 1))  # (C, H, W)

        rel_abs_error = compute_relative_absolute_error_cupy(result_tensor_chw, skimage_tensor_chw)

        # Use a more lenient threshold (5%) since implementations use different algorithms
        assert rel_abs_error < 0.05, f"Histogram matching relative absolute error too large: {rel_abs_error:.6f}, expected <0.05 (channel_axis={channel_axis})"


if __name__ == "__main__":
    pytest.main()
