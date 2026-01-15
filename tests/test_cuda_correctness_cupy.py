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

# Skip all tests if CuPy CUDA is not available
if not cp.cuda.is_available():
    pytest.skip("CuPy CUDA is not available", allow_module_level=True)

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter


def compute_relative_absolute_error_cupy(x: cp.ndarray, y: cp.ndarray) -> float:
    assert x.shape == y.shape, "Arrays must have the same shape"
    assert x.dtype == y.dtype, "Arrays must have the same dtype"
    assert x.device == y.device, "Arrays must be on the same device"

    epsilon = 1e-16
    norm_y = cp.linalg.norm(y, ord=2) + epsilon
    rel_abs_error = cp.linalg.norm(cp.abs(x - y), ord=2) / norm_y
    return float(rel_abs_error)


@pytest.mark.cuda
class TestCUDABackendComparisonCupy:
    @pytest.fixture
    def cuda_device(self):
        if not cp.cuda.is_available():
            pytest.skip("CUDA is not available")
        return cp.cuda.Device(0)

    @pytest.fixture
    def reference_image(self, cuda_device):
        cp.random.seed(42)
        with cp.cuda.Device(cuda_device):
            return (cp.random.rand(1, 3, 256, 256) * 255).round().astype(cp.uint8)

    @pytest.fixture
    def source_image_cupy(self, cuda_device):
        cp.random.seed(123)
        with cp.cuda.Device(cuda_device):
            return (cp.random.rand(1, 3, 256, 256) * 255).round().astype(cp.uint8)

    def test_reinhard_cuda_vs_cupy(self, reference_image, source_image_cupy, cuda_device):
        """Test CuPy backend for Reinhard normalization (self-consistency test)."""
        # Test CuPy backend
        normalizer_cupy = Reinhard(device=cuda_device, backend="cupy")
        normalizer_cupy.fit(reference_image)
        result_cupy = normalizer_cupy.transform(source_image_cupy)
        result_cupy_cpu = cp.asnumpy(result_cupy.squeeze(0)).astype(np.float32)

        # Verify result is valid (not NaN, not Inf, in expected range)
        assert not cp.isnan(result_cupy).any(), "Result contains NaN values"
        assert not cp.isinf(result_cupy).any(), "Result contains Inf values"
        assert result_cupy_cpu.min() >= 0, "Result contains negative values"
        assert result_cupy_cpu.max() <= 255, "Result exceeds maximum value"
        assert result_cupy_cpu.shape == reference_image.squeeze(0).shape, "Result shape mismatch"

    def test_macenko_cuda_vs_cupy(self, reference_image, source_image_cupy, cuda_device):
        """Test CuPy backend for Macenko normalization (self-consistency test)."""
        # Test CuPy backend
        normalizer_cupy = Macenko(device=cuda_device, backend="cupy")
        normalizer_cupy.fit(reference_image)
        result_cupy = normalizer_cupy.transform(source_image_cupy)
        result_cupy_cpu = cp.asnumpy(result_cupy.squeeze(0)).astype(np.float32)

        # Verify result is valid (not NaN, not Inf, in expected range)
        assert not cp.isnan(result_cupy).any(), "Result contains NaN values"
        assert not cp.isinf(result_cupy).any(), "Result contains Inf values"
        assert result_cupy_cpu.min() >= 0, "Result contains negative values"
        assert result_cupy_cpu.max() <= 255, "Result exceeds maximum value"
        assert result_cupy_cpu.shape == reference_image.squeeze(0).shape, "Result shape mismatch"

    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_cuda_vs_cupy(self, reference_image, source_image_cupy, cuda_device, channel_axis):
        """Test CuPy backend for histogram matching (self-consistency test)."""
        # Create converter
        converter = ChannelFormatConverter(channel_axis=channel_axis)

        # Prepare inputs
        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image_cupy)

        # CuPy backend
        normalizer_cupy = HistogramMatching(device=cuda_device, backend="cupy", channel_axis=channel_axis)
        normalizer_cupy.fit(ref_input)
        result_cupy = normalizer_cupy.transform(src_input)
        result_cupy_chw = converter.to_chw(result_cupy, squeeze_batch=True, return_torch=False)

        # Ensure result is a CuPy or numpy array
        result_cupy_cpu = cp.asnumpy(result_cupy_chw) if isinstance(result_cupy_chw, cp.ndarray) else result_cupy_chw

        # Verify result is valid (not NaN, not Inf, in expected range)
        assert not np.isnan(result_cupy_cpu).any(), "Result contains NaN values"
        assert not np.isinf(result_cupy_cpu).any(), "Result contains Inf values"
        assert result_cupy_cpu.min() >= 0, "Result contains negative values"
        assert result_cupy_cpu.max() <= 255, "Result exceeds maximum value"


if __name__ == "__main__":
    pytest.main()
