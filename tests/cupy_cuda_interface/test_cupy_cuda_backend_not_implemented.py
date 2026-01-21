# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""Parity tests: **CuPy CUDA extension backend** vs **CuPy backend**.

What this file tests
--------------------
On a CUDA-capable system, compare:
- `backend="cupy_cuda"` (custom CUDA extension via `stainx_cuda_cupy`)
- `backend="cupy"` (CuPy ops)

for the same inputs and assert outputs match within tolerance.
"""

import cupy as cp
import pytest

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter


def compute_relative_absolute_error_cupy(x: cp.ndarray, y: cp.ndarray) -> float:
    assert x.shape == y.shape, "Arrays must have the same shape"
    assert x.dtype == y.dtype, "Arrays must have the same dtype"
    assert x.device == y.device, "Arrays must be on the same device"

    epsilon = 1e-16
    y_flat = y.ravel()
    x_flat = x.ravel()
    norm_y = cp.linalg.norm(y_flat, ord=2) + epsilon
    rel_abs_error = cp.linalg.norm(cp.abs(x_flat - y_flat), ord=2) / norm_y
    return float(rel_abs_error)


def _cupy_cuda_available():
    """Safely check if CuPy CUDA is available."""
    try:
        return cp.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cupy_cuda_available(), reason="CUDA is not available")
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
        """Test CUDA backend vs CuPy backend for Reinhard normalization."""
        # CuPy backend
        normalizer_cupy = Reinhard(device=cuda_device, backend="cupy")
        normalizer_cupy.fit(reference_image)
        result_cupy = normalizer_cupy.transform(source_image_cupy)
        result_cupy_cpu = cp.asnumpy(result_cupy.squeeze(0)).astype(cp.float32)

        # CUDA backend
        normalizer_cuda = Reinhard(device=cuda_device, backend="cupy_cuda")
        normalizer_cuda.fit(reference_image)
        result_cuda = normalizer_cuda.transform(source_image_cupy)
        result_cuda_cpu = cp.asnumpy(result_cuda.squeeze(0)).astype(cp.float32)

        # Convert both to CuPy arrays on the same device for comparison
        with cp.cuda.Device(cuda_device):
            result_cupy_cp = cp.asarray(result_cupy_cpu).astype(cp.float32)
            result_cuda_cp = cp.asarray(result_cuda_cpu).astype(cp.float32)

        rel_abs_error = compute_relative_absolute_error_cupy(result_cuda_cp, result_cupy_cp)
        assert rel_abs_error < 0.01, f"CUDA vs CuPy Reinhard relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"

    def test_macenko_cuda_vs_cupy(self, reference_image, source_image_cupy, cuda_device):
        """Test CUDA backend vs CuPy backend for Macenko normalization."""
        # CuPy backend
        normalizer_cupy = Macenko(device=cuda_device, backend="cupy")
        normalizer_cupy.fit(reference_image)
        result_cupy = normalizer_cupy.transform(source_image_cupy)
        result_cupy_cpu = cp.asnumpy(result_cupy.squeeze(0)).astype(cp.float32)

        # CUDA backend
        normalizer_cuda = Macenko(device=cuda_device, backend="cupy_cuda")
        normalizer_cuda.fit(reference_image)
        result_cuda = normalizer_cuda.transform(source_image_cupy)
        result_cuda_cpu = cp.asnumpy(result_cuda.squeeze(0)).astype(cp.float32)

        # Convert both to CuPy arrays on the same device for comparison
        with cp.cuda.Device(cuda_device):
            result_cupy_cp = cp.asarray(result_cupy_cpu).astype(cp.float32)
            result_cuda_cp = cp.asarray(result_cuda_cpu).astype(cp.float32)

        rel_abs_error = compute_relative_absolute_error_cupy(result_cuda_cp, result_cupy_cp)
        assert rel_abs_error < 0.01, f"CUDA vs CuPy Macenko relative absolute error too large: {rel_abs_error:.6f}, expected <0.01"

    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_cuda_vs_cupy(self, reference_image, source_image_cupy, cuda_device, channel_axis):
        """Test CUDA backend vs CuPy backend for histogram matching."""
        converter = ChannelFormatConverter(channel_axis=channel_axis)

        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image_cupy)

        # CuPy backend
        normalizer_cupy = HistogramMatching(device=cuda_device, backend="cupy", channel_axis=channel_axis)
        normalizer_cupy.fit(ref_input)
        result_cupy = normalizer_cupy.transform(src_input)
        result_cupy_chw = converter.to_chw(result_cupy, squeeze_batch=True, return_torch=False)

        # CUDA backend
        normalizer_cuda = HistogramMatching(device=cuda_device, backend="cupy_cuda", channel_axis=channel_axis)
        normalizer_cuda.fit(ref_input)
        result_cuda = normalizer_cuda.transform(src_input)
        result_cuda_chw = converter.to_chw(result_cuda, squeeze_batch=True, return_torch=False)

        # Convert both to CuPy arrays on the same device for comparison
        with cp.cuda.Device(cuda_device):
            result_cupy_cp = cp.asarray(result_cupy_chw).astype(cp.float32)
            result_cuda_cp = cp.asarray(result_cuda_chw).astype(cp.float32)

        rel_abs_error = compute_relative_absolute_error_cupy(result_cuda_cp, result_cupy_cp)
        assert rel_abs_error < 0.01, f"CUDA vs CuPy histogram matching relative absolute error too large: {rel_abs_error:.6f}, expected <0.01 (channel_axis={channel_axis})"


if __name__ == "__main__":
    pytest.main()
