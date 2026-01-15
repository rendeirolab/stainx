# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cupy as cp
import pytest

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.backends.cupy_backend import HistogramMatchingCupy, MacenkoCupy, ReinhardCupy


class TestStainNormalizersCupy:
    @pytest.mark.parametrize("normalizer_class", [HistogramMatching, Reinhard, Macenko])
    def test_initialization(self, normalizer_class, device_cupy):
        normalizer = normalizer_class(device=device_cupy, backend="cupy")

        assert normalizer.device == device_cupy
        assert normalizer.backend == "cupy"
        assert not normalizer._is_fitted

    @pytest.mark.parametrize("normalizer_class", [HistogramMatching, Reinhard, Macenko])
    def test_fit(self, normalizer_class, sample_images_cupy, device_cupy):
        normalizer = normalizer_class(device=device_cupy, backend="cupy")

        result = normalizer.fit(sample_images_cupy)

        assert result is normalizer  # Should return self
        assert normalizer._is_fitted  # Should be fitted after fit()

    @pytest.mark.parametrize("normalizer_class", [HistogramMatching, Reinhard, Macenko])
    def test_transform_without_fit(self, normalizer_class, sample_images_cupy, device_cupy):
        normalizer = normalizer_class(device=device_cupy, backend="cupy")

        with pytest.raises(ValueError, match="Must call fit"):
            normalizer.transform(sample_images_cupy)

    @pytest.mark.parametrize("normalizer_class", [HistogramMatching, Reinhard, Macenko])
    def test_fit_transform(self, normalizer_class, sample_images_cupy, reference_images_cupy, device_cupy):
        normalizer = normalizer_class(device=device_cupy, backend="cupy")

        # Fit on reference images
        normalizer.fit(reference_images_cupy)
        assert normalizer._is_fitted

        # Transform sample images
        result = normalizer.transform(sample_images_cupy)

        # Check output shape matches input
        assert result.shape == sample_images_cupy.shape
        # Device comparison: both should be CuPy arrays
        assert isinstance(result, cp.ndarray)
        assert result.dtype == sample_images_cupy.dtype


class TestBackendImplementationsCupy:
    @pytest.mark.parametrize(("backend_class", "shapes"), [(HistogramMatchingCupy, [(4, 3, 16, 16), (16,)]), (ReinhardCupy, [(4, 3, 16, 16), (3,), (3,)]), (MacenkoCupy, [(4, 3, 16, 16), (3, 2), (2,)])])
    def test_backend_transform(self, backend_class, shapes, device_cupy):
        # Create arrays inside the test to avoid CUDA access during collection
        args = tuple(cp.random.rand(*shape) for shape in shapes)
        backend = backend_class(device=device_cupy)
        images = args[0]

        result = backend.transform(*args)

        assert result.shape == images.shape
        assert isinstance(result, cp.ndarray)
        assert result.dtype == images.dtype


if __name__ == "__main__":
    pytest.main()
