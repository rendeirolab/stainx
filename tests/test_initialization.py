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
from stainx.backends.torch_backend import (
    HistogramMatchingPyTorch,
    MacenkoPyTorch,
    ReinhardPyTorch,
)


class TestStainNormalizers:
    """Unified test cases for all stain normalizers."""

    @pytest.mark.parametrize("normalizer_class", [HistogramMatching, Reinhard, Macenko])
    def test_initialization(self, normalizer_class, device):
        """Test normalizer initialization."""
        normalizer = normalizer_class(device=device)

        assert normalizer.device == device
        assert normalizer.backend in ["cuda", "pytorch"]
        assert not normalizer._is_fitted

    @pytest.mark.parametrize("normalizer_class", [HistogramMatching, Reinhard, Macenko])
    def test_fit(self, normalizer_class, sample_images, device):
        """Test that fit works correctly."""
        normalizer = normalizer_class(device=device)

        # This should work when implemented, but will fail with NotImplementedError now
        result = normalizer.fit(sample_images)

        assert result is normalizer  # Should return self
        assert normalizer._is_fitted  # Should be fitted after fit()

    @pytest.mark.parametrize("normalizer_class", [HistogramMatching, Reinhard, Macenko])
    def test_transform_without_fit(self, normalizer_class, sample_images, device):
        """Test that transform raises error when not fitted."""
        normalizer = normalizer_class(device=device)

        with pytest.raises(ValueError, match="Must call fit"):
            normalizer.transform(sample_images)

    @pytest.mark.parametrize("normalizer_class", [HistogramMatching, Reinhard, Macenko])
    def test_fit_transform(
        self, normalizer_class, sample_images, reference_images, device
    ):
        """Test fit and transform workflow."""
        normalizer = normalizer_class(device=device)

        # Fit on reference images - should work when implemented
        normalizer.fit(reference_images)
        assert normalizer._is_fitted

        # Transform sample images - should work when implemented
        result = normalizer.transform(sample_images)

        # Check output shape matches input
        assert result.shape == sample_images.shape
        # Device comparison: cuda:0 == cuda (both are CUDA devices)
        assert result.device.type == device.type
        assert result.dtype == sample_images.dtype


class TestBackendImplementations:
    """Test backend implementations work correctly."""

    @pytest.mark.parametrize(
        ("backend_class", "args"),
        [
            (HistogramMatchingPyTorch, (torch.rand(4, 3, 16, 16), torch.rand(16))),
            (ReinhardPyTorch, (torch.rand(4, 3, 16, 16), torch.rand(3), torch.rand(3))),
            (
                MacenkoPyTorch,
                (torch.rand(4, 3, 16, 16), torch.rand(3, 2), torch.rand(2)),
            ),
        ],
    )
    def test_backend_transform(self, backend_class, args, device):
        """Test that backend transform methods work correctly."""
        backend = backend_class(device=device)
        images = args[0]

        result = backend.transform(*args)

        assert result.shape == images.shape
        assert result.device.type == device.type
        assert result.dtype == images.dtype


if __name__ == "__main__":
    pytest.main()
