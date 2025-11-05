# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
from stainx import HistogramMatching, Reinhard, Macenko, Vahadane
from stainx.backends.torch_backend import (
    HistogramMatchingPyTorch, ReinhardPyTorch, MacenkoPyTorch, VahadanePyTorch
)


class TestStainNormalizers:
    """Unified test cases for all stain normalizers."""
    
    @pytest.mark.parametrize("normalizer_class", [
        HistogramMatching, Reinhard, Macenko, Vahadane
    ])
    def test_initialization(self, normalizer_class, device):
        """Test normalizer initialization."""
        normalizer = normalizer_class(device=device)
        
        assert normalizer.device == device
        assert normalizer.backend in ["cuda", "pytorch"]
        assert not normalizer._is_fitted
    
    @pytest.mark.parametrize("normalizer_class", [
        HistogramMatching, Reinhard, Macenko, Vahadane
    ])
    def test_fit(self, normalizer_class, sample_images, device):
        """Test that fit works correctly."""
        normalizer = normalizer_class(device=device)
        
        # This should work when implemented, but will fail with NotImplementedError now
        result = normalizer.fit(sample_images)
        
        assert result is normalizer  # Should return self
        assert normalizer._is_fitted  # Should be fitted after fit()
    
    @pytest.mark.parametrize("normalizer_class", [
        HistogramMatching, Reinhard, Macenko, Vahadane
    ])
    def test_transform_without_fit(self, normalizer_class, sample_images, device):
        """Test that transform raises error when not fitted."""
        normalizer = normalizer_class(device=device)
        
        with pytest.raises(ValueError, match="Must call fit"):
            normalizer.transform(sample_images)
    
    @pytest.mark.parametrize("normalizer_class", [
        HistogramMatching, Reinhard, Macenko, Vahadane
    ])
    def test_fit_transform(self, normalizer_class, sample_images, reference_images, device):
        """Test fit and transform workflow."""
        normalizer = normalizer_class(device=device)
        
        # Fit on reference images - should work when implemented
        normalizer.fit(reference_images)
        assert normalizer._is_fitted
        
        # Transform sample images - should work when implemented
        result = normalizer.transform(sample_images)
        
        # Check output shape matches input
        assert result.shape == sample_images.shape
        assert result.device == device
        assert result.dtype == sample_images.dtype


class TestBackendImplementations:
    """Test backend implementations work correctly."""
    
    @pytest.mark.parametrize("backend_class,args", [
        (HistogramMatchingPyTorch, (torch.rand(4, 3, 256, 256), torch.rand(256))),
        (ReinhardPyTorch, (torch.rand(4, 3, 256, 256), torch.rand(3), torch.rand(3))),
        (MacenkoPyTorch, (torch.rand(4, 3, 256, 256), torch.rand(3, 3), torch.rand(3, 256))),
        (VahadanePyTorch, (torch.rand(4, 3, 256, 256), torch.rand(3, 3), torch.rand(3, 256))),
    ])
    def test_backend_transform(self, backend_class, args, device):
        """Test that backend transform methods work correctly."""
        backend = backend_class(device=device)
        images = args[0]
        
        # Transform should work when implemented
        result = backend.transform(*args)
        
        # Check output shape matches input
        assert result.shape == images.shape
        assert result.device == device
        assert result.dtype == images.dtype
