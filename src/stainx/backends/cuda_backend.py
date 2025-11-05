# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
CUDA backend implementations for stain normalization.

This module provides Python wrappers for CUDA-accelerated implementations.
The actual CUDA kernels are compiled in the stainx_cuda package.
"""

from typing import Optional, Union

import torch

try:
    import stainx_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    stainx_cuda = None


class CUDABackendBase:
    """Base class for CUDA backend implementations."""
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Initialize CUDA backend.
        
        Parameters
        ----------
        device : str or torch.device, optional
            Device to run computations on. Defaults to CUDA device.
        """
        if not CUDA_AVAILABLE:
            raise ImportError(
                "stainx_cuda package is not installed or built. "
                "CUDA backend is not available. Use backend='pytorch' instead."
            )
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("CUDA is not available on this system")
        else:
            self.device = torch.device(device)
        
        if self.device.type != "cuda":
            raise ValueError(f"CUDA backend requires CUDA device, got {self.device.type}")


class HistogramMatchingCUDA(CUDABackendBase):
    """CUDA implementation of histogram matching."""
    
    def transform(self, images: torch.Tensor, reference_histogram: torch.Tensor) -> torch.Tensor:
        """
        Apply histogram matching transformation using CUDA.
        
        Parameters
        ----------
        images : torch.Tensor
            Input images of shape (N, C, H, W) or (N, H, W, C)
        reference_histogram : torch.Tensor
            Reference histogram
            
        Returns
        -------
        torch.Tensor
            Histogram-matched images
        """
        # Move tensors to CUDA device
        images = images.to(self.device)
        reference_histogram = reference_histogram.to(self.device)
        
        # Check if CUDA function is available
        if not hasattr(stainx_cuda, 'histogram_matching'):
            raise NotImplementedError(
                "CUDA histogram matching not yet implemented. "
                "The stainx_cuda extension is not built or the function is not available."
            )
        
        # Call CUDA implementation
        # This will raise AT_ERROR("CUDA histogram matching not yet implemented") 
        # from the C++ code if not implemented
        try:
            return stainx_cuda.histogram_matching(images, reference_histogram)
        except RuntimeError as e:
            # Catch the AT_ERROR from C++ code and re-raise with clearer message
            error_msg = str(e)
            if "not yet implemented" in error_msg.lower():
                raise NotImplementedError(f"CUDA histogram matching not yet implemented: {error_msg}") from e
            raise


class ReinhardCUDA(CUDABackendBase):
    """CUDA implementation of Reinhard normalization."""
    
    def transform(
        self, 
        images: torch.Tensor, 
        target_mean: torch.Tensor, 
        target_std: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Reinhard normalization using CUDA.
        
        Parameters
        ----------
        images : torch.Tensor
            Input images
        target_mean : torch.Tensor
            Target mean values
        target_std : torch.Tensor
            Target std values
            
        Returns
        -------
        torch.Tensor
            Normalized images
        """
        images = images.to(self.device)
        target_mean = target_mean.to(self.device)
        target_std = target_std.to(self.device)
        
        # TODO: Implement when CUDA kernel is ready
        raise NotImplementedError("CUDA Reinhard normalization not yet implemented")


class MacenkoCUDA(CUDABackendBase):
    """CUDA implementation of Macenko normalization."""
    
    def transform(
        self,
        images: torch.Tensor,
        stain_matrix: torch.Tensor,
        concentration_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Macenko normalization using CUDA.
        
        Parameters
        ----------
        images : torch.Tensor
            Input images
        stain_matrix : torch.Tensor
            Stain matrix
        concentration_map : torch.Tensor
            Concentration map
            
        Returns
        -------
        torch.Tensor
            Normalized images
        """
        images = images.to(self.device)
        stain_matrix = stain_matrix.to(self.device)
        concentration_map = concentration_map.to(self.device)
        
        # TODO: Implement when CUDA kernel is ready
        raise NotImplementedError("CUDA Macenko normalization not yet implemented")


class VahadaneCUDA(CUDABackendBase):
    """CUDA implementation of Vahadane normalization."""
    
    def transform(
        self,
        images: torch.Tensor,
        stain_matrix: torch.Tensor,
        concentration_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Vahadane normalization using CUDA.
        
        Parameters
        ----------
        images : torch.Tensor
            Input images
        stain_matrix : torch.Tensor
            Stain matrix
        concentration_map : torch.Tensor
            Concentration map
            
        Returns
        -------
        torch.Tensor
            Normalized images
        """
        images = images.to(self.device)
        stain_matrix = stain_matrix.to(self.device)
        concentration_map = concentration_map.to(self.device)
        
        # TODO: Implement when CUDA kernel is ready
        raise NotImplementedError("CUDA Vahadane normalization not yet implemented")

