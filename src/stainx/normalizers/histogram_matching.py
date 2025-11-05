# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Histogram matching stain normalization implementation.
"""

from typing import Optional, Union

import torch

from ._template import NormalizerTemplate


class HistogramMatching(NormalizerTemplate):
    """
    Histogram matching stain normalization.
    
    Parameters
    ----------
    device : str or torch.device, optional
        Device to run computations on. If None, auto-detects best available.
    backend : str, optional
        Backend to use ('cuda', 'pytorch'). If None, auto-selects best.
    """
    
    def _init_algorithm_attributes(self):
        """Initialize HistogramMatching-specific attributes."""
        self._reference_histogram = None
    
    def _get_cuda_class(self):
        """Get the CUDA backend class for HistogramMatching."""
        from ..backends.cuda_backend import HistogramMatchingCUDA
        return HistogramMatchingCUDA
    
    def _get_pytorch_class(self):
        """Get the PyTorch backend class for HistogramMatching."""
        from ..backends.torch_backend import HistogramMatchingPyTorch
        return HistogramMatchingPyTorch
    
    def _compute_reference_params(self, images: torch.Tensor) -> None:
        """
        Compute reference histogram from images.
        
        Parameters
        ----------
        images : torch.Tensor
            Reference images
        """
        # TODO: Implement histogram computation
        raise NotImplementedError("Histogram computation not yet implemented")
    
    def _get_reference_params(self) -> tuple:
        """
        Get HistogramMatching reference parameters.
        
        Returns
        -------
        tuple
            (reference_histogram,)
        """
        return (self._reference_histogram,)
