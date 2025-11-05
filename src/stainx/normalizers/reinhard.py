# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Reinhard stain normalization implementation.
"""

from typing import Optional, Union

import torch

from ._template import NormalizerTemplate


class Reinhard(NormalizerTemplate):
    """
    Reinhard stain normalization.
    
    Parameters
    ----------
    device : str or torch.device, optional
        Device to run computations on. If None, auto-detects best available.
    backend : str, optional
        Backend to use ('cuda', 'pytorch'). If None, auto-selects best.
    """
    
    def _init_algorithm_attributes(self):
        """Initialize Reinhard-specific attributes."""
        self._reference_mean = None
        self._reference_std = None
    
    def _get_cuda_class(self):
        """Get the CUDA backend class for Reinhard."""
        from ..backends.cuda_backend import ReinhardCUDA
        return ReinhardCUDA
    
    def _get_pytorch_class(self):
        """Get the PyTorch backend class for Reinhard."""
        from ..backends.torch_backend import ReinhardPyTorch
        return ReinhardPyTorch
    
    def _compute_reference_params(self, images: torch.Tensor) -> None:
        """
        Compute reference mean and std from images.
        
        Parameters
        ----------
        images : torch.Tensor
            Reference images
        """
        # TODO: Implement Reinhard reference computation
        raise NotImplementedError("Reinhard reference computation not yet implemented")
    
    def _get_reference_params(self) -> tuple:
        """
        Get Reinhard reference parameters.
        
        Returns
        -------
        tuple
            (reference_mean, reference_std)
        """
        return (self._reference_mean, self._reference_std)
