# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Macenko stain normalization implementation.
"""

from typing import Optional, Union

import torch

from ._template import NormalizerTemplate


class Macenko(NormalizerTemplate):
    """
    Macenko stain normalization.
    
    Parameters
    ----------
    device : str or torch.device, optional
        Device to run computations on. If None, auto-detects best available.
    backend : str, optional
        Backend to use ('cuda', 'pytorch'). If None, auto-selects best.
    """
    
    def _init_algorithm_attributes(self):
        """Initialize Macenko-specific attributes."""
        self._stain_matrix = None
        self._concentration_matrix = None
    
    def _get_cuda_class(self):
        """Get the CUDA backend class for Macenko."""
        from ..backends.cuda_backend import MacenkoCUDA
        return MacenkoCUDA
    
    def _get_pytorch_class(self):
        """Get the PyTorch backend class for Macenko."""
        from ..backends.torch_backend import MacenkoPyTorch
        return MacenkoPyTorch
    
    def _compute_reference_params(self, images: torch.Tensor) -> None:
        """
        Compute stain and concentration matrices from images.
        
        Parameters
        ----------
        images : torch.Tensor
            Reference images
        """
        # TODO: Implement Macenko reference computation
        raise NotImplementedError("Macenko reference computation not yet implemented")
    
    def _get_reference_params(self) -> tuple:
        """
        Get Macenko reference parameters.
        
        Returns
        -------
        tuple
            (stain_matrix, concentration_matrix)
        """
        return (self._stain_matrix, self._concentration_matrix)
