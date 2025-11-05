# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Vahadane stain normalization implementation.
"""

from typing import Optional, Union

import torch

from ._template import NormalizerTemplate


class Vahadane(NormalizerTemplate):
    """
    Vahadane stain normalization.
    
    Parameters
    ----------
    device : str or torch.device, optional
        Device to run computations on. If None, auto-detects best available.
    backend : str, optional
        Backend to use ('cuda', 'pytorch'). If None, auto-selects best.
    """
    
    def _init_algorithm_attributes(self):
        """Initialize Vahadane-specific attributes."""
        self._stain_basis = None
        self._concentration_basis = None
    
    def _get_cuda_class(self):
        """Get the CUDA backend class for Vahadane."""
        from ..backends.cuda_backend import VahadaneCUDA
        return VahadaneCUDA
    
    def _get_pytorch_class(self):
        """Get the PyTorch backend class for Vahadane."""
        from ..backends.torch_backend import VahadanePyTorch
        return VahadanePyTorch
    
    def _compute_reference_params(self, images: torch.Tensor) -> None:
        """
        Compute stain and concentration basis from images.
        
        Parameters
        ----------
        images : torch.Tensor
            Reference images
        """
        # TODO: Implement Vahadane reference computation
        raise NotImplementedError("Vahadane reference computation not yet implemented")
    
    def _get_reference_params(self) -> tuple:
        """
        Get Vahadane reference parameters.
        
        Returns
        -------
        tuple
            (stain_basis, concentration_basis)
        """
        return (self._stain_basis, self._concentration_basis)
