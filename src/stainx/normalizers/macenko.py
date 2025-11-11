# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Macenko stain normalization implementation.
"""

import torch

from stainx.normalizers._template import NormalizerTemplate


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
        self._target_max_conc = None

    def _get_cuda_class(self):
        """Get the CUDA backend class for Macenko."""
        from stainx.backends.cuda_backend import MacenkoCUDA

        return MacenkoCUDA

    def _get_pytorch_class(self):
        """Get the PyTorch backend class for Macenko."""
        from stainx.backends.torch_backend import MacenkoPyTorch

        return MacenkoPyTorch

    def _compute_reference_params(self, images: torch.Tensor) -> None:
        """
        Compute stain matrix from reference images using SVD.

        Parameters
        ----------
        images : torch.Tensor
            Reference images of shape (N, C, H, W) with C=3
        """
        # Use backend to compute reference stain matrix
        pytorch_class = self._get_pytorch_class()
        backend = pytorch_class(self.device)
        self._stain_matrix, self._target_max_conc = (
            backend.compute_reference_stain_matrix(images)
        )
        self._concentration_matrix = None

    def _get_reference_params(self) -> tuple:
        """
        Get Macenko reference parameters.

        Returns
        -------
        tuple
            (stain_matrix, target_max_conc)
        """
        return (self._stain_matrix, self._target_max_conc)
