# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Template base class for stain normalization algorithms.

This module provides a template class that contains all shared implementation
for stain normalizers, reducing code duplication across algorithm-specific classes.
"""

import torch

from stainx.base import StainNormalizerBase


class NormalizerTemplate(StainNormalizerBase):
    """
    Template base class for stain normalization algorithms.

    This class provides all shared implementation for stain normalizers,
    including backend selection and common methods.
    Algorithm-specific classes only need to define their unique attributes
    and the _compute_reference_params method.

    Parameters
    ----------
    device : str or torch.device, optional
        Device to run computations on. If None, auto-detects best available.
    backend : str, optional
        Backend to use ('cuda', 'pytorch'). If None, auto-selects best.
    """

    def __init__(
        self, device: str | torch.device | None = None, backend: str | None = None
    ):
        super().__init__(device)
        self.backend = backend or self._select_backend()
        self._backend_impl = None

        self._init_algorithm_attributes()

    def _init_algorithm_attributes(self):
        """Initialize algorithm-specific attributes. Override in subclasses."""

    def _select_backend(self) -> str:
        """Select the best available backend."""
        return "pytorch"

    def _get_backend_impl(self):
        """Get the backend implementation."""
        if self._backend_impl is None:
            if self.backend == "cuda":
                cuda_class = self._get_cuda_class()
                self._backend_impl = cuda_class(self.device)
            else:
                pytorch_class = self._get_pytorch_class()
                self._backend_impl = pytorch_class(self.device)
        return self._backend_impl

    def _get_cuda_class(self):
        """Get the CUDA backend class name. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_cuda_class")

    def _get_pytorch_class(self):
        """Get the PyTorch backend class name. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_pytorch_class")

    def fit(self, images: torch.Tensor) -> "NormalizerTemplate":
        """
        Fit the normalizer to reference images.

        Parameters
        ----------
        images : torch.Tensor
            Reference images of shape (N, C, H, W) or (N, H, W, C)

        Returns
        -------
        self : NormalizerTemplate
            Returns self for method chaining
        """
        self._compute_reference_params(images)
        self._is_fitted = True
        return self

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        """
        Transform images using fitted parameters.

        Parameters
        ----------
        images : torch.Tensor
            Images to normalize of shape (N, C, H, W) or (N, H, W, C)

        Returns
        -------
        torch.Tensor
            Normalized images
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")

        reference_params = self._get_reference_params()
        backend_impl = self._get_backend_impl()
        return backend_impl.transform(images, *reference_params)

    def _compute_reference_params(self, images: torch.Tensor) -> None:
        """
        Compute algorithm-specific reference parameters from images.
        Override in subclasses.

        Parameters
        ----------
        images : torch.Tensor
            Reference images
        """
        raise NotImplementedError("Subclasses must implement _compute_reference_params")

    def _get_reference_params(self) -> tuple:
        """
        Get algorithm-specific reference parameters as a tuple.
        Override in subclasses.

        Returns
        -------
        tuple
            Reference parameters for the algorithm
        """
        raise NotImplementedError("Subclasses must implement _get_reference_params")
