# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Base classes for stain normalization algorithms.

This module provides abstract base class for scikit-learn style API.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from stainx.utils import get_device


class StainNormalizerBase(ABC, nn.Module):
    """
    Abstract base class for scikit-learn style stain normalization.

    This class provides the standard fit/transform interface
    compatible with scikit-learn pipelines.
    """

    def __init__(self, device: str | torch.device | None = None):
        """
        Initialize the stain normalizer.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to run computations on. If None, auto-detects best available.
        """
        super().__init__()
        self.device = get_device(device)
        self._is_fitted = False

    @abstractmethod
    def fit(self, images: torch.Tensor) -> "StainNormalizerBase":
        """
        Fit the normalizer to reference images.

        Parameters
        ----------
        images : torch.Tensor
            Reference images of shape (N, C, H, W) or (N, H, W, C)

        Returns
        -------
        self : StainNormalizerBase
            Returns self for method chaining
        """

    @abstractmethod
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

    def fit_transform(self, images: torch.Tensor) -> torch.Tensor:
        """
        Fit to reference images and transform them.

        Parameters
        ----------
        images : torch.Tensor
            Reference images to fit and transform

        Returns
        -------
        torch.Tensor
            Normalized images
        """
        self.fit(images)
        return self.transform(images)
