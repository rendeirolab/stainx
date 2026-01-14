# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from abc import ABC, abstractmethod
from typing import Any

from stainx.utils import get_device


class StainNormalizerBase(ABC):
    """Base class for stain normalizers.

    This class is backend-agnostic and does not depend on PyTorch.
    PyTorch support is optional and can be enabled by subclasses.
    """

    def __init__(self, device: str | Any | None = None):
        """Initialize the normalizer.

        Args:
            device: Device specification (string or device-like object).
                   Can be "cpu", "cuda", "mps", or a device object from any backend.
        """
        self.device = get_device(device)
        self._is_fitted = False

    @abstractmethod
    def fit(self, images: Any) -> "StainNormalizerBase":
        """Fit the normalizer to reference images.

        Args:
            images: Input images (tensor-like object from any backend).

        Returns:
            Self for method chaining.
        """

    @abstractmethod
    def transform(self, images: Any) -> Any:
        """Transform images using the fitted normalizer.

        Args:
            images: Input images (tensor-like object from any backend).

        Returns:
            Normalized images (same type as input).
        """

    def fit_transform(self, images: Any) -> Any:
        """Fit and transform images in one step.

        Args:
            images: Input images (tensor-like object from any backend).

        Returns:
            Normalized images (same type as input).
        """
        self.fit(images)
        return self.transform(images)
