# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from typing import Any

import cupy as cp
import torch

from stainx.base import StainNormalizerBase


class NormalizerTemplate(StainNormalizerBase):
    """Template class for normalizers with backend selection.

    This class is backend-agnostic and can work with Torch, CuPy, or other backends.
    Torch is used by default but can be replaced by other backends.
    """

    def __init__(self, device: str | Any | None = None, backend: str | None = None):
        """Initialize the normalizer template.

        Args:
            device: Device specification (string or device-like object).
            backend: Backend name ("torch", "cuda", "cupy", etc.). If None, auto-selects.
        """
        super().__init__(device)
        self.backend = backend or self._select_backend()
        print(f"Backend selected: {self.backend}")
        self._backend_impl = None

        self._init_algorithm_attributes()

    def _init_algorithm_attributes(self):
        """Initialize algorithm-specific attributes. Override in subclasses."""

    def _select_backend(self) -> str:
        """Select the best available backend based on device and availability.

        Returns:
            Backend name string ("torch", "cuda", "cupy", etc.).
        """
        # Check device type (try to get type attribute if available)
        device_type = None
        if hasattr(self.device, "type"):
            device_type = self.device.type
        elif isinstance(self.device, str):
            device_type = self.device
        elif isinstance(self.device, cp.cuda.Device):
            device_type = "cuda"

        # Try CUDA backend (Torch CUDA extension)
        if device_type == "cuda":
            from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

            # Check if Torch CUDA is available
            if CUDA_AVAILABLE and torch.cuda.is_available():
                return "cuda"

        # Try CuPy backend
        if device_type == "cuda" and cp.cuda.is_available():
            return "cupy"

        # Default to Torch backend
        return "torch"

    def _get_backend_impl(self):
        if self._backend_impl is None:
            if self.backend == "cuda":
                cuda_class = self._get_torch_cuda_class()
                self._backend_impl = cuda_class(self.device)
            elif self.backend == "cupy":
                cupy_class = self._get_cupy_class()
                kwargs = self._get_backend_kwargs()
                self._backend_impl = cupy_class(self.device, **kwargs)
            else:
                torch_class = self._get_torch_class()
                kwargs = self._get_backend_kwargs()
                self._backend_impl = torch_class(self.device, **kwargs)
        return self._backend_impl

    def _get_torch_cuda_class(self):
        """Get the Torch CUDA backend class. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_torch_cuda_class")

    def _get_torch_class(self):
        """Get the Torch backend class. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_torch_class")

    def _get_cupy_class(self):
        """Get the CuPy backend class. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_cupy_class")

    def fit(self, images: Any) -> "NormalizerTemplate":
        """Fit the normalizer to reference images.

        Args:
            images: Input images (tensor-like object from any backend).

        Returns:
            Self for method chaining.
        """
        self._compute_reference_params(images)
        self._is_fitted = True
        return self

    def transform(self, images: Any) -> Any:
        """Transform images using the fitted normalizer.

        Args:
            images: Input images (tensor-like object from any backend).

        Returns:
            Normalized images (same type as input).
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")

        reference_params = self._get_reference_params()
        backend_impl = self._get_backend_impl()
        return backend_impl.transform(images, *reference_params)

    def _get_backend_for_computation_torch(self):
        """Get the best available Torch backend for computation.

        This method is used for fitting operations that may need a specific backend.
        By default, uses Torch backend for fitting (CUDA backends typically don't have fit methods).

        Returns:
            Torch backend implementation instance.
        """
        # Get device type
        device_type = None
        if hasattr(self.device, "type"):
            device_type = self.device.type
        elif isinstance(self.device, str):
            device_type = self.device
        elif isinstance(self.device, cp.cuda.Device):
            device_type = "cuda"

        # Try to use CUDA device if available
        device = self.device
        if device_type == "cuda":
            from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

            if CUDA_AVAILABLE and torch.cuda.is_available():
                device = torch.device("cuda")

        # Use Torch backend for fitting (CUDA backends typically don't have fit methods)
        torch_class = self._get_torch_class()
        # Allow subclasses to override this to pass extra kwargs (e.g., channel_axis)
        kwargs = self._get_backend_kwargs()
        return torch_class(device, **kwargs)

    def _get_backend_for_computation_cupy(self):
        """Get the best available CuPy backend for computation.

        This method is used for fitting operations that may need a specific backend.
        By default, uses CuPy backend for fitting.

        Returns:
            CuPy backend implementation instance.
        """
        # Get device type
        device_type = None
        if hasattr(self.device, "type"):
            device_type = self.device.type
        elif isinstance(self.device, str):
            device_type = self.device
        elif isinstance(self.device, cp.cuda.Device):
            device_type = "cuda"

        # Try to use CUDA device if available
        device = self.device
        if device_type == "cuda" and cp.cuda.is_available():
            device = cp.cuda.Device(0)

        # Use CuPy backend for fitting
        cupy_class = self._get_cupy_class()
        # Allow subclasses to override this to pass extra kwargs (e.g., channel_axis)
        kwargs = self._get_backend_kwargs()
        return cupy_class(device, **kwargs)

    def _get_backend_kwargs(self) -> dict:
        """Override in subclasses to provide extra kwargs for backend initialization."""
        return {}

    def _compute_reference_params(self, images: Any) -> None:
        """Compute reference parameters from images. Override in subclasses.

        Args:
            images: Input images (tensor-like object from any backend).
        """
        raise NotImplementedError("Subclasses must implement _compute_reference_params")

    def _get_reference_params(self) -> tuple:
        """Get reference parameters for transformation. Override in subclasses.

        Returns:
            Tuple of reference parameters.
        """
        raise NotImplementedError("Subclasses must implement _get_reference_params")
