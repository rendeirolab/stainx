# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from typing import Any

# Optional backend imports - only used if available
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None

from stainx.base import StainNormalizerBase


class NormalizerTemplate(StainNormalizerBase):
    """Template class for normalizers with backend selection.

    This class is backend-agnostic and can work with PyTorch, CuPy, or other backends.
    PyTorch is used by default but can be replaced by other backends.
    """

    def __init__(self, device: str | Any | None = None, backend: str | None = None):
        """Initialize the normalizer template.

        Args:
            device: Device specification (string or device-like object).
            backend: Backend name ("pytorch", "cuda", "cupy", etc.). If None, auto-selects.
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
            Backend name string ("pytorch", "cuda", "cupy", etc.).
        """
        # Check device type (try to get type attribute if available)
        device_type = None
        if hasattr(self.device, "type"):
            device_type = self.device.type
        elif isinstance(self.device, str):
            device_type = self.device

        # Try CUDA backend (PyTorch CUDA extension)
        if device_type == "cuda":
            try:
                from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

                # Check if PyTorch CUDA is available
                if _TORCH_AVAILABLE and CUDA_AVAILABLE and torch.cuda.is_available():
                    return "cuda"
            except (ImportError, AttributeError):
                pass

        # Try CuPy backend (future support)
        if device_type == "cuda" and _CUPY_AVAILABLE:
            try:
                if cp.cuda.is_available():
                    # CuPy backend not yet implemented, fall through to PyTorch
                    pass
            except (AttributeError, RuntimeError):
                pass

        # Default to PyTorch backend
        return "pytorch"

    def _get_backend_impl(self):
        if self._backend_impl is None:
            if self.backend == "cuda":
                cuda_class = self._get_cuda_class()
                self._backend_impl = cuda_class(self.device)
            else:
                pytorch_class = self._get_pytorch_class()
                self._backend_impl = pytorch_class(self.device)
        return self._backend_impl

    def _get_cuda_class(self):
        """Get the CUDA backend class. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_cuda_class")

    def _get_pytorch_class(self):
        """Get the PyTorch backend class. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_pytorch_class")

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

    def _get_backend_for_computation(self):
        """Get the best available backend for computation.

        This method is used for fitting operations that may need a specific backend.
        By default, uses PyTorch backend for fitting (CUDA backends typically don't have fit methods).

        Returns:
            Backend implementation instance.
        """
        # Get device type
        device_type = None
        if hasattr(self.device, "type"):
            device_type = self.device.type
        elif isinstance(self.device, str):
            device_type = self.device

        # Try to use CUDA device if available
        device = self.device
        if device_type == "cuda":
            try:
                from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

                try:
                    import torch

                    if CUDA_AVAILABLE and torch.cuda.is_available():
                        device = torch.device("cuda")
                except ImportError:
                    pass
            except (ImportError, AttributeError):
                pass

        # Use PyTorch backend for fitting (CUDA backends typically don't have fit methods)
        pytorch_class = self._get_pytorch_class()
        # Allow subclasses to override this to pass extra kwargs (e.g., channel_axis)
        kwargs = self._get_backend_kwargs()
        return pytorch_class(device, **kwargs)

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
