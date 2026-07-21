# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from typing import Any

import torch

from stainx.base import StainNormalizerBase

_VALID_BACKENDS = frozenset({"torch", "torch_cuda"})


class NormalizerTemplate(StainNormalizerBase):
    """Template class for normalizers with Torch backend selection.

    Backends:
    - ``torch``: PyTorch ops on CPU / CUDA / MPS
    - ``torch_cuda``: compiled CUDA extension (optional). Auto-select falls back to
      ``torch`` if the extension is unavailable; ``backend="torch_cuda"`` raises.
    """

    def __init__(self, device: str | Any | None = None, backend: str | None = None):
        """Initialize the normalizer template.

        Args:
            device: Device specification (string or device-like object).
            backend: Backend name (``"torch"`` or ``"torch_cuda"``). If None, auto-selects.
        """
        super().__init__(device)
        if backend is not None and backend not in _VALID_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Valid backends: {sorted(_VALID_BACKENDS)}")
        if backend == "torch_cuda":
            from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

            if not CUDA_AVAILABLE:
                raise ImportError("Backend 'torch_cuda' requires the stainx_cuda_torch extension. Rebuild with CUDA/nvcc or use backend='torch'.")
        self.backend = backend or self._select_backend()
        self._backend_impl = None
        self._init_algorithm_attributes()

    def _init_algorithm_attributes(self):
        """Initialize algorithm-specific attributes. Override in subclasses."""

    def _select_backend(self) -> str:
        """Select the best available backend based on device and availability."""
        device_type = None
        if hasattr(self.device, "type"):
            device_type = self.device.type
        elif isinstance(self.device, str):
            device_type = self.device.split(":")[0]

        if device_type != "cuda":
            return "torch"

        from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE as TORCH_CUDA_AVAILABLE

        if TORCH_CUDA_AVAILABLE and torch.cuda.is_available():
            return "torch_cuda"
        return "torch"

    def _get_backend_impl(self):
        if self._backend_impl is None:
            if self.backend == "torch_cuda":
                cuda_class = self._get_torch_cuda_class()
                kwargs = self._get_backend_kwargs()
                self._backend_impl = cuda_class(self.device, **kwargs)
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

    def fit(self, images: Any) -> "NormalizerTemplate":
        """Fit the normalizer to reference images.

        Args:
            images: Input images (torch.Tensor).

        Returns:
            Self for method chaining.
        """
        self._compute_reference_params(images)
        self._is_fitted = True
        return self

    def transform(self, images: Any) -> Any:
        """Transform images using the fitted normalizer.

        Args:
            images: Input images (torch.Tensor).

        Returns:
            Normalized images (same type as input).
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")

        reference_params = self._get_reference_params()
        backend_impl = self._get_backend_impl()
        result = backend_impl.transform(images, *reference_params)
        if hasattr(self, "normalize_to_0_1") and self.normalize_to_0_1:
            result = result / 255.0
        return result

    def _get_backend_for_computation_torch(self):
        """Get a Torch backend instance for fit-time computation."""
        device_type = None
        if hasattr(self.device, "type"):
            device_type = self.device.type
        elif isinstance(self.device, str):
            device_type = self.device.split(":")[0]

        device = self.device
        if device_type == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")

        torch_class = self._get_torch_class()
        kwargs = self._get_backend_kwargs()
        return torch_class(device, **kwargs)

    def _get_backend_kwargs(self) -> dict:
        """Override in subclasses to provide extra kwargs for backend initialization."""
        return {}

    def _compute_reference_params(self, images: Any) -> None:
        """Compute reference parameters from images. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _compute_reference_params")

    def _get_reference_params(self) -> tuple:
        """Get reference parameters for transformation. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_reference_params")
