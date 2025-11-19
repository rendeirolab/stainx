# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import torch

from stainx.base import StainNormalizerBase


class NormalizerTemplate(StainNormalizerBase):
    def __init__(self, device: str | torch.device | None = None, backend: str | None = None):
        super().__init__(device)
        self.backend = backend or self._select_backend()
        self._backend_impl = None

        self._init_algorithm_attributes()

    def _init_algorithm_attributes(self):
        pass

    def _select_backend(self) -> str:
        try:
            from stainx.backends.cuda_backend import CUDA_AVAILABLE
            if CUDA_AVAILABLE and torch.cuda.is_available() and self.device.type == "cuda":
                return "cuda"
        except (ImportError, AttributeError):
            pass
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
        raise NotImplementedError("Subclasses must implement _get_cuda_class")

    def _get_pytorch_class(self):
        raise NotImplementedError("Subclasses must implement _get_pytorch_class")

    def fit(self, images: torch.Tensor) -> "NormalizerTemplate":
        self._compute_reference_params(images)
        self._is_fitted = True
        return self

    def transform(self, images: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")

        reference_params = self._get_reference_params()
        backend_impl = self._get_backend_impl()
        return backend_impl.transform(images, *reference_params)

    def _get_backend_for_computation(self):
        """Get the best available backend for computation (CUDA device if available, else PyTorch)."""
        # Check if CUDA is available and use CUDA device if possible
        use_cuda_device = False
        try:
            from stainx.backends.cuda_backend import CUDA_AVAILABLE

            # Use CUDA device if CUDA is available and device is CUDA
            if CUDA_AVAILABLE and torch.cuda.is_available():
                if self.device.type == "cuda" or (isinstance(self.device, str) and self.device == "cuda"):
                    device = torch.device("cuda")
                else:
                    device = self.device
            else:
                device = self.device
        except (ImportError, AttributeError):
            device = self.device
        
        # Use PyTorch backend (CUDA backends don't have compute_reference methods)
        pytorch_class = self._get_pytorch_class()
        # Allow subclasses to override this to pass extra kwargs (e.g., channel_axis)
        kwargs = self._get_backend_kwargs()
        return pytorch_class(device, **kwargs)

    def _get_backend_kwargs(self) -> dict:
        """Override in subclasses to provide extra kwargs for backend initialization."""
        return {}

    def _compute_reference_params(self, images: torch.Tensor) -> None:
        raise NotImplementedError("Subclasses must implement _compute_reference_params")

    def _get_reference_params(self) -> tuple:
        raise NotImplementedError("Subclasses must implement _get_reference_params")
