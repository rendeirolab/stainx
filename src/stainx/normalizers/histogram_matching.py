# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from typing import Any

from stainx.normalizers._template import NormalizerTemplate


class HistogramMatching(NormalizerTemplate):
    def __init__(self, device: Any | None = None, backend: str | None = None, channel_axis: int = 1):
        self.channel_axis = channel_axis
        super().__init__(device=device, backend=backend)

    def _init_algorithm_attributes(self):
        self._reference_histogram = None
        self._ref_vals = None
        self._ref_cdf = None
        self._ref_histograms_256 = None

    def _get_torch_cuda_class(self):
        from stainx.backends.torch_cuda_backend import HistogramMatchingCUDA

        return HistogramMatchingCUDA

    def _get_torch_class(self):
        from stainx.backends.torch_backend import HistogramMatchingTorch

        return HistogramMatchingTorch

    def _get_cupy_class(self):
        from stainx.backends.cupy_backend import HistogramMatchingCupy

        return HistogramMatchingCupy

    def _get_cupy_cuda_class(self):
        from stainx.backends.cupy_cuda_backend import HistogramMatchingCuPyCUDA

        return HistogramMatchingCuPyCUDA

    def _get_backend_impl(self):
        if self._backend_impl is None:
            if self.backend == "torch_cuda":
                cuda_class = self._get_torch_cuda_class()
                self._backend_impl = cuda_class(self.device, channel_axis=self.channel_axis)
            elif self.backend == "cupy_cuda":
                cupy_cuda_class = self._get_cupy_cuda_class()
                self._backend_impl = cupy_cuda_class(self.device, channel_axis=self.channel_axis)
            elif self.backend == "cupy":
                cupy_class = self._get_cupy_class()
                self._backend_impl = cupy_class(self.device, channel_axis=self.channel_axis)
            else:
                torch_class = self._get_torch_class()
                self._backend_impl = torch_class(self.device, channel_axis=self.channel_axis)
        return self._backend_impl

    def _get_backend_kwargs(self) -> dict:
        """Provide channel_axis for backend initialization."""
        return {"channel_axis": self.channel_axis}

    def _compute_reference_params(self, images: Any) -> None:
        import cupy as cp

        if isinstance(images, cp.ndarray):
            backend = self._get_backend_for_computation_cupy()
            (self._ref_vals, self._ref_cdf, self._ref_histograms_256, self._reference_histogram) = backend.compute_reference_histograms_cupy(images)
        else:
            backend = self._get_backend_for_computation_torch()
            (self._ref_vals, self._ref_cdf, self._ref_histograms_256, self._reference_histogram) = backend.compute_reference_histograms_torch(images)

    def _get_reference_params(self) -> tuple:
        if self._ref_histograms_256 is not None and len(self._ref_histograms_256) > 0:
            return (self._ref_histograms_256,)
        return (self._reference_histogram,)
