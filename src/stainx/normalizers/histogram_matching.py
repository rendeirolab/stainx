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

    def _get_backend_kwargs(self) -> dict:
        """Provide channel_axis for backend initialization."""
        return {"channel_axis": self.channel_axis}

    def _compute_reference_params(self, images: Any) -> None:
        backend = self._get_backend_for_computation_torch()
        (self._ref_vals, self._ref_cdf, self._ref_histograms_256, self._reference_histogram) = backend.compute_reference_histograms_torch(images)

    def _get_reference_params(self) -> tuple:
        if self._ref_histograms_256 is not None and len(self._ref_histograms_256) > 0:
            return (self._ref_histograms_256,)
        return (self._reference_histogram,)
