# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Histogram matching stain normalization implementation.
"""

import torch

from stainx.normalizers._template import NormalizerTemplate


class HistogramMatching(NormalizerTemplate):
    """
    Histogram matching stain normalization.

    Parameters
    ----------
    device : str or torch.device, optional
        Device to run computations on. If None, auto-detects best available.
    backend : str, optional
        Backend to use ('cuda', 'pytorch'). If None, auto-selects best.
    channel_axis : int, optional
        Axis containing color channels.
        - 1 or -3: channels-first format (N, C, H, W) - default
        - -1 or 3: channels-last format (N, H, W, C)
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        backend: str | None = None,
        channel_axis: int = 1,
    ):
        self.channel_axis = channel_axis
        super().__init__(device=device, backend=backend)

    def _init_algorithm_attributes(self):
        """Initialize HistogramMatching-specific attributes."""
        self._reference_histogram = None
        self._ref_vals = None
        self._ref_cdf = None
        self._ref_histograms_256 = None

    def _get_cuda_class(self):
        """Get the CUDA backend class for HistogramMatching."""
        from stainx.backends.cuda_backend import HistogramMatchingCUDA

        return HistogramMatchingCUDA

    def _get_pytorch_class(self):
        """Get the PyTorch backend class for HistogramMatching."""
        from stainx.backends.torch_backend import HistogramMatchingPyTorch

        return HistogramMatchingPyTorch

    def _get_backend_impl(self):
        """Get the backend implementation with channel_axis."""
        if self._backend_impl is None:
            if self.backend == "cuda":
                cuda_class = self._get_cuda_class()
                self._backend_impl = cuda_class(
                    self.device, channel_axis=self.channel_axis
                )
            else:
                pytorch_class = self._get_pytorch_class()
                self._backend_impl = pytorch_class(
                    self.device, channel_axis=self.channel_axis
                )
        return self._backend_impl

    def _compute_reference_params(self, images: torch.Tensor) -> None:
        """
        Compute reference histogram from images.

        Parameters
        ----------
        images : torch.Tensor
            Reference images of shape (N, C, H, W) with C=3
        """
        # Use backend to compute reference histograms
        pytorch_class = self._get_pytorch_class()
        backend = pytorch_class(self.device)
        (
            self._ref_vals,
            self._ref_cdf,
            self._ref_histograms_256,
            self._reference_histogram,
        ) = backend.compute_reference_histograms(images)

    def _get_reference_params(self) -> tuple:
        """
        Get HistogramMatching reference parameters.

        Returns
        -------
        tuple
            (per_channel_histograms,) where per_channel_histograms is a list of 256-bin histograms, one per channel
        """
        if self._ref_histograms_256 is not None and len(self._ref_histograms_256) > 0:
            return (self._ref_histograms_256,)
        return (self._reference_histogram,)
