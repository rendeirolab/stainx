# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""Torch ``nn.Module`` transforms for training pipelines."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from stainx.normalizers import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter, get_device

MethodName = Literal["macenko", "reinhard", "histogram_matching"]
ModeName = Literal["reference", "batch"]

_METHOD_MAP = {"macenko": Macenko, "reinhard": Reinhard, "histogram_matching": HistogramMatching}


class StainNormalizerTransform(nn.Module):
    """Apply stain normalization inside a Torch / torchvision pipeline.

    Modes
    -----
    ``reference`` (default)
        Fit once on a fixed reference image, then transform every batch.
        Prefer this for supervised training reproducibility.

    ``batch``
        Fit on the current batch (or a designated index) at every forward call.
        Useful for exploratory / domain-shift visualization. Usually unsafe for
        reproducible supervised training because statistics change every step.
    """

    def __init__(self, method: MethodName | Any = "macenko", *, mode: ModeName = "reference", reference: torch.Tensor | None = None, device: str | torch.device | None = None, backend: str | None = None, channel_axis: int = 1, batch_ref_index: int = 0, normalize_to_0_1: bool = False, normalizer: Any | None = None):
        """
        Args:
            method: Algorithm name, or a constructed normalizer instance when ``normalizer`` is not passed.
            mode: ``"reference"`` or ``"batch"``.
            reference: Required for ``mode="reference"`` unless ``normalizer`` is already fitted.
            device: Torch device for the normalizer.
            backend: ``"torch"`` or ``"torch_cuda"`` (auto-selected if None).
            channel_axis: Channel axis for histogram matching (default NCHW → 1).
            batch_ref_index: Index within the batch used as reference when ``mode="batch"``.
            normalize_to_0_1: Forwarded to Macenko when constructing from ``method``.
            normalizer: Optional pre-built normalizer (skips construction from ``method``).
        """
        super().__init__()
        self.mode = mode
        self.channel_axis = channel_axis
        self.batch_ref_index = batch_ref_index
        self.device = get_device(device)

        if mode not in ("reference", "batch"):
            raise ValueError(f"Unsupported mode '{mode}'. Use 'reference' or 'batch'.")

        if normalizer is not None:
            self.normalizer = normalizer
        elif isinstance(method, str):
            if method not in _METHOD_MAP:
                raise ValueError(f"Unknown method '{method}'. Choose from {sorted(_METHOD_MAP)}")
            cls = _METHOD_MAP[method]
            if method == "histogram_matching":
                self.normalizer = cls(device=self.device, backend=backend, channel_axis=channel_axis)
            elif method == "macenko":
                self.normalizer = cls(device=self.device, backend=backend, normalize_to_0_1=normalize_to_0_1)
            else:
                self.normalizer = cls(device=self.device, backend=backend)
        else:
            # Treat as a pre-built normalizer instance (legacy example style)
            self.normalizer = method

        self._uses_hm = isinstance(self.normalizer, HistogramMatching)
        self._converter = ChannelFormatConverter(channel_axis=channel_axis) if self._uses_hm else None

        if mode == "reference":
            if reference is None and not getattr(self.normalizer, "_is_fitted", False):
                raise ValueError("mode='reference' requires a reference tensor (or a pre-fitted normalizer).")
            if reference is not None:
                self.fit_reference(reference)

    def fit_reference(self, reference: torch.Tensor) -> StainNormalizerTransform:
        """Fit the underlying normalizer on a reference image or batch."""
        ref = self._prepare(reference)
        self.normalizer.fit(ref)
        return self

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        was_single = images.dim() == 3
        if was_single:
            images = images.unsqueeze(0)
        images = images.to(self.device)
        if self._converter is not None:
            images = self._converter.prepare_for_normalizer(images)
        return images

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        was_single = img.dim() == 3
        batch = self._prepare(img)

        if self.mode == "batch":
            idx = self.batch_ref_index
            if idx < 0 or idx >= batch.shape[0]:
                raise IndexError(f"batch_ref_index={idx} out of range for batch size {batch.shape[0]}")
            self.normalizer.fit(batch[idx : idx + 1])

        result = self.normalizer.transform(batch)
        return result.squeeze(0) if was_single else result
