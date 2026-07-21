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

MethodName = Literal["macenko", "reinhard", "histogram_matching"]
ModeName = Literal["reference", "batch"]

_METHOD_MAP = {"macenko": Macenko, "reinhard": Reinhard, "histogram_matching": HistogramMatching}
_CHANNELS_FIRST = frozenset({1, -3})
_CHANNELS_LAST = frozenset({-1, 3})
_FIT_TENSOR_ATTRS = ("_stain_matrix", "_target_max_conc", "_concentration_matrix", "_reference_histogram", "_ref_vals", "_ref_cdf", "_ref_histograms_256", "_reference_mean", "_reference_std")


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
        Do not use under ``DataLoader`` workers unless that mutability is intended.

    Layout
    ------
    Macenko / Reinhard expect **NCHW** with ``C=3`` (or CHW for a single image).
    ``channel_axis`` is only valid for histogram matching. Passing NHWC into
    Macenko/Reinhard raises — it would otherwise treat ``H`` as channels.

    Value range
    -----------
    ``uint8`` inputs are ``[0, 255]``; **float inputs are always treated as**
    ``[0, 1]`` (no ``max()>1`` heuristic — ColorJitter can exceed 1). For Macenko,
    ``normalize_to_0_1`` defaults to ``True`` so float pipelines stay compatible
    with ImageNet ``Normalize``. Pass ``normalize_to_0_1=False`` only for
    uint8 / ``[0, 255]`` Macenko output.

    Device
    ------
    Default ``device=None`` keeps tensors on the **input** device and syncs the
    inner normalizer / backend selection to that device on first fit/forward
    (so CUDA inputs can auto-select ``torch_cuda``). Pass ``device="cuda"`` to
    always move batches. ``backend="torch_cuda"`` with ``device=None`` requires
    CUDA inputs (or an explicit CUDA ``device``).

    Serialization
    -------------
    Fitted stain parameters live on the inner normalizer, not as registered
    buffers. ``state_dict()`` / DDP / checkpoints do **not** persist HE / maxC /
    histograms — call ``fit_reference`` again after loading weights.
    """

    def __init__(self, method: MethodName = "macenko", *, mode: ModeName = "reference", reference: torch.Tensor | None = None, device: str | torch.device | None = None, backend: str | None = None, channel_axis: int = 1, batch_ref_index: int = 0, normalize_to_0_1: bool | None = None, normalizer: Any | None = None):
        """
        Args:
            method: Algorithm name (``"macenko"``, ``"reinhard"``, ``"histogram_matching"``).
            mode: ``"reference"`` or ``"batch"``.
            reference: Required for ``mode="reference"`` unless ``normalizer`` is already fitted.
            device: If set, batches are moved here. If ``None``, keep the input tensor device.
            backend: ``"torch"`` or ``"torch_cuda"`` (auto-selected if None).
            channel_axis: Histogram matching only (``1``/``-3`` NCHW, ``-1``/``3`` NHWC).
            batch_ref_index: Index within the batch used as reference when ``mode="batch"``.
            normalize_to_0_1: Macenko output in ``[0, 1]``. Defaults to ``True`` for
                ``method="macenko"``; ignored for other methods. Always synced onto a
                prebuilt Macenko when set explicitly.
            normalizer: Optional pre-built normalizer (skips construction from ``method``).
        """
        super().__init__()
        self.mode = mode
        self.channel_axis = channel_axis
        self.batch_ref_index = batch_ref_index
        # None = follow input device each call (do not auto-pick CUDA at construct time).
        self.device = None if device is None else torch.device(device)
        self._requested_backend = backend

        if mode not in ("reference", "batch"):
            raise ValueError(f"Unsupported mode '{mode}'. Use 'reference' or 'batch'.")

        if backend == "torch_cuda" and self.device is not None and self.device.type != "cuda":
            raise ValueError(f"backend='torch_cuda' requires a CUDA device, got {self.device}.")

        # Training-safe default for method="macenko"; leave prebuilt Macenko alone unless set.
        explicit_n01 = normalize_to_0_1
        if normalize_to_0_1 is None:
            normalize_to_0_1 = method == "macenko" and normalizer is None

        if normalizer is not None:
            self.normalizer = normalizer
            if isinstance(self.normalizer, Macenko):
                if explicit_n01 is not None:
                    self.normalizer.normalize_to_0_1 = bool(explicit_n01)
            elif explicit_n01:
                raise ValueError("normalize_to_0_1 only applies to Macenko normalizers.")
            if not isinstance(self.normalizer, HistogramMatching) and channel_axis not in _CHANNELS_FIRST:
                raise ValueError(f"channel_axis={channel_axis} is only supported for histogram_matching; Macenko/Reinhard require NCHW (channel_axis=1).")
        else:
            if method not in _METHOD_MAP:
                raise ValueError(f"Unknown method '{method}'. Choose from {sorted(_METHOD_MAP)}")
            if method != "histogram_matching" and channel_axis not in _CHANNELS_FIRST:
                raise ValueError(f"channel_axis={channel_axis} is only supported for histogram_matching; {method} requires NCHW (channel_axis=1).")
            if explicit_n01 and method != "macenko":
                raise ValueError("normalize_to_0_1 only applies to Macenko (method='macenko').")
            cls = _METHOD_MAP[method]
            norm_device = self._initial_norm_device(backend)
            if method == "histogram_matching":
                self.normalizer = cls(device=norm_device, backend=backend, channel_axis=channel_axis)
            elif method == "macenko":
                self.normalizer = cls(device=norm_device, backend=backend, normalize_to_0_1=bool(normalize_to_0_1))
            else:
                self.normalizer = cls(device=norm_device, backend=backend)

        if mode == "reference":
            if reference is None and not getattr(self.normalizer, "_is_fitted", False):
                raise ValueError("mode='reference' requires a reference tensor (or a pre-fitted normalizer).")
            if reference is not None:
                self.fit_reference(reference)

    def _initial_norm_device(self, backend: str | None) -> torch.device | str:
        if self.device is not None:
            return self.device
        if backend == "torch_cuda":
            if not torch.cuda.is_available():
                raise ValueError("backend='torch_cuda' requires a CUDA device; pass device='cuda' or use CUDA input tensors with device=None.")
            return torch.device("cuda")
        # Placeholder until the first tensor reveals the input device.
        return "cpu"

    def fit_reference(self, reference: torch.Tensor) -> StainNormalizerTransform:
        """Fit the underlying normalizer on a reference image or batch."""
        ref = self._prepare(reference)
        self.normalizer.fit(ref)
        return self

    def _target_device(self, images: torch.Tensor) -> torch.device:
        return self.device if self.device is not None else images.device

    def _sync_normalizer_device(self, device: torch.device) -> None:
        """Keep inner normalizer + backend selection on the batch device."""
        device = torch.device(device)
        current = torch.device(self.normalizer.device)
        if self._requested_backend == "torch_cuda" and device.type != "cuda":
            raise ValueError(f"backend='torch_cuda' requires CUDA tensors when device=None; got {device}.")

        if current == device:
            # Still refresh auto backend if a placeholder CPU construct never saw CUDA.
            if self._requested_backend is None and device.type == "cuda" and getattr(self.normalizer, "backend", None) == "torch":
                self.normalizer.backend = self.normalizer._select_backend()
                self.normalizer._backend_impl = None
            return

        self.normalizer.device = device
        self.normalizer._backend_impl = None
        if self._requested_backend is None and hasattr(self.normalizer, "_select_backend"):
            self.normalizer.backend = self.normalizer._select_backend()

        for name in _FIT_TENSOR_ATTRS:
            value = getattr(self.normalizer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self.normalizer, name, value.to(device))
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, torch.Tensor) for v in value):
                moved = [v.to(device) for v in value]
                setattr(self.normalizer, name, type(value)(moved))

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError(f"Expected CHW/NCHW or HWC/NHWC image tensor, got shape {tuple(images.shape)}")

        if isinstance(self.normalizer, HistogramMatching) and self.channel_axis in _CHANNELS_LAST:
            if images.shape[-1] != 3:
                raise ValueError(f"channels-last histogram matching expects shape (N, H, W, 3), got {tuple(images.shape)}")
        else:
            # Macenko / Reinhard / HM-NCHW: channels must be dim 1.
            if images.shape[1] != 3:
                raise ValueError(f"Expected NCHW with C=3 (got shape {tuple(images.shape)}). Macenko/Reinhard do not accept NHWC; use channel_axis=-1 only with histogram_matching, or permute to NCHW first.")

        target = self._target_device(images)
        self._sync_normalizer_device(target)
        return images.to(target)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        was_single = img.dim() == 3
        batch = self._prepare(img)

        if self.mode == "batch":
            # Intentional: re-fits every forward (mutates state; not reproducible across steps).
            idx = self.batch_ref_index
            if idx < 0 or idx >= batch.shape[0]:
                raise IndexError(f"batch_ref_index={idx} out of range for batch size {batch.shape[0]}")
            self.normalizer.fit(batch[idx : idx + 1])

        result = self.normalizer.transform(batch)
        return result.squeeze(0) if was_single else result
