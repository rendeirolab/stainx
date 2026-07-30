# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from typing import Any, ClassVar

import numpy as np
import torch


def _get_default_device() -> torch.device:
    """Get default Torch device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device(device: str | Any | None) -> Any:
    """Resolve a Torch device from a string or device-like object.

    Args:
        device: Device specification (``"cpu"``, ``"cuda"``, ``"mps"``, or a ``torch.device``).

    Returns:
        A ``torch.device`` (or the original non-string object if already a device).
    """
    if device is None:
        return _get_default_device()
    if not isinstance(device, str):
        return device
    return torch.device(device)


class ChannelFormatConverter:
    # Mapping of channel_axis to format information
    _CHANNEL_AXIS_FORMAT: ClassVar[dict[int, dict[str, Any]]] = {1: {"is_channels_first": True, "permute_to_hwc": (1, 2, 0)}, -3: {"is_channels_first": True, "permute_to_hwc": (1, 2, 0)}, -1: {"is_channels_first": False, "permute_to_hwc": None}, 3: {"is_channels_first": False, "permute_to_hwc": None}}

    def __init__(self, channel_axis: int = 1):
        if channel_axis not in self._CHANNEL_AXIS_FORMAT:
            raise ValueError(f"Unsupported channel_axis={channel_axis}. Valid values: {sorted(self._CHANNEL_AXIS_FORMAT)}")

        self.channel_axis = channel_axis
        format_info = self._CHANNEL_AXIS_FORMAT[channel_axis]
        self.is_channels_first = format_info["is_channels_first"]
        self.permute_to_hwc = format_info["permute_to_hwc"]

    @staticmethod
    def _is_torch_tensor(x: Any) -> bool:
        return isinstance(x, torch.Tensor)

    @staticmethod
    def _is_numpy_array(x: Any) -> bool:
        return isinstance(x, np.ndarray)

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray | Any) -> np.ndarray:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.detach().cpu().numpy()
        return x

    @staticmethod
    def _transpose(x: torch.Tensor | np.ndarray | Any, axes: tuple) -> torch.Tensor | np.ndarray | Any:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.permute(*axes)
        return np.transpose(x, axes)

    def to_hwc(self, images: torch.Tensor | np.ndarray | Any, squeeze_batch: bool = False) -> np.ndarray:
        images_np = self._to_numpy(images)

        if squeeze_batch:
            images_np = np.squeeze(images_np, axis=0)

        if self.permute_to_hwc is not None:
            return np.transpose(images_np, self.permute_to_hwc)
        return images_np

    def prepare_for_normalizer(self, images: torch.Tensor | np.ndarray | Any) -> torch.Tensor | np.ndarray | Any:
        """Return tensors in channels-first layout for backends that expect NCHW.

        Channels-first inputs are returned unchanged (same device). Channels-last
        ``NHWC`` / ``HWC`` inputs are converted with ``(0, 3, 1, 2)`` / ``(2, 0, 1)``.
        After converting channels-last data, use ``channel_axis=1`` on the normalizer.
        """
        if self.is_channels_first:
            return images

        ndim = images.ndim if hasattr(images, "ndim") else len(images.shape)
        if ndim == 4:
            # (N, H, W, C) → (N, C, H, W) — do not squeeze the batch axis
            return self._transpose(images, (0, 3, 1, 2))
        if ndim == 3:
            # (H, W, C) → (1, C, H, W)
            images = self._transpose(images, (2, 0, 1))
            if self._is_torch_tensor(images):
                return images.unsqueeze(0)
            return np.expand_dims(images, axis=0)
        raise ValueError(f"prepare_for_normalizer expects 3D or 4D images, got ndim={ndim}")
