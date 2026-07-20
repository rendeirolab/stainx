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
    _CHANNEL_AXIS_FORMAT: ClassVar[dict[int, dict[str, Any]]] = {
        1: {"is_channels_first": True, "permute_to_hwc": (1, 2, 0), "permute_to_chw": None},
        -3: {"is_channels_first": True, "permute_to_hwc": (1, 2, 0), "permute_to_chw": None},
        -1: {"is_channels_first": False, "permute_to_hwc": None, "permute_to_chw": (2, 0, 1)},
        3: {"is_channels_first": False, "permute_to_hwc": None, "permute_to_chw": (2, 0, 1)},
    }

    def __init__(self, channel_axis: int = 1):
        if channel_axis not in self._CHANNEL_AXIS_FORMAT:
            channel_axis = 1

        self.channel_axis = channel_axis
        format_info = self._CHANNEL_AXIS_FORMAT[channel_axis]
        self.is_channels_first = format_info["is_channels_first"]
        self.permute_to_hwc = format_info["permute_to_hwc"]
        self.permute_to_chw = format_info["permute_to_chw"]

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
    def _to_torch(x: torch.Tensor | np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
        if ChannelFormatConverter._is_numpy_array(x):
            tensor = torch.from_numpy(x)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        if dtype is not None:
            return x.to(dtype)
        return x

    @staticmethod
    def _squeeze(x: torch.Tensor | np.ndarray | Any, dim: int | None = None) -> torch.Tensor | np.ndarray | Any:
        if dim is not None:
            if ChannelFormatConverter._is_torch_tensor(x):
                return x.squeeze(dim)
            return np.squeeze(x, axis=dim)
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.squeeze()
        return np.squeeze(x)

    @staticmethod
    def _transpose(x: torch.Tensor | np.ndarray | Any, axes: tuple) -> torch.Tensor | np.ndarray | Any:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.permute(*axes)
        return np.transpose(x, axes)

    @staticmethod
    def _cpu(x: torch.Tensor | np.ndarray | Any) -> torch.Tensor | np.ndarray | Any:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.cpu()
        return x

    @staticmethod
    def _float(x: torch.Tensor | np.ndarray | Any) -> torch.Tensor | np.ndarray | Any:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.float()
        return x.astype(np.float32)

    def to_hwc(self, images: torch.Tensor | np.ndarray | Any, squeeze_batch: bool = False) -> np.ndarray:
        images_np = self._to_numpy(images)

        if squeeze_batch:
            images_np = np.squeeze(images_np, axis=0)

        if self.permute_to_hwc is not None:
            return np.transpose(images_np, self.permute_to_hwc)
        return images_np

    def prepare_for_normalizer(self, images: torch.Tensor | np.ndarray | Any) -> torch.Tensor | np.ndarray | Any:
        if self.is_channels_first:
            return self._cpu(images)
        images = self._cpu(images)
        images = self._squeeze(images, dim=0)
        images = self._transpose(images, (2, 0, 1))
        if self._is_torch_tensor(images):
            return images.unsqueeze(0)
        return np.expand_dims(images, axis=0)

    def to_chw(self, images: torch.Tensor | np.ndarray | Any, squeeze_batch: bool = True, return_torch: bool = True) -> torch.Tensor | np.ndarray | Any:
        result = self._cpu(images)
        original_ndim = len(result.shape)

        if not self.is_channels_first:
            if original_ndim == 4:
                result = self._transpose(result, (0, 3, 1, 2))
            elif original_ndim == 3:
                result = self._transpose(result, (2, 0, 1))

        if squeeze_batch and len(result.shape) == 4:
            result = self._squeeze(result, dim=0)

        result = self._float(result)

        if not self.is_channels_first and len(result.shape) == 3 and result.shape[-1] in [1, 3, 4] and result.shape[0] not in [1, 3, 4]:
            result = self._transpose(result, (2, 0, 1))

        if return_torch and self._is_numpy_array(result):
            return self._to_torch(result)
        if not return_torch and self._is_torch_tensor(result):
            return self._to_numpy(result)

        return result
