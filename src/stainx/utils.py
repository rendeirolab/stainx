# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from typing import Any, ClassVar

import cupy as cp
import numpy as np
import torch


def _get_torch_device(device_str: str) -> Any | None:
    """Get Torch device if available."""
    return torch.device(device_str)


def _get_cupy_device(device_str: str) -> Any | None:
    """Get CuPy device if available."""
    if device_str != "cuda":
        return None
    if cp.cuda.is_available():
        return cp.cuda.Device(0)
    return None


def _get_default_device() -> Any:
    """Get default device from available backends."""
    # Priority: CUDA (Torch) > MPS (Torch) > CUDA (CuPy) > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # Try CuPy CUDA
    device = _get_cupy_device("cuda")
    if device is not None:
        return device

    # Fallback to CPU
    return _get_torch_device("cpu") or "cpu"


def get_device(device: str | Any | None) -> Any:
    """Get device object from any backend.

    This function is backend-agnostic and supports:
    - Torch devices (if Torch is available)
    - CuPy devices (if CuPy is available)
    - String device specifications ("cpu", "cuda", etc.)

    Args:
        device: Device specification (string or device-like object).

    Returns:
        Device object from the available backend, or string if no backend available.
    """
    # Return default device if None
    if device is None:
        return _get_default_device()

    # Return device object as-is if already a device object
    if not isinstance(device, str):
        return device

    # Try to create device from string
    # Priority: Torch > CuPy > string fallback
    device_obj = _get_torch_device(device)
    if device_obj is not None:
        return device_obj

    device_obj = _get_cupy_device(device)
    if device_obj is not None:
        return device_obj

    # Return string if no backend can handle it
    return device


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
            # Default to channels-first if unknown axis
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
    def _is_cupy_array(x: Any) -> bool:
        return isinstance(x, cp.ndarray)

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray | cp.ndarray) -> np.ndarray:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.cpu().numpy()
        if ChannelFormatConverter._is_cupy_array(x):
            return cp.asnumpy(x)
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
    def _squeeze(x: torch.Tensor | np.ndarray | cp.ndarray, dim: int | None = None) -> torch.Tensor | np.ndarray | cp.ndarray:
        if dim is not None:
            if ChannelFormatConverter._is_torch_tensor(x):
                return x.squeeze(dim)
            if ChannelFormatConverter._is_cupy_array(x):
                return cp.squeeze(x, axis=dim)
            return np.squeeze(x, axis=dim)
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.squeeze()
        if ChannelFormatConverter._is_cupy_array(x):
            return cp.squeeze(x)
        return np.squeeze(x)

    @staticmethod
    def _transpose(x: torch.Tensor | np.ndarray | cp.ndarray, axes: tuple) -> torch.Tensor | np.ndarray | cp.ndarray:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.permute(*axes)
        if ChannelFormatConverter._is_cupy_array(x):
            return cp.transpose(x, axes)
        return np.transpose(x, axes)

    @staticmethod
    def _cpu(x: torch.Tensor | np.ndarray | cp.ndarray) -> torch.Tensor | np.ndarray | cp.ndarray:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.cpu()
        # For CuPy arrays, return as-is (they're already on GPU or can be moved)
        return x

    @staticmethod
    def _float(x: torch.Tensor | np.ndarray | cp.ndarray) -> torch.Tensor | np.ndarray | cp.ndarray:
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.float()
        if ChannelFormatConverter._is_cupy_array(x):
            return x.astype(cp.float32)
        return x.astype(np.float32)

    def to_hwc(self, images: torch.Tensor | np.ndarray | cp.ndarray, squeeze_batch: bool = False) -> np.ndarray:
        images_np = self._to_numpy(images)

        if squeeze_batch:
            images_np = np.squeeze(images_np, axis=0)

        if self.permute_to_hwc is not None:
            return np.transpose(images_np, self.permute_to_hwc)
        return images_np

    def prepare_for_normalizer(self, images: torch.Tensor | np.ndarray | cp.ndarray) -> torch.Tensor | np.ndarray | cp.ndarray:
        if self.is_channels_first:
            # channels-first: return as-is on CPU
            return self._cpu(images)
        # channels-last: convert to channels-first
        # (N, H, W, C) -> (N, C, H, W)
        images = self._cpu(images)
        images = self._squeeze(images, dim=0)
        images = self._transpose(images, (2, 0, 1))
        # Add batch dimension back
        if self._is_torch_tensor(images):
            return images.unsqueeze(0)
        if self._is_cupy_array(images):
            return cp.expand_dims(images, axis=0)
        return np.expand_dims(images, axis=0)

    def to_chw(self, images: torch.Tensor | np.ndarray | cp.ndarray, squeeze_batch: bool = True, return_torch: bool = True) -> torch.Tensor | np.ndarray | cp.ndarray:
        result = self._cpu(images)
        original_ndim = len(result.shape)

        # When channel_axis indicates channels-last, the normalizer returns (N, H, W, C)
        # When channel_axis indicates channels-first, the normalizer returns (N, C, H, W)
        # We need to convert both to (C, H, W) for comparison

        # Handle channels-last format: convert to channels-first
        if not self.is_channels_first:
            if original_ndim == 4:
                # (N, H, W, C) -> (N, C, H, W)
                result = self._transpose(result, (0, 3, 1, 2))
            elif original_ndim == 3:
                # 3D result is (H, W, C) -> convert to (C, H, W)
                result = self._transpose(result, (2, 0, 1))

        # Squeeze batch dimension if needed
        if squeeze_batch and len(result.shape) == 4:
            result = self._squeeze(result, dim=0)  # (N, C, H, W) -> (C, H, W)

        result = self._float(result)

        # Final safeguard: ensure result is in CHW format
        # This catches cases where conversion might not have happened
        if not self.is_channels_first and len(result.shape) == 3 and result.shape[-1] in [1, 3, 4] and result.shape[0] not in [1, 3, 4]:
            result = self._transpose(result, (2, 0, 1))

        # Ensure return type matches request
        if return_torch and (self._is_numpy_array(result) or self._is_cupy_array(result)):
            return self._to_torch(self._to_numpy(result))
        if not return_torch and self._is_torch_tensor(result):
            return self._to_numpy(result)
        # If return_torch=False and result is CuPy array, keep as CuPy array
        if not return_torch and self._is_cupy_array(result):
            return result

        return result
