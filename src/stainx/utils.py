# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Utility functions for stain normalization.

This module provides shared utility functions used across the stainx package.
"""

from typing import Any, ClassVar

import numpy as np
import torch


def get_device(device: str | torch.device | None) -> torch.device:
    """
    Get the appropriate torch device.

    Parameters
    ----------
    device : str or torch.device, optional
        Device specification. If None, auto-detects best available.

    Returns
    -------
    torch.device
        The appropriate torch device

    Notes
    -----
    Device priority: CUDA > MPS > CPU
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class ChannelFormatConverter:
    """
    Converter for handling different channel axis formats in image tensors/arrays.

    This class provides a unified interface for converting between channels-first (CHW)
    and channels-last (HWC) formats, supporting various channel_axis specifications.
    Works with both PyTorch tensors and NumPy arrays.

    Parameters
    ----------
    channel_axis : int, default=1
        The axis index for channels. Supported values:
        - 1 or -3: channels-first format (N, C, H, W)
        - -1 or 3: channels-last format (N, H, W, C)

    Attributes
    ----------
    channel_axis : int
        The channel axis index
    is_channels_first : bool
        True if channels-first format, False if channels-last
    permute_to_hwc : tuple or None
        Permutation tuple to convert to HWC format, or None if already HWC
    permute_to_chw : tuple or None
        Permutation tuple to convert to CHW format, or None if already CHW

    Examples
    --------
    >>> converter = ChannelFormatConverter(channel_axis=1)
    >>> images_chw = torch.rand(1, 3, 256, 256)
    >>> images_hwc = converter.to_hwc(images_chw)
    >>> images_chw_restored = converter.to_chw(images_hwc)
    >>>
    >>> # Works with NumPy arrays too
    >>> images_np = np.random.rand(1, 3, 256, 256)
    >>> images_hwc_np = converter.to_hwc(images_np)
    """

    # Mapping of channel_axis to format information
    _CHANNEL_AXIS_FORMAT: ClassVar[dict[int, dict[str, Any]]] = {
        1: {
            "is_channels_first": True,
            "permute_to_hwc": (1, 2, 0),
            "permute_to_chw": None,
        },
        -3: {
            "is_channels_first": True,
            "permute_to_hwc": (1, 2, 0),
            "permute_to_chw": None,
        },
        -1: {
            "is_channels_first": False,
            "permute_to_hwc": None,
            "permute_to_chw": (2, 0, 1),
        },
        3: {
            "is_channels_first": False,
            "permute_to_hwc": None,
            "permute_to_chw": (2, 0, 1),
        },
    }

    def __init__(self, channel_axis: int = 1):
        """
        Initialize the channel format converter.

        Parameters
        ----------
        channel_axis : int, default=1
            The axis index for channels
        """
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
        """Check if input is a PyTorch tensor."""
        return isinstance(x, torch.Tensor)

    @staticmethod
    def _is_numpy_array(x: Any) -> bool:
        """Check if input is a NumPy array."""
        return isinstance(x, np.ndarray)

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert input to NumPy array if it's a tensor."""
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.cpu().numpy()
        return x

    @staticmethod
    def _to_torch(
        x: torch.Tensor | np.ndarray, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Convert input to PyTorch tensor if it's a NumPy array."""
        if ChannelFormatConverter._is_numpy_array(x):
            tensor = torch.from_numpy(x)
            if dtype is not None:
                tensor = tensor.to(dtype)
            return tensor
        if dtype is not None:
            return x.to(dtype)
        return x

    @staticmethod
    def _squeeze(
        x: torch.Tensor | np.ndarray, dim: int | None = None
    ) -> torch.Tensor | np.ndarray:
        """Squeeze dimension(s) from tensor/array."""
        if dim is not None:
            if ChannelFormatConverter._is_torch_tensor(x):
                return x.squeeze(dim)
            return np.squeeze(x, axis=dim)
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.squeeze()
        return np.squeeze(x)

    @staticmethod
    def _transpose(
        x: torch.Tensor | np.ndarray, axes: tuple
    ) -> torch.Tensor | np.ndarray:
        """Transpose tensor/array using given axes/permutation."""
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.permute(*axes)
        return np.transpose(x, axes)

    @staticmethod
    def _cpu(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Move tensor to CPU (no-op for NumPy arrays)."""
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.cpu()
        return x

    @staticmethod
    def _float(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Convert to float dtype."""
        if ChannelFormatConverter._is_torch_tensor(x):
            return x.float()
        return x.astype(np.float32)

    def to_hwc(
        self, images: torch.Tensor | np.ndarray, squeeze_batch: bool = False
    ) -> np.ndarray:
        """
        Convert tensor/array to numpy array in HWC format (channels-last).

        This is useful for libraries like skimage that expect channels-last format.
        Works with both PyTorch tensors and NumPy arrays.

        Parameters
        ----------
        images : torch.Tensor or np.ndarray
            Input tensor/array in format (N, C, H, W) or (N, H, W, C)
        squeeze_batch : bool, default=False
            If True, removes the batch dimension (assumes N=1)

        Returns
        -------
        np.ndarray
            Array in HWC format (or (H, W, C) if squeeze_batch=True)
        """
        images_np = self._to_numpy(images)

        if squeeze_batch:
            images_np = np.squeeze(images_np, axis=0)

        if self.permute_to_hwc is not None:
            return np.transpose(images_np, self.permute_to_hwc)
        return images_np

    def prepare_for_normalizer(
        self, images: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """
        Prepare input tensor/array in the format expected by the normalizer.

        For channels-first format, returns the tensor/array as-is (on CPU for tensors).
        For channels-last format, converts to channels-first format.
        Works with both PyTorch tensors and NumPy arrays.

        Parameters
        ----------
        images : torch.Tensor or np.ndarray
            Input tensor/array in format (N, C, H, W) or (N, H, W, C)

        Returns
        -------
        torch.Tensor or np.ndarray
            Tensor/array prepared for normalizer (channels-first format, on CPU for tensors)
        """
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
        return np.expand_dims(images, axis=0)

    def to_chw(
        self,
        images: torch.Tensor | np.ndarray,
        squeeze_batch: bool = True,
        return_torch: bool = True,
    ) -> torch.Tensor | np.ndarray:
        """
        Convert result tensor/array to CHW format (channels-first) for comparison.

        Works with both PyTorch tensors and NumPy arrays.
        The result format depends on the channel_axis setting - if channel_axis indicates
        channels-last format, the normalizer returns (N, H, W, C), which we convert to (C, H, W).

        Parameters
        ----------
        images : torch.Tensor or np.ndarray
            Input tensor/array in format (N, C, H, W) or (N, H, W, C) depending on channel_axis
        squeeze_batch : bool, default=True
            If True, removes the batch dimension (assumes N=1)
        return_torch : bool, default=True
            If True, returns PyTorch tensor; if False, returns NumPy array

        Returns
        -------
        torch.Tensor or np.ndarray
            Tensor/array in CHW format (or (C, H, W) if squeeze_batch=True)
        """
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
        if (
            not self.is_channels_first
            and len(result.shape) == 3
            and result.shape[-1] in [1, 3, 4]
            and result.shape[0] not in [1, 3, 4]
        ):
            result = self._transpose(result, (2, 0, 1))

        # Ensure return type matches request
        if return_torch and self._is_numpy_array(result):
            return self._to_torch(result)
        if not return_torch and self._is_torch_tensor(result):
            return self._to_numpy(result)

        return result
