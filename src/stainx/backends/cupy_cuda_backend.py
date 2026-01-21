# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import cupy as cp

from stainx.backends.cupy_backend import HistogramMatchingCupy, MacenkoCupy, ReinhardCupy

# For the initial implementation, `backend="cupy_cuda"` is a distinct backend name that
# guarantees the array is on a CUDA device, but uses the same CuPy implementation path.
# This unblocks users/tests while keeping the door open for a future compiled CuPy CUDA
# extension (mirroring the Torch CUDA extension).
CUDA_AVAILABLE = cp.cuda.is_available()


class CupyCUDABackendBase:
    """Base class for CuPy-specific CUDA backend implementations.

    This class handles CuPy array operations and device management for CUDA backends.
    For Torch-based CUDA backends, use TorchCUDABackendBase instead.
    """

    def __init__(self, device: str | cp.cuda.Device | None = None):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available on this system. Use backend='cupy' or backend='torch'.")

        if device is None:
            if cp.cuda.is_available():
                self.device = cp.cuda.Device(0)
            else:
                raise RuntimeError("CUDA is not available on this system")
        else:
            if isinstance(device, cp.cuda.Device):
                self.device = device
            elif isinstance(device, str) and device.startswith("cuda"):
                device_id = int(device.split(":")[-1]) if ":" in device else 0
                self.device = cp.cuda.Device(device_id)
            else:
                raise ValueError(f"CUDA backend requires CUDA device, got {device}")

        if not cp.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")


class HistogramMatchingCuPyCUDA(HistogramMatchingCupy, CupyCUDABackendBase):
    """CuPy CUDA backend (initial implementation).

    Uses the existing CuPy implementation, but enforces that CUDA is available.
    """

    def __init__(self, device: str | cp.cuda.Device | None = None, channel_axis: int = 1):
        CupyCUDABackendBase.__init__(self, device=device)
        HistogramMatchingCupy.__init__(self, device=self.device, channel_axis=channel_axis)


class ReinhardCuPyCUDA(ReinhardCupy, CupyCUDABackendBase):
    """CuPy CUDA backend (initial implementation)."""

    def __init__(self, device: str | cp.cuda.Device | None = None):
        CupyCUDABackendBase.__init__(self, device=device)
        ReinhardCupy.__init__(self, device=self.device)


class MacenkoCuPyCUDA(MacenkoCupy, CupyCUDABackendBase):
    """CuPy CUDA backend (initial implementation)."""

    def __init__(self, device: str | cp.cuda.Device | None = None):
        CupyCUDABackendBase.__init__(self, device=device)
        MacenkoCupy.__init__(self, device=self.device)
