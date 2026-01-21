# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from stainx.backends.torch_backend import HistogramMatchingTorch, MacenkoTorch, ReinhardTorch, TorchBackendBase

# Note: cupy_backend, torch_cuda_backend, and cupy_cuda_backend are imported lazily
# when needed to avoid import errors when optional dependencies are not installed

try:
    from stainx.backends.cupy_cuda_backend import CupyCUDABackendBase, HistogramMatchingCuPyCUDA, MacenkoCuPyCUDA, ReinhardCuPyCUDA

    __all__ = ["CupyCUDABackendBase", "HistogramMatchingCuPyCUDA", "HistogramMatchingTorch", "MacenkoCuPyCUDA", "MacenkoTorch", "ReinhardCuPyCUDA", "ReinhardTorch", "TorchBackendBase"]
except ImportError:
    __all__ = ["HistogramMatchingTorch", "MacenkoTorch", "ReinhardTorch", "TorchBackendBase"]
