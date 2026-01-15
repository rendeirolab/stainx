# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from stainx.backends.cupy_backend import CupyBackendBase, HistogramMatchingCupy, MacenkoCupy, ReinhardCupy
from stainx.backends.torch_backend import HistogramMatchingTorch, MacenkoTorch, ReinhardTorch, TorchBackendBase
from stainx.backends.torch_cuda_backend import HistogramMatchingCUDA, MacenkoCUDA, ReinhardCUDA, TorchCUDABackendBase

__all__ = ["CupyBackendBase", "HistogramMatchingCUDA", "HistogramMatchingCupy", "HistogramMatchingTorch", "MacenkoCUDA", "MacenkoCupy", "MacenkoTorch", "ReinhardCUDA", "ReinhardCupy", "ReinhardTorch", "TorchBackendBase", "TorchCUDABackendBase"]
