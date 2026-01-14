# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from stainx.backends.torch_backend import HistogramMatchingPyTorch, MacenkoPyTorch, PyTorchBackendBase, ReinhardPyTorch
from stainx.backends.torch_cuda_backend import HistogramMatchingCUDA, MacenkoCUDA, PyTorchCUDABackendBase, ReinhardCUDA

__all__ = ["HistogramMatchingCUDA", "HistogramMatchingPyTorch", "MacenkoCUDA", "MacenkoPyTorch", "PyTorchBackendBase", "PyTorchCUDABackendBase", "ReinhardCUDA", "ReinhardPyTorch"]
