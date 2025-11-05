# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Backend implementations for stain normalization.

This module provides access to different backend implementations
(CUDA, PyTorch) for stain normalization algorithms.
"""

from .torch_backend import (
    HistogramMatchingPyTorch,
    MacenkoPyTorch,
    PyTorchBackendBase,
    ReinhardPyTorch,
    VahadanePyTorch,
)

__all__ = [
    "PyTorchBackendBase",
    "HistogramMatchingPyTorch",
    "ReinhardPyTorch",
    "MacenkoPyTorch",
    "VahadanePyTorch",
]

