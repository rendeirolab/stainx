# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Stain normalization algorithms.

This module provides implementations of various stain normalization methods
for histopathology images.
"""

from .histogram_matching import HistogramMatching
from .macenko import Macenko
from .reinhard import Reinhard
from .vahadane import Vahadane

__all__ = [
    "HistogramMatching",
    "Reinhard", 
    "Macenko",
    "Vahadane",
]





