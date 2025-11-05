# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
Utility functions for stain normalization.

This module provides shared utility functions used across the stainx package.
"""

from typing import Optional, Union

import torch


def get_device(device: Optional[Union[str, torch.device]]) -> torch.device:
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
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)



