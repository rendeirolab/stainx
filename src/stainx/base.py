# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from stainx.utils import get_device


class StainNormalizerBase(ABC, nn.Module):
    def __init__(self, device: str | torch.device | None = None):
        super().__init__()
        self.device = get_device(device)
        self._is_fitted = False

    @abstractmethod
    def fit(self, images: torch.Tensor) -> "StainNormalizerBase":
        pass

    @abstractmethod
    def transform(self, images: torch.Tensor) -> torch.Tensor:
        pass

    def fit_transform(self, images: torch.Tensor) -> torch.Tensor:
        self.fit(images)
        return self.transform(images)
