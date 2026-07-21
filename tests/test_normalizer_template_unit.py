# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import pytest
import torch

from stainx.normalizers.reinhard import Reinhard


def test_transform_requires_fit():
    n = Reinhard(backend="torch", device="cpu")
    x = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="fit"):
        _ = n.transform(x)


def test_fit_transform_runs_and_preserves_shape():
    n = Reinhard(backend="torch", device="cpu")
    x = (torch.rand(1, 3, 8, 8) * 255).round().to(torch.uint8)
    y = n.fit_transform(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
