# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import pytest
import torch


@pytest.fixture
def sample_images():
    return (torch.rand(4, 3, 256, 256) * 255).round().to(torch.uint8)


@pytest.fixture
def reference_images():
    return (torch.rand(2, 3, 256, 256) * 255).round().to(torch.uint8)


@pytest.fixture
def single_image():
    return (torch.rand(3, 256, 256) * 255).round().to(torch.uint8)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path
