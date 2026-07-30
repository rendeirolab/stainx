# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure `src/` is importable regardless of test nesting depth
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@pytest.fixture
def sample_images_torch():
    return (torch.rand(4, 3, 256, 256) * 255).round().to(torch.uint8)


@pytest.fixture
def reference_images_torch():
    return (torch.rand(2, 3, 256, 256) * 255).round().to(torch.uint8)


@pytest.fixture
def single_image_torch():
    return (torch.rand(3, 256, 256) * 255).round().to(torch.uint8)


@pytest.fixture
def device_torch():
    if torch.cuda.is_available():
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
