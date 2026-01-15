# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import cupy as cp
import pytest
import torch

# Check if CuPy CUDA is available and compatible
_CUPY_CUDA_AVAILABLE = cp.cuda.is_available()


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


@pytest.fixture
def sample_images_cupy():
    if not _CUPY_CUDA_AVAILABLE:
        pytest.skip("CuPy CUDA is not available")
    cp.random.seed(42)
    return (cp.random.rand(4, 3, 256, 256) * 255).round().astype(cp.uint8)


@pytest.fixture
def reference_images_cupy():
    if not _CUPY_CUDA_AVAILABLE:
        pytest.skip("CuPy CUDA is not available")
    cp.random.seed(43)
    return (cp.random.rand(2, 3, 256, 256) * 255).round().astype(cp.uint8)


@pytest.fixture
def single_image_cupy():
    if not _CUPY_CUDA_AVAILABLE:
        pytest.skip("CuPy CUDA is not available")
    cp.random.seed(44)
    return (cp.random.rand(3, 256, 256) * 255).round().astype(cp.uint8)


@pytest.fixture
def device_cupy():
    if not _CUPY_CUDA_AVAILABLE:
        pytest.skip("CuPy CUDA is not available")
    # Create device - if this fails due to insufficient driver, pytest will handle it
    return cp.cuda.Device(0)
