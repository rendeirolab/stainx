# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

from __future__ import annotations

import sys
from pathlib import Path

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None  # type: ignore[assignment]
import pytest
import torch

# Note: We import cupy directly (assume it's installed like torch)
# We don't check cp.cuda.is_available() at module level because it may raise
# CUDARuntimeError if the driver is insufficient. We check it only in fixtures
# and in pytest hooks where pytest can handle exceptions properly.


# Ensure `src/` is importable regardless of test nesting depth
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# Pytest hook to skip CuPy(-interface) tests if CUDA is not available
# We check PyTorch CUDA to avoid calling `cp.cuda.is_available()` which may raise
def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip CuPy-interface tests when CuPy/CUDA is unavailable."""
    cupy_items = [item for item in items if "cupy" in item.nodeid.lower()]
    if not cupy_items:
        return

    if cp is None:
        for item in cupy_items:
            item.add_marker(pytest.mark.skip(reason="CuPy is not installed"))
        return

    if not torch.cuda.is_available():
        for item in cupy_items:
            item.add_marker(pytest.mark.skip(reason="CUDA is not available"))


# Pytest hook to skip CuPy tests if CUDA is not available
# This hook runs before each test, so we can check availability here
def pytest_runtest_setup(item):
    """Skip CuPy tests if CUDA is not available."""
    # Only process CuPy tests
    if "cupy" not in item.nodeid.lower():
        return

    if cp is None:
        pytest.skip("CuPy is not installed")

    # Check availability - this may raise CUDARuntimeError if driver is insufficient
    # If it raises, pytest will catch it during test setup
    if not cp.cuda.is_available():
        pytest.skip("CuPy CUDA is not available")


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
    # Check availability - may raise CUDARuntimeError if driver is insufficient
    # pytest will handle the exception and skip the test
    if cp is None:
        pytest.skip("CuPy is not installed")
    if not cp.cuda.is_available():
        pytest.skip("CuPy CUDA is not available")
    cp.random.seed(42)
    return (cp.random.rand(4, 3, 256, 256) * 255).round().astype(cp.uint8)


@pytest.fixture
def reference_images_cupy():
    # Check availability - may raise CUDARuntimeError if driver is insufficient
    # pytest will handle the exception and skip the test
    if cp is None:
        pytest.skip("CuPy is not installed")
    if not cp.cuda.is_available():
        pytest.skip("CuPy CUDA is not available")
    cp.random.seed(43)
    return (cp.random.rand(2, 3, 256, 256) * 255).round().astype(cp.uint8)


@pytest.fixture
def single_image_cupy():
    # Check availability - may raise CUDARuntimeError if driver is insufficient
    # pytest will handle the exception and skip the test
    if cp is None:
        pytest.skip("CuPy is not installed")
    if not cp.cuda.is_available():
        pytest.skip("CuPy CUDA is not available")
    cp.random.seed(44)
    return (cp.random.rand(3, 256, 256) * 255).round().astype(cp.uint8)


@pytest.fixture
def device_cupy():
    # Check availability - may raise CUDARuntimeError if driver is insufficient
    # pytest will handle the exception and skip the test
    if cp is None:
        pytest.skip("CuPy is not installed")
    if not cp.cuda.is_available():
        pytest.skip("CuPy CUDA is not available")
    # Create device - if this fails due to insufficient driver, pytest will handle it
    return cp.cuda.Device(0)
