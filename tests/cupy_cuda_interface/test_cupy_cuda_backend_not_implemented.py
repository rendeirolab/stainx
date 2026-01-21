"""Tests for the (future) **CUDA implementation with a CuPy interface**.

Context
-------
This project aims to provide 4 execution paths:

1) PyTorch implementation (Torch ops)                           -> `backend="torch"`
2) CUDA implementation with PyTorch interface (custom CUDA ext) -> `backend="cuda"`
3) CuPy implementation (CuPy ops)                               -> `backend="cupy"`
4) CUDA implementation with CuPy interface (custom CUDA ext)    -> (future) e.g. `backend="cupy_cuda"`

At the moment, (4) is **not implemented yet** (see `CONTRIBUTING.md`, "For CuPy CUDA (future)").
These tests are intentionally marked as **expected failures** so they:

- document the intended behavior and naming,
- don't silently pass by exercising the existing CuPy implementation,
- don't break CI until the backend exists.
"""

import cupy as cp
import pytest


@pytest.mark.cuda
class TestCupyCudaBackendNotImplemented:
    """Placeholder tests for `backend="cupy_cuda"` (not implemented)."""

    @pytest.fixture
    def cuda_device(self):
        if not cp.cuda.is_available():
            pytest.skip("CUDA is not available")
        return cp.cuda.Device(0)

    @pytest.mark.xfail(reason="CUDA backend with CuPy interface is not implemented yet (planned: src/stainx/backends/cupy_cuda_backend.py).", strict=True)
    def test_reinhard_backend_cupy_cuda_not_implemented(self, cuda_device):  # noqa: ARG002
        pytest.fail("Not implemented: expected to support `backend='cupy_cuda'` once CuPy CUDA backend exists.")

    @pytest.mark.xfail(reason="CUDA backend with CuPy interface is not implemented yet (planned: src/stainx/backends/cupy_cuda_backend.py).", strict=True)
    def test_macenko_backend_cupy_cuda_not_implemented(self, cuda_device):  # noqa: ARG002
        pytest.fail("Not implemented: expected to support `backend='cupy_cuda'` once CuPy CUDA backend exists.")

    @pytest.mark.xfail(reason="CUDA backend with CuPy interface is not implemented yet (planned: src/stainx/backends/cupy_cuda_backend.py).", strict=True)
    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_backend_cupy_cuda_not_implemented(self, cuda_device, channel_axis):  # noqa: ARG002
        pytest.fail("Not implemented: expected to support `backend='cupy_cuda'` once CuPy CUDA backend exists.")


if __name__ == "__main__":
    pytest.main()
