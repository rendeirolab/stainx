# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import torch

try:
    import stainx_cuda_torch

    CUDA_AVAILABLE = getattr(stainx_cuda_torch, "FUNCTIONS_AVAILABLE", False)
except Exception:
    stainx_cuda_torch = None  # type: ignore[assignment]
    CUDA_AVAILABLE = False


class TorchCUDABackendBase:
    """Base class for Torch CUDA extension backend implementations."""

    def __init__(self, device: str | torch.device | None = None):
        if not CUDA_AVAILABLE:
            raise ImportError("stainx_cuda_torch package is not installed or built. CUDA backend is not available. Use backend='torch' instead.")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("CUDA is not available on this system")
        else:
            self.device = torch.device(device)

        if self.device.type != "cuda":
            raise ValueError(f"CUDA backend requires CUDA device, got {self.device.type}")


class HistogramMatchingCUDA(TorchCUDABackendBase):
    def __init__(self, device: str | torch.device | None = None, channel_axis: int = 1):
        super().__init__(device)
        self.channel_axis = channel_axis

    def transform(self, images: torch.Tensor, reference_histogram: torch.Tensor | list) -> torch.Tensor:
        # Move tensors to CUDA device
        images = images.to(self.device)

        # Normalize to channels-first format for processing (matching Torch backend logic)
        needs_permute = False
        if self.channel_axis == -1 or (self.channel_axis == 3 and images.ndim == 4):
            images = images.permute(0, 3, 1, 2)
            needs_permute = True

        if isinstance(reference_histogram, list):
            if len(reference_histogram) == 0:
                raise ValueError("reference_histogram list cannot be empty")

            for i, h in enumerate(reference_histogram):
                if not isinstance(h, torch.Tensor):
                    raise TypeError(f"reference_histogram[{i}] must be a torch.Tensor, got {type(h)}")
                if h.dim() != 1 or h.size(0) != 256:
                    raise ValueError(f"Each histogram in reference_histogram list must be 1D with 256 elements. Got histogram at index {i} with shape {h.shape}")

            ref_hist = torch.stack(reference_histogram, dim=0)

            num_channels = images.size(1)
            if ref_hist.size(0) < num_channels:
                padding = ref_hist[0:1].repeat(num_channels - ref_hist.size(0), 1)
                ref_hist = torch.cat([ref_hist, padding], dim=0)
            elif ref_hist.size(0) > num_channels:
                ref_hist = ref_hist[:num_channels]

            ref_hist = ref_hist.to(self.device)
        else:
            ref_hist = reference_histogram.to(self.device)
            if ref_hist.dim() != 1 or ref_hist.size(0) != 256:
                raise ValueError(f"reference_histogram must be 1D with 256 elements. Got shape {ref_hist.shape}")

        if not hasattr(stainx_cuda_torch, "histogram_matching"):
            raise RuntimeError("stainx_cuda_torch.histogram_matching is not available. The CUDA extension may not be built correctly.")

        result = stainx_cuda_torch.histogram_matching(images, ref_hist)

        if needs_permute:
            result = result.permute(0, 2, 3, 1)

        return result


class ReinhardCUDA(TorchCUDABackendBase):
    def transform(self, images: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        target_mean = target_mean.to(self.device)
        target_std = target_std.to(self.device)

        if not hasattr(stainx_cuda_torch, "reinhard"):
            raise RuntimeError("stainx_cuda_torch.reinhard is not available. The CUDA extension may not be built correctly.")

        return stainx_cuda_torch.reinhard(images, target_mean, target_std)


class MacenkoCUDA(TorchCUDABackendBase):
    """Macenko CUDA backend.

    Parameters
    ----------
    device : str or torch.device, optional
        CUDA device to use.
    precision : str, optional
        Numerical precision mode. ``"stable"`` (default) uses the current
        fp64-covariance + fp32-pixel path. ``"fast"`` uses fp32 cov/eigh
        + fp16 for large pixel tensors (projection GEMM, phi sort,
        reconstruct matmul) while keeping the 2x2 solve at fp32.
        Both modes pass the same correctness suite vs torchstain.
    """

    def __init__(self, device: str | torch.device | None = None, precision: str = "stable"):
        super().__init__(device)
        if precision not in ("stable", "fast"):
            raise ValueError(f"precision must be 'stable' or 'fast', got {precision!r}")
        self._precision = precision

    def transform(self, images: torch.Tensor, stain_matrix: torch.Tensor, target_max_conc: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        stain_matrix = stain_matrix.to(self.device)
        target_max_conc = target_max_conc.to(self.device)

        if self._precision == "fast":
            if not hasattr(stainx_cuda_torch, "macenko_fast"):
                raise RuntimeError("stainx_cuda_torch.macenko_fast is not available. The CUDA extension may not be built correctly.")
            return stainx_cuda_torch.macenko_fast(images, stain_matrix, target_max_conc)
        if not hasattr(stainx_cuda_torch, "macenko"):
            raise RuntimeError("stainx_cuda_torch.macenko is not available. The CUDA extension may not be built correctly.")
        return stainx_cuda_torch.macenko(images, stain_matrix, target_max_conc)
