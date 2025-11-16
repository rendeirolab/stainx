# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import torch

try:
    import stainx_cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    stainx_cuda = None


class CUDABackendBase:
    def __init__(self, device: str | torch.device | None = None):
        if not CUDA_AVAILABLE:
            raise ImportError("stainx_cuda package is not installed or built. CUDA backend is not available. Use backend='pytorch' instead.")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("CUDA is not available on this system")
        else:
            self.device = torch.device(device)

        if self.device.type != "cuda":
            raise ValueError(f"CUDA backend requires CUDA device, got {self.device.type}")


class HistogramMatchingCUDA(CUDABackendBase):
    def __init__(self, device: str | torch.device | None = None, channel_axis: int = 1):
        super().__init__(device)
        self.channel_axis = channel_axis

    def transform(self, images: torch.Tensor, reference_histogram: torch.Tensor) -> torch.Tensor:
        # Move tensors to CUDA device
        images = images.to(self.device)
        reference_histogram = reference_histogram.to(self.device)

        # Check if CUDA function is available
        if not hasattr(stainx_cuda, "histogram_matching"):
            raise NotImplementedError("CUDA histogram matching not yet implemented. The stainx_cuda extension is not built or the function is not available.")

        # Call CUDA implementation
        # This will raise AT_ERROR("CUDA histogram matching not yet implemented")
        # from the C++ code if not implemented
        try:
            return stainx_cuda.histogram_matching(images, reference_histogram)
        except RuntimeError as e:
            # Catch the AT_ERROR from C++ code and re-raise with clearer message
            error_msg = str(e)
            if "not yet implemented" in error_msg.lower():
                raise NotImplementedError(f"CUDA histogram matching not yet implemented: {error_msg}") from e
            raise


class ReinhardCUDA(CUDABackendBase):
    def transform(self, images: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        target_mean = target_mean.to(self.device)
        target_std = target_std.to(self.device)

        # TODO: Implement when CUDA kernel is ready
        raise NotImplementedError("CUDA Reinhard normalization not yet implemented")


class MacenkoCUDA(CUDABackendBase):
    def transform(self, images: torch.Tensor, stain_matrix: torch.Tensor, concentration_map: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        stain_matrix = stain_matrix.to(self.device)
        if concentration_map is not None:
            concentration_map = concentration_map.to(self.device)

        # TODO: Implement when CUDA kernel is ready
        raise NotImplementedError("CUDA Macenko normalization not yet implemented")
