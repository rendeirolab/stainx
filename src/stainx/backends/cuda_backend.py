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

    def transform(self, images: torch.Tensor, reference_histogram: torch.Tensor | list) -> torch.Tensor:
        # Move tensors to CUDA device
        images = images.to(self.device)

        # Normalize to channels-first format for processing (matching PyTorch backend logic)
        # Match PyTorch backend's _normalize_to_channels_first method exactly
        # IMPORTANT: Do NOT fix corrupted formats - match PyTorch's behavior exactly
        # (even if PyTorch processes corrupted formats incorrectly)
        needs_permute = False
        if self.channel_axis == -1 or (self.channel_axis == 3 and images.ndim == 4):
            # Channels-last format (N, H, W, C) -> (N, C, H, W)
            # Trust channel_axis - assume input is in channels-last format
            # Note: This may produce wrong format if prepare_for_normalizer corrupted the input,
            # but we match PyTorch backend's behavior exactly
            images = images.permute(0, 3, 1, 2)
            needs_permute = True

        # Handle reference histogram (can be list or single tensor)
        # If list provided, stack into (C, 256) tensor for per-channel processing
        if isinstance(reference_histogram, list):
            # Stack per-channel histograms into (C, 256) tensor
            ref_hist_list = [h.to(self.device) for h in reference_histogram]
            # Ensure we have histograms for all channels (pad with first if needed)
            while len(ref_hist_list) < images.size(1):
                ref_hist_list.append(ref_hist_list[0])
            ref_hist = torch.stack(ref_hist_list[: images.size(1)], dim=0)  # (C, 256)
        else:
            # Single histogram for all channels: (256,)
            ref_hist = reference_histogram.to(self.device)

        # Check if CUDA function is available
        if not hasattr(stainx_cuda, "histogram_matching"):
            raise RuntimeError("stainx_cuda.histogram_matching is not available. The CUDA extension may not be built correctly.")

        # Call CUDA implementation (expects and returns (N, C, H, W) format)
        # CUDA function now accepts either (256,) or (C, 256) tensor
        result = stainx_cuda.histogram_matching(images, ref_hist)

        # Restore to original channel format if needed (matching PyTorch backend logic)
        if needs_permute:
            # Convert back to channels-last (N, C, H, W) -> (N, H, W, C)
            result = result.permute(0, 2, 3, 1)

        return result


class ReinhardCUDA(CUDABackendBase):
    def transform(self, images: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        target_mean = target_mean.to(self.device)
        target_std = target_std.to(self.device)

        # Check if CUDA function is available
        if not hasattr(stainx_cuda, "reinhard"):
            raise RuntimeError("stainx_cuda.reinhard is not available. The CUDA extension may not be built correctly.")

        # Call CUDA implementation
        return stainx_cuda.reinhard(images, target_mean, target_std)


class MacenkoCUDA(CUDABackendBase):
    def transform(self, images: torch.Tensor, stain_matrix: torch.Tensor, target_max_conc: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        stain_matrix = stain_matrix.to(self.device)
        target_max_conc = target_max_conc.to(self.device)

        # Check if CUDA function is available
        if not hasattr(stainx_cuda, "macenko"):
            raise RuntimeError("stainx_cuda.macenko is not available. The CUDA extension may not be built correctly.")

        # Call CUDA implementation
        return stainx_cuda.macenko(images, stain_matrix, target_max_conc)
