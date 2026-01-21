# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import torch

import stainx_cuda_torch

# Check if functions are actually available, not just if package imports
CUDA_AVAILABLE = getattr(stainx_cuda_torch, "FUNCTIONS_AVAILABLE", False)
if not CUDA_AVAILABLE:
    print("DEBUG: stainx_cuda_torch imported but FUNCTIONS_AVAILABLE=False - CUDA backend not available")


class TorchCUDABackendBase:
    """Base class for Torch-specific CUDA backend implementations.

    This class handles Torch tensor operations and device management for CUDA backends.
    For CuPy-based CUDA backends, use CupyCUDABackendBase instead.
    """

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
        # Match Torch backend's _normalize_to_channels_first method exactly
        # IMPORTANT: Do NOT fix corrupted formats - match Torch's behavior exactly
        # (even if Torch processes corrupted formats incorrectly)
        needs_permute = False
        if self.channel_axis == -1 or (self.channel_axis == 3 and images.ndim == 4):
            # Channels-last format (N, H, W, C) -> (N, C, H, W)
            # Trust channel_axis - assume input is in channels-last format
            # Note: This may produce wrong format if prepare_for_normalizer corrupted the input,
            # but we match Torch backend's behavior exactly
            images = images.permute(0, 3, 1, 2)
            needs_permute = True

        # Handle reference histogram (can be list or single tensor)
        # If list provided, stack into (C, 256) tensor for per-channel processing
        if isinstance(reference_histogram, list):
            if len(reference_histogram) == 0:
                raise ValueError("reference_histogram list cannot be empty")

            # Validate shapes first (quick check without moving to device)
            for i, h in enumerate(reference_histogram):
                if not isinstance(h, torch.Tensor):
                    raise TypeError(f"reference_histogram[{i}] must be a torch.Tensor, got {type(h)}")
                if h.dim() != 1 or h.size(0) != 256:
                    raise ValueError(f"Each histogram in reference_histogram list must be 1D with 256 elements. Got histogram at index {i} with shape {h.sizes()}")

            # Stack all histograms at once (more efficient than moving each individually)
            ref_hist = torch.stack(reference_histogram, dim=0)  # (num_histograms, 256)

            # Ensure we have histograms for all channels (pad with first if needed)
            num_channels = images.size(1)
            if ref_hist.size(0) < num_channels:
                # Pad by repeating the first histogram
                padding = ref_hist[0:1].repeat(num_channels - ref_hist.size(0), 1)
                ref_hist = torch.cat([ref_hist, padding], dim=0)
            elif ref_hist.size(0) > num_channels:
                # Truncate to match number of channels
                ref_hist = ref_hist[:num_channels]

            # Move to device once after stacking
            ref_hist = ref_hist.to(self.device)  # (C, 256)
        else:
            # Single histogram for all channels: (256,)
            ref_hist = reference_histogram.to(self.device)
            if ref_hist.dim() != 1 or ref_hist.size(0) != 256:
                raise ValueError(f"reference_histogram must be 1D with 256 elements. Got shape {ref_hist.sizes()}")

        # Check if CUDA function is available
        if not hasattr(stainx_cuda_torch, "histogram_matching"):
            raise RuntimeError("stainx_cuda_torch.histogram_matching is not available. The CUDA extension may not be built correctly.")

        # Call CUDA implementation (expects and returns (N, C, H, W) format)
        # CUDA function now accepts either (256,) or (C, 256) tensor
        result = stainx_cuda_torch.histogram_matching(images, ref_hist)

        # Restore to original channel format if needed (matching Torch backend logic)
        if needs_permute:
            # Convert back to channels-last (N, C, H, W) -> (N, H, W, C)
            result = result.permute(0, 2, 3, 1)

        return result


class ReinhardCUDA(TorchCUDABackendBase):
    def transform(self, images: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        target_mean = target_mean.to(self.device)
        target_std = target_std.to(self.device)

        # Check if CUDA function is available
        if not hasattr(stainx_cuda_torch, "reinhard"):
            raise RuntimeError("stainx_cuda_torch.reinhard is not available. The CUDA extension may not be built correctly.")

        # Call CUDA implementation
        return stainx_cuda_torch.reinhard(images, target_mean, target_std)


class MacenkoCUDA(TorchCUDABackendBase):
    def transform(self, images: torch.Tensor, stain_matrix: torch.Tensor, target_max_conc: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        stain_matrix = stain_matrix.to(self.device)
        target_max_conc = target_max_conc.to(self.device)

        # Check if CUDA function is available
        if not hasattr(stainx_cuda_torch, "macenko"):
            raise RuntimeError("stainx_cuda_torch.macenko is not available. The CUDA extension may not be built correctly.")

        # Call CUDA implementation
        return stainx_cuda_torch.macenko(images, stain_matrix, target_max_conc)
