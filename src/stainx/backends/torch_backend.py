# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
"""
PyTorch backend implementations for stain normalization.

This module provides simple PyTorch implementations of all stain normalization
algorithms. These serve as fallback implementations when CUDA backend
is not available.
"""

import torch


class PyTorchBackendBase:
    """Base class for PyTorch backend implementations."""

    def __init__(self, device: str | torch.device | None = None):
        """
        Initialize the PyTorch backend.

        Parameters
        ----------
        device : str or torch.device, optional
            Device to run computations on
        """
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    @staticmethod
    def rgb_to_lab(rgb: torch.Tensor, channel_axis: int = 1) -> torch.Tensor:
        """
        Convert RGB to LAB color space using PyTorch operations.

        This implementation follows the standard RGB -> XYZ -> LAB conversion
        using sRGB color space and D65 white point.

        Parameters
        ----------
        rgb : torch.Tensor
            RGB images of shape (N, C, H, W) or (N, H, W, C) with C=3
            Values should be in range [0, 1]
        channel_axis : int
            Axis containing color channels (1 for NCHW, -1 for NHWC)

        Returns
        -------
        torch.Tensor
            LAB images with same shape as input
        """
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        if channel_axis == -1 or (channel_axis == 3 and rgb.ndim == 4):
            rgb = rgb.permute(0, 3, 1, 2)
            needs_permute = True
        else:
            needs_permute = False

        # Gamma correction: sRGB to linear RGB
        mask = rgb > 0.04045
        linear_rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # RGB to XYZ conversion using einsum (highly optimized in PyTorch)
        transform_matrix = torch.tensor(
            [
                [0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227],
            ],
            dtype=rgb.dtype,
            device=rgb.device,
        )

        xyz = torch.einsum("ij,njhw->nihw", transform_matrix, linear_rgb)

        # Normalize by D65 white point
        xyz_ref = torch.tensor(
            [0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device
        ).view(1, 3, 1, 1)
        xyz_norm = xyz / xyz_ref

        # XYZ to LAB conversion
        mask = xyz_norm > 0.008856
        f_xyz = torch.where(
            mask, torch.pow(xyz_norm, 1.0 / 3.0), 7.787 * xyz_norm + 16.0 / 116.0
        )

        # Compute L, a, b channels vectorized (avoid separate slicing and concatenation)
        # f_xyz shape: (N, 3, H, W)
        f_x = f_xyz[:, 0:1, :, :]  # (N, 1, H, W)
        f_y = f_xyz[:, 1:2, :, :]  # (N, 1, H, W)
        f_z = f_xyz[:, 2:3, :, :]  # (N, 1, H, W)

        # Compute all channels at once
        L = (116.0 * f_y - 16.0) * 2.55
        a = 500.0 * (f_x - f_y) + 128.0
        b = 200.0 * (f_y - f_z) + 128.0

        lab = torch.cat([L, a, b], dim=1)

        if needs_permute:
            lab = lab.permute(0, 2, 3, 1)

        return lab

    @staticmethod
    def lab_to_rgb(lab: torch.Tensor, channel_axis: int = 1) -> torch.Tensor:
        """
        Convert LAB to RGB color space using PyTorch operations.

        This implementation follows the standard LAB -> XYZ -> RGB conversion
        using sRGB color space and D65 white point.

        Parameters
        ----------
        lab : torch.Tensor
            LAB images of shape (N, C, H, W) or (N, H, W, C) with C=3
        channel_axis : int
            Axis containing color channels (1 for NCHW, -1 for NHWC)

        Returns
        -------
        torch.Tensor
            RGB images with same shape as input, values in range [0, 1]
        """
        if channel_axis == -1 or (channel_axis == 3 and lab.ndim == 4):
            lab = lab.permute(0, 3, 1, 2)
            needs_permute = True
        else:
            needs_permute = False

        L = lab[:, 0:1, :, :] / 2.55
        a = lab[:, 1:2, :, :] - 128.0
        b = lab[:, 2:3, :, :] - 128.0

        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        def f_inv(t):
            mask = t > 0.2068966
            return torch.where(mask, t**3, (t - 16.0 / 116.0) / 7.787)

        xyz_norm = torch.cat([f_inv(fx), f_inv(fy), f_inv(fz)], dim=1)

        xyz_ref = torch.tensor(
            [0.95047, 1.0, 1.08883], dtype=xyz_norm.dtype, device=xyz_norm.device
        )
        xyz_ref = xyz_ref.view(1, 3, 1, 1)

        xyz = xyz_norm * xyz_ref

        transform_matrix = torch.tensor(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ],
            dtype=xyz.dtype,
            device=xyz.device,
        )

        linear_rgb = torch.einsum("ij,njhw->nihw", transform_matrix, xyz)

        mask = linear_rgb > 0.0031308
        rgb = torch.where(
            mask, 1.055 * torch.pow(linear_rgb, 1.0 / 2.4) - 0.055, 12.92 * linear_rgb
        )

        rgb = torch.clamp(rgb, 0.0, 1.0)

        if needs_permute:
            rgb = rgb.permute(0, 2, 3, 1)

        return rgb

    @staticmethod
    def normalize_to_float(images: torch.Tensor) -> torch.Tensor:
        """
        Normalize images to float in [0, 1] range.

        Parameters
        ----------
        images : torch.Tensor
            Input images, values in [0, 1] or [0, 255]

        Returns
        -------
        torch.Tensor
            Images normalized to float in [0, 1]
        """
        if images.max() > 1.0:
            return images.float() / 255.0
        return images.float()

    def images_to_uint8(self, images: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        Convert images to uint8 format.

        Parameters
        ----------
        images : torch.Tensor
            Input images, values in [0, 1] or [0, 255]

        Returns
        -------
        tuple[torch.Tensor, bool]
            (images_uint8, needs_scale_back) where needs_scale_back indicates
            if the original images were in [0, 1] range
        """
        if images.max() <= 1.0:
            images_uint8 = (images * 255.0).clamp(0, 255).to(torch.uint8)
            needs_scale_back = True
        else:
            images_uint8 = images.clamp(0, 255).to(torch.uint8)
            needs_scale_back = False
        return images_uint8, needs_scale_back

    def preserve_dtype(
        self,
        result: torch.Tensor,
        original_dtype: torch.dtype,
        was_uint8_or_high_range: bool = False,
        result_in_0_255_range: bool = False,
    ) -> torch.Tensor:
        """
        Preserve original dtype, handling uint8 and high-range inputs.

        Parameters
        ----------
        result : torch.Tensor
            Result tensor in float format
        original_dtype : torch.dtype
            Original dtype to restore
        was_uint8_or_high_range : bool
            Whether original input was uint8 or had values > 1.0
        result_in_0_255_range : bool
            Whether result is already in [0, 255] range (True) or [0, 1] range (False)

        Returns
        -------
        torch.Tensor
            Result tensor with original dtype preserved
        """
        # If result is in [0, 1] range but we need [0, 255], scale it
        if not result_in_0_255_range and (
            original_dtype == torch.uint8 or was_uint8_or_high_range
        ):
            result = (result * 255.0).clamp(0, 255)
        elif result_in_0_255_range:
            # Result is already in [0, 255], just clamp
            result = result.clamp(0, 255)

        # Convert to original dtype
        if original_dtype == torch.uint8:
            result = result.to(original_dtype)
        else:
            result = result.to(original_dtype)

        return result

    def compute_histogram_256(self, channel: torch.Tensor) -> torch.Tensor:
        """
        Compute 256-bin normalized histogram from a channel.

        Parameters
        ----------
        channel : torch.Tensor
            Channel values as uint8 (flat tensor)

        Returns
        -------
        torch.Tensor
            Normalized histogram of shape (256,)
        """
        counts = torch.bincount(channel, minlength=256).float()
        return counts / (counts.sum() + 1e-8)

    def compute_reference_histograms(
        self, images: torch.Tensor
    ) -> tuple[list, list, list, torch.Tensor]:
        """
        Compute reference histograms from images for histogram matching.

        Parameters
        ----------
        images : torch.Tensor
            Reference images of shape (N, C, H, W) or (N, H, W, C) with C=3

        Returns
        -------
        tuple[list, list, list, torch.Tensor]
            (ref_vals, ref_cdf, ref_histograms_256, reference_histogram)
            - ref_vals: List of value arrays for each channel
            - ref_cdf: List of CDF arrays for each channel
            - ref_histograms_256: List of 256-bin normalized histograms for each channel
            - reference_histogram: Single reference histogram (first channel's CDF)
        """
        # Normalize to channels-first format for processing
        images_normalized, _ = self._normalize_to_channels_first(images)
        images_uint8, _ = self.images_to_uint8(images_normalized)
        images_uint8 = images_uint8.to(self.device)

        _N, C, _H, _W = images_uint8.shape

        ref_vals = []
        ref_cdf = []
        ref_histograms_256 = []

        for c in range(C):
            channel = images_uint8[:, c, :, :]
            flat_channel = channel.reshape(-1)

            hist_256 = self.compute_histogram_256(flat_channel)
            ref_histograms_256.append(hist_256)

            counts = torch.bincount(flat_channel, minlength=256).float()
            nonzero_idx = torch.nonzero(counts, as_tuple=False).squeeze(-1)
            if len(nonzero_idx) > 0:
                vals = nonzero_idx.float()
                cnts = counts[nonzero_idx]
                cdf = torch.cumsum(cnts, dim=0)
                total = cdf[-1]
                cdf = cdf / (total + 1e-8)
            else:
                vals = torch.arange(256, dtype=torch.float32, device=self.device)
                cdf = torch.zeros(256, dtype=torch.float32, device=self.device)

            ref_vals.append(vals)
            ref_cdf.append(cdf)

        reference_histogram = ref_cdf[0]

        return ref_vals, ref_cdf, ref_histograms_256, reference_histogram


class HistogramMatchingPyTorch(PyTorchBackendBase):
    """PyTorch implementation of histogram matching."""

    def __init__(self, device: str | torch.device | None = None, channel_axis: int = 1):
        super().__init__(device)
        self.channel_axis = channel_axis

    def _normalize_to_channels_first(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """
        Normalize images to channels-first format (N, C, H, W).

        Parameters
        ----------
        images : torch.Tensor
            Input images in any format

        Returns
        -------
        tuple[torch.Tensor, bool]
            (normalized_images, needs_permute_back)
            - normalized_images: Images in (N, C, H, W) format
            - needs_permute_back: True if input was channels-last
        """
        if self.channel_axis == -1 or (self.channel_axis == 3 and images.ndim == 4):
            # Channels-last format (N, H, W, C) -> (N, C, H, W)
            return images.permute(0, 3, 1, 2), True
        # Already channels-first (N, C, H, W)
        return images, False

    def _restore_format(
        self, images: torch.Tensor, needs_permute: bool
    ) -> torch.Tensor:
        """
        Restore images to original format.

        Parameters
        ----------
        images : torch.Tensor
            Images in (N, C, H, W) format
        needs_permute : bool
            Whether to permute back to channels-last

        Returns
        -------
        torch.Tensor
            Images in original format
        """
        if needs_permute:
            # Convert back to channels-last (N, C, H, W) -> (N, H, W, C)
            return images.permute(0, 2, 3, 1)
        return images

    def transform(
        self, images: torch.Tensor, reference_histogram: torch.Tensor | list
    ) -> torch.Tensor:
        """
        Apply histogram matching transformation using PyTorch.

        Parameters
        ----------
        images : torch.Tensor
            Input images of shape (N, C, H, W) or (N, H, W, C) with C=3
        reference_histogram : torch.Tensor or list of torch.Tensor
            Reference histogram(s). Can be:
            - 1D tensor of shape (256,): target histogram for all channels
            - List of 1D tensors, each of shape (256,): per-channel histograms

        Returns
        -------
        torch.Tensor
            Histogram-matched images with same shape and dtype as input
        """
        # Normalize to channels-first format for processing
        images_normalized, needs_permute = self._normalize_to_channels_first(images)
        images_normalized = images_normalized.to(self.device)

        if isinstance(reference_histogram, list):
            per_channel_histograms = [h.to(self.device) for h in reference_histogram]
        else:
            reference_histogram = reference_histogram.to(self.device)
            per_channel_histograms = None

        original_dtype = images_normalized.dtype
        was_uint8_or_high_range = (
            images_normalized.dtype == torch.uint8 or images_normalized.max() > 1.0
        )

        images_uint8, needs_scale_back = self.images_to_uint8(images_normalized)

        N, C, H, W = images_uint8.shape
        matched_channels = []

        # Pre-compute reference values (same for all channels if using single reference histogram)
        ref_values = torch.arange(256, dtype=torch.float32, device=self.device)

        # Pre-compute reference CDFs for all channels if per-channel histograms provided
        precomputed_ref_cdfs = None
        precomputed_ref_cdf = None
        if per_channel_histograms is not None:
            precomputed_ref_cdfs = []
            for ref_hist in per_channel_histograms:
                ref_hist_norm = ref_hist.float() / (ref_hist.float().sum() + 1e-8)
                precomputed_ref_cdfs.append(torch.cumsum(ref_hist_norm, dim=0))
        elif reference_histogram.ndim == 1 and len(reference_histogram) == 256:
            ref_hist_norm = reference_histogram.float() / (
                reference_histogram.float().sum() + 1e-8
            )
            precomputed_ref_cdf = torch.cumsum(ref_hist_norm, dim=0)

        for c in range(C):
            channel = images_uint8[:, c, :, :]
            flat_channel = channel.reshape(-1)
            num_pixels = flat_channel.numel()

            # Build source histogram for all 256 values (much faster than unique)
            source_hist_counts = torch.bincount(flat_channel, minlength=256).float()
            source_hist = source_hist_counts / (num_pixels + 1e-8)
            source_cdf = torch.cumsum(source_hist, dim=0)

            # Get reference quantiles for this channel
            if precomputed_ref_cdfs is not None and c < len(precomputed_ref_cdfs):
                ref_quantiles = precomputed_ref_cdfs[c]
            elif precomputed_ref_cdf is not None:
                ref_quantiles = precomputed_ref_cdf
            else:
                # Fallback: compute on the fly (shouldn't happen with proper input)
                if per_channel_histograms is not None and c < len(
                    per_channel_histograms
                ):
                    ref_hist = per_channel_histograms[c].float()
                    ref_hist = ref_hist / (ref_hist.sum() + 1e-8)
                    ref_quantiles = torch.cumsum(ref_hist, dim=0)
                else:
                    ref_hist = reference_histogram.float()
                    ref_hist = ref_hist / (ref_hist.sum() + 1e-8)
                    ref_quantiles = torch.cumsum(ref_hist, dim=0)

            # Build lookup table: for each of 256 possible source values, find matched value
            # Vectorized matching: process all source quantiles at once
            ref_quantiles_min = ref_quantiles[0]
            ref_quantiles_max = ref_quantiles[-1]

            # Vectorized searchsorted for all quantiles at once
            indices = torch.searchsorted(ref_quantiles, source_cdf, right=False)
            indices = torch.clamp(indices, 1, len(ref_quantiles) - 1)

            # Vectorized interpolation
            quantile_left = ref_quantiles[indices - 1]
            quantile_right = ref_quantiles[indices]

            # Handle edge cases: values <= min or >= max
            below_min = source_cdf <= ref_quantiles_min
            above_max = source_cdf >= ref_quantiles_max

            # Compute alpha for interpolation (avoid division by zero)
            quantile_diff = quantile_right - quantile_left
            alpha = torch.where(
                quantile_diff > 1e-10,
                (source_cdf - quantile_left) / quantile_diff,
                torch.zeros_like(source_cdf),
            )

            # Interpolate values to build lookup table
            lookup_table = ref_values[indices - 1] + alpha * (
                ref_values[indices] - ref_values[indices - 1]
            )

            # Apply edge case handling
            lookup_table = torch.where(below_min, ref_values[0], lookup_table)
            lookup_table = torch.where(above_max, ref_values[-1], lookup_table)
            lookup_table = lookup_table.clamp(0, 255)

            # Direct lookup: map all pixels at once using the lookup table
            # flat_channel is uint8, so values are already in [0, 255], just convert to long for indexing
            matched_channel = lookup_table[flat_channel.long()].reshape(N, H, W)
            matched_channels.append(matched_channel)

        matched = torch.stack(matched_channels, dim=1)

        if needs_scale_back:
            matched = matched / 255.0
            result_in_0_255_range = False
        else:
            result_in_0_255_range = True

        matched = matched.clamp(0.0, 1.0 if needs_scale_back else 255.0)

        matched = self.preserve_dtype(
            matched, original_dtype, was_uint8_or_high_range, result_in_0_255_range
        )

        # Restore to original format
        return self._restore_format(matched, needs_permute)


class ReinhardPyTorch(PyTorchBackendBase):
    """PyTorch implementation of Reinhard normalization."""

    def __init__(self, device: str | torch.device | None = None):
        super().__init__(device)

    def compute_reference_mean_std(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reference mean and std from images in LAB color space.

        Parameters
        ----------
        images : torch.Tensor
            Reference images of shape (N, C, H, W) with C=3

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (reference_mean, reference_std) where both are shape (3,)
        """
        images = images.to(self.device, non_blocking=True)
        original_dtype = images.dtype
        was_uint8 = original_dtype == torch.uint8

        # Check range once and normalize inline
        if was_uint8 or images.max() > 1.0:
            images_float = images.float() / 255.0
        else:
            images_float = images.float()

        lab = self.rgb_to_lab(images_float, channel_axis=1)

        # Compute mean and std for all channels at once (vectorized)
        # Shape: (3,) - one value per channel
        reference_mean = lab.mean(dim=(0, 2, 3))
        reference_std = lab.std(dim=(0, 2, 3))

        return reference_mean, reference_std

    def transform(
        self,
        images: torch.Tensor,
        reference_mean: torch.Tensor,
        reference_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Reinhard normalization transformation using PyTorch.

        Parameters
        ----------
        images : torch.Tensor
            Input images of shape (N, C, H, W) with C=3, values in [0, 1] or [0, 255]
        reference_mean : torch.Tensor
            Reference mean in LAB color space, shape (3,) or (1, 3, 1, 1)
        reference_std : torch.Tensor
            Reference std in LAB color space, shape (3,) or (1, 3, 1, 1)

        Returns
        -------
        torch.Tensor
            Reinhard-normalized images with same shape and dtype as input
        """
        images = images.to(self.device, non_blocking=True)
        original_dtype = images.dtype
        was_uint8 = original_dtype == torch.uint8

        # Check range once and normalize
        if was_uint8 or images.max() > 1.0:
            images_float = images.float() / 255.0
            was_uint8_or_high_range = True
        else:
            images_float = images.float()
            was_uint8_or_high_range = False

        # Prepare reference tensors (avoid redundant device transfers)
        if reference_mean.ndim == 1:
            reference_mean = reference_mean.view(1, 3, 1, 1).to(
                self.device, non_blocking=True
            )
        else:
            reference_mean = reference_mean.to(self.device, non_blocking=True)
        if reference_std.ndim == 1:
            reference_std = reference_std.view(1, 3, 1, 1).to(
                self.device, non_blocking=True
            )
        else:
            reference_std = reference_std.to(self.device, non_blocking=True)

        lab = self.rgb_to_lab(images_float, channel_axis=1)

        # Compute mean and std for all channels at once (vectorized)
        lab_mean = lab.mean(dim=(0, 2, 3), keepdim=True)
        lab_std = lab.std(dim=(0, 2, 3), keepdim=True)

        # Vectorized normalization across all channels
        lab_normalized = (
            (lab - lab_mean) / (lab_std + 1e-8)
        ) * reference_std + reference_mean

        rgb_normalized = self.lab_to_rgb(lab_normalized, channel_axis=1)

        rgb_normalized = torch.clamp(rgb_normalized, 0.0, 1.0)

        return self.preserve_dtype(
            rgb_normalized,
            original_dtype,
            was_uint8_or_high_range,
            result_in_0_255_range=False,
        )


class MacenkoPyTorch(PyTorchBackendBase):
    """PyTorch implementation of Macenko normalization."""

    def __init__(self, device: str | torch.device | None = None):
        super().__init__(device)

    @staticmethod
    def _percentile(t: torch.Tensor, q: float) -> float:
        """Percentile matching torchstain's implementation."""
        k = 1 + round(0.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()

    @staticmethod
    def _eigh_with_mps_fallback(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigendecomposition with fallback to CPU for MPS devices.

        MPS (Metal Performance Shaders) on Apple Silicon doesn't support
        torch.linalg.eigh, so we fall back to CPU for this operation.

        Parameters
        ----------
        cov : torch.Tensor
            Covariance matrix of shape (3, 3)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (eigenvalues, eigenvectors) where eigenvectors are on the same device as input
        """
        device = cov.device
        # MPS doesn't support eigh, so move to CPU for computation
        # TODO[Samir]: In the future, if this issue (https://github.com/pytorch/pytorch/issues/141287) is resolved, we can remove this fallback.
        if device.type == "mps":
            cov_cpu = cov.cpu()
            eigvals, eigvecs = torch.linalg.eigh(cov_cpu)
            # Move eigenvectors back to original device
            eigvecs = eigvecs.to(device)
            eigvals = eigvals.to(device)
        else:
            eigvals, eigvecs = torch.linalg.eigh(cov)
        return eigvals, eigvecs

    def _process_single_image(
        self,
        od: torch.Tensor,
        stain_matrix: torch.Tensor,
        target_max_conc: torch.Tensor,
        beta: float,
        alpha: float,
        Io: float,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Process a single image for Macenko normalization.
        Extracted to reduce Python loop overhead.
        """
        # Reshape to (H*W, 3)
        od_reshaped = od.permute(1, 2, 0).reshape(-1, 3)

        # Filter by minimum OD
        od_min = od_reshaped.min(dim=1)[0]
        mask = od_min >= beta
        od_filtered = od_reshaped[mask]

        # Early exit if too few pixels (safety check)
        if od_filtered.shape[0] < 3:
            od_filtered = od_reshaped

        # Compute covariance matrix efficiently
        od_filtered_T = od_filtered.T  # (3, num_filtered)
        od_mean = od_filtered_T.mean(dim=1, keepdim=True)
        od_centered = od_filtered_T - od_mean
        num_pixels = od_filtered.shape[0]
        if num_pixels > 1:
            cov = torch.matmul(od_centered, od_centered.T) / (num_pixels - 1)
        else:
            cov = torch.zeros(
                (3, 3), dtype=od_centered.dtype, device=od_centered.device
            )

        _, eigvecs = self._eigh_with_mps_fallback(cov)
        eigvecs = eigvecs[:, [1, 2]]

        That = torch.matmul(od_filtered, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])

        # Compute percentiles
        min_phi = self._percentile(phi, alpha)
        max_phi = self._percentile(phi, 100 - alpha)

        # Pre-compute cos/sin values
        min_phi_t = torch.tensor(min_phi, dtype=torch.float32, device=od.device)
        max_phi_t = torch.tensor(max_phi, dtype=torch.float32, device=od.device)
        cos_min = torch.cos(min_phi_t)
        sin_min = torch.sin(min_phi_t)
        cos_max = torch.cos(max_phi_t)
        sin_max = torch.sin(max_phi_t)

        # Stack once
        angle_min = torch.stack((cos_min, sin_min))
        angle_max = torch.stack((cos_max, sin_max))

        vMin = torch.matmul(eigvecs, angle_min).unsqueeze(1)
        vMax = torch.matmul(eigvecs, angle_max).unsqueeze(1)

        HE_source = torch.where(
            vMin[0] > vMax[0],
            torch.cat((vMin, vMax), dim=1),
            torch.cat((vMax, vMin), dim=1),
        )

        # Reshape OD for concentration computation
        od_all = od.reshape(3, -1)

        # Check condition number and matrix size to decide on fallback
        # CUSOLVER fails on ill-conditioned matrices and very large matrices
        use_fallback = False
        # Use fallback for very large right-hand side matrices (CUSOLVER has issues)
        if od_all.shape[1] > 1000000:  # > 1M columns
            use_fallback = True

        if not use_fallback and HE_source.shape[0] >= HE_source.shape[1]:
            try:
                cond_num = torch.linalg.cond(HE_source)
                # Use fallback if condition number is too high (> 10)
                if cond_num.item() > 10.0:
                    use_fallback = True
            except Exception:
                use_fallback = True

        if use_fallback:
            # Use pseudoinverse for ill-conditioned or large matrices (more robust)
            HE_pinv = torch.linalg.pinv(HE_source)
            concentrations = torch.matmul(HE_pinv, od_all)
        else:
            try:
                concentrations, _, _, _ = torch.linalg.lstsq(
                    HE_source, od_all, rcond=None
                )
            except RuntimeError:
                # CUSOLVER error - fallback to pseudoinverse
                HE_pinv = torch.linalg.pinv(HE_source)
                concentrations = torch.matmul(HE_pinv, od_all)

        # Compute max concentrations
        max_conc_0 = self._percentile(concentrations[0, :], 99)
        max_conc_1 = self._percentile(concentrations[1, :], 99)
        max_conc = torch.stack(
            [
                torch.tensor(max_conc_0, dtype=torch.float32, device=od.device),
                torch.tensor(max_conc_1, dtype=torch.float32, device=od.device),
            ]
        )

        # Normalize concentrations
        norm_factor = target_max_conc / max_conc
        concentrations_norm = concentrations * norm_factor.unsqueeze(-1)

        # Reconstruct OD
        od_recon = torch.matmul(stain_matrix, concentrations_norm)

        # Convert back to RGB
        rgb_recon = Io * torch.exp(-od_recon)
        rgb_recon = torch.clamp(rgb_recon, 0, 255)

        return rgb_recon.reshape(3, H, W)

    def compute_reference_stain_matrix(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reference stain matrix and target max concentrations from images.

        Parameters
        ----------
        images : torch.Tensor
            Reference images of shape (N, C, H, W) with C=3

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (stain_matrix, target_max_conc) where:
            - stain_matrix: Shape (3, 2)
            - target_max_conc: Shape (2,)
        """
        images_float = self.normalize_to_float(images)
        images_float = images_float.to(self.device)

        _N, _C, _H, _W = images_float.shape

        Io = 240.0
        # Vectorize: process all images at once
        od = -torch.log((images_float * 255.0 + 1.0) / Io)
        # Reshape to (N, 3, H*W) then combine: (3, N*H*W)
        od_combined = od.permute(1, 0, 2, 3).reshape(3, -1)

        # Transpose once: (N*H*W, 3)
        od_combined_T = od_combined.T

        # Filter by minimum OD
        od_min = od_combined_T.min(dim=1)[0]
        mask = od_min >= 0.15
        od_for_cov = od_combined_T[mask]  # (num_pixels, 3)

        # Compute covariance matrix
        od_mean = od_for_cov.mean(dim=0, keepdim=True)
        od_centered = od_for_cov - od_mean
        cov = torch.matmul(od_centered.T, od_centered) / (od_for_cov.shape[0] - 1)

        _eigvals, eigvecs = self._eigh_with_mps_fallback(cov)

        stain_vectors = eigvecs[:, [1, 2]]

        That = torch.matmul(od_for_cov, stain_vectors)

        phi = torch.atan2(That[:, 1], That[:, 0])

        alpha = 1.0
        min_phi = self._percentile(phi, alpha)
        max_phi = self._percentile(phi, 100 - alpha)

        device = stain_vectors.device
        cos_min = torch.cos(torch.tensor(min_phi, device=device))
        sin_min = torch.sin(torch.tensor(min_phi, device=device))
        cos_max = torch.cos(torch.tensor(max_phi, device=device))
        sin_max = torch.sin(torch.tensor(max_phi, device=device))

        vMin = torch.matmul(stain_vectors, torch.stack((cos_min, sin_min))).unsqueeze(1)
        vMax = torch.matmul(stain_vectors, torch.stack((cos_max, sin_max))).unsqueeze(1)

        HE = torch.where(
            vMin[0] > vMax[0],
            torch.cat((vMin, vMax), dim=1),
            torch.cat((vMax, vMin), dim=1),
        )

        stain_matrix = HE

        # Check condition number and matrix size to decide on fallback
        # CUSOLVER fails on ill-conditioned matrices and very large matrices
        use_fallback = False
        # Use fallback for very large right-hand side matrices (CUSOLVER has issues)
        if od_combined.shape[1] > 1000000:  # > 1M columns
            use_fallback = True

        if not use_fallback and HE.shape[0] >= HE.shape[1]:
            try:
                cond_num = torch.linalg.cond(HE)
                # Use fallback if condition number is too high (> 10)
                if cond_num.item() > 10.0:
                    use_fallback = True
            except Exception:
                use_fallback = True

        if use_fallback:
            # Use pseudoinverse for ill-conditioned or large matrices (more robust)
            HE_pinv = torch.linalg.pinv(HE)
            concentrations = torch.matmul(HE_pinv, od_combined)
        else:
            try:
                concentrations, _, _, _ = torch.linalg.lstsq(
                    HE, od_combined, rcond=None
                )
            except RuntimeError:
                # CUSOLVER error - fallback to pseudoinverse
                HE_pinv = torch.linalg.pinv(HE)
                concentrations = torch.matmul(HE_pinv, od_combined)

        max_conc = torch.tensor(
            [
                self._percentile(concentrations[0, :], 99),
                self._percentile(concentrations[1, :], 99),
            ],
            device=self.device,
            dtype=torch.float32,
        )

        return stain_matrix, max_conc

    def transform(
        self,
        images: torch.Tensor,
        stain_matrix: torch.Tensor,
        target_max_conc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Macenko normalization transformation using PyTorch.

        Parameters
        ----------
        images : torch.Tensor
            Input images of shape (N, C, H, W) with C=3, values in [0, 1] or [0, 255]
        stain_matrix : torch.Tensor
            Reference stain matrix of shape (3, 2) or (3, 3)
        target_max_conc : torch.Tensor
            Target max concentrations from reference, shape (2,)

        Returns
        -------
        torch.Tensor
            Macenko-normalized images with same shape and dtype as input
        """
        images = images.to(self.device, non_blocking=True)
        stain_matrix = stain_matrix.to(self.device, non_blocking=True)
        target_max_conc = target_max_conc.to(self.device, non_blocking=True)

        original_dtype = images.dtype
        was_uint8_or_high_range = images.dtype == torch.uint8 or images.max() > 1.0

        images_float = self.normalize_to_float(images)

        if stain_matrix.shape != (3, 2):
            raise ValueError(
                f"stain_matrix must have shape (3, 2), got {stain_matrix.shape}"
            )

        N, C, H, W = images_float.shape

        # Pre-compute constants
        Io = 240.0
        beta = 0.15
        alpha = 1.0

        # Flatten target_max_conc once
        if target_max_conc.ndim > 1:
            target_max_conc = target_max_conc.flatten()

        # Vectorize OD computation for all images at once
        od_all_images = -torch.log((images_float * 255.0 + 1.0) / Io)  # (N, 3, H, W)

        # Pre-allocate output tensor
        normalized = torch.empty((N, C, H, W), dtype=torch.float32, device=self.device)

        # Process each image - loop is necessary due to variable filtered pixel counts
        for n in range(N):
            od = od_all_images[n]  # (3, H, W)
            normalized[n] = self._process_single_image(
                od, stain_matrix, target_max_conc, beta, alpha, Io, H, W
            )

        return self.preserve_dtype(
            normalized,
            original_dtype,
            was_uint8_or_high_range,
            result_in_0_255_range=True,
        )
