# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import cupy as cp


class CupyBackendBase:
    def __init__(self, device: str | cp.cuda.Device | None = None):
        if device is None:
            self.device = cp.cuda.Device(0) if cp.cuda.is_available() else None
        else:
            if isinstance(device, cp.cuda.Device):
                self.device = device
            else:
                self.device = cp.cuda.Device(0) if cp.cuda.is_available() else None

    @staticmethod
    def rgb_to_lab_cupy(rgb: cp.ndarray, channel_axis: int = 1) -> cp.ndarray:
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        if channel_axis == -1 or (channel_axis == 3 and rgb.ndim == 4):
            rgb = cp.transpose(rgb, (0, 3, 1, 2))
            needs_permute = True
        else:
            needs_permute = False

        # Gamma correction: sRGB to linear RGB
        mask = rgb > 0.04045
        linear_rgb = cp.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # RGB to XYZ conversion using einsum (highly optimized in CuPy)
        transform_matrix = cp.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]], dtype=rgb.dtype)

        xyz = cp.einsum("ij,njhw->nihw", transform_matrix, linear_rgb)

        # Normalize by D65 white point
        xyz_ref = cp.array([0.95047, 1.0, 1.08883], dtype=xyz.dtype).reshape(1, 3, 1, 1)
        xyz_norm = xyz / xyz_ref

        # XYZ to LAB conversion
        mask = xyz_norm > 0.008856
        f_xyz = cp.where(mask, cp.power(xyz_norm, 1.0 / 3.0), 7.787 * xyz_norm + 16.0 / 116.0)

        # Compute L, a, b channels vectorized (avoid separate slicing and concatenation)
        # f_xyz shape: (N, 3, H, W)
        f_x = f_xyz[:, 0:1, :, :]  # (N, 1, H, W)
        f_y = f_xyz[:, 1:2, :, :]  # (N, 1, H, W)
        f_z = f_xyz[:, 2:3, :, :]  # (N, 1, H, W)

        # Compute all channels at once
        L = (116.0 * f_y - 16.0) * 2.55
        a = 500.0 * (f_x - f_y) + 128.0
        b = 200.0 * (f_y - f_z) + 128.0

        lab = cp.concatenate([L, a, b], axis=1)

        if needs_permute:
            lab = cp.transpose(lab, (0, 2, 3, 1))

        return lab

    @staticmethod
    def lab_to_rgb_cupy(lab: cp.ndarray, channel_axis: int = 1) -> cp.ndarray:
        if channel_axis == -1 or (channel_axis == 3 and lab.ndim == 4):
            lab = cp.transpose(lab, (0, 3, 1, 2))
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
            return cp.where(mask, t**3, (t - 16.0 / 116.0) / 7.787)

        xyz_norm = cp.concatenate([f_inv(fx), f_inv(fy), f_inv(fz)], axis=1)

        xyz_ref = cp.array([0.95047, 1.0, 1.08883], dtype=xyz_norm.dtype)
        xyz_ref = xyz_ref.reshape(1, 3, 1, 1)

        xyz = xyz_norm * xyz_ref

        transform_matrix = cp.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]], dtype=xyz.dtype)

        linear_rgb = cp.einsum("ij,njhw->nihw", transform_matrix, xyz)

        mask = linear_rgb > 0.0031308
        rgb = cp.where(mask, 1.055 * cp.power(linear_rgb, 1.0 / 2.4) - 0.055, 12.92 * linear_rgb)

        rgb = cp.clip(rgb, 0.0, 1.0)

        if needs_permute:
            rgb = cp.transpose(rgb, (0, 2, 3, 1))

        return rgb

    @staticmethod
    def normalize_to_float_cupy(images: cp.ndarray) -> cp.ndarray:
        if images.max() > 1.0:
            return images.astype(cp.float32) / 255.0
        return images.astype(cp.float32)

    def images_to_uint8_cupy(self, images: cp.ndarray) -> tuple[cp.ndarray, bool]:
        if images.max() <= 1.0:
            images_uint8 = (images * 255.0).clip(0, 255).astype(cp.uint8)
            needs_scale_back = True
        else:
            images_uint8 = images.clip(0, 255).astype(cp.uint8)
            needs_scale_back = False
        return images_uint8, needs_scale_back

    def preserve_dtype_cupy(self, result: cp.ndarray, original_dtype: cp.dtype, was_uint8_or_high_range: bool = False, result_in_0_255_range: bool = False) -> cp.ndarray:
        # If result is in [0, 1] range but we need [0, 255], scale it
        if not result_in_0_255_range and (original_dtype == cp.uint8 or was_uint8_or_high_range):
            result = (result * 255.0).clip(0, 255)
        elif result_in_0_255_range:
            # Result is already in [0, 255], just clip
            result = result.clip(0, 255)

        # Convert to original dtype
        return result.astype(original_dtype)

    def compute_histogram_256_cupy(self, channel: cp.ndarray) -> cp.ndarray:
        counts = cp.bincount(channel, minlength=256).astype(cp.float32)
        return counts / (counts.sum() + 1e-8)

    def _normalize_to_channels_first_cupy(self, images: cp.ndarray) -> tuple[cp.ndarray, bool]:
        # Default implementation assumes channels-first
        return images, False

    def compute_reference_histograms_cupy(self, images: cp.ndarray) -> tuple[list, list, list, cp.ndarray]:
        # Normalize to channels-first format for processing
        images_normalized, _ = self._normalize_to_channels_first_cupy(images)
        images_uint8, _ = self.images_to_uint8_cupy(images_normalized)

        _N, C, _H, _W = images_uint8.shape

        ref_vals = []
        ref_cdf = []
        ref_histograms_256 = []

        for c in range(C):
            channel = images_uint8[:, c, :, :]
            flat_channel = channel.reshape(-1)

            hist_256 = self.compute_histogram_256_cupy(flat_channel)
            ref_histograms_256.append(hist_256)

            counts = cp.bincount(flat_channel, minlength=256).astype(cp.float32)
            nonzero_idx = cp.nonzero(counts)[0]
            if len(nonzero_idx) > 0:
                vals = nonzero_idx.astype(cp.float32)
                cnts = counts[nonzero_idx]
                cdf = cp.cumsum(cnts, axis=0)
                total = cdf[-1]
                cdf = cdf / (total + 1e-8)
            else:
                vals = cp.arange(256, dtype=cp.float32)
                cdf = cp.zeros(256, dtype=cp.float32)

            ref_vals.append(vals)
            ref_cdf.append(cdf)

        reference_histogram = ref_cdf[0]

        return ref_vals, ref_cdf, ref_histograms_256, reference_histogram


class HistogramMatchingCupy(CupyBackendBase):
    def __init__(self, device: str | cp.cuda.Device | None = None, channel_axis: int = 1):
        super().__init__(device)
        self.channel_axis = channel_axis

    def _normalize_to_channels_first_cupy(self, images: cp.ndarray) -> tuple[cp.ndarray, bool]:
        if self.channel_axis == -1 or (self.channel_axis == 3 and images.ndim == 4):
            # Channels-last format (N, H, W, C) -> (N, C, H, W)
            return cp.transpose(images, (0, 3, 1, 2)), True
        # Already channels-first (N, C, H, W)
        return images, False

    def _restore_format_cupy(self, images: cp.ndarray, needs_permute: bool) -> cp.ndarray:
        if needs_permute:
            # Convert back to channels-last (N, C, H, W) -> (N, H, W, C)
            return cp.transpose(images, (0, 2, 3, 1))
        return images

    def transform(self, images: cp.ndarray, reference_histogram: cp.ndarray | list) -> cp.ndarray:
        # Normalize to channels-first format for processing
        images_normalized, needs_permute = self._normalize_to_channels_first_cupy(images)

        per_channel_histograms = reference_histogram if isinstance(reference_histogram, list) else None

        original_dtype = images_normalized.dtype
        was_uint8_or_high_range = images_normalized.dtype == cp.uint8 or images_normalized.max() > 1.0

        images_uint8, needs_scale_back = self.images_to_uint8_cupy(images_normalized)

        N, C, H, W = images_uint8.shape
        matched_channels = []

        # Pre-compute reference values (same for all channels if using single reference histogram)
        ref_values = cp.arange(256, dtype=cp.float32)

        # Pre-compute reference CDFs for all channels if per-channel histograms provided
        precomputed_ref_cdfs = None
        precomputed_ref_cdf = None
        if per_channel_histograms is not None:
            precomputed_ref_cdfs = []
            for ref_hist in per_channel_histograms:
                ref_hist_norm = ref_hist.astype(cp.float32) / (ref_hist.astype(cp.float32).sum() + 1e-8)
                precomputed_ref_cdfs.append(cp.cumsum(ref_hist_norm, axis=0))
        elif reference_histogram.ndim == 1 and len(reference_histogram) == 256:
            ref_hist_norm = reference_histogram.astype(cp.float32) / (reference_histogram.astype(cp.float32).sum() + 1e-8)
            precomputed_ref_cdf = cp.cumsum(ref_hist_norm, axis=0)

        for c in range(C):
            channel = images_uint8[:, c, :, :]
            flat_channel = channel.reshape(-1)
            num_pixels = flat_channel.size

            # Build source histogram for all 256 values (much faster than unique)
            source_hist_counts = cp.bincount(flat_channel, minlength=256).astype(cp.float32)
            source_hist = source_hist_counts / (num_pixels + 1e-8)
            source_cdf = cp.cumsum(source_hist, axis=0)

            # Get reference quantiles for this channel
            if precomputed_ref_cdfs is not None and c < len(precomputed_ref_cdfs):
                ref_quantiles = precomputed_ref_cdfs[c]
            elif precomputed_ref_cdf is not None:
                ref_quantiles = precomputed_ref_cdf
            else:
                # Fallback: compute on the fly (shouldn't happen with proper input)
                if per_channel_histograms is not None and c < len(per_channel_histograms):
                    ref_hist = per_channel_histograms[c].astype(cp.float32)
                    ref_hist = ref_hist / (ref_hist.sum() + 1e-8)
                    ref_quantiles = cp.cumsum(ref_hist, axis=0)
                else:
                    ref_hist = reference_histogram.astype(cp.float32)
                    ref_hist = ref_hist / (ref_hist.sum() + 1e-8)
                    ref_quantiles = cp.cumsum(ref_hist, axis=0)

            # Build lookup table: for each of 256 possible source values, find matched value
            # Vectorized matching: process all source quantiles at once
            ref_quantiles_min = ref_quantiles[0]
            ref_quantiles_max = ref_quantiles[-1]

            # Vectorized searchsorted for all quantiles at once
            indices = cp.searchsorted(ref_quantiles, source_cdf, side="left")
            indices = cp.clip(indices, 1, len(ref_quantiles) - 1)

            # Vectorized interpolation
            quantile_left = ref_quantiles[indices - 1]
            quantile_right = ref_quantiles[indices]

            # Handle edge cases: values <= min or >= max
            below_min = source_cdf <= ref_quantiles_min
            above_max = source_cdf >= ref_quantiles_max

            # Compute alpha for interpolation (avoid division by zero)
            quantile_diff = quantile_right - quantile_left
            alpha = cp.where(quantile_diff > 1e-10, (source_cdf - quantile_left) / quantile_diff, cp.zeros_like(source_cdf))

            # Interpolate values to build lookup table
            lookup_table = ref_values[indices - 1] + alpha * (ref_values[indices] - ref_values[indices - 1])

            # Apply edge case handling
            lookup_table = cp.where(below_min, ref_values[0], lookup_table)
            lookup_table = cp.where(above_max, ref_values[-1], lookup_table)
            lookup_table = cp.clip(lookup_table, 0, 255)

            # Direct lookup: map all pixels at once using the lookup table
            # flat_channel is uint8, so values are already in [0, 255], just convert to int for indexing
            matched_channel = lookup_table[flat_channel.astype(cp.int32)].reshape(N, H, W)
            matched_channels.append(matched_channel)

        matched = cp.stack(matched_channels, axis=1)

        if needs_scale_back:
            matched = matched / 255.0
            result_in_0_255_range = False
        else:
            result_in_0_255_range = True

        matched = cp.clip(matched, 0.0, 1.0 if needs_scale_back else 255.0)

        matched = self.preserve_dtype_cupy(matched, original_dtype, was_uint8_or_high_range, result_in_0_255_range)

        # Restore to original format
        return self._restore_format_cupy(matched, needs_permute)


class ReinhardCupy(CupyBackendBase):
    def __init__(self, device: str | cp.cuda.Device | None = None):
        super().__init__(device)

    def compute_reference_mean_std_cupy(self, images: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        original_dtype = images.dtype
        was_uint8 = original_dtype == cp.uint8

        # Check range once and normalize inline
        images_float = images.astype(cp.float32) / 255.0 if was_uint8 or images.max() > 1.0 else images.astype(cp.float32)

        lab = self.rgb_to_lab_cupy(images_float, channel_axis=1)

        # Compute mean and std for all channels at once (vectorized)
        # Shape: (3,) - one value per channel
        reference_mean = lab.mean(axis=(0, 2, 3))
        reference_std = lab.std(axis=(0, 2, 3))

        return reference_mean, reference_std

    def transform(self, images: cp.ndarray, reference_mean: cp.ndarray, reference_std: cp.ndarray) -> cp.ndarray:
        original_dtype = images.dtype
        was_uint8 = original_dtype == cp.uint8

        # Check range once and normalize
        if was_uint8 or images.max() > 1.0:
            images_float = images.astype(cp.float32) / 255.0
            was_uint8_or_high_range = True
        else:
            images_float = images.astype(cp.float32)
            was_uint8_or_high_range = False

        # Prepare reference arrays
        if reference_mean.ndim == 1:
            reference_mean = reference_mean.reshape(1, 3, 1, 1)
        if reference_std.ndim == 1:
            reference_std = reference_std.reshape(1, 3, 1, 1)

        lab = self.rgb_to_lab_cupy(images_float, channel_axis=1)

        # Compute mean and std for all channels at once (vectorized)
        lab_mean = lab.mean(axis=(0, 2, 3), keepdims=True)
        lab_std = lab.std(axis=(0, 2, 3), keepdims=True)

        # Vectorized normalization across all channels
        lab_normalized = ((lab - lab_mean) / (lab_std + 1e-8)) * reference_std + reference_mean

        rgb_normalized = self.lab_to_rgb_cupy(lab_normalized, channel_axis=1)

        rgb_normalized = cp.clip(rgb_normalized, 0.0, 1.0)

        return self.preserve_dtype_cupy(rgb_normalized, original_dtype, was_uint8_or_high_range, result_in_0_255_range=False)


class MacenkoCupy(CupyBackendBase):
    def __init__(self, device: str | cp.cuda.Device | None = None):
        super().__init__(device)

    @staticmethod
    def _percentile_cupy(t: cp.ndarray, q: float) -> float:
        k = 1 + round(0.01 * float(q) * (t.size - 1))
        return float(cp.partition(t.flatten(), k - 1)[k - 1])

    def _process_single_image_cupy(self, od: cp.ndarray, stain_matrix: cp.ndarray, target_max_conc: cp.ndarray, beta: float, alpha: float, Io: float, H: int, W: int) -> cp.ndarray:
        # Reshape to (H*W, 3)
        od_reshaped = cp.transpose(od, (1, 2, 0)).reshape(-1, 3)

        # Filter by minimum OD
        od_min = od_reshaped.min(axis=1)
        mask = od_min >= beta
        od_filtered = od_reshaped[mask]

        # Early exit if too few pixels (safety check)
        if od_filtered.shape[0] < 3:
            od_filtered = od_reshaped

        # Compute covariance matrix efficiently
        od_filtered_T = od_filtered.T  # (3, num_filtered)
        od_mean = od_filtered_T.mean(axis=1, keepdims=True)
        od_centered = od_filtered_T - od_mean
        num_pixels = od_filtered.shape[0]
        cov = cp.dot(od_centered, od_centered.T) / (num_pixels - 1) if num_pixels > 1 else cp.zeros((3, 3), dtype=od_centered.dtype)

        _eigvals, eigvecs = cp.linalg.eigh(cov)
        eigvecs = eigvecs[:, [1, 2]]

        That = cp.dot(od_filtered, eigvecs)
        phi = cp.arctan2(That[:, 1], That[:, 0])

        # Compute percentiles
        min_phi = self._percentile_cupy(phi, alpha)
        max_phi = self._percentile_cupy(phi, 100 - alpha)

        # Pre-compute cos/sin values
        min_phi_t = cp.array(min_phi, dtype=cp.float32)
        max_phi_t = cp.array(max_phi, dtype=cp.float32)
        cos_min = cp.cos(min_phi_t)
        sin_min = cp.sin(min_phi_t)
        cos_max = cp.cos(max_phi_t)
        sin_max = cp.sin(max_phi_t)

        # Stack once
        angle_min = cp.stack((cos_min, sin_min))
        angle_max = cp.stack((cos_max, sin_max))

        vMin = cp.dot(eigvecs, angle_min).reshape(3, 1)
        vMax = cp.dot(eigvecs, angle_max).reshape(3, 1)

        HE_source = cp.where(vMin[0] > vMax[0], cp.concatenate((vMin, vMax), axis=1), cp.concatenate((vMax, vMin), axis=1))

        # Reshape OD for concentration computation
        od_all = od.reshape(3, -1)

        # Check condition number and matrix size to decide on fallback
        use_fallback = False
        # Use fallback for very large right-hand side matrices
        if od_all.shape[1] > 1000000:  # > 1M columns
            use_fallback = True

        if not use_fallback and HE_source.shape[0] >= HE_source.shape[1]:
            try:
                cond_num = cp.linalg.cond(HE_source)
                # Use fallback if condition number is too high (> 10)
                if cond_num > 10.0:
                    use_fallback = True
            except Exception:
                use_fallback = True

        if use_fallback:
            # Use pseudoinverse for ill-conditioned or large matrices (more robust)
            HE_pinv = cp.linalg.pinv(HE_source)
            concentrations = cp.dot(HE_pinv, od_all)
        else:
            try:
                concentrations = cp.linalg.lstsq(HE_source, od_all, rcond=None)[0]
            except Exception:
                # Fallback to pseudoinverse
                HE_pinv = cp.linalg.pinv(HE_source)
                concentrations = cp.dot(HE_pinv, od_all)

        # Compute max concentrations
        max_conc_0 = self._percentile_cupy(concentrations[0, :], 99)
        max_conc_1 = self._percentile_cupy(concentrations[1, :], 99)
        max_conc = cp.array([max_conc_0, max_conc_1], dtype=cp.float32)

        # Normalize concentrations
        norm_factor = target_max_conc / max_conc
        concentrations_norm = concentrations * norm_factor.reshape(-1, 1)

        # Reconstruct OD
        od_recon = cp.dot(stain_matrix, concentrations_norm)
        # Clamp negative OD values to 0 (OD must be >= 0)
        od_recon = cp.clip(od_recon, 0.0, float("inf"))

        # Convert back to RGB
        rgb_recon = Io * cp.exp(-od_recon)
        rgb_recon = cp.clip(rgb_recon, 0, 255)

        return rgb_recon.reshape(3, H, W)

    def compute_reference_stain_matrix_cupy(self, images: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        images_float = self.normalize_to_float_cupy(images)

        _N, _C, _H, _W = images_float.shape

        Io = 240.0
        # Vectorize: process all images at once
        od = -cp.log((images_float * 255.0 + 1.0) / Io)
        # Reshape to (N, 3, H*W) then combine: (3, N*H*W)
        od_combined = cp.transpose(od, (1, 0, 2, 3)).reshape(3, -1)

        # Transpose once: (N*H*W, 3)
        od_combined_T = od_combined.T

        # Filter by minimum OD
        od_min = od_combined_T.min(axis=1)
        mask = od_min >= 0.15
        od_for_cov = od_combined_T[mask]  # (num_pixels, 3)

        # Compute covariance matrix
        od_mean = od_for_cov.mean(axis=0, keepdims=True)
        od_centered = od_for_cov - od_mean
        cov = cp.dot(od_centered.T, od_centered) / (od_for_cov.shape[0] - 1)

        _eigvals, eigvecs = cp.linalg.eigh(cov)

        stain_vectors = eigvecs[:, [1, 2]]

        That = cp.dot(od_for_cov, stain_vectors)

        phi = cp.arctan2(That[:, 1], That[:, 0])

        alpha = 1.0
        min_phi = self._percentile_cupy(phi, alpha)
        max_phi = self._percentile_cupy(phi, 100 - alpha)

        cos_min = cp.cos(cp.array(min_phi))
        sin_min = cp.sin(cp.array(min_phi))
        cos_max = cp.cos(cp.array(max_phi))
        sin_max = cp.sin(cp.array(max_phi))

        vMin = cp.dot(stain_vectors, cp.stack((cos_min, sin_min))).reshape(3, 1)
        vMax = cp.dot(stain_vectors, cp.stack((cos_max, sin_max))).reshape(3, 1)

        HE = cp.where(vMin[0] > vMax[0], cp.concatenate((vMin, vMax), axis=1), cp.concatenate((vMax, vMin), axis=1))

        stain_matrix = HE

        # Check condition number and matrix size to decide on fallback
        use_fallback = False
        # Use fallback for very large right-hand side matrices
        if od_combined.shape[1] > 1000000:  # > 1M columns
            use_fallback = True

        if not use_fallback and HE.shape[0] >= HE.shape[1]:
            try:
                cond_num = cp.linalg.cond(HE)
                # Use fallback if condition number is too high (> 10)
                if cond_num > 10.0:
                    use_fallback = True
            except Exception:
                use_fallback = True

        if use_fallback:
            # Use pseudoinverse for ill-conditioned or large matrices (more robust)
            HE_pinv = cp.linalg.pinv(HE)
            concentrations = cp.dot(HE_pinv, od_combined)
        else:
            try:
                concentrations = cp.linalg.lstsq(HE, od_combined, rcond=None)[0]
            except Exception:
                # Fallback to pseudoinverse
                HE_pinv = cp.linalg.pinv(HE)
                concentrations = cp.dot(HE_pinv, od_combined)

        max_conc = cp.array([self._percentile_cupy(concentrations[0, :], 99), self._percentile_cupy(concentrations[1, :], 99)], dtype=cp.float32)

        return stain_matrix, max_conc

    def transform(self, images: cp.ndarray, stain_matrix: cp.ndarray, target_max_conc: cp.ndarray) -> cp.ndarray:
        stain_matrix = stain_matrix
        target_max_conc = target_max_conc

        original_dtype = images.dtype
        was_uint8_or_high_range = images.dtype == cp.uint8 or images.max() > 1.0

        images_float = self.normalize_to_float_cupy(images)

        if stain_matrix.shape != (3, 2):
            raise ValueError(f"stain_matrix must have shape (3, 2), got {stain_matrix.shape}")

        N, C, H, W = images_float.shape

        # Pre-compute constants
        Io = 240.0
        beta = 0.15
        alpha = 1.0

        # Flatten target_max_conc once
        if target_max_conc.ndim > 1:
            target_max_conc = target_max_conc.flatten()

        # Vectorize OD computation for all images at once
        od_all_images = -cp.log((images_float * 255.0 + 1.0) / Io)  # (N, 3, H, W)

        # Pre-allocate output array
        normalized = cp.empty((N, C, H, W), dtype=cp.float32)

        # Process each image - loop is necessary due to variable filtered pixel counts
        for n in range(N):
            od = od_all_images[n]  # (3, H, W)
            normalized[n] = self._process_single_image_cupy(od, stain_matrix, target_max_conc, beta, alpha, Io, H, W)

        return self.preserve_dtype_cupy(normalized, original_dtype, was_uint8_or_high_range, result_in_0_255_range=True)
