# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""Tests for ``StainNormalizerTransform`` and ``prepare_for_normalizer`` layout helpers."""

from __future__ import annotations

import pytest
import torch

from stainx import HistogramMatching, Macenko, StainNormalizerTransform
from stainx.utils import ChannelFormatConverter


class TestPrepareForNormalizer:
    def test_nhwc_batch_permute_keeps_n_and_device(self):
        device = torch.device("cpu")
        x = torch.randn(4, 32, 48, 3, device=device)
        out = ChannelFormatConverter(channel_axis=-1).prepare_for_normalizer(x)
        assert out.shape == (4, 3, 32, 48)
        assert out.device == x.device
        assert torch.allclose(out.permute(0, 2, 3, 1), x)

    def test_channels_first_passthrough_no_cpu_move(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        x = torch.randn(2, 3, 16, 16, device="cuda")
        out = ChannelFormatConverter(channel_axis=1).prepare_for_normalizer(x)
        assert out.data_ptr() == x.data_ptr()
        assert out.device.type == "cuda"

    def test_unknown_channel_axis_raises(self):
        with pytest.raises(ValueError, match="Unsupported channel_axis"):
            ChannelFormatConverter(channel_axis=0)


class TestStainNormalizerTransform:
    @pytest.fixture
    def reference(self):
        torch.manual_seed(0)
        return (torch.rand(1, 3, 64, 64) * 255).round().to(torch.uint8)

    @pytest.fixture
    def source(self):
        torch.manual_seed(1)
        return (torch.rand(2, 3, 64, 64) * 255).round().to(torch.uint8)

    def test_reference_mode_shape_device(self, reference, source):
        t = StainNormalizerTransform(method="reinhard", mode="reference", reference=reference, device="cpu")
        out = t(source)
        assert out.shape == source.shape
        assert out.device == source.device

    def test_default_device_follows_input(self, reference, source):
        t = StainNormalizerTransform(method="reinhard", mode="reference", reference=reference)
        out = t(source)
        assert out.device == source.device

    def test_default_device_follows_cuda_input(self, reference):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        torch.manual_seed(8)
        src = (torch.rand(2, 3, 64, 64, device="cuda") * 255).round().to(torch.uint8)
        ref = reference.to("cuda")
        t = StainNormalizerTransform(method="reinhard", mode="reference", reference=ref)
        out = t(src)
        assert out.device.type == "cuda"
        assert torch.device(t.normalizer.device).type == "cuda"

    def test_torch_cuda_backend_with_device_none_requires_cuda_input(self, reference):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

        if not CUDA_AVAILABLE:
            pytest.skip("torch_cuda extension unavailable")
        ref = reference.to("cuda")
        src = (torch.rand(1, 3, 64, 64, device="cuda") * 255).round().to(torch.uint8)
        t = StainNormalizerTransform(method="reinhard", mode="reference", reference=ref, backend="torch_cuda")
        out = t(src)
        assert out.device.type == "cuda"

    def test_normalize_to_0_1_rejected_for_reinhard(self, reference):
        with pytest.raises(ValueError, match="only applies to Macenko"):
            StainNormalizerTransform(method="reinhard", mode="reference", reference=reference, normalize_to_0_1=True)

    def test_single_image_roundtrip_rank(self, reference):
        torch.manual_seed(2)
        img = (torch.rand(3, 64, 64) * 255).round().to(torch.uint8)
        t = StainNormalizerTransform(method="reinhard", mode="reference", reference=reference, device="cpu")
        out = t(img)
        assert out.shape == img.shape

    def test_macenko_normalize_to_0_1_default(self):
        torch.manual_seed(3)
        src = torch.rand(2, 3, 64, 64)
        ref = torch.rand(1, 3, 64, 64)
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=ref, device="cpu")
        assert t.normalizer.normalize_to_0_1 is True
        out = t(src)
        assert out.dtype.is_floating_point
        assert float(out.amin()) >= -1e-5
        assert float(out.amax()) <= 1.0 + 1e-5
        assert float(out.mean()) < 1.0

    def test_macenko_without_flag_stays_0_255_for_unit_float(self):
        torch.manual_seed(4)
        src = torch.rand(2, 3, 64, 64)
        ref = torch.rand(1, 3, 64, 64)
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=ref, device="cpu", normalize_to_0_1=False)
        out = t(src)
        assert float(out.amax()) > 1.0

    def test_float_jitter_above_one_not_treated_as_255(self):
        """ColorJitter can push float >1; dtype gate must not silently /255."""
        torch.manual_seed(9)
        ref = torch.rand(1, 3, 64, 64)
        src = (torch.rand(2, 3, 64, 64) * 1.3).clamp(0.0, 1.5)
        assert float(src.amax()) > 1.0
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=ref, device="cpu")
        out = t(src)
        # Old max()>1 path would /255 → near-black mean.
        assert float(out.mean()) > 0.05
        assert float(out.amax()) <= 1.0 + 1e-4

    def test_normalize_to_0_1_matches_explicit_macenko(self):
        torch.manual_seed(5)
        src = torch.rand(2, 3, 64, 64)
        ref = torch.rand(1, 3, 64, 64)
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=ref, device="cpu")
        n = Macenko(device="cpu", normalize_to_0_1=True)
        n.fit(ref)
        # Independent fits + platform BLAS (esp. Windows) can differ by ~1e-5..1e-4
        assert torch.allclose(t(src), n.transform(src), rtol=0, atol=1e-4)

    def test_prebuilt_normalize_flag_can_clear(self, reference):
        n = Macenko(device="cpu", normalize_to_0_1=True)
        n.fit(reference.float() / 255.0)
        t = StainNormalizerTransform(mode="reference", normalizer=n, device="cpu", normalize_to_0_1=False)
        assert t.normalizer.normalize_to_0_1 is False

    def test_torch_cuda_plus_cpu_device_rejected(self, reference):
        with pytest.raises(ValueError, match="requires a CUDA device"):
            StainNormalizerTransform(method="reinhard", mode="reference", reference=reference, backend="torch_cuda", device="cpu")

    def test_device_none_fit_cpu_forward_cuda(self, reference):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        t = StainNormalizerTransform(method="reinhard", mode="reference", reference=reference)
        assert torch.device(t.normalizer.device).type == "cpu"
        src = (torch.rand(2, 3, 64, 64, device="cuda") * 255).round().to(torch.uint8)
        out = t(src)
        assert out.device.type == "cuda"
        assert torch.device(t.normalizer.device).type == "cuda"

    def test_macenko_normalize_to_0_1_torch_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

        if not CUDA_AVAILABLE:
            pytest.skip("torch_cuda extension unavailable")
        torch.manual_seed(10)
        ref = torch.rand(1, 3, 64, 64, device="cuda")
        src = torch.rand(2, 3, 64, 64, device="cuda")
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=ref, backend="torch_cuda", device="cuda")
        out = t(src)
        assert out.device.type == "cuda"
        assert float(out.amax()) <= 1.0 + 1e-4
        assert float(out.mean()) > 0.05

    def test_batch_mode_refits(self, source):
        t = StainNormalizerTransform(method="reinhard", mode="batch", device="cpu", batch_ref_index=0)
        out = t(source)
        assert out.shape == source.shape
        assert t.normalizer._is_fitted

    def test_hm_channels_last_no_double_permute(self):
        torch.manual_seed(6)
        ref = (torch.rand(1, 32, 32, 3) * 255).round().to(torch.uint8)
        src = (torch.rand(2, 32, 32, 3) * 255).round().to(torch.uint8)
        t = StainNormalizerTransform(method="histogram_matching", mode="reference", reference=ref, device="cpu", channel_axis=-1)
        out = t(src)
        assert out.shape == src.shape

    def test_prebuilt_hm_syncs_channel_axis_from_normalizer(self):
        """Default transform channel_axis=1 must not desync from prebuilt NHWC HM."""
        torch.manual_seed(11)
        ref = (torch.rand(1, 32, 32, 3) * 255).round().to(torch.uint8)
        src = (torch.rand(2, 32, 32, 3) * 255).round().to(torch.uint8)
        n = HistogramMatching(device="cpu", channel_axis=-1)
        n.fit(ref)
        t = StainNormalizerTransform(mode="reference", normalizer=n, device="cpu")
        assert t.channel_axis == -1
        out = t(src)
        assert out.shape == src.shape

    def test_prebuilt_hm_rejects_conflicting_channel_axis(self):
        n = HistogramMatching(device="cpu", channel_axis=1)
        with pytest.raises(ValueError, match="conflicts with prebuilt"):
            StainNormalizerTransform(mode="reference", normalizer=n, device="cpu", channel_axis=-1, reference=(torch.rand(1, 3, 8, 8) * 255).round().to(torch.uint8))

    def test_macenko_rejects_nhwc_channel_axis(self, reference):
        with pytest.raises(ValueError, match="only supported for histogram_matching"):
            StainNormalizerTransform(method="macenko", mode="reference", reference=reference, channel_axis=-1)

    def test_macenko_rejects_nhwc_tensor(self, reference):
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=reference, device="cpu")
        nhwc = (torch.rand(2, 64, 64, 3) * 255).round().to(torch.uint8)
        with pytest.raises(ValueError, match="Expected NCHW"):
            t(nhwc)

    def test_prebuilt_normalizer_honors_normalize_flag(self, reference, source):
        n = Macenko(device="cpu", normalize_to_0_1=False)
        n.fit(reference.float() / 255.0)
        t = StainNormalizerTransform(mode="reference", normalizer=n, device="cpu", normalize_to_0_1=True)
        assert t.normalizer.normalize_to_0_1 is True
        out = t(source.float() / 255.0)
        assert float(out.amax()) <= 1.0 + 1e-5

    def test_state_dict_does_not_include_stain_matrix(self, reference):
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=reference, device="cpu")
        keys = t.state_dict().keys()
        assert not any("stain" in k or "max_conc" in k for k in keys)
