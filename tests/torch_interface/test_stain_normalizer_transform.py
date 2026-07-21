# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""Tests for ``StainNormalizerTransform`` and ``prepare_for_normalizer`` layout helpers."""

from __future__ import annotations

import pytest
import torch

from stainx import Macenko, StainNormalizerTransform
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

    def test_single_image_roundtrip_rank(self, reference):
        torch.manual_seed(2)
        img = (torch.rand(3, 64, 64) * 255).round().to(torch.uint8)
        t = StainNormalizerTransform(method="reinhard", mode="reference", reference=reference, device="cpu")
        out = t(img)
        assert out.shape == img.shape

    def test_macenko_normalize_to_0_1(self, reference):
        torch.manual_seed(3)
        src = torch.rand(2, 3, 64, 64)
        ref = torch.rand(1, 3, 64, 64)
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=ref, device="cpu", normalize_to_0_1=True)
        out = t(src)
        assert out.dtype.is_floating_point
        assert float(out.amax()) <= 1.0 + 1e-5

    def test_macenko_auto_scales_unit_float_without_flag(self, reference):
        torch.manual_seed(4)
        src = torch.rand(2, 3, 64, 64)
        ref = torch.rand(1, 3, 64, 64)
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=ref, device="cpu", normalize_to_0_1=False)
        out = t(src)
        assert float(out.amax()) <= 1.0 + 1e-5

    def test_batch_mode_refits(self, source):
        t = StainNormalizerTransform(method="reinhard", mode="batch", device="cpu", batch_ref_index=0)
        out = t(source)
        assert out.shape == source.shape
        assert t.normalizer._is_fitted

    def test_hm_channels_last_no_double_permute(self):
        torch.manual_seed(5)
        ref = (torch.rand(1, 32, 32, 3) * 255).round().to(torch.uint8)
        src = (torch.rand(2, 32, 32, 3) * 255).round().to(torch.uint8)
        t = StainNormalizerTransform(method="histogram_matching", mode="reference", reference=ref, device="cpu", channel_axis=-1)
        out = t(src)
        assert out.shape == src.shape

    def test_prebuilt_normalizer(self, reference, source):
        n = Macenko(device="cpu")
        n.fit(reference)
        t = StainNormalizerTransform(mode="reference", normalizer=n, device="cpu")
        out = t(source)
        assert out.shape == source.shape

    def test_state_dict_does_not_include_stain_matrix(self, reference):
        t = StainNormalizerTransform(method="macenko", mode="reference", reference=reference, device="cpu")
        keys = t.state_dict().keys()
        assert not any("stain" in k or "max_conc" in k for k in keys)
