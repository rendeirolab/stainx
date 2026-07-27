# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""Correctness vs external baselines for each normalizer / backend.

Baselines
---------
- Reinhard / Macenko → ``torchstain``
- HistogramMatching → ``skimage.exposure.match_histograms``

Variants under test: ``backend="torch"`` and ``backend="torch_cuda"`` (skipped if the
CUDA extension is unavailable).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from skimage.exposure import match_histograms
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE as TORCH_CUDA_AVAILABLE
from stainx.utils import ChannelFormatConverter

# Image tensors are ~[0, 255]. Reinhard / HM: at most one grey level vs baselines.
# Macenko needs a well-defined stain plane: pure RGB noise makes the OD covariance
# near-isotropic, so the leading eigenspace (and H/E split) is eigensolver-dependent
# and parity vs torchstain is ill-posed. Macenko fixtures are therefore synthesized
# from the Beer-Lambert model with known HE vectors (Reinhard / HM stay on noise).
RTOL = 0.0
ATOL = 1.0
MACENKO_ATOL = 2.0
MACENKO_MAE = 0.35

# torchstain default HERef (columns = hematoxylin, eosin).
_HE_REF = torch.tensor([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=torch.float32)
_IO = 240.0


def _synthetic_he_tile(h: int, w: int, seed: int, he_scale: float = 1.0) -> torch.Tensor:
    """NCHW uint8 tile from Beer-Lambert: ``I = Io * exp(-(HE @ C))``."""
    g = torch.Generator().manual_seed(seed)
    # Low-frequency concentration maps (upsampled noise) → spatially coherent stain.
    gh, gw = max(h // 8, 1), max(w // 8, 1)
    c_h = F.interpolate(torch.rand(1, 1, gh, gw, generator=g), size=(h, w), mode="bilinear", align_corners=False).squeeze()
    c_e = F.interpolate(torch.rand(1, 1, gh, gw, generator=g), size=(h, w), mode="bilinear", align_corners=False).squeeze()
    concentrations = torch.stack([0.3 + 1.8 * c_h, 0.2 + 1.0 * c_e], dim=0)  # (2, H, W)
    od = torch.einsum("cs,shp->chp", _HE_REF * he_scale, concentrations)
    return (_IO * torch.exp(-od)).clamp(0, 255).round().to(torch.uint8).unsqueeze(0)


def _macenko_pair(image_hw: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference + source synthetic H&E tiles at ``image_hw`` (different stain strength)."""
    h, w = image_hw
    return _synthetic_he_tile(h, w, seed=42, he_scale=1.0).to(device), _synthetic_he_tile(h, w, seed=123, he_scale=1.15).to(device)


def _backend_params():
    yield pytest.param("torch", id="torch")
    if torch.cuda.is_available() and TORCH_CUDA_AVAILABLE:
        yield pytest.param("torch_cuda", id="torch_cuda")
    else:
        yield pytest.param("torch_cuda", marks=pytest.mark.skip(reason="torch_cuda extension unavailable"), id="torch_cuda")


@pytest.fixture(params=list(_backend_params()))
def backend(request):
    return request.param


@pytest.fixture
def device(backend):
    if backend == "torch_cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _hm_inputs(reference_image: torch.Tensor, source_image: torch.Tensor, channel_axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inputs in the layout expected by HistogramMatching for ``channel_axis``."""
    if channel_axis in (-1, 3):
        # True NHWC — pass through as-is (HistogramMatching honors channel_axis).
        # Do not NCHW-convert here unless the normalizer also uses channel_axis=1.
        return reference_image.permute(0, 2, 3, 1).contiguous(), source_image.permute(0, 2, 3, 1).contiguous()
    return reference_image, source_image


def _hm_to_chw(result: torch.Tensor, channel_axis: int) -> torch.Tensor:
    if channel_axis in (-1, 3):
        return result.squeeze(0).permute(2, 0, 1).float()
    return result.squeeze(0).float()


class TestAgainstBaselines:
    @pytest.fixture(params=[(64, 64), (128, 128), (256, 256), (256, 512), (321, 199), (384, 256), (480, 640), (512, 512), (1024, 1024), (2048, 2048)])
    def image_hw(self, request):
        return request.param

    @pytest.fixture
    def reference_image(self, device, image_hw):
        h, w = image_hw
        torch.manual_seed(42)
        return (torch.rand(1, 3, h, w, device=device) * 255).round().to(torch.uint8)

    @pytest.fixture
    def source_image(self, device, image_hw):
        h, w = image_hw
        torch.manual_seed(123)
        return (torch.rand(1, 3, h, w, device=device) * 255).round().to(torch.uint8)

    def test_reinhard_vs_torchstain(self, reference_image, source_image, device, backend, image_hw):
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image.squeeze(0).cpu()

        baseline = TorchReinhardNormalizer()
        baseline.fit(ref_chw)
        baseline_tensor = baseline.normalize(src_chw).permute(2, 0, 1).float()

        normalizer = Reinhard(device=device, backend=backend)
        normalizer.fit(reference_image)
        result = normalizer.transform(source_image).squeeze(0).cpu().float()

        assert torch.allclose(result, baseline_tensor, rtol=RTOL, atol=ATOL), f"Reinhard mismatch vs torchstain (backend={backend}, hw={image_hw})"

    @pytest.mark.parametrize("precision", ["stable", "fast"])
    def test_macenko_vs_torchstain(self, device, backend, image_hw, precision):
        # Synthetic Beer-Lambert H&E: Macenko needs a well-defined stain plane (see module note).
        if precision == "fast" and backend != "torch_cuda":
            pytest.skip("fast precision only applies to torch_cuda backend")

        reference_image, source_image = _macenko_pair(image_hw, device)
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image.squeeze(0).cpu()

        baseline = TorchMacenkoNormalizer()
        baseline.fit(ref_chw)
        baseline_rgb, _, _ = baseline.normalize(src_chw, stains=True)
        baseline_tensor = baseline_rgb.permute(2, 0, 1).float()

        fit_normalizer = Macenko(device=torch.device("cpu"), backend="torch")
        fit_normalizer.fit(reference_image.cpu())

        normalizer = Macenko(device=device, backend=backend, precision=precision)
        normalizer._stain_matrix = fit_normalizer._stain_matrix.to(device)
        normalizer._target_max_conc = fit_normalizer._target_max_conc.to(device)
        normalizer._is_fitted = True
        result = normalizer.transform(source_image).squeeze(0).cpu().float()

        assert torch.allclose(fit_normalizer._stain_matrix.cpu().float(), baseline.HERef.float(), rtol=1e-4, atol=1e-5), f"Macenko HE mismatch (backend={backend}, hw={image_hw})"
        assert torch.allclose(fit_normalizer._target_max_conc.cpu().float().flatten(), baseline.maxCRef.float().flatten(), rtol=1e-3, atol=1e-4), f"Macenko maxC mismatch (backend={backend}, hw={image_hw})"
        assert torch.allclose(result, baseline_tensor, rtol=RTOL, atol=MACENKO_ATOL), f"Macenko mismatch vs torchstain (backend={backend}, precision={precision}, hw={image_hw})"
        mae = (result - baseline_tensor).abs().mean().item()
        assert mae <= MACENKO_MAE, f"Macenko MAE {mae:.4f} > {MACENKO_MAE} (backend={backend}, precision={precision}, hw={image_hw})"
        assert result.max().item() <= 255.0 + MACENKO_ATOL
        if baseline_tensor.max().item() > 240.0:
            assert result.max().item() > 240.0, f"Macenko incorrectly capped at Io (backend={backend}, precision={precision}, hw={image_hw})"

    def test_macenko_stable_fast_not_identical(self, image_hw):
        """Fast and stable must produce different results (fp16 rounding)."""
        if not torch.cuda.is_available() or not TORCH_CUDA_AVAILABLE:
            pytest.skip("torch_cuda extension unavailable")
        cuda_device = torch.device("cuda")
        reference_image, source_image = _macenko_pair(image_hw, cuda_device)

        fit_normalizer = Macenko(device=torch.device("cpu"), backend="torch")
        fit_normalizer.fit(reference_image.cpu())

        normalizer_stable = Macenko(device=cuda_device, backend="torch_cuda", precision="stable")
        normalizer_stable._stain_matrix = fit_normalizer._stain_matrix.to(cuda_device)
        normalizer_stable._target_max_conc = fit_normalizer._target_max_conc.to(cuda_device)
        normalizer_stable._is_fitted = True
        result_stable = normalizer_stable.transform(source_image)

        normalizer_fast = Macenko(device=cuda_device, backend="torch_cuda", precision="fast")
        normalizer_fast._stain_matrix = fit_normalizer._stain_matrix.to(cuda_device)
        normalizer_fast._target_max_conc = fit_normalizer._target_max_conc.to(cuda_device)
        normalizer_fast._is_fitted = True
        result_fast = normalizer_fast.transform(source_image)

        # Both must pass correctness.
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image.squeeze(0).cpu()
        baseline = TorchMacenkoNormalizer()
        baseline.fit(ref_chw)
        baseline_rgb, _, _ = baseline.normalize(src_chw, stains=True)
        baseline_tensor = baseline_rgb.permute(2, 0, 1).float()

        for label, res in [("stable", result_stable), ("fast", result_fast)]:
            r = res.squeeze(0).cpu().float()
            mae = (r - baseline_tensor).abs().mean().item()
            assert torch.allclose(r, baseline_tensor, rtol=RTOL, atol=MACENKO_ATOL), f"{label} mismatch (hw={image_hw})"
            assert mae <= MACENKO_MAE, f"{label} MAE {mae:.4f} (hw={image_hw})"

        # Fast must differ from stable (fp16 rounding guarantees this on >= 256² tiles).
        if image_hw[0] >= 128 and image_hw[1] >= 128:
            assert not torch.equal(result_stable, result_fast), f"fast and stable results are bitwise identical — fast path may not be using fp16 (hw={image_hw})"

    def test_macenko_precision_validation(self):
        """Precision validation and backend compatibility checks."""
        with pytest.raises(ValueError, match="precision='fast' requires backend='torch_cuda'"):
            Macenko(backend="torch", precision="fast")

        # precision='fast' with auto-selected torch backend (no CUDA) should also raise.
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="precision='fast' requires backend='torch_cuda'"):
                Macenko(precision="fast")

        # Invalid precision value.
        with pytest.raises(ValueError, match="precision must be"):
            Macenko(precision="ultra")

    @pytest.mark.parametrize("channel_axis", [1, -1, 3, -3])
    def test_histogram_matching_vs_skimage(self, reference_image, source_image, device, backend, channel_axis, image_hw):
        converter = ChannelFormatConverter(channel_axis=1)
        skimage_chw = torch.from_numpy(match_histograms(converter.to_hwc(source_image, squeeze_batch=True), converter.to_hwc(reference_image, squeeze_batch=True), channel_axis=-1)).float().permute(2, 0, 1)

        ref_input, src_input = _hm_inputs(reference_image, source_image, channel_axis)
        normalizer = HistogramMatching(device=device, backend=backend, channel_axis=channel_axis)
        normalizer.fit(ref_input)
        result_chw = _hm_to_chw(normalizer.transform(src_input), channel_axis).cpu()

        assert torch.allclose(result_chw, skimage_chw, rtol=RTOL, atol=ATOL), f"HM mismatch vs skimage (backend={backend}, channel_axis={channel_axis}, hw={image_hw})"
