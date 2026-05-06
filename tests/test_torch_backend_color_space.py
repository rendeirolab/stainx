import torch

from stainx.backends.torch_backend import TorchBackendBase


def test_rgb_lab_roundtrip_close():
    # Small tensor to keep test fast; values in [0, 1]
    rgb = torch.rand(1, 3, 8, 8)
    lab = TorchBackendBase.rgb_to_lab_torch(rgb, channel_axis=1)
    rgb2 = TorchBackendBase.lab_to_rgb_torch(lab, channel_axis=1)

    assert rgb2.shape == rgb.shape
    # Conversion is lossy; just require approximate reconstruction
    assert torch.allclose(rgb2, rgb.clamp(0, 1), atol=3e-2, rtol=0)


def test_rgb_to_lab_accepts_uint8_range_and_returns_float():
    rgb_u8 = (torch.rand(1, 3, 8, 8) * 255).round().to(torch.uint8)
    lab = TorchBackendBase.rgb_to_lab_torch(rgb_u8, channel_axis=1)
    assert lab.dtype.is_floating_point
    assert lab.shape == rgb_u8.shape
