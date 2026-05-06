import numpy as np
import torch

from stainx.utils import ChannelFormatConverter, get_device


def test_get_device_default_returns_torch_device():
    d = get_device(None)
    assert isinstance(d, torch.device)


def test_get_device_string_cpu_returns_torch_device_cpu():
    d = get_device("cpu")
    assert isinstance(d, torch.device)
    assert d.type == "cpu"


def test_channel_format_converter_channels_first_to_hwc_numpy():
    conv = ChannelFormatConverter(channel_axis=1)
    chw = np.random.rand(3, 4, 5).astype(np.float32)
    hwc = conv.to_hwc(chw)
    assert hwc.shape == (4, 5, 3)


def test_channel_format_converter_prepare_for_normalizer_channels_last_to_channels_first_torch():
    conv = ChannelFormatConverter(channel_axis=-1)
    # (N, H, W, C) -> (N, C, H, W)
    images = torch.rand(1, 4, 5, 3)
    out = conv.prepare_for_normalizer(images)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 4, 5)


def test_channel_format_converter_to_chw_channels_last_torch_returns_chw():
    conv = ChannelFormatConverter(channel_axis=-1)
    # Pretend this is a normalizer output in channels-last format
    out_hwc = torch.rand(1, 4, 5, 3)
    chw = conv.to_chw(out_hwc, squeeze_batch=True, return_torch=True)
    assert isinstance(chw, torch.Tensor)
    assert chw.shape == (3, 4, 5)
