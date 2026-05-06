import torch
import pytest

from stainx.normalizers.reinhard import Reinhard


def test_transform_requires_fit():
    n = Reinhard(backend="torch", device="cpu")
    x = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="fit"):
        _ = n.transform(x)


def test_fit_transform_runs_and_preserves_shape():
    n = Reinhard(backend="torch", device="cpu")
    x = (torch.rand(1, 3, 8, 8) * 255).round().to(torch.uint8)
    y = n.fit_transform(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
