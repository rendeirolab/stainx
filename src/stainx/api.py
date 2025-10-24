import warnings

from .base import StainBase
from .utils import get_torch_device


class HistogramMatching(StainBase):
    def __init__(self, device=None):
        super().__init__()

        self.device = get_torch_device(device)
        if self.device == "cuda":
            try:
                from stainx_cuda import HistogramMatchingCUDA
            except (ImportError, ModuleNotFoundError):
                warnings.warn(
                    "GPU detected but stainx-cuda is not installed."
                    "Falling back to pure torch implementation."
                )
            self.model = HistogramMatchingCUDA()
        else:
            from .normalizer import TorchHistogramMatching

            self.model = TorchHistogramMatching()

    def fit(self, images):
        self.model.fit(images)

    def transform(self, images):
        return self.model.transform(images)
