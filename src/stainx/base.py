from abc import ABC, abstractmethod

import torch.nn as nn


class StainBase(ABC, nn.Module):
    @abstractmethod
    def fit(self, images):
        pass

    @abstractmethod
    def transform(self, images):
        pass

    @classmethod
    def from_pretrained(cls, path):
        """Load from huggingface"""
        pass

    def forward(self, images):
        return self.transform(images)

    def fit_transform(self, images):
        self.fit(images)
        return self.transform(images)
