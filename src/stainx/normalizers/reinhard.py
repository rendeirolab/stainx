# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import torch

from stainx.normalizers._template import NormalizerTemplate


class Reinhard(NormalizerTemplate):
    def _init_algorithm_attributes(self):
        self._reference_mean = None
        self._reference_std = None

    def _get_cuda_class(self):
        from stainx.backends.cuda_backend import ReinhardCUDA

        return ReinhardCUDA

    def _get_pytorch_class(self):
        from stainx.backends.torch_backend import ReinhardPyTorch

        return ReinhardPyTorch

    def _compute_reference_params(self, images: torch.Tensor) -> None:
        # Automatically use CUDA backend if available, otherwise fall back to PyTorch
        backend = self._get_backend_for_computation()
        self._reference_mean, self._reference_std = backend.compute_reference_mean_std(images)

    def _get_reference_params(self) -> tuple:
        return (self._reference_mean, self._reference_std)
