# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
import torch

from stainx.normalizers._template import NormalizerTemplate


class Macenko(NormalizerTemplate):
    def _init_algorithm_attributes(self):
        self._stain_matrix = None
        self._concentration_matrix = None
        self._target_max_conc = None

    def _get_torch_cuda_class(self):
        from stainx.backends.torch_cuda_backend import MacenkoCUDA

        return MacenkoCUDA

    def _get_torch_class(self):
        from stainx.backends.torch_backend import MacenkoTorch

        return MacenkoTorch

    def _compute_reference_params(self, images: torch.Tensor) -> None:
        # Automatically use CUDA backend if available, otherwise fall back to Torch
        backend = self._get_backend_for_computation_torch()
        self._stain_matrix, self._target_max_conc = backend.compute_reference_stain_matrix_torch(images)
        self._concentration_matrix = None

    def _get_reference_params(self) -> tuple:
        return (self._stain_matrix, self._target_max_conc)
