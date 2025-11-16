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

    def _get_cuda_class(self):
        from stainx.backends.cuda_backend import MacenkoCUDA

        return MacenkoCUDA

    def _get_pytorch_class(self):
        from stainx.backends.torch_backend import MacenkoPyTorch

        return MacenkoPyTorch

    def _compute_reference_params(self, images: torch.Tensor) -> None:
        # Use backend to compute reference stain matrix
        pytorch_class = self._get_pytorch_class()
        backend = pytorch_class(self.device)
        self._stain_matrix, self._target_max_conc = backend.compute_reference_stain_matrix(images)
        self._concentration_matrix = None

    def _get_reference_params(self) -> tuple:
        return (self._stain_matrix, self._target_max_conc)
