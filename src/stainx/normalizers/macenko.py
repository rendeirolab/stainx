# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from typing import Any

from stainx.normalizers._template import NormalizerTemplate


class Macenko(NormalizerTemplate):
    def __init__(self, device: Any | None = None, backend: str | None = None, normalize_to_0_1: bool = False):
        self.normalize_to_0_1 = normalize_to_0_1
        super().__init__(device=device, backend=backend)

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

    def _get_cupy_class(self):
        from stainx.backends.cupy_backend import MacenkoCupy

        return MacenkoCupy

    def _get_cupy_cuda_class(self):
        from stainx.backends.cupy_cuda_backend import MacenkoCuPyCUDA

        return MacenkoCuPyCUDA

    def _compute_reference_params(self, images: Any) -> None:
        from stainx.normalizers import _template

        if _template.cp is not None and isinstance(images, _template.cp.ndarray):
            backend = self._get_backend_for_computation_cupy()
            self._stain_matrix, self._target_max_conc = backend.compute_reference_stain_matrix_cupy(images)
        else:
            backend = self._get_backend_for_computation_torch()
            self._stain_matrix, self._target_max_conc = backend.compute_reference_stain_matrix_torch(images)
        self._concentration_matrix = None

    def _get_reference_params(self) -> tuple:
        return (self._stain_matrix, self._target_max_conc)
