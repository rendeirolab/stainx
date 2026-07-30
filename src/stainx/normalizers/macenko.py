# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
from typing import Any

from stainx.normalizers._template import NormalizerTemplate


class Macenko(NormalizerTemplate):
    """Macenko stain normalization.

    ``normalize_to_0_1`` defaults to ``False`` here (output ~``[0, 255]``).
    ``StainNormalizerTransform(method="macenko")`` defaults it to ``True`` for
    float ``[0, 1]`` training pipelines — prefer the transform there, or set the
    flag explicitly on this class.

    Parameters
    ----------
    device : str or torch.device, optional
        Device for computation.
    backend : str, optional
        Backend name (``"torch"`` or ``"torch_cuda"``).  Auto-selects when None.
    normalize_to_0_1 : bool, optional
        If True, divide output by 255.0 so the result is in [0, 1].
    precision : str, optional
        Numerical precision mode for the CUDA backend only.
        ``"stable"`` (default) — fp64 cov/eigh + fp32 pixels.
        ``"fast"`` — fp32 cov/eigh + fp16 large pixel tensors (projection
        GEMM, phi sort, reconstruct matmul); 2x2 solve stays fp32.
        Raises ``ValueError`` when set to ``"fast"`` with ``backend="torch"``.
    """

    def __init__(self, device: Any | None = None, backend: str | None = None, normalize_to_0_1: bool = False, precision: str = "stable"):
        if precision not in ("stable", "fast"):
            raise ValueError(f"precision must be 'stable' or 'fast', got {precision!r}")
        self._precision = precision
        self.normalize_to_0_1 = normalize_to_0_1
        super().__init__(device=device, backend=backend)
        # Validate precision/backend compatibility eagerly so the user gets
        # a clear error at construction time, not lazily during transform().
        if self._precision == "fast" and self.backend != "torch_cuda":
            raise ValueError(f"precision='fast' requires backend='torch_cuda', but backend is '{self.backend}'. Either set backend='torch_cuda' or use precision='stable'.")

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

    def _compute_reference_params(self, images: Any) -> None:
        backend = self._get_backend_for_computation_torch()
        self._stain_matrix, self._target_max_conc = backend.compute_reference_stain_matrix_torch(images)
        self._concentration_matrix = None

    def _get_backend_kwargs(self) -> dict:
        kwargs = super()._get_backend_kwargs()
        if self._precision != "stable":
            kwargs["precision"] = self._precision
        return kwargs

    def _get_reference_params(self) -> tuple:
        return (self._stain_matrix, self._target_max_conc)
