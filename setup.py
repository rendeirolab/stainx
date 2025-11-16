# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""Build CUDA extension for stainx if CUDA is available."""

import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if not torch.cuda.is_available():
    print("CUDA not available. Skipping CUDA extension build.")
    sys.exit(0)

print("Building CUDA extension...")

project_root = Path(__file__).parent
csrc_dir = project_root / "src" / "stainx_cuda" / "csrc"

extensions = [CUDAExtension(name="stainx_cuda", sources=[str(csrc_dir / "bindings.cpp"), str(csrc_dir / "histogram_matching.cu"), str(csrc_dir / "reinhard.cu"), str(csrc_dir / "macenko.cu")], include_dirs=[str(csrc_dir)], extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": ["-O3", "--use_fast_math"]})]

setup(name="stainx-cuda", ext_modules=extensions, cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)}, zip_safe=False)
