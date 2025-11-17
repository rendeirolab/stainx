# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""Build CUDA extension for stainx if CUDA is available."""

import contextlib
import os
import re
import sys
from pathlib import Path

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if not torch.cuda.is_available():
    print("CUDA not available. Skipping CUDA extension build.")
    sys.exit(0)

print("Building CUDA extension...")

project_root = Path(__file__).parent
csrc_dir = project_root / "src" / "stainx_cuda" / "csrc"

# Get current device info
device = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(device)
device_capability = torch.cuda.get_device_capability(device)
print(f"Current device: {device_name}")
print(f"Current CUDA capability: {device_capability}")


# Detect compute capability
def detect_cc():
    dev = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(dev)
    return major * 10 + minor


cc = detect_cc()
print(f"Detected compute capability: {cc}")


# Remove unwanted PyTorch NVCC flags
def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]
    for flag in REMOVE_NVCC_FLAGS:
        with contextlib.suppress(ValueError):
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)


# Get CUDA architecture flags
def get_cuda_arch_flags():
    flags = ["--expt-relaxed-constexpr", "--use_fast_math", "-std=c++17", "-O3", "-DNDEBUG", "-Xcompiler", "-funroll-loops", "-Xcompiler", "-ffast-math", "-Xcompiler", "-finline-functions"]

    # Add architecture-specific flags
    # Support for newer architectures (compute capability >= 10.0)
    if cc >= 100:
        # Add support for the detected architecture
        compute_arch = f"compute_{cc // 10}{cc % 10}"
        sm_arch = f"sm_{cc // 10}{cc % 10}"
        flags.extend(["-gencode", f"arch={compute_arch},code={sm_arch}"])

        # Also add support for common newer architectures (for compatibility)
        # Only add if different from detected arch to avoid duplicates
        if cc >= 120:
            if cc != 120:
                flags.extend(["-gencode", "arch=compute_120,code=sm_120"])
            if cc >= 100 and cc != 100:
                flags.extend(["-gencode", "arch=compute_100,code=sm_100"])
        elif cc >= 100:
            if cc != 100:
                flags.extend(["-gencode", "arch=compute_100,code=sm_100"])
    else:
        # For older architectures, add common ones
        if cc >= 75:
            flags.extend(["-gencode", "arch=compute_75,code=sm_75"])
        if cc >= 80:
            flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
        if cc >= 86:
            flags.extend(["-gencode", "arch=compute_86,code=sm_86"])

    return flags


# Check PyTorch version
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")
m = re.match(r"^(\d+)\.(\d+)", torch_version)
if not m:
    raise RuntimeError(f"Cannot parse PyTorch version '{torch_version}'")
major, minor = map(int, m.groups())
if major < 2 or (major == 2 and minor < 0):
    print(f"Warning: PyTorch version {torch_version} may not be fully supported. Recommended: >= 2.0")

# Remove unwanted flags
remove_unwanted_pytorch_nvcc_flags()

# Get CUDA architecture flags
nvcc_flags = get_cuda_arch_flags()

extensions = [
    CUDAExtension(
        name="stainx_cuda",
        sources=[str(csrc_dir / "bindings.cpp"), str(csrc_dir / "histogram_matching.cu"), str(csrc_dir / "reinhard.cu"), str(csrc_dir / "macenko.cu")],
        include_dirs=[str(csrc_dir)],
        define_macros=[("TARGET_CUDA_ARCH", str(cc))],
        extra_compile_args={"cxx": ["-std=c++17", "-O3", "-DNDEBUG"], "nvcc": nvcc_flags},
        extra_link_args=["-lcudart", "-lcublas", "-lcusolver"],
    )
]

# Enable ninja for faster builds (requires ninja to be installed)
use_ninja = os.environ.get("USE_NINJA", "true").lower() == "true"

setup(name="stainx-cuda", ext_modules=extensions, cmdclass={"build_ext": BuildExtension.with_options(use_ninja=use_ninja)}, zip_safe=False)
