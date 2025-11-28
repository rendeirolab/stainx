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
from typing import ClassVar

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class CUDADeviceInfo:
    """Handles CUDA device detection and information retrieval."""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            print("CUDA not available. Skipping CUDA extension build.")
            self.device = None
            self.device_name = None
            self.device_capability = None
            self.compute_capability = None
            return

        self.device = torch.cuda.current_device()
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_capability = torch.cuda.get_device_capability(self.device)
        self.compute_capability = self._detect_compute_capability()

    def _detect_compute_capability(self):
        """Detect compute capability from current device."""
        major, minor = torch.cuda.get_device_capability(self.device)
        return major * 10 + minor

    def print_info(self):
        """Print device information."""
        print("Building CUDA extension...")
        print(f"Current device: {self.device_name}")
        print(f"Current CUDA capability: {self.device_capability}")
        print(f"Detected compute capability: {self.compute_capability}")


class PyTorchVersionChecker:
    """Handles PyTorch version checking and validation."""

    MIN_MAJOR = 2
    MIN_MINOR = 0

    def __init__(self):
        self.version = torch.__version__
        self.major, self.minor = self._parse_version()

    def _parse_version(self):
        """Parse PyTorch version string."""
        m = re.match(r"^(\d+)\.(\d+)", self.version)
        if not m:
            raise RuntimeError(f"Cannot parse PyTorch version '{self.version}'")
        return tuple(map(int, m.groups()))

    def check(self):
        """Check if PyTorch version meets requirements."""
        print(f"PyTorch version: {self.version}")
        if self.major < self.MIN_MAJOR or (self.major == self.MIN_MAJOR and self.minor < self.MIN_MINOR):
            print(f"Warning: PyTorch version {self.version} may not be fully supported. Recommended: >= {self.MIN_MAJOR}.{self.MIN_MINOR}")


class NVCCFlagsManager:
    """Manages NVCC compilation flags."""

    UNWANTED_FLAGS: ClassVar[list[str]] = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]

    BASE_FLAGS: ClassVar[list[str]] = ["--expt-relaxed-constexpr", "--use_fast_math", "-std=c++17", "-O3", "-DNDEBUG", "-Xcompiler", "-funroll-loops", "-Xcompiler", "-ffast-math", "-Xcompiler", "-finline-functions"]

    def __init__(self):
        self._remove_unwanted_flags()

    def _remove_unwanted_flags(self):
        """Remove unwanted PyTorch NVCC flags."""
        for flag in self.UNWANTED_FLAGS:
            with contextlib.suppress(ValueError):
                torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)

    def get_architecture_flags(self, compute_capability):
        """Generate architecture-specific flags for the given compute capability."""
        flags = self.BASE_FLAGS.copy()

        # Format: compute_XX and sm_XX
        major = compute_capability // 10
        minor = compute_capability % 10
        compute_arch = f"compute_{major}{minor}"
        sm_arch = f"sm_{major}{minor}"

        flags.extend(["-gencode", f"arch={compute_arch},code={sm_arch}"])
        return flags


class CUDAExtensionBuilder:
    """Builds and configures the CUDA extension."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.csrc_dir = project_root / "src" / "stainx_cuda" / "csrc"
        self.device_info = CUDADeviceInfo()
        self.version_checker = PyTorchVersionChecker()
        self.flags_manager = NVCCFlagsManager()

    def build(self):
        """Build the CUDA extension configuration."""
        if not self.device_info.cuda_available:
            return []

        # Print device and version info
        self.device_info.print_info()
        self.version_checker.check()

        # Get architecture flags
        nvcc_flags = self.flags_manager.get_architecture_flags(self.device_info.compute_capability)

        # Define source files
        sources = [str(self.csrc_dir / "bindings.cpp"), str(self.csrc_dir / "histogram_matching.cu"), str(self.csrc_dir / "reinhard.cu"), str(self.csrc_dir / "macenko.cu")]

        # Create extension
        extension = CUDAExtension(
            name="stainx_cuda", sources=sources, include_dirs=[str(self.csrc_dir)], define_macros=[("TARGET_CUDA_ARCH", str(self.device_info.compute_capability))], extra_compile_args={"cxx": ["-std=c++17", "-O3", "-DNDEBUG"], "nvcc": nvcc_flags}, extra_link_args=["-lcudart", "-lcublas", "-lcusolver"]
        )

        return [extension]

    def get_build_ext_class(self):
        """Get the build extension class with ninja support."""
        use_ninja = os.environ.get("USE_NINJA", "true").lower() == "true"
        return BuildExtension.with_options(use_ninja=use_ninja)


# Main setup execution
project_root = Path(__file__).parent
builder = CUDAExtensionBuilder(project_root)

extensions = builder.build()
build_ext = builder.get_build_ext_class()

# Package discovery - setuptools will auto-discover packages in src/
# but we explicitly configure it to ensure both stainx and stainx_cuda are included
setup_kwargs = {
    "packages": ["stainx", "stainx_cuda"],
    "package_dir": {"": "src"},
    "zip_safe": False,
}

# Only add extension-related arguments if CUDA extension is being built
if extensions:
    setup_kwargs["ext_modules"] = extensions
    setup_kwargs["cmdclass"] = {"build_ext": build_ext}

setup(**setup_kwargs)
