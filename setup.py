# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""Build CUDA extension for stainx if CUDA is available."""

import contextlib
import os
import re
from pathlib import Path
from typing import ClassVar

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_version_from_pyproject():
    """Read version from pyproject.toml (single source of truth)."""
    project_root = Path(__file__).parent
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path) as f:
        for line in f:
            if line.strip().startswith("version ="):
                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', line)
                if match:
                    return match.group(1)
    raise RuntimeError("Could not find version in pyproject.toml")


class CUDADeviceInfo:
    """Handles CUDA device detection and information retrieval."""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            print("CUDA device not available at build time. Will build for architectures specified in TORCH_CUDA_ARCH_LIST or common architectures.")
            self.device = None
            self.device_name = None
            self.device_capability = None
            # Use a default architecture for define_macros (actual archs come from TORCH_CUDA_ARCH_LIST)
            self.compute_capability = 75  # Default fallback
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
        if self.cuda_available:
            print("Building CUDA extension...")
            print(f"Current device: {self.device_name}")
            print(f"Current CUDA capability: {self.device_capability}")
            print(f"Detected compute capability: {self.compute_capability}")
        else:
            print("Building CUDA extension (CUDA device not available at build time)...")
            if "TORCH_CUDA_ARCH_LIST" in os.environ:
                print(f"Using TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")


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
        if not torch.cuda.is_available():
            return []

        # Print device and version info
        self.device_info.print_info()
        self.version_checker.check()

        # Get architecture flags
        nvcc_flags = self.flags_manager.get_architecture_flags(self.device_info.compute_capability)

        # Define source files - use relative paths from setup.py directory
        source_files = ["bindings.cpp", "histogram_matching.cu", "reinhard.cu", "macenko.cu"]
        sources = [str(Path("src") / "stainx_cuda" / "csrc" / f) for f in source_files]

        # Include directory - use relative path
        include_dir = str(Path("src") / "stainx_cuda" / "csrc")

        # Create extension
        extension = CUDAExtension(
            name="stainx_cuda.stainx_cuda", sources=sources, include_dirs=[include_dir], define_macros=[("TARGET_CUDA_ARCH", str(self.device_info.compute_capability))], extra_compile_args={"cxx": ["-std=c++17", "-O3", "-DNDEBUG"], "nvcc": nvcc_flags}, extra_link_args=["-lcudart", "-lcublas", "-lcusolver"]
        )

        return [extension]

    def get_build_ext_class(self):
        """Get the build extension class with ninja support."""
        use_ninja = os.environ.get("USE_NINJA", "true").lower() == "true"
        return BuildExtension.with_options(use_ninja=use_ninja)


# Automatically detect and set CUDA architectures (like torch-floating-point)
if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    from torch import cuda

    arch_list = []
    for i in range(cuda.device_count()):
        capability = cuda.get_device_capability(i)
        arch = f"{capability[0]}.{capability[1]}"
        arch_list.append(arch)

    # Add PTX for the highest architecture for forward compatibility
    if arch_list:
        highest_arch = arch_list[-1]
        arch_list.append(f"{highest_arch}+PTX")

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)
    print(f"Setting TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

# Set PyTorch library path for runtime linking
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = torch_lib_path
else:
    os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"

if torch.cuda.is_available():
    print("CUDA detected, building with CUDA support.")
    project_root = Path(__file__).parent
    builder = CUDAExtensionBuilder(project_root)
    extensions = builder.build()
    build_ext = builder.get_build_ext_class()
else:
    print("No CUDA detected, building without CUDA support.")
    extensions = []
    build_ext = BuildExtension.with_options(use_ninja=os.environ.get("USE_NINJA", "true").lower() == "true")

# Package discovery - use find_packages to automatically discover all packages and subpackages
packages = find_packages(where="src")
# Ensure stainx_cuda is included even if it doesn't have __init__.py with Python code
if "stainx_cuda" not in packages:
    packages.append("stainx_cuda")

with open("README.md") as f:
    long_description = f.read()

setup_kwargs = {
    "packages": packages,
    "package_dir": {"": "src"},
    "zip_safe": False,
    "version": get_version_from_pyproject(),
    "description": "GPU-accelerated stain normalization",
    "author": "Samir Moustafa",
    "author_email": "smoustafa@cemm.oeaw.ac.at",
    "url": "https://github.com/rendeirolab/stainx",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "classifiers": ["Environment :: GPU :: NVIDIA CUDA", "Intended Audience :: Developers", "Intended Audience :: Healthcare Industry", "Intended Audience :: Science/Research", "Programming Language :: Python :: 3", "Topic :: Scientific/Engineering", "Topic :: Software Development"],
}

# Always include extension (like torch-floating-point)
setup_kwargs["ext_modules"] = extensions
cmdclass = {"build_ext": build_ext}
setup_kwargs["cmdclass"] = cmdclass

setup(**setup_kwargs)
