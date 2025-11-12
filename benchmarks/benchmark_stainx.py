# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import argparse
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import torch
from prettytable import PrettyTable
from skimage.exposure import match_histograms
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from logger import setup_logger

# Global logger instance (will be initialized in main)
logger = None


class ImageGenerator:
    """Generate test images for benchmarking."""

    @staticmethod
    def generate_image(
        height: int, width: int, channels: int = 3, seed: int = 42, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Generate a random test image.

        Parameters
        ----------
        height : int
            Image height
        width : int
            Image width
        channels : int, default=3
            Number of channels
        seed : int, default=42
            Random seed
        device : str, default="cpu"
            Device to create tensor on

        Returns
        -------
        torch.Tensor
            Image tensor of shape (1, C, H, W) with uint8 dtype
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        device_obj = torch.device(device)
        return (
            (torch.rand(1, channels, height, width, device=device_obj) * 255)
            .round()
            .to(torch.uint8)
        )


class BenchmarkExecutor:
    """Execute benchmark operations with warmup and timing."""

    def __init__(self, warmup_iterations: int = 3, benchmark_iterations: int = 10):
        """
        Initialize benchmark executor.

        Parameters
        ----------
        warmup_iterations : int, default=3
            Number of warmup iterations
        benchmark_iterations : int, default=10
            Number of benchmark iterations
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def _execute_benchmark(
        self, operation_func: Callable, name: str, device: str = "cpu"
    ) -> dict[str, Any]:
        """
        Execute a benchmark operation.

        Parameters
        ----------
        operation_func : Callable
            Function to benchmark (should return the result)
        name : str
            Name of the operation for logging
        device : str, default="cpu"
            Device being used

        Returns
        -------
        Dict[str, Any]
            Dictionary with 'result', 'time_ms', and 'success' keys
        """
        try:
            # Warmup
            for _ in range(self.warmup_iterations):
                result = operation_func()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Synchronize before timing
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            start_time = time.time()
            for _ in range(self.benchmark_iterations):
                result = operation_func()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
            total_time = time.time() - start_time

            avg_time_ms = (total_time / self.benchmark_iterations) * 1000
            return {"result": result, "time_ms": avg_time_ms, "success": True}

        except Exception as e:
            if logger:
                logger.info(f"    {name} failed: {e}")
            return {"result": None, "time_ms": float("inf"), "success": False}
        finally:
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run_stainx_reinhard(
        self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str
    ) -> dict[str, Any]:
        """Benchmark stainx Reinhard implementation."""
        normalizer = Reinhard(device=device)

        def operation():
            normalizer.fit(reference_image)
            return normalizer.transform(source_image)

        return self._execute_benchmark(operation, "StainX Reinhard", device)

    def run_torchstain_reinhard(
        self, reference_image: torch.Tensor, source_image: torch.Tensor
    ) -> dict[str, Any]:
        """Benchmark torchstain Reinhard implementation."""
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image.squeeze(0).cpu()
        normalizer = TorchReinhardNormalizer()

        def operation():
            normalizer.fit(ref_chw)
            return normalizer.normalize(src_chw)

        return self._execute_benchmark(operation, "TorchStain Reinhard", "cpu")

    def run_stainx_macenko(
        self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str
    ) -> dict[str, Any]:
        """Benchmark stainx Macenko implementation."""
        normalizer = Macenko(device=device)

        def operation():
            normalizer.fit(reference_image)
            return normalizer.transform(source_image)

        return self._execute_benchmark(operation, "StainX Macenko", device)

    def run_torchstain_macenko(
        self, reference_image: torch.Tensor, source_image: torch.Tensor
    ) -> dict[str, Any]:
        """Benchmark torchstain Macenko implementation."""
        ref_chw = reference_image.squeeze(0).cpu()
        src_chw = source_image.squeeze(0).cpu()
        normalizer = TorchMacenkoNormalizer()

        def operation():
            normalizer.fit(ref_chw)
            result, _, _ = normalizer.normalize(src_chw, stains=True)
            return result

        return self._execute_benchmark(operation, "TorchStain Macenko", "cpu")

    def run_stainx_histogram_matching(
        self,
        reference_image: torch.Tensor,
        source_image: torch.Tensor,
        device: str,
        channel_axis: int = 1,
    ) -> dict[str, Any]:
        """Benchmark stainx HistogramMatching implementation."""
        converter = ChannelFormatConverter(channel_axis=channel_axis)
        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image)
        normalizer = HistogramMatching(device=device, channel_axis=channel_axis)

        def operation():
            normalizer.fit(ref_input)
            return normalizer.transform(src_input)

        return self._execute_benchmark(operation, "StainX HistogramMatching", device)

    def run_skimage_histogram_matching(
        self, reference_image: torch.Tensor, source_image: torch.Tensor
    ) -> dict[str, Any]:
        """Benchmark skimage histogram matching implementation."""
        converter = ChannelFormatConverter(channel_axis=1)
        ref_np_uint8 = converter.to_hwc(reference_image, squeeze_batch=True)
        src_np_uint8 = converter.to_hwc(source_image, squeeze_batch=True)

        def operation():
            return match_histograms(src_np_uint8, ref_np_uint8, channel_axis=-1)

        return self._execute_benchmark(operation, "Skimage HistogramMatching", "cpu")


class BenchmarkRunner:
    """Run comprehensive benchmarks comparing stainx against other implementations."""

    def __init__(self, warmup_iterations: int = 3, benchmark_iterations: int = 10):
        """
        Initialize benchmark runner.

        Parameters
        ----------
        warmup_iterations : int, default=3
            Number of warmup iterations
        benchmark_iterations : int, default=10
            Number of benchmark iterations
        """
        self.executor = BenchmarkExecutor(warmup_iterations, benchmark_iterations)

    def run_reinhard_benchmarks(
        self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str
    ) -> dict[str, dict[str, Any]]:
        """
        Run Reinhard normalization benchmarks.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with benchmark results for stainx and torchstain
        """
        return {
            "stainx": self.executor.run_stainx_reinhard(
                reference_image, source_image, device
            ),
            "torchstain": self.executor.run_torchstain_reinhard(
                reference_image, source_image
            ),
        }

    def run_macenko_benchmarks(
        self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str
    ) -> dict[str, dict[str, Any]]:
        """
        Run Macenko normalization benchmarks.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with benchmark results for stainx and torchstain
        """
        return {
            "stainx": self.executor.run_stainx_macenko(
                reference_image, source_image, device
            ),
            "torchstain": self.executor.run_torchstain_macenko(
                reference_image, source_image
            ),
        }

    def run_histogram_matching_benchmarks(
        self,
        reference_image: torch.Tensor,
        source_image: torch.Tensor,
        device: str,
        channel_axis: int = 1,
    ) -> dict[str, dict[str, Any]]:
        """
        Run HistogramMatching normalization benchmarks.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with benchmark results for stainx and skimage
        """
        return {
            "stainx": self.executor.run_stainx_histogram_matching(
                reference_image, source_image, device, channel_axis
            ),
            "skimage": self.executor.run_skimage_histogram_matching(
                reference_image, source_image
            ),
        }

    @staticmethod
    def calculate_speedup(baseline_time: float, comparison_time: float) -> str:
        """
        Calculate speedup factor.

        Parameters
        ----------
        baseline_time : float
            Baseline implementation time in ms
        comparison_time : float
            Comparison implementation time in ms

        Returns
        -------
        str
            Speedup factor as "X.XXx" or "N/A"
        """
        if (
            baseline_time != float("inf")
            and comparison_time != float("inf")
            and comparison_time > 0
        ):
            speedup = baseline_time / comparison_time
            return f"{speedup:.2f}x"
        return "N/A"

    @staticmethod
    def calculate_relative_error(y1: torch.Tensor, y2: torch.Tensor) -> float:
        """
        Calculate relative error between two tensors.

        Parameters
        ----------
        y1 : torch.Tensor
            First tensor
        y2 : torch.Tensor
            Second tensor

        Returns
        -------
        float
            Relative error, or inf if tensors are incompatible
        """
        if y1 is None or y2 is None:
            return float("inf")

        # Handle different formats - convert to same format for comparison
        if isinstance(y1, np.ndarray):
            y1 = torch.from_numpy(y1).float()
        if isinstance(y2, np.ndarray):
            y2 = torch.from_numpy(y2).float()

        # Ensure same device and dtype
        y1 = y1.cpu().float()
        y2 = y2.cpu().float()

        # Handle different shapes (e.g., HWC vs CHW)
        if y1.shape != y2.shape and len(y1.shape) == 3 and len(y2.shape) == 3:
            # Try to reshape/permute to match
            # Check if one is CHW and other is HWC
            if y1.shape[0] == y2.shape[2] and y1.shape[2] == y2.shape[0]:
                y2 = y2.permute(2, 0, 1)
            elif y1.shape[2] == y2.shape[0] and y1.shape[0] == y2.shape[2]:
                y1 = y1.permute(2, 0, 1)

        if y1.shape != y2.shape:
            if logger:
                logger.info(f"    Warning: Shape mismatch - {y1.shape} vs {y2.shape}")
            return float("inf")

        epsilon = 1e-16
        norm_y1 = torch.norm(y1, p=2) + epsilon
        rel_error = torch.norm((y1 - y2).abs(), p=2) / norm_y1

        if rel_error > 1e-2 and logger:
            logger.info(f"    Warning: High relative error {rel_error:.6f}")

        return rel_error.item()


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(
        description="Benchmark StainX runtime against torchstain and skimage"
    )
    parser.add_argument(
        "--image-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Multiple image sizes to test (format: H W)",
    )
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    # Setup logging first
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name = f"{current_time}"

    global logger
    logger = setup_logger(
        filename=os.path.join(
            os.path.dirname(__file__), "logs", f"stainx_benchmark_{log_file_name}.log"
        ),
        verbose=True,
    )

    if device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if device == "mps" and not torch.backends.mps.is_available():
        logger.info("MPS requested but not available, falling back to CPU")
        device = "cpu"

    logger.info(
        f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}"
    )
    logger.info(f"Warmup runs: {args.warmup}, Benchmark runs: {args.runs}")
    logger.info("")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    benchmark_runner = BenchmarkRunner(args.warmup, args.runs)

    # Determine image sizes to test
    if args.image_sizes:
        # Parse pairs of height, width
        image_sizes = []
        for i in range(0, len(args.image_sizes), 2):
            if i + 1 < len(args.image_sizes):
                image_sizes.append((args.image_sizes[i], args.image_sizes[i + 1]))
    else:
        image_sizes = [(args.height, args.width)]

    # Create tables for each method
    reinhard_table = None
    macenko_table = None
    histogram_table = None

    reinhard_table = PrettyTable()
    reinhard_table.title = f"Reinhard Benchmark ({device.upper()})"
    reinhard_table.field_names = [
        "Image Size (HxW)",
        "StainX (ms)",
        "TorchStain (ms)",
        "Speedup",
        "Relative Error",
    ]

    macenko_table = PrettyTable()
    macenko_table.title = f"Macenko Benchmark ({device.upper()})"
    macenko_table.field_names = [
        "Image Size (HxW)",
        "StainX (ms)",
        "TorchStain (ms)",
        "Speedup",
        "Relative Error",
    ]

    histogram_table = PrettyTable()
    histogram_table.title = f"HistogramMatching Benchmark ({device.upper()})"
    histogram_table.field_names = [
        "Image Size (HxW)",
        "StainX (ms)",
        "Skimage (ms)",
        "Speedup",
        "Relative Error",
    ]

    logger.info("Starting benchmark...")
    logger.info("=" * 100)

    for height, width in image_sizes:
        logger.info(f"Testing image size: {height}x{width}")

        # Generate test images
        reference_image = ImageGenerator.generate_image(
            height, width, args.channels, seed=args.seed, device=device
        )
        source_image = ImageGenerator.generate_image(
            height, width, args.channels, seed=args.seed + 1, device=device
        )

        # Reinhard benchmarks
        logger.info("  Running Reinhard benchmarks...")
        reinhard_results = benchmark_runner.run_reinhard_benchmarks(
            reference_image, source_image, device
        )

        # Calculate speedup (stainx vs torchstain)
        speedup = BenchmarkRunner.calculate_speedup(
            reinhard_results["torchstain"]["time_ms"],
            reinhard_results["stainx"]["time_ms"],
        )

        # Calculate relative error
        if (
            reinhard_results["stainx"]["result"] is not None
            and reinhard_results["torchstain"]["result"] is not None
        ):
            # Convert torchstain result to CHW format
            torchstain_result = (
                reinhard_results["torchstain"]["result"].permute(2, 0, 1).float()
            )
            stainx_result = (
                reinhard_results["stainx"]["result"].squeeze(0).cpu().float()
            )
            rel_error = BenchmarkRunner.calculate_relative_error(
                stainx_result, torchstain_result
            )
        else:
            rel_error = float("inf")

        reinhard_table.add_row(
            [
                f"{height}x{width}",
                f"{reinhard_results['stainx']['time_ms']:.3f}"
                if reinhard_results["stainx"]["success"]
                else "ERROR",
                f"{reinhard_results['torchstain']['time_ms']:.3f}"
                if reinhard_results["torchstain"]["success"]
                else "ERROR",
                speedup,
                f"{rel_error:.6f}" if rel_error != float("inf") else "N/A",
            ]
        )

        # Macenko benchmarks
        logger.info("  Running Macenko benchmarks...")
        macenko_results = benchmark_runner.run_macenko_benchmarks(
            reference_image, source_image, device
        )

        # Calculate speedup
        speedup = BenchmarkRunner.calculate_speedup(
            macenko_results["torchstain"]["time_ms"],
            macenko_results["stainx"]["time_ms"],
        )

        # Calculate relative error
        if (
            macenko_results["stainx"]["result"] is not None
            and macenko_results["torchstain"]["result"] is not None
        ):
            # Convert torchstain result to CHW format
            torchstain_result = (
                macenko_results["torchstain"]["result"].permute(2, 0, 1).float()
            )
            stainx_result = macenko_results["stainx"]["result"].squeeze(0).cpu().float()
            rel_error = BenchmarkRunner.calculate_relative_error(
                stainx_result, torchstain_result
            )
        else:
            rel_error = float("inf")

        macenko_table.add_row(
            [
                f"{height}x{width}",
                f"{macenko_results['stainx']['time_ms']:.3f}"
                if macenko_results["stainx"]["success"]
                else "ERROR",
                f"{macenko_results['torchstain']['time_ms']:.3f}"
                if macenko_results["torchstain"]["success"]
                else "ERROR",
                speedup,
                f"{rel_error:.6f}" if rel_error != float("inf") else "N/A",
            ]
        )

        # HistogramMatching benchmarks
        logger.info("  Running HistogramMatching benchmarks...")
        histogram_results = benchmark_runner.run_histogram_matching_benchmarks(
            reference_image, source_image, device, channel_axis=1
        )

        # Calculate speedup
        speedup = BenchmarkRunner.calculate_speedup(
            histogram_results["skimage"]["time_ms"],
            histogram_results["stainx"]["time_ms"],
        )

        # Calculate relative error
        if (
            histogram_results["stainx"]["result"] is not None
            and histogram_results["skimage"]["result"] is not None
        ):
            # Convert results to CHW format for comparison
            converter = ChannelFormatConverter(channel_axis=1)
            stainx_result = converter.to_chw(
                histogram_results["stainx"]["result"], squeeze_batch=True
            ).float()
            skimage_result = (
                torch.from_numpy(histogram_results["skimage"]["result"])
                .permute(2, 0, 1)
                .float()
            )
            rel_error = BenchmarkRunner.calculate_relative_error(
                stainx_result, skimage_result
            )
        else:
            rel_error = float("inf")

        histogram_table.add_row(
            [
                f"{height}x{width}",
                f"{histogram_results['stainx']['time_ms']:.3f}"
                if histogram_results["stainx"]["success"]
                else "ERROR",
                f"{histogram_results['skimage']['time_ms']:.3f}"
                if histogram_results["skimage"]["success"]
                else "ERROR",
                speedup,
                f"{rel_error:.6f}" if rel_error != float("inf") else "N/A",
            ]
        )

    logger.info("=" * 100)
    logger.info("Benchmark Results:")
    logger.info("")

    if reinhard_table:
        logger.info(reinhard_table)
        logger.info("")

    if macenko_table:
        logger.info(macenko_table)
        logger.info("")

    if histogram_table:
        logger.info(histogram_table)
        logger.info("")

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
