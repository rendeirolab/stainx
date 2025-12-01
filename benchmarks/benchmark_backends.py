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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from logger import setup_logger

# Global logger instance (will be initialized in main)
logger = None


class ImageGenerator:
    @staticmethod
    def generate_image(height: int, width: int, channels: int = 3, batch_size: int = 1, seed: int = 42, device: str = "cpu") -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device_obj = torch.device(device)
        return (torch.rand(batch_size, channels, height, width, device=device_obj) * 255).round().to(torch.uint8)


class BenchmarkExecutor:
    def __init__(self, warmup_iterations: int = 3, benchmark_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def _execute_benchmark(self, operation_func: Callable, name: str, device: str = "cpu") -> dict[str, Any]:
        """Execute a benchmark operation."""
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

    def run_reinhard_cuda(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str) -> dict[str, Any]:
        try:
            normalizer = Reinhard(device=device, backend="cuda")
            # Fit once outside the benchmark loop
            normalizer.fit(reference_image)

            def operation():
                # Only benchmark transform
                return normalizer.transform(source_image)

            return self._execute_benchmark(operation, "Reinhard CUDA", device)
        except (ImportError, NotImplementedError, RuntimeError) as e:
            if logger:
                logger.info(f"    CUDA backend not available: {e}")
            return {"result": None, "time_ms": float("inf"), "success": False}

    def run_reinhard_pytorch(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str) -> dict[str, Any]:
        normalizer = Reinhard(device=device, backend="pytorch")
        # Fit once outside the benchmark loop
        normalizer.fit(reference_image)

        def operation():
            # Only benchmark transform
            return normalizer.transform(source_image)

        return self._execute_benchmark(operation, "Reinhard PyTorch", device)

    def run_macenko_cuda(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str) -> dict[str, Any]:
        try:
            normalizer = Macenko(device=device, backend="cuda")
            # Fit once outside the benchmark loop
            normalizer.fit(reference_image)

            def operation():
                # Only benchmark transform
                return normalizer.transform(source_image)

            return self._execute_benchmark(operation, "Macenko CUDA", device)
        except (ImportError, NotImplementedError, RuntimeError) as e:
            if logger:
                logger.info(f"    CUDA backend not available: {e}")
            return {"result": None, "time_ms": float("inf"), "success": False}

    def run_macenko_pytorch(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str) -> dict[str, Any]:
        normalizer = Macenko(device=device, backend="pytorch")
        # Fit once outside the benchmark loop
        normalizer.fit(reference_image)

        def operation():
            # Only benchmark transform
            return normalizer.transform(source_image)

        return self._execute_benchmark(operation, "Macenko PyTorch", device)

    def run_histogram_matching_cuda(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str, channel_axis: int = 1) -> dict[str, Any]:
        try:
            converter = ChannelFormatConverter(channel_axis=channel_axis)
            ref_input = converter.prepare_for_normalizer(reference_image)
            src_input = converter.prepare_for_normalizer(source_image)
            normalizer = HistogramMatching(device=device, backend="cuda", channel_axis=channel_axis)
            # Fit once outside the benchmark loop
            normalizer.fit(ref_input)

            def operation():
                # Only benchmark transform
                return normalizer.transform(src_input)

            return self._execute_benchmark(operation, "HistogramMatching CUDA", device)
        except (ImportError, NotImplementedError, RuntimeError) as e:
            if logger:
                logger.info(f"    CUDA backend not available: {e}")
            return {"result": None, "time_ms": float("inf"), "success": False}

    def run_histogram_matching_pytorch(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str, channel_axis: int = 1) -> dict[str, Any]:
        converter = ChannelFormatConverter(channel_axis=channel_axis)
        ref_input = converter.prepare_for_normalizer(reference_image)
        src_input = converter.prepare_for_normalizer(source_image)
        normalizer = HistogramMatching(device=device, backend="pytorch", channel_axis=channel_axis)
        # Fit once outside the benchmark loop
        normalizer.fit(ref_input)

        def operation():
            # Only benchmark transform
            return normalizer.transform(src_input)

        return self._execute_benchmark(operation, "HistogramMatching PyTorch", device)


class BenchmarkRunner:
    def __init__(self, warmup_iterations: int = 3, benchmark_iterations: int = 10):
        self.executor = BenchmarkExecutor(warmup_iterations, benchmark_iterations)

    def run_reinhard_benchmarks(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str) -> dict[str, dict[str, Any]]:
        return {
            "cuda": self.executor.run_reinhard_cuda(reference_image, source_image, device),
            "pytorch": self.executor.run_reinhard_pytorch(reference_image, source_image, device),
        }

    def run_macenko_benchmarks(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str) -> dict[str, dict[str, Any]]:
        return {
            "cuda": self.executor.run_macenko_cuda(reference_image, source_image, device),
            "pytorch": self.executor.run_macenko_pytorch(reference_image, source_image, device),
        }

    def run_histogram_matching_benchmarks(self, reference_image: torch.Tensor, source_image: torch.Tensor, device: str, channel_axis: int = 1) -> dict[str, dict[str, Any]]:
        return {
            "cuda": self.executor.run_histogram_matching_cuda(reference_image, source_image, device, channel_axis),
            "pytorch": self.executor.run_histogram_matching_pytorch(reference_image, source_image, device, channel_axis),
        }

    @staticmethod
    def calculate_speedup(baseline_time: float, comparison_time: float) -> str:
        """Calculate speedup: baseline_time / comparison_time"""
        if baseline_time != float("inf") and comparison_time != float("inf") and comparison_time > 0:
            speedup = baseline_time / comparison_time
            return f"{speedup:.2f}x"
        return "N/A"

    @staticmethod
    def calculate_relative_error(y1: torch.Tensor, y2: torch.Tensor) -> float:
        """Calculate relative error between two tensors."""
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

        # Handle different shapes (e.g., HWC vs CHW, or BCHW vs CHW)
        if y1.shape != y2.shape:
            # Handle 3D tensors (CHW or HWC)
            if len(y1.shape) == 3 and len(y2.shape) == 3:
                # Try to reshape/permute to match
                # Check if one is CHW and other is HWC
                if y1.shape[0] == y2.shape[2] and y1.shape[2] == y2.shape[0]:
                    y2 = y2.permute(2, 0, 1)
                elif y1.shape[2] == y2.shape[0] and y1.shape[0] == y2.shape[2]:
                    y1 = y1.permute(2, 0, 1)
            # Handle 4D vs 3D (BCHW vs CHW)
            elif len(y1.shape) == 4 and len(y2.shape) == 3:
                # Assume first dimension is batch, squeeze it
                if y1.shape[0] == 1:
                    y1 = y1.squeeze(0)
                else:
                    # For batch > 1, we can't directly compare, so flatten batch dimension
                    y1 = y1.view(-1, *y1.shape[2:])
                    y2 = y2.unsqueeze(0).expand(y1.shape[0], -1, -1, -1).contiguous()
            elif len(y1.shape) == 3 and len(y2.shape) == 4:
                # Same as above but reversed
                if y2.shape[0] == 1:
                    y2 = y2.squeeze(0)
                else:
                    y2 = y2.view(-1, *y2.shape[2:])
                    y1 = y1.unsqueeze(0).expand(y2.shape[0], -1, -1, -1).contiguous()

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
    parser = argparse.ArgumentParser(description="Benchmark CUDA backend vs PyTorch backend for StainX normalizers")
    parser.add_argument("--image-sizes", nargs="+", type=int, default=None, help="Multiple image sizes to test (format: H W)")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (CUDA backend requires CUDA device)")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Determine device
    device = args.device

    # Setup logging first
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name = f"{current_time}"

    global logger
    logger = setup_logger(filename=os.path.join(os.path.dirname(__file__), "logs", f"backend_benchmark_{log_file_name}.log"), verbose=True)

    if device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA requested but not available. CUDA backend requires CUDA device.")
        logger.info("Falling back to CPU (PyTorch backend will still work, but CUDA backend will fail)")
        device = "cpu"

    if device == "cuda":
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Device: CPU")
    logger.info(f"Batch size: {args.batch_size}")
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
    reinhard_table = PrettyTable()
    reinhard_table.title = f"Reinhard Backend Comparison ({device.upper()}, Batch={args.batch_size})"
    reinhard_table.field_names = ["Image Size (HxW)", "CUDA (ms)", "PyTorch (ms)", "Speedup (PyTorch/CUDA)", "Relative Error"]

    macenko_table = PrettyTable()
    macenko_table.title = f"Macenko Backend Comparison ({device.upper()}, Batch={args.batch_size})"
    macenko_table.field_names = ["Image Size (HxW)", "CUDA (ms)", "PyTorch (ms)", "Speedup (PyTorch/CUDA)", "Relative Error"]

    histogram_table = PrettyTable()
    histogram_table.title = f"HistogramMatching Backend Comparison ({device.upper()}, Batch={args.batch_size})"
    histogram_table.field_names = ["Image Size (HxW)", "CUDA (ms)", "PyTorch (ms)", "Speedup (PyTorch/CUDA)", "Relative Error"]

    logger.info("Starting backend benchmark...")
    logger.info("=" * 100)

    for height, width in image_sizes:
        logger.info(f"Testing image size: {height}x{width} (batch size: {args.batch_size})")

        # Generate test images
        reference_image = ImageGenerator.generate_image(height, width, args.channels, batch_size=args.batch_size, seed=args.seed, device=device)
        source_image = ImageGenerator.generate_image(height, width, args.channels, batch_size=args.batch_size, seed=args.seed + 1, device=device)

        # Reinhard benchmarks
        logger.info("  Running Reinhard benchmarks...")
        reinhard_results = benchmark_runner.run_reinhard_benchmarks(reference_image, source_image, device)

        # Calculate speedup (CUDA vs PyTorch)
        speedup = BenchmarkRunner.calculate_speedup(reinhard_results["pytorch"]["time_ms"], reinhard_results["cuda"]["time_ms"])

        # Calculate relative error
        if reinhard_results["cuda"]["result"] is not None and reinhard_results["pytorch"]["result"] is not None:
            cuda_result = reinhard_results["cuda"]["result"].cpu().float()
            pytorch_result = reinhard_results["pytorch"]["result"].cpu().float()
            # Handle batch dimension - if batch_size is 1, squeeze it for comparison
            if args.batch_size == 1:
                cuda_result = cuda_result.squeeze(0)
                pytorch_result = pytorch_result.squeeze(0)
            rel_error = BenchmarkRunner.calculate_relative_error(cuda_result, pytorch_result)
        else:
            rel_error = float("inf")

        reinhard_table.add_row([
            f"{height}x{width}",
            f"{reinhard_results['cuda']['time_ms']:.3f}" if reinhard_results["cuda"]["success"] else "N/A",
            f"{reinhard_results['pytorch']['time_ms']:.3f}" if reinhard_results["pytorch"]["success"] else "ERROR",
            speedup,
            f"{rel_error:.6f}" if rel_error != float("inf") else "N/A",
        ])

        # Macenko benchmarks
        logger.info("  Running Macenko benchmarks...")
        macenko_results = benchmark_runner.run_macenko_benchmarks(reference_image, source_image, device)

        # Calculate speedup
        speedup = BenchmarkRunner.calculate_speedup(macenko_results["pytorch"]["time_ms"], macenko_results["cuda"]["time_ms"])

        # Calculate relative error
        if macenko_results["cuda"]["result"] is not None and macenko_results["pytorch"]["result"] is not None:
            cuda_result = macenko_results["cuda"]["result"].cpu().float()
            pytorch_result = macenko_results["pytorch"]["result"].cpu().float()
            # Handle batch dimension - if batch_size is 1, squeeze it for comparison
            if args.batch_size == 1:
                cuda_result = cuda_result.squeeze(0)
                pytorch_result = pytorch_result.squeeze(0)
            rel_error = BenchmarkRunner.calculate_relative_error(cuda_result, pytorch_result)
        else:
            rel_error = float("inf")

        macenko_table.add_row([
            f"{height}x{width}",
            f"{macenko_results['cuda']['time_ms']:.3f}" if macenko_results["cuda"]["success"] else "N/A",
            f"{macenko_results['pytorch']['time_ms']:.3f}" if macenko_results["pytorch"]["success"] else "ERROR",
            speedup,
            f"{rel_error:.6f}" if rel_error != float("inf") else "N/A",
        ])

        # HistogramMatching benchmarks
        logger.info("  Running HistogramMatching benchmarks...")
        histogram_results = benchmark_runner.run_histogram_matching_benchmarks(reference_image, source_image, device, channel_axis=1)

        # Calculate speedup
        speedup = BenchmarkRunner.calculate_speedup(histogram_results["pytorch"]["time_ms"], histogram_results["cuda"]["time_ms"])

        # Calculate relative error
        if histogram_results["cuda"]["result"] is not None and histogram_results["pytorch"]["result"] is not None:
            cuda_result = histogram_results["cuda"]["result"]
            pytorch_result = histogram_results["pytorch"]["result"]
            # Both should be in the same format from the normalizer
            if isinstance(cuda_result, torch.Tensor):
                cuda_result = cuda_result.cpu().float()
                # Handle batch dimension - if batch_size is 1, squeeze it for comparison
                if args.batch_size == 1 and cuda_result.dim() == 4:
                    cuda_result = cuda_result.squeeze(0)
            if isinstance(pytorch_result, torch.Tensor):
                pytorch_result = pytorch_result.cpu().float()
                # Handle batch dimension - if batch_size is 1, squeeze it for comparison
                if args.batch_size == 1 and pytorch_result.dim() == 4:
                    pytorch_result = pytorch_result.squeeze(0)
            rel_error = BenchmarkRunner.calculate_relative_error(cuda_result, pytorch_result)
        else:
            rel_error = float("inf")

        histogram_table.add_row([
            f"{height}x{width}",
            f"{histogram_results['cuda']['time_ms']:.3f}" if histogram_results["cuda"]["success"] else "N/A",
            f"{histogram_results['pytorch']['time_ms']:.3f}" if histogram_results["pytorch"]["success"] else "ERROR",
            speedup,
            f"{rel_error:.6f}" if rel_error != float("inf") else "N/A",
        ])

    logger.info("=" * 100)
    logger.info("Benchmark Results:")
    logger.info("")

    logger.info(reinhard_table)
    logger.info("")

    logger.info(macenko_table)
    logger.info("")

    logger.info(histogram_table)
    logger.info("")

    logger.info("Backend benchmark complete!")


if __name__ == "__main__":
    main()


