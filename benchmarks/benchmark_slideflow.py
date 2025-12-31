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

from stainx import Macenko, Reinhard

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

# Import SlideFlow normalizers
import slideflow.norm as sf_norm
from utils import setup_logger

# Global logger instance (will be initialized in main)
logger = None


class ImageGenerator:
    @staticmethod
    def generate_batch(batch_size: int, height: int, width: int, channels: int = 3, seed: int = 42, device: str = "cpu") -> torch.Tensor:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device_obj = torch.device(device)
        return (torch.rand(batch_size, channels, height, width, device=device_obj) * 255).round().to(torch.uint8)


class BenchmarkExecutor:
    def __init__(self, warmup_iterations: int = 3, benchmark_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def _execute_benchmark(self, operation_func: Callable, name: str, device: str = "cpu") -> dict[str, Any]:
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

    def run_stainx_reinhard(self, reference_batch: torch.Tensor, source_batch: torch.Tensor, device: str) -> dict[str, Any]:
        normalizer = Reinhard(device=device)
        # Fit once outside the benchmark loop on first image
        normalizer.fit(reference_batch[:1])

        def operation():
            # Only benchmark transform
            return normalizer.transform(source_batch)

        return self._execute_benchmark(operation, "StainX Reinhard", device)

    def run_slideflow_reinhard(self, reference_batch: torch.Tensor, source_batch: torch.Tensor, device: str) -> dict[str, Any]:
        # Use SlideFlow autoselect to get PyTorch normalizer (supports batches and CUDA)
        normalizer = sf_norm.autoselect("reinhard_fast")
        normalizer.device = device

        # Ensure on correct device (data is already uint8 from generator)
        ref_batch = reference_batch.to(device)
        src_batch = source_batch.to(device)

        # Extract first image (index 0) - same as reference_batch[:1] but without batch dim
        # Convert to numpy HWC format for SlideFlow fit (permute reorders dims, doesn't change image)
        ref_first_np = ref_batch[0].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

        # Fit once outside the benchmark loop on the same first image (index 0) as StainX
        normalizer.fit(ref_first_np)

        def operation():
            return normalizer.transform(src_batch)

        return self._execute_benchmark(operation, "SlideFlow Reinhard", device)

    def run_stainx_macenko(self, reference_batch: torch.Tensor, source_batch: torch.Tensor, device: str) -> dict[str, Any]:
        normalizer = Macenko(device=device)
        # Fit once outside the benchmark loop on first image
        normalizer.fit(reference_batch[:1])

        def operation():
            # Only benchmark transform
            return normalizer.transform(source_batch)

        return self._execute_benchmark(operation, "StainX Macenko", device)

    def run_slideflow_macenko(self, reference_batch: torch.Tensor, source_batch: torch.Tensor, device: str) -> dict[str, Any]:
        normalizer = sf_norm.autoselect("macenko_fast")
        normalizer.device = device

        # Ensure on correct device (data is already uint8 from generator)
        ref_batch = reference_batch.to(device)
        src_batch = source_batch.to(device)

        # Extract first image (index 0) - same as reference_batch[:1] but without batch dim
        # Convert to numpy HWC format for SlideFlow fit (permute reorders dims, doesn't change image)
        ref_first_np = ref_batch[0].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

        # Fit once outside the benchmark loop on the same first image (index 0) as StainX
        normalizer.fit(ref_first_np)

        def operation():
            return normalizer.transform(src_batch)

        return self._execute_benchmark(operation, "SlideFlow Macenko", device)


class BenchmarkRunner:
    def __init__(self, warmup_iterations: int = 3, benchmark_iterations: int = 10):
        self.executor = BenchmarkExecutor(warmup_iterations, benchmark_iterations)

    def run_reinhard_benchmarks(self, reference_batch: torch.Tensor, source_batch: torch.Tensor, device: str) -> dict[str, dict[str, Any]]:
        return {"stainx": self.executor.run_stainx_reinhard(reference_batch, source_batch, device), "slideflow": self.executor.run_slideflow_reinhard(reference_batch, source_batch, device)}

    def run_macenko_benchmarks(self, reference_batch: torch.Tensor, source_batch: torch.Tensor, device: str) -> dict[str, dict[str, Any]]:
        return {"stainx": self.executor.run_stainx_macenko(reference_batch, source_batch, device), "slideflow": self.executor.run_slideflow_macenko(reference_batch, source_batch, device)}

    @staticmethod
    def calculate_speedup(baseline_ips: float, comparison_ips: float) -> str:
        if baseline_ips != float("inf") and comparison_ips != float("inf") and baseline_ips > 0:
            speedup = comparison_ips / baseline_ips
            return f"{speedup:.2f}x"
        return "N/A"

    @staticmethod
    def calculate_relative_error(y1: torch.Tensor, y2: torch.Tensor) -> float:
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
    parser = argparse.ArgumentParser(description="Benchmark StainX runtime against SlideFlow")
    parser.add_argument("--image-sizes", nargs="+", type=int, default=None, help="Multiple image sizes to test (format: H W)")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to use")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for images")

    args = parser.parse_args()

    # Determine device
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device

    # Setup logging first
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    log_file_name = f"{current_time}"

    global logger
    logger = setup_logger(filename=os.path.join(os.path.dirname(__file__), "logs", f"slideflow_benchmark_{log_file_name}.log"), verbose=True)

    if device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if device == "mps" and not torch.backends.mps.is_available():
        logger.info("MPS requested but not available, falling back to CPU")
        device = "cpu"

    logger.info(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}")
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
    reinhard_table.title = f"Reinhard Benchmark ({device.upper()}, batch={args.batch_size})"
    reinhard_table.field_names = ["Image Size (HxW)", "StainX (img/s)", "SlideFlow (img/s)", "Speedup", "Relative Error"]

    macenko_table = PrettyTable()
    macenko_table.title = f"Macenko Benchmark ({device.upper()}, batch={args.batch_size})"
    macenko_table.field_names = ["Image Size (HxW)", "StainX (img/s)", "SlideFlow (img/s)", "Speedup", "Relative Error"]

    logger.info("Starting benchmark...")
    logger.info("=" * 100)

    for height, width in image_sizes:
        logger.info(f"Testing image size: {height}x{width} (batch size: {args.batch_size})")

        # Generate test image batches
        reference_batch = ImageGenerator.generate_batch(args.batch_size, height, width, args.channels, seed=args.seed, device=device)
        source_batch = ImageGenerator.generate_batch(args.batch_size, height, width, args.channels, seed=args.seed + 1, device=device)

        # Reinhard benchmarks
        logger.info("  Running Reinhard benchmarks...")
        reinhard_results = benchmark_runner.run_reinhard_benchmarks(reference_batch, source_batch, device)

        # Calculate images per second from time_ms
        batch_size = reference_batch.shape[0]
        stainx_ips = batch_size * 1000 / reinhard_results["stainx"]["time_ms"] if reinhard_results["stainx"]["success"] and reinhard_results["stainx"]["time_ms"] > 0 else 0.0

        slideflow_ips = batch_size * 1000 / reinhard_results["slideflow"]["time_ms"] if reinhard_results["slideflow"]["success"] and reinhard_results["slideflow"]["time_ms"] > 0 else 0.0

        # Calculate speedup (stainx vs slideflow)
        speedup = BenchmarkRunner.calculate_speedup(slideflow_ips, stainx_ips)

        # Calculate relative error (compare first image of batch)
        if reinhard_results["stainx"]["result"] is not None and reinhard_results["slideflow"]["result"] is not None:
            # TorchStainNormalizer returns PyTorch tensors in (N, C, H, W) format
            slideflow_result = reinhard_results["slideflow"]["result"]
            slideflow_result = torch.from_numpy(slideflow_result).float() if isinstance(slideflow_result, np.ndarray) else slideflow_result.cpu().float()

            # Compare first image of batch
            stainx_result = reinhard_results["stainx"]["result"][0].cpu().float()
            slideflow_first = slideflow_result[0] if len(slideflow_result.shape) == 4 else slideflow_result
            rel_error = BenchmarkRunner.calculate_relative_error(stainx_result, slideflow_first)
        else:
            rel_error = float("inf")

        reinhard_table.add_row([f"{height}x{width}", f"{stainx_ips:.2f}" if reinhard_results["stainx"]["success"] else "ERROR", f"{slideflow_ips:.2f}" if reinhard_results["slideflow"]["success"] else "ERROR", speedup, f"{rel_error:.6f}" if rel_error != float("inf") else "N/A"])

        # Macenko benchmarks
        logger.info("  Running Macenko benchmarks...")
        macenko_results = benchmark_runner.run_macenko_benchmarks(reference_batch, source_batch, device)

        # Calculate images per second from time_ms
        batch_size = reference_batch.shape[0]
        stainx_ips = batch_size * 1000 / macenko_results["stainx"]["time_ms"] if macenko_results["stainx"]["success"] and macenko_results["stainx"]["time_ms"] > 0 else 0.0

        slideflow_ips = batch_size * 1000 / macenko_results["slideflow"]["time_ms"] if macenko_results["slideflow"]["success"] and macenko_results["slideflow"]["time_ms"] > 0 else 0.0

        # Calculate speedup
        speedup = BenchmarkRunner.calculate_speedup(slideflow_ips, stainx_ips)

        # Calculate relative error (compare first image of batch)
        if macenko_results["stainx"]["result"] is not None and macenko_results["slideflow"]["result"] is not None:
            # TorchStainNormalizer returns PyTorch tensors in (N, C, H, W) format
            slideflow_result = macenko_results["slideflow"]["result"]
            slideflow_result = torch.from_numpy(slideflow_result).float() if isinstance(slideflow_result, np.ndarray) else slideflow_result.cpu().float()

            # Compare first image of batch
            stainx_result = macenko_results["stainx"]["result"][0].cpu().float()
            slideflow_first = slideflow_result[0] if len(slideflow_result.shape) == 4 else slideflow_result
            rel_error = BenchmarkRunner.calculate_relative_error(stainx_result, slideflow_first)
        else:
            rel_error = float("inf")

        macenko_table.add_row([f"{height}x{width}", f"{stainx_ips:.2f}" if macenko_results["stainx"]["success"] else "ERROR", f"{slideflow_ips:.2f}" if macenko_results["slideflow"]["success"] else "ERROR", speedup, f"{rel_error:.6f}" if rel_error != float("inf") else "N/A"])

    logger.info("=" * 100)
    logger.info("Benchmark Results:")
    logger.info("")

    logger.info(reinhard_table)
    logger.info("")

    logger.info(macenko_table)
    logger.info("")

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
