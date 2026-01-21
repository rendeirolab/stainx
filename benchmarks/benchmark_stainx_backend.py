# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import cupy as cp
import numpy as np
import torch
from prettytable import PrettyTable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stainx import HistogramMatching, Macenko, Reinhard

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from plot_bars import plot_3d_bars
from utils import benchmark_operation, calculate_relative_error, calculate_speedup, generate_batch, setup_logger

logger = None


def convert_to_backend_format(data: np.ndarray, backend: str) -> Any:
    """Convert numpy array to the appropriate backend format."""
    if backend in ("torch", "torch_cuda"):
        return torch.from_numpy(data).to("cuda" if backend == "torch_cuda" else "cpu")
    if backend in ("cupy", "cupy_cuda"):
        return cp.asarray(data)
    return data


def get_backend_device(backend: str):
    """Get the appropriate device object for the backend."""
    if backend in ("torch", "torch_cuda"):
        return torch.device("cuda")
    if backend in ("cupy", "cupy_cuda"):
        return cp.cuda.Device(0)
    return "cuda"


def run_benchmark(method: str, reference_batch_np: np.ndarray, source_batch_np: np.ndarray, backend1: str, backend2: str, warmup: int, runs: int, logger=None) -> dict[str, dict[str, Any]]:
    # Convert numpy arrays to backend-specific formats
    reference_batch_backend1 = convert_to_backend_format(reference_batch_np, backend1)
    source_batch_backend1 = convert_to_backend_format(source_batch_np, backend1)
    reference_batch_backend2 = convert_to_backend_format(reference_batch_np, backend2)
    source_batch_backend2 = convert_to_backend_format(source_batch_np, backend2)

    # Get appropriate device objects for each backend
    device1 = get_backend_device(backend1)
    device2 = get_backend_device(backend2)

    # Initialize normalizers with appropriate devices
    if method == "reinhard":
        stainx_backend1 = Reinhard(device=device1, backend=backend1)
        stainx_backend2 = Reinhard(device=device2, backend=backend2)
    elif method == "macenko":
        stainx_backend1 = Macenko(device=device1, backend=backend1)
        stainx_backend2 = Macenko(device=device2, backend=backend2)
    elif method == "histogram_matching":
        stainx_backend1 = HistogramMatching(device=device1, backend=backend1)
        stainx_backend2 = HistogramMatching(device=device2, backend=backend2)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Fit with reference batch (first image)
    stainx_reference1 = reference_batch_backend1[:1]
    stainx_reference2 = reference_batch_backend2[:1]

    stainx_backend1.fit(stainx_reference1)
    stainx_backend2.fit(stainx_reference2)

    # Benchmark transform operations
    stainx_backend1_result = benchmark_operation(lambda: stainx_backend1.transform(source_batch_backend1), warmup, runs, "cuda", logger)
    stainx_backend2_result = benchmark_operation(lambda: stainx_backend2.transform(source_batch_backend2), warmup, runs, "cuda", logger)

    return {"backend1": stainx_backend1_result, "backend2": stainx_backend2_result}


def main():
    parser = argparse.ArgumentParser(description="Benchmark two StainX backends against each other")
    parser.add_argument("--method", type=str, required=True, choices=["reinhard", "macenko", "histogram_matching"], help="Normalization method to benchmark (reinhard, macenko, or histogram_matching)")
    parser.add_argument("--image-size", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512], help="Image sizes to test (single number per size, creates square images: size x size)")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", nargs="+", type=int, default=[32, 64, 128, 256, 512], help="Batch sizes for images (can specify multiple)")
    parser.add_argument("--backend1", type=str, default="torch_cuda", choices=["torch", "torch_cuda", "cupy", "cupy_cuda"], help="First backend to benchmark")
    parser.add_argument("--backend2", type=str, default="torch", choices=["torch", "torch_cuda", "cupy", "cupy_cuda"], help="Second backend to benchmark")
    parser.add_argument("--plot-path", action="store_true", help="Generate and save the 3D speedup plot")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires CUDA.")
        sys.exit(1)

    device = "cuda"

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    global logger
    logger = setup_logger(filename=os.path.join(os.path.dirname(__file__), "logs", f"stainx_backend_benchmark_{current_time}.log"), verbose=True)

    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Backend 1: {args.backend1}")
    logger.info(f"Backend 2: {args.backend2}")
    logger.info(f"Batch sizes: {args.batch_size}")
    logger.info(f"Warmup runs: {args.warmup}, Benchmark runs: {args.runs}")
    logger.info("")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_sizes = [(size, size) for size in args.image_size]
    # Format method name: "histogram_matching" -> "Histogram Matching", "reinhard" -> "Reinhard"
    method_name = " ".join(word.capitalize() for word in args.method.split("_"))
    total_tests = len(args.batch_size) * len(image_sizes)
    current_test = 0
    result_tables = []
    speedup_matrix = np.zeros((len(args.batch_size), len(image_sizes)))

    logger.info("Starting benchmark...")
    logger.info("=" * 100)

    for batch_idx, batch_size in enumerate(args.batch_size):
        result_table = PrettyTable()
        result_table.title = f"{method_name} Benchmark (CUDA, batch={batch_size})"
        result_table.field_names = ["Image Size (HxW)", f"{args.backend1} (img/s)", f"{args.backend2} (img/s)", "Speedup", "Relative Error"]

        for img_idx, (height, width) in enumerate(image_sizes):
            logger.info(f"[{current_test}/{total_tests}] Testing: batch={batch_size}, size={height}x{width}")

            # Generate data as torch tensors first, then convert to numpy
            reference_batch_torch = generate_batch(batch_size, height, width, args.channels, seed=args.seed, device=device)
            source_batch_torch = generate_batch(batch_size, height, width, args.channels, seed=args.seed + 1, device=device)

            # Convert to numpy arrays
            reference_batch_np = reference_batch_torch.cpu().numpy()
            source_batch_np = source_batch_torch.cpu().numpy()

            results = run_benchmark(args.method, reference_batch_np, source_batch_np, args.backend1, args.backend2, args.warmup, args.runs, logger)

            backend1_ips = batch_size * 1000 / results["backend1"]["time_ms"] if results["backend1"]["success"] and results["backend1"]["time_ms"] > 0 else 0.0
            backend2_ips = batch_size * 1000 / results["backend2"]["time_ms"] if results["backend2"]["success"] and results["backend2"]["time_ms"] > 0 else 0.0

            speedup_numeric = backend1_ips / backend2_ips if backend2_ips > 0 and backend2_ips != float("inf") and backend1_ips != float("inf") else 0.0
            speedup_matrix[batch_idx, img_idx] = speedup_numeric

            speedup = calculate_speedup(backend2_ips, backend1_ips)
            logger.info(f"[{current_test}/{total_tests}] Speedup ({args.backend1} vs {args.backend2}): {speedup}")

            if results["backend1"]["result"] is not None and results["backend2"]["result"] is not None:
                # Convert results to torch tensors for error calculation
                backend1_result = results["backend1"]["result"][0]
                backend2_result = results["backend2"]["result"][0]

                # Convert cupy arrays to numpy, then to torch if needed
                if isinstance(backend1_result, cp.ndarray):
                    backend1_result = torch.from_numpy(cp.asnumpy(backend1_result))
                elif not isinstance(backend1_result, torch.Tensor):
                    backend1_result = torch.from_numpy(backend1_result)

                if isinstance(backend2_result, cp.ndarray):
                    backend2_result = torch.from_numpy(cp.asnumpy(backend2_result))
                elif not isinstance(backend2_result, torch.Tensor):
                    backend2_result = torch.from_numpy(backend2_result)

                # Ensure we're comparing the first image in the batch
                if backend1_result.ndim == 4:
                    backend1_result = backend1_result[0]
                if backend2_result.ndim == 4:
                    backend2_result = backend2_result[0]

                backend1_result = backend1_result.cpu().float()
                backend2_result = backend2_result.cpu().float()
                rel_error = calculate_relative_error(backend1_result, backend2_result, logger)
            else:
                rel_error = float("inf")

            result_table.add_row([f"{height}x{width}", f"{backend1_ips:.0f}" if results["backend1"]["success"] else "ERROR", f"{backend2_ips:.0f}" if results["backend2"]["success"] else "ERROR", speedup, f"{rel_error:.6f}" if rel_error != float("inf") else "N/A"])
            current_test += 1

        result_tables.append(result_table)

    logger.info("=" * 100)
    logger.info("Benchmark Results:")
    logger.info("")
    for table in result_tables:
        logger.info(table)
        logger.info("")

    if args.plot_path:
        logger.info("Generating 3D speedup plot...")
        plot_3d_bars(
            z=speedup_matrix, x=np.array(args.image_size), y=np.array(args.batch_size), xlabel="Image Size (HxW)", ylabel="Batch Size", zlabel=f"Speedup ({args.backend1} / {args.backend2})", label_fontsize=12, save_path=f"speedup_{args.backend1}_vs_{args.backend2}_plot_{method_name}_cuda_{current_time}.pdf", show=False
        )
        logger.info(f"3D plot saved to: speedup_{args.backend1}_vs_{args.backend2}_plot_{method_name}_cuda_{current_time}.pdf")

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
