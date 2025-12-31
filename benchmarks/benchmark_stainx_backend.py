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

import numpy as np
import torch
from prettytable import PrettyTable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stainx import Macenko, Reinhard

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from plot_bars import plot_3d_bars
from utils import benchmark_operation, calculate_relative_error, calculate_speedup, generate_batch, setup_logger

logger = None


def run_benchmark(method: str, reference_batch: torch.Tensor, source_batch: torch.Tensor, backend: str, warmup: int, runs: int, logger=None) -> dict[str, dict[str, Any]]:
    if method == "reinhard":
        stainx_cuda_backend = Reinhard(device="cuda", backend=backend)
        stainx_pytorch_cuda = Reinhard(device="cuda", backend="pytorch")
    elif method == "macenko":
        stainx_cuda_backend = Macenko(device="cuda", backend=backend)
        stainx_pytorch_cuda = Macenko(device="cuda", backend="pytorch")
    else:
        raise ValueError(f"Invalid method: {method}")

    stainx_reference = reference_batch[:1]

    stainx_cuda_backend.fit(stainx_reference)
    stainx_pytorch_cuda.fit(stainx_reference)

    stainx_cuda_backend_result = benchmark_operation(lambda: stainx_cuda_backend.transform(source_batch), warmup, runs, "cuda", logger)
    stainx_pytorch_cuda_result = benchmark_operation(lambda: stainx_pytorch_cuda.transform(source_batch), warmup, runs, "cuda", logger)

    return {"cuda_backend": stainx_cuda_backend_result, "pytorch_cuda": stainx_pytorch_cuda_result}


def main():
    parser = argparse.ArgumentParser(description="Benchmark StainX CUDA backend against PyTorch CUDA device")
    parser.add_argument("--method", type=str, required=True, choices=["reinhard", "macenko"], help="Normalization method to benchmark (reinhard or macenko)")
    parser.add_argument("--image-size", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512], help="Image sizes to test (single number per size, creates square images: size x size)")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", nargs="+", type=int, default=[32, 64, 128, 256, 512], help="Batch sizes for images (can specify multiple)")
    parser.add_argument("--backend", type=str, default="cuda", help="CUDA backend to use (e.g., 'cuda', 'cudnn')")
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
    logger.info(f"CUDA Backend: {args.backend}")
    logger.info(f"Batch sizes: {args.batch_size}")
    logger.info(f"Warmup runs: {args.warmup}, Benchmark runs: {args.runs}")
    logger.info("")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_sizes = [(size, size) for size in args.image_size]
    method_name = args.method.capitalize()
    total_tests = len(args.batch_size) * len(image_sizes)
    current_test = 0
    result_tables = []
    speedup_matrix = np.zeros((len(args.batch_size), len(image_sizes)))

    logger.info("Starting benchmark...")
    logger.info("=" * 100)

    for batch_idx, batch_size in enumerate(args.batch_size):
        result_table = PrettyTable()
        result_table.title = f"{method_name} Benchmark (CUDA, batch={batch_size})"
        result_table.field_names = ["Image Size (HxW)", "CUDA Backend (img/s)", "PyTorch CUDA (img/s)", "Speedup", "Relative Error"]

        for img_idx, (height, width) in enumerate(image_sizes):
            logger.info(f"[{current_test}/{total_tests}] Testing: batch={batch_size}, size={height}x{width}")

            reference_batch = generate_batch(batch_size, height, width, args.channels, seed=args.seed, device=device)
            source_batch = generate_batch(batch_size, height, width, args.channels, seed=args.seed + 1, device=device)

            results = run_benchmark(args.method, reference_batch, source_batch, args.backend, args.warmup, args.runs, logger)

            cuda_backend_ips = batch_size * 1000 / results["cuda_backend"]["time_ms"] if results["cuda_backend"]["success"] and results["cuda_backend"]["time_ms"] > 0 else 0.0
            pytorch_cuda_ips = batch_size * 1000 / results["pytorch_cuda"]["time_ms"] if results["pytorch_cuda"]["success"] and results["pytorch_cuda"]["time_ms"] > 0 else 0.0

            speedup_numeric = cuda_backend_ips / pytorch_cuda_ips if pytorch_cuda_ips > 0 and pytorch_cuda_ips != float("inf") and cuda_backend_ips != float("inf") else 0.0
            speedup_matrix[batch_idx, img_idx] = speedup_numeric

            speedup = calculate_speedup(pytorch_cuda_ips, cuda_backend_ips)
            logger.info(f"[{current_test}/{total_tests}] Speedup: {speedup}")

            if results["cuda_backend"]["result"] is not None and results["pytorch_cuda"]["result"] is not None:
                cuda_backend_result = results["cuda_backend"]["result"][0].cpu().float()
                pytorch_cuda_result = results["pytorch_cuda"]["result"][0].cpu().float()
                rel_error = calculate_relative_error(cuda_backend_result, pytorch_cuda_result, logger)
            else:
                rel_error = float("inf")

            result_table.add_row([f"{height}x{width}", f"{cuda_backend_ips:.0f}" if results["cuda_backend"]["success"] else "ERROR", f"{pytorch_cuda_ips:.0f}" if results["pytorch_cuda"]["success"] else "ERROR", speedup, f"{rel_error:.6f}" if rel_error != float("inf") else "N/A"])
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
        plot_3d_bars(z=speedup_matrix, x=np.array(args.image_size), y=np.array(args.batch_size), xlabel="Image Size (HxW)", ylabel="Batch Size", zlabel="Speedup (CUDA Backend / PyTorch CUDA)", label_fontsize=12, save_path=f"speedup_cuda_backend_vs_pytorch_plot_{method_name}_cuda_{current_time}.pdf", show=False)
        logger.info(f"3D plot saved to: speedup_cuda_backend_vs_pytorch_plot_{method_name}_cuda_{current_time}.pdf")

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
