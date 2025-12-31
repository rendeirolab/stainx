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
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stainx import Macenko, Reinhard

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from plot_bars import plot_2d_bars
from utils import benchmark_operation, calculate_relative_error, calculate_speedup, generate_batch, setup_logger

logger = None


def run_benchmark(method: str, reference_image: torch.Tensor, source_image: torch.Tensor, device: str, warmup: int, runs: int, logger=None) -> dict[str, dict[str, Any]]:
    if method == "reinhard":
        stainx_norm = Reinhard(device=device)
        torchstain_norm = TorchReinhardNormalizer()
    elif method == "macenko":
        stainx_norm = Macenko(device=device)
        torchstain_norm = TorchMacenkoNormalizer()
    else:
        raise ValueError(f"Invalid method: {method}")

    stainx_reference = reference_image.unsqueeze(0)
    torchstain_reference = reference_image.to(device)

    stainx_norm.fit(stainx_reference)
    torchstain_norm.fit(torchstain_reference)

    torchstain_source = source_image.to(device)

    stainx_result = benchmark_operation(lambda: stainx_norm.transform(source_image.unsqueeze(0)), warmup, runs, device, logger)
    torchstain_result = benchmark_operation(lambda: torchstain_norm.normalize(torchstain_source), warmup, runs, device, logger)

    return {"stainx": stainx_result, "torchstain": torchstain_result}


def main():
    parser = argparse.ArgumentParser(description="Benchmark StainX runtime against torchstain")
    parser.add_argument("--method", type=str, required=True, choices=["reinhard", "macenko"], help="Normalization method to benchmark")
    parser.add_argument("--image-size", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512], help="Image sizes to test (single number per size, creates square images: size x size)")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to use")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot-path", action="store_true", help="Generate and save the 2D speedup plot")

    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    global logger
    logger = setup_logger(filename=os.path.join(os.path.dirname(__file__), "logs", f"torchstain_benchmark_{current_time}.log"), verbose=True)

    logger.info(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}")
    logger.info(f"Warmup runs: {args.warmup}, Benchmark runs: {args.runs}")
    logger.info("")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    image_sizes = [(size, size) for size in args.image_size]
    method_name = args.method.capitalize()
    total_tests = len(image_sizes)

    result_table = PrettyTable()
    result_table.title = f"{method_name} Benchmark ({device.upper()})"
    result_table.field_names = ["Image Size (HxW)", "StainX (img/s)", "TorchStain (img/s)", "Speedup", "Relative Error"]

    # Store speedup values for plotting
    speedup_values = np.zeros(len(image_sizes))

    logger.info("Starting benchmark...")
    logger.info("=" * 100)

    for img_idx, (height, width) in enumerate(image_sizes):
        current_test = img_idx + 1
        logger.info(f"[{current_test}/{total_tests}] Testing: size={height}x{width}")

        reference_image = generate_batch(1, height, width, args.channels, seed=args.seed, device=device).squeeze(0)
        source_image = generate_batch(1, height, width, args.channels, seed=args.seed + 1, device=device).squeeze(0)

        results = run_benchmark(args.method, reference_image, source_image, device, args.warmup, args.runs, logger)

        stainx_ips = 1000 / results["stainx"]["time_ms"] if results["stainx"]["success"] and results["stainx"]["time_ms"] > 0 else 0.0
        comparison_ips = 1000 / results["torchstain"]["time_ms"] if results["torchstain"]["success"] and results["torchstain"]["time_ms"] > 0 else 0.0

        speedup_numeric = stainx_ips / comparison_ips if comparison_ips > 0 and comparison_ips != float("inf") and stainx_ips != float("inf") else 0.0
        speedup_values[img_idx] = speedup_numeric

        speedup = calculate_speedup(comparison_ips, stainx_ips)
        logger.info(f"[{current_test}/{total_tests}] Speedup: {speedup}")

        if results["stainx"]["result"] is not None and results["torchstain"]["result"] is not None:
            torchstain_result = (t := (results["torchstain"]["result"][0] if isinstance(results["torchstain"]["result"], tuple) else results["torchstain"]["result"]), t.permute(2, 0, 1) if len(t.shape) == 3 and t.shape[2] == 3 else t)[-1].float()
            stainx_result = results["stainx"]["result"].squeeze(0).cpu().float()
            rel_error = calculate_relative_error(stainx_result, torchstain_result, logger)
        else:
            rel_error = float("inf")

        result_table.add_row([f"{height}x{width}", f"{stainx_ips:.0f}" if results["stainx"]["success"] else "ERROR", f"{comparison_ips:.0f}" if results["torchstain"]["success"] else "ERROR", speedup, f"{rel_error:.6f}" if rel_error != float("inf") else "N/A"])

    logger.info("=" * 100)
    logger.info("Benchmark Results:")
    logger.info("")
    logger.info(result_table)
    logger.info("")

    if args.plot_path:
        logger.info("Generating 2D speedup plot...")
        plot_2d_bars(x=np.array(args.image_size), y=speedup_values, xlabel="Image Size (HxW)", ylabel="Speedup (StainX / TorchStain)", label_fontsize=12, color="#00aecc", save_path=f"speedup_over_torchstain_plot_{method_name}_{device}_{current_time}.pdf", show=False)
        logger.info(f"2D plot saved to: speedup_over_torchstain_plot_{method_name}_{device}_{current_time}.pdf")

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
