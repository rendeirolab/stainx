#!/usr/bin/env python3
# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter


def main():
    parser = argparse.ArgumentParser(description="Simple benchmark for a single StainX method")
    parser.add_argument("method", type=str, choices=["reinhard", "macenko", "histogram_matching"], help="StainX method to benchmark")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to use")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}")
    print(f"Method: {args.method}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.height}x{args.width}")
    print()

    # Set seeds
    torch.manual_seed(args.seed)

    # Generate test images
    reference_image = (torch.rand(args.batch_size, args.channels, args.height, args.width, device=device) * 255).round().to(torch.uint8)
    source_image = (torch.rand(args.batch_size, args.channels, args.height, args.width, device=device) * 255).round().to(torch.uint8)

    # Create normalizer based on method
    if args.method == "reinhard":
        normalizer = Reinhard(device=device)
    elif args.method == "macenko":
        normalizer = Macenko(device=device)
    elif args.method == "histogram_matching":
        converter = ChannelFormatConverter(channel_axis=0)
        reference_image = converter.prepare_for_normalizer(reference_image)
        source_image = converter.prepare_for_normalizer(source_image)
        normalizer = HistogramMatching(device=device, channel_axis=0)

    # Fit on reference image
    print("Fitting normalizer...")
    normalizer.fit(reference_image)

    # Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Run transform once
    print("Running transform...")
    start_time = time.time()
    for _ in range(args.runs):
        result = normalizer.transform(source_image)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_time = ((time.time() - start_time) * 1000) / args.runs

    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    print(f"Time: {elapsed_time:.3f} ms")
    print(f"Images per second: {args.batch_size * 1000 / elapsed_time:.2f}")


if __name__ == "__main__":
    main()
