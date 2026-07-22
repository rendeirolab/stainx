# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""StainX batched transform vs Slideflow batched transform (same device)."""

from __future__ import annotations

import time

import slideflow.norm as sf_norm
import torch

from stainx import Macenko, Reinhard

BATCH_SIZES = [32, 64, 128]
IMAGE_SIZES = [128, 256, 512]
WARMUP = 10
RUNS = 30
SEED = 42


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_ms(fn, device: torch.device, warmup: int = WARMUP, runs: int = RUNS) -> float:
    for _ in range(warmup):
        fn()
        _sync(device)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
        _sync(device)
    return (time.perf_counter() - t0) / runs * 1000.0


def _rand_batch(n: int, h: int, w: int, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.rand(n, 3, h, w, device=device) * 255).round().to(torch.uint8)


def _bench_method(name: str, stainx_cls, slideflow_method: str, device: torch.device) -> None:
    print(f"\n=== {name} (device={device}) ===")
    print(f"{'batch':>6} {'HxW':>9} {'stainx ms':>12} {'slideflow ms':>14} {'speedup':>9} {'stainx img/s':>13} {'slideflow img/s':>16}")

    for batch_size in BATCH_SIZES:
        for size in IMAGE_SIZES:
            ref = _rand_batch(1, size, size, device, SEED)
            src = _rand_batch(batch_size, size, size, device, SEED + 1)

            stainx = stainx_cls(device=device)
            stainx.fit(ref)

            # Slideflow fit wants HWC numpy; transform accepts a batched tensor on device.
            slideflow = sf_norm.autoselect(slideflow_method)
            slideflow.device = str(device)
            slideflow.fit(ref[0].permute(1, 2, 0).cpu().numpy())

            def stainx_step(_stainx=stainx, _src=src):
                return _stainx.transform(_src)

            def slideflow_step(_slideflow=slideflow, _src=src):
                return _slideflow.transform(_src)

            stainx_ms = _time_ms(stainx_step, device)
            slideflow_ms = _time_ms(slideflow_step, device)
            speedup = slideflow_ms / stainx_ms if stainx_ms > 0 else float("nan")
            print(f"{batch_size:6d} {size:4d}x{size:<4d} {stainx_ms:12.1f} {slideflow_ms:14.1f} {speedup:8.2f}x {batch_size / stainx_ms * 1000:13.0f} {batch_size / slideflow_ms * 1000:16.0f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"warmup={WARMUP}, runs={RUNS}")
    print("StainX: one batched transform on device | Slideflow: batched transform on same device")

    _bench_method("Reinhard", Reinhard, "reinhard_fast", device)
    _bench_method("Macenko", Macenko, "macenko_fast", device)


if __name__ == "__main__":
    main()
