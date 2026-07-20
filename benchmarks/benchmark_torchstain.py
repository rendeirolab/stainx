# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""StainX batched transform vs torchstain (Python for-loop over the batch)."""

from __future__ import annotations

import time

import torch
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

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


def _bench_method(name: str, stainx_cls, torchstain_cls, device: torch.device) -> None:
    print(f"\n=== {name} (device={device}) ===")
    print(f"{'batch':>6} {'HxW':>9} {'stainx ms':>12} {'torchstain ms':>14} {'speedup':>9} {'stainx img/s':>13} {'torchstain img/s':>17}")

    for batch_size in BATCH_SIZES:
        for size in IMAGE_SIZES:
            ref = _rand_batch(1, size, size, device, SEED)
            src = _rand_batch(batch_size, size, size, device, SEED + 1)

            stainx = stainx_cls(device=device)
            stainx.fit(ref)

            # torchstain keeps its color matrices on CPU and has no batch API.
            torchstain = torchstain_cls()
            torchstain.fit(ref.squeeze(0).float().cpu())
            src_ts = src.float().cpu()

            def stainx_step(_stainx=stainx, _src=src):
                return _stainx.transform(_src)

            def torchstain_step(_torchstain=torchstain, _src_ts=src_ts):
                outs = []
                for i in range(_src_ts.shape[0]):
                    out = _torchstain.normalize(_src_ts[i])
                    outs.append(out[0] if isinstance(out, tuple) else out)
                return outs

            stainx_ms = _time_ms(stainx_step, device)
            torchstain_ms = _time_ms(torchstain_step, device)
            speedup = torchstain_ms / stainx_ms if stainx_ms > 0 else float("nan")
            print(f"{batch_size:6d} {size:4d}x{size:<4d} {stainx_ms:12.1f} {torchstain_ms:14.1f} {speedup:8.2f}x {batch_size / stainx_ms * 1000:13.0f} {batch_size / torchstain_ms * 1000:17.0f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"warmup={WARMUP}, runs={RUNS}")
    print("StainX: one batched transform on device | torchstain: CPU for-loop over batch")

    _bench_method("Reinhard", Reinhard, TorchReinhardNormalizer, device)
    _bench_method("Macenko", Macenko, TorchMacenkoNormalizer, device)


if __name__ == "__main__":
    main()
