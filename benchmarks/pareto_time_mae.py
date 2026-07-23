# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""MAE vs throughput Pareto plot across packages / devices.

Sources: ``examples/data/target.png`` + ``test_1..5.png`` (resized, repeated to batch).
MAE baselines (CPU): Macenko→torchstain | Reinhard→StainTools | HM→skimage.
Shapes = device (CPU ○ / GPU □), colors = method; dashed = Pareto; dotted = max MAE (255).
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import slideflow.norm as sf_norm
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.exposure import match_histograms
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

BATCH, H, W = 128, 256, 256
WARMUP, RUNS = 5, 20
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
OUT = Path(__file__).resolve().parent / "logs" / "pareto_time_mae.png"

METHOD_COLOR = {"Macenko": "#96324b", "Reinhard": "#e0a02e", "HistogramMatching": "#c4788a"}
DEVICE_MARKER = {"CPU": "o", "GPU": "s"}
MAE_FLOOR = 1e-4
MAE_MAX = 255.0  # vertical reference: max possible grey-level MAE


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_ms(fn, device: torch.device) -> float:
    for _ in range(WARMUP):
        fn()
        _sync(device)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        fn()
        _sync(device)
    return (time.perf_counter() - t0) / RUNS * 1000.0


def _mae(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return float((pred.float() - ref.float()).abs().mean())


def _as_nchw_float(x) -> torch.Tensor:
    """Batch → NCHW float CPU. Accepts tensor, ndarray, or list of images."""
    if isinstance(x, (list, tuple)):
        parts = [_as_nchw_float(xi) for xi in x]
        return torch.stack([p[0] if p.dim() == 4 else p for p in parts], 0)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    x = x.detach().cpu().float()
    if x.dim() == 3:
        return (x.permute(2, 0, 1) if x.shape[-1] == 3 else x).unsqueeze(0)
    if x.dim() == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
        return x.permute(0, 3, 1, 2)
    return x


def _to_nhwc(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu()
    return x.permute(0, 2, 3, 1).numpy() if x.dim() == 4 else x.permute(1, 2, 0).numpy()


def _load_he_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``target.png`` + ``test_1..5.png`` resized to HxW and repeated to length ``n``."""

    def load(name: str) -> torch.Tensor:
        arr = np.array(Image.open(DATA / name).convert("RGB"), copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        if tuple(t.shape[-2:]) == (H, W):
            return t
        return F.interpolate(t.float(), size=(H, W), mode="bilinear", align_corners=False).round().clamp(0, 255).to(torch.uint8)

    ref = load("target.png")
    src = torch.cat([load(f"test_{i}.png") for i in range(1, 6)], 0)
    return ref, src.repeat((n + src.shape[0] - 1) // src.shape[0], 1, 1, 1)[:n]


def _staintools_reinhard():
    """Load Reinhard only — avoid ``staintools.__init__`` (needs spams)."""
    root = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/staintools"
    if not root.is_dir():
        raise RuntimeError("staintools not installed; uv sync --group benchmark --python 3.11")
    if "staintools" not in sys.modules:
        pkg = types.ModuleType("staintools")
        pkg.__path__ = [str(root)]
        sys.modules["staintools"] = pkg
        pre = types.ModuleType("staintools.preprocessing")
        pre.__path__ = [str(root / "preprocessing")]
        sys.modules["staintools.preprocessing"] = pre
        for name, rel in (
            ("staintools.preprocessing.input_validation", "preprocessing/input_validation.py"),
            ("staintools.reinhard_color_normalizer", "reinhard_color_normalizer.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, root / rel)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
    return sys.modules["staintools.reinhard_color_normalizer"].ReinhardColorNormalizer


def _pareto_front(points: list[dict]) -> list[dict]:
    """Minimize MAE, maximize img/s."""
    front, best_tps = [], -1.0
    for p in sorted(points, key=lambda p: (p["mae"], -p["imgs_per_s"])):
        if p["imgs_per_s"] > best_tps:
            front.append(p)
            best_tps = p["imgs_per_s"]
    return front


def build_baselines(ref: torch.Tensor, src: torch.Tensor) -> dict[str, torch.Tensor]:
    ref_chw = ref.squeeze(0).float()
    ref_hwc = _to_nhwc(ref)[0]
    src_nhwc = _to_nhwc(src)
    n = src.shape[0]

    ts = TorchMacenkoNormalizer()
    ts.fit(ref_chw)
    macenko = torch.stack([_as_nchw_float(ts.normalize(src[i].float(), stains=True)[0])[0] for i in range(n)])

    st = _staintools_reinhard()()
    st.fit(ref_hwc)
    reinhard = torch.stack([torch.from_numpy(st.transform(src_nhwc[i])).permute(2, 0, 1).float() for i in range(n)])

    hm = torch.stack([torch.from_numpy(match_histograms(src_nhwc[i], ref_hwc, channel_axis=-1)).permute(2, 0, 1).float() for i in range(n)])
    return {"Macenko": macenko, "Reinhard": reinhard, "HistogramMatching": hm}


def collect_results(ref: torch.Tensor, src: torch.Tensor, baselines: dict[str, torch.Tensor]) -> list[dict]:
    results: list[dict] = []
    ref_chw = ref.squeeze(0).float()
    ref_hwc = _to_nhwc(ref)[0]
    src = src.contiguous()
    src_nhwc = _to_nhwc(src)
    src_f = src.float()
    cpu = torch.device("cpu")
    cuda = torch.device("cuda") if torch.cuda.is_available() else None

    def add(method: str, package: str, device: torch.device, fn) -> None:
        ms = _time_ms(fn, device)
        tps = BATCH * 1000.0 / ms if ms > 0 else float("inf")
        mae = _mae(_as_nchw_float(fn()), baselines[method])
        results.append({"method": method, "package": package, "device": "GPU" if device.type == "cuda" else "CPU", "time_ms": ms, "imgs_per_s": tps, "mae": mae})
        print(f"{method:<20} {package:<24} {device.type:<5}  {tps:10.1f} img/s  MAE={mae:.4f}")

    # torchstain (CPU for-loop)
    ts_m = TorchMacenkoNormalizer()
    ts_m.fit(ref_chw)
    add("Macenko", "torchstain", cpu, lambda: [ts_m.normalize(src_f[i], stains=True)[0] for i in range(BATCH)])

    ts_r = TorchReinhardNormalizer()
    ts_r.fit(ref_chw)
    add("Reinhard", "torchstain", cpu, lambda: [ts_r.normalize(src_f[i]) for i in range(BATCH)])

    # StainTools / skimage (CPU for-loop)
    st = _staintools_reinhard()()
    st.fit(ref_hwc)
    add("Reinhard", "StainTools", cpu, lambda: [st.transform(src_nhwc[i]) for i in range(BATCH)])
    add("HistogramMatching", "skimage", cpu, lambda: [match_histograms(src_nhwc[i], ref_hwc, channel_axis=-1) for i in range(BATCH)])

    # StainX (native batch; H2D outside timer)
    backends = [("torch", cpu)]
    if cuda is not None:
        backends.append(("torch", cuda))
        if CUDA_AVAILABLE:
            backends.append(("torch_cuda", cuda))

    for method, cls, kwargs in (("Macenko", Macenko, {}), ("Reinhard", Reinhard, {}), ("HistogramMatching", HistogramMatching, {"channel_axis": 1})):
        for backend, dev in backends:
            n = cls(device=dev, backend=backend, **kwargs)
            n.fit(ref.to(dev))
            x = src.to(dev)
            add(method, f"stainx[{backend}]", dev, lambda n=n, x=x: n.transform(x))

    # Slideflow
    devices = [cpu] + ([cuda] if cuda is not None else [])
    for method, sf_name in (("Reinhard", "reinhard"), ("Reinhard", "reinhard_fast"), ("Macenko", "macenko"), ("Macenko", "macenko_fast")):
        for dev in devices:
            sf = sf_norm.autoselect(sf_name)
            sf.device = str(dev)
            sf.fit(ref_hwc)
            x = src.to(dev)
            add(method, f"slideflow[{sf_name}]", dev, lambda sf=sf, x=x: sf.transform(x))

    return results


def plot(results: list[dict], path: Path, mae_max: float = MAE_MAX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    for p in results:
        x = max(p["mae"], MAE_FLOOR)
        ax.scatter(x, p["imgs_per_s"], marker=DEVICE_MARKER[p["device"]], s=110, c=METHOD_COLOR[p["method"]], edgecolors="k", linewidths=0.6, zorder=3)
        ax.annotate(p["package"], (x, p["imgs_per_s"]), textcoords="offset points", xytext=(6, 4), fontsize=7, alpha=0.85)

    for method, color in METHOD_COLOR.items():
        front = _pareto_front([p for p in results if p["method"] == method])
        label = f"Pareto {method if method != 'HistogramMatching' else 'HM'}"
        xs = [max(p["mae"], MAE_FLOOR) for p in front]
        ys = [p["imgs_per_s"] for p in front]
        if len(front) >= 2:
            ax.plot(xs, ys, "--", color=color, linewidth=1.6, zorder=2, label=label)
        elif len(front) == 1:
            ax.scatter(xs, ys, marker="x", c=color, s=80, zorder=2, label=label)

    ax.axvline(mae_max, color="#333333", linestyle=":", linewidth=1.8, zorder=1, label=f"max MAE ({mae_max:.0f})")

    for method, color in METHOD_COLOR.items():
        ax.scatter([], [], marker="o", c=color, s=100, label=method, edgecolors="k")
    for device, marker in DEVICE_MARKER.items():
        ax.scatter([], [], marker=marker, c="gray", s=100, label=device, edgecolors="k")

    ax.set_xlabel("MAE vs stable baseline  ↓ better")
    ax.set_ylabel("Throughput (img/s)  ↑ better")
    ax.set_title(f"Stain normalization: MAE vs throughput  (batch={BATCH}, {H}x{W}, warmup={WARMUP}, runs={RUNS})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {path}")


def main() -> None:
    print(f"Batch={BATCH} | {H}x{W} | warmup={WARMUP} runs={RUNS}")
    print("Baselines: Macenko→torchstain | Reinhard→StainTools | HM→skimage")
    print(f"Images: {DATA.name}/target + test_1..5 → {H}x{W}, ×{BATCH}\n")

    ref, src = _load_he_batch(BATCH)
    print("Building baselines...")
    baselines = build_baselines(ref, src)
    results = collect_results(ref, src, baselines)
    if not results:
        raise SystemExit("No results to plot.")
    plot(results, OUT, MAE_MAX)


if __name__ == "__main__":
    main()
