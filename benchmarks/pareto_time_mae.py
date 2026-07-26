# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

"""MAE vs throughput Pareto plot across packages / devices.

Sources: ``examples/data/target.png`` + ``test_1..5.png`` (resized, repeated to batch).
MAE baselines (CPU): Macenko→torchstain | Reinhard→StainTools | HM→skimage.
Those reference libs (StainTools, skimage) are omitted from the plot; torchstain is kept.

Peers surveyed for Macenko / Reinhard / HM (installable + working):
  stainx, torchstain (+modified Reinhard), slideflow, color-matcher, tiatoolbox,
  torch-staintools, wsi-normalizer (Reinhard), colortrans, color_transfer.
Skipped: colorcast (HM wraps skimage); StainTools/wsi Macenko (spams Fortran failure);
  stainlib/histomicstk/pathml (not pip-installable here); Vahadane-only tools.

One panel per method (horizontal); color = package; shapes = device (CPU ○ / GPU ★); dashed = Pareto; dotted = max MAE (255).
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import slideflow.norm as sf_norm
import torch
import torch.nn.functional as F  # noqa: N812
from adjustText import adjust_text
from color_matcher import ColorMatcher
from color_matcher.hist_matcher import HistogramMatcher as ColorMatcherHM
from color_transfer import color_transfer as pyimagesearch_color_transfer
from PIL import Image
from skimage.exposure import match_histograms
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.backends.torch_cuda_backend import CUDA_AVAILABLE

BATCH, H, W = 128, 256, 256
WARMUP, RUNS = 30, 100
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
OUT = Path(__file__).resolve().parent / "logs" / "pareto_time_mae.png"

METHODS = ("Macenko", "Reinhard", "HistogramMatching")
DEVICE_MARKER = {"CPU": "o", "GPU": "*"}
# Color encodes package. #9b2335 dominates for stainx; peers get distinct accents.
PACKAGE_COLOR = {
    "torchstain": "#5c7a8a",
    "torchstain[modified]": "#7a9aab",
    "stainx[torch]": "#9b2335",
    "stainx[torch_cuda]": "#6e1826",
    "slideflow": "#d4a017",
    "slideflow[fast]": "#b8860b",
    "color-matcher": "#2f6f4e",
    "tiatoolbox": "#6b4c9a",
    "torch-staintools": "#c45c26",
    "wsi-normalizer": "#2a6f97",
    "colortrans": "#8b5a2b",
    "color_transfer": "#4a6741",
}
PACKAGE_COLOR_FALLBACK = "#555555"
PARETO_COLOR = "#444444"
# MAE reference used for error (build_baselines). Omit pure reference libs from the
# scatter; keep torchstain plotted so the package is visible (Macenko MAE≈0 by construction).
MAE_BASELINE_PACKAGE = {"Macenko": "torchstain", "Reinhard": "StainTools", "HistogramMatching": "skimage"}
PLOT_EXCLUDE = {(method, package) for method, package in MAE_BASELINE_PACKAGE.items() if package != "torchstain"}
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
        for name, rel in (("staintools.preprocessing.input_validation", "preprocessing/input_validation.py"), ("staintools.reinhard_color_normalizer", "reinhard_color_normalizer.py")):
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

    # StainTools / skimage / color-matcher (CPU for-loop)
    st = _staintools_reinhard()()
    st.fit(ref_hwc)
    add("Reinhard", "StainTools", cpu, lambda: [st.transform(src_nhwc[i]) for i in range(BATCH)])
    add("HistogramMatching", "skimage", cpu, lambda: [match_histograms(src_nhwc[i], ref_hwc, channel_axis=-1) for i in range(BATCH)])
    # color-matcher: photo Reinhard + independent HM (no Macenko). Baselines stay StainTools / skimage.
    ref_f64 = ref_hwc.astype(np.float64)
    cm = ColorMatcher()
    cm_hm = ColorMatcherHM()
    add("Reinhard", "color-matcher", cpu, lambda: [cm.transfer(src=src_nhwc[i].astype(np.float64), ref=ref_f64, method="reinhard") for i in range(BATCH)])
    add("HistogramMatching", "color-matcher", cpu, lambda: [cm_hm.hist_match(src=src_nhwc[i].astype(np.float64), ref=ref_f64) for i in range(BATCH)])

    # torchstain modified Reinhard (Roy et al.)
    ts_rm = TorchReinhardNormalizer(method="modified")
    ts_rm.fit(ref_chw)
    add("Reinhard", "torchstain[modified]", cpu, lambda: [ts_rm.normalize(src_f[i]) for i in range(BATCH)])

    # colortrans / color_transfer — general Reinhard color transfer (not pathology-specific).
    import colortrans

    add("Reinhard", "colortrans", cpu, lambda: [colortrans.transfer_reinhard(src_nhwc[i], ref_hwc) for i in range(BATCH)])
    ref_bgr = cv2.cvtColor(ref_hwc, cv2.COLOR_RGB2BGR)
    add(
        "Reinhard",
        "color_transfer",
        cpu,
        lambda: [
            cv2.cvtColor(pyimagesearch_color_transfer(ref_bgr, cv2.cvtColor(src_nhwc[i], cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB) for i in range(BATCH)
        ],
    )

    # tiatoolbox (StainTools-derived Reinhard / Macenko)
    from tiatoolbox.tools import stainnorm

    for method in ("Reinhard", "Macenko"):
        tn = stainnorm.get_normalizer(method)
        tn.fit(ref_hwc)
        add(method, "tiatoolbox", cpu, lambda tn=tn: [tn.transform(src_nhwc[i]) for i in range(BATCH)])

    # wsi-normalizer (CPU for-loop; Macenko needs working SPAMS Fortran arrays — skip)
    from wsi_normalizer import ReinhardNormalizer as WsiReinhard

    wsi_r = WsiReinhard()
    wsi_r.fit(ref_hwc)
    add("Reinhard", "wsi-normalizer", cpu, lambda: [wsi_r.transform(src_nhwc[i]) for i in range(BATCH)])

    # torch-staintools (native batch; float [0,1] NCHW)
    from torch_staintools.normalizer import NormalizerBuilder

    tst_devices = [cpu] + ([cuda] if cuda is not None else [])
    for method, builder in (("Macenko", "macenko"), ("Reinhard", "reinhard")):
        for dev in tst_devices:
            n = NormalizerBuilder.build(builder, use_cache=False).to(dev).eval()
            ref01 = (ref.float() / 255.0).to(dev)
            src01 = (src.float() / 255.0).to(dev)
            with torch.no_grad():
                n.fit(ref01)

            def _tst_fn(n=n, x=src01):
                with torch.no_grad():
                    return n(x) * 255.0

            add(method, "torch-staintools", dev, _tst_fn)

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


def _panel_label(package: str, method: str) -> str:
    """Full package names; drop redundant method stem from slideflow tags (panel title has it)."""
    stem = {"Macenko": "macenko", "Reinhard": "reinhard", "HistogramMatching": "histogram"}.get(method, method.lower())
    if package == f"slideflow[{stem}_fast]":
        return "slideflow[fast]"
    if package == f"slideflow[{stem}]":
        return "slideflow"
    return package


def _package_color(package: str, method: str) -> str:
    return PACKAGE_COLOR.get(_panel_label(package, method), PACKAGE_COLOR_FALLBACK)


def plot(results: list[dict], path: Path, mae_max: float = MAE_MAX) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop pure MAE-definition libs (StainTools / skimage). Keep torchstain + color-matcher + others.
    results = [p for p in results if (p["method"], p["package"]) not in PLOT_EXCLUDE]
    if not results:
        raise SystemExit("No results to plot after excluding MAE baselines.")

    # Bioinformatics / OUP figure typography: Arial or Helvetica, ≥7 pt.
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.bottom": True,
        "ytick.left": True,
        "axes.labelpad": 2,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), dpi=350, sharex=False, sharey=False)
    fig.patch.set_facecolor("white")
    x_ticks_all = [MAE_FLOOR, 0.001, 0.01, 0.1, 1, 10, 100, mae_max]

    def _plain_num(v: float, _pos=None) -> str:
        if abs(v - mae_max) / mae_max < 1e-6:
            return f"{mae_max:.0f}"
        if v >= 1:
            return f"{v:.0f}"
        return f"{v:.4g}"

    for ax, method in zip(axes, METHODS, strict=True):
        pts = [p for p in results if p["method"] == method]
        ax.set_facecolor("white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", color="#e8e8e8", linewidth=0.7)
        ax.grid(False, which="minor")
        ax.set_title(method, fontsize=9, pad=4)

        # Plot markers at true (mae, tps) so the Pareto line matches the points.
        # (Jittering markers previously made torchstain look left of stainx and broke the front.)
        xs_pts = [max(p["mae"], MAE_FLOOR) for p in pts]
        ys_pts = [p["imgs_per_s"] for p in pts]

        texts = []
        for p, x, y in zip(pts, xs_pts, ys_pts, strict=True):
            color = _package_color(p["package"], method)
            marker = DEVICE_MARKER[p["device"]]
            # Stars read smaller than circles at the same ``s``.
            size = 110 if marker == "*" else 55
            ax.scatter(x, y, marker=marker, s=size, c=color, edgecolors="black", linewidths=0.7, zorder=3, alpha=0.9)
            texts.append(ax.text(x, y, _panel_label(p["package"], method), fontsize=6.5, color="#222222", alpha=0.95, bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}, zorder=4))

        ax.axvline(mae_max, color="#555555", linestyle=":", linewidth=1.4, zorder=1, alpha=0.85)

        front = _pareto_front(pts)
        xs = [max(p["mae"], MAE_FLOOR) for p in front]
        ys = [p["imgs_per_s"] for p in front]
        if len(front) >= 2:
            ax.plot(xs, ys, "--", color=PARETO_COLOR, linewidth=1.3, zorder=2, alpha=0.75)
        elif len(front) == 1:
            ax.scatter(xs, ys, marker="x", c=PARETO_COLOR, s=45, zorder=2)

        # Always show max MAE (255) on every panel, not only when data reach it.
        x_min = min(xs_pts)
        y_min, y_max = min(ys_pts), max(ys_pts)
        ax.set_xlim(max(x_min / 4, MAE_FLOOR * 0.5), mae_max * 1.25)
        ax.set_ylim(y_min / 2.2, y_max * 2.5)
        ticks = [t for t in x_ticks_all if ax.get_xlim()[0] <= t <= mae_max * 1.25]
        if mae_max not in ticks:
            ticks.append(mae_max)
        ax.set_xticks(sorted(set(ticks)))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_plain_num))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("Mean absolute error vs stable baseline  (lower is better)", fontsize=8, labelpad=2)

        if texts:
            adjust_text(
                texts,
                x=xs_pts,
                y=ys_pts,
                ax=ax,
                force_text=(0.9, 1.1),
                force_static=(0.6, 0.8),
                force_explode=(1.2, 1.4),
                force_pull=(0.015, 0.015),
                expand=(1.25, 1.35),
                max_move=(45, 55),
                explode_radius="auto",
                ensure_inside_axes=True,
                expand_axes=True,
                iter_lim=500,
                min_arrow_len=3,
                arrowprops={"arrowstyle": "-", "color": "#aaaaaa", "lw": 0.45, "alpha": 0.7, "shrinkA": 2, "shrinkB": 2},
            )

        sns.despine(ax=ax, left=False, bottom=False)
        ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8, pad=2, color="#222222", bottom=True, left=True, top=False, right=False)
        ax.tick_params(axis="x", which="major", labelrotation=35)
        ax.tick_params(axis="both", which="minor", length=0)

    # Shared legend on leftmost panel: device shapes + reference lines.
    axes[0].scatter([], [], marker="o", c="#666666", s=50, label="CPU", edgecolors="black", linewidths=0.7, alpha=0.9)
    axes[0].scatter([], [], marker="*", c="#666666", s=100, label="GPU", edgecolors="black", linewidths=0.7, alpha=0.9)
    axes[0].plot([], [], "--", color=PARETO_COLOR, linewidth=1.3, label="Pareto front")
    axes[0].plot([], [], ":", color="#555555", linewidth=1.4, label=f"Max. mean abs. error ({mae_max:.0f})")
    leg = axes[0].legend(loc="upper left", frameon=True, fancybox=False, facecolor="white", framealpha=0.95, edgecolor="#cccccc", fontsize=6.5, borderpad=0.35, labelspacing=0.3, handletextpad=0.35)
    leg.get_frame().set_linewidth(0.6)

    axes[0].set_ylabel("Throughput (img/s)", fontsize=8, labelpad=2)

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.3)
    fig.savefig(path, dpi=350, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
    plt.close(fig)
    sns.reset_defaults()
    print(f"\nWrote {path}")


def main() -> None:
    print(f"Batch={BATCH} | {H}x{W} | warmup={WARMUP} runs={RUNS}")
    print("Baselines: Macenko→torchstain | Reinhard→StainTools | HM→skimage")
    print(f"Images: {DATA.name}/target + test_1..5 → {H}x{W}, x{BATCH}\n")

    ref, src = _load_he_batch(BATCH)
    print("Building baselines...")
    baselines = build_baselines(ref, src)
    results = collect_results(ref, src, baselines)
    if not results:
        raise SystemExit("No results to plot.")
    plot(results, OUT, MAE_MAX)


if __name__ == "__main__":
    main()
