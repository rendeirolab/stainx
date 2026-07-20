# Backend correctness report (vs baselines)

Audit of StainX algorithm backends against project oracles:
**torchstain 1.4.1** (Reinhard / Macenko) and **scikit-image** (histogram matching).

## Summary

| Method | Step | Verdict | Notes |
|--------|------|---------|-------|
| Macenko | RGB→OD (`Io=240`, `+1`) | match | Same as torchstain |
| Macenko | OD filter (`beta=0.15`) | match | `min>=beta` ≡ `~any(OD<beta)` |
| Macenko | Cov / `eigh` / top-2 | **fixed** | Manual cov ≡ `torch.cov`; `eigh` forced onto **Torch CPU** on torch / cupy / torch_cuda — GPU/NumPy eigh can flip the 2D stain plane when eigenvalues are nearly degenerate |
| Macenko | Angular percentiles / H-E order | match | Nearest-rank percentile (`kthvalue` / `cp.partition`) on all shipped backends |
| Macenko | Concentrations | **fixed** | Was: `pinv` when `cond>10`; now always `lstsq` on **CPU** (torchstain parity; CUDA gels can diverge) |
| Macenko | Reconstruct OD→RGB | **fixed** | Was: clamp OD≥0 (cap RGB at 240); now allow negative OD, clip RGB to [0,255] only |
| Macenko | Output layout | intentional | StainX CHW float; torchstain HWC int — compared as float CHW |
| Reinhard | sRGB↔LAB (OpenCV-scaled) | match | Same matrices / white point |
| Reinhard | mean/std standardize | match | Rel error ~2e-4 |
| Histogram matching | CDF / LUT | intentional | Close but not identical to `skimage.exposure.match_histograms`; tolerances 0.05 (0.09 if min side ≤64) |

## Confirmed bugs (fixed)

### 1. OD≥0 clamp on reconstruction

In torch, cupy, and the former pure-CUDA Macenko kernels (`fmaxf(0, sum)`). Capped RGB at `Io=240` while torchstain reaches 255. With HE/maxC already matching, this alone caused ~1.5–3.3% rel L2 on CPU.

After removal, **CPU torch Macenko vs torchstain rel error is 0** on audited seeds/sizes.

### 2. `pinv` / cond fallback

Torch/cupy used `pinv` when `cond(HE)>10` or RHS >1M columns. Now always `lstsq` like torchstain.

### 3. GPU / NumPy `eigh` plane flip

Near-degenerate top eigenvalues (common on random uint8 tiles) made CUDA `eigh` / NumPy `eigh` reflect the 2D stain plane vs Torch CPU → wrong angular percentiles and ~40% rel error. Fix: run 3×3 `eigh` via **Torch CPU** in torch, cupy, and torch_cuda.

### 4. CUDA gels / histogram Macenko (torch_cuda)

The old pure-CUDA Macenko path (histogram percentiles + SVD/`gels`) diverged from torchstain on transform even when HE/maxC looked fine. That kernel file was **removed**. Shipping `torch_cuda` Macenko is an **ATen parity path** in [`src/stainx_cuda_torch/csrc/macenko.cu`](../src/stainx_cuda_torch/csrc/macenko.cu): `kthvalue`, CPU `eigh`, CPU `lstsq`.

## Test / process changes

- Macenko oracle tolerance tightened: `0.1` → `0.01` (torch + cupy).
- Intermediate asserts: HE matrix and maxC vs torchstain after `fit`.
- Assert RGB can exceed 240 when torchstain does (guards against OD-clamp regressions).
- CUDA Macenko Io-cap spot-check folded into [`tests/torch_cuda_interface/test_cuda_backend_parity_against_torch.py`](../tests/torch_cuda_interface/test_cuda_backend_parity_against_torch.py).
- **torchstain pinned to `==1.4.1`** in both `dev` and `benchmark` dependency groups.

## Backend matrix

| Backend | Macenko path | OD clamp | percentiles | lstsq / eigh | Notes |
|---------|--------------|----------|-------------|--------------|-------|
| `torch` | `MacenkoTorch` | fixed | `kthvalue` | CPU lstsq + CPU eigh | Matches torchstain on CPU |
| `cupy` | `MacenkoCupy` | fixed | `cp.partition` | CPU lstsq (via Torch) + CPU eigh | Same numerical policy as torch |
| `torch_cuda` | ATen in `stainx_cuda_torch` | fixed | `kthvalue` | CPU lstsq + CPU eigh | Parity path; not custom Macenko kernels |
| `cupy_cuda` | inherits `MacenkoCupy` | fixed | same as cupy | same as cupy | Name enforces CUDA device; no compiled Macenko kernels |

`csrc/` still holds Reinhard and histogram-matching kernels used by `torch_cuda`. Pure-CUDA Macenko kernels were deleted.

## Out of scope (unchanged)

- Slideflow remains a speed baseline, not a correctness oracle.
- Histogram matching algorithm not rewritten to bit-match skimage.
- Reinhard residual vs torchstain (~2e-4) not further polished.
- A future fast custom-CUDA Macenko that also matches torchstain bit-close.
