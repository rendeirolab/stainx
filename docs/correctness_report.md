# Correctness report

Audit of StainX Torch backends against project oracles in
`tests/torch_interface/test_correctness_against_references.py`
(and CUDA parity in `tests/torch_cuda_interface/`).

| Algorithm | Check | Status | Notes |
|-----------|-------|--------|-------|
| Macenko | Cov / `eigh` / top-2 | **fixed** | `torch`: manual cov, `eigh` on **Torch CPU** (GPU eigh can flip the stain plane when eigenvalues are nearly degenerate). `torch_cuda`: custom fp64 cov + analytic 3×3 eigh on device (`precision="stable"`); optional `precision="fast"` (fp32) |
| Macenko | Angular percentiles / H-E order | match | Nearest-rank percentile (`kthvalue` / on-device gather) |
| Macenko | Concentrations / maxC | match | `torch`: `lstsq` on Torch CPU; `torch_cuda`: on-device 2×2 solve |
| Macenko | RGB reconstruct / Io cap | **fixed** | Match torchstain (allow > Io) |
| Reinhard | LAB mean/std match | match | vs `torchstain` — `atol=1` |
| Histogram Matching | CDF / LUT | match | vs `skimage.match_histograms` — `atol=1` |

## Thresholds (executable oracle)

| Algorithm | Baseline | Pixel gate | Extra |
|-----------|----------|------------|-------|
| Reinhard | torchstain 1.4.1 | `atol=1` | — |
| Histogram Matching | skimage | `atol=1` | — |
| Macenko | torchstain 1.4.1 | `atol=2` | HE/maxC `allclose`; MAE ≤ `0.35`; **synthetic** Beer–Lambert H&E tiles (not random RGB) |

## Backend matrix (since 0.1.0)

| Backend | Engine | Notes |
|---------|--------|-------|
| `torch` | `MacenkoTorch` / etc. | Primary CPU / CUDA / MPS path |
| `torch_cuda` | Extension | Reinhard / HM / Macenko: custom CUDA kernels |

CuPy backends were removed in 0.1.0. Package version is independent of this matrix
(see `stainx.__version__`).
