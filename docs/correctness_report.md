# Correctness report

Audit of StainX Torch backends against project oracles:

| Algorithm | Check | Status | Notes |
|-----------|-------|--------|-------|
| Macenko | Cov / `eigh` / top-2 | **fixed** | `torch`: manual cov ≡ `torch.cov`, `eigh` on **Torch CPU** (GPU eigh can flip the stain plane when eigenvalues are nearly degenerate). `torch_cuda`: custom fp64 cov + analytic 3×3 eigh on device (`precision="stable"`); optional `precision="fast"` (fp32) |
| Macenko | Angular percentiles / H-E order | match | Nearest-rank percentile (`kthvalue` / on-device gather) |
| Macenko | Concentrations / maxC | match | `torch`: `lstsq` on Torch CPU; `torch_cuda`: on-device 2×2 solve |
| Macenko | RGB reconstruct / Io cap | **fixed** | Match torchstain (allow > Io) |
| Reinhard | LAB mean/std match | match | Torch + CUDA kernels |
| Histogram Matching | CDF / LUT | match | vs `skimage.match_histograms` |

## Backend matrix (0.1.0)

| Backend | Engine | Notes |
|---------|--------|-------|
| `torch` | `MacenkoTorch` / etc. | Primary CPU / CUDA / MPS path |
| `torch_cuda` | Extension | Reinhard / HM / Macenko: custom CUDA kernels |

CuPy backends were removed in 0.1.0.
