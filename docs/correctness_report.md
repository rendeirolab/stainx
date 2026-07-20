# Correctness report

Audit of StainX Torch backends against project oracles:

| Algorithm | Check | Status | Notes |
|-----------|-------|--------|-------|
| Macenko | Cov / `eigh` / top-2 | **fixed** | Manual cov ≡ `torch.cov`; `eigh` forced onto **Torch CPU** on torch / torch_cuda — GPU eigh can flip the 2D stain plane when eigenvalues are nearly degenerate |
| Macenko | Angular percentiles / H-E order | match | Nearest-rank percentile (`kthvalue`) |
| Macenko | Concentrations / maxC | match | Always `lstsq` on Torch CPU |
| Macenko | RGB reconstruct / Io cap | **fixed** | Match torchstain (allow > Io) |
| Reinhard | LAB mean/std match | match | Torch + CUDA kernels |
| Histogram Matching | CDF / LUT | match | vs `skimage.match_histograms` |

## Backend matrix (0.1.0)

| Backend | Engine | Notes |
|---------|--------|-------|
| `torch` | `MacenkoTorch` / etc. | Primary CPU / CUDA / MPS path |
| `torch_cuda` | Extension | Reinhard/HM: pure kernels; Macenko: ATen parity |

CuPy backends were removed in 0.1.0.
