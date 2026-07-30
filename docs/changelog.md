# Changelog

## [0.1.2] - 2026-07-30

### Changed

- Macenko pure CUDA cov / analytic 3×3 eigh kernels live under `csrc/macenko.cu` (Torch wrapper includes them; ATen pipeline unchanged)
- Docs MkDocs theme primary/accent color set to `#ffc3e0` (custom CSS)

## [0.1.1] - 2026-07-30

### Changed

- Replace package logo with new `StainX-logo.svg` (README, docs, MkDocs theme logo/favicon)

## [0.1.0] - 2026-07-20

### Breaking

- **Torch-only**: removed CuPy backends, `stainx_cuda_cupy`, and `stainx[cupy]`
- Valid backends are now only `"torch"` and `"torch_cuda"`
- No backward compatibility with 0.0.x import paths or CuPy array inputs
- **Input range is dtype-gated**: `uint8` → `[0, 255]`; **float is always treated as `[0, 1]`** (no `max()>1` / `amax()` heuristic). Calling `.float()` on uint8 without `/255` is silently wrong — keep `uint8` or scale explicitly. ColorJitter may push floats slightly above 1; that is fine and is not treated as a `[0, 255]` signal.
- **Migration**: CuPy users should pin `stainx<0.1` or switch inputs/backends to Torch (`backend="torch"` / `"torch_cuda"`)

### Added

- Public `StainNormalizerTransform` (`mode="reference"` | `"batch"`) for DataLoader pipelines (`normalize_to_0_1` defaults to `True` for Macenko)
- `torch_cuda` Macenko CUDA kernel (fp64 covariance + analytic 3×3 eigh + on-device concentration solve); optional `precision="fast"` (fp32 cov/eigh, fp16 pixels)

### Removed

- `stainx.viz` / `stainx[viz]` (matplotlib helpers)

### Testing

- Correctness suite compares each normalizer × backend (`torch`, `torch_cuda`) to external baselines (torchstain / skimage) with **absolute** `atol` / `MACENKO_ATOL` (plus HE / maxC floors for Macenko)
- Macenko correctness uses real H&E tiles instead of random noise: noise gives a near-isotropic OD covariance with degenerate leading eigenvectors, so the stain plane (and H/E split) is eigensolver-dependent and parity there is ill-posed

### Fixed

- Macenko: remove reconstructed OD≥0 clamp so RGB can exceed `Io` (torchstain parity)
- Macenko: always use `lstsq` (drop `pinv`/cond fallback)
- Macenko (`torch` backend): run 3×3 `eigh` on Torch CPU to avoid GPU plane flips when eigenvalues are nearly degenerate

### Changed

- Version bump to 0.1.0
- Install docs emphasize `make install*` and Linux-first support (macOS/Windows best-effort)
- Silent backend selection (no print side-effect on init)
- `make build` syncs `--group dev` instead of `--all-groups`
- Pin correctness/benchmark dependency `torchstain==1.4.1`
- Tighten Macenko vs torchstain relative-error threshold to `0.01`
- `torch_cuda` Macenko is a fully on-GPU path (custom fp64 covariance reduction + analytic 3×3 symmetric eigendecomposition matching `torch.linalg.eigh` to machine precision + on-device concentration solve) — no per-image CPU round-trip. ~5–9× faster than the previous ATen/CPU-offload path while staying within a grey level of torchstain on H&E tiles (e.g. 555 → 5177 img/s at 64×150², 86 → 476 img/s at 32×512²)
