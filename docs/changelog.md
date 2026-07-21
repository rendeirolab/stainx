# Changelog

## [0.1.0] - 2026-07-20

### Breaking

- **Torch-only**: removed CuPy backends, `stainx_cuda_cupy`, and `stainx[cupy]`
- Valid backends are now only `"torch"` and `"torch_cuda"`
- No backward compatibility with 0.0.x import paths or CuPy array inputs
- **Input range is dtype-gated**: `uint8` → `[0, 255]`; **float is always treated as `[0, 1]`** (no `max()>1` / `amax()` heuristic). Calling `.float()` on uint8 without `/255` is silently wrong — keep `uint8` or scale explicitly. ColorJitter may push floats slightly above 1; that is fine and is not treated as a `[0, 255]` signal.
- **Migration**: CuPy users should pin `stainx<0.1` or switch inputs/backends to Torch (`backend="torch"` / `"torch_cuda"`)

### Added

- Public `StainNormalizerTransform` (`mode="reference"` | `"batch"`) for DataLoader pipelines (`normalize_to_0_1` defaults to `True` for Macenko)

### Removed

- `stainx.viz` / `stainx[viz]` (matplotlib helpers)

### Testing

- Correctness suite compares each normalizer × backend (`torch`, `torch_cuda`) to external baselines (torchstain / skimage) with **absolute** `atol` / `MACENKO_ATOL` (plus HE / maxC floors for Macenko)

### Changed

- Version bump to 0.1.0
- Install docs emphasize `make install*` and Linux-first support (macOS/Windows best-effort)
- Silent backend selection (no print side-effect on init)
- `make build` syncs `--group dev` instead of `--all-groups`

## Unreleased (pre-0.1.0 notes)

### Fixed

- Macenko: remove reconstructed OD≥0 clamp so RGB can exceed `Io` (torchstain parity)
- Macenko: always use `lstsq` (drop `pinv`/cond fallback)
- Macenko: run 3×3 `eigh` on Torch CPU to avoid GPU plane flips

### Changed

- Pin correctness/benchmark dependency `torchstain==1.4.1`
- Tighten Macenko vs torchstain relative-error threshold to `0.01`
- `torch_cuda` Macenko: ATen parity path; Reinhard/HM keep real CUDA kernels
