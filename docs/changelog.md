# Changelog

## Unreleased

### Fixed

- Macenko: remove reconstructed OD≥0 clamp so RGB can exceed `Io` (torchstain parity)
- Macenko: always use `lstsq` (drop `pinv`/cond fallback) on torch and cupy backends
- Macenko: run 3×3 `eigh` on Torch CPU across torch/cupy/torch_cuda to avoid GPU plane flips
- Macenko: force `lstsq` onto Torch CPU on cupy (same gels-parity policy as torch / torch_cuda)

### Changed

- Pin correctness/benchmark dependency `torchstain==1.4.1`
- Tighten Macenko vs torchstain relative-error threshold to `0.01`; add HE/maxC asserts
- `torch_cuda` Macenko: use ATen parity path (`kthvalue`, CPU eigh/lstsq); expect ~1× vs torch (Reinhard still uses real CUDA kernels)
- Remove unused pure-CUDA Macenko kernels (`csrc/macenko.cu`); `csrc/` keeps Reinhard + histogram matching only

## [0.1.0] - 2025-12-02

### Added

- Histogram Matching, Reinhard, and Macenko normalization
- PyTorch and CUDA backends
- Automatic backend selection
- Batch processing support
