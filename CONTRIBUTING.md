# Contributing to StainX

## Architecture

StainX uses a layered architecture with automatic backend selection:

```mermaid
---
config:
  layout: elk
---
flowchart TB
 subgraph subGraph0["User API Layer"]
        User["User Code"]
        HM["HistogramMatching"]
        RE["Reinhard"]
        MA["Macenko"]
  end
 subgraph subGraph1["Normalizer Template Layer"]
        NT["NormalizerTemplate"]
        SNB["StainNormalizerBase"]
        nn.Module["torch.nn.Module"]
  end
 subgraph subGraph4["PyTorch Backend Implementations"]
        PTBB["PyTorchBackendBase"]
        HMPT["HistogramMatchingPyTorch"]
        REPT["ReinhardPyTorch"]
        MAPT["MacenkoPyTorch"]
        RGB2LAB["rgb_to_lab"]
        LAB2RGB["lab_to_rgb"]
  end
 subgraph subGraph5["CUDA Backend Implementations"]
        CUBB["CUDABackendBase"]
        HMCU["HistogramMatchingCUDA"]
        RECU["ReinhardCUDA"]
        MACU["MacenkoCUDA"]
  end
 subgraph subGraph6["CUDA Extension"]
        SC["stainx_cuda"]
        CU[".cu files"]
        HM_CU["histogram_matching.cu"]
        RE_CU["reinhard.cu"]
        MA_CU["macenko.cu"]
  end
 subgraph Utilities["Utilities"]
        UTILS["utils.py"]
        GD["get_device"]
        CFC["ChannelFormatConverter"]
  end
    NT -- inherits --> SNB
    SNB -- inherits --> nn.Module
    HM -- inherits --> NT
    RE -- inherits --> NT
    MA -- inherits --> NT
    NT -- selects backend via _select_backend --> PTBB
    NT -- selects backend via _select_backend --> CUBB
    HMPT -- inherits --> PTBB
    REPT -- inherits --> PTBB
    MAPT -- inherits --> PTBB
    PTBB -- provides static methods --> RGB2LAB & LAB2RGB
    HMCU -- inherits --> CUBB
    RECU -- inherits --> CUBB
    MACU -- inherits --> CUBB
    HMCU -- calls --> SC
    RECU -- calls --> SC
    MACU -- calls --> SC
    SC -- compiled from --> CU
    CU --> HM_CU & RE_CU & MA_CU
    UTILS --> GD & CFC
    REPT -- uses --> RGB2LAB & LAB2RGB
    SNB -- uses --> GD
    User -- creates --> HM & RE & MA
    style User fill:#e1f5ff
    style NT fill:#fff4e1
    style SNB fill:#fff4e1
    style PTBB fill:#e8f5e9
    style CUBB fill:#fce4ec
    style SC fill:#f3e5f5
```

**Key Components:**
- **User API**: `HistogramMatching`, `Reinhard`, `Macenko` classes
- **Template Layer**: `NormalizerTemplate` handles backend selection
- **Backends**: PyTorch (pure Python) and CUDA (optimized kernels)
- **Utilities**: Device detection, channel format conversion

## Contributing

### Development Setup

```bash
make install-dev  # Install with dev dependencies
make test         # Run tests
make lint         # Check code quality
make fix          # Auto-fix issues
```

### Adding New Normalization Methods

1. **Create normalizer class** in `src/stainx/normalizers/`:
   - Inherit from `NormalizerTemplate`
   - Implement `fit()` and `transform()` methods

2. **Implement PyTorch backend** in `src/stainx/backends/torch_backend.py`:
   - Inherit from `PyTorchBackendBase`
   - Implement algorithm in PyTorch

3. **Implement CUDA backend** (optional):
   - Add `.cu` file in `src/stainx_cuda/csrc/`
   - Add bindings in `src/stainx_cuda/__init__.py`
   - Create backend class in `src/stainx/backends/cuda_backend.py`

4. **Add tests** in `tests/`:
   - Test correctness against reference implementation
   - Test both PyTorch and CUDA backends

5. **Update documentation**:
   - Add usage examples
   - Update README if needed

### Code Style

- Python: Follow PEP 8, use `ruff` for linting
- C++/CUDA: Use `clang-format` for formatting
- Run `make fix` before committing

### Build the project

- To build the project (including the CUDA extension if you have CUDA available) `make build`
- This runs the package build process and compiles any C++/CUDA extensions. If you want to force a clean rebuild, run `make clean  build`
- Check the `Makefile` for additional commands.

### Testing Requirements

- All tests must pass: `make test`
- New features require tests
- CUDA tests should gracefully skip if CUDA unavailable

