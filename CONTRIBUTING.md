# Contributing to StainX

## Architecture

StainX uses a layered architecture with automatic backend selection. PyTorch and CuPy are supported as framework backends, with at least one required.

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
        ABC["abc.ABC"]
  end
 subgraph subGraph2["Framework Backends (Equal Priority)"]
  subgraph subGraph4["PyTorch Backend Implementations"]
        PTBB["PyTorchBackendBase"]
        HMPT["HistogramMatchingPyTorch"]
        REPT["ReinhardPyTorch"]
        MAPT["MacenkoPyTorch"]
        RGB2LAB["rgb_to_lab"]
        LAB2RGB["lab_to_rgb"]
  end
  subgraph subGraph5["PyTorch CUDA Backend Implementations"]
        CUBB["PyTorchCUDABackendBase"]
        HMCU["HistogramMatchingCUDA"]
        RECU["ReinhardCUDA"]
        MACU["MacenkoCUDA"]
  end
  subgraph subGraph8["CuPy Backend Implementations"]
        CPBB["CuPyBackendBase"]
        HMCP["HistogramMatchingCuPy"]
        RECP["ReinhardCuPy"]
        MACP["MacenkoCuPy"]
  end
  subgraph subGraph9["CuPy CUDA Backend Implementations"]
        CPUBB["CuPyCUDABackendBase"]
        HMCPCU["HistogramMatchingCuPyCUDA"]
        RECPCU["ReinhardCuPyCUDA"]
        MACPCU["MacenkoCuPyCUDA"]
  end
 end
 subgraph subGraph6["Pure CUDA Kernels (csrc/)"]
        HM_PURE["histogram_matching.cu"]
        RE_PURE["reinhard.cu"]
        MA_PURE["macenko.cu"]
  end
 subgraph subGraph7["PyTorch CUDA Extension"]
        SC["stainx_cuda_torch"]
        HM_WRAP["histogram_matching.cu"]
        RE_WRAP["reinhard.cu"]
        MA_WRAP["macenko.cu"]
        BIND["bindings.cpp"]
  end
 subgraph subGraph10["CuPy CUDA Extension"]
        SCCP["stainx_cuda_cupy"]
        HM_WRAP_CP["histogram_matching.cu"]
        RE_WRAP_CP["reinhard.cu"]
        MA_WRAP_CP["macenko.cu"]
        BIND_CP["bindings.cpp"]
  end
 subgraph Utilities["Utilities"]
        UTILS["utils.py"]
        GD["get_device"]
        CFC["ChannelFormatConverter"]
  end
    NT -- inherits --> SNB
    SNB -- inherits --> ABC
    HM -- inherits --> NT
    RE -- inherits --> NT
    MA -- inherits --> NT
    NT -- selects backend via _select_backend --> PTBB
    NT -- selects backend via _select_backend --> CUBB
    NT -- selects backend via _select_backend --> CPBB
    NT -- selects backend via _select_backend --> CPUBB
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
    HMCP -- inherits --> CPBB
    RECP -- inherits --> CPBB
    MACP -- inherits --> CPBB
    HMCPCU -- inherits --> CPUBB
    RECPCU -- inherits --> CPUBB
    MACPCU -- inherits --> CPUBB
    HMCPCU -- calls --> SCCP
    RECPCU -- calls --> SCCP
    MACPCU -- calls --> SCCP
    SC -- compiled from --> HM_WRAP
    SC -- compiled from --> RE_WRAP
    SC -- compiled from --> MA_WRAP
    SC -- compiled from --> BIND
    HM_WRAP -- includes kernels from --> HM_PURE
    RE_WRAP -- includes kernels from --> RE_PURE
    MA_WRAP -- includes kernels from --> MA_PURE
    SCCP -- compiled from --> HM_WRAP_CP
    SCCP -- compiled from --> RE_WRAP_CP
    SCCP -- compiled from --> MA_WRAP_CP
    SCCP -- compiled from --> BIND_CP
    HM_WRAP_CP -- includes kernels from --> HM_PURE
    RE_WRAP_CP -- includes kernels from --> RE_PURE
    MA_WRAP_CP -- includes kernels from --> MA_PURE
    UTILS --> GD & CFC
    REPT -- uses --> RGB2LAB & LAB2RGB
    SNB -- uses --> GD
    User -- creates --> HM & RE & MA
    style User fill:#e1f5ff
    style NT fill:#fff4e1
    style SNB fill:#fff4e1
    style PTBB fill:#e8f5e9
    style CUBB fill:#fce4ec
    style CPBB fill:#e3f2fd
    style CPUBB fill:#fff3e0
    style SC fill:#f3e5f5
    style SCCP fill:#e8eaf6
    style HM_PURE fill:#fff9c4
    style RE_PURE fill:#fff9c4
    style MA_PURE fill:#fff9c4
```

**Key Components:**
- **User API**: `HistogramMatching`, `Reinhard`, `Macenko` classes
- **Template Layer**: `NormalizerTemplate` handles backend selection (PyTorch, CuPy, etc.)
- **Base Classes**: `StainNormalizerBase` and `NormalizerTemplate` are backend-agnostic (no hard framework dependency)
- **Framework Backends**: PyTorch and CuPy (at least one required)
  - **PyTorch**: Pure Python implementation (CPU, CUDA, MPS)
  - **PyTorch CUDA**: Optimized CUDA kernels via PyTorch extension
  - **CuPy**: Pure Python implementation using CuPy (CPU, CUDA)
  - **CuPy CUDA**: Optimized CUDA kernels via CuPy extension (future)
- **Pure CUDA Kernels**: Located in `csrc/`, no dependencies, reusable by any CUDA interface (PyTorch or CuPy)
- **Framework Extensions**: 
  - **PyTorch CUDA Extension**: Wrappers in `src/stainx_cuda_torch/csrc/` that include pure kernels
  - **CuPy CUDA Extension**: Wrappers in `src/stainx_cuda_cupy/csrc/` that include pure kernels (future)
- **Utilities**: Backend-agnostic device detection, channel format conversion

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
   - Implement `_get_pytorch_class()` and optionally `_get_cupy_class()`, `_get_torch_cuda_class()`, `_get_cupy_cuda_class()`

2. **Implement PyTorch backend** in `src/stainx/backends/torch_backend.py`:
   - Inherit from `PyTorchBackendBase`
   - Implement algorithm in PyTorch

3. **Implement CuPy backend** (optional):
   - Create backend class in `src/stainx/backends/cupy_backend.py`
   - Inherit from `CuPyBackendBase`
   - Implement algorithm using CuPy

4. **Implement CUDA backend** (optional):
   - Add pure CUDA kernels in `csrc/` (no dependencies, reusable by both PyTorch and CuPy)
   - **For PyTorch CUDA**:
     - Add PyTorch wrapper in `src/stainx_cuda_torch/csrc/` that includes kernels from `csrc/`
     - Add bindings in `src/stainx_cuda_torch/csrc/bindings.cpp`
     - Create backend class in `src/stainx/backends/torch_cuda_backend.py`
   - **For CuPy CUDA** (future):
     - Add CuPy wrapper in `src/stainx_cuda_cupy/csrc/` that includes kernels from `csrc/`
     - Add bindings in `src/stainx_cuda_cupy/csrc/bindings.cpp`
     - Create backend class in `src/stainx/backends/cupy_cuda_backend.py`

5. **Add tests** in `tests/`:
   - Test correctness against reference implementation
   - Test all available backends (PyTorch, CuPy, CUDA variants)

6. **Update documentation**:
   - Add usage examples
   - Update README if needed

### Adding New Backends

To add a new backend (e.g., OpenCL, Metal):


1. **Create backend base class** in `src/stainx/backends/`:
   - Inherit from a common interface or create new base class
   - Implement `__init__()` with device handling
   - Define required methods (e.g., `transform()`)

2. **Implement backend classes** for each normalizer:
   - Create classes like `HistogramMatching<Backend>`, `Reinhard<Backend>`, `Macenko<Backend>`
   - Inherit from the backend base class
   - Implement `transform()` method with algorithm-specific logic

3. **Update backend selection** in `src/stainx/normalizers/_template.py`:
   - Modify `_select_backend()` to detect and select new backend
   - Add availability check (similar to `CUDA_AVAILABLE`)

4. **Update normalizer classes**:
   - Add `_get_<backend>_class()` method in each normalizer
   - Update `_get_backend_impl()` to handle new backend

5. **Export backend classes** in `src/stainx/backends/__init__.py`

6. **Add tests**:
   - Test backend availability detection
   - Test correctness against reference implementation
   - Ensure backend works independently (not as fallback)

**Note:** Currently, the `fit()` function executes only PyTorch routines. To improve flexibility and fully support all backends (such as CUDA), consider implementing backend-specific `fit()` logic so that fitting is performed using the selected backend, not just PyTorch. This will ensure consistency and performance across all supported backends.

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

