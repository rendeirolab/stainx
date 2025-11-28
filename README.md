# stainx

GPU-accelerated stain normalization.

## Installation

### Requirements

- Python >=3.12
- PyTorch >=2.0.0
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
pip install .
```

The package will automatically build CUDA extensions if CUDA is available. If CUDA is not available, the package will install without CUDA support.

### Using Make

Alternatively, you can use the provided Makefile:

```bash
make build      # Build the package (includes CUDA extension if available)
make clean      # Clean build artifacts and cache files
make test       # Run tests
make lint       # Check code for linting issues
make fix        # Auto-fix linting issues and format code
make help       # Show all available targets
```

### Verify Installation

You can verify the installation and check which backend is being used (CUDA or PyTorch) with the following command:

```bash
uvx --with "stainx @ git+https://github.com/rendeirolab/stainx.git" --with numpy --with torch python -c "import torch; from stainx import Reinhard; cuda_avail = torch.cuda.is_available(); print(f'CUDA available: {cuda_avail}'); device = 'cuda' if cuda_avail else 'cpu'; n = Reinhard(device=device); print(f'Backend selected: {n.backend}'); print(f'Backend implementation: {n._get_backend_impl().__class__.__name__}'); print('✅ CUDA backend' if n.backend == 'cuda' else '✅ PyTorch backend')"
```

This command uses `uvx` to temporarily install stainx from GitHub along with its dependencies (numpy and torch), then runs a Python script that:
- Checks if CUDA is available
- Creates a Reinhard normalizer instance
- Prints which backend is selected (CUDA or PyTorch)
- Displays the backend implementation class name
