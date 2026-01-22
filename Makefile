# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
.PHONY: build clean test install install-dev help lint fix docs

# Variables
UV := $(shell PATH="$(HOME)/.local/bin:$$PATH" command -v uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")
PYTHON := $(UV) run python
PIP := $(UV) run pip
PYTEST := $(UV) run pytest
MKDOCS := $(UV) run mkdocs

# Default target
help:
	@echo "Available targets:"
	@echo "  make build       - Build distribution packages (wheels + sdist) using uv build"
	@echo "  make install     - Fast editable install for development (production deps only)"
	@echo "  make install-dev - Install with dev dependencies"
	@echo "  make clean       - Clean build artifacts and cache files"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Check code for linting issues"
	@echo "  make fix         - Auto-fix linting issues and format code"
	@echo "  make docs        - Build documentation (matches Read the Docs)"

# Build distribution packages (wheels + sdist) - optimized for PyPI publishing
build:
	@if [ ! -f "$(UV)" ] && ! PATH="$(HOME)/.local/bin:$$PATH" command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Creating virtual environment if not exists..."
	@$(UV) venv .venv --seed
	@$(UV) sync --all-groups
	@echo "Building distribution packages with uv build..."
	$(UV) build
	@echo "Build complete! Artifacts are in dist/"

# Fast editable install for development
install:
	@if [ ! -f "$(UV)" ] && ! PATH="$(HOME)/.local/bin:$$PATH" command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Installing stainx in editable mode (fast, production deps only)..."
	$(UV) pip install -e .
	@echo "Installing ninja for faster builds..."
	@$(UV) pip install ninja || echo "Warning: ninja installation failed, builds will be slower"
	@echo "Building CUDA extension in-place (if CUDA is available)..."
	@$(UV) run python setup.py build_ext --inplace || echo "Warning: CUDA extension build failed or skipped, continuing with PyTorch backend only"
	@echo "Installation complete!"

# Install with dev dependencies
install-dev:
	@if [ ! -f "$(UV)" ] && ! PATH="$(HOME)/.local/bin:$$PATH" command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Syncing uv environment with dev dependencies..."
	$(UV) sync --all-groups
	@echo "Installing stainx in editable mode..."
	$(UV) pip install -e .
	@echo "Building CUDA extension in-place (if CUDA is available)..."
	@$(UV) run python setup.py build_ext --inplace || echo "Warning: CUDA extension build failed or skipped, continuing with PyTorch backend only"
	@echo "Installation with dev dependencies complete!"

# Clean build artifacts and cache files
clean:
	@echo "Cleaning build artifacts and caches..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .tox/
	rm -rf .hypothesis/
	rm -rf site/
	rm -rf .venv-rtd/
	# Remove Python cache files and directories
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	# Remove compiled extensions (but not in .venv)
	find . -path "./.venv" -prune -o -type f -name "*.so" -print | xargs rm -f 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	# Remove any .cache directories (e.g., from pytest-xdist)
	find . -type d -name ".cache" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete! (Note: Python's in-memory import cache will be cleared on next Python restart)"

# Run tests
test:
	@echo "Running tests..."
	@if [ ! -d ".venv" ]; then \
		echo "Virtual environment not found. Running 'make install-dev' first..."; \
		$(MAKE) install-dev; \
	fi
	@TORCH_LIB_PATH=$$($(PYTHON) -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null); \
	if [ -z "$$TORCH_LIB_PATH" ] || [ ! -d "$$TORCH_LIB_PATH" ]; then \
		echo "Warning: Could not find PyTorch library path, tests may fail if CUDA extension is used"; \
		$(PYTEST) tests/ -v; \
	else \
		LD_LIBRARY_PATH="$$TORCH_LIB_PATH:$$LD_LIBRARY_PATH" $(PYTEST) tests/ -v; \
	fi

# Check code for linting issues
lint:
	@echo "Checking code for linting issues..."
	@find src/stainx_cuda_torch/csrc/ -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format --dry-run --Werror
	$(UV) run ruff check .

# Auto-fix linting issues and format code
fix:
	@echo "Fixing linting issues and formatting code..."
	$(UV) run ruff check --fix --unsafe-fixes .
	$(UV) run ruff format .
	@find src/stainx_cuda_torch/csrc/ -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i
	@echo "Code formatting and linting fixes complete!"

# Build documentation (matches Read the Docs build process)
docs:
	@echo "Building documentation (matching Read the Docs process)..."
	@if [ ! -d ".venv-rtd" ]; then \
		echo "Creating isolated virtual environment..."; \
		$(UV) venv .venv-rtd --seed; \
	fi
	@echo "Installing dependencies from requirements-docs.txt..."
	@$(UV) pip install --python .venv-rtd/bin/python -r requirements-docs.txt
	@echo "Installing stainx package..."
	@$(UV) pip install --python .venv-rtd/bin/python -e .
	@echo "Building documentation with mkdocs..."
	@.venv-rtd/bin/mkdocs build --strict
	@echo "Documentation build successful! Output in site/"

