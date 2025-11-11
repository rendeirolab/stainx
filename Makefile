# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
.PHONY: build clean test install install-dev help lint fix

# Variables
PYTHON := $(UV) run python
UV := $(shell PATH="$(HOME)/.local/bin:$$PATH" command -v uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")
PIP := $(UV) run pip
PYTEST := $(UV) run pytest

# Default target
help:
	@echo "Available targets:"
	@echo "  make build      - Build the package"
	@echo "  make clean      - Clean build artifacts and cache files"
	@echo "  make test       - Run tests"
	@echo "  make install    - Install package in editable mode"
	@echo "  make install-dev - Install package with dev dependencies"
	@echo "  make lint       - Check code for linting issues"
	@echo "  make fix        - Auto-fix linting issues and format code"

# Build and install the package
build:
	@if [ ! -f "$(UV)" ] && ! PATH="$(HOME)/.local/bin:$$PATH" command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Cleaning build artifacts..."
	rm -rf build/
	find . -path "./.venv" -prune -o -type f -name "*.so" -print | xargs rm -f 2>/dev/null || true
	@echo "Syncing uv environment and installing dependencies..."
	$(UV) sync
	@echo "Installing development dependencies from requirements-dev.txt..."
	$(UV) pip install -r requirements-dev.txt
	@PYTHON_VER=$$($(UV) run python --version 2>&1 | cut -d' ' -f2 | cut -d. -f1,2); \
	if [ ! -f ".venv/lib/python$$PYTHON_VER/site-packages/torch/lib/libtorch_global_deps.so" ]; then \
		$(UV) pip uninstall torch 2>/dev/null || true; \
		$(UV) pip install --no-cache torch; \
	fi
	@echo "Uninstalling stainx if installed..."
	-$(UV) pip uninstall -y stainx 2>/dev/null || true
	@echo "Building and installing stainx in editable mode..."
	$(UV) pip install -e .
	@echo "Build and install complete!"

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
	$(UV) sync --dev
	@PYTHON_VER=$$($(UV) run python --version 2>&1 | cut -d' ' -f2 | cut -d. -f1,2); \
	if [ ! -f ".venv/lib/python$$PYTHON_VER/site-packages/torch/lib/libtorch_global_deps.so" ]; then \
		$(UV) pip uninstall torch 2>/dev/null || true; \
		$(UV) pip install --no-cache torch; \
	fi
	$(PYTEST) tests/ -v

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	$(UV) sync --dev
	$(PYTEST) tests/ -v --cov=src/stainx --cov-report=term-missing --cov-report=html

# Install package in editable mode
install:
	@echo "Syncing uv environment and installing dependencies..."
	$(UV) sync
	@echo "Installing stainx in editable mode..."
	$(UV) pip install -e .

# Install package with dev dependencies
install-dev:
	@echo "Installing stainx with dev dependencies..."
	$(UV) sync --dev

# Check code for linting issues
lint:
	@echo "Checking code for linting issues..."
	$(UV) run ruff check .

# Auto-fix linting issues and format code
fix:
	@echo "Fixing linting issues and formatting code..."
	$(UV) run ruff check --fix --unsafe-fixes .
	$(UV) run ruff format .
	@echo "Code formatting and linting fixes complete!"

# Type check
typecheck:
	@echo "Type checking..."
	$(UV) run mypy src/ || echo "mypy not installed, skipping type check"

