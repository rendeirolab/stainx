# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.
.PHONY: build clean test install install-dev help

# Variables
PYTHON := python
UV := uv
PIP := pip
PYTEST := pytest

# Default target
help:
	@echo "Available targets:"
	@echo "  make build      - Build the package"
	@echo "  make clean      - Clean build artifacts and cache files"
	@echo "  make test       - Run tests"
	@echo "  make install    - Install package in editable mode"
	@echo "  make install-dev - Install package with dev dependencies"

# Build and install the package
build:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	find . -type f -name "*.so" -delete
	@echo "Uninstalling stainx if installed..."
	-$(PIP) uninstall -y stainx 2>/dev/null || true
	@echo "Building and installing stainx in editable mode..."
	$(PIP) install -e .
	@echo "Build and install complete!"

# Clean build artifacts and cache files
clean:
	@echo "Cleaning build artifacts..."
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
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

# Run tests
test:
	@echo "Running tests..."
	$(PYTEST) tests/ -v

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ -v --cov=src/stainx --cov-report=term-missing --cov-report=html

# Install package in editable mode
install:
	@echo "Installing stainx in editable mode..."
	$(PIP) install -e .

# Install package with dev dependencies
install-dev:
	@echo "Installing stainx with dev dependencies..."
	$(UV) sync --dev

# Format code
format:
	@echo "Formatting code..."
	ruff format src/ tests/

# Lint code
lint:
	@echo "Linting code..."
	ruff check src/ tests/

# Type check
typecheck:
	@echo "Type checking..."
	mypy src/ || echo "mypy not installed, skipping type check"

