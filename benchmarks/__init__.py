"""Benchmark entrypoints for local profiling.

This package intentionally re-exports the `run_stainx` CLI entry so that
Sentrux's dependency graph sees `benchmarks/run_stainx.py` as having at least
one internal importer (reducing file instability) while keeping the script
directly runnable.
"""

from benchmarks.run_stainx import main as run_stainx_main

__all__ = ["run_stainx_main"]
