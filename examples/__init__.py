"""Runnable examples and demo entrypoints.

This package exists mainly to make entrypoints importable (and thus visible as
having internal importers in dependency graphs), while keeping scripts directly
executable.
"""

from examples.simple_example import main as simple_example_main

__all__ = ["simple_example_main"]
