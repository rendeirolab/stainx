import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


class NoColorFormatter(logging.Formatter):
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

    def format(self, record):
        original = super().format(record)
        return self.ansi_escape.sub("", original)


def setup_logger(filename, verbose, logger_name=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    name = logger_name or __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = NoColorFormatter("%(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger


def generate_batch(batch_size: int, height: int, width: int, channels: int = 3, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    max_value = 255
    x = torch.rand(batch_size, channels, height, width, device=device) * max_value
    return x.round().to(torch.uint8)


def benchmark_operation(operation_func: Callable, warmup: int, runs: int, device: str, logger=None) -> dict[str, Any]:
    try:
        for _ in range(warmup):
            operation_func()
            if device == "cuda":
                torch.cuda.synchronize()

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(runs):
            result = operation_func()
            if device == "cuda":
                torch.cuda.synchronize()

        return {"result": result, "time_ms": (time.time() - start) / runs * 1000, "success": True}
    except Exception as e:
        if logger:
            logger.info(f"    Benchmark failed: {e}")
        return {"result": None, "time_ms": float("inf"), "success": False}
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()


def calculate_speedup(baseline: float, comparison: float) -> str:
    if baseline > 0 and comparison > 0 and baseline != float("inf") and comparison != float("inf"):
        return f"{comparison / baseline:.2f}x"
    return "N/A"


def calculate_relative_error(y1: torch.Tensor, y2: torch.Tensor, logger=None) -> float:
    if y1 is None or y2 is None:
        return float("inf")

    if isinstance(y1, np.ndarray):
        y1 = torch.from_numpy(y1).float()
    if isinstance(y2, np.ndarray):
        y2 = torch.from_numpy(y2).float()

    y1, y2 = y1.cpu().float(), y2.cpu().float()

    if y1.shape != y2.shape and len(y1.shape) == 3 and len(y2.shape) == 3:
        if y1.shape[0] == y2.shape[2] and y1.shape[2] == y2.shape[0]:
            y2 = y2.permute(2, 0, 1)
        elif y1.shape[2] == y2.shape[0] and y1.shape[0] == y2.shape[2]:
            y1 = y1.permute(2, 0, 1)

    if y1.shape != y2.shape:
        if logger:
            logger.info(f"[Warning]: Shape mismatch - {y1.shape} vs {y2.shape}")
        return float("inf")

    epsilon = 1e-16
    rel_error = torch.norm((y1 - y2).abs(), p=2) / (torch.norm(y1, p=2) + epsilon)

    if rel_error > 1e-1 and logger:
        logger.info(f"[Warning]: High relative error {rel_error:.6f}")

    return rel_error.item()
