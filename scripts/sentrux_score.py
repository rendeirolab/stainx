from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class SentruxScore:
    quality: int
    raw_output: str


_QUALITY_RE = re.compile(r"^\s*Quality:\s*(\d+)\s*$", re.MULTILINE)


def _run_sentrux_check(path: str) -> SentruxScore:
    proc = subprocess.run(
        ["sentrux", "check", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout or ""

    m = _QUALITY_RE.search(out)
    if not m:
        raise RuntimeError(f"Could not parse Sentrux quality from output:\n{out}")

    return SentruxScore(quality=int(m.group(1)), raw_output=out)


def _copy_scoped_project(src_root: Path, dst_root: Path, *, include: list[str], exclude: list[str]) -> None:
    def included(p: Path) -> bool:
        rel = p.relative_to(src_root).as_posix()
        return any(p.match(glob) or Path(rel).match(glob) for glob in include)

    def excluded(p: Path) -> bool:
        rel = p.relative_to(src_root).as_posix()
        return any(p.match(glob) or Path(rel).match(glob) for glob in exclude)

    for p in src_root.rglob("*"):
        if p.is_dir():
            continue
        if not included(p) or excluded(p):
            continue

        rel = p.relative_to(src_root)
        out_path = dst_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Print (and optionally gate) Sentrux Quality score.")
    p.add_argument("path", nargs="?", default=".", help="Repo root to score (default: .)")
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob(s) to exclude from scoring (repeatable). Example: --exclude 'examples/**'",
    )
    p.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob(s) to include in scoring (repeatable). If omitted, includes everything not excluded.",
    )
    p.add_argument("--min", dest="min_quality", type=int, default=None, help="Fail (exit 1) if quality is below this value.")
    args = p.parse_args(argv)

    root = Path(args.path).resolve()
    # Default behavior:
    # - If no scoping flags are provided, score the repo directly.
    # - If the user provides *any* include/exclude flags, score a lightweight
    #   "library-only" snapshot by default (can be overridden with --include).
    if args.include:
        include = args.include
    elif args.exclude:
        include = ["src/**", "tests/**", ".sentrux/**", "pyproject.toml", "README*", "LICENSE*"]
    else:
        include = ["**/*"]

    exclude = list(args.exclude)
    if args.exclude:
        # Always drop common non-source dirs when running in snapshot mode.
        exclude.extend(
            [
                ".git/**",
                ".venv/**",
                ".pytest_cache/**",
                ".ruff_cache/**",
                "dist/**",
                "build/**",
                "site/**",
                "htmlcov/**",
                "**/__pycache__/**",
            ]
        )

    # If we need include/exclude scoping, run Sentrux on a temporary copy.
    if include != ["**/*"] or exclude:
        with tempfile.TemporaryDirectory(prefix="sentrux-scope-") as td:
            scoped = Path(td)
            _copy_scoped_project(root, scoped, include=include, exclude=exclude)
            score = _run_sentrux_check(str(scoped))
    else:
        score = _run_sentrux_check(str(root))
    print(score.quality)

    if args.min_quality is not None and score.quality < args.min_quality:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
