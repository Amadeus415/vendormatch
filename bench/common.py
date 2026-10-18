"""Shared helpers for the benchmark scripts: loading splits, run metadata, saving results."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from contextlib import suppress
from datetime import date
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

DATA = Path("data")
RESULTS = Path("results")


def load_split(name: str) -> list[dict]:
    with (DATA / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


@cache
def hardware() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        return f"{chip.strip()}, {mem // 2**30} GB, macOS {platform.mac_ver()[0]}"
    except (OSError, ValueError):
        return platform.platform()


def commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src", "bench"], capture_output=True, text=True)
        return sha + ("-dirty" if dirty.stdout.strip() else "") if sha else "uncommitted"
    except OSError:
        return "unknown"


def metadata(**extra) -> dict:
    libs = {}
    for lib in ("numpy", "scikit-learn", "rapidfuzz", "mlx", "mlx-lm"):
        with suppress(PackageNotFoundError):  # mlx is optional
            libs[lib] = version(lib)
    return {
        "date": date.today().isoformat(),
        "hardware": hardware(),
        "python": platform.python_version(),
        "libraries": libs,
        "commit": commit(),
        **extra,
    }


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple | set):
        return list(o)
    raise TypeError(type(o))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
    print(f"wrote {path}")


def load_json(path: Path):
    return json.loads(Path(path).read_text())
