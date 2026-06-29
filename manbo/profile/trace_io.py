from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

TRACE_HEADER = ["step", "elapsed_ms", "step_ms", "loss", "psnr"]


@dataclass(slots=True)
class TraceSeries:
    label: str
    step: list[int]
    elapsed_ms: list[float]
    psnr: list[float]


def parse_trace_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path_str = spec.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
    else:
        path = Path(spec.strip())
        label = path.stem
    if not label:
        raise ValueError(f"invalid trace spec: {spec!r}")
    return label, path


def load_trace_series(
    label: str,
    path: Path,
    *,
    drop_nan_psnr: bool = True,
) -> TraceSeries:
    if not path.exists():
        raise FileNotFoundError(f"trace csv not found: {path}")

    steps: list[int] = []
    elapsed_ms: list[float] = []
    psnr_vals: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = [
            name for name in TRACE_HEADER if name not in (reader.fieldnames or [])
        ]
        if missing_cols:
            raise ValueError(f"trace csv missing columns {missing_cols}: {path}")

        for row in reader:
            psnr = float(row["psnr"])
            if drop_nan_psnr and math.isnan(psnr):
                continue
            steps.append(int(float(row["step"])))
            elapsed_ms.append(float(row["elapsed_ms"]))
            psnr_vals.append(psnr)

    if not steps:
        raise ValueError(f"trace csv has no valid PSNR rows: {path}")

    return TraceSeries(label=label, step=steps, elapsed_ms=elapsed_ms, psnr=psnr_vals)


def write_trace_csv(
    path: Path,
    rows: Iterable[Sequence[float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TRACE_HEADER)
        for row in rows:
            writer.writerow(row)


__all__ = [
    "TRACE_HEADER",
    "TraceSeries",
    "parse_trace_spec",
    "load_trace_series",
    "write_trace_csv",
]
