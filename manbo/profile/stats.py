from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass(slots=True)
class TimingRow:
    label: str
    ms: float
    std: float
    var: float
    n: int


class TimingStats:
    """Container for per-label timing samples in milliseconds."""

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = {}
        self._pending_cuda_events: list[tuple[str, Any, Any]] = []

    def add(self, label: str, ms: float) -> None:
        self._samples.setdefault(label, []).append(float(ms))

    def add_cuda_event(self, label: str, start_event: Any, end_event: Any) -> None:
        self._pending_cuda_events.append((str(label), start_event, end_event))

    def flush_pending_cuda_events(self, *, sync: bool = True) -> None:
        if not self._pending_cuda_events:
            return
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        pending = list(self._pending_cuda_events)
        self._pending_cuda_events.clear()
        for label, start_event, end_event in pending:
            ms = float(start_event.elapsed_time(end_event))
            self._samples.setdefault(label, []).append(ms)

    def reset(self) -> None:
        self._samples.clear()
        self._pending_cuda_events.clear()

    def samples(self, label: str) -> List[float]:
        self.flush_pending_cuda_events()
        return self._samples.get(label, [])

    @property
    def labels(self) -> List[str]:
        self.flush_pending_cuda_events()
        return sorted(self._samples.keys())

    @staticmethod
    def _percentile(xs: Sequence[float], q: float) -> float:
        if not xs:
            return 0.0
        xs2 = sorted(xs)
        idx = int(round((len(xs2) - 1) * q))
        idx = max(0, min(idx, len(xs2) - 1))
        return float(xs2[idx])

    @staticmethod
    def _std_var(xs: Sequence[float]) -> tuple[float, float]:
        if len(xs) < 2:
            return 0.0, 0.0
        var = float(statistics.pvariance(xs))
        std = float(var**0.5)
        return std, var

    def metric(self, label: str, mode: str = "mean") -> float:
        xs = self.samples(label)
        if mode == "n":
            return float(len(xs))
        if not xs:
            return 0.0
        if mode == "sum":
            return float(sum(xs))
        if mode == "mean":
            return float(sum(xs) / len(xs))
        if mode == "median":
            return float(statistics.median(xs))
        if mode == "p90":
            return self._percentile(xs, 0.90)
        if mode == "min":
            return float(min(xs))
        if mode == "max":
            return float(max(xs))
        if mode == "std":
            return self._std_var(xs)[0]
        if mode == "var":
            return self._std_var(xs)[1]
        raise ValueError(f"Unknown metric mode: {mode}")

    def rows(
        self, *, metric: str = "mean", prefix: Optional[str] = None
    ) -> list[TimingRow]:
        rows: list[TimingRow] = []
        for label in self.labels:
            if prefix and not label.startswith(prefix):
                continue
            xs = self.samples(label)
            if not xs:
                continue
            std, var = self._std_var(xs)
            rows.append(
                TimingRow(
                    label=label,
                    ms=self.metric(label, metric),
                    std=std,
                    var=var,
                    n=len(xs),
                )
            )
        return rows

    def report(
        self,
        *,
        title: str,
        metric: str = "mean",
        top_k: int = 40,
        prefix: Optional[str] = None,
        show_hotspots: bool = True,
        show_hierarchy: bool = True,
    ) -> None:
        rows = self.rows(metric=metric, prefix=prefix)
        if not rows:
            print(f"\n{title}: no profiling samples")
            return

        total = float(sum(row.ms for row in rows))
        counts = sorted({row.n for row in rows})
        sample_info = (
            f"samples/label={counts[0]}"
            if len(counts) == 1
            else f"samples/label range=[{counts[0]}, {counts[-1]}]"
        )
        print(f"\n{title} (metric={metric}, {sample_info})")

        if show_hotspots:
            hot_rows = sorted(rows, key=lambda row: row.ms, reverse=True)[
                : max(1, int(top_k))
            ]
            label_w = min(64, max(18, max(len(row.label) for row in hot_rows)))
            print("-" * (label_w + 44))
            print(
                f"{'label':<{label_w}}  {'ms':>10}  {'std':>10}  {'var':>10}  {'share%':>8}"
            )
            print("-" * (label_w + 44))
            for row in hot_rows:
                share = 100.0 * row.ms / max(total, 1e-12)
                print(
                    f"{row.label:<{label_w}}  {row.ms:>10.4f}  {row.std:>10.4f}  {row.var:>10.4f}  {share:>7.2f}%"
                )
            print("-" * (label_w + 44))

        if show_hierarchy:
            print("Hierarchy")
            print("-" * 96)
            prev_parts: list[str] = []
            for row in sorted(rows, key=lambda item: item.label.split(".")):
                disp = row.label
                if prefix and disp.startswith(prefix):
                    disp = disp[len(prefix) :]
                parts = [p for p in disp.split(".") if p]
                if not parts:
                    continue

                common = 0
                while (
                    common < len(prev_parts)
                    and common < len(parts)
                    and prev_parts[common] == parts[common]
                ):
                    common += 1

                for i in range(common, len(parts) - 1):
                    indent = "  " * i
                    print(f"{indent}{parts[i]}")

                leaf_indent = "  " * (len(parts) - 1)
                share = 100.0 * row.ms / max(total, 1e-12)
                print(
                    f"{leaf_indent}- {parts[-1]}: ms={row.ms:.4f} std={row.std:.4f} var={row.var:.4f} share={share:.2f}%"
                )
                prev_parts = parts
            print("-" * 96)


__all__ = ["TimingRow", "TimingStats"]
