from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .stats import TimingStats

_ACTIVE_STEP_PROFILER: ContextVar[Optional["StepProfiler"]] = ContextVar(
    "active_step_profiler", default=None
)
_ACTIVE_SPAN_STACK: ContextVar[tuple[int, ...]] = ContextVar(
    "active_span_stack", default=()
)
_ACTIVE_SPAN_LABEL_STACK: ContextVar[tuple[str, ...]] = ContextVar(
    "active_span_label_stack", default=()
)
_ACTIVE_STEP_PROFILER_GLOBAL: list["StepProfiler"] = []
_ACTIVE_STEP_PROFILER_GLOBAL_LOCK = threading.Lock()


@dataclass(slots=True)
class _SpanRecord:
    span_id: int
    parent_id: Optional[int]
    label: str
    depth: int
    meta: dict[str, Any]
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    start_wall_ns: int
    end_wall_ns: int
    start_perf_ns: int
    end_perf_ns: int


class SpanTraceRecorder:
    """Collect and write per-step nested span traces."""

    def __init__(self, path: Path, *, trace_format: str = "json") -> None:
        self.path = Path(path)
        self.trace_format = str(trace_format)
        if self.trace_format not in {"json", "chrome"}:
            raise ValueError("trace_format must be one of: json, chrome")
        self._steps: list[dict[str, Any]] = []

    @property
    def num_steps(self) -> int:
        return len(self._steps)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(k): SpanTraceRecorder._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [SpanTraceRecorder._jsonable(v) for v in value]
        return repr(value)

    def record_step(self, payload: Mapping[str, Any]) -> None:
        self._steps.append(self._jsonable(dict(payload)))

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.trace_format == "json":
            doc = {
                "format": "manbo-span-trace-v1",
                "steps": self._steps,
            }
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            return

        events: list[dict[str, Any]] = []
        for step in self._steps:
            step_name = str(step.get("name", "step"))
            step_start = int(step.get("start_perf_ns", 0))
            step_end = int(step.get("end_perf_ns", step_start))
            step_dur_us = max(0.0, (step_end - step_start) / 1000.0)
            events.append(
                {
                    "name": step_name,
                    "cat": "step",
                    "ph": "X",
                    "ts": step_start / 1000.0,
                    "dur": step_dur_us,
                    "pid": 1,
                    "tid": 0,
                    "args": {
                        "total_ms": step.get("total_ms", 0.0),
                        "step_index": step.get("step_index"),
                    },
                }
            )

            for span_payload in step.get("spans", []):
                span_start = int(span_payload.get("start_perf_ns", step_start))
                span_end = int(span_payload.get("end_perf_ns", span_start))
                dur_us = max(0.0, (span_end - span_start) / 1000.0)
                events.append(
                    {
                        "name": str(span_payload.get("label", "span")),
                        "cat": "span",
                        "ph": "X",
                        "ts": span_start / 1000.0,
                        "dur": dur_us,
                        "pid": 1,
                        "tid": int(span_payload.get("depth", 0)) + 1,
                        "args": {
                            "span_id": span_payload.get("id"),
                            "parent_id": span_payload.get("parent_id"),
                            "elapsed_ms": span_payload.get("elapsed_ms", 0.0),
                            "meta": span_payload.get("meta", {}),
                        },
                    }
                )

        doc = {
            "traceEvents": events,
            "displayTimeUnit": "ms",
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)


class _NoopCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _SpanCtx:
    __slots__ = (
        "_prof",
        "_label",
        "_meta",
        "_span_id",
        "_parent_id",
        "_depth",
        "_start_event",
        "_end_event",
        "_start_wall_ns",
        "_end_wall_ns",
        "_start_perf_ns",
        "_end_perf_ns",
        "_stack_token",
        "_label_token",
    )

    def __init__(
        self, prof: "StepProfiler", label: str, meta: Mapping[str, Any]
    ) -> None:
        self._prof = prof
        self._label = label
        self._meta = dict(meta)
        self._span_id = -1
        self._parent_id: Optional[int] = None
        self._depth = 0
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
        self._start_wall_ns = 0
        self._end_wall_ns = 0
        self._start_perf_ns = 0
        self._end_perf_ns = 0
        self._stack_token: Optional[Token[tuple[int, ...]]] = None
        self._label_token: Optional[Token[tuple[str, ...]]] = None

    def __enter__(self):
        if not self._prof.enabled:
            return self

        stack = _ACTIVE_SPAN_STACK.get()
        self._span_id = self._prof._alloc_span_id()
        self._parent_id = stack[-1] if stack else None
        self._depth = len(stack)
        self._stack_token = _ACTIVE_SPAN_STACK.set(stack + (self._span_id,))
        label_stack = _ACTIVE_SPAN_LABEL_STACK.get()
        self._label_token = _ACTIVE_SPAN_LABEL_STACK.set(label_stack + (self._label,))

        self._start_wall_ns = time.time_ns()
        self._start_perf_ns = time.perf_counter_ns()
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._prof.enabled:
            return False

        self._end_wall_ns = time.time_ns()
        self._end_perf_ns = time.perf_counter_ns()
        assert self._end_event is not None and self._start_event is not None
        self._end_event.record()
        self._prof._add_span_record(
            _SpanRecord(
                span_id=self._span_id,
                parent_id=self._parent_id,
                label=self._label,
                depth=self._depth,
                meta=dict(self._meta),
                start_event=self._start_event,
                end_event=self._end_event,
                start_wall_ns=self._start_wall_ns,
                end_wall_ns=self._end_wall_ns,
                start_perf_ns=self._start_perf_ns,
                end_perf_ns=self._end_perf_ns,
            )
        )

        if self._stack_token is not None:
            _ACTIVE_SPAN_STACK.reset(self._stack_token)
            self._stack_token = None
        if self._label_token is not None:
            _ACTIVE_SPAN_LABEL_STACK.reset(self._label_token)
            self._label_token = None
        return False


class StepProfiler:
    """CUDA-event profiler with nested span tracking and optional trace export."""

    __slots__ = (
        "name",
        "_stats",
        "_enabled",
        "_spans",
        "_lock",
        "_next_span_id",
        "_step_total_start",
        "_step_total_end",
        "_start_wall_ns",
        "_end_wall_ns",
        "_start_perf_ns",
        "_end_perf_ns",
        "_profile_token",
        "_stack_token",
        "_label_stack_token",
        "_trace_recorder",
        "_step_index",
    )

    def __init__(
        self,
        name: str,
        stats: TimingStats,
        enabled: bool = True,
        *,
        trace_recorder: Optional[SpanTraceRecorder] = None,
        step_index: Optional[int] = None,
    ) -> None:
        self.name = name
        self._stats = stats
        self._enabled = bool(enabled and torch.cuda.is_available())
        self._spans: list[_SpanRecord] = []
        self._lock = threading.Lock()
        self._next_span_id = 1
        self._step_total_start: Optional[torch.cuda.Event] = None
        self._step_total_end: Optional[torch.cuda.Event] = None
        self._start_wall_ns = 0
        self._end_wall_ns = 0
        self._start_perf_ns = 0
        self._end_perf_ns = 0
        self._profile_token: Optional[Token[Optional[StepProfiler]]] = None
        self._stack_token: Optional[Token[tuple[int, ...]]] = None
        self._label_stack_token: Optional[Token[tuple[str, ...]]] = None
        self._trace_recorder = trace_recorder
        self._step_index = step_index

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __enter__(self):
        if not self._enabled:
            return self
        self._profile_token = _ACTIVE_STEP_PROFILER.set(self)
        self._stack_token = _ACTIVE_SPAN_STACK.set(())
        self._label_stack_token = _ACTIVE_SPAN_LABEL_STACK.set(())
        with _ACTIVE_STEP_PROFILER_GLOBAL_LOCK:
            _ACTIVE_STEP_PROFILER_GLOBAL.append(self)

        self._start_wall_ns = time.time_ns()
        self._start_perf_ns = time.perf_counter_ns()
        self._step_total_start = torch.cuda.Event(enable_timing=True)
        self._step_total_end = torch.cuda.Event(enable_timing=True)
        self._step_total_start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._enabled:
            return False

        self._end_wall_ns = time.time_ns()
        self._end_perf_ns = time.perf_counter_ns()
        assert self._step_total_end is not None and self._step_total_start is not None
        self._step_total_end.record()

        if self._stack_token is not None:
            _ACTIVE_SPAN_STACK.reset(self._stack_token)
            self._stack_token = None
        if self._label_stack_token is not None:
            _ACTIVE_SPAN_LABEL_STACK.reset(self._label_stack_token)
            self._label_stack_token = None
        if self._profile_token is not None:
            _ACTIVE_STEP_PROFILER.reset(self._profile_token)
            self._profile_token = None
        with _ACTIVE_STEP_PROFILER_GLOBAL_LOCK:
            if (
                _ACTIVE_STEP_PROFILER_GLOBAL
                and _ACTIVE_STEP_PROFILER_GLOBAL[-1] is self
            ):
                _ACTIVE_STEP_PROFILER_GLOBAL.pop()
            else:
                for idx in range(len(_ACTIVE_STEP_PROFILER_GLOBAL) - 1, -1, -1):
                    if _ACTIVE_STEP_PROFILER_GLOBAL[idx] is self:
                        del _ACTIVE_STEP_PROFILER_GLOBAL[idx]
                        break

        with self._lock:
            span_records = list(self._spans)

        if self._trace_recorder is None:
            for rec in span_records:
                self._stats.add_cuda_event(rec.label, rec.start_event, rec.end_event)
            self._stats.add_cuda_event(
                f"{self.name}.total",
                self._step_total_start,
                self._step_total_end,
            )
            return False

        torch.cuda.synchronize()

        span_payloads: list[dict[str, Any]] = []
        for rec in span_records:
            elapsed_ms = float(rec.start_event.elapsed_time(rec.end_event))
            self._stats.add(rec.label, elapsed_ms)
            span_payloads.append(
                {
                    "id": rec.span_id,
                    "parent_id": rec.parent_id,
                    "label": rec.label,
                    "depth": rec.depth,
                    "elapsed_ms": elapsed_ms,
                    "start_wall_ns": rec.start_wall_ns,
                    "end_wall_ns": rec.end_wall_ns,
                    "start_perf_ns": rec.start_perf_ns,
                    "end_perf_ns": rec.end_perf_ns,
                    "meta": rec.meta,
                }
            )

        total_ms = float(self._step_total_start.elapsed_time(self._step_total_end))
        self._stats.add(f"{self.name}.total", total_ms)

        if self._trace_recorder is not None:
            self._trace_recorder.record_step(
                {
                    "name": self.name,
                    "step_index": self._step_index,
                    "total_ms": total_ms,
                    "start_wall_ns": self._start_wall_ns,
                    "end_wall_ns": self._end_wall_ns,
                    "start_perf_ns": self._start_perf_ns,
                    "end_perf_ns": self._end_perf_ns,
                    "spans": sorted(
                        span_payloads, key=lambda row: row["start_perf_ns"]
                    ),
                }
            )
        return False

    def section(self, label: str, **meta: Any):
        return _SpanCtx(self, label, meta)

    def _alloc_span_id(self) -> int:
        with self._lock:
            current = int(self._next_span_id)
            self._next_span_id += 1
        return current

    def _add_span_record(self, rec: _SpanRecord) -> None:
        with self._lock:
            self._spans.append(rec)


def _get_active_profiler() -> Optional[StepProfiler]:
    ctx_prof = _ACTIVE_STEP_PROFILER.get()
    if ctx_prof is not None:
        return ctx_prof
    with _ACTIVE_STEP_PROFILER_GLOBAL_LOCK:
        return (
            _ACTIVE_STEP_PROFILER_GLOBAL[-1] if _ACTIVE_STEP_PROFILER_GLOBAL else None
        )


class _SpanAPI:
    def __call__(self, label: str, **meta: Any):
        p = _get_active_profiler()
        if p is None:
            return _NoopCtx()
        return p.section(str(label), **meta)

    @staticmethod
    def _scoped_label(scope: str, name: str) -> str:
        seg = str(name).strip()
        if not seg:
            raise ValueError(f"{scope} name must be non-empty")
        label = f"{scope}.{seg}"
        stack = _ACTIVE_SPAN_LABEL_STACK.get()
        if stack:
            return f"{stack[-1]}.{label}"
        return label

    def phase(self, name: str, **meta: Any):
        return self(self._scoped_label("phase", name), **meta)

    def step(self, name: str, **meta: Any):
        return self(self._scoped_label("step", name), **meta)

    def stage(self, name: str, **meta: Any):
        return self(self._scoped_label("stage", name), **meta)


span = _SpanAPI()


def prof(label: str):
    return span(label)


__all__ = [
    "StepProfiler",
    "SpanTraceRecorder",
    "span",
    "prof",
]
