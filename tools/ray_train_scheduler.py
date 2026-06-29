from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import ray
except ModuleNotFoundError:
    ray = None  # type: ignore[assignment]
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    task_name: str
    target: Path
    output_dir: Path
    log_path: Path


def _parse_numeric_stem(path: Path) -> int | None:
    stem = path.stem
    if not stem.isdigit():
        return None
    try:
        return int(stem)
    except ValueError:
        return None


def _target_sort_key(path: Path) -> tuple[int, int, str]:
    idx = _parse_numeric_stem(path)
    if idx is None:
        return (1, 0, path.name)
    return (0, idx, path.name)


def _resolve_targets(
    *,
    target_dir: Path,
    target_glob: str,
    explicit_targets: list[Path],
    max_targets: int,
    target_modulo: int,
    target_remainder: int,
) -> list[Path]:
    collected_from_dir = False
    if explicit_targets:
        targets = [p for p in explicit_targets if p.is_file()]
        missing = [str(p) for p in explicit_targets if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "some explicit targets are missing or not files: "
                + ", ".join(missing[:8])
            )
    else:
        if not target_dir.exists():
            raise FileNotFoundError(f"target_dir does not exist: {target_dir}")
        if not target_dir.is_dir():
            raise ValueError(f"target_dir must be a directory: {target_dir}")
        targets = sorted(
            (p for p in target_dir.glob(target_glob) if p.is_file()),
            key=_target_sort_key,
        )
        collected_from_dir = True

    modulo = int(target_modulo)
    if collected_from_dir and modulo > 0:
        remainder = int(target_remainder) % modulo
        targets = [
            p
            for p in targets
            if (
                (idx := _parse_numeric_stem(p)) is not None
                and idx % modulo == remainder
            )
        ]

    if max_targets > 0:
        targets = targets[:max_targets]
    if not targets:
        raise FileNotFoundError(
            "no targets found. "
            f"target_dir={target_dir} target_glob={target_glob!r} "
            f"explicit_targets={len(explicit_targets)} max_targets={max_targets} "
            f"target_modulo={target_modulo} target_remainder={target_remainder}"
        )
    return targets


def _build_tasks(targets: list[Path], output_root: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    log_root = output_root / "logs"
    for i, target in enumerate(targets, start=1):
        task_name = f"{i:04d}_{target.stem}"
        output_dir = output_root / task_name
        log_path = log_root / f"{task_name}.log"
        tasks.append(
            TaskSpec(
                task_id=i,
                task_name=task_name,
                target=target.resolve(),
                output_dir=output_dir.resolve(),
                log_path=log_path.resolve(),
            )
        )
    return tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ray Actor scheduler: launch one training subprocess per GPU and "
            "dynamically assign target images."
        )
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("datasets/DIV2K_HR"),
        help="Directory of target images. Ignored when --target is provided.",
    )
    parser.add_argument(
        "--target-glob",
        type=str,
        default="*.png",
        help="Glob for collecting targets from --target-dir.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        action="append",
        default=[],
        help="Explicit target image path(s). Can be repeated.",
    )
    parser.add_argument(
        "--target-modulo",
        type=int,
        default=0,
        help=(
            "Keep only numeric stems where int(stem) %% modulo == remainder. "
            "<=0 disables modulo filtering."
        ),
    )
    parser.add_argument(
        "--target-remainder",
        type=int,
        default=0,
        help="Remainder used with --target-modulo filtering.",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=200,
        help="Maximum number of targets to schedule. <=0 means all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./output/ray_scheduler"),
        help="Root directory for per-task output and logs.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Path to JSON run summary. Default: <output-root>/run_summary.json",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="",
        help="Ray cluster address. Empty means start/use local ray runtime.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Override total worker count. <=0 means auto as "
            "cluster_gpu * workers_per_gpu."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help=(
            "How many persistent workers to start per GPU. "
            "Total auto workers = cluster_gpu * workers_per_gpu."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip task when expected output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved tasks and exit without launching Ray jobs.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help=(
            "Python executable for --train-runner=subprocess and for "
            "rendered command templates."
        ),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train.py"),
        help="Path to training entry script.",
    )
    parser.add_argument(
        "--train-runner",
        type=str,
        choices=["inproc", "subprocess"],
        default="inproc",
        help=(
            "How each worker launches training. "
            "'inproc' calls train.py main() inside the Ray actor process; "
            "'subprocess' spawns a new process per task."
        ),
    )
    parser.add_argument(
        "--train-arg",
        type=str,
        action="append",
        default=[],
        help=(
            "Single argument forwarded to train.py; repeat as needed. "
            "For options, prefer --train-arg=--iterations --train-arg=300."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling new tasks after the first training failure.",
    )

    raw = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in raw:
        sep = raw.index("--")
        parser_argv = raw[:sep]
        passthrough = raw[sep + 1 :]
    else:
        parser_argv = raw

    args = parser.parse_args(parser_argv)
    args.train_args = [*args.train_arg, *passthrough]
    return args


def _expects_checkpoint(train_args: list[str]) -> bool:
    return "--no-save-checkpoint" not in train_args


class _TrainWorker:
    def __init__(
        self,
        *,
        worker_id: int,
        project_root: str,
        python_exe: str,
        train_script: str,
        train_args: list[str],
        train_runner: str,
        skip_existing: bool,
        expect_checkpoint: bool,
    ) -> None:
        self.worker_id = int(worker_id)
        self.project_root = Path(project_root)
        self.python_exe = str(python_exe)
        self.train_script = str(train_script)
        self.train_script_path = Path(self.train_script)
        self.train_args = list(train_args)
        self.train_runner = str(train_runner).strip().lower()
        if self.train_runner not in {"inproc", "subprocess"}:
            raise ValueError(f"unsupported train_runner: {self.train_runner}")
        self.skip_existing = bool(skip_existing)
        self.expect_checkpoint = bool(expect_checkpoint)
        self._cached_gpu_id = self._gpu_id()
        if self._cached_gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._cached_gpu_id
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        self._train_main: Any | None = None

    def _gpu_id(self) -> str:
        cached = str(getattr(self, "_cached_gpu_id", ""))
        if cached:
            return cached
        if ray is None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if not visible:
                return ""
            return visible.split(",", maxsplit=1)[0].strip()
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        gpu_ids = accelerator_ids.get("GPU", [])
        if not gpu_ids:
            return ""
        return str(gpu_ids[0])

    def worker_meta(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "gpu_id": self._gpu_id(),
            "train_runner": self.train_runner,
        }

    def _load_train_main(self) -> Any:
        if self._train_main is not None:
            return self._train_main
        spec = importlib.util.spec_from_file_location(
            f"_ray_train_entry_worker_{self.worker_id}",
            self.train_script_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load train script: {self.train_script_path}")
        module = importlib.util.module_from_spec(spec)
        script_dir = str(self.train_script_path.parent)
        inserted_path = False
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            inserted_path = True
        try:
            spec.loader.exec_module(module)
        finally:
            if inserted_path:
                with contextlib.suppress(ValueError):
                    sys.path.remove(script_dir)
        main_fn = getattr(module, "main", None)
        if not callable(main_fn):
            raise AttributeError(
                f"train script does not define callable main(): {self.train_script_path}"
            )
        self._train_main = main_fn
        return main_fn

    def _invoke_train_main(self, train_argv: list[str]) -> None:
        main_fn = self._load_train_main()
        can_pass_argv = False
        try:
            sig = inspect.signature(main_fn)
            params = list(sig.parameters.values())
            if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
                can_pass_argv = True
            else:
                positional_required = [
                    p
                    for p in params
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    and p.default is inspect.Parameter.empty
                ]
                positional_total = [
                    p
                    for p in params
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                can_pass_argv = (
                    len(positional_required) <= 1 and len(positional_total) >= 1
                )
        except (TypeError, ValueError):
            can_pass_argv = False

        if can_pass_argv:
            main_fn(train_argv)
            return

        saved_argv = list(sys.argv)
        sys.argv = [str(self.train_script_path), *train_argv]
        try:
            main_fn()
        finally:
            sys.argv = saved_argv

    def _run_task_subprocess(
        self, *, cmd: list[str], cmd_str: str, log_path: Path, gpu_id: str
    ) -> tuple[int, float]:
        env = os.environ.copy()
        if gpu_id:
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("PYTHONUNBUFFERED", "1")

        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(
                f"worker_id={self.worker_id} gpu_id={gpu_id} "
                f"runner={self.train_runner}\n"
            )
            log_file.write(f"cmd={cmd_str}\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.perf_counter() - start
        return int(proc.returncode), float(elapsed)

    def _run_task_inproc(
        self, *, train_argv: list[str], cmd_str: str, log_path: Path, gpu_id: str
    ) -> tuple[int, float]:
        start = time.perf_counter()
        return_code = 0
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(
                f"worker_id={self.worker_id} gpu_id={gpu_id} "
                f"runner={self.train_runner}\n"
            )
            log_file.write(f"cmd={cmd_str}\n")
            log_file.flush()
            try:
                with contextlib.chdir(self.project_root):
                    with contextlib.redirect_stdout(log_file):
                        with contextlib.redirect_stderr(log_file):
                            self._invoke_train_main(train_argv)
            except Exception:  # noqa: BLE001
                return_code = 1
                log_file.write("\n")
                traceback.print_exc(file=log_file)
            log_file.flush()
        elapsed = time.perf_counter() - start
        return int(return_code), float(elapsed)

    def run_task(self, task: dict[str, Any]) -> dict[str, Any]:
        task_id = int(task["task_id"])
        task_name = str(task["task_name"])
        target = Path(task["target"])
        output_dir = Path(task["output_dir"])
        log_path = Path(task["log_path"])

        output_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        train_argv = [
            "--target",
            str(target),
            "--output-dir",
            str(output_dir),
            *self.train_args,
        ]
        cmd = [self.python_exe, self.train_script, *train_argv]
        cmd_str = shlex.join(cmd)

        target_stem = target.stem
        final_png = output_dir / f"{target_stem}_final.png"
        model_ckpt = output_dir / f"{target_stem}_model.pt"
        if self.skip_existing:
            if final_png.exists() and (
                (not self.expect_checkpoint) or model_ckpt.exists()
            ):
                return {
                    "task_id": task_id,
                    "task_name": task_name,
                    "status": "skipped",
                    "target": str(target),
                    "worker_id": self.worker_id,
                    "gpu_id": self._gpu_id(),
                    "return_code": 0,
                    "elapsed_sec": 0.0,
                    "log_path": str(log_path),
                    "output_dir": str(output_dir),
                    "command": cmd_str,
                }

        gpu_id = self._gpu_id()
        if self.train_runner == "subprocess":
            return_code, elapsed = self._run_task_subprocess(
                cmd=cmd, cmd_str=cmd_str, log_path=log_path, gpu_id=gpu_id
            )
        else:
            return_code, elapsed = self._run_task_inproc(
                train_argv=train_argv,
                cmd_str=cmd_str,
                log_path=log_path,
                gpu_id=gpu_id,
            )

        return {
            "task_id": task_id,
            "task_name": task_name,
            "status": ("ok" if return_code == 0 else "failed"),
            "target": str(target),
            "worker_id": self.worker_id,
            "gpu_id": gpu_id,
            "return_code": int(return_code),
            "elapsed_sec": float(elapsed),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "command": cmd_str,
        }


if ray is not None:
    TrainWorker = ray.remote(max_restarts=0)(_TrainWorker)
else:
    TrainWorker = _TrainWorker


def _render_task_preview(
    console: Console, tasks: list[TaskSpec], max_rows: int = 10
) -> None:
    table = Table(title=f"Resolved Tasks ({len(tasks)})", show_lines=False)
    table.add_column("id", justify="right", style="cyan")
    table.add_column("task", style="white")
    table.add_column("target", style="green")
    table.add_column("output_dir", style="yellow")
    for task in tasks[:max_rows]:
        table.add_row(
            str(task.task_id),
            task.task_name,
            str(task.target),
            str(task.output_dir),
        )
    if len(tasks) > max_rows:
        table.add_row("...", "...", "...", "...")
    console.print(table)


def _render_result_summary(console: Console, results: list[dict[str, Any]]) -> None:
    table = Table(title="Scheduler Summary", show_lines=False)
    table.add_column("id", justify="right", style="cyan")
    table.add_column("status", style="white")
    table.add_column("worker", justify="right")
    table.add_column("gpu", justify="right")
    table.add_column("secs", justify="right")
    table.add_column("target", style="green")
    table.add_column("log", style="yellow")
    for item in sorted(results, key=lambda x: int(x["task_id"])):
        table.add_row(
            str(item["task_id"]),
            str(item["status"]),
            str(item["worker_id"]),
            str(item["gpu_id"] if item["gpu_id"] != "" else "-"),
            f"{float(item['elapsed_sec']):.2f}",
            str(item["target"]),
            str(item["log_path"]),
        )
    console.print(table)


def _append_event(events: list[str], message: str, *, max_items: int = 12) -> None:
    events.append(message)
    if len(events) > max_items:
        del events[: len(events) - max_items]


def _build_scheduler_state_table(
    *,
    total: int,
    done: int,
    ok: int,
    skipped: int,
    failed: int,
    pending: int,
    inflight: int,
) -> Table:
    table = Table(title="Scheduler State", show_header=False)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    table.add_row("total", str(total))
    table.add_row("done", str(done))
    table.add_row("ok", str(ok))
    table.add_row("skipped", str(skipped))
    table.add_row("failed", str(failed))
    table.add_row("pending", str(pending))
    table.add_row("active", str(inflight))
    return table


def _build_active_tasks_table(
    inflight: dict[ray.ObjectRef, dict[str, Any]],
    *,
    now_perf: float,
    max_rows: int = 8,
) -> Table:
    table = Table(
        title=f"Active Tasks ({len(inflight)})", show_lines=False, expand=True
    )
    table.add_column("worker", justify="right", no_wrap=True)
    table.add_column("gpu", justify="right", no_wrap=True)
    table.add_column("task", justify="right", no_wrap=True)
    table.add_column("target", style="green", no_wrap=True, overflow="ellipsis")
    table.add_column("elapsed", justify="right", no_wrap=True)

    rows: list[tuple[int, str, int, str, float]] = []
    for meta in inflight.values():
        task = meta["task"]
        elapsed = float(max(0.0, now_perf - float(meta["start_perf"])))
        rows.append(
            (
                int(meta["worker_id"]),
                str(meta["gpu_id"]) if str(meta["gpu_id"]) != "" else "-",
                int(task.task_id),
                str(task.target.name),
                elapsed,
            )
        )
    rows.sort(key=lambda x: (x[0], x[2]))
    if not rows:
        table.add_row("-", "-", "-", "-", "-")
        shown_rows = 1
    else:
        display_rows = rows
        overflow_count = 0
        if len(rows) > max_rows:
            display_rows = rows[: max_rows - 1]
            overflow_count = len(rows) - len(display_rows)
        for worker_id, gpu_id, task_id, target_name, elapsed in display_rows:
            table.add_row(
                str(worker_id),
                gpu_id,
                str(task_id),
                target_name,
                f"{elapsed:.1f}s",
            )
        if overflow_count > 0:
            table.add_row("...", "...", f"+{overflow_count} more", "...", "...")
        shown_rows = len(display_rows) + (1 if overflow_count > 0 else 0)

    for _ in range(max(0, max_rows - shown_rows)):
        table.add_row(
            "",
            "",
            "",
            "",
            "",
        )
    return table


def _build_recent_events_table(events: list[str], *, max_rows: int = 8) -> Table:
    table = Table(title="Recent Events", show_header=False, expand=True)
    table.add_column("event", style="white", no_wrap=True, overflow="ellipsis")
    if not events:
        table.add_row("-")
        shown_rows = 1
    else:
        items = events[-max_rows:]
        for item in items:
            table.add_row(item)
        shown_rows = len(items)
    for _ in range(max(0, max_rows - shown_rows)):
        table.add_row("")
    return table


def _build_live_renderable(
    *,
    progress: Progress,
    total: int,
    done: int,
    ok: int,
    skipped: int,
    failed: int,
    pending: int,
    inflight: dict[ray.ObjectRef, dict[str, Any]],
    events: list[str],
) -> Group:
    now_perf = time.perf_counter()
    return Group(
        progress,
        _build_scheduler_state_table(
            total=total,
            done=done,
            ok=ok,
            skipped=skipped,
            failed=failed,
            pending=pending,
            inflight=len(inflight),
        ),
        _build_recent_events_table(events),
        _build_active_tasks_table(inflight, now_perf=now_perf),
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _summary_path(output_root: Path, user_path: Path | None) -> Path:
    if user_path is None:
        return (output_root / "run_summary.json").resolve()
    if user_path.is_absolute():
        return user_path.resolve()
    return (Path.cwd() / user_path).resolve()


def _build_worker_stats(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[int, list[dict[str, Any]]] = {}
    for item in results:
        worker_id = int(item.get("worker_id", -1))
        bucket.setdefault(worker_id, []).append(item)

    worker_rows: list[dict[str, Any]] = []
    for worker_id in sorted(bucket.keys()):
        items = bucket[worker_id]
        elapsed_all = [float(x.get("elapsed_sec", 0.0)) for x in items]
        elapsed_exec = [
            float(x.get("elapsed_sec", 0.0))
            for x in items
            if str(x.get("status")) in {"ok", "failed"}
        ]
        worker_rows.append(
            {
                "worker_id": worker_id,
                "tasks": len(items),
                "ok": sum(1 for x in items if str(x.get("status")) == "ok"),
                "skipped": sum(1 for x in items if str(x.get("status")) == "skipped"),
                "failed": sum(1 for x in items if str(x.get("status")) == "failed"),
                "avg_elapsed_sec_all": _mean(elapsed_all),
                "avg_elapsed_sec_executed": _mean(elapsed_exec),
            }
        )
    return worker_rows


def _write_summary_files(
    summary_path: Path, summary: dict[str, Any]
) -> tuple[Path, Path]:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    markdown_path = summary_path.with_suffix(".md")
    counts = summary.get("counts", {})
    timing = summary.get("timing", {})
    ray_info = summary.get("ray", {})
    commands = summary.get("commands", {})
    failed = summary.get("failed_tasks", [])

    lines = [
        "# Ray Train Scheduler Summary",
        "",
        f"- status: `{summary.get('status', 'unknown')}`",
        f"- start_utc: `{summary.get('start_utc', '')}`",
        f"- end_utc: `{summary.get('end_utc', '')}`",
        f"- wall_elapsed_sec: `{float(timing.get('wall_elapsed_sec', 0.0)):.3f}`",
        f"- cluster_gpu: `{int(ray_info.get('cluster_gpu', 0))}`",
        f"- workers: `{int(ray_info.get('num_workers', 0))}`",
        f"- workers_per_gpu: `{int(ray_info.get('workers_per_gpu', 1))}`",
        f"- worker_gpu_fraction: `{float(ray_info.get('worker_gpu_fraction', 1.0)):.6f}`",
        f"- max_workers_by_gpu: `{int(ray_info.get('max_workers_by_gpu', 0))}`",
        f"- tasks_total: `{int(counts.get('total', 0))}`",
        f"- tasks_started: `{int(counts.get('started', 0))}`",
        f"- ok: `{int(counts.get('ok', 0))}`",
        f"- skipped: `{int(counts.get('skipped', 0))}`",
        f"- failed: `{int(counts.get('failed', 0))}`",
        f"- pending_unstarted: `{int(counts.get('pending_unstarted', 0))}`",
        "",
        "## Timing",
        "",
        f"- avg_task_elapsed_sec_all: `{float(timing.get('avg_task_elapsed_sec_all', 0.0)):.3f}`",
        f"- avg_task_elapsed_sec_executed: `{float(timing.get('avg_task_elapsed_sec_executed', 0.0)):.3f}`",
        f"- avg_task_elapsed_sec_ok: `{float(timing.get('avg_task_elapsed_sec_ok', 0.0)):.3f}`",
        "",
        "## Commands",
        "",
        f"- train_runner: `{str(commands.get('train_runner', 'unknown'))}`",
        "",
        "### scheduler_command",
        "```bash",
        str(commands.get("scheduler_command", "")),
        "```",
        "",
        "### train_command_template",
        "```bash",
        str(commands.get("train_command_template", "")),
        "```",
        "",
        "### forwarded_train_args",
        "```text",
        " ".join(str(x) for x in commands.get("forwarded_train_args", [])),
        "```",
    ]
    if failed:
        lines += ["", "## Failed Tasks", ""]
        for row in failed:
            lines.append(
                f"- task={row.get('task_id')} return_code={row.get('return_code')} "
                f"target={row.get('target')} log={row.get('log_path')}"
            )
    lines += ["", f"json_path={summary_path}"]

    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary_path, markdown_path


def _build_summary_payload(
    *,
    status: str,
    error: str | None,
    start_utc: str,
    end_utc: str,
    wall_elapsed_sec: float,
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path,
    train_script: Path,
    train_args: list[str],
    tasks: list[TaskSpec],
    results: list[dict[str, Any]],
    pending_tasks: int,
    inflight_tasks: int,
    cluster_gpu: int,
    num_workers: int,
    cluster_resources: dict[str, float],
) -> dict[str, Any]:
    workers_per_gpu = int(max(1, int(getattr(args, "workers_per_gpu", 1))))
    worker_gpu_fraction = 1.0 / float(workers_per_gpu)
    max_workers_by_gpu = int(max(0, int(cluster_gpu) * workers_per_gpu))
    total = len(tasks)
    started = len(results)
    ok = sum(1 for x in results if str(x.get("status")) == "ok")
    skipped = sum(1 for x in results if str(x.get("status")) == "skipped")
    failed = sum(1 for x in results if str(x.get("status")) == "failed")

    elapsed_all = [float(x.get("elapsed_sec", 0.0)) for x in results]
    elapsed_exec = [
        float(x.get("elapsed_sec", 0.0))
        for x in results
        if str(x.get("status")) in {"ok", "failed"}
    ]
    elapsed_ok = [
        float(x.get("elapsed_sec", 0.0))
        for x in results
        if str(x.get("status")) == "ok"
    ]

    script_path = Path(__file__).resolve()
    scheduler_command = shlex.join(
        [str(args.python_exe), str(script_path), *list(sys.argv[1:])]
    )
    train_command_template = shlex.join(
        [
            str(args.python_exe),
            str(train_script),
            "--target",
            "<target_path>",
            "--output-dir",
            "<task_output_dir>",
            *train_args,
        ]
    )
    failed_tasks = [
        {
            "task_id": int(x.get("task_id", -1)),
            "target": str(x.get("target", "")),
            "return_code": int(x.get("return_code", -1)),
            "log_path": str(x.get("log_path", "")),
            "command": str(x.get("command", "")),
            "error": str(x.get("error", "")),
        }
        for x in results
        if str(x.get("status")) == "failed"
    ]

    summary = {
        "status": status,
        "error": error,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "ray": {
            "address": str(args.ray_address),
            "cluster_gpu": int(cluster_gpu),
            "num_workers": int(num_workers),
            "workers_per_gpu": workers_per_gpu,
            "worker_gpu_fraction": worker_gpu_fraction,
            "max_workers_by_gpu": max_workers_by_gpu,
            "cluster_resources": cluster_resources,
        },
        "target_selection": {
            "target_dir": str(args.target_dir),
            "target_glob": str(args.target_glob),
            "explicit_targets": [str(p) for p in list(args.target)],
            "target_modulo": int(args.target_modulo),
            "target_remainder": int(args.target_remainder),
            "max_targets": int(args.max_targets),
            "resolved_targets": [str(x.target) for x in tasks],
            "resolved_first": (str(tasks[0].target) if tasks else ""),
            "resolved_last": (str(tasks[-1].target) if tasks else ""),
        },
        "counts": {
            "total": total,
            "started": started,
            "ok": ok,
            "skipped": skipped,
            "failed": failed,
            "pending_unstarted": int(max(0, pending_tasks)),
            "inflight_unfinished": int(max(0, inflight_tasks)),
            "not_finished_total": int(max(0, total - started)),
        },
        "timing": {
            "wall_elapsed_sec": float(max(0.0, wall_elapsed_sec)),
            "avg_task_elapsed_sec_all": _mean(elapsed_all),
            "avg_task_elapsed_sec_executed": _mean(elapsed_exec),
            "avg_task_elapsed_sec_ok": _mean(elapsed_ok),
        },
        "commands": {
            "scheduler_command": scheduler_command,
            "train_command_template": train_command_template,
            "forwarded_train_args": train_args,
            "train_runner": str(args.train_runner),
            "fail_fast": bool(args.fail_fast),
        },
        "worker_stats": _build_worker_stats(results),
        "failed_tasks": failed_tasks,
        "results": sorted(results, key=lambda x: int(x.get("task_id", -1))),
    }
    return summary


def main() -> None:
    args = _parse_args()
    console = Console()
    project_root = Path.cwd().resolve()
    train_script = args.train_script
    if not train_script.is_absolute():
        train_script = (project_root / train_script).resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"train_script not found: {train_script}")

    targets = _resolve_targets(
        target_dir=args.target_dir.resolve(),
        target_glob=str(args.target_glob),
        explicit_targets=list(args.target),
        max_targets=int(args.max_targets),
        target_modulo=int(args.target_modulo),
        target_remainder=int(args.target_remainder),
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = _build_tasks(targets=targets, output_root=output_root)
    _render_task_preview(console, tasks)

    summary_path = _summary_path(output_root, args.summary_path)
    run_start_utc = _utc_now_iso()
    run_start_perf = time.perf_counter()
    train_args: list[str] = list(args.train_args)
    results: list[dict[str, Any]] = []
    cluster_gpu = 0
    num_workers = 0
    workers_per_gpu = int(args.workers_per_gpu)
    if workers_per_gpu <= 0:
        raise ValueError(f"workers_per_gpu must be > 0, got {workers_per_gpu}.")
    worker_gpu_fraction = 1.0 / float(workers_per_gpu)
    max_workers_by_gpu = 0
    cluster_resources: dict[str, float] = {}
    pending_tasks_count = len(tasks)
    inflight_tasks_count = 0
    run_status = "dry_run" if args.dry_run else "running"
    run_error: str | None = None

    if args.dry_run:
        console.print("dry-run enabled: no Ray actors or training tasks were started.")
        console.print(f"python_exe={args.python_exe}")
        console.print(f"train_script={train_script}")
        console.print(f"train_runner={args.train_runner}")
        console.print(
            "workers_per_gpu="
            f"{workers_per_gpu} (per_worker_gpu_fraction={worker_gpu_fraction:.6f})"
        )
        console.print(f"train_args={train_args}")
        run_end_utc = _utc_now_iso()
        summary = _build_summary_payload(
            status=run_status,
            error=run_error,
            start_utc=run_start_utc,
            end_utc=run_end_utc,
            wall_elapsed_sec=time.perf_counter() - run_start_perf,
            args=args,
            project_root=project_root,
            output_root=output_root,
            train_script=train_script,
            train_args=train_args,
            tasks=tasks,
            results=results,
            pending_tasks=pending_tasks_count,
            inflight_tasks=inflight_tasks_count,
            cluster_gpu=cluster_gpu,
            num_workers=num_workers,
            cluster_resources=cluster_resources,
        )
        saved_json, saved_md = _write_summary_files(summary_path, summary)
        console.print(f"summary saved: {saved_json}")
        console.print(f"summary saved: {saved_md}")
        return

    if ray is None:
        console.print(
            "[yellow]Ray is not installed; running tasks locally in sequence.[/yellow]"
        )
        expect_ckpt = _expects_checkpoint(train_args)
        num_workers = 1
        cluster_resources = {"local_sequential_fallback": 1.0}
        worker = _TrainWorker(
            worker_id=0,
            project_root=str(project_root),
            python_exe=str(args.python_exe),
            train_script=str(train_script),
            train_args=train_args,
            train_runner=str(args.train_runner),
            skip_existing=bool(args.skip_existing),
            expect_checkpoint=expect_ckpt,
        )
        progress = Progress(
            TextColumn("[bold cyan]Progress"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
            console=console,
        )
        progress_task = progress.add_task("all_tasks", total=len(tasks), completed=0)
        with progress:
            for task in tasks:
                pending_tasks_count -= 1
                result = worker.run_task(task.__dict__)
                results.append(result)
                progress.update(progress_task, completed=len(results))
                if str(result.get("status")) == "failed" and bool(args.fail_fast):
                    break

        _render_result_summary(console, results)
        ok_count = sum(1 for x in results if str(x.get("status")) == "ok")
        skipped_count = sum(1 for x in results if str(x.get("status")) == "skipped")
        failed_count = sum(1 for x in results if str(x.get("status")) == "failed")
        console.print(
            f"done: ok={ok_count} skipped={skipped_count} failed={failed_count} "
            f"total={len(results)}"
        )
        if failed_count > 0:
            run_status = "failed"
            run_error = f"{failed_count} task(s) failed"
        else:
            run_status = "finished"
        run_end_utc = _utc_now_iso()
        summary = _build_summary_payload(
            status=run_status,
            error=run_error,
            start_utc=run_start_utc,
            end_utc=run_end_utc,
            wall_elapsed_sec=time.perf_counter() - run_start_perf,
            args=args,
            project_root=project_root,
            output_root=output_root,
            train_script=train_script,
            train_args=train_args,
            tasks=tasks,
            results=results,
            pending_tasks=pending_tasks_count,
            inflight_tasks=0,
            cluster_gpu=cluster_gpu,
            num_workers=num_workers,
            cluster_resources=cluster_resources,
        )
        saved_json, saved_md = _write_summary_files(summary_path, summary)
        console.print(f"summary saved: {saved_json}")
        console.print(f"summary saved: {saved_md}")
        if run_error is not None:
            raise RuntimeError(run_error)
        return

    ray_address = str(args.ray_address).strip()
    caught_exc: Exception | None = None
    ray_initialized = False

    try:
        if ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=True)
        else:
            ray.init(ignore_reinit_error=True, log_to_driver=True)
        ray_initialized = True

        cluster_resources = {
            str(k): float(v) for k, v in ray.cluster_resources().items()
        }
        cluster_gpu = int(cluster_resources.get("GPU", 0))
        if cluster_gpu <= 0:
            raise RuntimeError(
                "No GPU resources found in Ray cluster. "
                "Please run on the target 5090 machine (or initialize Ray with GPUs)."
            )
        worker_gpu_fraction = 1.0 / float(workers_per_gpu)
        max_workers_by_gpu = int(cluster_gpu * workers_per_gpu)

        requested_workers = int(args.num_workers)
        if requested_workers > 0:
            num_workers = requested_workers
        else:
            num_workers = max_workers_by_gpu
        if num_workers > max_workers_by_gpu:
            raise ValueError(
                "num_workers "
                f"({num_workers}) exceeds GPU capacity "
                f"({cluster_gpu} * {workers_per_gpu} = {max_workers_by_gpu})."
            )
        num_workers = min(num_workers, len(tasks))
        if num_workers <= 0:
            raise ValueError("num_workers resolved to 0; nothing to run.")

        console.print(
            f"Ray GPUs={cluster_gpu}, workers={num_workers}, tasks={len(tasks)}, "
            f"workers_per_gpu={workers_per_gpu}, "
            f"per_worker_gpu_fraction={worker_gpu_fraction:.6f}, "
            f"skip_existing={bool(args.skip_existing)} "
            f"train_runner={str(args.train_runner)}"
        )

        expect_ckpt = _expects_checkpoint(train_args)
        workers = [
            TrainWorker.options(num_gpus=worker_gpu_fraction).remote(
                worker_id=i,
                project_root=str(project_root),
                python_exe=str(args.python_exe),
                train_script=str(train_script),
                train_args=train_args,
                train_runner=str(args.train_runner),
                skip_existing=bool(args.skip_existing),
                expect_checkpoint=expect_ckpt,
            )
            for i in range(num_workers)
        ]

        worker_metas = ray.get([worker.worker_meta.remote() for worker in workers])
        worker_slots: list[dict[str, Any]] = []
        for worker, meta in zip(workers, worker_metas):
            worker_slots.append(
                {
                    "worker": worker,
                    "worker_id": int(meta.get("worker_id", -1)),
                    "gpu_id": str(meta.get("gpu_id", "")),
                }
            )

        pending_tasks: list[TaskSpec] = list(tasks)
        pending_tasks_count = len(pending_tasks)
        inflight: dict[ray.ObjectRef, dict[str, Any]] = {}
        failed = False
        events: list[str] = []
        progress = Progress(
            TextColumn("[bold cyan]Progress"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
            console=console,
        )
        progress_task = progress.add_task("all_tasks", total=len(tasks), completed=0)
        with Live(
            _build_live_renderable(
                progress=progress,
                total=len(tasks),
                done=0,
                ok=0,
                skipped=0,
                failed=0,
                pending=pending_tasks_count,
                inflight=inflight,
                events=events,
            ),
            console=console,
            refresh_per_second=4,
        ) as live:
            for slot in worker_slots:
                if not pending_tasks:
                    break
                task = pending_tasks.pop(0)
                pending_tasks_count = len(pending_tasks)
                ref = slot["worker"].run_task.remote(task=task.__dict__)
                inflight[ref] = {
                    "worker": slot["worker"],
                    "worker_id": int(slot["worker_id"]),
                    "gpu_id": str(slot["gpu_id"]),
                    "task": task,
                    "start_perf": time.perf_counter(),
                }
                inflight_tasks_count = len(inflight)
                _append_event(
                    events,
                    f"dispatch task={task.task_id} worker={slot['worker_id']} "
                    f"gpu={slot['gpu_id'] if slot['gpu_id'] else '-'} "
                    f"target={task.target.name}",
                )
                live.update(
                    _build_live_renderable(
                        progress=progress,
                        total=len(tasks),
                        done=len(results),
                        ok=sum(1 for x in results if str(x.get("status")) == "ok"),
                        skipped=sum(
                            1 for x in results if str(x.get("status")) == "skipped"
                        ),
                        failed=sum(
                            1 for x in results if str(x.get("status")) == "failed"
                        ),
                        pending=pending_tasks_count,
                        inflight=inflight,
                        events=events,
                    )
                )

            while inflight:
                inflight_tasks_count = len(inflight)
                ready, _ = ray.wait(
                    list(inflight.keys()),
                    num_returns=1,
                    timeout=0.5,
                )
                if not ready:
                    live.update(
                        _build_live_renderable(
                            progress=progress,
                            total=len(tasks),
                            done=len(results),
                            ok=sum(1 for x in results if str(x.get("status")) == "ok"),
                            skipped=sum(
                                1 for x in results if str(x.get("status")) == "skipped"
                            ),
                            failed=sum(
                                1 for x in results if str(x.get("status")) == "failed"
                            ),
                            pending=pending_tasks_count,
                            inflight=inflight,
                            events=events,
                        )
                    )
                    continue

                ref = ready[0]
                meta = inflight.pop(ref)
                try:
                    result = ray.get(ref)
                except Exception as exc:  # noqa: BLE001
                    task = meta.get("task")
                    result = {
                        "task_id": int(task.task_id) if task is not None else -1,
                        "task_name": (
                            str(task.task_name) if task is not None else "<unknown>"
                        ),
                        "status": "failed",
                        "target": str(task.target) if task is not None else "<unknown>",
                        "worker_id": int(meta.get("worker_id", -1)),
                        "gpu_id": str(meta.get("gpu_id", "")),
                        "return_code": -1,
                        "elapsed_sec": float(
                            max(
                                0.0,
                                time.perf_counter()
                                - float(meta.get("start_perf", time.perf_counter())),
                            )
                        ),
                        "log_path": (
                            str(task.log_path) if task is not None else "<unknown>"
                        ),
                        "output_dir": (
                            str(task.output_dir) if task is not None else "<unknown>"
                        ),
                        "error": str(exc),
                        "command": "",
                    }

                results.append(result)
                progress.update(progress_task, completed=len(results))

                status = str(result.get("status", "unknown"))
                _append_event(
                    events,
                    f"done task={result.get('task_id')} status={status} "
                    f"worker={result.get('worker_id')} "
                    f"gpu={result.get('gpu_id') if result.get('gpu_id') else '-'} "
                    f"secs={float(result.get('elapsed_sec', 0.0)):.2f}",
                )
                if status == "failed":
                    failed = True

                if pending_tasks and not (args.fail_fast and failed):
                    next_task = pending_tasks.pop(0)
                    pending_tasks_count = len(pending_tasks)
                    next_ref = meta["worker"].run_task.remote(task=next_task.__dict__)
                    inflight[next_ref] = {
                        "worker": meta["worker"],
                        "worker_id": int(meta.get("worker_id", -1)),
                        "gpu_id": str(meta.get("gpu_id", "")),
                        "task": next_task,
                        "start_perf": time.perf_counter(),
                    }
                    _append_event(
                        events,
                        f"dispatch task={next_task.task_id} "
                        f"worker={meta.get('worker_id')} "
                        f"gpu={meta.get('gpu_id') if meta.get('gpu_id') else '-'} "
                        f"target={next_task.target.name}",
                    )
                inflight_tasks_count = len(inflight)
                live.update(
                    _build_live_renderable(
                        progress=progress,
                        total=len(tasks),
                        done=len(results),
                        ok=sum(1 for x in results if str(x.get("status")) == "ok"),
                        skipped=sum(
                            1 for x in results if str(x.get("status")) == "skipped"
                        ),
                        failed=sum(
                            1 for x in results if str(x.get("status")) == "failed"
                        ),
                        pending=pending_tasks_count,
                        inflight=inflight,
                        events=events,
                    )
                )

        _render_result_summary(console, results)
        ok_count = sum(1 for x in results if str(x.get("status")) == "ok")
        skipped_count = sum(1 for x in results if str(x.get("status")) == "skipped")
        failed_count = sum(1 for x in results if str(x.get("status")) == "failed")
        console.print(
            f"done: ok={ok_count} skipped={skipped_count} failed={failed_count} "
            f"total={len(results)}"
        )
        if failed_count > 0:
            run_status = "failed"
            run_error = f"{failed_count} task(s) failed"
        else:
            run_status = "finished"
    except Exception as exc:  # noqa: BLE001
        caught_exc = exc
        run_status = "failed"
        if run_error is None:
            run_error = str(exc)
    finally:
        if ray is not None and ray_initialized and ray.is_initialized():
            ray.shutdown()
        run_end_utc = _utc_now_iso()
        summary = _build_summary_payload(
            status=run_status,
            error=run_error,
            start_utc=run_start_utc,
            end_utc=run_end_utc,
            wall_elapsed_sec=time.perf_counter() - run_start_perf,
            args=args,
            project_root=project_root,
            output_root=output_root,
            train_script=train_script,
            train_args=train_args,
            tasks=tasks,
            results=results,
            pending_tasks=pending_tasks_count,
            inflight_tasks=inflight_tasks_count,
            cluster_gpu=cluster_gpu,
            num_workers=num_workers,
            cluster_resources=cluster_resources,
        )
        saved_json, saved_md = _write_summary_files(summary_path, summary)
        console.print(f"summary saved: {saved_json}")
        console.print(f"summary saved: {saved_md}")

    if caught_exc is not None:
        raise caught_exc
    if run_error is not None:
        raise RuntimeError(run_error)


if __name__ == "__main__":
    main()
