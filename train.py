from __future__ import annotations

import argparse
import logging
import math
import random
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
import torch
from PIL import Image
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.profiler import (
    ProfilerActivity,
    profile as torch_profile,
    record_function,
    schedule as torch_schedule,
    tensorboard_trace_handler,
)

from manbo.gaussian_trace import GaussianTrace
from manbo import set_gs_profile_ctx_factory
from manbo.profile import (
    SpanTraceRecorder,
    StepProfiler,
    TimingStats,
    collect_timing_rows,
    prof,
    render_hierarchy_rich,
    top_rows_for_structlog,
    write_trace_csv,
)
from train_schedules import CurveScheduleController, SparseInitConfig


DEFAULT_TARGET = Path("datasets/DIV2K_HR/0001.png")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GaussianTrace on one or multiple target images."
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--target-glob",
        type=str,
        default="*.png",
        help="When --target is a directory, collect files matching this glob.",
    )
    parser.add_argument(
        "--target-modulo",
        type=int,
        default=0,
        help=(
            "When --target is a directory, keep only numeric stems where "
            "int(stem) %% modulo == remainder. <=0 disables modulo filtering."
        ),
    )
    parser.add_argument(
        "--target-remainder",
        type=int,
        default=0,
        help="Remainder used with --target-modulo filtering.",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--mode", type=str, choices=["closed", "unclosed"], default="closed"
    )
    parser.add_argument(
        "--renderer-backend",
        type=str,
        choices=["gaussian", "cubic", "cubic_fill"],
        default="gaussian",
    )
    parser.add_argument("--cubic-distance-samples-train", type=int, default=12)
    parser.add_argument("--cubic-distance-samples-eval", type=int, default=18)
    parser.add_argument(
        "--cubic-flatten-method",
        type=str,
        choices=["bernstein", "de_casteljau"],
        default="bernstein",
        help=(
            "Curve-to-edge flattening method for cubic polyline/cubic_fill backends. "
            "'de_casteljau' is a trial path for performance experiments."
        ),
    )
    parser.add_argument("--num-curves", type=int, default=512)
    parser.add_argument("--bezier-degree", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--optimizer", type=str, choices=["adam", "adan"], default="adan"
    )
    parser.add_argument(
        "--use-reg-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable regularization loss term during training.",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--block-h", type=int, default=16)
    parser.add_argument("--block-w", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--psnr-every",
        type=int,
        default=0,
        help="Compute train-step PSNR every N steps. <=0 means fallback to --log-every.",
    )
    parser.add_argument("--eval-iters", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("./output/train"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument(
        "--restore-best-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to restore the lowest-loss model state before final eval/export. "
            "Disable to keep last-step parameters."
        ),
    )
    parser.add_argument("--profile-ops", action="store_true")
    parser.add_argument("--profile-every", type=int, default=1)
    parser.add_argument("--profile-skip-first", type=int, default=0)
    parser.add_argument(
        "--profile-metric",
        type=str,
        choices=["sum", "mean", "median", "p90", "min", "max"],
        default="mean",
    )
    parser.add_argument("--profile-topk", type=int, default=40)
    parser.add_argument(
        "--profile-span-trace",
        action="store_true",
        help="Export nested span traces for profiled train steps.",
    )
    parser.add_argument(
        "--profile-span-trace-path",
        type=Path,
        default=None,
        help="Output path for span trace export. Default: <output-dir>/<target>_span_trace.<ext>.",
    )
    parser.add_argument(
        "--profile-span-trace-format",
        type=str,
        choices=["json", "chrome"],
        default="json",
        help="Span trace export format.",
    )
    parser.add_argument(
        "--profile-overlap",
        action="store_true",
        help="Collect per-step overlap stats from renderer payload and correlate with step time.",
    )
    parser.add_argument(
        "--profile-contrib",
        action="store_true",
        help="Collect per-primitive contribution-rate stats from rasterize backward vectors on profiled steps.",
    )
    parser.add_argument(
        "--profile-contrib-topk",
        type=int,
        default=16,
        help="Top-K primitives retained in contribution-rate summary.",
    )
    parser.add_argument(
        "--no-torch-profiler",
        action="store_true",
        help="Disable torch.profiler trace export even when --profile-ops is enabled.",
    )
    parser.add_argument(
        "--torch-profiler-dir",
        type=Path,
        default=None,
        help="Directory for torch.profiler traces (default: <output-dir>/torch_profiler).",
    )
    parser.add_argument("--torch-profiler-record-shapes", action="store_true")
    parser.add_argument("--torch-profiler-profile-memory", action="store_true")
    parser.add_argument("--torch-profiler-with-stack", action="store_true")

    parser.add_argument("--debug-visual", action="store_true")
    parser.add_argument("--debug-every", type=int, default=100)
    parser.add_argument("--debug-dir", type=Path, default=Path("./output/train/debug"))
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=None,
        help="Optional CSV path to dump step/elapsed/loss/psnr traces.",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=1,
        help="Record one trace row every N steps (plus first/last).",
    )
    curve_group = parser.add_argument_group("Curve Schedule")
    curve_group.add_argument(
        "--curve-schedule",
        type=str,
        choices=["none", "layerwised", "prune_densify", "gini_prune", "adaptive"],
        default="none",
        help="Curve-count scheduling strategy during training.",
    )
    curve_group.add_argument("--layerwised-base-curves", type=int, default=24)
    curve_group.add_argument("--layerwised-step-every", type=int, default=1000)
    curve_group.add_argument("--layerwised-max-chunk", type=int, default=64)
    curve_group.add_argument("--layerwised-radii-switch-iter", type=int, default=9200)
    curve_group.add_argument("--layerwised-radii-before", type=float, default=0.02)
    curve_group.add_argument("--layerwised-radii-after", type=float, default=0.01)
    curve_group.add_argument("--layerwised-prune-by-contrib", action="store_true")
    curve_group.add_argument("--layerwised-prune-ratio", type=float, default=0.05)
    curve_group.add_argument("--layerwised-prune-min-keep", type=int, default=24)
    curve_group.add_argument("--layerwised-prune-max-remove", type=int, default=0)
    curve_group.add_argument("--layerwised-prune-warmup-iter", type=int, default=0)
    curve_group.add_argument("--prune-densify-every", type=int, default=500)
    curve_group.add_argument("--prune-densify-start-iter", type=int, default=1000)
    curve_group.add_argument("--prune-densify-end-closed", type=int, default=9200)
    curve_group.add_argument("--prune-densify-end-unclosed", type=int, default=14000)
    curve_group.add_argument("--gini-step-every", type=int, default=500)
    curve_group.add_argument("--gini-warmup-iter", type=int, default=1000)
    curve_group.add_argument("--gini-trigger-low", type=float, default=0.35)
    curve_group.add_argument("--gini-trigger-high", type=float, default=0.85)
    curve_group.add_argument("--gini-prune-min-ratio", type=float, default=0.0)
    curve_group.add_argument("--gini-prune-max-ratio", type=float, default=0.2)
    curve_group.add_argument("--gini-prune-min-keep", type=int, default=24)
    curve_group.add_argument("--gini-prune-max-remove", type=int, default=0)
    curve_group.add_argument("--gini-densify-radii", type=float, default=0.01)
    curve_group.add_argument("--gini-ema-beta", type=float, default=0.8)
    curve_group.add_argument("--merge-step-every", type=int, default=1000)
    curve_group.add_argument("--merge-warmup-iter", type=int, default=2000)
    curve_group.add_argument("--merge-jaccard-min", type=float, default=0.35)
    curve_group.add_argument("--merge-max-pairs", type=int, default=24)
    curve_group.add_argument("--merge-densify-radii", type=float, default=0.01)
    curve_group.add_argument("--merge-tile-block", type=int, default=16)
    curve_group.add_argument(
        "--contrib-area-normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use area-normalized contribution rate eta=C_i/Clamp(A_i) "
            "for schedule contribution vectors. "
            "Default: disabled for closed mode, enabled otherwise."
        ),
    )
    curve_group.add_argument(
        "--contrib-area-clamp-min",
        type=float,
        default=9.0,
        help="Lower clamp bound for A_i when computing eta=C_i/Clamp(A_i).",
    )
    curve_group.add_argument(
        "--contrib-area-clamp-max",
        type=float,
        default=0.0,
        help=(
            "Upper clamp bound for A_i when computing eta. "
            "<=0 means auto (image area H*W)."
        ),
    )
    curve_group.add_argument(
        "--adaptive-freeze-last-steps",
        type=int,
        default=2000,
        help=(
            "For --curve-schedule=adaptive, disable structure adjustment "
            "in the final N training steps."
        ),
    )
    curve_group.add_argument(
        "--curve-schedule-verbose",
        action="store_true",
        help="Print rich per-trigger diagnostics for curve schedule decisions.",
    )

    sparse_group = parser.add_argument_group("Sparse Init / Legacy Alignment")
    sparse_group.add_argument("--sparse-quantile-interval", type=int, default=200)
    sparse_group.add_argument("--sparse-nodiff-thres", type=float, default=0.05)
    sparse_group.add_argument("--sparse-tolerance", type=int, default=1)
    sparse_group.add_argument("--sparse-connectivity", type=int, choices=[4], default=4)
    sparse_group.add_argument("--sparse-max-iters", type=int, default=4096)
    sparse_group.add_argument("--sparse-topk", type=int, default=4096)
    sparse_group.add_argument("--sparse-suppress-radius", type=int, default=None)
    return parser.parse_args(argv)


def configure_logger() -> Any:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    data = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = data.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.dim() != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"expected image [1, 3, H, W], got {tuple(image.shape)}")
    chw = image.detach().clamp(0.0, 1.0).squeeze(0)
    hwc = (chw.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().contiguous()
    h, w, _ = hwc.shape
    return Image.frombytes("RGB", (w, h), hwc.numpy().tobytes())


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().to(device="cpu").clone()
        else:
            out[key] = value
    return out


def build_model(
    args: argparse.Namespace,
    target: torch.Tensor,
    device: torch.device,
    *,
    num_curves_override: Optional[int] = None,
) -> GaussianTrace:
    _, _, h, w = target.shape
    num_curves = int(
        args.num_curves if num_curves_override is None else num_curves_override
    )
    model = GaussianTrace(
        H=h,
        W=w,
        BLOCK_W=args.block_w,
        BLOCK_H=args.block_h,
        device=device,
        mode=args.mode,
        num_curves=num_curves,
        bezier_degree=args.bezier_degree,
        num_samples=args.num_samples,
        lr=args.lr,
        opt_type=args.optimizer,
        use_reg_loss=getattr(args, "use_reg_loss", False),
        renderer_backend=getattr(args, "renderer_backend", "gaussian"),
        cubic_distance_samples_train=getattr(args, "cubic_distance_samples_train", 12),
        cubic_distance_samples_eval=getattr(args, "cubic_distance_samples_eval", 18),
        cubic_flatten_method=getattr(args, "cubic_flatten_method", "bernstein"),
        quantize=False,
    ).to(device)

    if args.model_path is not None:
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state, strict=False)
    return model


def run_eval(
    model: GaussianTrace, target: torch.Tensor, eval_iters: int
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        pred = model()["render"]
        mse = torch.mean((pred - target) ** 2).item()
        final_psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()

        torch.cuda.synchronize() if target.is_cuda else None
        t0 = time.perf_counter()
        for _ in range(eval_iters):
            _ = model()["render"]
        torch.cuda.synchronize() if target.is_cuda else None
        avg_eval_ms = (time.perf_counter() - t0) * 1000.0 / max(1, eval_iters)

    model.train()
    return final_psnr, avg_eval_ms


def _render_kv_table(console: Console, title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


def _fmt_schedule_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000.0 or abs(value) < 1e-3:
            return f"{value:.4e}"
        return f"{value:.6f}"
    return str(value)


def _render_curve_schedule_events(
    console: Console, events: list[dict[str, Any]]
) -> None:
    if not events:
        return
    preferred = [
        "phase",
        "status",
        "reason",
        "curves_before",
        "curves_after",
        "removed",
        "densified",
        "remove_count",
        "remove_ratio",
        "gini_now",
        "gini_ema_prev",
        "gini_ema",
        "saturation",
        "merge_trigger",
        "tile_count",
        "candidate_pairs",
        "selected_pairs",
        "merged_pairs",
        "jaccard_threshold",
    ]
    for event in events:
        if not isinstance(event, dict):
            continue
        step = int(event.get("step", -1))
        sched = str(event.get("schedule", "unknown"))
        table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        table.add_column("key", style="cyan", no_wrap=True)
        table.add_column("value", style="white")
        used: set[str] = set()
        for key in preferred:
            if key in event:
                table.add_row(key, _fmt_schedule_value(event[key]))
                used.add(key)
        for key in sorted(event.keys()):
            if key in used or key in {"step", "schedule"}:
                continue
            table.add_row(str(key), _fmt_schedule_value(event[key]))
        console.print(
            Panel(
                table,
                title=f"Curve Schedule step={step} [{sched}]",
                border_style="yellow",
            )
        )


def _fmt_prof_us(value_us: float) -> str:
    val = float(max(0.0, value_us))
    if val >= 1_000_000.0:
        return f"{val / 1_000_000.0:.3f}s"
    if val >= 1_000.0:
        return f"{val / 1_000.0:.3f}ms"
    return f"{val:.1f}us"


def _render_torch_profiler_summary(console: Console, key_averages: Any) -> None:
    events = list(key_averages)
    if not events:
        console.print("\ntorch.profiler key_averages: no events")
        return

    def _self_cuda_total(evt: Any) -> float:
        return float(
            getattr(
                evt,
                "self_cuda_time_total",
                getattr(evt, "self_device_time_total", 0.0),
            )
        )

    def _cuda_total(evt: Any) -> float:
        return float(
            getattr(
                evt,
                "cuda_time_total",
                getattr(evt, "device_time_total", 0.0),
            )
        )

    def _self_cuda(evt: Any) -> float:
        return _self_cuda_total(evt)

    def _self_cpu(evt: Any) -> float:
        return float(getattr(evt, "self_cpu_time_total", 0.0))

    sort_key = _self_cuda if any(_self_cuda(evt) > 0.0 for evt in events) else _self_cpu
    events = sorted(events, key=sort_key, reverse=True)[:20]

    table = Table(
        title="torch.profiler key_averages (top20)",
        box=box.SIMPLE,
        show_lines=False,
        pad_edge=False,
    )
    table.add_column(
        "op", style="cyan", no_wrap=True, overflow="ellipsis", max_width=42
    )
    table.add_column("self_cuda", justify="right")
    table.add_column("cuda_total", justify="right")
    table.add_column("self_cpu", justify="right")
    table.add_column("cpu_total", justify="right")
    table.add_column("calls", justify="right")

    for evt in events:
        table.add_row(
            str(getattr(evt, "key", "<unknown>")),
            _fmt_prof_us(_self_cuda_total(evt)),
            _fmt_prof_us(_cuda_total(evt)),
            _fmt_prof_us(getattr(evt, "self_cpu_time_total", 0.0)),
            _fmt_prof_us(getattr(evt, "cpu_time_total", 0.0)),
            str(int(getattr(evt, "count", 0))),
        )
    console.print()
    console.print(table)


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _corr(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a2 = a[:n]
    b2 = b[:n]
    ma = sum(a2) / n
    mb = sum(b2) / n
    va = sum((x - ma) * (x - ma) for x in a2) / n
    vb = sum((y - mb) * (y - mb) for y in b2) / n
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a2, b2)) / n
    return cov / math.sqrt(va * vb)


def _parse_target_index(path: Path) -> Optional[int]:
    stem = path.stem
    if not stem.isdigit():
        return None
    try:
        return int(stem)
    except ValueError:
        return None


def _target_sort_key(path: Path) -> tuple[int, int, str]:
    index = _parse_target_index(path)
    if index is None:
        return (1, 0, path.name)
    return (0, index, path.name)


def resolve_target_paths(
    target: Path,
    *,
    target_glob: str,
    target_modulo: int,
    target_remainder: int,
) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"target path not found: {target}")
    if not target.is_dir():
        raise ValueError(f"target path must be a file or directory: {target}")

    candidates = sorted(
        (p for p in target.glob(target_glob) if p.is_file()), key=_target_sort_key
    )
    modulo = int(target_modulo)
    if modulo > 0:
        remainder = int(target_remainder) % modulo
        candidates = [
            p
            for p in candidates
            if (
                (idx := _parse_target_index(p)) is not None
                and idx % modulo == remainder
            )
        ]

    if not candidates:
        raise FileNotFoundError(
            "no target files matched "
            f"path={target} glob={target_glob!r} "
            f"target_modulo={target_modulo} target_remainder={target_remainder}"
        )
    return candidates


def _suffix_path_for_target(path: Path, target_stem: str) -> Path:
    suffix = "".join(path.suffixes)
    if suffix:
        base_name = path.name[: -len(suffix)]
    else:
        base_name = path.name
    return path.with_name(f"{base_name}_{target_stem}{suffix}")


def _target_output_paths(output_dir: Path, target_stem: str) -> tuple[Path, Path]:
    out_path = output_dir / f"{target_stem}_final.png"
    ckpt_path = output_dir / f"{target_stem}_model.pt"
    return out_path, ckpt_path


def _run_single_target(
    args: argparse.Namespace,
    *,
    target_path: Path,
    logger: Any,
    console: Console,
) -> dict[str, Any]:

    backend_norm = str(getattr(args, "renderer_backend", "gaussian")).strip().lower()
    mode_norm = str(args.mode).strip().lower()
    if backend_norm == "cubic" and mode_norm == "closed":
        logger.warning(
            "renderer_backend.cubic.redirect_to_fill",
            requested_backend="cubic",
            applied_backend="cubic_fill",
            mode="closed",
            reason="closed mode should use fill backend",
        )
        args.renderer_backend = "cubic_fill"
        backend_norm = "cubic_fill"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = load_image_tensor(target_path, device)
    curve_schedule_mode = str(args.curve_schedule)
    if args.model_path is not None and curve_schedule_mode != "none":
        logger.warning(
            "curve_schedule.disabled_for_resume",
            model_path=str(args.model_path),
            requested=curve_schedule_mode,
        )
        curve_schedule_mode = "none"
    if args.layerwised_prune_by_contrib and curve_schedule_mode != "layerwised":
        logger.warning(
            "layerwised_prune_by_contrib.ignored",
            reason="requires --curve-schedule=layerwised",
            curve_schedule=curve_schedule_mode,
        )
    contrib_backend_supported = backend_norm in {"gaussian", "cubic", "cubic_fill"}
    layerwised_prune_requested = bool(
        curve_schedule_mode == "layerwised" and args.layerwised_prune_by_contrib
    )
    layerwised_prune_contrib_enabled = bool(
        layerwised_prune_requested and contrib_backend_supported
    )
    if layerwised_prune_requested and not contrib_backend_supported:
        logger.warning(
            "layerwised_prune_by_contrib.disabled",
            reason="currently supported only for gaussian/cubic/cubic_fill backend",
            renderer_backend=backend_norm,
        )
    gini_prune_requested = bool(curve_schedule_mode == "gini_prune")
    gini_prune_enabled = bool(gini_prune_requested and contrib_backend_supported)
    if gini_prune_requested and not contrib_backend_supported:
        logger.warning(
            "gini_prune.disabled",
            reason="requires contribution stats on gaussian/cubic/cubic_fill backend",
            renderer_backend=backend_norm,
        )
        curve_schedule_mode = "none"
    adaptive_requested = bool(curve_schedule_mode == "adaptive")
    adaptive_enabled = bool(adaptive_requested and contrib_backend_supported)
    if adaptive_requested and not contrib_backend_supported:
        logger.warning(
            "adaptive.disabled",
            reason="requires contribution stats on gaussian/cubic/cubic_fill backend",
            renderer_backend=backend_norm,
        )
        curve_schedule_mode = "none"
    if adaptive_requested:
        logger.info(
            "adaptive.merge_trigger_mode",
            mode="gini_state",
            rule="gini_now > gini_ema switches to merge",
            merge_step_every_ignored=True,
        )
    schedule_contrib_enabled = bool(
        layerwised_prune_contrib_enabled or gini_prune_enabled or adaptive_enabled
    )
    schedule_isect_contrib_enabled = bool(adaptive_enabled)

    model_init_curves = int(args.num_curves)
    if curve_schedule_mode == "layerwised" and args.model_path is None:
        model_init_curves = int(
            min(args.num_curves, max(1, args.layerwised_base_curves))
        )

    model = build_model(
        args,
        target,
        device,
        num_curves_override=model_init_curves,
    )
    curve_scheduler = CurveScheduleController(
        schedule=curve_schedule_mode,
        target_num_curves=int(args.num_curves),
        model_mode=str(args.mode),
        total_steps=int(args.iterations),
        adaptive_freeze_last_steps=int(args.adaptive_freeze_last_steps),
        sparse_init_config=SparseInitConfig(
            quantile_interval=int(args.sparse_quantile_interval),
            nodiff_thres=float(args.sparse_nodiff_thres),
            tolerance=int(args.sparse_tolerance),
            connectivity=int(args.sparse_connectivity),
            max_iters=int(args.sparse_max_iters),
            topk=int(args.sparse_topk),
            suppress_radius=(
                None
                if args.sparse_suppress_radius is None
                else int(args.sparse_suppress_radius)
            ),
        ),
        layerwised_base_curves=int(model_init_curves),
        layerwised_step_every=int(args.layerwised_step_every),
        layerwised_max_chunk=int(args.layerwised_max_chunk),
        layerwised_radii_switch_iter=int(args.layerwised_radii_switch_iter),
        layerwised_radii_before=float(args.layerwised_radii_before),
        layerwised_radii_after=float(args.layerwised_radii_after),
        layerwised_prune_by_contrib=bool(layerwised_prune_contrib_enabled),
        layerwised_prune_ratio=float(args.layerwised_prune_ratio),
        layerwised_prune_min_keep=int(args.layerwised_prune_min_keep),
        layerwised_prune_max_remove=int(args.layerwised_prune_max_remove),
        layerwised_prune_warmup_iter=int(args.layerwised_prune_warmup_iter),
        prune_densify_every=int(args.prune_densify_every),
        prune_densify_start_iter=int(args.prune_densify_start_iter),
        prune_densify_end_closed=int(args.prune_densify_end_closed),
        prune_densify_end_unclosed=int(args.prune_densify_end_unclosed),
        gini_step_every=int(args.gini_step_every),
        gini_warmup_iter=int(args.gini_warmup_iter),
        gini_trigger_low=float(args.gini_trigger_low),
        gini_trigger_high=float(args.gini_trigger_high),
        gini_prune_min_ratio=float(args.gini_prune_min_ratio),
        gini_prune_max_ratio=float(args.gini_prune_max_ratio),
        gini_prune_min_keep=int(args.gini_prune_min_keep),
        gini_prune_max_remove=int(args.gini_prune_max_remove),
        gini_densify_radii=float(args.gini_densify_radii),
        gini_ema_beta=float(args.gini_ema_beta),
        merge_step_every=int(args.merge_step_every),
        merge_warmup_iter=int(args.merge_warmup_iter),
        merge_jaccard_min=float(args.merge_jaccard_min),
        merge_max_pairs=int(args.merge_max_pairs),
        merge_densify_radii=float(args.merge_densify_radii),
        merge_tile_block=int(args.merge_tile_block),
    )
    stats = TimingStats()

    profiling_enabled = bool(args.profile_ops and device.type == "cuda")
    set_gs_profile_ctx_factory(prof if profiling_enabled else None)
    profile_contrib_enabled = bool(
        args.profile_contrib
        and profiling_enabled
        and backend_norm in {"gaussian", "cubic", "cubic_fill"}
    )
    if args.profile_contrib and not profiling_enabled:
        logger.warning(
            "profile_contrib.disabled",
            reason="requires --profile-ops on cuda",
        )
    if args.profile_contrib and backend_norm not in {"gaussian", "cubic", "cubic_fill"}:
        logger.warning(
            "profile_contrib.disabled",
            reason="currently supported only for gaussian/cubic/cubic_fill backend",
            renderer_backend=backend_norm,
        )
    model.contrib_stats_enabled = bool(
        profile_contrib_enabled or schedule_contrib_enabled
    )
    model.contrib_stats_topk = int(max(1, int(args.profile_contrib_topk)))
    model.contrib_stats_keep_vector = bool(schedule_contrib_enabled)
    model.contrib_stats_collect_isect = bool(schedule_isect_contrib_enabled)
    area_norm_arg = getattr(args, "contrib_area_normalize", None)
    if area_norm_arg is None:
        model.contrib_stats_area_normalize = str(args.mode).lower() != "closed"
    else:
        model.contrib_stats_area_normalize = bool(area_norm_arg)
    area_clamp_min = float(max(0.0, float(args.contrib_area_clamp_min)))
    area_clamp_max = float(args.contrib_area_clamp_max)
    if area_clamp_max <= 0.0:
        area_clamp_max = float(max(1, int(model.H) * int(model.W)))
    if area_clamp_max < area_clamp_min:
        area_clamp_max = area_clamp_min
    model.contrib_stats_area_clamp_min = float(area_clamp_min)
    model.contrib_stats_area_clamp_max = float(area_clamp_max)
    model.contrib_stats_collect = False
    model._last_contrib_payload = {}
    span_trace_recorder: Optional[SpanTraceRecorder] = None
    if args.profile_span_trace and not profiling_enabled:
        logger.warning(
            "profile_span_trace.disabled",
            reason="requires --profile-ops on cuda",
        )
    if args.profile_span_trace and profiling_enabled:
        if args.profile_span_trace_path is None:
            ext = "json" if args.profile_span_trace_format == "json" else "chrome.json"
            span_trace_path = args.output_dir / f"{target_path.stem}_span_trace.{ext}"
        else:
            span_trace_path = args.profile_span_trace_path
        span_trace_recorder = SpanTraceRecorder(
            span_trace_path,
            trace_format=args.profile_span_trace_format,
        )
        logger.info(
            "profile_span_trace.enabled",
            path=str(span_trace_path),
            fmt=args.profile_span_trace_format,
        )

    torch_profiler = None
    torch_profiler_active = False
    torch_profiler_started = False
    torch_profiler_start_step: Optional[int] = None
    torch_profiler_end_step: Optional[int] = None
    torch_profiler_trace_dir: Optional[Path] = None
    if profiling_enabled and not args.no_torch_profiler:
        wait_steps = max(0, int(args.profile_skip_first))
        remaining_steps = max(0, int(args.iterations - wait_steps))
        desired_warmup = 1
        desired_active = 20
        schedule_source = "default_short_window"

        warmup_steps = min(desired_warmup, max(0, remaining_steps - 1))
        active_capture_steps = min(
            desired_active, max(0, remaining_steps - warmup_steps)
        )
        start_step = wait_steps + 1
        total_capture_steps = warmup_steps + active_capture_steps
        if active_capture_steps >= 1:
            trace_root = args.torch_profiler_dir or (args.output_dir / "torch_profiler")
            torch_profiler_trace_dir = trace_root / target_path.stem
            torch_profiler_trace_dir.mkdir(parents=True, exist_ok=True)
            torch_profiler = torch_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch_schedule(
                    wait=0,
                    warmup=warmup_steps,
                    active=active_capture_steps,
                    repeat=1,
                ),
                on_trace_ready=tensorboard_trace_handler(str(torch_profiler_trace_dir)),
                record_shapes=args.torch_profiler_record_shapes,
                profile_memory=args.torch_profiler_profile_memory,
                with_stack=args.torch_profiler_with_stack,
                acc_events=True,
            )
            logger.info(
                "torch_profiler.enabled",
                output_dir=str(torch_profiler_trace_dir),
                schedule_source=schedule_source,
                start_step=start_step,
                end_step=(start_step + total_capture_steps - 1),
                warmup_steps=warmup_steps,
                active_steps=active_capture_steps,
                remaining_steps=remaining_steps,
            )
            torch_profiler_start_step = start_step
            torch_profiler_end_step = start_step + total_capture_steps - 1
        else:
            logger.warning(
                "torch_profiler.skipped",
                reason="no_active_steps_after_schedule",
                iterations=args.iterations,
                profile_skip_first=args.profile_skip_first,
                remaining_steps=remaining_steps,
                desired_warmup=desired_warmup,
                desired_active=desired_active,
            )

    console.print(
        Panel.fit(
            (
                f"target={target_path}\n"
                f"shape={tuple(target.shape)}\n"
                f"mode={args.mode} curves={args.num_curves} iters={args.iterations}\n"
                f"optimizer={args.optimizer}\n"
                f"curve_schedule={curve_schedule_mode} init_curves={model_init_curves}"
            ),
            title="Training",
            border_style="cyan",
        )
    )
    logger.info(
        "train.start",
        target=str(target_path),
        shape=tuple(target.shape),
        mode=args.mode,
        curves=args.num_curves,
        iterations=args.iterations,
        optimizer=args.optimizer,
        curve_schedule=curve_schedule_mode,
        init_curves=model_init_curves,
        profile_contrib=bool(profile_contrib_enabled),
        profile_contrib_topk=int(max(1, int(args.profile_contrib_topk))),
        schedule_contrib_enabled=bool(schedule_contrib_enabled),
        layerwised_prune_by_contrib=bool(layerwised_prune_contrib_enabled),
        layerwised_prune_ratio=float(max(0.0, float(args.layerwised_prune_ratio))),
        gini_prune_enabled=bool(gini_prune_enabled),
        gini_prune_max_ratio=float(max(0.0, float(args.gini_prune_max_ratio))),
        gini_trigger_low=float(args.gini_trigger_low),
        gini_trigger_high=float(args.gini_trigger_high),
        adaptive_enabled=bool(adaptive_enabled),
        merge_jaccard_min=float(args.merge_jaccard_min),
        merge_max_pairs=int(max(0, int(args.merge_max_pairs))),
        schedule_isect_contrib_enabled=bool(schedule_isect_contrib_enabled),
        contrib_area_normalize=bool(model.contrib_stats_area_normalize),
        contrib_area_clamp_min=float(model.contrib_stats_area_clamp_min),
        contrib_area_clamp_max=float(model.contrib_stats_area_clamp_max),
        curve_schedule_verbose=bool(args.curve_schedule_verbose),
        restore_best_at_end=bool(args.restore_best_at_end),
    )

    t0 = time.perf_counter()
    last_loss = 0.0
    last_psnr = 0.0
    best_loss = float("inf")
    best_step = 0
    best_state_dict: Optional[dict[str, Any]] = None
    profiled_steps = [
        s
        for s in range(1, args.iterations + 1)
        if s > args.profile_skip_first and s % max(1, args.profile_every) == 0
    ]
    profile_start_step = profiled_steps[0] if profiled_steps else None
    profile_end_step = profiled_steps[-1] if profiled_steps else None
    profile_wall_start_ns: Optional[int] = None
    profile_wall_end_ns: Optional[int] = None
    profile_perf_start_ns: Optional[int] = None
    profile_perf_end_ns: Optional[int] = None

    overlap_steps: list[int] = []
    overlap_num_points: list[float] = []
    overlap_num_intersects: list[float] = []
    overlap_hits_per_gaussian: list[float] = []
    overlap_hits_max: list[float] = []
    overlap_area_ratio: list[float] = []
    contrib_steps: list[int] = []
    contrib_top1_rate: list[float] = []
    contrib_topk_rate_sum: list[float] = []
    contrib_effective_curves: list[float] = []
    contrib_active_ratio: list[float] = []
    contrib_r_share: list[float] = []
    contrib_g_share: list[float] = []
    contrib_b_share: list[float] = []
    contrib_topk_used: Optional[int] = None

    trace_rows: list[tuple[int, float, float, float, float]] = []
    torch_profiler_summary: Any = None
    progress = Progress(
        TextColumn("[bold cyan]Training[/bold cyan]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("loss {task.fields[loss]}"),
        TextColumn("psnr {task.fields[psnr]}"),
        TextColumn("it/s {task.fields[it_s]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    with progress:
        task_id = progress.add_task(
            "train",
            total=args.iterations,
            loss="--",
            psnr="--",
            it_s="--",
        )
        last_step_perf = t0
        try:
            for step in range(1, args.iterations + 1):
                if (
                    torch_profiler is not None
                    and not torch_profiler_active
                    and torch_profiler_start_step is not None
                    and step == torch_profiler_start_step
                ):
                    torch_profiler.__enter__()
                    torch_profiler_active = True
                    torch_profiler_started = True

                if (
                    profiling_enabled
                    and profile_start_step is not None
                    and step == profile_start_step
                ):
                    profile_wall_start_ns = time.time_ns()
                    profile_perf_start_ns = time.perf_counter_ns()

                step_profile_on = (
                    profiling_enabled
                    and step > args.profile_skip_first
                    and step % max(1, args.profile_every) == 0
                )
                contrib_profile_collect = bool(
                    profile_contrib_enabled and step_profile_on
                )
                contrib_schedule_collect = bool(
                    schedule_contrib_enabled and curve_scheduler.needs_contrib(step)
                )
                model.contrib_stats_collect = bool(
                    contrib_profile_collect or contrib_schedule_collect
                )
                psnr_every = args.psnr_every if args.psnr_every > 0 else args.log_every
                need_psnr = (
                    step == 1
                    or step % max(1, psnr_every) == 0
                    or step == args.iterations
                )

                step_record_ctx = (
                    record_function("train.step")
                    if torch_profiler_active
                    else nullcontext()
                )
                with step_record_ctx:
                    with StepProfiler(
                        "train.step",
                        stats,
                        enabled=step_profile_on,
                        trace_recorder=span_trace_recorder,
                        step_index=step,
                    ):
                        loss, step_psnr, pred_image = model.train_step(
                            target, compute_psnr=need_psnr
                        )

                if curve_scheduler.enabled:
                    with torch.no_grad():
                        schedule_events = curve_scheduler.after_step(
                            step=step,
                            model=model,
                            gt_image=target,
                            pred_image=pred_image.detach(),
                        )
                    if args.curve_schedule_verbose and schedule_events:
                        _render_curve_schedule_events(console, schedule_events)

                if torch_profiler_active:
                    torch_profiler.step()

                if (
                    torch_profiler_active
                    and torch_profiler_end_step is not None
                    and step >= torch_profiler_end_step
                ):
                    torch_profiler.__exit__(None, None, None)
                    torch_profiler_active = False

                last_loss = float(loss.item())
                if step_psnr is not None:
                    last_psnr = float(step_psnr)
                if last_loss < best_loss:
                    best_loss = float(last_loss)
                    best_step = int(step)
                    if args.restore_best_at_end:
                        best_state_dict = _clone_state_dict_to_cpu(model)
                now_perf = time.perf_counter()
                step_ms = (now_perf - last_step_perf) * 1000.0
                elapsed_ms = (now_perf - t0) * 1000.0
                last_step_perf = now_perf
                progress.advance(task_id, 1)

                if args.trace_csv is not None:
                    trace_on = (
                        step == 1
                        or step == args.iterations
                        or step % max(1, args.trace_every) == 0
                    )
                    if trace_on:
                        trace_rows.append(
                            (
                                step,
                                elapsed_ms,
                                step_ms,
                                last_loss,
                                float("nan") if step_psnr is None else float(step_psnr),
                            )
                        )

                if step_profile_on and args.profile_overlap:
                    payload = getattr(model, "_last_project_payload", None)
                    if (
                        isinstance(payload, dict)
                        and "num_tiles_hit" in payload
                        and "radii" in payload
                    ):
                        num_tiles_hit = payload["num_tiles_hit"]
                        radii = payload["radii"].float()
                        num_points = int(num_tiles_hit.numel())
                        num_intersects = int(num_tiles_hit.sum().item())
                        hits_per_gaussian = float(num_intersects / max(1, num_points))
                        hits_max = float(num_tiles_hit.max().item())
                        total_area = float((torch.pi * (radii * radii)).sum().item())
                        area_ratio = total_area / max(1.0, float(model.H * model.W))

                        overlap_steps.append(step)
                        overlap_num_points.append(float(num_points))
                        overlap_num_intersects.append(float(num_intersects))
                        overlap_hits_per_gaussian.append(hits_per_gaussian)
                        overlap_hits_max.append(hits_max)
                        overlap_area_ratio.append(area_ratio)

                if step_profile_on and profile_contrib_enabled:
                    payload = getattr(model, "_last_contrib_payload", None)
                    if isinstance(payload, dict) and int(
                        payload.get("step", -1)
                    ) == int(step):
                        score_sum = max(
                            float(payload.get("curve_contrib_score_sum", 0.0)),
                            1e-12,
                        )
                        r_sum = float(payload.get("curve_contrib_r_sum", 0.0))
                        g_sum = float(payload.get("curve_contrib_g_sum", 0.0))
                        b_sum = float(payload.get("curve_contrib_b_sum", 0.0))
                        contrib_steps.append(step)
                        contrib_top1_rate.append(float(payload.get("top1_rate", 0.0)))
                        contrib_topk_rate_sum.append(
                            float(payload.get("topk_rate_sum", 0.0))
                        )
                        contrib_effective_curves.append(
                            float(payload.get("effective_curves", 0.0))
                        )
                        contrib_active_ratio.append(
                            float(payload.get("active_ratio", 0.0))
                        )
                        contrib_r_share.append(r_sum / score_sum)
                        contrib_g_share.append(g_sum / score_sum)
                        contrib_b_share.append(b_sum / score_sum)
                        contrib_topk_used = int(payload.get("topk", 0))

                if step == 1 or step % args.log_every == 0 or step == args.iterations:
                    elapsed = max(time.perf_counter() - t0, 1e-12)
                    it_s = step / elapsed
                    progress.update(
                        task_id,
                        loss=f"{last_loss:.6f}",
                        psnr=f"{last_psnr:.4f}",
                        it_s=f"{it_s:.2f}",
                    )

                if (
                    profiling_enabled
                    and profile_end_step is not None
                    and step == profile_end_step
                ):
                    profile_perf_end_ns = time.perf_counter_ns()
                    profile_wall_end_ns = time.time_ns()
        finally:
            if torch_profiler_active and torch_profiler is not None:
                torch_profiler.__exit__(None, None, None)
                torch_profiler_active = False

    if torch_profiler is not None and torch_profiler_started:
        torch_profiler_summary = torch_profiler.key_averages()

    train_sec = time.perf_counter() - t0
    export_from_best = bool(args.restore_best_at_end and best_state_dict is not None)
    if export_from_best:
        assert best_state_dict is not None
        model.load_state_dict(best_state_dict, strict=True)
    if best_step <= 0:
        best_loss = float(last_loss)
        best_step = int(args.iterations)

    model.eval()
    with torch.no_grad():
        final = model()["render"]

    final_psnr, eval_ms = run_eval(model, target, args.eval_iters)

    stem = target_path.stem
    out_img = tensor_to_pil(final)
    out_path, ckpt_path = _target_output_paths(args.output_dir, stem)
    out_img.save(out_path)

    if not args.no_save_checkpoint:
        torch.save(model.state_dict(), ckpt_path)
    else:
        ckpt_path = None

    _render_kv_table(
        console,
        "Done",
        [
            ("final_loss", f"{last_loss:.6f}"),
            ("best_loss", f"{best_loss:.6f}"),
            ("best_step", str(int(best_step))),
            ("final_step_psnr", f"{last_psnr:.4f}"),
            ("eval_psnr", f"{final_psnr:.4f}"),
            ("train_time", f"{train_sec:.3f}s"),
            ("avg_eval_time", f"{eval_ms:.3f}ms"),
            ("restore_best_at_end", str(bool(args.restore_best_at_end))),
            ("export_state", "best_loss" if export_from_best else "last"),
            ("saved_image", str(out_path)),
            ("saved_ckpt", str(ckpt_path) if ckpt_path is not None else "<disabled>"),
        ],
    )
    logger.info(
        "train.done",
        final_loss=round(last_loss, 6),
        best_loss=round(best_loss, 6),
        best_step=int(best_step),
        final_step_psnr=round(last_psnr, 4),
        eval_psnr=round(final_psnr, 4),
        train_time_sec=round(train_sec, 3),
        avg_eval_ms=round(eval_ms, 3),
        restore_best_at_end=bool(args.restore_best_at_end),
        export_state=("best_loss" if export_from_best else "last"),
        saved_image=str(out_path),
        saved_ckpt=str(ckpt_path) if ckpt_path is not None else None,
    )

    if args.trace_csv is not None:
        trace_path = args.trace_csv
        write_trace_csv(trace_path, trace_rows)
        logger.info("trace_csv.saved", path=str(trace_path), rows=len(trace_rows))

    if profiling_enabled:
        if device.type == "cuda":
            torch.cuda.synchronize()
        stats.flush_pending_cuda_events(sync=False)
        profile_samples = int(stats.metric("train.step.total", "n"))
        gpu_total_ms = stats.metric("train.step.total", "sum")
        gpu_mean_ms = stats.metric("train.step.total", "mean")

        if (
            profile_start_step is not None
            and profile_end_step is not None
            and profile_wall_start_ns is not None
            and profile_wall_end_ns is not None
            and profile_perf_start_ns is not None
            and profile_perf_end_ns is not None
            and profile_samples > 0
        ):
            wall_window_ms = (profile_wall_end_ns - profile_wall_start_ns) / 1e6
            perf_window_ms = (profile_perf_end_ns - profile_perf_start_ns) / 1e6
            sampled_step_count = len(profiled_steps)
            span_step_count = profile_end_step - profile_start_step + 1
            wall_avg_sampled_ms = wall_window_ms / max(1, sampled_step_count)
            wall_avg_span_ms = wall_window_ms / max(1, span_step_count)
            gap_ms = wall_avg_sampled_ms - gpu_mean_ms
            gap_pct = 100.0 * gap_ms / max(gpu_mean_ms, 1e-12)

            start_iso = datetime.fromtimestamp(
                profile_wall_start_ns / 1e9, tz=timezone.utc
            ).isoformat()
            end_iso = datetime.fromtimestamp(
                profile_wall_end_ns / 1e9, tz=timezone.utc
            ).isoformat()

            _render_kv_table(
                console,
                "Profile Window Audit",
                [
                    (
                        "window_steps",
                        f"{profile_start_step}..{profile_end_step} "
                        f"(sampled={sampled_step_count}, span={span_step_count})",
                    ),
                    ("start_utc", f"{start_iso}  start_ns={profile_wall_start_ns}"),
                    ("end_utc", f"{end_iso}  end_ns={profile_wall_end_ns}"),
                    ("window_wall_ms", f"{wall_window_ms:.4f}"),
                    ("window_perf_ms", f"{perf_window_ms:.4f}"),
                    ("gpu_event_total_ms", f"{gpu_total_ms:.4f}"),
                    ("gpu_event_mean_ms", f"{gpu_mean_ms:.4f}"),
                    ("wall_avg_sampled_ms", f"{wall_avg_sampled_ms:.4f}"),
                    ("wall_avg_span_ms", f"{wall_avg_span_ms:.4f}"),
                    (
                        "capture_gap_ms",
                        f"{gap_ms:+.4f} (wall_avg_sampled_ms - gpu_event_mean_ms, {gap_pct:+.2f}%)",
                    ),
                ],
            )

        all_rows = collect_timing_rows(stats, metric=args.profile_metric)
        render_hierarchy_rich(
            console,
            title="Profile Hierarchy (phase)",
            rows=all_rows,
            metric=args.profile_metric,
            prefix=None,
            global_take_label="train.step.total",
        )
        logger.info(
            "profile.summary",
            metric=args.profile_metric,
            top_rows=top_rows_for_structlog(all_rows, top_k=min(args.profile_topk, 10)),
        )

        if args.profile_overlap and overlap_steps:
            step_ms_series = stats.samples("train.step.total")
            corr_intersects = _corr(step_ms_series, overlap_num_intersects)
            corr_hits = _corr(step_ms_series, overlap_hits_per_gaussian)
            corr_area = _corr(step_ms_series, overlap_area_ratio)
            _render_kv_table(
                console,
                "Overlap Trend (profiled steps)",
                [
                    (
                        "steps",
                        f"{overlap_steps[0]}..{overlap_steps[-1]} sampled={len(overlap_steps)}",
                    ),
                    (
                        "num_points",
                        f"mean={_mean(overlap_num_points):.1f} "
                        f"min={min(overlap_num_points):.0f} max={max(overlap_num_points):.0f}",
                    ),
                    (
                        "num_intersects",
                        f"mean={_mean(overlap_num_intersects):.1f} "
                        f"min={min(overlap_num_intersects):.0f} max={max(overlap_num_intersects):.0f}",
                    ),
                    (
                        "hits_per_gaussian",
                        f"mean={_mean(overlap_hits_per_gaussian):.4f} "
                        f"min={min(overlap_hits_per_gaussian):.4f} max={max(overlap_hits_per_gaussian):.4f}",
                    ),
                    (
                        "hits_max",
                        f"mean={_mean(overlap_hits_max):.1f} "
                        f"min={min(overlap_hits_max):.0f} max={max(overlap_hits_max):.0f}",
                    ),
                    (
                        "area_ratio(sum(pi*r^2)/canvas)",
                        f"mean={_mean(overlap_area_ratio):.4f} "
                        f"min={min(overlap_area_ratio):.4f} max={max(overlap_area_ratio):.4f}",
                    ),
                    ("corr(step_ms, num_intersects)", f"{corr_intersects:.4f}"),
                    ("corr(step_ms, hits_per_gaussian)", f"{corr_hits:.4f}"),
                    ("corr(step_ms, area_ratio)", f"{corr_area:.4f}"),
                ],
            )

        if profile_contrib_enabled and contrib_steps:
            step_ms_series = stats.samples("train.step.total")
            corr_top1 = _corr(step_ms_series, contrib_top1_rate)
            corr_topk_sum = _corr(step_ms_series, contrib_topk_rate_sum)
            corr_effective = _corr(step_ms_series, contrib_effective_curves)
            _render_kv_table(
                console,
                "Primitive Contribution Trend (profiled steps)",
                [
                    (
                        "steps",
                        f"{contrib_steps[0]}..{contrib_steps[-1]} sampled={len(contrib_steps)}",
                    ),
                    ("topk_used", str(contrib_topk_used if contrib_topk_used else 0)),
                    (
                        "top1_rate",
                        f"mean={_mean(contrib_top1_rate):.6f} "
                        f"min={min(contrib_top1_rate):.6f} max={max(contrib_top1_rate):.6f}",
                    ),
                    (
                        "topk_rate_sum",
                        f"mean={_mean(contrib_topk_rate_sum):.6f} "
                        f"min={min(contrib_topk_rate_sum):.6f} max={max(contrib_topk_rate_sum):.6f}",
                    ),
                    (
                        "effective_curves",
                        f"mean={_mean(contrib_effective_curves):.2f} "
                        f"min={min(contrib_effective_curves):.2f} max={max(contrib_effective_curves):.2f}",
                    ),
                    (
                        "active_ratio",
                        f"mean={_mean(contrib_active_ratio):.6f} "
                        f"min={min(contrib_active_ratio):.6f} max={max(contrib_active_ratio):.6f}",
                    ),
                    (
                        "score_share(R/G/B)",
                        f"{_mean(contrib_r_share):.4f} / "
                        f"{_mean(contrib_g_share):.4f} / "
                        f"{_mean(contrib_b_share):.4f}",
                    ),
                    ("corr(step_ms, top1_rate)", f"{corr_top1:.4f}"),
                    ("corr(step_ms, topk_rate_sum)", f"{corr_topk_sum:.4f}"),
                    ("corr(step_ms, effective_curves)", f"{corr_effective:.4f}"),
                ],
            )

    if torch_profiler_trace_dir is not None:
        logger.info("torch_profiler.trace_saved", path=str(torch_profiler_trace_dir))
    if torch_profiler_summary is not None:
        _render_torch_profiler_summary(console, torch_profiler_summary)
    if span_trace_recorder is not None:
        span_trace_recorder.flush()
        logger.info(
            "profile_span_trace.saved",
            path=str(span_trace_recorder.path),
            fmt=span_trace_recorder.trace_format,
            steps=span_trace_recorder.num_steps,
        )

    set_gs_profile_ctx_factory(None)

    return {
        "target": str(target_path),
        "target_stem": stem,
        "eval_psnr": float(final_psnr),
        "train_time_sec": float(train_sec),
        "best_loss": float(best_loss),
        "best_step": int(best_step),
        "saved_image": str(out_path),
        "saved_ckpt": str(ckpt_path) if ckpt_path is not None else None,
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logger = configure_logger()
    console = Console()

    target_paths = resolve_target_paths(
        args.target,
        target_glob=str(args.target_glob),
        target_modulo=int(args.target_modulo),
        target_remainder=int(args.target_remainder),
    )
    if len(target_paths) > 1:
        _render_kv_table(
            console,
            "Batch Targets",
            [
                ("count", str(len(target_paths))),
                ("target_dir", str(args.target)),
                ("glob", str(args.target_glob)),
                ("modulo", str(int(args.target_modulo))),
                ("remainder", str(int(args.target_remainder))),
                ("first", str(target_paths[0])),
                ("last", str(target_paths[-1])),
            ],
        )

    batch_results: list[dict[str, Any]] = []
    total = len(target_paths)
    for idx, target_path in enumerate(target_paths, start=1):
        run_args = argparse.Namespace(**vars(args))
        if total > 1:
            if args.trace_csv is not None:
                run_args.trace_csv = _suffix_path_for_target(
                    args.trace_csv, target_path.stem
                )
            if args.profile_span_trace_path is not None:
                run_args.profile_span_trace_path = _suffix_path_for_target(
                    args.profile_span_trace_path, target_path.stem
                )
            if args.debug_visual:
                run_args.debug_dir = Path(args.debug_dir) / target_path.stem

        out_path, ckpt_path = _target_output_paths(
            run_args.output_dir, target_path.stem
        )
        if out_path.exists() and ckpt_path.exists():
            logger.info(
                "batch.target.skip_existing_outputs",
                index=idx,
                total=total,
                target=str(target_path),
                saved_image=str(out_path),
                saved_ckpt=str(ckpt_path),
            )
            batch_results.append(
                {
                    "target": str(target_path),
                    "target_stem": target_path.stem,
                    "eval_psnr": float("nan"),
                    "train_time_sec": 0.0,
                    "best_loss": float("nan"),
                    "best_step": 0,
                    "saved_image": str(out_path),
                    "saved_ckpt": str(ckpt_path),
                    "skipped": True,
                }
            )
            continue

        logger.info(
            "batch.target.start",
            index=idx,
            total=total,
            target=str(target_path),
        )
        if total > 1:
            console.print(
                Panel.fit(
                    f"[{idx}/{total}] target={target_path}",
                    title="Batch Progress",
                    border_style="magenta",
                )
            )
        batch_results.append(
            _run_single_target(
                run_args,
                target_path=target_path,
                logger=logger,
                console=console,
            )
        )

    if total > 1:
        table = Table(title="Batch Summary", box=box.SIMPLE_HEAVY)
        table.add_column("idx", justify="right", style="cyan")
        table.add_column("status", style="yellow")
        table.add_column("target", style="white")
        table.add_column("eval_psnr", justify="right")
        table.add_column("train_time", justify="right")
        table.add_column("saved_image", style="green")
        for idx, item in enumerate(batch_results, start=1):
            skipped = bool(item.get("skipped", False))
            eval_psnr_disp = "-" if skipped else f"{float(item['eval_psnr']):.4f}"
            train_time_disp = (
                "-" if skipped else f"{float(item['train_time_sec']):.3f}s"
            )
            table.add_row(
                str(idx),
                ("skipped" if skipped else "trained"),
                str(item["target"]),
                eval_psnr_disp,
                train_time_disp,
                str(item["saved_image"]),
            )
        console.print(table)


if __name__ == "__main__":
    main()
