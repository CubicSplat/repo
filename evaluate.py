from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from rich import box
from rich.console import Console
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
class EvalPair:
    pred_path: Path
    ref_path: Path
    pred_rel: Path
    ref_rel: Path


@dataclass(frozen=True)
class EvalResult:
    pred_rel: str
    ref_rel: str
    psnr: float
    ssim: float
    lpips: float | None
    eval_ms: float
    height: int
    width: int
    pred_path: str = ""
    ref_path: str = ""


@dataclass(frozen=True)
class RayEvalRun:
    summary_path: Path
    report_csv_path: Path
    pairs: tuple[EvalPair, ...]
    missing_refs: tuple[tuple[Path, Path], ...]
    missing_preds: tuple[Path, ...]


@dataclass(frozen=True)
class RayAutoScanPlan:
    run: RayEvalRun
    pairs: tuple[EvalPair, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate *_final.png against reference *.png using TileLang PSNR/SSIM."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Prediction root directory (contains *_final.png).",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Reference root directory. Default: same as --path.",
    )
    parser.add_argument(
        "--pred-pattern",
        type=str,
        default="*_final.png",
        help="Prediction filename glob pattern.",
    )
    parser.add_argument(
        "--pred-suffix",
        type=str,
        default="_final",
        help="Suffix removed from prediction stem to get reference stem.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan subdirectories.",
    )
    parser.add_argument(
        "--ref-match-mode",
        type=str,
        choices=["auto", "relative", "flat", "basename"],
        default="auto",
        help=(
            "How to map prediction file to reference file. "
            "'relative': keep relative subdir mapping; "
            "'flat': use <ref_root>/<stem>.png; "
            "'basename': find unique basename under ref_root; "
            "'auto': try relative, then flat, then basename (recommended for ray scheduler outputs)."
        ),
    )
    parser.add_argument(
        "--ray-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable ray scheduler mode. "
            "Use run_summary.json to pair outputs with references precisely."
        ),
    )
    parser.add_argument(
        "--ray-distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use Ray to evaluate pairs in parallel across multiple GPUs. "
            "Default: enabled when --ray-mode is enabled."
        ),
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="",
        help="Ray cluster address for distributed evaluation. Empty means local Ray runtime.",
    )
    parser.add_argument(
        "--ray-num-workers",
        type=int,
        default=0,
        help="Number of ray GPU workers for evaluation. <=0 means auto by available GPUs.",
    )
    parser.add_argument(
        "--ray-gpus-per-worker",
        type=float,
        default=0.5,
        help=(
            "Ray GPU resource reserved per worker actor. "
            "Smaller value allows multiple workers on one GPU to hide IO latency."
        ),
    )
    parser.add_argument(
        "--ray-cpus-per-worker",
        type=float,
        default=0.25,
        help=(
            "Ray CPU resource reserved per worker actor. "
            "Lower value helps avoid CPU resource bottleneck that leaves GPUs idle."
        ),
    )
    parser.add_argument(
        "--ray-summary",
        type=Path,
        action="append",
        default=[],
        help=(
            "Path to run_summary.json or a directory containing it. "
            "Can be repeated. If omitted in ray mode, discover run_summary.json under <path>."
        ),
    )
    parser.add_argument(
        "--ray-auto-scan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In ray mode, evaluate each discovered run_summary.json separately. "
            "If a run already has evaluation report CSV, skip it by default."
        ),
    )
    parser.add_argument(
        "--ray-eval-report-name",
        type=str,
        default="evaluation_report.csv",
        help=(
            "Report CSV filename used in ray auto-scan mode, stored next to each run_summary.json."
        ),
    )
    parser.add_argument(
        "--ray-eval-force",
        action="store_true",
        help="In ray auto-scan mode, re-evaluate even when report CSV already exists.",
    )
    parser.add_argument(
        "--allow-missing-ref",
        action="store_true",
        help="Skip predictions whose reference file is missing.",
    )
    parser.add_argument(
        "--fail-on-shape-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, skip image pairs with shape mismatch.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Evaluate only the first N matched pairs. <=0 means all.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-range", type=float, default=1.0)
    parser.add_argument("--ssim-win-size", type=int, default=11)
    parser.add_argument("--ssim-win-sigma", type=float, default=1.5)
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Enable LPIPS metric via torchmetrics (expensive).",
    )
    parser.add_argument(
        "--lpips-net-type",
        type=str,
        choices=["vgg", "alex", "squeeze"],
        default="vgg",
        help="Backbone used by LPIPS when --lpips is enabled.",
    )
    parser.add_argument("--report-csv", type=Path, default=None)
    return parser.parse_args()


def _render_kv_table(console: Console, title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


def _iter_pred_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in root.rglob(pattern) if p.is_file()]
    else:
        files = [p for p in root.glob(pattern) if p.is_file()]
    return sorted(files, key=lambda p: str(p.relative_to(root)))


def _build_ref_name_index(ref_root: Path, recursive: bool) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in _iter_pred_files(ref_root, "*.png", recursive):
        index.setdefault(path.name, []).append(path)
    return index


def _relative_to_or_none(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _find_subpath_relative(path: Path, subpath: Path) -> Path | None:
    subparts = subpath.parts
    if not subparts:
        return None
    parts = path.parts
    if len(parts) <= len(subparts):
        return None
    for i in range(0, len(parts) - len(subparts) + 1):
        if parts[i : i + len(subparts)] == subparts:
            tail = parts[i + len(subparts) :]
            if tail:
                return Path(*tail)
    return None


def _to_ref_name(pred_name: str, pred_suffix: str) -> str | None:
    stem = Path(pred_name).stem
    if not stem.endswith(pred_suffix):
        return None
    ref_stem = stem[: -len(pred_suffix)]
    if not ref_stem:
        return None
    return f"{ref_stem}.png"


def _iter_run_summaries(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"ray summary source not found: {root}")
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise ValueError(f"ray summary source must be file/dir: {root}")
    if recursive:
        files = [p for p in root.rglob("run_summary.json") if p.is_file()]
    else:
        files = [p for p in root.glob("run_summary.json") if p.is_file()]
    return sorted(files, key=lambda p: str(p.relative_to(root)))


def _resolve_run_summary_paths(
    pred_root: Path,
    *,
    sources: list[Path],
    recursive: bool,
) -> list[Path]:
    summary_paths: list[Path] = []
    raw_sources = sources if sources else [pred_root]
    for src in raw_sources:
        summary_paths.extend(_iter_run_summaries(src, recursive))

    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in summary_paths:
        resolved = p.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        uniq.append(resolved)
    if not uniq:
        roots_preview = ", ".join(str(x) for x in raw_sources[:5])
        raise FileNotFoundError(
            "no run_summary.json found for ray mode. "
            f"sources={roots_preview} recursive={recursive}"
        )
    return sorted(uniq, key=str)


def _default_ray_eval_report_csv(summary_path: Path, report_name: str) -> Path:
    name = str(report_name).strip()
    if not name:
        raise ValueError("--ray-eval-report-name must be non-empty")
    return summary_path.with_name(name)


def _resolve_ray_output_dir(
    *,
    raw_output_dir: str,
    pred_root: Path,
    summary_path: Path,
) -> Path:
    candidates: list[Path] = []
    output_dir = Path(raw_output_dir) if raw_output_dir else Path()
    if raw_output_dir:
        if output_dir.is_absolute():
            candidates.extend(
                [
                    output_dir,
                    pred_root / output_dir.name,
                    summary_path.parent / output_dir.name,
                ]
            )
        else:
            candidates.extend(
                [summary_path.parent / output_dir, pred_root / output_dir]
            )
    candidates.append(summary_path.parent)

    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.exists() and cand.is_dir():
            return cand
    return candidates[0]


def _resolve_ray_ref_path(
    *,
    target_path: Path,
    target_name: str,
    ref_root: Path,
    summary_target_dir: Path | None,
    ref_name_index: dict[str, list[Path]],
) -> Path | None:
    if target_path.exists() and target_path.is_file():
        return target_path

    candidates: list[Path] = []
    if summary_target_dir is not None and str(summary_target_dir):
        rel = _relative_to_or_none(target_path, summary_target_dir)
        if rel is None:
            rel = _find_subpath_relative(target_path, summary_target_dir)
        if rel is not None:
            candidates.append(ref_root / rel)
    if target_path.name:
        candidates.append(ref_root / target_path.name)
    if (not target_path.is_absolute()) and str(target_path) not in {"", "."}:
        candidates.append(ref_root / target_path)
    candidates.append(ref_root / target_name)

    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.exists() and cand.is_file():
            return cand

    matches = ref_name_index.get(target_name, [])
    if len(matches) == 1 and matches[0].exists() and matches[0].is_file():
        return matches[0]
    return None


def _collect_eval_pairs_from_single_ray_summary(
    *,
    summary_path: Path,
    pred_root: Path,
    ref_root: Path,
    ref_name_index: dict[str, list[Path]],
) -> tuple[list[EvalPair], list[tuple[Path, Path]], list[Path]]:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", [])
    if not isinstance(results, list):
        return [], [], []

    target_sel = payload.get("target_selection", {})
    raw_target_dir = ""
    if isinstance(target_sel, dict):
        raw_target_dir = str(target_sel.get("target_dir", "")).strip()
    summary_target_dir = Path(raw_target_dir) if raw_target_dir else None

    pairs: list[EvalPair] = []
    missing_refs: list[tuple[Path, Path]] = []
    missing_preds: list[Path] = []
    seen_pred_paths: set[Path] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status not in {"ok", "skipped"}:
            continue
        target_raw = str(item.get("target", "")).strip()
        target_path = Path(target_raw) if target_raw else Path()
        target_stem = target_path.stem
        if not target_stem:
            task_name = str(item.get("task_name", "")).strip()
            target_stem = task_name.split("_", maxsplit=1)[-1] if task_name else ""
        if not target_stem:
            continue
        target_name = f"{target_stem}.png"

        output_dir = _resolve_ray_output_dir(
            raw_output_dir=str(item.get("output_dir", "")).strip(),
            pred_root=pred_root,
            summary_path=summary_path,
        )
        pred_path = output_dir / f"{target_stem}_final.png"
        if (not pred_path.exists()) or (not pred_path.is_file()):
            missing_preds.append(pred_path)
            continue

        ref_path = _resolve_ray_ref_path(
            target_path=target_path,
            target_name=target_name,
            ref_root=ref_root,
            summary_target_dir=summary_target_dir,
            ref_name_index=ref_name_index,
        )
        if ref_path is None:
            missing_refs.append((pred_path, ref_root / target_name))
            continue

        pred_resolved = pred_path.resolve()
        if pred_resolved in seen_pred_paths:
            continue
        seen_pred_paths.add(pred_resolved)
        pred_rel = _relative_to_or_none(pred_path, pred_root) or Path(pred_path.name)
        ref_rel = _relative_to_or_none(ref_path, ref_root) or Path(ref_path.name)
        pairs.append(
            EvalPair(
                pred_path=pred_path,
                ref_path=ref_path,
                pred_rel=pred_rel,
                ref_rel=ref_rel,
            )
        )
    return pairs, missing_refs, missing_preds


def _resolve_ref_path(
    *,
    ref_root: Path,
    pred_rel: Path,
    ref_name: str,
    ref_match_mode: str,
    ref_name_index: dict[str, list[Path]] | None,
) -> Path | None:
    mode = str(ref_match_mode)
    candidates: list[Path] = []
    if mode in {"auto", "relative"}:
        candidates.append(ref_root / pred_rel.with_name(ref_name))
    if mode in {"auto", "flat"}:
        candidates.append(ref_root / ref_name)

    seen: set[Path] = set()
    for ref_path in candidates:
        if ref_path in seen:
            continue
        seen.add(ref_path)
        if ref_path.exists() and ref_path.is_file():
            return ref_path

    if mode in {"auto", "basename"} and ref_name_index is not None:
        matches = ref_name_index.get(ref_name, [])
        if len(matches) == 1:
            ref_path = matches[0]
            if ref_path.exists() and ref_path.is_file():
                return ref_path
    return None


def collect_eval_pairs_from_ray(
    pred_root: Path,
    *,
    ref_root: Path,
    recursive: bool,
    ray_summary_paths: list[Path] | None = None,
) -> tuple[list[EvalPair], list[tuple[Path, Path]], list[Path], list[Path]]:
    runs = collect_eval_pairs_from_ray_runs(
        pred_root,
        ref_root=ref_root,
        recursive=recursive,
        ray_summary_paths=ray_summary_paths,
    )
    pairs: list[EvalPair] = []
    missing_refs: list[tuple[Path, Path]] = []
    missing_preds: list[Path] = []
    summary_paths: list[Path] = []
    seen_pred_paths: set[Path] = set()
    for run in runs:
        summary_paths.append(run.summary_path)
        missing_refs.extend(list(run.missing_refs))
        missing_preds.extend(list(run.missing_preds))
        for pair in run.pairs:
            pred_resolved = pair.pred_path.resolve()
            if pred_resolved in seen_pred_paths:
                continue
            seen_pred_paths.add(pred_resolved)
            pairs.append(pair)
    return pairs, missing_refs, missing_preds, summary_paths


def collect_eval_pairs_from_ray_runs(
    pred_root: Path,
    *,
    ref_root: Path,
    recursive: bool,
    ray_summary_paths: list[Path] | None = None,
    report_name: str = "evaluation_report.csv",
) -> list[RayEvalRun]:
    summary_paths = _resolve_run_summary_paths(
        pred_root,
        sources=list(ray_summary_paths or []),
        recursive=recursive,
    )
    ref_name_index = _build_ref_name_index(ref_root, recursive=True)
    runs: list[RayEvalRun] = []
    for summary_path in summary_paths:
        pairs, missing_refs, missing_preds = (
            _collect_eval_pairs_from_single_ray_summary(
                summary_path=summary_path,
                pred_root=pred_root,
                ref_root=ref_root,
                ref_name_index=ref_name_index,
            )
        )
        runs.append(
            RayEvalRun(
                summary_path=summary_path,
                report_csv_path=_default_ray_eval_report_csv(summary_path, report_name),
                pairs=tuple(pairs),
                missing_refs=tuple(missing_refs),
                missing_preds=tuple(missing_preds),
            )
        )
    return runs


def _filter_ray_auto_scan_runs(
    runs: list[RayEvalRun], *, force: bool
) -> tuple[list[RayEvalRun], list[Path]]:
    pending: list[RayEvalRun] = []
    skipped_reports: list[Path] = []
    for run in runs:
        if (not bool(force)) and run.report_csv_path.exists():
            skipped_reports.append(run.report_csv_path)
            continue
        pending.append(run)
    return pending, skipped_reports


def _build_ray_auto_scan_plan(
    runs: list[RayEvalRun], *, max_items: int
) -> tuple[list[EvalPair], list[RayAutoScanPlan]]:
    limit = int(max_items)
    apply_limit = limit > 0
    selected_pairs: list[EvalPair] = []
    plans: list[RayAutoScanPlan] = []
    seen_pred_paths: set[Path] = set()
    stop = False
    for run in runs:
        run_pairs: list[EvalPair] = []
        for pair in run.pairs:
            if apply_limit and len(selected_pairs) >= limit:
                stop = True
                break
            pred_resolved = pair.pred_path.resolve()
            if pred_resolved in seen_pred_paths:
                continue
            seen_pred_paths.add(pred_resolved)
            selected_pairs.append(pair)
            run_pairs.append(pair)
        if run_pairs:
            plans.append(RayAutoScanPlan(run=run, pairs=tuple(run_pairs)))
        if stop:
            break
    return selected_pairs, plans


def collect_eval_pairs(
    pred_root: Path,
    *,
    ref_root: Path,
    pred_pattern: str,
    pred_suffix: str,
    recursive: bool,
    ref_match_mode: str = "auto",
) -> tuple[list[EvalPair], list[tuple[Path, Path]], list[Path]]:
    pred_files = _iter_pred_files(pred_root, pred_pattern, recursive)
    mode = str(ref_match_mode).strip().lower()
    if mode not in {"auto", "relative", "flat", "basename"}:
        raise ValueError(
            f"unsupported ref_match_mode={ref_match_mode!r}; "
            "expected one of: auto, relative, flat, basename"
        )
    need_ref_index = mode in {"auto", "basename"}
    ref_name_index = (
        _build_ref_name_index(ref_root, recursive) if need_ref_index else None
    )
    pairs: list[EvalPair] = []
    missing_refs: list[tuple[Path, Path]] = []
    invalid_pred_names: list[Path] = []

    for pred_path in pred_files:
        pred_rel = pred_path.relative_to(pred_root)
        ref_name = _to_ref_name(pred_path.name, pred_suffix)
        if ref_name is None:
            invalid_pred_names.append(pred_path)
            continue
        ref_path = _resolve_ref_path(
            ref_root=ref_root,
            pred_rel=pred_rel,
            ref_name=ref_name,
            ref_match_mode=mode,
            ref_name_index=ref_name_index,
        )
        if ref_path is None:
            missing_refs.append((pred_path, ref_root / pred_rel.with_name(ref_name)))
            continue
        ref_rel = ref_path.relative_to(ref_root)
        pairs.append(
            EvalPair(
                pred_path=pred_path,
                ref_path=ref_path,
                pred_rel=pred_rel,
                ref_rel=ref_rel,
            )
        )
    return pairs, missing_refs, invalid_pred_names


def _load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    data = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = data.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def _build_lpips_metric(device: torch.device, net_type: str) -> Any:
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(
            net_type=str(net_type),
            reduction="mean",
            normalize=True,
        ).to(device)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LPIPS requires torchvision. Install it first, e.g. `uv add torchvision`."
        ) from exc
    metric.eval()
    return metric


def evaluate_pair(
    pair: EvalPair,
    *,
    device: torch.device,
    data_range: float,
    ssim_win_size: int,
    ssim_win_sigma: float,
    lpips_metric: Any | None,
) -> EvalResult:
    # Lazy import keeps path-pairing tests lightweight and still uses manbo TileLang ops.
    from manbo.metric import psnr, ssim

    pred = _load_image_tensor(pair.pred_path, device=device)
    ref = _load_image_tensor(pair.ref_path, device=device)
    if pred.shape != ref.shape:
        raise ValueError(
            f"shape mismatch for pred={pair.pred_path} and ref={pair.ref_path}: "
            f"{tuple(pred.shape)} vs {tuple(ref.shape)}"
        )

    t0 = time.perf_counter()
    with torch.no_grad():
        psnr_tensor = psnr(pred, ref, data_range=data_range)
        ssim_tensor = ssim(
            pred,
            ref,
            data_range=data_range,
            size_average=True,
            win_size=ssim_win_size,
            win_sigma=ssim_win_sigma,
        )
        lpips_tensor: torch.Tensor | None = None
        if lpips_metric is not None:
            lpips_tensor = lpips_metric(pred, ref)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_ms = (time.perf_counter() - t0) * 1000.0

    psnr_value = float(psnr_tensor.item())
    ssim_value = float(ssim_tensor.item())
    lpips_value: float | None = None
    if lpips_tensor is not None:
        lpips_value = float(lpips_tensor.item())
        lpips_metric.reset()

    _, _, h, w = pred.shape
    return EvalResult(
        pred_rel=str(pair.pred_rel),
        ref_rel=str(pair.ref_rel),
        psnr=psnr_value,
        ssim=ssim_value,
        lpips=lpips_value,
        eval_ms=eval_ms,
        height=int(h),
        width=int(w),
        pred_path=str(pair.pred_path.resolve()),
        ref_path=str(pair.ref_path.resolve()),
    )


def _write_report_csv(path: Path, rows: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pred_rel",
                "ref_rel",
                "psnr",
                "ssim",
                "lpips",
                "eval_ms",
                "height",
                "width",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.pred_rel,
                    row.ref_rel,
                    f"{row.psnr:.6f}",
                    f"{row.ssim:.6f}",
                    (f"{float(row.lpips):.6f}" if row.lpips is not None else ""),
                    f"{row.eval_ms:.6f}",
                    row.height,
                    row.width,
                ]
            )


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _render_results(console: Console, rows: list[EvalResult]) -> None:
    psnr_vals = [r.psnr for r in rows]
    ssim_vals = [r.ssim for r in rows]
    lpips_vals = [float(r.lpips) for r in rows if r.lpips is not None]
    eval_vals = [r.eval_ms for r in rows]

    summary_rows = [
        ("count", str(len(rows))),
        ("psnr_mean", f"{_mean(psnr_vals):.6f}"),
        ("psnr_median", f"{statistics.median(psnr_vals):.6f}"),
        ("psnr_min", f"{min(psnr_vals):.6f}"),
        ("psnr_max", f"{max(psnr_vals):.6f}"),
        ("ssim_mean", f"{_mean(ssim_vals):.6f}"),
        ("ssim_median", f"{statistics.median(ssim_vals):.6f}"),
        ("ssim_min", f"{min(ssim_vals):.6f}"),
        ("ssim_max", f"{max(ssim_vals):.6f}"),
    ]
    if lpips_vals:
        summary_rows.extend(
            [
                ("lpips_mean", f"{_mean(lpips_vals):.6f}"),
                ("lpips_median", f"{statistics.median(lpips_vals):.6f}"),
                ("lpips_min", f"{min(lpips_vals):.6f}"),
                ("lpips_max", f"{max(lpips_vals):.6f}"),
            ]
        )
    summary_rows.extend(
        [
            ("eval_ms_total", f"{sum(eval_vals):.3f}"),
            ("eval_ms_mean", f"{_mean(eval_vals):.3f}"),
        ]
    )
    _render_kv_table(console, "Evaluation Summary", summary_rows)

    detail = Table(title="Per-Image Metrics", box=box.SIMPLE)
    detail.add_column("idx", justify="right", style="cyan")
    detail.add_column("pred_rel", style="white")
    detail.add_column("ref_rel", style="white")
    detail.add_column("shape", justify="right")
    detail.add_column("psnr", justify="right")
    detail.add_column("ssim", justify="right")
    detail.add_column("lpips", justify="right")
    detail.add_column("eval_ms", justify="right")
    for idx, row in enumerate(rows, start=1):
        detail.add_row(
            str(idx),
            row.pred_rel,
            row.ref_rel,
            f"{row.height}x{row.width}",
            f"{row.psnr:.6f}",
            f"{row.ssim:.6f}",
            f"{float(row.lpips):.6f}" if row.lpips is not None else "--",
            f"{row.eval_ms:.3f}",
        )
    console.print(detail)


def _evaluate_pairs_local(
    *,
    pairs: list[EvalPair],
    device: torch.device,
    data_range: float,
    ssim_win_size: int,
    ssim_win_sigma: float,
    lpips_metric: Any | None,
    fail_on_shape_mismatch: bool,
    console: Console,
) -> tuple[list[EvalResult], int]:
    rows: list[EvalResult] = []
    skipped_shape_mismatch = 0
    progress = Progress(
        TextColumn("[bold cyan]Evaluating[/bold cyan]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("psnr {task.fields[psnr]}"),
        TextColumn("ssim {task.fields[ssim]}"),
        TextColumn("lpips {task.fields[lpips]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    with progress:
        task_id = progress.add_task(
            "eval",
            total=len(pairs),
            psnr="--",
            ssim="--",
            lpips="--",
        )
        for pair in pairs:
            try:
                row = evaluate_pair(
                    pair,
                    device=device,
                    data_range=float(data_range),
                    ssim_win_size=int(ssim_win_size),
                    ssim_win_sigma=float(ssim_win_sigma),
                    lpips_metric=lpips_metric,
                )
                rows.append(row)
                progress.update(
                    task_id,
                    psnr=f"{row.psnr:.4f}",
                    ssim=f"{row.ssim:.4f}",
                    lpips=(
                        f"{float(row.lpips):.4f}" if row.lpips is not None else "--"
                    ),
                )
            except ValueError as exc:
                if bool(fail_on_shape_mismatch):
                    raise
                skipped_shape_mismatch += 1
                console.print(f"[yellow]Warning:[/yellow] {exc}")
            progress.advance(task_id, 1)
    return rows, int(skipped_shape_mismatch)


def _evaluate_pairs_ray(
    *,
    pairs: list[EvalPair],
    data_range: float,
    ssim_win_size: int,
    ssim_win_sigma: float,
    lpips: bool,
    lpips_net_type: str,
    fail_on_shape_mismatch: bool,
    ray_address: str,
    ray_num_workers: int,
    ray_gpus_per_worker: float,
    ray_cpus_per_worker: float,
    console: Console,
) -> tuple[list[EvalResult], int]:
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ray distributed evaluation requires `ray`. Install it first, e.g. `uv add ray`."
        ) from exc

    if not ray.is_initialized():
        if str(ray_address).strip():
            ray.init(address=str(ray_address).strip(), ignore_reinit_error=True)
        else:
            ray.init(ignore_reinit_error=True)

    cluster_gpu = float(ray.cluster_resources().get("GPU", 0.0))
    if cluster_gpu <= 0.0:
        raise RuntimeError(
            "Ray distributed evaluation requested but no GPU resource found in ray cluster."
        )
    gpu_per_worker = float(ray_gpus_per_worker)
    if gpu_per_worker <= 0.0:
        raise ValueError("--ray-gpus-per-worker must be > 0")
    cpu_per_worker = float(ray_cpus_per_worker)
    if cpu_per_worker < 0.0:
        raise ValueError("--ray-cpus-per-worker must be >= 0")

    max_workers_by_gpu = int(cluster_gpu / gpu_per_worker)
    if max_workers_by_gpu <= 0:
        raise RuntimeError(
            "ray worker resource is too large for current cluster: "
            f"cluster_gpu={cluster_gpu:.3f}, ray_gpus_per_worker={gpu_per_worker:.3f}"
        )
    worker_count = (
        int(ray_num_workers) if int(ray_num_workers) > 0 else int(max_workers_by_gpu)
    )
    worker_count = max(
        1,
        min(int(worker_count), int(max_workers_by_gpu), len(pairs)),
    )
    if len(pairs) < int(max_workers_by_gpu):
        console.print(
            "[yellow]Ray note:[/yellow] "
            f"pairs={len(pairs)} < max_workers_by_gpu={max_workers_by_gpu}; "
            "some GPUs may stay idle in this run."
        )
    console.print(
        "[cyan]Ray workers:[/cyan] "
        f"cluster_gpu={cluster_gpu:.3f}, "
        f"gpus/worker={gpu_per_worker:.3f}, "
        f"cpus/worker={cpu_per_worker:.3f}, "
        f"workers={worker_count}"
    )

    @ray.remote(
        num_gpus=float(gpu_per_worker),
        num_cpus=float(cpu_per_worker),
        max_restarts=0,
    )
    class _EvalWorker:
        def __init__(self, *, enable_lpips: bool, lpips_net: str) -> None:
            self.device = torch.device("cuda")
            self.lpips_metric = (
                _build_lpips_metric(self.device, lpips_net) if enable_lpips else None
            )

        def evaluate_batch(
            self,
            payloads: list[dict[str, str]],
            *,
            data_range: float,
            ssim_win_size: int,
            ssim_win_sigma: float,
            fail_on_shape_mismatch: bool,
        ) -> dict[str, Any]:
            rows: list[dict[str, Any]] = []
            skipped_shape_mismatch = 0
            for payload in payloads:
                pair = EvalPair(
                    pred_path=Path(str(payload["pred_path"])),
                    ref_path=Path(str(payload["ref_path"])),
                    pred_rel=Path(str(payload["pred_rel"])),
                    ref_rel=Path(str(payload["ref_rel"])),
                )
                try:
                    row = evaluate_pair(
                        pair,
                        device=self.device,
                        data_range=float(data_range),
                        ssim_win_size=int(ssim_win_size),
                        ssim_win_sigma=float(ssim_win_sigma),
                        lpips_metric=self.lpips_metric,
                    )
                    rows.append(
                        {
                            "pred_rel": row.pred_rel,
                            "ref_rel": row.ref_rel,
                            "psnr": float(row.psnr),
                            "ssim": float(row.ssim),
                            "lpips": (None if row.lpips is None else float(row.lpips)),
                            "eval_ms": float(row.eval_ms),
                            "height": int(row.height),
                            "width": int(row.width),
                            "pred_path": str(row.pred_path),
                            "ref_path": str(row.ref_path),
                        }
                    )
                except ValueError:
                    if bool(fail_on_shape_mismatch):
                        raise
                    skipped_shape_mismatch += 1
            return {
                "rows": rows,
                "skipped_shape_mismatch": int(skipped_shape_mismatch),
                "processed": int(len(payloads)),
            }

    workers = [
        _EvalWorker.remote(enable_lpips=bool(lpips), lpips_net=str(lpips_net_type))
        for _ in range(worker_count)
    ]
    payloads = [
        {
            "pred_path": str(pair.pred_path),
            "ref_path": str(pair.ref_path),
            "pred_rel": str(pair.pred_rel),
            "ref_rel": str(pair.ref_rel),
        }
        for pair in pairs
    ]
    # Keep each GPU fed with smaller chunks; this avoids long-tail idle workers.
    target_batches_per_worker = 8
    chunk_size = max(
        1,
        len(payloads) // max(1, worker_count * int(target_batches_per_worker)),
    )
    chunk_size = min(chunk_size, 32)
    chunks = [
        payloads[i : i + chunk_size] for i in range(0, len(payloads), int(chunk_size))
    ]

    next_chunk_idx = 0
    inflight: dict[Any, tuple[int, int]] = {}

    def _submit_next_chunk(worker_idx: int) -> bool:
        nonlocal next_chunk_idx
        if next_chunk_idx >= len(chunks):
            return False
        chunk = chunks[next_chunk_idx]
        next_chunk_idx += 1
        ref = workers[worker_idx].evaluate_batch.remote(
            chunk,
            data_range=float(data_range),
            ssim_win_size=int(ssim_win_size),
            ssim_win_sigma=float(ssim_win_sigma),
            fail_on_shape_mismatch=bool(fail_on_shape_mismatch),
        )
        inflight[ref] = (int(worker_idx), int(len(chunk)))
        return True

    for wi in range(worker_count):
        submitted = _submit_next_chunk(wi)
        if not submitted:
            break

    rows: list[EvalResult] = []
    skipped_shape_mismatch = 0
    progress = Progress(
        TextColumn("[bold cyan]Evaluating (Ray)[/bold cyan]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    with progress:
        task_id = progress.add_task("eval_ray", total=len(pairs))
        while inflight:
            ready, _ = ray.wait(list(inflight.keys()), num_returns=1)
            ref = ready[0]
            worker_idx, batch_size = inflight.pop(ref)
            result = ray.get(ref)
            progress.advance(task_id, batch_size)
            for item in result.get("rows", []):
                rows.append(
                    EvalResult(
                        pred_rel=str(item["pred_rel"]),
                        ref_rel=str(item["ref_rel"]),
                        psnr=float(item["psnr"]),
                        ssim=float(item["ssim"]),
                        lpips=(
                            None
                            if item.get("lpips", None) is None
                            else float(item["lpips"])
                        ),
                        eval_ms=float(item["eval_ms"]),
                        height=int(item["height"]),
                        width=int(item["width"]),
                        pred_path=str(item.get("pred_path", "")),
                        ref_path=str(item.get("ref_path", "")),
                    )
                )
            skipped_shape_mismatch += int(result.get("skipped_shape_mismatch", 0))
            _submit_next_chunk(int(worker_idx))
    rows = sorted(rows, key=lambda x: x.pred_rel)
    return rows, int(skipped_shape_mismatch)


def _write_ray_auto_scan_reports(
    *,
    plans: list[RayAutoScanPlan],
    rows: list[EvalResult],
) -> int:
    rows_by_pred_path: dict[Path, EvalResult] = {}
    for row in rows:
        raw_pred_path = str(row.pred_path).strip()
        if not raw_pred_path:
            continue
        key = Path(raw_pred_path).resolve()
        rows_by_pred_path[key] = row

    written_reports = 0
    for plan in plans:
        run_rows = []
        for pair in plan.pairs:
            row = rows_by_pred_path.get(pair.pred_path.resolve())
            if row is not None:
                run_rows.append(row)
        if not run_rows:
            continue
        _write_report_csv(plan.run.report_csv_path, run_rows)
        written_reports += 1
    return int(written_reports)


def _default_ray_auto_scan_summary_csv(pred_root: Path, report_name: str) -> Path:
    name = str(report_name).strip()
    stem = Path(name).stem if name else "evaluation_report"
    if not stem:
        stem = "evaluation_report"
    return pred_root / f"{stem}_summary.csv"


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _parse_float_or_none(raw: str) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int_or_none(raw: Any) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _collect_report_stats(report_csv_path: Path) -> dict[str, float]:
    count = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    lpips_count = 0
    with report_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            psnr_value = _parse_float_or_none(str(row.get("psnr", "")))
            ssim_value = _parse_float_or_none(str(row.get("ssim", "")))
            if psnr_value is None or ssim_value is None:
                continue
            count += 1
            psnr_sum += float(psnr_value)
            ssim_sum += float(ssim_value)
            lpips_value = _parse_float_or_none(str(row.get("lpips", "")))
            if lpips_value is not None:
                lpips_sum += float(lpips_value)
                lpips_count += 1
    if count <= 0:
        return {
            "count": 0.0,
            "psnr_mean": 0.0,
            "ssim_mean": 0.0,
            "lpips_mean": 0.0,
            "lpips_count": 0.0,
        }
    return {
        "count": float(count),
        "psnr_mean": float(psnr_sum / count),
        "ssim_mean": float(ssim_sum / count),
        "lpips_mean": (float(lpips_sum / lpips_count) if lpips_count > 0 else 0.0),
        "lpips_count": float(lpips_count),
    }


def _load_run_summary_payload(summary_path: Path) -> dict[str, Any]:
    if (not summary_path.exists()) or (not summary_path.is_file()):
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_ray_auto_scan_summary_report(
    *,
    pred_root: Path,
    runs: list[RayEvalRun],
    report_name: str,
) -> tuple[Path, int, int]:
    summary_csv_path = _default_ray_auto_scan_summary_csv(pred_root, report_name)
    run_rows: list[dict[str, str]] = []
    total_eval_count = 0
    included_run_count = 0

    for run in runs:
        report_csv_path = run.report_csv_path
        if (not report_csv_path.exists()) or (not report_csv_path.is_file()):
            continue
        stats = _collect_report_stats(report_csv_path)
        eval_count = int(stats["count"])
        if eval_count <= 0:
            continue
        included_run_count += 1
        total_eval_count += eval_count
        lpips_count = int(stats["lpips_count"])
        payload = _load_run_summary_payload(run.summary_path)
        counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
        timing = payload.get("timing", {}) if isinstance(payload, dict) else {}
        ray_info = payload.get("ray", {}) if isinstance(payload, dict) else {}
        if not isinstance(counts, dict):
            counts = {}
        if not isinstance(timing, dict):
            timing = {}
        if not isinstance(ray_info, dict):
            ray_info = {}
        task_count = _parse_int_or_none(counts.get("total"))
        if task_count is None or task_count <= 0:
            task_count = _parse_int_or_none(counts.get("started"))
        if task_count is None or task_count <= 0:
            task_count = int(eval_count)
        wall_elapsed_sec = _parse_float_or_none(str(timing.get("wall_elapsed_sec", "")))
        cluster_gpu = _parse_int_or_none(ray_info.get("cluster_gpu"))
        throughput_tpm: float | None = None
        if (
            wall_elapsed_sec is not None
            and wall_elapsed_sec > 0.0
            and cluster_gpu is not None
            and cluster_gpu > 0
            and task_count > 0
        ):
            throughput_tpm = (
                float(task_count)
                * 60.0
                / (float(wall_elapsed_sec) * float(cluster_gpu))
            )

        run_dir = run.summary_path.parent
        run_rel = _relative_to_or_none(run_dir, pred_root)
        if run_rel is None or str(run_rel) in {"", "."}:
            run_name = str(run_dir.name or run_dir)
        else:
            run_name = str(run_rel.as_posix())
        run_rows.append(
            {
                "scope": "run",
                "run_name": run_name,
                "eval_count": str(eval_count),
                "throughput_tpm": _format_metric(throughput_tpm),
                "psnr_mean": _format_metric(float(stats["psnr_mean"])),
                "ssim_mean": _format_metric(float(stats["ssim_mean"])),
                "lpips_mean": _format_metric(
                    float(stats["lpips_mean"]) if lpips_count > 0 else None
                ),
            }
        )

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "run_name",
                "eval_count",
                "throughput_tpm",
                "psnr_mean",
                "ssim_mean",
                "lpips_mean",
            ],
        )
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)
    return summary_csv_path, int(included_run_count), int(total_eval_count)


def main() -> None:
    args = parse_args()
    console = Console()

    ray_mode = bool(args.ray_mode or len(args.ray_summary) > 0)
    if args.ray_distributed is None:
        ray_distributed = bool(ray_mode)
    else:
        ray_distributed = bool(args.ray_distributed)
    pred_root = Path(args.path)
    if not pred_root.exists() or not pred_root.is_dir():
        raise FileNotFoundError(f"prediction directory not found: {pred_root}")
    ref_root = Path(args.ref_dir) if args.ref_dir is not None else pred_root
    if not ref_root.exists() or not ref_root.is_dir():
        raise FileNotFoundError(f"reference directory not found: {ref_root}")

    if not args.pred_suffix:
        raise ValueError("--pred-suffix must be non-empty")
    if args.data_range <= 0.0:
        raise ValueError("--data-range must be > 0")
    if args.lpips and args.device.strip().lower() != "cuda":
        console.print(
            "[yellow]Warning:[/yellow] LPIPS is enabled on non-CUDA device; this may be slow."
        )
    if ray_distributed and args.device.strip().lower() != "cuda":
        raise ValueError("Ray distributed evaluation requires --device cuda")

    device = torch.device(args.device)
    if (
        device.type == "cuda"
        and (not ray_distributed)
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA requested but unavailable")

    ray_summary_count = 0
    ray_auto_skipped_reports: list[Path] = []
    ray_auto_plan: list[RayAutoScanPlan] = []
    ray_auto_runs: list[RayEvalRun] = []
    invalid_pred_names: list[Path] = []
    missing_refs: list[tuple[Path, Path]] = []
    missing_preds: list[Path] = []
    if ray_mode:
        if bool(args.ray_auto_scan):
            runs = collect_eval_pairs_from_ray_runs(
                pred_root,
                ref_root=ref_root,
                recursive=bool(args.recursive),
                ray_summary_paths=list(args.ray_summary),
                report_name=str(args.ray_eval_report_name),
            )
            ray_auto_runs = list(runs)
            ray_summary_count = len(runs)
            pending_runs, ray_auto_skipped_reports = _filter_ray_auto_scan_runs(
                runs,
                force=bool(args.ray_eval_force),
            )
            for run in pending_runs:
                missing_refs.extend(list(run.missing_refs))
                missing_preds.extend(list(run.missing_preds))
            pairs, ray_auto_plan = _build_ray_auto_scan_plan(
                pending_runs,
                max_items=int(args.max_items),
            )
        else:
            pairs, missing_refs, missing_preds, summary_paths = (
                collect_eval_pairs_from_ray(
                    pred_root,
                    ref_root=ref_root,
                    recursive=bool(args.recursive),
                    ray_summary_paths=list(args.ray_summary),
                )
            )
            ray_summary_count = len(summary_paths)
    else:
        pairs, missing_refs, invalid_pred_names = collect_eval_pairs(
            pred_root,
            ref_root=ref_root,
            pred_pattern=str(args.pred_pattern),
            pred_suffix=str(args.pred_suffix),
            recursive=bool(args.recursive),
            ref_match_mode=str(args.ref_match_mode),
        )

    if invalid_pred_names:
        preview = ", ".join(
            str(p.relative_to(pred_root)) for p in invalid_pred_names[:5]
        )
        console.print(
            f"[yellow]Warning:[/yellow] {len(invalid_pred_names)} prediction files do not match pred-suffix={args.pred_suffix!r}; examples: {preview}"
        )

    if missing_refs:
        preview = "; ".join(
            f"{pred.relative_to(pred_root)} -> {ref.relative_to(ref_root) if ref.is_relative_to(ref_root) else ref}"
            for pred, ref in missing_refs[:5]
        )
        if args.allow_missing_ref:
            console.print(
                f"[yellow]Warning:[/yellow] {len(missing_refs)} prediction files have no reference; skipped. examples: {preview}"
            )
        else:
            raise FileNotFoundError(
                f"{len(missing_refs)} prediction files have no reference. "
                f"Use --allow-missing-ref to skip. examples: {preview}"
            )

    if missing_preds:
        preview = "; ".join(str(p) for p in missing_preds[:5])
        console.print(
            f"[yellow]Warning:[/yellow] {len(missing_preds)} ray tasks have no prediction file; skipped. examples: {preview}"
        )

    if (not bool(args.ray_auto_scan)) and args.max_items and args.max_items > 0:
        pairs = pairs[: int(args.max_items)]

    if not pairs:
        if bool(args.ray_auto_scan) and len(ray_auto_skipped_reports) > 0:
            summary_csv_path, included_run_count, total_count = (
                _write_ray_auto_scan_summary_report(
                    pred_root=pred_root,
                    runs=ray_auto_runs,
                    report_name=str(args.ray_eval_report_name),
                )
            )
            console.print(
                "[green]Auto-scan summary saved:[/green] "
                f"{summary_csv_path} (runs={included_run_count}, images={total_count})"
            )
            console.print(
                f"[green]No pending runs.[/green] {len(ray_auto_skipped_reports)} runs already have {args.ray_eval_report_name}."
            )
            return
        raise RuntimeError("no valid prediction-reference pairs found")

    _render_kv_table(
        console,
        "Evaluation Config",
        [
            ("pred_root", str(pred_root)),
            ("ref_root", str(ref_root)),
            ("ray_mode", str(ray_mode).lower()),
            ("ray_distributed", str(bool(ray_distributed)).lower()),
            ("ray_auto_scan", str(bool(args.ray_auto_scan)).lower()),
            ("ray_summaries", str(ray_summary_count)),
            ("ray_auto_skipped", str(len(ray_auto_skipped_reports))),
            ("ray_eval_force", str(bool(args.ray_eval_force)).lower()),
            ("ray_eval_report_name", str(args.ray_eval_report_name)),
            ("pred_pattern", str(args.pred_pattern)),
            ("pred_suffix", str(args.pred_suffix)),
            ("ref_match_mode", str(args.ref_match_mode)),
            ("recursive", str(bool(args.recursive)).lower()),
            ("pairs", str(len(pairs))),
            ("device", str(device)),
            ("ray_address", str(args.ray_address)),
            ("ray_num_workers", str(int(args.ray_num_workers))),
            ("ray_gpus_per_worker", f"{float(args.ray_gpus_per_worker):.3f}"),
            ("ray_cpus_per_worker", f"{float(args.ray_cpus_per_worker):.3f}"),
            ("ssim_win_size", str(int(args.ssim_win_size))),
            ("ssim_win_sigma", f"{float(args.ssim_win_sigma):.3f}"),
            ("lpips", str(bool(args.lpips)).lower()),
            ("lpips_net_type", str(args.lpips_net_type)),
        ],
    )

    if ray_distributed:
        rows, skipped_shape_mismatch = _evaluate_pairs_ray(
            pairs=pairs,
            data_range=float(args.data_range),
            ssim_win_size=int(args.ssim_win_size),
            ssim_win_sigma=float(args.ssim_win_sigma),
            lpips=bool(args.lpips),
            lpips_net_type=str(args.lpips_net_type),
            fail_on_shape_mismatch=bool(args.fail_on_shape_mismatch),
            ray_address=str(args.ray_address),
            ray_num_workers=int(args.ray_num_workers),
            ray_gpus_per_worker=float(args.ray_gpus_per_worker),
            ray_cpus_per_worker=float(args.ray_cpus_per_worker),
            console=console,
        )
    else:
        lpips_metric = (
            _build_lpips_metric(device, str(args.lpips_net_type))
            if args.lpips
            else None
        )
        rows, skipped_shape_mismatch = _evaluate_pairs_local(
            pairs=pairs,
            device=device,
            data_range=float(args.data_range),
            ssim_win_size=int(args.ssim_win_size),
            ssim_win_sigma=float(args.ssim_win_sigma),
            lpips_metric=lpips_metric,
            fail_on_shape_mismatch=bool(args.fail_on_shape_mismatch),
            console=console,
        )

    if not rows:
        raise RuntimeError(
            "all pairs were skipped; no evaluation results were produced"
        )

    if skipped_shape_mismatch > 0:
        console.print(
            f"[yellow]Warning:[/yellow] skipped {skipped_shape_mismatch} pairs due to shape mismatch"
        )

    _render_results(console, rows)
    if bool(args.ray_auto_scan):
        written_reports = _write_ray_auto_scan_reports(
            plans=ray_auto_plan,
            rows=rows,
        )
        if written_reports > 0:
            console.print(
                f"[green]Auto-scan reports saved:[/green] {written_reports} runs -> {args.ray_eval_report_name}"
            )
        summary_csv_path, included_run_count, total_count = (
            _write_ray_auto_scan_summary_report(
                pred_root=pred_root,
                runs=ray_auto_runs,
                report_name=str(args.ray_eval_report_name),
            )
        )
        console.print(
            "[green]Auto-scan summary saved:[/green] "
            f"{summary_csv_path} (runs={included_run_count}, images={total_count})"
        )
    if args.report_csv is not None:
        _write_report_csv(Path(args.report_csv), rows)
        console.print(f"[green]CSV saved:[/green] {args.report_csv}")


if __name__ == "__main__":
    main()
