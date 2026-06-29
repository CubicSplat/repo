from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from rich import box
from rich.console import Console
from rich.table import Table

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from manbo import psnr  # noqa: E402
from train import build_model, set_seed  # noqa: E402


@dataclass(frozen=True)
class CheckpointMeta:
    num_curves: int
    num_control_points: int
    base_num_samples: int
    mode: str
    bezier_degree: int


@dataclass(frozen=True)
class EvalRow:
    scale: int
    sample_count: int
    height: int
    width: int
    compare_height: int
    compare_width: int
    psnr: float
    render_ms: float
    missing_keys: int
    unexpected_keys: int
    skipped_shape_keys: int
    export_render_path: str


@dataclass(frozen=True)
class BackendProbeRow:
    backend: str
    psnr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained checkpoint and evaluate PSNR across magnification "
            "factors and sampling densities."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./output/scale_psnr"))
    parser.add_argument(
        "--scales",
        type=str,
        default="1,2,4",
        help="Comma-separated integer magnification factors.",
    )
    parser.add_argument(
        "--sample-counts",
        type=str,
        default="",
        help=(
            "Comma-separated absolute sample counts. "
            "If empty, sample counts are derived from --sample-ratios."
        ),
    )
    parser.add_argument(
        "--sample-ratios",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated ratios multiplied by checkpoint base sample count.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "closed", "unclosed"],
        default="auto",
    )
    parser.add_argument("--bezier-degree", type=int, default=0)
    parser.add_argument(
        "--renderer-backend",
        type=str,
        choices=["auto", "gaussian", "cubic", "cubic_fill"],
        default="auto",
    )
    parser.add_argument("--block-h", type=int, default=16)
    parser.add_argument("--block-w", type=int, default=16)
    parser.add_argument(
        "--target-resize-mode",
        type=str,
        choices=["nearest", "bilinear", "bicubic"],
        default="bicubic",
        help="Interpolation mode used to resize GT for scale>1.",
    )
    parser.add_argument(
        "--psnr-compare-space",
        type=str,
        choices=["base_target", "scaled_target"],
        default="base_target",
        help=(
            "PSNR comparison space. "
            "'base_target': downsample render back to base resolution and compare to original GT. "
            "'scaled_target': compare render to upscaled GT at current scale."
        ),
    )
    parser.add_argument(
        "--pred-downsample-mode",
        type=str,
        choices=["area", "bilinear", "bicubic"],
        default="area",
        help=(
            "Downsample mode for render->base conversion when "
            "--psnr-compare-space=base_target and scale>1."
        ),
    )
    parser.add_argument(
        "--denser-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pass denser_sample to model.forward; currently only affects parts "
            "of closed-mode gaussian sampling."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cubic-distance-samples-train-base", type=int, default=12)
    parser.add_argument("--cubic-distance-samples-eval-base", type=int, default=18)
    parser.add_argument(
        "--export-renders",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export rendered images for each evaluated configuration to the "
            "same directory as --target with auto-generated filenames."
        ),
    )
    parser.add_argument("--csv-name", type=str, default="scale_sampling_psnr.csv")
    parser.add_argument("--json-name", type=str, default="scale_sampling_psnr.json")
    return parser.parse_args()


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    vals: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"{name} expects positive integers, got {value}")
        vals.append(value)
    if not vals:
        raise ValueError(f"{name} cannot be empty")
    return vals


def _parse_float_list(raw: str, *, name: str) -> list[float]:
    vals: list[float] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError(f"{name} expects positive floats, got {value}")
        vals.append(value)
    if not vals:
        raise ValueError(f"{name} cannot be empty")
    return vals


def _unique_preserve_order(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _resolve_mode(mode: str, num_control_points: int, hint_path: Path) -> str:
    mode_norm = str(mode).lower()
    if mode_norm != "auto":
        return mode_norm

    candidates: list[str] = []
    if num_control_points % 2 == 0:
        candidates.append("closed")
    if (num_control_points - 1) % 3 == 0:
        candidates.append("unclosed")
    candidates = sorted(set(candidates))
    if not candidates:
        raise ValueError(
            "cannot infer mode from control-point count: "
            f"num_control_points={num_control_points}"
        )
    if len(candidates) == 1:
        return candidates[0]

    hint = str(hint_path).lower()
    for hinted in ("unclosed", "closed"):
        if hinted in hint and hinted in candidates:
            return hinted
    if "closed" in candidates:
        return "closed"
    return candidates[0]


def _infer_bezier_degree(mode: str, num_control_points: int) -> int:
    mode_norm = str(mode).lower()
    if mode_norm == "closed":
        if num_control_points % 2 != 0:
            raise ValueError(
                f"closed mode expects even control-point count, got {num_control_points}"
            )
        return num_control_points // 2 - 1
    if mode_norm == "unclosed":
        if (num_control_points - 1) % 3 != 0:
            raise ValueError(
                f"unclosed mode expects (cp-1) % 3 == 0, got cp={num_control_points}"
            )
        return (num_control_points - 1) // 3
    raise ValueError(f"unsupported mode: {mode}")


def _read_checkpoint_meta(
    state: dict[str, torch.Tensor],
    *,
    mode_arg: str,
    bezier_degree_arg: int,
    hint_path: Path,
) -> CheckpointMeta:
    control_points = state.get("_control_points")
    xyz = state.get("_xyz")
    if not isinstance(control_points, torch.Tensor):
        raise KeyError("checkpoint missing tensor key '_control_points'")
    if control_points.ndim != 3 or int(control_points.shape[2]) != 2:
        raise ValueError(
            f"_control_points must be [N,K,2], got {tuple(control_points.shape)}"
        )
    if not isinstance(xyz, torch.Tensor):
        raise KeyError("checkpoint missing tensor key '_xyz'")
    if xyz.ndim != 3 or int(xyz.shape[2]) != 2:
        raise ValueError(f"_xyz must be [N,S,2], got {tuple(xyz.shape)}")

    num_curves = int(control_points.shape[0])
    num_control_points = int(control_points.shape[1])
    base_num_samples = int(xyz.shape[1])
    mode = _resolve_mode(str(mode_arg), num_control_points, hint_path)
    bezier_degree = (
        int(bezier_degree_arg)
        if int(bezier_degree_arg) > 0
        else _infer_bezier_degree(mode, num_control_points)
    )
    return CheckpointMeta(
        num_curves=num_curves,
        num_control_points=num_control_points,
        base_num_samples=base_num_samples,
        mode=mode,
        bezier_degree=bezier_degree,
    )


def _resolve_sweep_values(
    *,
    selected_backend: str,
    sample_counts_raw: str,
    sample_ratios_raw: str,
    base_num_samples: int,
    base_cubic_distance_samples_eval: int,
) -> tuple[str, list[int]]:
    backend = str(selected_backend).strip().lower()
    sweep_key = "num_samples" if backend == "gaussian" else "distance_samples"
    if str(sample_counts_raw).strip():
        counts = _parse_int_list(sample_counts_raw, name="--sample-counts")
        return sweep_key, _unique_preserve_order(counts)

    ratios = _parse_float_list(sample_ratios_raw, name="--sample-ratios")
    base_value = (
        int(base_num_samples)
        if sweep_key == "num_samples"
        else int(base_cubic_distance_samples_eval)
    )
    counts = [max(2, int(round(float(base_value) * r))) for r in ratios]
    return sweep_key, _unique_preserve_order(counts)


def _load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"target image not found: {path}")
    image = Image.open(path).convert("RGB")
    w, h = image.size
    data = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = data.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def _scaled_target(
    target_base: torch.Tensor, *, scale: int, resize_mode: str
) -> torch.Tensor:
    if int(scale) == 1:
        return target_base
    h = int(target_base.shape[-2]) * int(scale)
    w = int(target_base.shape[-1]) * int(scale)
    kwargs: dict[str, object] = {}
    if resize_mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(target_base, size=(h, w), mode=resize_mode, **kwargs)


def _resize_image_tensor(
    image: torch.Tensor,
    *,
    size_hw: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if tuple(image.shape[-2:]) == tuple(size_hw):
        return image
    kwargs: dict[str, object] = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    return F.interpolate(image, size=size_hw, mode=mode, **kwargs)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.dim() != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"expected image [1, 3, H, W], got {tuple(image.shape)}")
    chw = image.detach().clamp(0.0, 1.0).squeeze(0)
    hwc = (chw.permute(1, 2, 0) * 255.0).round().to(torch.uint8).cpu().contiguous()
    h, w, _ = hwc.shape
    return Image.frombytes("RGB", (w, h), hwc.numpy().tobytes())


def _safe_name_token(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    out = "".join(ch if ch in allowed else "_" for ch in str(text))
    out = out.strip("._")
    return out or "x"


def _build_export_render_path(
    *,
    export_dir: Path,
    target_stem: str,
    ckpt_path: Path,
    backend: str,
    scale: int,
    sample_key: str,
    sample_count: int,
) -> Path:
    stem = "__".join(
        [
            _safe_name_token(target_stem),
            _safe_name_token(ckpt_path.stem),
            _safe_name_token(backend),
            f"x{int(scale)}",
            f"{_safe_name_token(sample_key)}{int(sample_count)}",
        ]
    )
    return export_dir / f"{stem}.png"


def _save_render_image(path: Path, render_nchw: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(render_nchw).save(path)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _filter_state_dict_by_shape(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    for key, value in state.items():
        if key not in model_state:
            continue
        model_value = model_state[key]
        if isinstance(value, torch.Tensor) and isinstance(model_value, torch.Tensor):
            if tuple(value.shape) != tuple(model_value.shape):
                skipped_shape.append(key)
                continue
        filtered[key] = value
    return filtered, skipped_shape


def _apply_sampling_density(model, sample_count: int) -> None:
    sample_count_i = max(2, int(sample_count))
    model.num_samples = sample_count_i
    model.total_num_sample = int(model.num_beziers) * sample_count_i

    dev = model._control_points.device
    for deg in (int(model.bezier_degree) + 1, int(model.bezier_degree)):
        for mul in (1, 2, 4, 8, 16):
            model._update_bernstein_cache(int(deg), sample_count_i * int(mul), dev)


def _build_model_for_sampling(
    *,
    meta: CheckpointMeta,
    target_base: torch.Tensor,
    device: torch.device,
    sample_count: int,
    sweep_key: str,
    renderer_backend: str,
    args: argparse.Namespace,
):
    sweep = str(sweep_key).strip().lower()
    if sweep not in {"num_samples", "distance_samples"}:
        raise ValueError(f"unsupported sweep key: {sweep_key}")
    if sweep == "num_samples":
        model_num_samples = int(sample_count)
        cubic_distance_train = int(args.cubic_distance_samples_train_base)
        cubic_distance_eval = int(args.cubic_distance_samples_eval_base)
    else:
        model_num_samples = int(meta.base_num_samples)
        cubic_distance_train = int(sample_count)
        cubic_distance_eval = int(sample_count)

    model_args = SimpleNamespace(
        block_w=int(args.block_w),
        block_h=int(args.block_h),
        mode=str(meta.mode),
        renderer_backend=str(renderer_backend),
        num_curves=int(meta.num_curves),
        bezier_degree=int(meta.bezier_degree),
        num_samples=int(model_num_samples),
        lr=1e-2,
        optimizer="adan",
        model_path=None,
        use_reg_loss=False,
        cubic_distance_samples_train=max(2, int(cubic_distance_train)),
        cubic_distance_samples_eval=max(2, int(cubic_distance_eval)),
        cubic_flatten_method="bernstein",
    )
    model = build_model(model_args, target_base, device, None)
    model.requires_grad_(False)
    if sweep == "num_samples":
        _apply_sampling_density(model, int(sample_count))
    else:
        v = max(2, int(sample_count))
        model.cubic_distance_samples_train = v
        model.cubic_distance_samples_eval = v
    return model


def _backend_supported(*, mode: str, bezier_degree: int, backend: str) -> bool:
    mode_norm = str(mode).lower()
    backend_norm = str(backend).lower()
    if backend_norm == "gaussian":
        return True
    if backend_norm == "cubic":
        return mode_norm == "unclosed" and int(bezier_degree) == 3
    if backend_norm == "cubic_fill":
        return mode_norm == "closed"
    return False


def _evaluate_model_once(
    model,
    *,
    scale: int,
    target_base: torch.Tensor,
    target_scaled: torch.Tensor,
    warmup: int,
    eval_iters: int,
    denser_sample: bool,
    psnr_compare_space: str,
    pred_downsample_mode: str,
) -> tuple[float, float, int, int, int, int, torch.Tensor]:
    warmup_i = max(0, int(warmup))
    eval_iters_i = max(1, int(eval_iters))
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_i):
            _ = _forward_with_scale_compat(
                model,
                scale=int(scale),
                denser_sample=bool(denser_sample),
            )

        _sync_if_cuda(target_scaled.device)
        t0 = time.perf_counter()
        pred = None
        for _ in range(eval_iters_i):
            pred = _forward_with_scale_compat(
                model,
                scale=int(scale),
                denser_sample=bool(denser_sample),
            )
        _sync_if_cuda(target_scaled.device)
        render_ms = (time.perf_counter() - t0) * 1000.0 / float(eval_iters_i)

        assert isinstance(pred, torch.Tensor)
        if tuple(pred.shape) != tuple(target_scaled.shape):
            raise RuntimeError(
                "pred/target shape mismatch at eval: "
                f"pred={tuple(pred.shape)} target={tuple(target_scaled.shape)}"
            )

        compare_space = str(psnr_compare_space).strip().lower()
        if compare_space == "scaled_target":
            pred_cmp = pred
            ref_cmp = target_scaled
        elif compare_space == "base_target":
            base_h = int(target_base.shape[-2])
            base_w = int(target_base.shape[-1])
            pred_cmp = _resize_image_tensor(
                pred,
                size_hw=(base_h, base_w),
                mode=str(pred_downsample_mode),
            )
            ref_cmp = target_base
        else:
            raise ValueError(f"unsupported psnr compare space: {psnr_compare_space}")

        if tuple(pred_cmp.shape) != tuple(ref_cmp.shape):
            raise RuntimeError(
                "PSNR compare tensor shape mismatch: "
                f"pred_cmp={tuple(pred_cmp.shape)} ref_cmp={tuple(ref_cmp.shape)} "
                f"compare_space={compare_space}"
            )
        score = float(psnr(pred_cmp.float(), ref_cmp.float()).item())
    return (
        score,
        float(render_ms),
        int(pred.shape[-2]),
        int(pred.shape[-1]),
        int(ref_cmp.shape[-2]),
        int(ref_cmp.shape[-1]),
        pred.detach().contiguous(),
    )


def _forward_with_scale_compat(
    model,
    *,
    scale: int,
    denser_sample: bool,
) -> torch.Tensor:
    mode_norm = str(getattr(model, "mode", "")).lower()
    backend_norm = str(getattr(model, "renderer_backend", "gaussian")).lower()
    scale_i = int(scale)
    # Workaround is needed only for gaussian+unclosed where factor>1 may hit
    # get_scaling_open shape assumptions. Cubic backends should keep native factor.
    if mode_norm != "unclosed" or backend_norm != "gaussian" or scale_i == 1:
        return model(factor=scale_i, denser_sample=bool(denser_sample))["render"]

    # Unclosed + factor>1 currently has a shape mismatch in get_scaling_open;
    # emulate magnification by scaling H/W for this forward call.
    h0 = int(getattr(model, "H", 0))
    w0 = int(getattr(model, "W", 0))
    if h0 <= 0 or w0 <= 0:
        raise ValueError(f"invalid model image size: H={h0}, W={w0}")
    model.H = h0 * scale_i
    model.W = w0 * scale_i
    try:
        return model(factor=1, denser_sample=bool(denser_sample))["render"]
    finally:
        model.H = h0
        model.W = w0


def _render_results(
    console: Console, rows: list[EvalRow], *, sample_label: str
) -> None:
    table = Table(title="Scale x Sampling PSNR", box=box.SIMPLE_HEAVY)
    table.add_column("scale", justify="right", style="cyan")
    table.add_column(str(sample_label), justify="right", style="cyan")
    table.add_column("render_size", justify="right")
    table.add_column("psnr_ref", justify="right")
    table.add_column("psnr", justify="right", style="green")
    table.add_column("render_ms", justify="right")
    table.add_column("missing", justify="right")
    table.add_column("skipped_shape", justify="right")
    for row in rows:
        table.add_row(
            str(int(row.scale)),
            str(int(row.sample_count)),
            f"{int(row.height)}x{int(row.width)}",
            f"{int(row.compare_height)}x{int(row.compare_width)}",
            f"{float(row.psnr):.4f}",
            f"{float(row.render_ms):.3f}",
            str(int(row.missing_keys)),
            str(int(row.skipped_shape_keys)),
        )
    console.print(table)


def _render_best_per_scale(
    console: Console, rows: list[EvalRow], *, sample_label: str
) -> None:
    by_scale: dict[int, EvalRow] = {}
    for row in rows:
        prev = by_scale.get(int(row.scale))
        if prev is None or float(row.psnr) > float(prev.psnr):
            by_scale[int(row.scale)] = row

    table = Table(title="Best PSNR Per Scale", box=box.SIMPLE_HEAVY)
    table.add_column("scale", justify="right", style="cyan")
    table.add_column(f"best_{sample_label}", justify="right", style="cyan")
    table.add_column("best_psnr", justify="right", style="green")
    for scale in sorted(by_scale):
        row = by_scale[scale]
        table.add_row(
            str(int(scale)),
            str(int(row.sample_count)),
            f"{float(row.psnr):.4f}",
        )
    console.print(table)


def _resolve_renderer_backend(
    *,
    requested_backend: str,
    meta: CheckpointMeta,
    state: dict[str, torch.Tensor],
    target_base: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[str, list[BackendProbeRow]]:
    backend_req = str(requested_backend).strip().lower()
    if backend_req != "auto":
        if not _backend_supported(
            mode=str(meta.mode),
            bezier_degree=int(meta.bezier_degree),
            backend=backend_req,
        ):
            raise ValueError(
                "renderer backend unsupported by checkpoint mode/degree: "
                f"backend={backend_req} mode={meta.mode} bezier_degree={meta.bezier_degree}"
            )
        return backend_req, []

    candidates = [
        b
        for b in ("gaussian", "cubic", "cubic_fill")
        if _backend_supported(
            mode=str(meta.mode),
            bezier_degree=int(meta.bezier_degree),
            backend=b,
        )
    ]
    if not candidates:
        raise RuntimeError(
            "no supported renderer backend candidates for "
            f"mode={meta.mode} bezier_degree={meta.bezier_degree}"
        )

    probe_rows: list[BackendProbeRow] = []
    best_backend = candidates[0]
    best_psnr = float("-inf")
    for backend in candidates:
        probe_sweep_key = (
            "num_samples" if str(backend) == "gaussian" else "distance_samples"
        )
        probe_value = (
            int(meta.base_num_samples)
            if probe_sweep_key == "num_samples"
            else int(args.cubic_distance_samples_eval_base)
        )
        model = _build_model_for_sampling(
            meta=meta,
            target_base=target_base,
            device=device,
            sample_count=int(probe_value),
            sweep_key=str(probe_sweep_key),
            renderer_backend=str(backend),
            args=args,
        )
        filtered_state, _ = _filter_state_dict_by_shape(model, state)
        model.load_state_dict(filtered_state, strict=False)
        score, *_ = _evaluate_model_once(
            model,
            scale=1,
            target_base=target_base,
            target_scaled=target_base,
            warmup=0,
            eval_iters=1,
            denser_sample=bool(args.denser_sample),
            psnr_compare_space=str(args.psnr_compare_space),
            pred_downsample_mode=str(args.pred_downsample_mode),
        )
        probe_rows.append(BackendProbeRow(backend=str(backend), psnr=float(score)))
        if float(score) > float(best_psnr):
            best_psnr = float(score)
            best_backend = str(backend)
        del model
        _sync_if_cuda(device)
    return best_backend, probe_rows


def _render_backend_probe(
    console: Console, rows: list[BackendProbeRow], selected: str
) -> None:
    if not rows:
        return
    table = Table(
        title="Backend Auto Probe (scale=1, base samples)", box=box.SIMPLE_HEAVY
    )
    table.add_column("backend", style="cyan")
    table.add_column("psnr", justify="right", style="green")
    table.add_column("selected", justify="center")
    for row in rows:
        is_sel = str(row.backend) == str(selected)
        table.add_row(
            str(row.backend),
            f"{float(row.psnr):.4f}",
            "yes" if is_sel else "",
        )
    console.print(table)


def _run_main(args: argparse.Namespace, console: Console) -> None:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    target_path = Path(args.target)

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_seed(int(args.seed))
    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint must contain model state dict, got {type(state)}")

    meta = _read_checkpoint_meta(
        state,
        mode_arg=str(args.mode),
        bezier_degree_arg=int(args.bezier_degree),
        hint_path=ckpt_path,
    )
    scales = _parse_int_list(str(args.scales), name="--scales")

    target_base = _load_image_tensor(target_path, device)
    target_by_scale: dict[int, torch.Tensor] = {}
    for scale in scales:
        target_by_scale[int(scale)] = _scaled_target(
            target_base,
            scale=int(scale),
            resize_mode=str(args.target_resize_mode),
        )

    selected_backend, backend_probe_rows = _resolve_renderer_backend(
        requested_backend=str(args.renderer_backend),
        meta=meta,
        state=state,
        target_base=target_base,
        device=device,
        args=args,
    )
    sweep_key, sample_counts = _resolve_sweep_values(
        selected_backend=str(selected_backend),
        sample_counts_raw=str(args.sample_counts),
        sample_ratios_raw=str(args.sample_ratios),
        base_num_samples=int(meta.base_num_samples),
        base_cubic_distance_samples_eval=int(args.cubic_distance_samples_eval_base),
    )
    sample_label = "num_samples" if sweep_key == "num_samples" else "distance_samples"

    out_dir = Path(args.output_dir) / ckpt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    export_render_dir = out_dir / "renders"
    csv_path = out_dir / str(args.csv_name)
    json_path = out_dir / str(args.json_name)

    console.print(
        "[bold]Config[/bold] "
        f"checkpoint={ckpt_path} target={args.target} mode={meta.mode} "
        f"degree={meta.bezier_degree} base_samples={meta.base_num_samples} "
        f"scales={scales} {sample_label}={sample_counts} "
        f"renderer_backend={selected_backend} "
        f"psnr_compare_space={args.psnr_compare_space}"
    )
    _render_backend_probe(console, backend_probe_rows, selected_backend)
    if sweep_key == "distance_samples":
        console.print(
            "[yellow]note:[/yellow] current sweep is `distance_samples` for cubic backend."
        )

    rows: list[EvalRow] = []
    exported_render_count = 0
    for sample_count in sample_counts:
        model = _build_model_for_sampling(
            meta=meta,
            target_base=target_base,
            device=device,
            sample_count=int(sample_count),
            sweep_key=str(sweep_key),
            renderer_backend=str(selected_backend),
            args=args,
        )
        filtered_state, skipped_shape = _filter_state_dict_by_shape(model, state)
        load_info = model.load_state_dict(filtered_state, strict=False)
        model.eval()

        for scale in scales:
            target_scaled = target_by_scale[int(scale)]
            (
                score,
                render_ms,
                render_h,
                render_w,
                compare_h,
                compare_w,
                render_image,
            ) = _evaluate_model_once(
                model,
                scale=int(scale),
                target_base=target_base,
                target_scaled=target_scaled,
                warmup=int(args.warmup),
                eval_iters=int(args.eval_iters),
                denser_sample=bool(args.denser_sample),
                psnr_compare_space=str(args.psnr_compare_space),
                pred_downsample_mode=str(args.pred_downsample_mode),
            )
            export_render_path = ""
            if bool(args.export_renders):
                export_path = _build_export_render_path(
                    export_dir=export_render_dir,
                    target_stem=target_path.stem,
                    ckpt_path=ckpt_path,
                    backend=str(selected_backend),
                    scale=int(scale),
                    sample_key=str(sweep_key),
                    sample_count=int(sample_count),
                )
                _save_render_image(export_path, render_image)
                export_render_path = str(export_path)
                exported_render_count += 1
            rows.append(
                EvalRow(
                    scale=int(scale),
                    sample_count=int(sample_count),
                    height=int(render_h),
                    width=int(render_w),
                    compare_height=int(compare_h),
                    compare_width=int(compare_w),
                    psnr=float(score),
                    render_ms=float(render_ms),
                    missing_keys=len(load_info.missing_keys),
                    unexpected_keys=len(load_info.unexpected_keys),
                    skipped_shape_keys=len(skipped_shape),
                    export_render_path=export_render_path,
                )
            )

        del model
        _sync_if_cuda(device)

    _render_results(console, rows, sample_label=str(sample_label))
    _render_best_per_scale(console, rows, sample_label=str(sample_label))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_key",
                "scale",
                "sample_count",
                "height",
                "width",
                "compare_height",
                "compare_width",
                "psnr",
                "render_ms",
                "missing_keys",
                "unexpected_keys",
                "skipped_shape_keys",
                "export_render_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            out_row = asdict(row)
            out_row = {"sample_key": str(sweep_key), **out_row}
            writer.writerow(out_row)

    payload = {
        "checkpoint": str(ckpt_path),
        "target": str(args.target),
        "mode": meta.mode,
        "bezier_degree": int(meta.bezier_degree),
        "num_curves": int(meta.num_curves),
        "base_num_samples": int(meta.base_num_samples),
        "renderer_backend": str(selected_backend),
        "scales": [int(x) for x in scales],
        "sample_counts": [int(x) for x in sample_counts],
        "sample_key": str(sweep_key),
        "psnr_compare_space": str(args.psnr_compare_space),
        "pred_downsample_mode": str(args.pred_downsample_mode),
        "target_resize_mode": str(args.target_resize_mode),
        "denser_sample": bool(args.denser_sample),
        "export_renders": bool(args.export_renders),
        "export_render_dir": str(export_render_dir),
        "backend_probe": [asdict(row) for row in backend_probe_rows],
        "rows": [asdict(row) for row in rows],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    console.print(
        f"[bold]Saved[/bold] csv={csv_path} json={json_path} "
        f"exported_renders={int(exported_render_count)}"
    )


def main() -> None:
    args = parse_args()
    console = Console()
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        _run_main(args, console)
    finally:
        torch.set_grad_enabled(prev_grad_state)


if __name__ == "__main__":
    main()
