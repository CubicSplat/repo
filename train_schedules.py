from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch

from manbo.sparse_init import SparseCoordInit, SparseInitConfig


@torch.no_grad()
def assign_curves_to_tiles(
    xys: torch.Tensor,
    N: int,
    H: int,
    W: int,
    block: int = 16,
) -> dict[int, list[int]]:
    if N <= 0:
        return {}
    if not isinstance(xys, torch.Tensor):
        return {}
    if xys.ndim < 2 or int(xys.shape[-1]) != 2:
        return {}

    pts = xys.detach().to(torch.float32)
    if pts.ndim == 2:
        if int(pts.shape[0]) % int(N) != 0:
            return {}
        pts = pts.view(int(N), -1, 2)
    elif int(pts.shape[0]) == int(N):
        pts = pts.reshape(int(N), -1, 2)
    else:
        flat = pts.reshape(-1, 2)
        if int(flat.shape[0]) % int(N) != 0:
            return {}
        pts = flat.view(int(N), -1, 2)
    if int(pts.shape[1]) <= 0:
        return {}

    pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    x = pts[..., 0]
    y = pts[..., 1]
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    y_min = float(y.min().item())
    y_max = float(y.max().item())
    normalized_coords = (
        x_min >= -1.5 and x_max <= 1.5 and y_min >= -1.5 and y_max <= 1.5
    )

    w_i = max(1, int(W))
    h_i = max(1, int(H))
    block_i = max(1, int(block))
    if normalized_coords:
        px = (x + 1.0) * 0.5 * float(max(1, w_i - 1))
        py = (y + 1.0) * 0.5 * float(max(1, h_i - 1))
    else:
        px = x
        py = y
    px = px.clamp(0.0, float(max(0, w_i - 1)))
    py = py.clamp(0.0, float(max(0, h_i - 1)))

    gw = max(1, (w_i + block_i - 1) // block_i)
    gh = max(1, (h_i + block_i - 1) // block_i)
    tx = torch.div(px, float(block_i), rounding_mode="floor").to(torch.int64)
    ty = torch.div(py, float(block_i), rounding_mode="floor").to(torch.int64)
    tx = tx.clamp(0, gw - 1)
    ty = ty.clamp(0, gh - 1)
    tile_ids = (ty * gw + tx).to(torch.int64).to(device="cpu")

    tile_to_curves: dict[int, list[int]] = defaultdict(list)
    for ci in range(int(N)):
        uniq_tiles = torch.unique(tile_ids[ci])
        for tile_id in uniq_tiles.tolist():
            tile_to_curves[int(tile_id)].append(int(ci))
    return dict(tile_to_curves)


def compute_cooc_jaccard(
    tile_to_curves: dict[int, list[int]],
    N: int,
    contrib_rates: Optional[torch.Tensor] = None,
) -> List[Tuple[int, int, float]]:
    if int(N) <= 1:
        return []

    curve_tile_count = [0 for _ in range(int(N))]
    cooc: dict[Tuple[int, int], int] = defaultdict(int)
    rates: Optional[list[float]] = None
    if isinstance(contrib_rates, torch.Tensor):
        flat = contrib_rates.detach().view(-1).to(torch.float32)
        if int(flat.numel()) == int(N):
            rates = flat.to(device="cpu").tolist()

    for curves in tile_to_curves.values():
        if not curves:
            continue
        uniq = sorted({int(ci) for ci in curves if int(ci) >= 0 and int(ci) < int(N)})
        k = len(uniq)
        if k <= 0:
            continue
        for ci in uniq:
            curve_tile_count[ci] += 1
        if k < 2:
            continue
        for i in range(k):
            ci = uniq[i]
            for j in range(i + 1, k):
                cj = uniq[j]
                cooc[(ci, cj)] += 1

    results: list[Tuple[int, int, float]] = []
    for (ci, cj), inter in cooc.items():
        union = int(curve_tile_count[ci] + curve_tile_count[cj] - inter)
        if union <= 0:
            continue
        score = float(float(inter) / float(union))
        if rates is not None:
            wi = max(0.0, float(rates[ci]))
            wj = max(0.0, float(rates[cj]))
            low_contrib_weight = min(2.0, 1.0 + 1.0 / (wi + wj + 1e-4))
            score = score * low_contrib_weight
        results.append((int(ci), int(cj), float(score)))
    return results


def compute_contrib_jaccard(
    tile_ids: torch.Tensor,
    curve_ids: torch.Tensor,
    contrib_values: torch.Tensor,
    *,
    N: int,
    eps: float = 1e-8,
) -> List[Tuple[int, int, float]]:
    if int(N) <= 1:
        return []
    if (
        not isinstance(tile_ids, torch.Tensor)
        or not isinstance(curve_ids, torch.Tensor)
        or not isinstance(contrib_values, torch.Tensor)
    ):
        return []
    if tile_ids.ndim != 1 or curve_ids.ndim != 1 or contrib_values.ndim != 1:
        return []
    m = min(int(tile_ids.numel()), int(curve_ids.numel()), int(contrib_values.numel()))
    if m <= 0:
        return []

    tiles = tile_ids.detach().to(device="cpu", dtype=torch.int64)[:m]
    curves = curve_ids.detach().to(device="cpu", dtype=torch.int64)[:m]
    vals = (
        torch.nan_to_num(
            contrib_values.detach().to(device="cpu", dtype=torch.float32)[:m],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        .clamp(min=0.0)
        .tolist()
    )
    tile_list = tiles.tolist()
    curve_list = curves.tolist()

    tile_to_curve_vals: dict[int, dict[int, float]] = defaultdict(dict)
    for t, c, v in zip(tile_list, curve_list, vals):
        t_i = int(t)
        c_i = int(c)
        if c_i < 0 or c_i >= int(N):
            continue
        if v <= 0.0:
            continue
        prev = tile_to_curve_vals[t_i].get(c_i, 0.0)
        tile_to_curve_vals[t_i][c_i] = float(prev + float(v))

    curve_total: list[float] = [0.0 for _ in range(int(N))]
    pair_min: dict[tuple[int, int], float] = defaultdict(float)
    for curve_val_dict in tile_to_curve_vals.values():
        if len(curve_val_dict) <= 0:
            continue
        items = list(curve_val_dict.items())
        k = len(items)
        for ci, vi in items:
            curve_total[int(ci)] += float(vi)
        if k < 2:
            continue
        for i in range(k):
            ci, vi = items[i]
            for j in range(i + 1, k):
                cj, vj = items[j]
                if ci == cj:
                    continue
                key = (ci, cj) if ci < cj else (cj, ci)
                pair_min[key] += float(min(vi, vj))

    out: list[tuple[int, int, float]] = []
    denom_eps = float(max(1e-12, eps))
    for key, num in pair_min.items():
        den = float(curve_total[key[0]] + curve_total[key[1]] - float(num))
        if den <= denom_eps:
            continue
        out.append((int(key[0]), int(key[1]), float(num / den)))
    return out


def greedy_merge_pairs(
    candidates: List[Tuple[int, int, float]],
    jaccard_threshold: float,
    max_pairs: int,
    min_keep: int,
    N: int,
) -> List[Tuple[int, int]]:
    if int(max_pairs) <= 0 or int(N) <= 1:
        return []
    threshold = float(jaccard_threshold)
    candidates_sorted = sorted(candidates, key=lambda x: float(x[2]), reverse=True)
    used: set[int] = set()
    pairs: list[Tuple[int, int]] = []
    n_cur = int(N)
    for ci, cj, score in candidates_sorted:
        ci_i = int(ci)
        cj_i = int(cj)
        if ci_i == cj_i:
            continue
        if float(score) < threshold:
            break
        if ci_i in used or cj_i in used:
            continue
        if n_cur - 1 < int(min_keep):
            break
        pairs.append((ci_i, cj_i))
        used.add(ci_i)
        used.add(cj_i)
        n_cur -= 1
        if len(pairs) >= int(max_pairs):
            break
    return pairs


@torch.no_grad()
def execute_merge(
    model,
    pairs: List[Tuple[int, int]],
    contrib_rates: Optional[torch.Tensor],
    gt_image: torch.Tensor,
    pred_image: Optional[torch.Tensor],
    sparse_init_config: SparseInitConfig,
    densify_radii: float,
) -> int:
    if not pairs:
        return 0
    N = int(model._control_points.shape[0])
    if N <= 1:
        return 0

    device = model._control_points.device
    rates = torch.ones((N,), dtype=torch.float32, device=device)
    if isinstance(contrib_rates, torch.Tensor):
        flat = contrib_rates.detach().view(-1).to(device=device, dtype=torch.float32)
        if int(flat.numel()) == N:
            rates = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0).clamp(
                min=0.0
            )

    cp = model._control_points.detach()
    feat = model._features_dc.detach()
    opa = model._opacity.detach()
    dep = model._depth.detach()
    scl = model._scaling.detach()
    cho = model._cholesky.detach()

    remove_set: set[int] = set()
    new_cp: list[torch.Tensor] = []
    new_feat: list[torch.Tensor] = []
    new_opa: list[torch.Tensor] = []
    new_dep: list[torch.Tensor] = []
    new_scl: list[torch.Tensor] = []
    new_cho: list[torch.Tensor] = []

    for ci, cj in pairs:
        ci_i = int(ci)
        cj_i = int(cj)
        if ci_i < 0 or cj_i < 0 or ci_i >= N or cj_i >= N:
            continue
        if ci_i == cj_i:
            continue
        if ci_i in remove_set or cj_i in remove_set:
            continue
        wi = float(rates[ci_i].item())
        wj = float(rates[cj_i].item())
        denom = wi + wj
        if denom <= 1e-8:
            a = 0.5
        else:
            a = wi / denom
        b = 1.0 - a

        new_cp.append((a * cp[ci_i] + b * cp[cj_i]).unsqueeze(0))
        new_feat.append((a * feat[ci_i] + b * feat[cj_i]).unsqueeze(0))
        new_opa.append((a * opa[ci_i] + b * opa[cj_i]).unsqueeze(0))
        new_dep.append((a * dep[ci_i] + b * dep[cj_i]).unsqueeze(0))
        new_scl.append((a * scl[ci_i] + b * scl[cj_i]).unsqueeze(0))
        new_cho.append((a * cho[ci_i] + b * cho[cj_i]).unsqueeze(0))
        remove_set.add(ci_i)
        remove_set.add(cj_i)

    if not new_cp:
        return 0

    merged_count = int(len(new_cp))
    keep_mask = torch.ones((N,), dtype=torch.bool, device=device)
    for idx in remove_set:
        keep_mask[int(idx)] = False
    model.prune_beizer_curves(keep_mask)
    model.num_curves = int(model._control_points.shape[0])

    model.densification_postfix(
        torch.cat(new_cp, dim=0).to(device=device, dtype=cp.dtype).contiguous(),
        torch.cat(new_feat, dim=0).to(device=device, dtype=feat.dtype).contiguous(),
        torch.cat(new_cho, dim=0).to(device=device, dtype=cho.dtype).contiguous(),
        torch.cat(new_dep, dim=0).to(device=device, dtype=dep.dtype).contiguous(),
        torch.cat(new_opa, dim=0).to(device=device, dtype=opa.dtype).contiguous(),
        torch.cat(new_scl, dim=0).to(device=device, dtype=scl.dtype).contiguous(),
    )
    model.num_curves = int(model._control_points.shape[0])

    if merged_count > 0:
        try:
            pred = pred_image if isinstance(pred_image, torch.Tensor) else gt_image
            init = SparseCoordInit(pred, gt_image, config=sparse_init_config)
            model.densify(int(merged_count), init, gt_image, float(densify_radii))
            model.num_curves = int(model._control_points.shape[0])
        except Exception:
            pass

    return int(merged_count)


@dataclass(slots=True)
class CurveScheduleController:
    schedule: str
    target_num_curves: int
    model_mode: str
    total_steps: int = 0
    adaptive_freeze_last_steps: int = 2000
    sparse_init_config: SparseInitConfig = field(default_factory=SparseInitConfig)
    layerwised_base_curves: int = 24
    layerwised_step_every: int = 1000
    layerwised_max_chunk: int = 64
    layerwised_radii_switch_iter: int = 9200
    layerwised_radii_before: float = 0.02
    layerwised_radii_after: float = 0.01
    layerwised_prune_by_contrib: bool = False
    layerwised_prune_ratio: float = 0.0
    layerwised_prune_min_keep: int = 1
    layerwised_prune_max_remove: int = 0
    layerwised_prune_warmup_iter: int = 0
    prune_densify_every: int = 500
    prune_densify_start_iter: int = 1000
    prune_densify_end_closed: int = 9200
    prune_densify_end_unclosed: int = 14000
    gini_step_every: int = 500
    gini_warmup_iter: int = 1000
    gini_trigger_low: float = 0.35
    gini_trigger_high: float = 0.85
    gini_prune_min_ratio: float = 0.0
    gini_prune_max_ratio: float = 0.2
    gini_prune_min_keep: int = 24
    gini_prune_max_remove: int = 0
    gini_densify_radii: float = 0.01
    gini_ema_beta: float = 0.8
    merge_step_every: int = 1000
    merge_warmup_iter: int = 2000
    merge_jaccard_min: float = 0.35
    merge_max_pairs: int = 24
    merge_densify_radii: float = 0.01
    merge_tile_block: int = 16
    _pending_removed: int = field(init=False, default=0)
    _allocations: list[int] = field(init=False, default_factory=list)
    _gini_ema: Optional[float] = field(init=False, default=None)
    _adaptive_tail_guard_notified: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.schedule = str(self.schedule)
        self.target_num_curves = int(self.target_num_curves)
        self.total_steps = int(max(0, self.total_steps))
        self.adaptive_freeze_last_steps = int(max(0, self.adaptive_freeze_last_steps))
        self.layerwised_base_curves = int(self.layerwised_base_curves)
        self.layerwised_step_every = int(max(1, self.layerwised_step_every))
        self.layerwised_max_chunk = int(max(1, self.layerwised_max_chunk))
        self.layerwised_prune_by_contrib = bool(self.layerwised_prune_by_contrib)
        self.layerwised_prune_ratio = float(
            min(1.0, max(0.0, self.layerwised_prune_ratio))
        )
        self.layerwised_prune_min_keep = int(max(1, self.layerwised_prune_min_keep))
        self.layerwised_prune_max_remove = int(max(0, self.layerwised_prune_max_remove))
        self.layerwised_prune_warmup_iter = int(
            max(0, self.layerwised_prune_warmup_iter)
        )
        self.prune_densify_every = int(max(1, self.prune_densify_every))
        self.prune_densify_start_iter = int(max(1, self.prune_densify_start_iter))
        self.prune_densify_end_closed = int(max(1, self.prune_densify_end_closed))
        self.prune_densify_end_unclosed = int(max(1, self.prune_densify_end_unclosed))
        self.gini_step_every = int(max(1, self.gini_step_every))
        self.gini_warmup_iter = int(max(0, self.gini_warmup_iter))
        self.gini_trigger_low = float(min(1.0, max(0.0, self.gini_trigger_low)))
        self.gini_trigger_high = float(min(1.0, max(0.0, self.gini_trigger_high)))
        if self.gini_trigger_high < self.gini_trigger_low:
            self.gini_trigger_high = self.gini_trigger_low
        self.gini_prune_min_ratio = float(min(1.0, max(0.0, self.gini_prune_min_ratio)))
        self.gini_prune_max_ratio = float(min(1.0, max(0.0, self.gini_prune_max_ratio)))
        if self.gini_prune_max_ratio < self.gini_prune_min_ratio:
            self.gini_prune_max_ratio = self.gini_prune_min_ratio
        self.gini_prune_min_keep = int(max(1, self.gini_prune_min_keep))
        self.gini_prune_max_remove = int(max(0, self.gini_prune_max_remove))
        self.gini_densify_radii = float(max(0.0, self.gini_densify_radii))
        self.gini_ema_beta = float(min(1.0, max(0.0, self.gini_ema_beta)))
        self.merge_step_every = int(max(1, self.merge_step_every))
        self.merge_warmup_iter = int(max(0, self.merge_warmup_iter))
        self.merge_jaccard_min = float(max(0.0, self.merge_jaccard_min))
        self.merge_max_pairs = int(max(0, self.merge_max_pairs))
        self.merge_densify_radii = float(max(0.0, self.merge_densify_radii))
        self.merge_tile_block = int(max(1, self.merge_tile_block))

        if isinstance(self.sparse_init_config, dict):
            cfg_dict = dict(self.sparse_init_config)
            cfg_dict.pop("backend", None)
            self.sparse_init_config = SparseInitConfig(**cfg_dict)
        elif not isinstance(self.sparse_init_config, SparseInitConfig):
            raise TypeError("sparse_init_config must be SparseInitConfig or dict")

        self._pending_removed = 0
        self._gini_ema = None
        self._adaptive_tail_guard_notified = False
        if self.schedule == "layerwised":
            remaining = max(0, self.target_num_curves - self.layerwised_base_curves)
            self._allocations = self._build_layerwised_allocations(
                remaining,
                self.layerwised_max_chunk,
            )
        else:
            self._allocations = []

    def _adaptive_tail_guard_start_step(self) -> Optional[int]:
        if self.schedule != "adaptive":
            return None
        total = int(self.total_steps)
        freeze = int(self.adaptive_freeze_last_steps)
        if total <= 0 or freeze <= 0:
            return None
        return int(max(1, total - freeze + 1))

    def _adaptive_tail_guard_active(self, step: int) -> bool:
        start = self._adaptive_tail_guard_start_step()
        if start is None:
            return False
        return int(step) >= int(start)

    @property
    def enabled(self) -> bool:
        return self.schedule != "none"

    @property
    def allocations(self) -> list[int]:
        return list(self._allocations)

    @staticmethod
    def _build_layerwised_allocations(remaining: int, max_chunk: int) -> List[int]:
        if remaining <= 0:
            return []
        alloc: List[int] = []
        chunk = 1
        total = 0
        while total < remaining:
            take = min(chunk, remaining - total)
            alloc.append(int(take))
            total += int(take)
            chunk = min(chunk * 2, max_chunk)

        # Legacy-compatible behavior:
        # - drop the first stage (usually 1)
        # - preserve target endpoint by compensating the dropped count in the last stage
        if alloc:
            dropped = int(alloc.pop(0))
            if alloc:
                alloc[-1] = int(alloc[-1] + dropped)
            else:
                alloc = [dropped]
        return alloc

    def _end_iter(self) -> int:
        return (
            self.prune_densify_end_unclosed
            if self.model_mode == "unclosed"
            else self.prune_densify_end_closed
        )

    def _make_sparse_init_method(
        self, pred_image: torch.Tensor, gt_image: torch.Tensor
    ) -> SparseCoordInit:
        return SparseCoordInit(
            pred_image,
            gt_image,
            config=self.sparse_init_config,
        )

    def needs_contrib(self, step: int) -> bool:
        if self.schedule == "layerwised":
            if not self.layerwised_prune_by_contrib:
                return False
            if step < self.layerwised_prune_warmup_iter:
                return False
            if step % self.layerwised_step_every != 0:
                return False
            if not self._allocations:
                return False
            return True
        if self.schedule == "gini_prune":
            if step < self.gini_warmup_iter:
                return False
            if step % self.gini_step_every != 0:
                return False
            return True
        if self.schedule == "adaptive":
            if self._adaptive_tail_guard_active(step):
                return False
            gini_due = (
                step >= self.gini_warmup_iter and step % self.gini_step_every == 0
            )
            return bool(gini_due)
        return False

    def _extract_contrib_rates(
        self, model, *, expected_step: int
    ) -> Optional[torch.Tensor]:
        payload = getattr(model, "_last_contrib_payload", None)
        if not isinstance(payload, dict):
            return None
        step_value = payload.get("step", None)
        if step_value is not None and int(step_value) != int(expected_step):
            return None
        rate_tensor = payload.get("rate_tensor")
        if not isinstance(rate_tensor, torch.Tensor):
            return None
        score_sum = float(payload.get("curve_contrib_score_sum", 0.0))
        if score_sum <= 0.0:
            return None

        num_curves = int(model._control_points.shape[0])
        rates = (
            rate_tensor.detach()
            .view(-1)
            .to(
                device=model._control_points.device,
                dtype=torch.float32,
            )
        )
        if int(rates.numel()) != num_curves:
            return None
        rates = torch.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        rates = torch.clamp(rates, min=0.0)
        return rates

    def _extract_tile_contrib_sparse(
        self, model, *, expected_step: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        payload = getattr(model, "_last_contrib_payload", None)
        if not isinstance(payload, dict):
            return None
        step_value = payload.get("step", None)
        if step_value is not None and int(step_value) != int(expected_step):
            return None
        score_sum = float(payload.get("curve_contrib_score_sum", 0.0))
        if score_sum <= 0.0:
            return None

        tile_ids = payload.get("tile_contrib_tile_ids")
        curve_ids = payload.get("tile_contrib_curve_ids")
        values = payload.get("tile_contrib_values")
        if (
            not isinstance(tile_ids, torch.Tensor)
            or not isinstance(curve_ids, torch.Tensor)
            or not isinstance(values, torch.Tensor)
        ):
            return None
        if tile_ids.ndim != 1 or curve_ids.ndim != 1 or values.ndim != 1:
            return None
        m = min(int(tile_ids.numel()), int(curve_ids.numel()), int(values.numel()))
        if m <= 0:
            return None

        num_curves = int(model._control_points.shape[0])
        tile_ids_i64 = tile_ids.detach().to(device="cpu", dtype=torch.int64)[:m]
        curve_ids_i64 = curve_ids.detach().to(device="cpu", dtype=torch.int64)[:m]
        values_f32 = (
            torch.nan_to_num(
                values.detach().to(device="cpu", dtype=torch.float32)[:m],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            .clamp(min=0.0)
            .contiguous()
        )
        valid = (
            (values_f32 > 0.0)
            & (tile_ids_i64 >= 0)
            & (curve_ids_i64 >= 0)
            & (curve_ids_i64 < num_curves)
        )
        if not bool(valid.any().item()):
            return None
        tile_ids_i64 = tile_ids_i64[valid]
        curve_ids_i64 = curve_ids_i64[valid]
        values_f32 = values_f32[valid]
        num_tiles = int(payload.get("tile_contrib_num_tiles", 0))
        if num_tiles <= 0 and int(tile_ids_i64.numel()) > 0:
            num_tiles = int(tile_ids_i64.max().item()) + 1
        return tile_ids_i64, curve_ids_i64, values_f32, int(max(0, num_tiles))

    @staticmethod
    def _gini_from_rates(rates: torch.Tensor) -> float:
        if int(rates.numel()) <= 1:
            return 0.0
        vals = torch.clamp(rates.detach().to(torch.float32), min=0.0)
        total = float(vals.sum().item())
        if total <= 0.0:
            return 0.0
        vals = vals / total
        vals, _ = torch.sort(vals)
        n = int(vals.numel())
        idx = torch.arange(1, n + 1, dtype=torch.float32, device=vals.device)
        gini = (2.0 * (idx * vals).sum() - (n + 1.0) * vals.sum()) / (n * vals.sum())
        return float(torch.clamp(gini, 0.0, 1.0).item())

    def _gini_remove_ratio(self, gini_value: float) -> float:
        low = float(self.gini_trigger_low)
        high = float(self.gini_trigger_high)
        if high <= low:
            t = 1.0 if gini_value >= high else 0.0
        else:
            t = (gini_value - low) / (high - low)
        t = min(1.0, max(0.0, t))
        ratio = self.gini_prune_min_ratio + t * (
            self.gini_prune_max_ratio - self.gini_prune_min_ratio
        )
        return float(min(1.0, max(0.0, ratio)))

    def _layerwised_contrib_keep_mask(
        self, model, *, expected_step: int
    ) -> Optional[torch.Tensor]:
        rates = self._extract_contrib_rates(model, expected_step=expected_step)
        if rates is None:
            return None
        num_curves = int(model._control_points.shape[0])
        if num_curves <= self.layerwised_prune_min_keep:
            return None

        remove_count = int(num_curves * self.layerwised_prune_ratio)
        if self.layerwised_prune_ratio > 0.0 and remove_count <= 0:
            remove_count = 1
        if self.layerwised_prune_max_remove > 0:
            remove_count = min(remove_count, self.layerwised_prune_max_remove)
        remove_count = min(remove_count, num_curves - self.layerwised_prune_min_keep)
        if remove_count <= 0:
            return None

        _, remove_idx = torch.topk(
            rates,
            k=int(remove_count),
            largest=False,
            sorted=False,
        )
        keep_mask = torch.ones((num_curves,), dtype=torch.bool, device=rates.device)
        keep_mask[remove_idx] = False
        return keep_mask

    def _gini_prune_plan(self, model, *, expected_step: int) -> dict[str, Any]:
        num_curves = int(model._control_points.shape[0])
        gini_ema_prev = float(self._gini_ema) if self._gini_ema is not None else 0.0
        plan: dict[str, Any] = {
            "keep_mask": None,
            "num_curves": int(num_curves),
            "gini_now": 0.0,
            "gini_ema_prev": float(gini_ema_prev),
            "gini_ema": float(gini_ema_prev),
            "saturation": False,
            "remove_ratio": 0.0,
            "remove_count": 0,
            "reason": "unknown",
        }

        rates = self._extract_contrib_rates(model, expected_step=expected_step)
        if rates is None:
            plan["reason"] = "no_rates"
            return plan
        if num_curves <= self.gini_prune_min_keep:
            plan["reason"] = "min_keep_guard"
            return plan

        gini_now = self._gini_from_rates(rates)
        if self._gini_ema is None:
            self._gini_ema = gini_now
        else:
            beta = float(self.gini_ema_beta)
            self._gini_ema = beta * self._gini_ema + (1.0 - beta) * gini_now
        gini_used = float(self._gini_ema)
        remove_ratio = self._gini_remove_ratio(gini_used)

        remove_count = int(num_curves * remove_ratio)
        if remove_ratio > 0.0 and remove_count <= 0:
            remove_count = 1
        if self.gini_prune_max_remove > 0:
            remove_count = min(remove_count, self.gini_prune_max_remove)
        remove_count = min(remove_count, num_curves - self.gini_prune_min_keep)

        plan["gini_now"] = float(gini_now)
        plan["gini_ema"] = float(gini_used)
        plan["saturation"] = bool(gini_now > (gini_used + 1e-9))
        plan["remove_ratio"] = float(remove_ratio)
        plan["remove_count"] = int(max(0, remove_count))

        if remove_count <= 0:
            plan["reason"] = "zero_remove"
            return plan

        _, remove_idx = torch.topk(
            rates,
            k=int(remove_count),
            largest=False,
            sorted=False,
        )
        keep_mask = torch.ones((num_curves,), dtype=torch.bool, device=rates.device)
        keep_mask[remove_idx] = False
        plan["keep_mask"] = keep_mask
        plan["reason"] = "ready"
        return plan

    def _gini_prune_keep_mask(
        self, model, *, expected_step: int
    ) -> Optional[torch.Tensor]:
        return self._gini_prune_plan(model, expected_step=expected_step).get(
            "keep_mask"
        )

    def _gini_step(
        self,
        *,
        step: int,
        model,
        gt_image: torch.Tensor,
        pred_image: torch.Tensor,
        prefer_merge_on_saturation: bool = False,
        merge_ready: bool = False,
    ) -> tuple[bool, Optional[dict[str, Any]], bool]:
        if step < self.gini_warmup_iter:
            return False, None, False
        if step % self.gini_step_every != 0:
            return False, None, False

        before_curves = int(model._control_points.shape[0])
        plan = self._gini_prune_plan(
            model,
            expected_step=int(step),
        )
        keep_mask = plan.get("keep_mask")
        saturation = bool(plan.get("saturation", False))
        event: dict[str, Any] = {
            "step": int(step),
            "schedule": str(self.schedule),
            "phase": "gini",
            "status": "skip",
            "reason": str(plan.get("reason", "unknown")),
            "curves_before": int(before_curves),
            "curves_after": int(before_curves),
            "gini_now": float(plan.get("gini_now", 0.0)),
            "gini_ema_prev": float(plan.get("gini_ema_prev", 0.0)),
            "gini_ema": float(plan.get("gini_ema", 0.0)),
            "saturation": bool(saturation),
            "remove_ratio": float(plan.get("remove_ratio", 0.0)),
            "remove_count": int(plan.get("remove_count", 0)),
        }
        if prefer_merge_on_saturation and saturation:
            if merge_ready:
                event["status"] = "switch"
                event["reason"] = "switch_to_merge"
                event["merge_trigger"] = "gini_saturation"
                return False, event, True
            event["merge_ready"] = False
            event["reason"] = "saturation_wait_merge_warmup"
        if keep_mask is None:
            return False, event, False
        removed = int((~keep_mask).sum().item())
        if removed <= 0:
            event["reason"] = "zero_remove"
            return False, event, False
        model.prune_beizer_curves(keep_mask)
        model.num_curves = int(model._control_points.shape[0])
        pos_init_method = self._make_sparse_init_method(pred_image, gt_image)
        model.densify(int(removed), pos_init_method, gt_image, self.gini_densify_radii)
        model.num_curves = int(model._control_points.shape[0])
        event.update(
            {
                "status": "applied",
                "reason": "prune_and_refill",
                "removed": int(removed),
                "densified": int(removed),
                "curves_after": int(model._control_points.shape[0]),
            }
        )
        return True, event, False

    def _merge_step(
        self,
        *,
        step: int,
        model,
        gt_image: torch.Tensor,
        pred_image: torch.Tensor,
        force_run: bool = False,
        trigger: str = "schedule",
    ) -> Optional[dict[str, Any]]:
        if step < self.merge_warmup_iter:
            return None
        if (not force_run) and step % self.merge_step_every != 0:
            return None
        num_curves = int(model._control_points.shape[0])
        event: dict[str, Any] = {
            "step": int(step),
            "schedule": str(self.schedule),
            "phase": "merge",
            "status": "skip",
            "reason": "unknown",
            "curves_before": int(num_curves),
            "curves_after": int(num_curves),
            "jaccard_threshold": float(self.merge_jaccard_min),
            "merge_trigger": str(trigger),
        }
        if num_curves <= max(1, int(self.gini_prune_min_keep) * 2):
            event["reason"] = "min_keep_guard"
            return event

        tile_sparse = self._extract_tile_contrib_sparse(model, expected_step=int(step))
        if tile_sparse is None:
            event["reason"] = "no_tile_contrib"
            return event
        tile_ids, curve_ids, contrib_vals, num_tiles = tile_sparse
        event["tile_count"] = int(num_tiles)
        event["tile_contrib_nnz"] = int(contrib_vals.numel())
        rates = self._extract_contrib_rates(model, expected_step=int(step))
        candidates = compute_contrib_jaccard(
            tile_ids,
            curve_ids,
            contrib_vals,
            N=num_curves,
        )
        event["merge_metric"] = "contrib_jaccard"
        event["candidate_pairs"] = int(len(candidates))
        pairs = greedy_merge_pairs(
            candidates,
            jaccard_threshold=self.merge_jaccard_min,
            max_pairs=self.merge_max_pairs,
            min_keep=self.gini_prune_min_keep,
            N=num_curves,
        )
        event["selected_pairs"] = int(len(pairs))
        if not pairs:
            event["reason"] = "no_pairs_above_threshold"
            return event

        merged = execute_merge(
            model,
            pairs,
            rates,
            gt_image,
            pred_image,
            self.sparse_init_config,
            self.merge_densify_radii,
        )
        if merged > 0:
            event.update(
                {
                    "status": "applied",
                    "reason": "merge_and_refill",
                    "merged_pairs": int(merged),
                    "densified": int(merged),
                    "curves_after": int(model._control_points.shape[0]),
                }
            )
        else:
            event["reason"] = "merge_failed"
        return event

    def after_step(
        self, *, step: int, model, gt_image: torch.Tensor, pred_image: torch.Tensor
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.schedule == "none":
            return events

        if self.schedule == "layerwised":
            if step % self.layerwised_step_every != 0:
                return events
            if not self._allocations:
                return events
            before_curves = int(model._control_points.shape[0])
            use_contrib_prune = self.needs_contrib(step)
            add_path_num = int(self._allocations.pop(0))
            removed_by_contrib = 0
            if use_contrib_prune:
                keep_mask = self._layerwised_contrib_keep_mask(
                    model,
                    expected_step=int(step),
                )
                if keep_mask is not None:
                    removed_by_contrib = int((~keep_mask).sum().item())
                    if removed_by_contrib > 0:
                        model.prune_beizer_curves(keep_mask)
                        model.num_curves = int(model._control_points.shape[0])
            densify_num = int(add_path_num + removed_by_contrib)
            if densify_num <= 0:
                events.append(
                    {
                        "step": int(step),
                        "schedule": str(self.schedule),
                        "phase": "layerwised",
                        "status": "skip",
                        "reason": "zero_densify",
                        "curves_before": int(before_curves),
                        "curves_after": int(model._control_points.shape[0]),
                        "planned_add": int(add_path_num),
                        "removed_by_contrib": int(removed_by_contrib),
                    }
                )
                return events
            radii = (
                self.layerwised_radii_before
                if step < self.layerwised_radii_switch_iter
                else self.layerwised_radii_after
            )
            pos_init_method = self._make_sparse_init_method(pred_image, gt_image)
            model.densify(densify_num, pos_init_method, gt_image, radii)
            model.num_curves = int(model._control_points.shape[0])
            events.append(
                {
                    "step": int(step),
                    "schedule": str(self.schedule),
                    "phase": "layerwised",
                    "status": "applied",
                    "reason": "grow_and_refill",
                    "curves_before": int(before_curves),
                    "curves_after": int(model._control_points.shape[0]),
                    "planned_add": int(add_path_num),
                    "removed_by_contrib": int(removed_by_contrib),
                    "densified": int(densify_num),
                }
            )
            return events

        if self.schedule == "gini_prune":
            _, gini_event, _ = self._gini_step(
                step=step,
                model=model,
                gt_image=gt_image,
                pred_image=pred_image,
            )
            if gini_event is not None:
                events.append(gini_event)
            return events

        if self.schedule == "adaptive":
            if self._adaptive_tail_guard_active(step):
                if not self._adaptive_tail_guard_notified:
                    self._adaptive_tail_guard_notified = True
                    start = self._adaptive_tail_guard_start_step()
                    events.append(
                        {
                            "step": int(step),
                            "schedule": str(self.schedule),
                            "phase": "adaptive",
                            "status": "skip",
                            "reason": "tail_guard",
                            "curves_before": int(model._control_points.shape[0]),
                            "curves_after": int(model._control_points.shape[0]),
                            "guard_start_step": int(start) if start is not None else -1,
                            "freeze_last_steps": int(self.adaptive_freeze_last_steps),
                            "total_steps": int(self.total_steps),
                        }
                    )
                return events
            gini_changed, gini_event, switch_to_merge = self._gini_step(
                step=step,
                model=model,
                gt_image=gt_image,
                pred_image=pred_image,
                prefer_merge_on_saturation=True,
                merge_ready=bool(step >= self.merge_warmup_iter),
            )
            if gini_event is not None:
                events.append(gini_event)
            if switch_to_merge:
                merge_event = self._merge_step(
                    step=step,
                    model=model,
                    gt_image=gt_image,
                    pred_image=pred_image,
                    force_run=True,
                    trigger="gini_saturation",
                )
                if merge_event is not None:
                    events.append(merge_event)
            elif not gini_changed and gini_event is None:
                # No gini trigger on this step; adaptive now switches merge only
                # by gini saturation, not by an independent merge cadence.
                pass
            return events

        if self.schedule == "prune_densify":
            end_iter = self._end_iter()
            if step < self.prune_densify_start_iter or step >= end_iter:
                return events
            if step % self.prune_densify_every != 0:
                return events

            cycle = step // self.prune_densify_every
            if cycle % 2 == 1:
                before_curves = int(model._control_points.shape[0])
                prune_mask = model.remove_curves_mask()
                removed = int((~prune_mask).sum().item())
                model.prune_beizer_curves(prune_mask)
                model.num_curves = int(model._control_points.shape[0])
                self._pending_removed += max(0, removed)
                events.append(
                    {
                        "step": int(step),
                        "schedule": str(self.schedule),
                        "phase": "prune_densify.prune",
                        "status": "applied",
                        "reason": "pending_refill",
                        "curves_before": int(before_curves),
                        "curves_after": int(model._control_points.shape[0]),
                        "removed": int(removed),
                        "pending_removed": int(self._pending_removed),
                    }
                )
                return events

            if self._pending_removed <= 0:
                events.append(
                    {
                        "step": int(step),
                        "schedule": str(self.schedule),
                        "phase": "prune_densify.refill",
                        "status": "skip",
                        "reason": "no_pending_removed",
                        "curves_before": int(model._control_points.shape[0]),
                        "curves_after": int(model._control_points.shape[0]),
                    }
                )
                return events
            before_curves = int(model._control_points.shape[0])
            pos_init_method = self._make_sparse_init_method(pred_image, gt_image)
            model.densify(int(self._pending_removed), pos_init_method, gt_image)
            model.num_curves = int(model._control_points.shape[0])
            densified = int(self._pending_removed)
            self._pending_removed = 0
            events.append(
                {
                    "step": int(step),
                    "schedule": str(self.schedule),
                    "phase": "prune_densify.refill",
                    "status": "applied",
                    "reason": "refilled_pending",
                    "curves_before": int(before_curves),
                    "curves_after": int(model._control_points.shape[0]),
                    "densified": int(densified),
                }
            )
            return events

        raise ValueError(f"Unsupported curve schedule: {self.schedule}")


__all__ = [
    "SparseInitConfig",
    "CurveScheduleController",
    "assign_curves_to_tiles",
    "compute_cooc_jaccard",
    "compute_contrib_jaccard",
    "greedy_merge_pairs",
    "execute_merge",
]
