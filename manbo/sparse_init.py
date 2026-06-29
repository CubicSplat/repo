from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from manbo.ops import (
    connected_components_with_stats_4_tilelang,
    select_largest_component_center,
)


@dataclass(slots=True)
class SparseInitConfig:
    quantile_interval: int = 200
    nodiff_thres: float = 0.05
    tolerance: int = 1
    connectivity: int = 4
    max_iters: int = 4096
    topk: int = 4096
    suppress_radius: int | None = None

    def __post_init__(self) -> None:
        self.quantile_interval = int(max(2, self.quantile_interval))
        self.nodiff_thres = float(max(0.0, self.nodiff_thres))
        self.tolerance = int(max(0, self.tolerance))
        self.connectivity = int(self.connectivity)
        self.max_iters = int(max(1, self.max_iters))
        self.topk = int(max(1, self.topk))
        self.suppress_radius = (
            None
            if self.suppress_radius is None
            else int(max(1, int(self.suppress_radius)))
        )

        if self.connectivity != 4:
            raise ValueError("only 4-connectivity is currently supported")


def _build_quantized_error_map_torch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    quantile_interval: int,
    nodiff_thres: float,
) -> torch.Tensor:
    err = ((pred.detach().float()[0] - gt.detach().float()[0]) ** 2).sum(dim=0)
    err = torch.where(
        err < float(nodiff_thres),
        torch.zeros((), dtype=err.dtype, device=err.device),
        err,
    )

    quantiles = torch.linspace(
        0.0,
        1.0,
        int(max(2, quantile_interval)),
        dtype=err.dtype,
        device=err.device,
    )
    qv = torch.quantile(err.reshape(-1), quantiles)
    qv = torch.unique(qv, sorted=True)
    bins = qv[1:-1] if int(qv.numel()) > 2 else err.new_empty((0,))
    out = torch.bucketize(err, bins, right=False)
    return out.clamp(0, 255).to(torch.uint8).contiguous()


def _build_idcnt_from_map_torch(map_u8: torch.Tensor) -> Dict[int, int]:
    uniq, cnt = torch.unique(map_u8, return_counts=True)
    idcnt: Dict[int, int] = {
        int(k): int(v)
        for k, v in zip(
            uniq.to(dtype=torch.int64).tolist(), cnt.to(dtype=torch.int64).tolist()
        )
    }
    if idcnt:
        idcnt.pop(min(idcnt.keys()), None)
    return idcnt


class SparseCoordInit:
    """
    Sparse coordinate initializer.
    """

    def __init__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        config: SparseInitConfig | None = None,
    ) -> None:
        if pred.dim() != 4 or gt.dim() != 4:
            raise ValueError(
                f"Expect pred/gt [B, C, H, W], got {tuple(pred.shape)} and {tuple(gt.shape)}"
            )
        if pred.shape != gt.shape:
            raise ValueError(
                f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}"
            )
        if pred.shape[0] != 1:
            raise ValueError(
                f"SparseCoordInit expects batch size 1, got {pred.shape[0]}"
            )

        cfg = SparseInitConfig() if config is None else config
        if cfg.connectivity != 4:
            raise ValueError("only 4-connectivity is currently supported")

        self.config = cfg
        self.h = int(pred.shape[2])
        self.w = int(pred.shape[3])

        self.map = _build_quantized_error_map_torch(
            pred,
            gt,
            quantile_interval=int(cfg.quantile_interval),
            nodiff_thres=float(cfg.nodiff_thres),
        )
        self.idcnt = _build_idcnt_from_map_torch(self.map)

    def _random_coord(self) -> list[float]:
        rc = torch.rand((2,), dtype=torch.float32, device=self.map.device)
        return [
            float(rc[0].item() * float(self.h)),
            float(rc[1].item() * float(self.w)),
        ]

    def _select_target_id(self) -> int | None:
        if not self.idcnt:
            return None
        return int(max(self.idcnt, key=lambda k: int(k) * int(self.idcnt[k])))

    def _consume_target(self, target_id: int, removed: int) -> None:
        remain = int(self.idcnt.get(int(target_id), 0)) - int(max(0, removed))
        if remain <= 0:
            self.idcnt.pop(int(target_id), None)
        else:
            self.idcnt[int(target_id)] = int(remain)

    def _call_connected_components(self, target_id: int) -> list[float]:
        tol = int(self.config.tolerance)
        mask = (
            (self.map >= int(target_id - tol))
            & (self.map <= int(target_id + tol + 2))
            & (self.map > 0)
        ).to(torch.uint8)
        if not bool(mask.any().item()):
            return self._random_coord()

        _, labels, stats, centers = connected_components_with_stats_4_tilelang(
            mask,
            max_iters=int(self.config.max_iters),
        )

        if stats.shape[0] <= 0:
            return self._random_coord()

        row, col, target_label, target_area, component_mask = (
            select_largest_component_center(
                labels,
                stats,
                centers,
            )
        )
        self._consume_target(target_id, int(target_area))
        self.map[component_mask] = 0
        return [float(row), float(col)]

    def __call__(self) -> list[float]:
        target_id = self._select_target_id()
        if target_id is None:
            return self._random_coord()
        return self._call_connected_components(int(target_id))


__all__ = [
    "SparseCoordInit",
    "SparseInitConfig",
]
