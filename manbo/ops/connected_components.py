from __future__ import annotations

from typing import Tuple

import tilelang
import tilelang.language as T
import torch

DEFAULT_THREADS = 256
DEFAULT_MAX_ITERS = 4096


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def cc_init_labels_kernel(
    threads: int = DEFAULT_THREADS,
    numel=T.dynamic("numel"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        mask: T.Tensor[[numel], T.uint8],
        labels: T.Tensor[[numel], T.int32],
    ):
        grid = T.ceildiv(numel, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                idx = bx * threads + tx
                if idx < numel:
                    T.assume(idx >= 0)
                    labels[idx] = T.if_then_else(
                        mask[idx] != T.uint8(0), idx + 1, T.int32(0)
                    )

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def cc_relax_4_kernel(
    threads: int = DEFAULT_THREADS,
    numel=T.dynamic("numel"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        labels_in: T.Tensor[[numel], T.int32],
        mask: T.Tensor[[numel], T.uint8],
        labels_out: T.Tensor[[numel], T.int32],
        changed: T.Tensor[[1], T.int32],
        height: T.int32,
        width: T.int32,
    ):
        grid = T.ceildiv(numel, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                idx = bx * threads + tx
                if idx < numel:
                    T.assume(idx >= 0)
                    if mask[idx] == T.uint8(0):
                        labels_out[idx] = T.int32(0)
                    else:
                        row = idx // width
                        col = idx - row * width
                        best = T.alloc_var("int32")
                        best = labels_in[idx]

                        if row > 0:
                            up_idx = idx - width
                            up_label = labels_in[up_idx]
                            if up_label > T.int32(0):
                                best = T.min(best, up_label)
                        if row + 1 < height:
                            down_idx = idx + width
                            down_label = labels_in[down_idx]
                            if down_label > T.int32(0):
                                best = T.min(best, down_label)
                        if col > 0:
                            left_idx = idx - 1
                            left_label = labels_in[left_idx]
                            if left_label > T.int32(0):
                                best = T.min(best, left_label)
                        if col + 1 < width:
                            right_idx = idx + 1
                            right_label = labels_in[right_idx]
                            if right_label > T.int32(0):
                                best = T.min(best, right_label)

                        labels_out[idx] = best
                        if best != labels_in[idx]:
                            T.atomic_max(changed[0], T.int32(1))

    return kernel


def connected_components_labels_4_tilelang(
    mask: torch.Tensor,
    *,
    max_iters: int = DEFAULT_MAX_ITERS,
    threads: int = DEFAULT_THREADS,
) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H, W], got {tuple(mask.shape)}")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")

    mask_u8 = mask.to(dtype=torch.uint8).contiguous()
    h, w = mask_u8.shape
    numel = int(mask_u8.numel())
    mask_flat = mask_u8.view(-1)

    init_kernel = cc_init_labels_kernel(threads=int(threads))
    relax_kernel = cc_relax_4_kernel(threads=int(threads))

    labels_a = torch.empty((numel,), dtype=torch.int32, device=mask_u8.device)
    labels_b = torch.empty_like(labels_a)
    init_kernel(mask_flat, labels_a)

    for _ in range(int(max_iters)):
        changed = torch.zeros((1,), dtype=torch.int32, device=mask_u8.device)
        relax_kernel(labels_a, mask_flat, labels_b, changed, int(h), int(w))
        if int(changed.item()) == 0:
            labels_a = labels_b
            break
        labels_a, labels_b = labels_b, labels_a
    else:
        raise RuntimeError(f"CCL did not converge within max_iters={max_iters}")

    return labels_a.view(h, w)


def connected_components_with_stats_4_tilelang(
    mask: torch.Tensor,
    *,
    max_iters: int = DEFAULT_MAX_ITERS,
    threads: int = DEFAULT_THREADS,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns `(num_labels, labels, stats, centers)` following OpenCV layout:
    - `num_labels`: includes background.
    - `labels`: [H, W] int32 with compact ids in [0, num_labels-1].
    - `stats`: [num_labels-1, 5] int32 columns [x, y, width, height, area].
    - `centers`: [num_labels-1, 2] float32 columns [cx, cy].
    """

    labels_raw = connected_components_labels_4_tilelang(
        mask,
        max_iters=int(max_iters),
        threads=int(threads),
    )
    h, w = labels_raw.shape
    flat_labels = labels_raw.reshape(-1)
    max_label = int(flat_labels.max().item())
    if max_label <= 0:
        empty_stats = torch.empty((0, 5), dtype=torch.int32, device=labels_raw.device)
        empty_centers = torch.empty(
            (0, 2), dtype=torch.float32, device=labels_raw.device
        )
        return 1, torch.zeros_like(labels_raw), empty_stats, empty_centers

    valid = flat_labels > 0
    if not bool(valid.any().item()):
        empty_stats = torch.empty((0, 5), dtype=torch.int32, device=labels_raw.device)
        empty_centers = torch.empty(
            (0, 2), dtype=torch.float32, device=labels_raw.device
        )
        return 1, torch.zeros_like(labels_raw), empty_stats, empty_centers

    ids = flat_labels[valid].to(torch.int64)
    ys = (
        torch.arange(h, dtype=torch.int32, device=labels_raw.device)
        .view(h, 1)
        .expand(h, w)
        .reshape(-1)[valid]
    )
    xs = (
        torch.arange(w, dtype=torch.int32, device=labels_raw.device)
        .view(1, w)
        .expand(h, w)
        .reshape(-1)[valid]
    )
    area = torch.bincount(ids, minlength=max_label + 1).to(torch.int32)
    sum_x = torch.zeros((max_label + 1,), dtype=torch.float32, device=labels_raw.device)
    sum_y = torch.zeros((max_label + 1,), dtype=torch.float32, device=labels_raw.device)
    sum_x.scatter_add_(0, ids, xs.to(torch.float32))
    sum_y.scatter_add_(0, ids, ys.to(torch.float32))
    min_x = torch.full(
        (max_label + 1,), int(w), dtype=torch.int32, device=labels_raw.device
    )
    min_y = torch.full(
        (max_label + 1,), int(h), dtype=torch.int32, device=labels_raw.device
    )
    max_x = torch.full(
        (max_label + 1,), -1, dtype=torch.int32, device=labels_raw.device
    )
    max_y = torch.full(
        (max_label + 1,), -1, dtype=torch.int32, device=labels_raw.device
    )
    min_x.scatter_reduce_(0, ids, xs, reduce="amin", include_self=True)
    min_y.scatter_reduce_(0, ids, ys, reduce="amin", include_self=True)
    max_x.scatter_reduce_(0, ids, xs, reduce="amax", include_self=True)
    max_y.scatter_reduce_(0, ids, ys, reduce="amax", include_self=True)

    valid_labels = torch.nonzero(area > 0, as_tuple=False).view(-1)
    valid_labels = valid_labels[valid_labels > 0]
    if valid_labels.numel() == 0:
        empty_stats = torch.empty((0, 5), dtype=torch.int32, device=labels_raw.device)
        empty_centers = torch.empty(
            (0, 2), dtype=torch.float32, device=labels_raw.device
        )
        return 1, torch.zeros_like(labels_raw), empty_stats, empty_centers

    mapping = torch.zeros((max_label + 1,), dtype=torch.int32, device=labels_raw.device)
    mapping[valid_labels] = torch.arange(
        1,
        int(valid_labels.numel()) + 1,
        dtype=torch.int32,
        device=labels_raw.device,
    )
    labels_compact = mapping[labels_raw]

    vx = min_x[valid_labels]
    vy = min_y[valid_labels]
    widths = max_x[valid_labels] - vx + 1
    heights = max_y[valid_labels] - vy + 1
    areas = area[valid_labels]
    stats = torch.stack([vx, vy, widths, heights, areas], dim=1).to(torch.int32)

    denom = areas.to(torch.float32).clamp_min(1.0)
    centers = torch.stack(
        [sum_x[valid_labels] / denom, sum_y[valid_labels] / denom], dim=1
    ).to(torch.float32)

    return int(valid_labels.numel()) + 1, labels_compact, stats, centers


def select_largest_component_center(
    labels: torch.Tensor,
    stats: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[int, int, int, int, torch.Tensor]:
    if labels.ndim != 2:
        raise ValueError(f"labels must be [H, W], got {tuple(labels.shape)}")
    if stats.ndim != 2 or stats.shape[-1] != 5:
        raise ValueError(f"stats must be [N, 5], got {tuple(stats.shape)}")
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError(f"centers must be [N, 2], got {tuple(centers.shape)}")
    if stats.shape[0] <= 0:
        raise ValueError("stats must contain at least one foreground component")

    areas = stats[:, 4]
    target_idx = int(torch.argmax(areas).item())
    target_label = target_idx + 1
    target_area = int(areas[target_idx].item())
    component_mask = labels == int(target_label)
    coords = torch.nonzero(component_mask, as_tuple=False)
    if coords.numel() <= 0:
        raise RuntimeError("selected connected component has zero pixels")

    w = int(labels.shape[1])
    linear = coords[:, 0].to(torch.int64) * int(w) + coords[:, 1].to(torch.int64)
    coords = coords[torch.argsort(linear)]

    center_rc = torch.stack([centers[target_idx, 1], centers[target_idx, 0]], dim=0).to(
        torch.float32
    )
    dists = torch.norm(coords.to(torch.float32) - center_rc.unsqueeze(0), dim=1)
    best_idx = int(torch.argmin(dists).item())
    row = int(coords[best_idx, 0].item())
    col = int(coords[best_idx, 1].item())
    return row, col, int(target_label), target_area, component_mask


__all__ = [
    "DEFAULT_MAX_ITERS",
    "DEFAULT_THREADS",
    "cc_init_labels_kernel",
    "cc_relax_4_kernel",
    "connected_components_labels_4_tilelang",
    "connected_components_with_stats_4_tilelang",
    "select_largest_component_center",
]
