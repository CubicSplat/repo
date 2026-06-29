from __future__ import annotations

import math

import tilelang
import tilelang.language as T
import torch

from .metric import reduce_sum_1d_tilelang
from .utils import get_warp_size

DEFAULT_REG_THREADS = 256
DEFAULT_REG_ITEMS_PER_THREAD = 8
CURVATURE_ROLL_OFFSET = 5


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _as_cuda_float32_contiguous(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()
    return x


@T.macro
def _shape_proj_penalty_alpha(
    p0_x: T.float32,
    p0_y: T.float32,
    pend_x: T.float32,
    pend_y: T.float32,
    mid_x: T.float32,
    mid_y: T.float32,
    one: T.float32,
    zero: T.float32,
    eps: T.float32,
    out: T.Tensor,
):
    v_x = pend_x - p0_x
    v_y = pend_y - p0_y
    u_x = mid_x - p0_x
    u_y = mid_y - p0_y
    denom = v_x * v_x + v_y * v_y + eps
    alpha = (u_x * v_x + u_y * v_y) / denom
    pos = T.max(alpha - one, zero)
    neg = T.max(-alpha, zero)
    out[0] = alpha
    out[1] = pos * pos + neg * neg


@T.macro
def _shape_proj_backward_terms(
    p0_x: T.float32,
    p0_y: T.float32,
    pend_x: T.float32,
    pend_y: T.float32,
    mid_x: T.float32,
    mid_y: T.float32,
    one: T.float32,
    zero: T.float32,
    two: T.float32,
    eps: T.float32,
    scale: T.float32,
    out: T.Tensor,
):
    v_x = pend_x - p0_x
    v_y = pend_y - p0_y
    u_x = mid_x - p0_x
    u_y = mid_y - p0_y
    denom = v_x * v_x + v_y * v_y + eps
    inv_denom = one / denom
    alpha = (u_x * v_x + u_y * v_y) * inv_denom
    d_alpha = T.if_then_else(
        alpha > one,
        two * (alpha - one),
        T.if_then_else(alpha < zero, two * alpha, zero),
    )
    d_alpha_scaled = d_alpha * scale

    out[0] = d_alpha_scaled * v_x * inv_denom
    out[1] = d_alpha_scaled * v_y * inv_denom
    out[2] = d_alpha_scaled * (u_x * inv_denom - two * alpha * v_x * inv_denom)
    out[3] = d_alpha_scaled * (u_y * inv_denom - two * alpha * v_y * inv_denom)


@T.macro
def _reduce_block_sum_to_partial(
    acc: T.float32,
    warp_partial: T.Tensor,
    lane_id: T.int32,
    warp_id: T.int32,
    warp_num: int,
    bx: T.int32,
    zero: T.float32,
    partial: T.Tensor,
):
    warp_acc = T.warp_reduce_sum(acc)
    if lane_id == 0:
        warp_partial[warp_id, 0] = warp_acc

    T.sync_threads()

    if warp_id == 0:
        block_acc = T.alloc_var("float32")
        block_acc = zero
        if lane_id < warp_num:
            block_acc = warp_partial[lane_id, 0]
        block_acc = T.warp_reduce_sum(block_acc)
        if lane_id == 0:
            partial[bx] = block_acc


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def bezier_shape_partial_sum_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    num_curves=T.dynamic("num_curves"),
    num_ctrl=T.dynamic("num_ctrl"),
    num_blocks=T.dynamic("num_blocks"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"bezier_shape_partial_sum_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    warp_num = threads // warp_size
    warp_stride = warp_size + 1
    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        control_points: T.Tensor[[num_curves, num_ctrl, 2], dtype],
        split_idx: T.int32,
        partial: T.Tensor[[num_blocks], "float32"],
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            T.annotate_safe_value({control_points: T.Cast(dtype, 0.0)})
            warp_partial = T.alloc_shared((warp_num, warp_stride), "float32")
            tx = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            base = bx * block_span + tx
            mid_count = num_ctrl - 2
            num_terms = num_curves * mid_count
            T.assume(mid_count > 0)
            T.assume(num_terms >= 0)
            T.assume(split_idx > 0)
            T.assume(split_idx < num_ctrl - 1)
            one = T.Cast("float32", 1.0)
            zero = T.Cast("float32", 0.0)
            eps = T.Cast("float32", 1e-8)

            acc = T.alloc_var("float32")
            acc = zero

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < num_terms:
                    curve_id = idx // mid_count
                    mid_id = idx - curve_id * mid_count
                    cp_idx = mid_id + 1
                    cp_idx_adj = T.if_then_else(cp_idx >= split_idx, cp_idx + 1, cp_idx)
                    T.assume((curve_id >= 0) & (curve_id < num_curves))
                    T.assume((cp_idx_adj >= 1) & (cp_idx_adj < num_ctrl))

                    p0_x = T.Cast("float32", control_points[curve_id, 0, 0])
                    p0_y = T.Cast("float32", control_points[curve_id, 0, 1])
                    pend_x = T.Cast("float32", control_points[curve_id, split_idx, 0])
                    pend_y = T.Cast("float32", control_points[curve_id, split_idx, 1])
                    mid_x = T.Cast("float32", control_points[curve_id, cp_idx_adj, 0])
                    mid_y = T.Cast("float32", control_points[curve_id, cp_idx_adj, 1])

                    proj_terms = T.alloc_local((2,), "float32")
                    _shape_proj_penalty_alpha(
                        p0_x,
                        p0_y,
                        pend_x,
                        pend_y,
                        mid_x,
                        mid_y,
                        one,
                        zero,
                        eps,
                        proj_terms,
                    )
                    acc += proj_terms[1]

            _reduce_block_sum_to_partial(
                acc, warp_partial, lane_id, warp_id, warp_num, bx, zero, partial
            )

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def bezier_open_partial_sum_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    num_curves=T.dynamic("num_curves"),
    num_ctrl=T.dynamic("num_ctrl"),
    num_blocks=T.dynamic("num_blocks"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"bezier_open_partial_sum_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    warp_num = threads // warp_size
    warp_stride = warp_size + 1
    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        control_points: T.Tensor[[num_curves, num_ctrl, 2], dtype],
        partial: T.Tensor[[num_blocks], "float32"],
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            T.annotate_safe_value({control_points: T.Cast(dtype, 0.0)})
            warp_partial = T.alloc_shared((warp_num, warp_stride), "float32")
            tx = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            base = bx * block_span + tx
            mid_count = num_ctrl - 2
            num_terms = num_curves * mid_count
            pend_idx = num_ctrl - 1
            T.assume(mid_count > 0)
            T.assume(num_terms >= 0)
            T.assume(pend_idx > 0)
            T.assume(pend_idx < num_ctrl)
            one = T.Cast("float32", 1.0)
            zero = T.Cast("float32", 0.0)
            eps = T.Cast("float32", 1e-8)

            acc = T.alloc_var("float32")
            acc = zero

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < num_terms:
                    curve_id = idx // mid_count
                    mid_id = idx - curve_id * mid_count
                    cp_idx = mid_id + 1
                    T.assume((curve_id >= 0) & (curve_id < num_curves))
                    T.assume((cp_idx >= 1) & (cp_idx < num_ctrl - 1))

                    p0_x = T.Cast("float32", control_points[curve_id, 0, 0])
                    p0_y = T.Cast("float32", control_points[curve_id, 0, 1])
                    pend_x = T.Cast("float32", control_points[curve_id, pend_idx, 0])
                    pend_y = T.Cast("float32", control_points[curve_id, pend_idx, 1])
                    mid_x = T.Cast("float32", control_points[curve_id, cp_idx, 0])
                    mid_y = T.Cast("float32", control_points[curve_id, cp_idx, 1])

                    proj_terms = T.alloc_local((2,), "float32")
                    _shape_proj_penalty_alpha(
                        p0_x,
                        p0_y,
                        pend_x,
                        pend_y,
                        mid_x,
                        mid_y,
                        one,
                        zero,
                        eps,
                        proj_terms,
                    )
                    acc += proj_terms[1]

            _reduce_block_sum_to_partial(
                acc, warp_partial, lane_id, warp_id, warp_num, bx, zero, partial
            )

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def opacity_partial_sum_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    numel=T.dynamic("numel"),
    num_blocks=T.dynamic("num_blocks"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"opacity_partial_sum_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    warp_num = threads // warp_size
    warp_stride = warp_size + 1
    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        opacity: T.Tensor[[numel], dtype],
        partial: T.Tensor[[num_blocks], "float32"],
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            T.annotate_safe_value({opacity: T.Cast(dtype, 0.0)})
            warp_partial = T.alloc_shared((warp_num, warp_stride), "float32")
            tx = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            base = bx * block_span + tx
            T.assume(numel >= 0)
            one = T.Cast("float32", 1.0)
            zero = T.Cast("float32", 0.0)

            acc = T.alloc_var("float32")
            acc = zero

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < numel:
                    T.assume((idx >= 0) & (idx < numel))
                    x = T.Cast("float32", opacity[idx])
                    sig = one / (one + T.exp(-x))
                    delta = sig - one
                    acc += T.max(delta, -delta)

            _reduce_block_sum_to_partial(
                acc, warp_partial, lane_id, warp_id, warp_num, bx, zero, partial
            )

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def boundary_partial_sum_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    num_curves=T.dynamic("num_curves"),
    num_joints=T.dynamic("num_joints"),
    num_blocks=T.dynamic("num_blocks"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"boundary_partial_sum_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    warp_num = threads // warp_size
    warp_stride = warp_size + 1
    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        joints: T.Tensor[[num_curves, num_joints, 2], dtype],
        bound: T.float32,
        partial: T.Tensor[[num_blocks], "float32"],
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            T.annotate_safe_value({joints: T.Cast(dtype, 0.0)})
            warp_partial = T.alloc_shared((warp_num, warp_stride), "float32")
            tx = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            base = bx * block_span + tx
            num_terms = num_curves * num_joints * 2
            T.assume(num_terms >= 0)
            zero = T.Cast("float32", 0.0)

            acc = T.alloc_var("float32")
            acc = zero

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < num_terms:
                    curve_id = idx // (num_joints * 2)
                    rem = idx - curve_id * (num_joints * 2)
                    joint_id = rem // 2
                    comp = rem - joint_id * 2
                    T.assume((curve_id >= 0) & (curve_id < num_curves))
                    T.assume((joint_id >= 0) & (joint_id < num_joints))
                    x = T.Cast("float32", joints[curve_id, joint_id, comp])
                    over = T.max(x - bound, zero)
                    under = T.max(-bound - x, zero)
                    acc += over + under

            _reduce_block_sum_to_partial(
                acc, warp_partial, lane_id, warp_id, warp_num, bx, zero, partial
            )

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def curvature_partial_sum_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    num_curves=T.dynamic("num_curves"),
    num_samples=T.dynamic("num_samples"),
    num_blocks=T.dynamic("num_blocks"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"curvature_partial_sum_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    warp_num = threads // warp_size
    warp_stride = warp_size + 1
    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        prev_pts: T.Tensor[[num_curves, num_samples, 2], dtype],
        curr_pts: T.Tensor[[num_curves, num_samples, 2], dtype],
        next_pts: T.Tensor[[num_curves, num_samples, 2], dtype],
        cos_thresh: T.float32,
        partial: T.Tensor[[num_blocks], "float32"],
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            T.annotate_safe_value(
                {
                    prev_pts: T.Cast(dtype, 0.0),
                    curr_pts: T.Cast(dtype, 0.0),
                    next_pts: T.Cast(dtype, 0.0),
                }
            )
            warp_partial = T.alloc_shared((warp_num, warp_stride), "float32")
            tx = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            base = bx * block_span + tx
            num_terms = num_curves * num_samples
            T.assume(num_terms >= 0)
            zero = T.Cast("float32", 0.0)
            one = T.Cast("float32", 1.0)
            n_one = T.Cast("float32", -1.0)
            two = T.Cast("float32", 2.0)
            eps = T.Cast("float32", 1e-8)

            acc = T.alloc_var("float32")
            acc = zero

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < num_terms:
                    curve_id = idx // num_samples
                    sample_id = idx - curve_id * num_samples
                    T.assume((curve_id >= 0) & (curve_id < num_curves))
                    T.assume((sample_id >= 0) & (sample_id < num_samples))

                    prev_x = T.Cast("float32", prev_pts[curve_id, sample_id, 0])
                    prev_y = T.Cast("float32", prev_pts[curve_id, sample_id, 1])
                    curr_x = T.Cast("float32", curr_pts[curve_id, sample_id, 0])
                    curr_y = T.Cast("float32", curr_pts[curve_id, sample_id, 1])
                    next_x = T.Cast("float32", next_pts[curve_id, sample_id, 0])
                    next_y = T.Cast("float32", next_pts[curve_id, sample_id, 1])

                    sec_x = prev_x - two * curr_x + next_x
                    sec_y = prev_y - two * curr_y + next_y
                    curvature = sec_x * sec_x + sec_y * sec_y

                    v1_x = prev_x - curr_x
                    v1_y = prev_y - curr_y
                    v2_x = next_x - curr_x
                    v2_y = next_y - curr_y

                    n1 = T.sqrt(v1_x * v1_x + v1_y * v1_y + eps)
                    n2 = T.sqrt(v2_x * v2_x + v2_y * v2_y + eps)
                    dot_raw = (v1_x / n1) * (v2_x / n2) + (v1_y / n1) * (v2_y / n2)
                    dot_clamped = T.min(one, T.max(n_one, dot_raw))

                    masked = T.if_then_else(dot_clamped > cos_thresh, curvature, zero)
                    acc += masked

            _reduce_block_sum_to_partial(
                acc, warp_partial, lane_id, warp_id, warp_num, bx, zero, partial
            )

    return kernel


def bezier_shape_proj_outside_sum_tilelang(
    control_points: torch.Tensor,
    *,
    split_idx: int,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> torch.Tensor:
    cp = _as_cuda_float32_contiguous(control_points, "control_points")
    if cp.dim() != 3 or cp.shape[-1] != 2:
        raise ValueError(
            f"control_points must have shape [N, K, 2], got {tuple(cp.shape)}"
        )

    num_curves, num_ctrl, _ = cp.shape
    if num_curves == 0 or num_ctrl <= 2:
        return cp.new_tensor(0.0, dtype=torch.float32)
    if int(split_idx) <= 0 or int(split_idx) >= int(num_ctrl - 1):
        raise ValueError(f"split_idx must be in [1, {num_ctrl - 2}], got {split_idx}")

    block_span = int(threads) * int(items_per_thread)
    num_terms = int(num_curves) * int(num_ctrl - 2)
    num_blocks = _ceildiv(num_terms, block_span)
    partial = torch.empty((num_blocks,), dtype=torch.float32, device=cp.device)

    kernel = bezier_shape_partial_sum_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    kernel(cp, int(split_idx), partial)
    return reduce_sum_1d_tilelang(
        partial, threads=int(threads), items_per_thread=int(items_per_thread)
    )


def bezier_open_proj_outside_sum_tilelang(
    control_points: torch.Tensor,
    *,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> torch.Tensor:
    cp = _as_cuda_float32_contiguous(control_points, "control_points")
    if cp.dim() != 3 or cp.shape[-1] != 2:
        raise ValueError(
            f"control_points must have shape [N, K, 2], got {tuple(cp.shape)}"
        )

    num_curves, num_ctrl, _ = cp.shape
    if num_curves == 0 or num_ctrl <= 2:
        return cp.new_tensor(0.0, dtype=torch.float32)

    block_span = int(threads) * int(items_per_thread)
    num_terms = int(num_curves) * int(num_ctrl - 2)
    num_blocks = _ceildiv(num_terms, block_span)
    partial = torch.empty((num_blocks,), dtype=torch.float32, device=cp.device)

    kernel = bezier_open_partial_sum_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    kernel(cp, partial)
    return reduce_sum_1d_tilelang(
        partial, threads=int(threads), items_per_thread=int(items_per_thread)
    )


def opacity_abs_sigmoid_delta_sum_tilelang(
    opacity: torch.Tensor,
    *,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> torch.Tensor:
    op = _as_cuda_float32_contiguous(opacity, "opacity").view(-1)
    if op.numel() == 0:
        return op.new_tensor(0.0, dtype=torch.float32)

    block_span = int(threads) * int(items_per_thread)
    num_blocks = _ceildiv(int(op.numel()), block_span)
    partial = torch.empty((num_blocks,), dtype=torch.float32, device=op.device)

    kernel = opacity_partial_sum_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    kernel(op, partial)
    return reduce_sum_1d_tilelang(
        partial, threads=int(threads), items_per_thread=int(items_per_thread)
    )


def boundary_joints_penalty_sum_tilelang(
    points: torch.Tensor,
    *,
    degree: int,
    bound: float = 1.0,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> torch.Tensor:
    pts = _as_cuda_float32_contiguous(points, "points")
    if pts.dim() != 3 or pts.shape[-1] != 2:
        raise ValueError(f"points must have shape [N, K, 2], got {tuple(pts.shape)}")
    if int(degree) <= 0:
        raise ValueError("degree must be > 0")

    joint_indices = torch.arange(0, pts.shape[1], int(degree), device=pts.device)
    joints = pts.index_select(dim=1, index=joint_indices).contiguous()
    if joints.numel() == 0:
        return pts.new_tensor(0.0, dtype=torch.float32)

    num_curves, num_joints, _ = joints.shape
    block_span = int(threads) * int(items_per_thread)
    num_terms = int(num_curves) * int(num_joints) * 2
    num_blocks = _ceildiv(num_terms, block_span)
    partial = torch.empty((num_blocks,), dtype=torch.float32, device=pts.device)

    kernel = boundary_partial_sum_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    kernel(joints, float(bound), partial)
    return reduce_sum_1d_tilelang(
        partial, threads=int(threads), items_per_thread=int(items_per_thread)
    )


def curvature_masked_second_diff_sum_tilelang(
    paths: torch.Tensor,
    *,
    stride: int,
    angle_thresh_deg: float = 60.0,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> tuple[torch.Tensor, int]:
    x = _as_cuda_float32_contiguous(paths, "paths")
    if x.dim() != 4 or x.shape[1] != 2 or x.shape[-1] != 2:
        raise ValueError(f"paths must have shape [N, 2, M, 2], got {tuple(x.shape)}")
    if int(stride) <= 0:
        raise ValueError("stride must be > 0")

    path1 = x[:, 0, :, :]
    path2 = torch.flip(x[:, 1, :, :], dims=[1])
    full_path = torch.cat([path1, path2], dim=1)

    total_len = int(full_path.shape[1])
    indices = torch.arange(0, total_len, int(stride), device=x.device)
    if indices.numel() == 0:
        return x.new_tensor(0.0, dtype=torch.float32), 0

    prev = (
        torch.roll(full_path, CURVATURE_ROLL_OFFSET, dims=1)
        .index_select(1, indices)
        .contiguous()
    )
    curr = full_path.index_select(1, indices).contiguous()
    nex = (
        torch.roll(full_path, -CURVATURE_ROLL_OFFSET, dims=1)
        .index_select(1, indices)
        .contiguous()
    )

    num_curves, num_samples, _ = prev.shape
    block_span = int(threads) * int(items_per_thread)
    num_terms = int(num_curves) * int(num_samples)
    num_blocks = _ceildiv(num_terms, block_span)
    partial = torch.empty((num_blocks,), dtype=torch.float32, device=x.device)

    kernel = curvature_partial_sum_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    cos_thresh = float(math.cos(math.radians(float(angle_thresh_deg))))
    kernel(prev, curr, nex, cos_thresh, partial)

    s = reduce_sum_1d_tilelang(
        partial, threads=int(threads), items_per_thread=int(items_per_thread)
    )
    return s, int(num_samples)


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def bezier_shape_backward_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    num_curves=T.dynamic("num_curves"),
    num_ctrl=T.dynamic("num_ctrl"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"bezier_shape_backward_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        control_points: T.Tensor[[num_curves, num_ctrl, 2], dtype],
        split_idx: T.int32,
        scale: T.float32,
        grad: T.Tensor[[num_curves, num_ctrl, 2], "float32"],
    ):
        mid_count = num_ctrl - 2
        num_terms = num_curves * mid_count
        T.assume(mid_count > 0)
        T.assume(num_terms >= 0)
        T.assume(split_idx > 0)
        T.assume(split_idx < num_ctrl - 1)
        with T.Kernel(T.ceildiv(num_terms, block_span), threads=threads) as bx:
            T.annotate_safe_value({control_points: T.Cast(dtype, 0.0)})
            tx = T.get_thread_binding()
            base = bx * block_span + tx
            zero = T.Cast("float32", 0.0)
            one = T.Cast("float32", 1.0)
            two = T.Cast("float32", 2.0)
            eps = T.Cast("float32", 1e-8)

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < num_terms:
                    curve_id = idx // mid_count
                    mid_id = idx - curve_id * mid_count
                    cp_idx = mid_id + 1
                    cp_idx_adj = T.if_then_else(cp_idx >= split_idx, cp_idx + 1, cp_idx)
                    T.assume((curve_id >= 0) & (curve_id < num_curves))
                    T.assume((cp_idx_adj >= 1) & (cp_idx_adj < num_ctrl))

                    p0_x = T.Cast("float32", control_points[curve_id, 0, 0])
                    p0_y = T.Cast("float32", control_points[curve_id, 0, 1])
                    pend_x = T.Cast("float32", control_points[curve_id, split_idx, 0])
                    pend_y = T.Cast("float32", control_points[curve_id, split_idx, 1])
                    mid_x = T.Cast("float32", control_points[curve_id, cp_idx_adj, 0])
                    mid_y = T.Cast("float32", control_points[curve_id, cp_idx_adj, 1])

                    grad_terms = T.alloc_local((4,), "float32")
                    _shape_proj_backward_terms(
                        p0_x,
                        p0_y,
                        pend_x,
                        pend_y,
                        mid_x,
                        mid_y,
                        one,
                        zero,
                        two,
                        eps,
                        scale,
                        grad_terms,
                    )

                    T.atomic_add(grad[curve_id, 0, 0], -(grad_terms[0] + grad_terms[2]))
                    T.atomic_add(grad[curve_id, 0, 1], -(grad_terms[1] + grad_terms[3]))
                    T.atomic_add(grad[curve_id, split_idx, 0], grad_terms[2])
                    T.atomic_add(grad[curve_id, split_idx, 1], grad_terms[3])
                    T.atomic_add(grad[curve_id, cp_idx_adj, 0], grad_terms[0])
                    T.atomic_add(grad[curve_id, cp_idx_adj, 1], grad_terms[1])

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def bezier_open_backward_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    num_curves=T.dynamic("num_curves"),
    num_ctrl=T.dynamic("num_ctrl"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"bezier_open_backward_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        control_points: T.Tensor[[num_curves, num_ctrl, 2], dtype],
        scale: T.float32,
        grad: T.Tensor[[num_curves, num_ctrl, 2], "float32"],
    ):
        mid_count = num_ctrl - 2
        num_terms = num_curves * mid_count
        T.assume(mid_count > 0)
        T.assume(num_terms >= 0)
        with T.Kernel(T.ceildiv(num_terms, block_span), threads=threads) as bx:
            T.annotate_safe_value({control_points: T.Cast(dtype, 0.0)})
            tx = T.get_thread_binding()
            base = bx * block_span + tx
            pend_idx = num_ctrl - 1
            T.assume(pend_idx > 0)
            T.assume(pend_idx < num_ctrl)
            zero = T.Cast("float32", 0.0)
            one = T.Cast("float32", 1.0)
            two = T.Cast("float32", 2.0)
            eps = T.Cast("float32", 1e-8)

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < num_terms:
                    curve_id = idx // mid_count
                    mid_id = idx - curve_id * mid_count
                    cp_idx = mid_id + 1
                    T.assume((curve_id >= 0) & (curve_id < num_curves))
                    T.assume((cp_idx >= 1) & (cp_idx < num_ctrl - 1))

                    p0_x = T.Cast("float32", control_points[curve_id, 0, 0])
                    p0_y = T.Cast("float32", control_points[curve_id, 0, 1])
                    pend_x = T.Cast("float32", control_points[curve_id, pend_idx, 0])
                    pend_y = T.Cast("float32", control_points[curve_id, pend_idx, 1])
                    mid_x = T.Cast("float32", control_points[curve_id, cp_idx, 0])
                    mid_y = T.Cast("float32", control_points[curve_id, cp_idx, 1])

                    grad_terms = T.alloc_local((4,), "float32")
                    _shape_proj_backward_terms(
                        p0_x,
                        p0_y,
                        pend_x,
                        pend_y,
                        mid_x,
                        mid_y,
                        one,
                        zero,
                        two,
                        eps,
                        scale,
                        grad_terms,
                    )

                    T.atomic_add(grad[curve_id, 0, 0], -(grad_terms[0] + grad_terms[2]))
                    T.atomic_add(grad[curve_id, 0, 1], -(grad_terms[1] + grad_terms[3]))
                    T.atomic_add(grad[curve_id, pend_idx, 0], grad_terms[2])
                    T.atomic_add(grad[curve_id, pend_idx, 1], grad_terms[3])
                    T.atomic_add(grad[curve_id, cp_idx, 0], grad_terms[0])
                    T.atomic_add(grad[curve_id, cp_idx, 1], grad_terms[1])

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def opacity_backward_kernel(
    threads: int = DEFAULT_REG_THREADS,
    dtype: str = "float32",
    numel=T.dynamic("numel"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        opacity: T.Tensor[[numel], dtype],
        scale: T.float32,
        grad: T.Tensor[[numel], "float32"],
    ):
        T.assume(numel >= 0)
        with T.Kernel(T.ceildiv(numel, threads), threads=threads) as bx:
            T.annotate_safe_value({opacity: T.Cast(dtype, 0.0)})
            tx = T.get_thread_binding()
            idx = bx * threads + tx
            if idx < numel:
                T.assume((idx >= 0) & (idx < numel))
                zero = T.Cast("float32", 0.0)
                one = T.Cast("float32", 1.0)
                n_one = T.Cast("float32", -1.0)
                x = T.Cast("float32", opacity[idx])
                sig = one / (one + T.exp(-x))
                delta = sig - one
                sign = T.if_then_else(
                    delta > zero,
                    one,
                    T.if_then_else(delta < zero, n_one, zero),
                )
                grad[idx] = scale * sign * sig * (one - sig)

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def boundary_backward_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    num_curves=T.dynamic("num_curves"),
    num_ctrl=T.dynamic("num_ctrl"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"boundary_backward_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        points: T.Tensor[[num_curves, num_ctrl, 2], dtype],
        degree: T.int32,
        bound: T.float32,
        scale: T.float32,
        grad: T.Tensor[[num_curves, num_ctrl, 2], "float32"],
    ):
        T.assume(degree > 0)
        num_joints = T.ceildiv(num_ctrl, degree)
        num_terms = num_curves * num_joints * 2
        T.assume(num_terms >= 0)
        with T.Kernel(T.ceildiv(num_terms, block_span), threads=threads) as bx:
            T.annotate_safe_value({points: T.Cast(dtype, 0.0)})
            tx = T.get_thread_binding()
            base = bx * block_span + tx
            zero = T.Cast("float32", 0.0)
            neg_scale = -scale

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < num_terms:
                    curve_id = idx // (num_joints * 2)
                    rem = idx - curve_id * (num_joints * 2)
                    joint_id = rem // 2
                    comp = rem - joint_id * 2
                    cp_idx = joint_id * degree
                    T.assume((curve_id >= 0) & (curve_id < num_curves))
                    T.assume((joint_id >= 0) & (joint_id < num_joints))
                    if cp_idx < num_ctrl:
                        x = T.Cast("float32", points[curve_id, cp_idx, comp])
                        g = T.if_then_else(
                            x > bound,
                            scale,
                            T.if_then_else(x < -bound, neg_scale, zero),
                        )
                        T.atomic_add(grad[curve_id, cp_idx, comp], g)

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def curvature_backward_kernel(
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    num_curves=T.dynamic("num_curves"),
    num_points=T.dynamic("num_points"),
):
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"curvature_backward_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        xyz: T.Tensor[[num_curves, 2, num_points, 2], dtype],
        stride: T.int32,
        cos_thresh: T.float32,
        scale: T.float32,
        grad_xyz: T.Tensor[[num_curves, 2, num_points, 2], "float32"],
    ):
        T.assume(stride > 0)
        total_len = num_points * 2
        sampled_per_curve = T.ceildiv(total_len, stride)
        num_terms = num_curves * sampled_per_curve
        T.assume(num_terms >= 0)
        with T.Kernel(T.ceildiv(num_terms, block_span), threads=threads) as bx:
            T.annotate_safe_value({xyz: T.Cast(dtype, 0.0)})
            tx = T.get_thread_binding()
            base = bx * block_span + tx
            zero = T.Cast("float32", 0.0)
            one = T.Cast("float32", 1.0)
            n_one = T.Cast("float32", -1.0)
            two = T.Cast("float32", 2.0)
            eps = T.Cast("float32", 1e-8)

            for i in T.serial(items_per_thread):
                idx = base + i * threads
                if idx < num_terms:
                    curve_id = idx // sampled_per_curve
                    sample_id = idx - curve_id * sampled_per_curve
                    full_idx = sample_id * stride
                    T.assume((curve_id >= 0) & (curve_id < num_curves))
                    T.assume((full_idx >= 0) & (full_idx < total_len))

                    prev_idx = (
                        full_idx - CURVATURE_ROLL_OFFSET + total_len
                    ) % total_len
                    next_idx = (full_idx + CURVATURE_ROLL_OFFSET) % total_len

                    curr_side = T.if_then_else(full_idx < num_points, 0, 1)
                    curr_pos = T.if_then_else(
                        full_idx < num_points, full_idx, total_len - 1 - full_idx
                    )
                    prev_side = T.if_then_else(prev_idx < num_points, 0, 1)
                    prev_pos = T.if_then_else(
                        prev_idx < num_points, prev_idx, total_len - 1 - prev_idx
                    )
                    next_side = T.if_then_else(next_idx < num_points, 0, 1)
                    next_pos = T.if_then_else(
                        next_idx < num_points, next_idx, total_len - 1 - next_idx
                    )

                    prev_x = T.Cast("float32", xyz[curve_id, prev_side, prev_pos, 0])
                    prev_y = T.Cast("float32", xyz[curve_id, prev_side, prev_pos, 1])
                    curr_x = T.Cast("float32", xyz[curve_id, curr_side, curr_pos, 0])
                    curr_y = T.Cast("float32", xyz[curve_id, curr_side, curr_pos, 1])
                    next_x = T.Cast("float32", xyz[curve_id, next_side, next_pos, 0])
                    next_y = T.Cast("float32", xyz[curve_id, next_side, next_pos, 1])

                    sec_x = prev_x - two * curr_x + next_x
                    sec_y = prev_y - two * curr_y + next_y

                    v1_x = prev_x - curr_x
                    v1_y = prev_y - curr_y
                    v2_x = next_x - curr_x
                    v2_y = next_y - curr_y
                    n1 = T.sqrt(v1_x * v1_x + v1_y * v1_y + eps)
                    n2 = T.sqrt(v2_x * v2_x + v2_y * v2_y + eps)
                    dot_raw = (v1_x / n1) * (v2_x / n2) + (v1_y / n1) * (v2_y / n2)
                    dot_clamped = T.min(one, T.max(n_one, dot_raw))
                    mask = T.if_then_else(dot_clamped > cos_thresh, one, zero)

                    d_sec_x = scale * mask * sec_x
                    d_sec_y = scale * mask * sec_y

                    T.atomic_add(grad_xyz[curve_id, prev_side, prev_pos, 0], d_sec_x)
                    T.atomic_add(grad_xyz[curve_id, prev_side, prev_pos, 1], d_sec_y)
                    T.atomic_add(
                        grad_xyz[curve_id, curr_side, curr_pos, 0], -two * d_sec_x
                    )
                    T.atomic_add(
                        grad_xyz[curve_id, curr_side, curr_pos, 1], -two * d_sec_y
                    )
                    T.atomic_add(grad_xyz[curve_id, next_side, next_pos, 0], d_sec_x)
                    T.atomic_add(grad_xyz[curve_id, next_side, next_pos, 1], d_sec_y)

    return kernel


def bezier_shape_proj_outside_grad_tilelang(
    control_points: torch.Tensor,
    *,
    split_idx: int,
    lambda_proj: float,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> torch.Tensor:
    cp = _as_cuda_float32_contiguous(control_points, "control_points")
    if cp.dim() != 3 or cp.shape[-1] != 2:
        raise ValueError(
            f"control_points must have shape [N, K, 2], got {tuple(cp.shape)}"
        )

    num_curves, num_ctrl, _ = cp.shape
    grad = torch.zeros_like(cp, dtype=torch.float32)
    if num_curves == 0 or num_ctrl <= 2:
        return grad
    if int(split_idx) <= 0 or int(split_idx) >= int(num_ctrl - 1):
        raise ValueError(f"split_idx must be in [1, {num_ctrl - 2}], got {split_idx}")

    scale = float(lambda_proj) / float(max(1, int(num_curves)))

    kernel = bezier_shape_backward_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    kernel(cp, int(split_idx), float(scale), grad)
    return grad


def bezier_open_proj_outside_grad_tilelang(
    control_points: torch.Tensor,
    *,
    lambda_proj: float,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> torch.Tensor:
    cp = _as_cuda_float32_contiguous(control_points, "control_points")
    if cp.dim() != 3 or cp.shape[-1] != 2:
        raise ValueError(
            f"control_points must have shape [N, K, 2], got {tuple(cp.shape)}"
        )

    num_curves, num_ctrl, _ = cp.shape
    grad = torch.zeros_like(cp, dtype=torch.float32)
    if num_curves == 0 or num_ctrl <= 2:
        return grad

    scale = float(lambda_proj) / float(max(1, int(num_curves)))

    kernel = bezier_open_backward_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    kernel(cp, float(scale), grad)
    return grad


def opacity_abs_sigmoid_delta_grad_tilelang(
    opacity: torch.Tensor,
    *,
    opacity_weight: float,
    threads: int = DEFAULT_REG_THREADS,
) -> torch.Tensor:
    op_in = _as_cuda_float32_contiguous(opacity, "opacity")
    op = op_in.view(-1)
    grad_flat = torch.empty_like(op, dtype=torch.float32)
    if op.numel() == 0:
        return torch.zeros_like(op_in, dtype=torch.float32)

    scale = float(opacity_weight) / float(max(1, int(op.numel())))
    kernel = opacity_backward_kernel(
        threads=int(threads),
        dtype="float32",
    )
    kernel(op, float(scale), grad_flat)
    return grad_flat.view_as(op_in)


def boundary_joints_penalty_grad_tilelang(
    points: torch.Tensor,
    *,
    degree: int,
    bound: float = 1.0,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> torch.Tensor:
    pts = _as_cuda_float32_contiguous(points, "points")
    if pts.dim() != 3 or pts.shape[-1] != 2:
        raise ValueError(f"points must have shape [N, K, 2], got {tuple(pts.shape)}")
    if int(degree) <= 0:
        raise ValueError("degree must be > 0")

    num_curves, num_ctrl, _ = pts.shape
    grad = torch.zeros_like(pts, dtype=torch.float32)
    if num_curves == 0 or num_ctrl == 0:
        return grad

    joint_count = _ceildiv(int(num_ctrl), int(degree))
    if joint_count <= 0:
        return grad

    scale = 1.0 / float(max(1, int(num_curves) * int(joint_count) * 2))

    kernel = boundary_backward_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    kernel(pts, int(degree), float(bound), float(scale), grad)
    return grad


def curvature_masked_second_diff_grad_tilelang(
    paths: torch.Tensor,
    *,
    stride: int,
    angle_thresh_deg: float = 60.0,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
) -> torch.Tensor:
    xyz = _as_cuda_float32_contiguous(paths, "paths")
    if xyz.dim() != 4 or xyz.shape[1] != 2 or xyz.shape[-1] != 2:
        raise ValueError(f"paths must have shape [N, 2, M, 2], got {tuple(xyz.shape)}")
    if int(stride) <= 0:
        raise ValueError("stride must be > 0")

    grad_xyz = torch.zeros_like(xyz, dtype=torch.float32)
    if xyz.numel() == 0:
        return grad_xyz

    num_curves = int(xyz.shape[0])
    num_points = int(xyz.shape[2])
    if num_curves == 0 or num_points == 0:
        return grad_xyz

    total_len = int(num_points) * 2
    sampled_per_curve = _ceildiv(total_len, int(stride))
    if sampled_per_curve <= 0:
        return grad_xyz

    denom = max(1, int(num_curves) * int(sampled_per_curve))
    scale = 2.0 / float(denom)
    cos_thresh = float(math.cos(math.radians(float(angle_thresh_deg))))

    kernel = curvature_backward_kernel(
        threads=int(threads),
        items_per_thread=int(items_per_thread),
        dtype="float32",
    )
    kernel(xyz, int(stride), float(cos_thresh), float(scale), grad_xyz)
    return grad_xyz


__all__ = [
    "DEFAULT_REG_ITEMS_PER_THREAD",
    "DEFAULT_REG_THREADS",
    "bezier_shape_backward_kernel",
    "bezier_open_backward_kernel",
    "opacity_backward_kernel",
    "boundary_backward_kernel",
    "curvature_backward_kernel",
    "bezier_open_partial_sum_kernel",
    "bezier_open_proj_outside_grad_tilelang",
    "bezier_open_proj_outside_sum_tilelang",
    "bezier_shape_partial_sum_kernel",
    "bezier_shape_proj_outside_grad_tilelang",
    "bezier_shape_proj_outside_sum_tilelang",
    "boundary_joints_penalty_grad_tilelang",
    "boundary_joints_penalty_sum_tilelang",
    "boundary_partial_sum_kernel",
    "curvature_masked_second_diff_grad_tilelang",
    "curvature_masked_second_diff_sum_tilelang",
    "curvature_partial_sum_kernel",
    "opacity_abs_sigmoid_delta_grad_tilelang",
    "opacity_abs_sigmoid_delta_sum_tilelang",
    "opacity_partial_sum_kernel",
]
