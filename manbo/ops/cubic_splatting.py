from __future__ import annotations

import math
from typing import Any, Callable, Mapping, MutableMapping, Optional

import tilelang
import tilelang.language as T
import torch

from .tile_bins import get_tile_bin_edges_tilelang

BLOCK_X = 16
BLOCK_Y = 16
DEFAULT_THREADS = 256
DEFAULT_AABB_PAD = 1.0
DEFAULT_AA_WIDTH = 1.0
DEFAULT_DISTANCE_SAMPLES = 24
DEFAULT_DE_CASTELJAU_FLATNESS_TOL_PX = 0.5
CUBIC_FLATTEN_METHOD_BERNSTEIN = "bernstein"
CUBIC_FLATTEN_METHOD_DE_CASTELJAU = "de_casteljau"
DEFAULT_CUBIC_FLATTEN_METHOD = CUBIC_FLATTEN_METHOD_BERNSTEIN
_CUBIC_FLATTEN_METHOD_ALIASES = {
    "bernstein": CUBIC_FLATTEN_METHOD_BERNSTEIN,
    "de_casteljau": CUBIC_FLATTEN_METHOD_DE_CASTELJAU,
    "decasteljau": CUBIC_FLATTEN_METHOD_DE_CASTELJAU,
    "de-casteljau": CUBIC_FLATTEN_METHOD_DE_CASTELJAU,
}
DYN_NUM_SEGMENTS = T.dynamic("num_segments")
DYN_NUM_EDGES = T.dynamic("num_edges")
DYN_NUM_PRIMITIVES = T.dynamic("num_primitives")
DYN_NUM_PRIMITIVE_OFFSETS = T.dynamic("num_primitive_offsets")
DYN_NUM_SEGMENT_OFFSETS = T.dynamic("num_segment_offsets")
DYN_NUM_CURVES_FILL = T.dynamic("num_curves_fill")
DYN_NUM_FILL_SEGMENTS = T.dynamic("num_fill_segments")
CubicProcessPayload = MutableMapping[str, Any]
CubicProcessFn = Callable[[str, CubicProcessPayload], Optional[Mapping[str, Any]]]
# Backward-compat aliases for existing fill-only naming.
CubicFillProcessPayload = CubicProcessPayload
CubicFillProcessFn = CubicProcessFn


def _run_cubic_process_fn(
    process_fn: Optional[CubicProcessFn],
    stage: str,
    payload: CubicProcessPayload,
) -> CubicProcessPayload:
    if process_fn is None:
        return payload
    updates = process_fn(stage, payload)
    if updates:
        payload.update(updates)
    return payload


def _as_f32_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float32:
        return t if t.is_contiguous() else t.contiguous()
    return t.contiguous().to(torch.float32)


def _as_i32_contig(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.int32:
        return t if t.is_contiguous() else t.contiguous()
    return t.contiguous().to(torch.int32)


def normalize_cubic_flatten_method(method: object) -> str:
    raw = str(method).strip().lower()
    key = _CUBIC_FLATTEN_METHOD_ALIASES.get(raw, raw)
    if key not in {
        CUBIC_FLATTEN_METHOD_BERNSTEIN,
        CUBIC_FLATTEN_METHOD_DE_CASTELJAU,
    }:
        raise ValueError(
            "flatten_method must be one of "
            f"{sorted(_CUBIC_FLATTEN_METHOD_ALIASES.keys())}, got {method!r}"
        )
    return key


def _closed_fill_piecewise_basis_tensors(
    *,
    control_points_per_bezier: int,
    subdiv: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build basis tensors used by closed cubic fill segment conversion.

    Spec:
    - basis: [subdiv+1, Kb], Kb=control_points_per_bezier
    - deriv_basis: [subdiv+1, Kb-1]
    - dt: [subdiv], adjacent sample spacing in [0,1]
    """
    kb = int(control_points_per_bezier)
    sub = int(subdiv)
    if kb < 2:
        raise ValueError(f"control_points_per_bezier must be >=2, got {kb}")
    if sub <= 0:
        raise ValueError(f"subdiv must be >0, got {sub}")

    sample_n = sub + 1
    dtype = torch.float32
    t = torch.linspace(0.0, 1.0, sample_n, device=device, dtype=dtype)
    t_col = t.view(sample_n, 1)
    omt_col = (1.0 - t).view(sample_n, 1)

    n = kb - 1
    idx = torch.arange(kb, device=device, dtype=dtype)
    comb = torch.tensor(
        [math.comb(n, int(i)) for i in range(kb)],
        device=device,
        dtype=dtype,
    )
    basis = (
        comb.view(1, kb)
        * torch.pow(omt_col, (n - idx).view(1, kb))
        * torch.pow(t_col, idx.view(1, kb))
    ).contiguous()

    deriv_k = kb - 1
    deriv_idx = torch.arange(deriv_k, device=device, dtype=dtype)
    deriv_comb = torch.tensor(
        [math.comb(n - 1, int(i)) for i in range(deriv_k)],
        device=device,
        dtype=dtype,
    )
    deriv_basis = (
        deriv_comb.view(1, deriv_k)
        * torch.pow(omt_col, ((n - 1) - deriv_idx).view(1, deriv_k))
        * torch.pow(t_col, deriv_idx.view(1, deriv_k))
    ).contiguous()

    dt = (t[1:] - t[:-1]).contiguous()
    return basis, deriv_basis, dt


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def closed_cubic_fill_segments_forward_kernel(
    control_points: int,
    subdiv: int = 4,
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
):
    """Convert closed high-degree control polygons to piecewise cubic segments.

    Spec:
    - control_points_in: [N, K, 2] float32, K>=4 and (K-2)%2==0.
    - basis: [S+1, Kb], deriv_basis: [S+1, Kb-1], dt: [S], where
      S=subdiv, Kb=(K+2)/2.
    - out_segments: [N*2*S, 4, 2] float32
      two piecewise-Hermite branches per curve, each with S cubic segments.
    - one thread handles one output segment.
    """
    k = int(control_points)
    sub = int(subdiv)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if k < 4 or ((k - 2) % 2) != 0:
        raise ValueError(
            "closed cubic fill expects control_points K with K>=4 and (K-2)%2==0, "
            f"got K={k}"
        )
    if sub <= 0:
        raise ValueError(f"subdiv must be > 0, got {sub}")

    k_branch = (k + 2) // 2
    seg_per_curve = 2 * sub

    @T.prim_func
    def kernel(
        control_points_in: T.Tensor[[DYN_NUM_CURVES_FILL, k, 2], dtype],
        basis: T.Tensor[[sub + 1, k_branch], dtype],
        deriv_basis: T.Tensor[[sub + 1, k_branch - 1], dtype],
        dt: T.Tensor[[sub], dtype],
        out_segments: T.Tensor[[DYN_NUM_FILL_SEGMENTS, 4, 2], dtype],
        num_curves_i32: T.int32,
        num_segments_i32: T.int32,
    ):
        zero_f = T.Cast(dtype, 0.0)
        one_third = T.Cast(dtype, 1.0 / 3.0)
        degree_f = T.Cast(dtype, float(k_branch - 1))
        seg_per_curve_i32 = T.int32(seg_per_curve)
        subdiv_i32 = T.int32(sub)
        control_points_i32 = T.int32(k)
        k_branch_minus_one_i32 = T.int32(k_branch - 1)

        grid = T.ceildiv(num_segments_i32, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                sid = bx * threads + tx
                if sid < num_segments_i32:
                    curve_id = sid // seg_per_curve_i32
                    if curve_id < num_curves_i32:
                        seg_local = sid - curve_id * seg_per_curve_i32
                        branch = seg_local // subdiv_i32
                        seg_in_branch = seg_local - branch * subdiv_i32
                        seg_next = seg_in_branch + T.int32(1)
                        cp_base = branch * k_branch_minus_one_i32

                        p0x = T.alloc_var(dtype, init=zero_f)
                        p0y = T.alloc_var(dtype, init=zero_f)
                        p3x = T.alloc_var(dtype, init=zero_f)
                        p3y = T.alloc_var(dtype, init=zero_f)
                        d0x = T.alloc_var(dtype, init=zero_f)
                        d0y = T.alloc_var(dtype, init=zero_f)
                        d1x = T.alloc_var(dtype, init=zero_f)
                        d1y = T.alloc_var(dtype, init=zero_f)

                        for i in T.serial(0, k_branch):
                            cp_idx_raw = cp_base + i
                            cp_idx = T.if_then_else(
                                cp_idx_raw >= control_points_i32,
                                cp_idx_raw - control_points_i32,
                                cp_idx_raw,
                            )
                            x = control_points_in[curve_id, cp_idx, 0]
                            y = control_points_in[curve_id, cp_idx, 1]
                            w0 = basis[seg_in_branch, i]
                            w1 = basis[seg_next, i]
                            p0x = p0x + w0 * x
                            p0y = p0y + w0 * y
                            p3x = p3x + w1 * x
                            p3y = p3y + w1 * y

                        for i in T.serial(0, k_branch - 1):
                            cp_cur_raw = cp_base + i
                            cp_nxt_raw = cp_cur_raw + T.int32(1)
                            cp_cur = T.if_then_else(
                                cp_cur_raw >= control_points_i32,
                                cp_cur_raw - control_points_i32,
                                cp_cur_raw,
                            )
                            cp_nxt = T.if_then_else(
                                cp_nxt_raw >= control_points_i32,
                                cp_nxt_raw - control_points_i32,
                                cp_nxt_raw,
                            )
                            dx = (
                                control_points_in[curve_id, cp_nxt, 0]
                                - control_points_in[curve_id, cp_cur, 0]
                            )
                            dy = (
                                control_points_in[curve_id, cp_nxt, 1]
                                - control_points_in[curve_id, cp_cur, 1]
                            )
                            db0 = deriv_basis[seg_in_branch, i]
                            db1 = deriv_basis[seg_next, i]
                            d0x = d0x + db0 * dx
                            d0y = d0y + db0 * dy
                            d1x = d1x + db1 * dx
                            d1y = d1y + db1 * dy

                        d0x = d0x * degree_f
                        d0y = d0y * degree_f
                        d1x = d1x * degree_f
                        d1y = d1y * degree_f
                        dt_seg_third = dt[seg_in_branch] * one_third

                        out_segments[sid, 0, 0] = p0x
                        out_segments[sid, 0, 1] = p0y
                        out_segments[sid, 3, 0] = p3x
                        out_segments[sid, 3, 1] = p3y
                        out_segments[sid, 1, 0] = p0x + d0x * dt_seg_third
                        out_segments[sid, 1, 1] = p0y + d0y * dt_seg_third
                        out_segments[sid, 2, 0] = p3x - d1x * dt_seg_third
                        out_segments[sid, 2, 1] = p3y - d1y * dt_seg_third

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def closed_cubic_fill_segments_backward_kernel(
    control_points: int,
    subdiv: int = 4,
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
):
    """Backward kernel for closed_cubic_fill_segments_forward_kernel.

    Spec:
    - grad_segments: [N*2*S, 4, 2] float32
    - grad_control_points: [N, K, 2] float32 (accumulated with atomics)
    - basis/deriv_basis/dt follow the forward kernel shapes.
    """
    k = int(control_points)
    sub = int(subdiv)
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if k < 4 or ((k - 2) % 2) != 0:
        raise ValueError(
            "closed cubic fill expects control_points K with K>=4 and (K-2)%2==0, "
            f"got K={k}"
        )
    if sub <= 0:
        raise ValueError(f"subdiv must be > 0, got {sub}")

    k_branch = (k + 2) // 2
    seg_per_curve = 2 * sub

    @T.prim_func
    def kernel(
        grad_segments: T.Tensor[[DYN_NUM_FILL_SEGMENTS, 4, 2], dtype],
        basis: T.Tensor[[sub + 1, k_branch], dtype],
        deriv_basis: T.Tensor[[sub + 1, k_branch - 1], dtype],
        dt: T.Tensor[[sub], dtype],
        grad_control_points: T.Tensor[[DYN_NUM_CURVES_FILL, k, 2], dtype],
        num_curves_i32: T.int32,
        num_segments_i32: T.int32,
    ):
        one_third = T.Cast(dtype, 1.0 / 3.0)
        degree_f = T.Cast(dtype, float(k_branch - 1))
        seg_per_curve_i32 = T.int32(seg_per_curve)
        subdiv_i32 = T.int32(sub)
        control_points_i32 = T.int32(k)
        k_branch_minus_one_i32 = T.int32(k_branch - 1)

        grid = T.ceildiv(num_segments_i32, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                sid = bx * threads + tx
                if sid < num_segments_i32:
                    curve_id = sid // seg_per_curve_i32
                    if curve_id < num_curves_i32:
                        seg_local = sid - curve_id * seg_per_curve_i32
                        branch = seg_local // subdiv_i32
                        seg_in_branch = seg_local - branch * subdiv_i32
                        seg_next = seg_in_branch + T.int32(1)
                        cp_base = branch * k_branch_minus_one_i32
                        dt_seg_third = dt[seg_in_branch] * one_third

                        gp0x = grad_segments[sid, 0, 0]
                        gp0y = grad_segments[sid, 0, 1]
                        gp1x = grad_segments[sid, 1, 0]
                        gp1y = grad_segments[sid, 1, 1]
                        gp2x = grad_segments[sid, 2, 0]
                        gp2y = grad_segments[sid, 2, 1]
                        gp3x = grad_segments[sid, 3, 0]
                        gp3y = grad_segments[sid, 3, 1]

                        g0x = gp0x + gp1x
                        g0y = gp0y + gp1y
                        g3x = gp3x + gp2x
                        g3y = gp3y + gp2y
                        gd0x = gp1x * dt_seg_third
                        gd0y = gp1y * dt_seg_third
                        gd1x = -gp2x * dt_seg_third
                        gd1y = -gp2y * dt_seg_third

                        for i in T.serial(0, k_branch):
                            cp_idx_raw = cp_base + i
                            cp_idx = T.if_then_else(
                                cp_idx_raw >= control_points_i32,
                                cp_idx_raw - control_points_i32,
                                cp_idx_raw,
                            )
                            w0 = basis[seg_in_branch, i]
                            w1 = basis[seg_next, i]
                            add_x = g0x * w0 + g3x * w1
                            add_y = g0y * w0 + g3y * w1
                            T.atomic_add(
                                grad_control_points[curve_id, cp_idx, 0], add_x
                            )
                            T.atomic_add(
                                grad_control_points[curve_id, cp_idx, 1], add_y
                            )

                        for i in T.serial(0, k_branch - 1):
                            cp_cur_raw = cp_base + i
                            cp_nxt_raw = cp_cur_raw + T.int32(1)
                            cp_cur = T.if_then_else(
                                cp_cur_raw >= control_points_i32,
                                cp_cur_raw - control_points_i32,
                                cp_cur_raw,
                            )
                            cp_nxt = T.if_then_else(
                                cp_nxt_raw >= control_points_i32,
                                cp_nxt_raw - control_points_i32,
                                cp_nxt_raw,
                            )
                            db0 = deriv_basis[seg_in_branch, i]
                            db1 = deriv_basis[seg_next, i]
                            dcoef_x = degree_f * (gd0x * db0 + gd1x * db1)
                            dcoef_y = degree_f * (gd0y * db0 + gd1y * db1)
                            T.atomic_add(
                                grad_control_points[curve_id, cp_cur, 0], -dcoef_x
                            )
                            T.atomic_add(
                                grad_control_points[curve_id, cp_cur, 1], -dcoef_y
                            )
                            T.atomic_add(
                                grad_control_points[curve_id, cp_nxt, 0], dcoef_x
                            )
                            T.atomic_add(
                                grad_control_points[curve_id, cp_nxt, 1], dcoef_y
                            )

    return kernel


class _ClosedCubicFillSegmentsFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        control_points_in: torch.Tensor,
        basis: torch.Tensor,
        deriv_basis: torch.Tensor,
        dt: torch.Tensor,
        threads: int,
        subdiv: int,
    ) -> torch.Tensor:
        cp = _as_f32_contig(control_points_in)
        basis_f32 = _as_f32_contig(basis)
        deriv_basis_f32 = _as_f32_contig(deriv_basis)
        dt_f32 = _as_f32_contig(dt).view(-1)

        num_curves = int(cp.shape[0])
        control_points = int(cp.shape[1])
        sub = int(subdiv)
        seg_per_curve = 2 * sub
        num_segments = num_curves * seg_per_curve

        out = torch.empty((num_segments, 4, 2), dtype=torch.float32, device=cp.device)
        if num_segments > 0:
            kernel = closed_cubic_fill_segments_forward_kernel(
                control_points=control_points,
                subdiv=sub,
                threads=int(threads),
                dtype="float32",
            )
            kernel(
                cp,
                basis_f32,
                deriv_basis_f32,
                dt_f32,
                out,
                int(num_curves),
                int(num_segments),
            )

        ctx.save_for_backward(basis_f32, deriv_basis_f32, dt_f32)
        ctx.num_curves = num_curves
        ctx.control_points = control_points
        ctx.subdiv = sub
        ctx.threads = int(threads)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        basis, deriv_basis, dt = ctx.saved_tensors
        num_curves = int(ctx.num_curves)
        control_points = int(ctx.control_points)
        sub = int(ctx.subdiv)
        threads = int(ctx.threads)
        seg_per_curve = 2 * sub
        num_segments = num_curves * seg_per_curve

        grad_cp = torch.zeros(
            (num_curves, control_points, 2),
            dtype=torch.float32,
            device=grad_out.device,
        )
        if num_segments > 0:
            kernel = closed_cubic_fill_segments_backward_kernel(
                control_points=control_points,
                subdiv=sub,
                threads=threads,
                dtype="float32",
            )
            kernel(
                _as_f32_contig(grad_out),
                basis,
                deriv_basis,
                dt,
                grad_cp,
                int(num_curves),
                int(num_segments),
            )
        return grad_cp, None, None, None, None, None


def closed_cubic_fill_segments_tilelang(
    control_points: torch.Tensor,
    *,
    subdiv: int = 4,
    threads: int = DEFAULT_THREADS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build closed-fill cubic ring segments from high-degree control polygons.

    Args:
    - control_points: [N, K, 2] float32 CUDA tensor.
      Requires K>=4 and (K-2)%2==0.
    - subdiv: piecewise subdivisions per Bezier branch.
    Returns:
    - ring_segments: [N * 2 * subdiv, 4, 2] float32
    - ring_offsets: [N+1] int32, CSR offsets with constant stride 2*subdiv.
    """
    if control_points.ndim != 3 or int(control_points.shape[-1]) != 2:
        raise ValueError(
            f"control_points must be [N,K,2], got {tuple(control_points.shape)}"
        )
    if not control_points.is_cuda:
        raise ValueError("closed_cubic_fill_segments_tilelang requires CUDA tensor")
    if int(subdiv) <= 0:
        raise ValueError(f"subdiv must be >0, got {int(subdiv)}")

    num_curves = int(control_points.shape[0])
    k = int(control_points.shape[1])
    if k < 4 or ((k - 2) % 2) != 0:
        raise ValueError(
            "closed cubic fill expects control point count K that satisfies K>=4 and (K-2)%2==0, "
            f"got K={k}"
        )

    seg_per_curve = 2 * int(subdiv)
    ring_offsets = torch.arange(
        0,
        (num_curves + 1) * seg_per_curve,
        seg_per_curve,
        dtype=torch.int32,
        device=control_points.device,
    )
    if num_curves <= 0:
        return (
            torch.empty((0, 4, 2), dtype=torch.float32, device=control_points.device),
            ring_offsets,
        )

    cp_f32 = _as_f32_contig(control_points)
    k_branch = (k + 2) // 2
    basis, deriv_basis, dt = _closed_fill_piecewise_basis_tensors(
        control_points_per_bezier=k_branch,
        subdiv=int(subdiv),
        device=cp_f32.device,
    )
    ring_segments = _ClosedCubicFillSegmentsFn.apply(
        cp_f32,
        basis,
        deriv_basis,
        dt,
        int(threads),
        int(subdiv),
    )
    return ring_segments.contiguous(), ring_offsets.contiguous()


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def project_cubics_2d_forward_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    threads: int = DEFAULT_THREADS,
    aabb_pad: float = DEFAULT_AABB_PAD,
    aniso_intersects: bool = False,
    dtype: str = "float32",
    num_curves=T.dynamic("num_curves"),
):
    @T.prim_func
    def kernel(
        cubics_norm: T.Tensor[[num_curves, 4, 2], dtype],
        stroke_widths: T.Tensor[[num_curves], dtype],
        support_radii: T.Tensor[[num_curves], dtype],
        cubics_px: T.Tensor[[num_curves, 4, 2], dtype],
        tile_ranges: T.Tensor[[num_curves, 4], T.int32],
        num_tiles_hit: T.Tensor[[num_curves], T.int32],
        tile_bound_x: T.int32,
        tile_bound_y: T.int32,
        img_height: T.int32,
        img_width: T.int32,
    ):
        half_w_f = T.Cast(dtype, 0.5) * T.Cast(dtype, img_width)
        half_h_f = T.Cast(dtype, 0.5) * T.Cast(dtype, img_height)
        zero_f = T.Cast(dtype, 0.0)
        one_f = T.Cast(dtype, 1.0)
        two_f = T.Cast(dtype, 2.0)
        three_f = T.Cast(dtype, 3.0)
        four_f = T.Cast(dtype, 4.0)
        pad_f = T.Cast(dtype, aabb_pad)
        eps_f = T.Cast(dtype, 1e-6)
        block_x_f = T.Cast(dtype, block_x)
        block_y_f = T.Cast(dtype, block_y)

        grid = T.ceildiv(num_curves, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                gid = bx * threads + tx
                if gid < num_curves:
                    T.assume(gid >= 0)
                    # Project cubic control points from normalized space to pixel space.
                    p0x = cubics_norm[gid, 0, 0] * half_w_f + half_w_f
                    p0y = cubics_norm[gid, 0, 1] * half_h_f + half_h_f
                    p1x = cubics_norm[gid, 1, 0] * half_w_f + half_w_f
                    p1y = cubics_norm[gid, 1, 1] * half_h_f + half_h_f
                    p2x = cubics_norm[gid, 2, 0] * half_w_f + half_w_f
                    p2y = cubics_norm[gid, 2, 1] * half_h_f + half_h_f
                    p3x = cubics_norm[gid, 3, 0] * half_w_f + half_w_f
                    p3y = cubics_norm[gid, 3, 1] * half_h_f + half_h_f

                    cubics_px[gid, 0, 0] = p0x
                    cubics_px[gid, 0, 1] = p0y
                    cubics_px[gid, 1, 0] = p1x
                    cubics_px[gid, 1, 1] = p1y
                    cubics_px[gid, 2, 0] = p2x
                    cubics_px[gid, 2, 1] = p2y
                    cubics_px[gid, 3, 0] = p3x
                    cubics_px[gid, 3, 1] = p3y

                    # Exact cubic AABB via extrema roots of dP/dt in x/y.
                    ax = -p0x + three_f * p1x - three_f * p2x + p3x
                    bx_pow = three_f * p0x - T.Cast(dtype, 6.0) * p1x + three_f * p2x
                    cx_pow = -three_f * p0x + three_f * p1x
                    dx_pow = p0x

                    ay = -p0y + three_f * p1y - three_f * p2y + p3y
                    by_pow = three_f * p0y - T.Cast(dtype, 6.0) * p1y + three_f * p2y
                    cy_pow = -three_f * p0y + three_f * p1y
                    dy_pow = p0y

                    min_x = T.alloc_var(dtype)
                    max_x = T.alloc_var(dtype)
                    min_y = T.alloc_var(dtype)
                    max_y = T.alloc_var(dtype)
                    min_x = T.min(p0x, p3x)
                    max_x = T.max(p0x, p3x)
                    min_y = T.min(p0y, p3y)
                    max_y = T.max(p0y, p3y)

                    # Solve x'(t)=0 for t in (0,1).
                    q2x = three_f * ax
                    q1x = two_f * bx_pow
                    q0x = cx_pow
                    if (q2x > eps_f) | (q2x < -eps_f):
                        disc = q1x * q1x - four_f * q2x * q0x
                        if disc >= -eps_f:
                            sqrt_disc = T.sqrt(T.max(disc, zero_f))
                            inv2q2 = one_f / (two_f * q2x)
                            tx0 = (-q1x - sqrt_disc) * inv2q2
                            tx1 = (-q1x + sqrt_disc) * inv2q2
                            if (tx0 > zero_f) & (tx0 < one_f):
                                vx0 = (
                                    (ax * tx0 + bx_pow) * tx0 + cx_pow
                                ) * tx0 + dx_pow
                                min_x = T.min(min_x, vx0)
                                max_x = T.max(max_x, vx0)
                            if (tx1 > zero_f) & (tx1 < one_f):
                                vx1 = (
                                    (ax * tx1 + bx_pow) * tx1 + cx_pow
                                ) * tx1 + dx_pow
                                min_x = T.min(min_x, vx1)
                                max_x = T.max(max_x, vx1)
                    elif (q1x > eps_f) | (q1x < -eps_f):
                        t_lin_x = -q0x / q1x
                        if (t_lin_x > zero_f) & (t_lin_x < one_f):
                            vx = (
                                (ax * t_lin_x + bx_pow) * t_lin_x + cx_pow
                            ) * t_lin_x + dx_pow
                            min_x = T.min(min_x, vx)
                            max_x = T.max(max_x, vx)

                    # Solve y'(t)=0 for t in (0,1).
                    q2y = three_f * ay
                    q1y = two_f * by_pow
                    q0y = cy_pow
                    if (q2y > eps_f) | (q2y < -eps_f):
                        disc = q1y * q1y - four_f * q2y * q0y
                        if disc >= -eps_f:
                            sqrt_disc = T.sqrt(T.max(disc, zero_f))
                            inv2q2 = one_f / (two_f * q2y)
                            ty0 = (-q1y - sqrt_disc) * inv2q2
                            ty1 = (-q1y + sqrt_disc) * inv2q2
                            if (ty0 > zero_f) & (ty0 < one_f):
                                vy0 = (
                                    (ay * ty0 + by_pow) * ty0 + cy_pow
                                ) * ty0 + dy_pow
                                min_y = T.min(min_y, vy0)
                                max_y = T.max(max_y, vy0)
                            if (ty1 > zero_f) & (ty1 < one_f):
                                vy1 = (
                                    (ay * ty1 + by_pow) * ty1 + cy_pow
                                ) * ty1 + dy_pow
                                min_y = T.min(min_y, vy1)
                                max_y = T.max(max_y, vy1)
                    elif (q1y > eps_f) | (q1y < -eps_f):
                        t_lin_y = -q0y / q1y
                        if (t_lin_y > zero_f) & (t_lin_y < one_f):
                            vy = (
                                (ay * t_lin_y + by_pow) * t_lin_y + cy_pow
                            ) * t_lin_y + dy_pow
                            min_y = T.min(min_y, vy)
                            max_y = T.max(max_y, vy)

                    half_sw = T.max(zero_f, stroke_widths[gid]) * T.Cast(dtype, 0.5)
                    support = T.max(zero_f, support_radii[gid])
                    if aniso_intersects:
                        # More aggressive chord-oriented anisotropic expansion.
                        cx = p3x - p0x
                        cy = p3y - p0y
                        chord2 = cx * cx + cy * cy
                        inv_chord = one_f / T.sqrt(T.max(chord2, eps_f))
                        nx0 = T.abs(cy) * inv_chord
                        ny0 = T.abs(cx) * inv_chord
                        nx = T.max(T.Cast(dtype, 0.20), nx0)
                        ny = T.max(T.Cast(dtype, 0.20), ny0)
                        radial = half_sw + support
                        pad_x = pad_f + radial * nx
                        pad_y = pad_f + radial * ny

                        min_x = min_x - pad_x
                        max_x = max_x + pad_x
                        min_y = min_y - pad_y
                        max_y = max_y + pad_y
                    else:
                        pad_i = T.max(pad_f, support)
                        min_x = min_x - half_sw - pad_i
                        min_y = min_y - half_sw - pad_i
                        max_x = max_x + half_sw + pad_i
                        max_y = max_y + half_sw + pad_i

                    tile_min_x = T.clamp(
                        T.Cast(T.int32, min_x / block_x_f),
                        T.int32(0),
                        tile_bound_x,
                    )
                    tile_min_y = T.clamp(
                        T.Cast(T.int32, min_y / block_y_f),
                        T.int32(0),
                        tile_bound_y,
                    )
                    tile_max_x = T.clamp(
                        T.Cast(T.int32, max_x / block_x_f + one_f),
                        T.int32(0),
                        tile_bound_x,
                    )
                    tile_max_y = T.clamp(
                        T.Cast(T.int32, max_y / block_y_f + one_f),
                        T.int32(0),
                        tile_bound_y,
                    )

                    tile_w = tile_max_x - tile_min_x
                    tile_h = tile_max_y - tile_min_y
                    area = tile_w * tile_h
                    tile_ranges[gid, 0] = tile_min_x
                    tile_ranges[gid, 1] = tile_min_y
                    tile_ranges[gid, 2] = tile_max_x
                    tile_ranges[gid, 3] = tile_max_y
                    num_tiles_hit[gid] = T.max(T.int32(0), area)

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def map_cubic_to_intersects_kernel(
    threads: int = DEFAULT_THREADS,
    num_curves=T.dynamic("num_curves"),
    num_intersects=T.dynamic("num_intersects"),
):
    @T.prim_func
    def kernel(
        tile_ranges: T.Tensor[[num_curves, 4], T.int32],
        depths_i32: T.Tensor[[num_curves], T.int32],
        cum_tiles_hit: T.Tensor[[num_curves], T.int32],
        isect_ids: T.Tensor[[num_intersects], T.int64],
        curve_ids: T.Tensor[[num_intersects], T.int32],
        tile_bound_x: T.int32,
        tile_bound_y: T.int32,
    ):
        _ = tile_bound_y

        grid = T.ceildiv(num_curves, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                gid = bx * threads + tx
                if gid < num_curves:
                    T.assume(gid >= 0)
                    tile_min_x = tile_ranges[gid, 0]
                    tile_min_y = tile_ranges[gid, 1]
                    tile_max_x = tile_ranges[gid, 2]
                    tile_max_y = tile_ranges[gid, 3]

                    tile_w = tile_max_x - tile_min_x
                    tile_h = tile_max_y - tile_min_y
                    tile_area = tile_w * tile_h
                    if tile_area > T.int32(0):
                        base_out = T.alloc_var("int32")
                        row_base_out = T.alloc_var("int32")
                        row_base_tile = T.alloc_var("int32")
                        out_idx = T.alloc_var("int32")
                        tile_id64 = T.alloc_var("int64")
                        depth_sign_mask = T.alloc_var("int32")
                        depth_key_i32 = T.alloc_var("int32")
                        depth_u64 = T.alloc_var("int64")

                        base_out = T.if_then_else(
                            gid == 0,
                            T.int32(0),
                            cum_tiles_hit[gid - 1],
                        )
                        T.assume(base_out >= 0)
                        T.assume(base_out + tile_area <= num_intersects)
                        # Encode float bits so integer ascending order matches float
                        # ascending order for both negative and positive depths.
                        depth_sign_mask = (depths_i32[gid] >> 31) | T.int32(-2147483648)
                        depth_key_i32 = depths_i32[gid] ^ depth_sign_mask
                        depth_u64 = T.Cast(T.int64, depth_key_i32) & T.Cast(
                            T.int64, 0xFFFFFFFF
                        )

                        for row in T.serial(tile_h):
                            row_base_out = base_out + row * tile_w
                            row_base_tile = (
                                tile_min_y + row
                            ) * tile_bound_x + tile_min_x
                            for col in T.serial(tile_w):
                                out_idx = row_base_out + col
                                T.assume((out_idx >= 0) & (out_idx < num_intersects))
                                tile_id64 = T.Cast(T.int64, row_base_tile + col)
                                isect_ids[out_idx] = (tile_id64 << 32) | depth_u64
                                curve_ids[out_idx] = gid

    return kernel


def project_cubics_2d_forward_tilelang(
    cubics_norm: torch.Tensor,
    stroke_widths: torch.Tensor,
    support_radii: torch.Tensor,
    img_height: int,
    img_width: int,
    tile_bounds: tuple[int, int, int],
    *,
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    threads: int = DEFAULT_THREADS,
    aabb_pad: float = DEFAULT_AABB_PAD,
    aniso_intersects: bool = False,
):
    num_curves = int(cubics_norm.shape[0])
    cubics_px = torch.empty(
        (num_curves, 4, 2), dtype=torch.float32, device=cubics_norm.device
    )
    tile_ranges = torch.empty(
        (num_curves, 4), dtype=torch.int32, device=cubics_norm.device
    )
    num_tiles_hit = torch.empty(
        (num_curves,), dtype=torch.int32, device=cubics_norm.device
    )

    tile_bound_x, tile_bound_y, _ = tile_bounds
    kernel = project_cubics_2d_forward_kernel(
        block_x=int(block_x),
        block_y=int(block_y),
        threads=int(threads),
        aabb_pad=float(aabb_pad),
        aniso_intersects=bool(aniso_intersects),
        dtype="float32",
    )
    cubics_norm_f32 = _as_f32_contig(cubics_norm)
    stroke_widths_f32 = _as_f32_contig(stroke_widths).view(-1)
    support_radii_f32 = _as_f32_contig(support_radii).view(-1)
    kernel(
        cubics_norm_f32,
        stroke_widths_f32,
        support_radii_f32,
        cubics_px,
        tile_ranges,
        num_tiles_hit,
        int(tile_bound_x),
        int(tile_bound_y),
        int(img_height),
        int(img_width),
    )
    return cubics_px, tile_ranges, num_tiles_hit


def map_cubic_to_intersects_tilelang(
    tile_ranges: torch.Tensor,
    depths: torch.Tensor,
    cum_tiles_hit: torch.Tensor,
    tile_bounds: tuple[int, int, int],
    *,
    threads: int = DEFAULT_THREADS,
):
    num_intersects = (
        int(cum_tiles_hit[-1].item()) if int(cum_tiles_hit.numel()) > 0 else 0
    )
    if num_intersects <= 0:
        dev = tile_ranges.device
        return (
            torch.empty((0,), dtype=torch.int64, device=dev),
            torch.empty((0,), dtype=torch.int32, device=dev),
        )

    isect_ids = torch.empty(
        (num_intersects,), dtype=torch.int64, device=tile_ranges.device
    )
    curve_ids = torch.empty(
        (num_intersects,), dtype=torch.int32, device=tile_ranges.device
    )
    tile_bound_x, tile_bound_y, _ = tile_bounds

    kernel = map_cubic_to_intersects_kernel(
        threads=int(threads),
    )
    depths_i32 = _as_f32_contig(depths).view(torch.int32)
    tile_ranges_i32 = _as_i32_contig(tile_ranges)
    cum_tiles_hit_i32 = _as_i32_contig(cum_tiles_hit)
    kernel(
        tile_ranges_i32,
        depths_i32,
        cum_tiles_hit_i32,
        isect_ids,
        curve_ids,
        int(tile_bound_x),
        int(tile_bound_y),
    )
    return isect_ids, curve_ids


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def rasterize_cubic_fill_forward_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    distance_samples: int = DEFAULT_DISTANCE_SAMPLES,
    dtype: str = "float32",
):
    # Kept for API compatibility; cubic flattening now happens once on host.
    _ = distance_samples
    num_primitives = T.dynamic("num_primitives")
    num_edges = T.dynamic("num_edges")
    num_intersects = T.dynamic("num_intersects")
    num_tiles = T.dynamic("num_tiles")
    img_height = T.dynamic("img_height")
    img_width = T.dynamic("img_width")
    block_size = block_x * block_y

    @T.prim_func
    def kernel(
        primitive_ids_sorted: T.Tensor[[num_intersects], T.int32],
        tile_bins: T.Tensor[[num_tiles, 2], T.int32],
        edges_px: T.Tensor[[num_edges, 2, 2], dtype],
        isect_edge_ranges: T.Tensor[[num_intersects, 2], T.int32],
        colors: T.Tensor[[num_primitives, 3], dtype],
        opacities: T.Tensor[[num_primitives], dtype],
        inv_sigma2s: T.Tensor[[num_primitives], dtype],
        fill_rules: T.Tensor[[num_primitives], T.int32],
        background: T.Tensor[[3], dtype],
        out_img: T.Tensor[[img_height, img_width, 3], dtype],
        final_Ts: T.Tensor[[img_height, img_width], dtype],
        final_idx: T.Tensor[[img_height, img_width], T.int32],
    ):
        tile_bound_x = T.ceildiv(img_width, block_x)
        tile_bound_y = T.ceildiv(img_height, block_y)
        alpha_cap = T.Cast(dtype, 0.999)
        alpha_min = T.Cast(dtype, 1.0 / 255.0)
        trans_stop = T.Cast(dtype, 1e-4)
        zero_f = T.Cast(dtype, 0.0)
        one_f = T.Cast(dtype, 1.0)
        half_f = T.Cast(dtype, 0.5)
        eps_f = T.Cast(dtype, 1e-6)
        two_f = T.Cast(dtype, 2.0)
        neg_lim = T.Cast(dtype, -60.0)
        pos_lim = T.Cast(dtype, 60.0)

        with T.Kernel(tile_bound_x, tile_bound_y, threads=(block_x, block_y)) as (
            bx,
            by,
        ):
            T.annotate_safe_value(
                {
                    primitive_ids_sorted: T.int32(0),
                    tile_bins: T.int32(0),
                    edges_px: T.Cast(dtype, 0.0),
                    isect_edge_ranges: T.int32(0),
                    colors: T.Cast(dtype, 0.0),
                    opacities: T.Cast(dtype, 0.0),
                    inv_sigma2s: T.Cast(dtype, 1.0),
                    fill_rules: T.int32(1),
                }
            )

            tile_id = by * tile_bound_x + bx
            T.assume(tile_id < num_tiles)
            range_start = tile_bins[tile_id, 0]
            range_end = tile_bins[tile_id, 1]
            T.assume(range_start >= 0)
            T.assume(range_end >= range_start)
            T.assume(range_end <= num_intersects)
            num_batches = T.ceildiv(range_end - range_start, block_size)

            color0_batch = T.alloc_shared((block_size,), dtype)
            color1_batch = T.alloc_shared((block_size,), dtype)
            color2_batch = T.alloc_shared((block_size,), dtype)
            opacity_batch = T.alloc_shared((block_size,), dtype)
            inv_sigma2_batch = T.alloc_shared((block_size,), dtype)
            edge_start_batch = T.alloc_shared((block_size,), "int32")
            edge_end_batch = T.alloc_shared((block_size,), "int32")
            rule_batch = T.alloc_shared((block_size,), "int32")
            edge_x0_batch = T.alloc_shared((block_size,), dtype)
            edge_y0_batch = T.alloc_shared((block_size,), dtype)
            edge_x1_batch = T.alloc_shared((block_size,), dtype)
            edge_y1_batch = T.alloc_shared((block_size,), dtype)
            edge_vx_batch = T.alloc_shared((block_size,), dtype)
            edge_vy_batch = T.alloc_shared((block_size,), dtype)
            edge_inv_vv_batch = T.alloc_shared((block_size,), dtype)

            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            tr = ty * block_x + tx

            i = by * block_y + ty
            j = bx * block_x + tx
            pix_id = T.min(i * img_width + j, img_height * img_width - 1)
            T.assume((pix_id >= 0) & (pix_id < img_height * img_width))
            px = T.Cast(dtype, j) + half_f
            py = T.Cast(dtype, i) + half_f
            inside_pix_i32 = T.if_then_else(
                (i < img_height) & (j < img_width),
                T.int32(1),
                T.int32(0),
            )

            T_local = T.alloc_var(dtype)
            T_local = one_f
            done = T.alloc_var("int32")
            done = T.if_then_else(
                inside_pix_i32 != T.int32(0),
                T.int32(0),
                T.int32(1),
            )
            cur_idx = T.alloc_var("int32")
            cur_idx = T.int32(0)
            pix_out = T.alloc_local((3,), dtype)
            pix_out[0] = zero_f
            pix_out[1] = zero_f
            pix_out[2] = zero_f

            for b in T.serial(num_batches):
                done_count = T.call_extern("int32", "__syncthreads_count", done)
                if done_count >= block_size:
                    break

                batch_start = range_start + block_size * b
                idx = batch_start + tr
                if idx < range_end:
                    pid = primitive_ids_sorted[idx]
                    T.assume((pid >= 0) & (pid < num_primitives))
                    edge_start_batch[tr] = isect_edge_ranges[idx, 0]
                    edge_end_batch[tr] = isect_edge_ranges[idx, 1]
                    color0_batch[tr] = colors[pid, 0]
                    color1_batch[tr] = colors[pid, 1]
                    color2_batch[tr] = colors[pid, 2]
                    opacity_batch[tr] = opacities[pid]
                    inv_sigma2_batch[tr] = inv_sigma2s[pid]
                    rule_batch[tr] = fill_rules[pid]

                T.sync_threads()

                batch_size = T.min(block_size, range_end - batch_start)
                for t in T.serial(batch_size):
                    edge_start = edge_start_batch[t]
                    edge_end = edge_end_batch[t]
                    n_edge = T.max(T.int32(0), edge_end - edge_start)

                    min_d2 = T.alloc_var(dtype)
                    min_d2 = T.Cast(dtype, 1e30)
                    parity = T.alloc_var("int32")
                    winding = T.alloc_var("int32")
                    parity = T.int32(0)
                    winding = T.int32(0)

                    num_chunks = T.ceildiv(n_edge, block_size)
                    for c in T.serial(num_chunks):
                        load_idx = c * block_size + tr
                        if load_idx < n_edge:
                            eid = edge_start + load_idx
                            T.assume((eid >= T.int32(0)) & (eid < num_edges))
                            edge_x0 = edges_px[eid, 0, 0]
                            edge_y0 = edges_px[eid, 0, 1]
                            edge_x1 = edges_px[eid, 1, 0]
                            edge_y1 = edges_px[eid, 1, 1]
                            edge_x0_batch[tr] = edge_x0
                            edge_y0_batch[tr] = edge_y0
                            edge_x1_batch[tr] = edge_x1
                            edge_y1_batch[tr] = edge_y1
                            vx_edge = edge_x1 - edge_x0
                            vy_edge = edge_y1 - edge_y0
                            vv_edge = vx_edge * vx_edge + vy_edge * vy_edge
                            edge_vx_batch[tr] = vx_edge
                            edge_vy_batch[tr] = vy_edge
                            edge_inv_vv_batch[tr] = one_f / T.max(vv_edge, eps_f)

                        T.sync_threads()

                        if done == T.int32(0):
                            chunk_len = T.min(block_size, n_edge - c * block_size)
                            for e in T.serial(chunk_len):
                                x0 = edge_x0_batch[e]
                                y0 = edge_y0_batch[e]
                                y1 = edge_y1_batch[e]
                                vx = edge_vx_batch[e]
                                vy = edge_vy_batch[e]
                                wx = px - x0
                                wy = py - y0
                                proj = T.alloc_var(dtype)
                                proj = (wx * vx + wy * vy) * edge_inv_vv_batch[e]
                                proj = T.max(zero_f, T.min(one_f, proj))
                                cx = x0 + proj * vx
                                cy = y0 + proj * vy
                                dx = cx - px
                                dy = cy - py
                                d2 = dx * dx + dy * dy
                                min_d2 = T.min(min_d2, d2)

                                cross_cond = ((y0 <= py) & (y1 > py)) | (
                                    (y0 > py) & (y1 <= py)
                                )
                                if cross_cond:
                                    dy_seg = vy
                                    lhs = x0 * dy_seg + (py - y0) * vx
                                    rhs = px * dy_seg
                                    hit_i32 = T.if_then_else(
                                        T.if_then_else(
                                            dy_seg > zero_f,
                                            lhs > rhs,
                                            lhs < rhs,
                                        ),
                                        T.int32(1),
                                        T.int32(0),
                                    )
                                    parity = parity ^ hit_i32
                                    winding = winding + hit_i32 * T.if_then_else(
                                        dy_seg > zero_f,
                                        T.int32(1),
                                        T.int32(-1),
                                    )

                        T.sync_threads()

                    if done == T.int32(0):
                        rule = rule_batch[t]
                        rule_evenodd_i32 = T.if_then_else(
                            rule == T.int32(0),
                            T.int32(1),
                            T.int32(0),
                        )
                        inside_parity_i32 = T.if_then_else(
                            parity != T.int32(0),
                            T.int32(1),
                            T.int32(0),
                        )
                        inside_winding_i32 = T.if_then_else(
                            winding != T.int32(0),
                            T.int32(1),
                            T.int32(0),
                        )
                        inside_shape_i32 = (
                            rule_evenodd_i32 * inside_parity_i32
                            + (T.int32(1) - rule_evenodd_i32) * inside_winding_i32
                        )

                        dist = T.sqrt(T.max(min_d2, zero_f))
                        inv_sigma = T.sqrt(T.max(inv_sigma2_batch[t], eps_f))
                        sign = T.if_then_else(
                            inside_shape_i32 != T.int32(0),
                            -one_f,
                            one_f,
                        )
                        logit = T.alloc_var(dtype)
                        logit = sign * dist * inv_sigma * two_f
                        logit = T.max(neg_lim, T.min(pos_lim, logit))
                        cov = one_f / (one_f + T.exp(logit))
                        alpha = T.min(alpha_cap, opacity_batch[t] * cov)
                        if alpha < alpha_min:
                            continue

                        next_T = T_local * (one_f - alpha)
                        vis = alpha * T_local
                        pix_out[0] += color0_batch[t] * vis
                        pix_out[1] += color1_batch[t] * vis
                        pix_out[2] += color2_batch[t] * vis
                        T_local = next_T
                        cur_idx = batch_start + t
                        if next_T <= trans_stop:
                            done = T.int32(1)

                T.sync_threads()

            if inside_pix_i32 != T.int32(0):
                final_Ts[i, j] = T_local
                final_idx[i, j] = cur_idx
                out_img[i, j, 0] = pix_out[0] + T_local * background[0]
                out_img[i, j, 1] = pix_out[1] + T_local * background[1]
                out_img[i, j, 2] = pix_out[2] + T_local * background[2]

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def rasterize_cubic_fill_backward_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    distance_samples: int = DEFAULT_DISTANCE_SAMPLES,
    dtype: str = "float32",
):
    _ = distance_samples
    num_primitives = T.dynamic("num_primitives")
    num_edges = T.dynamic("num_edges")
    num_intersects = T.dynamic("num_intersects")
    num_tiles = T.dynamic("num_tiles")
    img_height = T.dynamic("img_height")
    img_width = T.dynamic("img_width")
    num_isect_contrib = T.dynamic("num_isect_contrib")
    block_size = block_x * block_y

    @T.prim_func
    def kernel(
        primitive_ids_sorted: T.Tensor[[num_intersects], T.int32],
        tile_bins: T.Tensor[[num_tiles, 2], T.int32],
        edges_px: T.Tensor[[num_edges, 2, 2], dtype],
        isect_edge_ranges: T.Tensor[[num_intersects, 2], T.int32],
        colors: T.Tensor[[num_primitives, 3], dtype],
        opacities: T.Tensor[[num_primitives], dtype],
        inv_sigma2s: T.Tensor[[num_primitives], dtype],
        fill_rules: T.Tensor[[num_primitives], T.int32],
        background: T.Tensor[[3], dtype],
        final_Ts: T.Tensor[[img_height, img_width], dtype],
        final_idx: T.Tensor[[img_height, img_width], T.int32],
        v_output: T.Tensor[[img_height, img_width, 3], dtype],
        v_edges: T.Tensor[[num_edges, 2, 2], dtype],
        v_colors: T.Tensor[[num_primitives, 3], dtype],
        v_opacities: T.Tensor[[num_primitives], dtype],
        v_inv_sigma2s: T.Tensor[[num_primitives], dtype],
        collect_curve_contrib: T.int32,
        curve_contrib_rgb: T.Tensor[[num_primitives, 3], dtype],
        collect_curve_isect_contrib: T.int32,
        curve_isect_contrib: T.Tensor[[num_isect_contrib], dtype],
    ):
        tile_bound_x = T.ceildiv(img_width, block_x)
        tile_bound_y = T.ceildiv(img_height, block_y)
        alpha_cap = T.Cast(dtype, 0.999)
        alpha_min = T.Cast(dtype, 1.0 / 255.0)
        zero_f = T.Cast(dtype, 0.0)
        one_f = T.Cast(dtype, 1.0)
        half_f = T.Cast(dtype, 0.5)
        two_f = T.Cast(dtype, 2.0)
        neg_lim = T.Cast(dtype, -60.0)
        pos_lim = T.Cast(dtype, 60.0)
        eps_f = T.Cast(dtype, 1e-6)

        with T.Kernel(tile_bound_x, tile_bound_y, threads=(block_x, block_y)) as (
            bx,
            by,
        ):
            T.annotate_safe_value(
                {
                    primitive_ids_sorted: T.int32(0),
                    tile_bins: T.int32(0),
                    edges_px: T.Cast(dtype, 0.0),
                    isect_edge_ranges: T.int32(0),
                    colors: T.Cast(dtype, 0.0),
                    opacities: T.Cast(dtype, 0.0),
                    inv_sigma2s: T.Cast(dtype, 1.0),
                    fill_rules: T.int32(1),
                }
            )

            tile_id = by * tile_bound_x + bx
            T.assume(tile_id < num_tiles)
            range_start = tile_bins[tile_id, 0]
            range_end = tile_bins[tile_id, 1]
            T.assume(range_start >= 0)
            T.assume(range_end >= range_start)
            T.assume(range_end <= num_intersects)
            num_batches = T.ceildiv(range_end - range_start, block_size)

            color0_batch = T.alloc_shared((block_size,), dtype)
            color1_batch = T.alloc_shared((block_size,), dtype)
            color2_batch = T.alloc_shared((block_size,), dtype)
            opacity_batch = T.alloc_shared((block_size,), dtype)
            inv_sigma2_batch = T.alloc_shared((block_size,), dtype)
            edge_start_batch = T.alloc_shared((block_size,), "int32")
            edge_end_batch = T.alloc_shared((block_size,), "int32")
            rule_batch = T.alloc_shared((block_size,), "int32")
            id_batch = T.alloc_shared((block_size,), "int32")
            final_Ts_flat = T.reshape(final_Ts, [img_height * img_width])
            final_idx_flat = T.reshape(final_idx, [img_height * img_width])
            v_output_flat = T.reshape(v_output, [img_height * img_width, 3])

            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            tr = ty * block_x + tx
            lane = T.get_lane_idx()

            i = by * block_y + ty
            j = bx * block_x + tx
            pix_id = T.min(i * img_width + j, img_height * img_width - 1)
            T.assume((pix_id >= 0) & (pix_id < img_height * img_width))
            px = T.Cast(dtype, j) + half_f
            py = T.Cast(dtype, i) + half_f

            inside_pix_i32 = T.if_then_else(
                (i < img_height) & (j < img_width),
                T.int32(1),
                T.int32(0),
            )
            bin_final = T.if_then_else(
                inside_pix_i32 != T.int32(0),
                final_idx_flat[pix_id],
                T.int32(0),
            )
            T_local = T.alloc_var(dtype)
            T_local = final_Ts_flat[pix_id]
            T_final = T_local

            v_out0 = v_output_flat[pix_id, 0]
            v_out1 = v_output_flat[pix_id, 1]
            v_out2 = v_output_flat[pix_id, 2]
            v_out_alpha = zero_f
            bg_dot_vout = (
                background[0] * v_out0 + background[1] * v_out1 + background[2] * v_out2
            )

            buffer0 = T.alloc_var(dtype, init=zero_f)
            buffer1 = T.alloc_var(dtype, init=zero_f)
            buffer2 = T.alloc_var(dtype, init=zero_f)
            valid_i32 = T.alloc_var("int32")
            contrib_i32 = T.alloc_var("int32")
            opacity_valid_i32 = T.alloc_var("int32")
            geom_valid_i32 = T.alloc_var("int32")

            lane_mask = T.Cast("uint32", T.uint32(0xFFFFFFFF))
            use_bin_final = T.warp_reduce_max(bin_final)
            v_rgb_local = T.alloc_local((3,), dtype)
            contrib_rgb_local = T.alloc_local((3,), dtype)
            v_opacity_local = T.alloc_var(dtype)
            v_inv_sigma2_local = T.alloc_var(dtype)
            v_edge_local = T.alloc_local((4,), dtype)
            best_eid_local = T.alloc_var("int32")

            for b in T.serial(num_batches):
                T.sync_threads()
                batch_end = range_end - 1 - block_size * b
                batch_size = T.min(block_size, batch_end + 1 - range_start)
                idx = batch_end - tr
                if idx >= range_start:
                    T.assume(idx < num_intersects)
                    pid = primitive_ids_sorted[idx]
                    T.assume((pid >= 0) & (pid < num_primitives))
                    id_batch[tr] = pid
                    edge_start_batch[tr] = isect_edge_ranges[idx, 0]
                    edge_end_batch[tr] = isect_edge_ranges[idx, 1]
                    color0_batch[tr] = colors[pid, 0]
                    color1_batch[tr] = colors[pid, 1]
                    color2_batch[tr] = colors[pid, 2]
                    opacity_batch[tr] = opacities[pid]
                    inv_sigma2_batch[tr] = inv_sigma2s[pid]
                    rule_batch[tr] = fill_rules[pid]

                T.sync_threads()

                t_start = T.max(T.int32(0), batch_end - use_bin_final)
                for t in T.serial(t_start, batch_size):
                    gidx = T.Cast(T.int32, batch_end) - T.Cast(T.int32, t)
                    valid_i32 = inside_pix_i32
                    if gidx > bin_final:
                        valid_i32 = T.int32(0)
                    warp_any_base = T.call_extern(
                        "int32",
                        "__any_sync",
                        lane_mask,
                        valid_i32,
                    )
                    if warp_any_base == T.int32(0):
                        continue

                    v_rgb_local[0] = zero_f
                    v_rgb_local[1] = zero_f
                    v_rgb_local[2] = zero_f
                    contrib_rgb_local[0] = zero_f
                    contrib_rgb_local[1] = zero_f
                    contrib_rgb_local[2] = zero_f
                    v_opacity_local = zero_f
                    v_inv_sigma2_local = zero_f
                    for k in T.serial(4):
                        v_edge_local[k] = zero_f
                    best_eid_local = T.int32(-1)
                    contrib_i32 = T.int32(0)
                    opacity_valid_i32 = T.int32(0)
                    geom_valid_i32 = T.int32(0)

                    rgb0 = T.alloc_var(dtype)
                    rgb1 = T.alloc_var(dtype)
                    rgb2 = T.alloc_var(dtype)
                    opac = T.alloc_var(dtype)
                    inv_sigma2 = T.alloc_var(dtype)
                    rule = T.alloc_var("int32")
                    rgb0 = zero_f
                    rgb1 = zero_f
                    rgb2 = zero_f
                    opac = zero_f
                    inv_sigma2 = one_f
                    rule = T.int32(1)
                    if valid_i32 != T.int32(0):
                        rgb0 = color0_batch[t]
                        rgb1 = color1_batch[t]
                        rgb2 = color2_batch[t]
                        opac = opacity_batch[t]
                        inv_sigma2 = T.max(inv_sigma2_batch[t], eps_f)
                        rule = rule_batch[t]

                    edge_start = edge_start_batch[t]
                    edge_end = edge_end_batch[t]
                    n_edge = T.max(T.int32(0), edge_end - edge_start)

                    min_d2 = T.alloc_var(dtype)
                    min_d2 = T.Cast(dtype, 1e30)
                    parity = T.alloc_var("int32")
                    winding = T.alloc_var("int32")
                    best_eid = T.alloc_var("int32")
                    best_proj = T.alloc_var(dtype)
                    best_cx = T.alloc_var(dtype)
                    best_cy = T.alloc_var(dtype)
                    parity = T.int32(0)
                    winding = T.int32(0)
                    best_eid = T.int32(-1)
                    best_proj = zero_f
                    best_cx = zero_f
                    best_cy = zero_f

                    num_chunks = T.ceildiv(n_edge, T.int32(32))
                    for c in T.serial(num_chunks):
                        load_idx = c * T.int32(32) + lane
                        eid_lane = T.alloc_var("int32")
                        x0_lane = T.alloc_var(dtype)
                        y0_lane = T.alloc_var(dtype)
                        x1_lane = T.alloc_var(dtype)
                        y1_lane = T.alloc_var(dtype)
                        eid_lane = T.int32(-1)
                        x0_lane = zero_f
                        y0_lane = zero_f
                        x1_lane = zero_f
                        y1_lane = zero_f
                        if load_idx < n_edge:
                            eid_lane = edge_start + load_idx
                            T.assume((eid_lane >= T.int32(0)) & (eid_lane < num_edges))
                            x0_lane = edges_px[eid_lane, 0, 0]
                            y0_lane = edges_px[eid_lane, 0, 1]
                            x1_lane = edges_px[eid_lane, 1, 0]
                            y1_lane = edges_px[eid_lane, 1, 1]

                        chunk_len = T.min(T.int32(32), n_edge - c * T.int32(32))
                        for e in T.serial(chunk_len):
                            eid = T.shfl_sync(eid_lane, e, mask=0xFFFFFFFF)
                            x0 = T.shfl_sync(x0_lane, e, mask=0xFFFFFFFF)
                            y0 = T.shfl_sync(y0_lane, e, mask=0xFFFFFFFF)
                            x1 = T.shfl_sync(x1_lane, e, mask=0xFFFFFFFF)
                            y1 = T.shfl_sync(y1_lane, e, mask=0xFFFFFFFF)

                            if valid_i32 != T.int32(0):
                                vx = x1 - x0
                                vy = y1 - y0
                                wx = px - x0
                                wy = py - y0
                                vv = vx * vx + vy * vy
                                inv_vv = one_f / T.max(vv, eps_f)
                                proj = T.alloc_var(dtype)
                                proj = (wx * vx + wy * vy) * inv_vv
                                proj = T.max(zero_f, T.min(one_f, proj))
                                cx = x0 + proj * vx
                                cy = y0 + proj * vy
                                dx = cx - px
                                dy = cy - py
                                d2 = dx * dx + dy * dy
                                if d2 < min_d2:
                                    min_d2 = d2
                                    best_eid = eid
                                    best_proj = proj
                                    best_cx = cx
                                    best_cy = cy

                                cross_cond = ((y0 <= py) & (y1 > py)) | (
                                    (y0 > py) & (y1 <= py)
                                )
                                if cross_cond:
                                    dy_seg = y1 - y0
                                    lhs = x0 * dy_seg + (py - y0) * (x1 - x0)
                                    rhs = px * dy_seg
                                    hit_i32 = T.if_then_else(
                                        T.if_then_else(
                                            dy_seg > zero_f,
                                            lhs > rhs,
                                            lhs < rhs,
                                        ),
                                        T.int32(1),
                                        T.int32(0),
                                    )
                                    parity = parity ^ hit_i32
                                    winding = winding + hit_i32 * T.if_then_else(
                                        dy_seg > zero_f,
                                        T.int32(1),
                                        T.int32(-1),
                                    )

                    if valid_i32 != T.int32(0):
                        inside_shape_i32 = T.alloc_var("int32")
                        if rule == T.int32(0):
                            inside_shape_i32 = T.if_then_else(
                                parity != T.int32(0),
                                T.int32(1),
                                T.int32(0),
                            )
                        else:
                            inside_shape_i32 = T.if_then_else(
                                winding != T.int32(0),
                                T.int32(1),
                                T.int32(0),
                            )

                        dist = T.sqrt(T.max(min_d2, zero_f))
                        inv_sigma = T.sqrt(inv_sigma2)
                        sign = T.if_then_else(
                            inside_shape_i32 != T.int32(0),
                            -one_f,
                            one_f,
                        )
                        logit = T.alloc_var(dtype)
                        logit = sign * dist * inv_sigma * two_f
                        logit = T.max(neg_lim, T.min(pos_lim, logit))
                        cov = one_f / (one_f + T.exp(logit))
                        alpha = T.min(alpha_cap, opac * cov)

                        if alpha >= alpha_min:
                            contrib_i32 = T.int32(1)
                            ra = one_f / (one_f - alpha)
                            T_local = T_local * ra
                            fac = alpha * T_local

                            v_rgb_local[0] = fac * v_out0
                            v_rgb_local[1] = fac * v_out1
                            v_rgb_local[2] = fac * v_out2
                            if (collect_curve_contrib != T.int32(0)) | (
                                collect_curve_isect_contrib != T.int32(0)
                            ):
                                contrib_rgb_local[0] = fac * rgb0
                                contrib_rgb_local[1] = fac * rgb1
                                contrib_rgb_local[2] = fac * rgb2

                            v_alpha = T.alloc_var(dtype)
                            v_alpha = zero_f
                            v_alpha = v_alpha + (rgb0 * T_local - buffer0 * ra) * v_out0
                            v_alpha = v_alpha + (rgb1 * T_local - buffer1 * ra) * v_out1
                            v_alpha = v_alpha + (rgb2 * T_local - buffer2 * ra) * v_out2
                            v_alpha = v_alpha + T_final * ra * (
                                v_out_alpha - bg_dot_vout
                            )

                            buffer0 = buffer0 + rgb0 * fac
                            buffer1 = buffer1 + rgb1 * fac
                            buffer2 = buffer2 + rgb2 * fac

                            if alpha < alpha_cap:
                                opacity_valid_i32 = T.int32(1)
                                v_opacity_local = cov * v_alpha

                                v_cov = opac * v_alpha
                                v_logit = -v_cov * cov * (one_f - cov)
                                v_inv_sigma = v_logit * sign * dist * two_f
                                v_inv_sigma2_local = (
                                    v_inv_sigma * half_f / T.max(inv_sigma, eps_f)
                                )

                                if (dist > eps_f) & (best_eid >= T.int32(0)):
                                    geom_valid_i32 = T.int32(1)
                                    v_dist = v_logit * sign * inv_sigma * two_f
                                    gx = v_dist * (best_cx - px) / dist
                                    gy = v_dist * (best_cy - py) / dist
                                    v_edge_local[0] = (one_f - best_proj) * gx
                                    v_edge_local[1] = (one_f - best_proj) * gy
                                    v_edge_local[2] = best_proj * gx
                                    v_edge_local[3] = best_proj * gy
                                    best_eid_local = best_eid

                    warp_any = T.call_extern(
                        "int32",
                        "__any_sync",
                        lane_mask,
                        contrib_i32,
                    )
                    if warp_any == T.int32(0):
                        continue

                    warp_any_opacity = T.call_extern(
                        "int32",
                        "__any_sync",
                        lane_mask,
                        opacity_valid_i32,
                    )

                    v_rgb_local[0] = T.warp_reduce_sum(v_rgb_local[0])
                    v_rgb_local[1] = T.warp_reduce_sum(v_rgb_local[1])
                    v_rgb_local[2] = T.warp_reduce_sum(v_rgb_local[2])
                    if (collect_curve_contrib != T.int32(0)) | (
                        collect_curve_isect_contrib != T.int32(0)
                    ):
                        contrib_rgb_local[0] = T.warp_reduce_sum(contrib_rgb_local[0])
                        contrib_rgb_local[1] = T.warp_reduce_sum(contrib_rgb_local[1])
                        contrib_rgb_local[2] = T.warp_reduce_sum(contrib_rgb_local[2])
                    if warp_any_opacity != T.int32(0):
                        v_opacity_local = T.warp_reduce_sum(v_opacity_local)
                        v_inv_sigma2_local = T.warp_reduce_sum(v_inv_sigma2_local)

                    if lane == 0:
                        pid = id_batch[t]
                        T.assume((pid >= 0) & (pid < num_primitives))
                        T.atomic_add(v_colors[pid, 0], v_rgb_local[0])
                        T.atomic_add(v_colors[pid, 1], v_rgb_local[1])
                        T.atomic_add(v_colors[pid, 2], v_rgb_local[2])
                        if collect_curve_contrib != T.int32(0):
                            T.atomic_add(
                                curve_contrib_rgb[pid, 0], contrib_rgb_local[0]
                            )
                            T.atomic_add(
                                curve_contrib_rgb[pid, 1], contrib_rgb_local[1]
                            )
                            T.atomic_add(
                                curve_contrib_rgb[pid, 2], contrib_rgb_local[2]
                            )
                        if collect_curve_isect_contrib != T.int32(0):
                            isect_idx = gidx
                            T.assume(
                                (isect_idx >= T.int32(0))
                                & (isect_idx < num_isect_contrib)
                            )
                            T.atomic_add(
                                curve_isect_contrib[isect_idx],
                                contrib_rgb_local[0]
                                + contrib_rgb_local[1]
                                + contrib_rgb_local[2],
                            )
                        if warp_any_opacity != T.int32(0):
                            T.atomic_add(v_opacities[pid], v_opacity_local)
                            T.atomic_add(v_inv_sigma2s[pid], v_inv_sigma2_local)

                    if (valid_i32 != T.int32(0)) & (geom_valid_i32 != T.int32(0)):
                        eid = best_eid_local
                        T.assume((eid >= T.int32(0)) & (eid < num_edges))
                        T.atomic_add(v_edges[eid, 0, 0], v_edge_local[0])
                        T.atomic_add(v_edges[eid, 0, 1], v_edge_local[1])
                        T.atomic_add(v_edges[eid, 1, 0], v_edge_local[2])
                        T.atomic_add(v_edges[eid, 1, 1], v_edge_local[3])

    return kernel


def _flatten_cubics_to_edges(
    cubics_px: torch.Tensor,
    primitive_seg_offsets: torch.Tensor,
    *,
    distance_samples: int,
    flatten_method: str = DEFAULT_CUBIC_FLATTEN_METHOD,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Flatten cubic segments into straight edges once per frame.

    Shapes:
    - cubics_px: [num_segments, 4, 2] float32
    - primitive_seg_offsets: [num_primitives + 1] int32
    - edges_px (output): [num_edges, 2, 2] float32
    - primitive_edge_offsets (output): [num_primitives + 1] int32
    - segment_edge_offsets (output; de_casteljau only): [num_segments + 1] int32
    - edge_t_starts (output; de_casteljau only): [num_edges] float32
    - edge_t_ends (output; de_casteljau only): [num_edges] float32
    """
    cubics = _as_f32_contig(cubics_px)
    offs = _as_i32_contig(primitive_seg_offsets).view(-1)
    sample_n = max(2, int(distance_samples))
    edges_per_segment = sample_n - 1
    flatten_method_norm = normalize_cubic_flatten_method(flatten_method)

    if cubics.ndim != 3 or tuple(cubics.shape[1:]) != (4, 2):
        raise ValueError(f"cubics_px must be [N,4,2], got {tuple(cubics.shape)}")
    if offs.ndim != 1 or int(offs.numel()) <= 0:
        raise ValueError(
            f"primitive_seg_offsets must be non-empty 1D tensor, got {tuple(offs.shape)}"
        )

    num_segments = int(cubics.shape[0])
    num_primitives = max(0, int(offs.numel()) - 1)
    edge_offsets = (
        offs.to(dtype=torch.int64)
        .mul(int(edges_per_segment))
        .to(dtype=torch.int32)
        .contiguous()
    )

    if num_segments <= 0 or edges_per_segment <= 0:
        return (
            torch.empty((0, 2, 2), dtype=torch.float32, device=cubics.device),
            edge_offsets,
            torch.zeros((1,), dtype=torch.int32, device=cubics.device),
            None,
            None,
        )

    if flatten_method_norm == CUBIC_FLATTEN_METHOD_BERNSTEIN:
        t = torch.linspace(0.0, 1.0, sample_n, device=cubics.device, dtype=cubics.dtype)
        omt = 1.0 - t
        b0 = omt * omt * omt
        b1 = 3.0 * omt * omt * t
        b2 = 3.0 * omt * t * t
        b3 = t * t * t
        basis = torch.stack([b0, b1, b2, b3], dim=0).view(1, 4, sample_n, 1)
        pts = (cubics.unsqueeze(2) * basis).sum(dim=1)
        edges = torch.stack([pts[:, :-1, :], pts[:, 1:, :]], dim=2).contiguous()
        edges_px = edges.view(-1, 2, 2).contiguous()
        segment_edge_offsets = torch.arange(
            0,
            (num_segments + 1) * edges_per_segment,
            edges_per_segment,
            dtype=torch.int32,
            device=cubics.device,
        ).contiguous()
        edge_t_starts = None
        edge_t_ends = None
    else:
        tol_px = float(DEFAULT_DE_CASTELJAU_FLATNESS_TOL_PX)
        tol_px = max(1e-4, tol_px)
        # Keep worst-case work bounded by distance_samples.
        max_depth = int(math.ceil(math.log2(max(1, edges_per_segment))))
        active_cp = cubics
        active_sid = torch.arange(num_segments, device=cubics.device, dtype=torch.int64)
        active_t0 = torch.zeros(
            (num_segments,), device=cubics.device, dtype=torch.float32
        )
        active_t1 = torch.ones(
            (num_segments,), device=cubics.device, dtype=torch.float32
        )
        leaf_p0: list[torch.Tensor] = []
        leaf_p3: list[torch.Tensor] = []
        leaf_sid: list[torch.Tensor] = []
        leaf_t0: list[torch.Tensor] = []
        leaf_t1: list[torch.Tensor] = []

        for depth in range(max_depth + 1):
            if int(active_cp.shape[0]) <= 0:
                break
            flat = cubic_adaptive_flatness_flags_tilelang(
                active_cp,
                tol_px=float(tol_px),
                force_flat=bool(depth >= max_depth),
                threads=int(DEFAULT_THREADS),
            ).to(torch.bool)

            if bool(flat.any().item()):
                flat_cp = active_cp[flat]
                leaf_p0.append(flat_cp[:, 0, :])
                leaf_p3.append(flat_cp[:, 3, :])
                leaf_sid.append(active_sid[flat])
                leaf_t0.append(active_t0[flat])
                leaf_t1.append(active_t1[flat])

            split = ~flat
            if not bool(split.any().item()):
                break
            split_cp = active_cp[split]
            split_sid = active_sid[split]
            split_t0 = active_t0[split]
            split_t1 = active_t1[split]
            left, right = cubic_split_half_tilelang(
                split_cp,
                threads=int(DEFAULT_THREADS),
            )
            tmid = 0.5 * (split_t0 + split_t1)
            active_cp = torch.cat([left, right], dim=0)
            active_sid = torch.cat([split_sid, split_sid], dim=0)
            active_t0 = torch.cat([split_t0, tmid], dim=0)
            active_t1 = torch.cat([tmid, split_t1], dim=0)

        if not leaf_p0:
            return (
                torch.empty((0, 2, 2), dtype=torch.float32, device=cubics.device),
                edge_offsets,
                torch.zeros(
                    (num_segments + 1,), dtype=torch.int32, device=cubics.device
                ),
                torch.empty((0,), dtype=torch.float32, device=cubics.device),
                torch.empty((0,), dtype=torch.float32, device=cubics.device),
            )

        p0_all = torch.cat(leaf_p0, dim=0).contiguous()
        p3_all = torch.cat(leaf_p3, dim=0).contiguous()
        sid_all = torch.cat(leaf_sid, dim=0).to(torch.int64).contiguous()
        t0_all = torch.cat(leaf_t0, dim=0).to(torch.float32).contiguous()
        t1_all = torch.cat(leaf_t1, dim=0).to(torch.float32).contiguous()

        if int(p0_all.shape[0]) != int(sid_all.shape[0]):
            raise RuntimeError("adaptive flatten internal size mismatch")

        order_base = int((1 << max_depth) + 1)
        t_token = torch.clamp(
            torch.round(t0_all * float(order_base - 1)).to(torch.int64),
            min=0,
            max=order_base - 1,
        )
        sort_key = sid_all * int(order_base) + t_token
        order = torch.argsort(sort_key)
        p0_all = p0_all.index_select(0, order)
        p3_all = p3_all.index_select(0, order)
        sid_all = sid_all.index_select(0, order)
        t0_all = t0_all.index_select(0, order)
        t1_all = t1_all.index_select(0, order)

        edges_px = torch.stack([p0_all, p3_all], dim=1).contiguous()
        seg_edge_counts = torch.bincount(sid_all, minlength=num_segments).to(
            torch.int64
        )
        seg_prefix = torch.zeros(
            (num_segments + 1,), dtype=torch.int64, device=cubics.device
        )
        seg_prefix[1:] = torch.cumsum(seg_edge_counts, dim=0)
        edge_offsets = (
            seg_prefix.index_select(0, offs.to(torch.int64))
            .to(torch.int32)
            .contiguous()
        )
        segment_edge_offsets = seg_prefix.to(torch.int32).contiguous()
        edge_t_starts = t0_all.contiguous()
        edge_t_ends = t1_all.contiguous()

    expected_edges = int(num_segments * edges_per_segment)
    if flatten_method_norm == CUBIC_FLATTEN_METHOD_BERNSTEIN:
        if int(edges_px.shape[0]) != expected_edges:
            raise RuntimeError(
                "flattened edge count mismatch: "
                f"got {int(edges_px.shape[0])}, expected {expected_edges}"
            )
    if num_primitives > 0 and int(edge_offsets[-1].item()) != int(edges_px.shape[0]):
        raise ValueError(
            "primitive_seg_offsets end does not match segment count during flatten: "
            f"edge_offsets[-1]={int(edge_offsets[-1].item())}, expected={int(edges_px.shape[0])}"
        )
    return edges_px, edge_offsets, segment_edge_offsets, edge_t_starts, edge_t_ends


def _build_fill_isect_edge_ranges(
    primitive_ids_sorted: torch.Tensor,
    primitive_edge_offsets: torch.Tensor,
    *,
    num_edges: int,
) -> torch.Tensor:
    _ = num_edges
    offs_i64 = _as_i32_contig(primitive_edge_offsets).to(torch.int64).view(-1)
    prim_ids_i64 = _as_i32_contig(primitive_ids_sorted).to(torch.int64).view(-1)
    edge_start = torch.gather(offs_i64, 0, prim_ids_i64)
    edge_end = torch.gather(offs_i64, 0, prim_ids_i64 + 1)
    return torch.stack([edge_start, edge_end], dim=1).to(torch.int32).contiguous()


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def cubic_adaptive_flatness_flags_kernel(
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
):
    """Compute per-cubic flatness flags for adaptive de Casteljau splitting."""
    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        cubics: T.Tensor[[DYN_NUM_SEGMENTS, 4, 2], dtype],
        out_flat: T.Tensor[[DYN_NUM_SEGMENTS], T.uint8],
        num_segments_i32: T.int32,
        tol_px: T.float32,
        force_flat_i32: T.int32,
    ):
        zero_f = T.Cast(dtype, 0.0)
        one_u8 = T.uint8(1)
        zero_u8 = T.uint8(0)
        eps_f = T.Cast(dtype, 1e-6)
        grid = T.ceildiv(num_segments_i32, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                sid = bx * threads + tx
                if sid < num_segments_i32:
                    if force_flat_i32 != T.int32(0):
                        out_flat[sid] = one_u8
                    else:
                        x0 = cubics[sid, 0, 0]
                        y0 = cubics[sid, 0, 1]
                        x1 = cubics[sid, 1, 0]
                        y1 = cubics[sid, 1, 1]
                        x2 = cubics[sid, 2, 0]
                        y2 = cubics[sid, 2, 1]
                        x3 = cubics[sid, 3, 0]
                        y3 = cubics[sid, 3, 1]

                        lx = x3 - x0
                        ly = y3 - y0
                        chord2 = lx * lx + ly * ly
                        chord = T.sqrt(T.max(chord2, eps_f))

                        v1x = x1 - x0
                        v1y = y1 - y0
                        v2x = x2 - x0
                        v2y = y2 - y0

                        cross1 = T.abs(lx * v1y - ly * v1x)
                        cross2 = T.abs(lx * v2y - ly * v2x)
                        d1 = cross1 / chord
                        d2 = cross2 / chord
                        dmax_line = T.max(d1, d2)

                        n1 = T.sqrt(T.max(v1x * v1x + v1y * v1y, zero_f))
                        n2 = T.sqrt(T.max(v2x * v2x + v2y * v2y, zero_f))
                        n3 = T.sqrt(T.max(chord2, zero_f))
                        dmax_degen = T.max(n1, T.max(n2, n3))
                        dmax = T.if_then_else(chord2 <= eps_f, dmax_degen, dmax_line)

                        out_flat[sid] = T.if_then_else(dmax <= tol_px, one_u8, zero_u8)

    return kernel


def cubic_adaptive_flatness_flags_tilelang(
    cubics: torch.Tensor,
    *,
    tol_px: float,
    force_flat: bool = False,
    threads: int = DEFAULT_THREADS,
) -> torch.Tensor:
    c = _as_f32_contig(cubics)
    n = int(c.shape[0])
    if n <= 0:
        return torch.empty((0,), dtype=torch.uint8, device=c.device)
    if not c.is_cuda:
        p0 = c[:, 0, :]
        p1 = c[:, 1, :]
        p2 = c[:, 2, :]
        p3 = c[:, 3, :]
        line = p3 - p0
        lx = line[:, 0]
        ly = line[:, 1]
        chord2 = lx * lx + ly * ly
        chord = torch.sqrt(torch.clamp(chord2, min=1e-6))
        v1 = p1 - p0
        v2 = p2 - p0
        cross1 = torch.abs(lx * v1[:, 1] - ly * v1[:, 0])
        cross2 = torch.abs(lx * v2[:, 1] - ly * v2[:, 0])
        dmax = torch.maximum(cross1 / chord, cross2 / chord)
        deg = chord2 <= 1e-6
        if bool(deg.any().item()):
            d_alt = torch.maximum(
                torch.linalg.norm(v1, dim=1),
                torch.maximum(
                    torch.linalg.norm(v2, dim=1), torch.sqrt(chord2.clamp_min(0.0))
                ),
            )
            dmax = torch.where(deg, d_alt, dmax)
        if bool(force_flat):
            return torch.ones((n,), dtype=torch.uint8, device=c.device)
        return (dmax <= float(tol_px)).to(torch.uint8).contiguous()
    out = torch.empty((n,), dtype=torch.uint8, device=c.device)
    kernel = cubic_adaptive_flatness_flags_kernel(
        threads=int(threads),
        dtype="float32",
    )
    kernel(c, out, int(n), float(max(1e-4, float(tol_px))), int(bool(force_flat)))
    return out.contiguous()


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def cubic_split_half_kernel(
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
):
    """Split cubics at t=0.5 into left/right child cubics."""
    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        cubics: T.Tensor[[DYN_NUM_SEGMENTS, 4, 2], dtype],
        left_out: T.Tensor[[DYN_NUM_SEGMENTS, 4, 2], dtype],
        right_out: T.Tensor[[DYN_NUM_SEGMENTS, 4, 2], dtype],
        num_segments_i32: T.int32,
    ):
        half = T.Cast(dtype, 0.5)
        grid = T.ceildiv(num_segments_i32, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                sid = bx * threads + tx
                if sid < num_segments_i32:
                    x0 = cubics[sid, 0, 0]
                    y0 = cubics[sid, 0, 1]
                    x1 = cubics[sid, 1, 0]
                    y1 = cubics[sid, 1, 1]
                    x2 = cubics[sid, 2, 0]
                    y2 = cubics[sid, 2, 1]
                    x3 = cubics[sid, 3, 0]
                    y3 = cubics[sid, 3, 1]

                    p01x = half * (x0 + x1)
                    p01y = half * (y0 + y1)
                    p12x = half * (x1 + x2)
                    p12y = half * (y1 + y2)
                    p23x = half * (x2 + x3)
                    p23y = half * (y2 + y3)
                    p012x = half * (p01x + p12x)
                    p012y = half * (p01y + p12y)
                    p123x = half * (p12x + p23x)
                    p123y = half * (p12y + p23y)
                    p0123x = half * (p012x + p123x)
                    p0123y = half * (p012y + p123y)

                    left_out[sid, 0, 0] = x0
                    left_out[sid, 0, 1] = y0
                    left_out[sid, 1, 0] = p01x
                    left_out[sid, 1, 1] = p01y
                    left_out[sid, 2, 0] = p012x
                    left_out[sid, 2, 1] = p012y
                    left_out[sid, 3, 0] = p0123x
                    left_out[sid, 3, 1] = p0123y

                    right_out[sid, 0, 0] = p0123x
                    right_out[sid, 0, 1] = p0123y
                    right_out[sid, 1, 0] = p123x
                    right_out[sid, 1, 1] = p123y
                    right_out[sid, 2, 0] = p23x
                    right_out[sid, 2, 1] = p23y
                    right_out[sid, 3, 0] = x3
                    right_out[sid, 3, 1] = y3

    return kernel


def cubic_split_half_tilelang(
    cubics: torch.Tensor,
    *,
    threads: int = DEFAULT_THREADS,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = _as_f32_contig(cubics)
    n = int(c.shape[0])
    if n <= 0:
        e = torch.empty((0, 4, 2), dtype=torch.float32, device=c.device)
        return e, e
    if not c.is_cuda:
        cp0 = c[:, 0, :]
        cp1 = c[:, 1, :]
        cp2 = c[:, 2, :]
        cp3 = c[:, 3, :]
        p01 = 0.5 * (cp0 + cp1)
        p12 = 0.5 * (cp1 + cp2)
        p23 = 0.5 * (cp2 + cp3)
        p012 = 0.5 * (p01 + p12)
        p123 = 0.5 * (p12 + p23)
        p0123 = 0.5 * (p012 + p123)
        left = torch.stack([cp0, p01, p012, p0123], dim=1).contiguous()
        right = torch.stack([p0123, p123, p23, cp3], dim=1).contiguous()
        return left, right
    left = torch.empty_like(c, dtype=torch.float32)
    right = torch.empty_like(c, dtype=torch.float32)
    kernel = cubic_split_half_kernel(threads=int(threads), dtype="float32")
    kernel(c, left, right, int(n))
    return left.contiguous(), right.contiguous()


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def primitive_tile_ranges_from_segment_ranges_kernel(
    threads: int = DEFAULT_THREADS,
):
    """Reduce segment tile AABBs to primitive tile AABBs from CSR offsets.

    Spec:
    - segment_tile_ranges: [num_segments, 4] int32 => [min_x, min_y, max_x, max_y)
    - primitive_seg_offsets: [num_primitives + 1] int32 CSR offsets over segments
    - primitive_tile_ranges (out): [num_primitives, 4] int32
    - num_tiles_hit (out): [num_primitives] int32, tile area per primitive
    - one thread handles one primitive; segment span is reduced serially.
    """
    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        segment_tile_ranges: T.Tensor[[DYN_NUM_SEGMENTS, 4], T.int32],
        primitive_seg_offsets: T.Tensor[[DYN_NUM_PRIMITIVE_OFFSETS], T.int32],
        primitive_tile_ranges: T.Tensor[[DYN_NUM_PRIMITIVES, 4], T.int32],
        num_tiles_hit: T.Tensor[[DYN_NUM_PRIMITIVES], T.int32],
        num_primitives_i32: T.int32,
        num_segments_i32: T.int32,
        tile_bound_x: T.int32,
        tile_bound_y: T.int32,
    ):
        grid = T.ceildiv(num_primitives_i32, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                pid = bx * threads + tx
                if pid < num_primitives_i32:
                    start = primitive_seg_offsets[pid]
                    end = primitive_seg_offsets[pid + T.int32(1)]

                    if end > start:
                        T.assume(start >= 0)
                        T.assume(end <= num_segments_i32)
                        min_x = T.alloc_var("int32")
                        min_y = T.alloc_var("int32")
                        max_x = T.alloc_var("int32")
                        max_y = T.alloc_var("int32")
                        min_x = tile_bound_x
                        min_y = tile_bound_y
                        max_x = T.int32(0)
                        max_y = T.int32(0)
                        for sid in T.serial(start, end):
                            min_x = T.min(min_x, segment_tile_ranges[sid, 0])
                            min_y = T.min(min_y, segment_tile_ranges[sid, 1])
                            max_x = T.max(max_x, segment_tile_ranges[sid, 2])
                            max_y = T.max(max_y, segment_tile_ranges[sid, 3])

                        tile_w = T.max(T.int32(0), max_x - min_x)
                        tile_h = T.max(T.int32(0), max_y - min_y)
                        primitive_tile_ranges[pid, 0] = min_x
                        primitive_tile_ranges[pid, 1] = min_y
                        primitive_tile_ranges[pid, 2] = max_x
                        primitive_tile_ranges[pid, 3] = max_y
                        num_tiles_hit[pid] = tile_w * tile_h
                    else:
                        primitive_tile_ranges[pid, 0] = T.int32(0)
                        primitive_tile_ranges[pid, 1] = T.int32(0)
                        primitive_tile_ranges[pid, 2] = T.int32(0)
                        primitive_tile_ranges[pid, 3] = T.int32(0)
                        num_tiles_hit[pid] = T.int32(0)

    return kernel


def _primitive_tile_ranges_from_segment_tile_ranges(
    segment_tile_ranges: torch.Tensor,
    primitive_seg_offsets: torch.Tensor,
    tile_bounds: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    num_primitives = int(max(0, int(primitive_seg_offsets.numel()) - 1))
    device = segment_tile_ranges.device
    if num_primitives <= 0:
        return (
            torch.empty((0, 4), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int32, device=device),
        )

    seg_ranges = _as_i32_contig(segment_tile_ranges)
    offs = _as_i32_contig(primitive_seg_offsets).view(-1)

    if int(offs.shape[0]) != num_primitives + 1:
        raise ValueError(
            "primitive_seg_offsets shape mismatch: "
            f"got {int(offs.shape[0])}, expected {num_primitives + 1}"
        )

    tile_bound_x, tile_bound_y, _ = tile_bounds
    tile_ranges = torch.empty((num_primitives, 4), dtype=torch.int32, device=device)
    num_tiles_hit = torch.empty((num_primitives,), dtype=torch.int32, device=device)
    kernel = primitive_tile_ranges_from_segment_ranges_kernel(
        threads=int(DEFAULT_THREADS)
    )
    kernel(
        seg_ranges,
        offs,
        tile_ranges,
        num_tiles_hit,
        int(num_primitives),
        int(seg_ranges.shape[0]),
        int(tile_bound_x),
        int(tile_bound_y),
    )
    return tile_ranges.contiguous(), num_tiles_hit.contiguous()


def rasterize_cubic_fill_forward_tilelang(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    primitive_ids_sorted: torch.Tensor,
    tile_bins: torch.Tensor,
    cubics_px: torch.Tensor,
    primitive_seg_offsets: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    fill_rules: torch.Tensor,
    background: torch.Tensor,
    *,
    aa_widths: torch.Tensor | None = None,
    inv_sigma2s: torch.Tensor | None = None,
    edges_px: torch.Tensor,
    isect_edge_ranges: torch.Tensor,
    aa_width: float = DEFAULT_AA_WIDTH,
    distance_samples: int = DEFAULT_DISTANCE_SAMPLES,
    map_threads: int = DEFAULT_THREADS,
):
    # Kept for API compatibility; no longer used after grouped-edge path removal.
    _ = map_threads
    img_width, img_height, _ = img_size
    num_primitives = int(colors.shape[0])
    num_intersects = int(primitive_ids_sorted.shape[0])
    out_img = torch.empty(
        (img_height, img_width, 3), dtype=torch.float32, device=cubics_px.device
    )
    final_Ts = torch.empty(
        (img_height, img_width), dtype=torch.float32, device=cubics_px.device
    )
    final_idx = torch.empty(
        (img_height, img_width), dtype=torch.int32, device=cubics_px.device
    )

    if num_intersects <= 0 or num_primitives <= 0:
        bg = background.to(device=out_img.device, dtype=out_img.dtype).view(1, 1, 3)
        out_img.copy_(bg.expand(img_height, img_width, 3))
        final_Ts.fill_(1.0)
        final_idx.fill_(0)
        return out_img, final_Ts, final_idx

    if inv_sigma2s is None:
        if aa_widths is None:
            aa_widths_in = torch.full(
                (num_primitives,),
                float(aa_width),
                device=cubics_px.device,
                dtype=torch.float32,
            )
        else:
            aa_widths_in = aa_widths.to(
                device=cubics_px.device, dtype=torch.float32
            ).view(-1)
            if int(aa_widths_in.numel()) == 1 and num_primitives > 1:
                aa_widths_in = aa_widths_in.expand(num_primitives)
            if int(aa_widths_in.numel()) != num_primitives:
                raise ValueError(
                    f"aa_widths size must match num_primitives ({num_primitives}), got {int(aa_widths_in.numel())}"
                )
        aa_widths_in = torch.clamp(_as_f32_contig(aa_widths_in), min=1e-4)
        inv_sigma2s_in = 1.0 / torch.clamp(aa_widths_in * aa_widths_in, min=1e-6)
    else:
        inv_sigma2s_in = inv_sigma2s.to(
            device=cubics_px.device, dtype=torch.float32
        ).view(-1)
        if int(inv_sigma2s_in.numel()) == 1 and num_primitives > 1:
            inv_sigma2s_in = inv_sigma2s_in.expand(num_primitives)
        if int(inv_sigma2s_in.numel()) != num_primitives:
            raise ValueError(
                f"inv_sigma2s size must match num_primitives ({num_primitives}), got {int(inv_sigma2s_in.numel())}"
            )
        inv_sigma2s_in = _as_f32_contig(inv_sigma2s_in)

    edges_px = _as_f32_contig(edges_px)
    if int(edges_px.shape[0]) <= 0:
        bg = background.to(device=out_img.device, dtype=out_img.dtype).view(1, 1, 3)
        out_img.copy_(bg.expand(img_height, img_width, 3))
        final_Ts.fill_(1.0)
        final_idx.fill_(0)
        return out_img, final_Ts, final_idx

    isect_edge_ranges_i32 = _as_i32_contig(isect_edge_ranges)
    if isect_edge_ranges_i32.ndim != 2 or int(isect_edge_ranges_i32.shape[1]) != 2:
        raise ValueError(
            f"isect_edge_ranges must be [num_intersects,2], got {tuple(isect_edge_ranges_i32.shape)}"
        )
    if int(isect_edge_ranges_i32.shape[0]) != num_intersects:
        raise ValueError(
            "isect_edge_ranges first dim must match num_intersects "
            f"({num_intersects}), got {int(isect_edge_ranges_i32.shape[0])}"
        )

    kernel = rasterize_cubic_fill_forward_kernel(
        block_x=int(block[0]),
        block_y=int(block[1]),
        dtype="float32",
    )

    primitive_ids_i32 = _as_i32_contig(primitive_ids_sorted).view(-1)
    tile_bins_i32 = _as_i32_contig(tile_bins)
    edges_px_f32 = _as_f32_contig(edges_px)
    colors_f32 = _as_f32_contig(colors)
    opacities_f32 = _as_f32_contig(opacities).view(-1)
    fill_rules_i32 = _as_i32_contig(fill_rules).view(-1)
    background_f32 = _as_f32_contig(
        background.to(device=cubics_px.device, dtype=torch.float32).view(3)
    )

    kernel(
        primitive_ids_i32,
        tile_bins_i32,
        edges_px_f32,
        isect_edge_ranges_i32,
        colors_f32,
        opacities_f32,
        inv_sigma2s_in.view(-1),
        fill_rules_i32,
        background_f32,
        out_img,
        final_Ts,
        final_idx,
    )
    return out_img, final_Ts, final_idx


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def _accumulate_edge_grads_to_cubics_kernel(
    threads: int = DEFAULT_THREADS,
    edges_per_segment: int = DEFAULT_DISTANCE_SAMPLES - 1,
    dtype: str = "float32",
):
    """Accumulate edge endpoint grads into cubic control-point grads.

    Spec:
    - v_edges: [num_segments, edges_per_segment, 2, 2] float32
      axis2: 0=start endpoint grad, 1=end endpoint grad
      axis3: xy components
    - v_cubics: [num_segments, 4, 2] float32
      accumulated grads for cubic control points [P0..P3]
    - target: one thread accumulates one segment.
    """
    if threads <= 0:
        raise ValueError("threads must be > 0")
    if int(edges_per_segment) <= 0:
        raise ValueError("edges_per_segment must be > 0")

    @T.prim_func
    def kernel(
        v_edges: T.Tensor[[DYN_NUM_SEGMENTS, edges_per_segment, 2, 2], dtype],
        v_cubics: T.Tensor[[DYN_NUM_SEGMENTS, 4, 2], dtype],
        num_segments_i32: T.int32,
    ):
        one_f = T.Cast(dtype, 1.0)
        three_f = T.Cast(dtype, 3.0)
        zero_f = T.Cast(dtype, 0.0)
        inv_e = one_f / T.Cast(dtype, edges_per_segment)
        grid = T.ceildiv(num_segments_i32, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                sid = bx * threads + tx
                if sid < num_segments_i32:
                    acc00 = T.alloc_var(dtype, init=zero_f)
                    acc01 = T.alloc_var(dtype, init=zero_f)
                    acc10 = T.alloc_var(dtype, init=zero_f)
                    acc11 = T.alloc_var(dtype, init=zero_f)
                    acc20 = T.alloc_var(dtype, init=zero_f)
                    acc21 = T.alloc_var(dtype, init=zero_f)
                    acc30 = T.alloc_var(dtype, init=zero_f)
                    acc31 = T.alloc_var(dtype, init=zero_f)

                    for e in T.serial(0, edges_per_segment):
                        t0 = T.Cast(dtype, e) * inv_e
                        t1 = T.Cast(dtype, e + 1) * inv_e
                        omt0 = one_f - t0
                        omt1 = one_f - t1
                        omt0_2 = omt0 * omt0
                        omt1_2 = omt1 * omt1
                        t0_2 = t0 * t0
                        t1_2 = t1 * t1

                        b00 = omt0_2 * omt0
                        b01 = three_f * omt0_2 * t0
                        b02 = three_f * omt0 * t0_2
                        b03 = t0_2 * t0

                        b10 = omt1_2 * omt1
                        b11 = three_f * omt1_2 * t1
                        b12 = three_f * omt1 * t1_2
                        b13 = t1_2 * t1

                        gsx = v_edges[sid, e, 0, 0]
                        gsy = v_edges[sid, e, 0, 1]
                        gex = v_edges[sid, e, 1, 0]
                        gey = v_edges[sid, e, 1, 1]

                        acc00 = acc00 + gsx * b00 + gex * b10
                        acc01 = acc01 + gsy * b00 + gey * b10
                        acc10 = acc10 + gsx * b01 + gex * b11
                        acc11 = acc11 + gsy * b01 + gey * b11
                        acc20 = acc20 + gsx * b02 + gex * b12
                        acc21 = acc21 + gsy * b02 + gey * b12
                        acc30 = acc30 + gsx * b03 + gex * b13
                        acc31 = acc31 + gsy * b03 + gey * b13

                    v_cubics[sid, 0, 0] = acc00
                    v_cubics[sid, 0, 1] = acc01
                    v_cubics[sid, 1, 0] = acc10
                    v_cubics[sid, 1, 1] = acc11
                    v_cubics[sid, 2, 0] = acc20
                    v_cubics[sid, 2, 1] = acc21
                    v_cubics[sid, 3, 0] = acc30
                    v_cubics[sid, 3, 1] = acc31

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def _accumulate_edge_grads_to_cubics_csr_kernel(
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
):
    """Accumulate edge grads into cubics using CSR segment->edge offsets.

    Spec:
    - v_edges: [num_edges,2,2] float32
    - edge_t_starts/edge_t_ends: [num_edges] float32
    - segment_edge_offsets: [num_segments+1] int32 CSR row pointers
    - v_cubics: [num_segments,4,2] float32
    """
    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        v_edges: T.Tensor[[DYN_NUM_EDGES, 2, 2], dtype],
        edge_t_starts: T.Tensor[[DYN_NUM_EDGES], dtype],
        edge_t_ends: T.Tensor[[DYN_NUM_EDGES], dtype],
        segment_edge_offsets: T.Tensor[[DYN_NUM_SEGMENT_OFFSETS], T.int32],
        v_cubics: T.Tensor[[DYN_NUM_SEGMENTS, 4, 2], dtype],
        num_segments_i32: T.int32,
    ):
        one_f = T.Cast(dtype, 1.0)
        three_f = T.Cast(dtype, 3.0)
        zero_f = T.Cast(dtype, 0.0)
        grid = T.ceildiv(num_segments_i32, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                sid = bx * threads + tx
                if sid < num_segments_i32:
                    e_start = segment_edge_offsets[sid]
                    e_end = segment_edge_offsets[sid + T.int32(1)]

                    acc00 = T.alloc_var(dtype, init=zero_f)
                    acc01 = T.alloc_var(dtype, init=zero_f)
                    acc10 = T.alloc_var(dtype, init=zero_f)
                    acc11 = T.alloc_var(dtype, init=zero_f)
                    acc20 = T.alloc_var(dtype, init=zero_f)
                    acc21 = T.alloc_var(dtype, init=zero_f)
                    acc30 = T.alloc_var(dtype, init=zero_f)
                    acc31 = T.alloc_var(dtype, init=zero_f)

                    for e in T.serial(e_start, e_end):
                        t0 = edge_t_starts[e]
                        t1 = edge_t_ends[e]
                        omt0 = one_f - t0
                        omt1 = one_f - t1
                        omt0_2 = omt0 * omt0
                        omt1_2 = omt1 * omt1
                        t0_2 = t0 * t0
                        t1_2 = t1 * t1

                        b00 = omt0_2 * omt0
                        b01 = three_f * omt0_2 * t0
                        b02 = three_f * omt0 * t0_2
                        b03 = t0_2 * t0

                        b10 = omt1_2 * omt1
                        b11 = three_f * omt1_2 * t1
                        b12 = three_f * omt1 * t1_2
                        b13 = t1_2 * t1

                        gsx = v_edges[e, 0, 0]
                        gsy = v_edges[e, 0, 1]
                        gex = v_edges[e, 1, 0]
                        gey = v_edges[e, 1, 1]

                        acc00 = acc00 + gsx * b00 + gex * b10
                        acc01 = acc01 + gsy * b00 + gey * b10
                        acc10 = acc10 + gsx * b01 + gex * b11
                        acc11 = acc11 + gsy * b01 + gey * b11
                        acc20 = acc20 + gsx * b02 + gex * b12
                        acc21 = acc21 + gsy * b02 + gey * b12
                        acc30 = acc30 + gsx * b03 + gex * b13
                        acc31 = acc31 + gsy * b03 + gey * b13

                    v_cubics[sid, 0, 0] = acc00
                    v_cubics[sid, 0, 1] = acc01
                    v_cubics[sid, 1, 0] = acc10
                    v_cubics[sid, 1, 1] = acc11
                    v_cubics[sid, 2, 0] = acc20
                    v_cubics[sid, 2, 1] = acc21
                    v_cubics[sid, 3, 0] = acc30
                    v_cubics[sid, 3, 1] = acc31

    return kernel


def _accumulate_edge_grads_to_cubics(
    v_edges: torch.Tensor,
    num_segments: int,
    *,
    distance_samples: int,
    flatten_method: str = DEFAULT_CUBIC_FLATTEN_METHOD,
    segment_edge_offsets: torch.Tensor | None = None,
    edge_t_starts: torch.Tensor | None = None,
    edge_t_ends: torch.Tensor | None = None,
    threads: int = DEFAULT_THREADS,
) -> torch.Tensor:
    sample_n = max(2, int(distance_samples))
    edges_per_segment = sample_n - 1
    dev = v_edges.device
    flatten_method_norm = normalize_cubic_flatten_method(flatten_method)
    if int(num_segments) <= 0:
        return torch.empty((0, 4, 2), dtype=torch.float32, device=dev)
    if flatten_method_norm == CUBIC_FLATTEN_METHOD_BERNSTEIN:
        expected_edges = int(num_segments) * int(edges_per_segment)
        if int(v_edges.shape[0]) != expected_edges:
            raise ValueError(
                "v_edges size mismatch: "
                f"got {int(v_edges.shape[0])}, expected {expected_edges}"
            )
        if expected_edges <= 0:
            return torch.zeros((num_segments, 4, 2), dtype=torch.float32, device=dev)
        v_edges_seg = _as_f32_contig(v_edges).view(
            num_segments, edges_per_segment, 2, 2
        )
        v_cubics = torch.empty((num_segments, 4, 2), dtype=torch.float32, device=dev)
        kernel = _accumulate_edge_grads_to_cubics_kernel(
            threads=int(threads),
            edges_per_segment=int(edges_per_segment),
            dtype="float32",
        )
        kernel(v_edges_seg, v_cubics, int(num_segments))
        return v_cubics.contiguous()

    if segment_edge_offsets is None or edge_t_starts is None or edge_t_ends is None:
        raise ValueError(
            "de_casteljau flatten backward requires CSR metadata "
            "(segment_edge_offsets, edge_t_starts, edge_t_ends)"
        )
    seg_offs_i32 = _as_i32_contig(segment_edge_offsets).view(-1)
    if int(seg_offs_i32.numel()) != int(num_segments) + 1:
        raise ValueError(
            "segment_edge_offsets size mismatch: "
            f"got {int(seg_offs_i32.numel())}, expected {int(num_segments) + 1}"
        )
    if int(seg_offs_i32[0].item()) != 0:
        raise ValueError(
            f"segment_edge_offsets must start at 0, got {int(seg_offs_i32[0].item())}"
        )
    expected_edges = int(v_edges.shape[0])
    if int(seg_offs_i32[-1].item()) != expected_edges:
        raise ValueError(
            "segment_edge_offsets end must equal v_edges size, "
            f"got {int(seg_offs_i32[-1].item())} vs {expected_edges}"
        )
    t0 = _as_f32_contig(edge_t_starts).view(-1)
    t1 = _as_f32_contig(edge_t_ends).view(-1)
    if int(t0.numel()) != expected_edges or int(t1.numel()) != expected_edges:
        raise ValueError(
            "edge_t_starts/edge_t_ends size mismatch with v_edges: "
            f"{int(t0.numel())}, {int(t1.numel())} vs {expected_edges}"
        )
    v_cubics = torch.empty((num_segments, 4, 2), dtype=torch.float32, device=dev)
    kernel = _accumulate_edge_grads_to_cubics_csr_kernel(
        threads=int(threads),
        dtype="float32",
    )
    kernel(
        _as_f32_contig(v_edges),
        t0,
        t1,
        seg_offs_i32,
        v_cubics,
        int(num_segments),
    )
    return v_cubics.contiguous()


def rasterize_cubic_fill_backward_tilelang(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    primitive_ids_sorted: torch.Tensor,
    tile_bins: torch.Tensor,
    cubics_px: torch.Tensor,
    primitive_seg_offsets: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    fill_rules: torch.Tensor,
    background: torch.Tensor,
    final_Ts: torch.Tensor,
    final_idx: torch.Tensor,
    v_output: torch.Tensor,
    *,
    aa_widths: torch.Tensor | None = None,
    inv_sigma2s: torch.Tensor | None = None,
    edges_px: torch.Tensor,
    isect_edge_ranges: torch.Tensor,
    segment_edge_offsets: torch.Tensor | None = None,
    edge_t_starts: torch.Tensor | None = None,
    edge_t_ends: torch.Tensor | None = None,
    aa_width: float = DEFAULT_AA_WIDTH,
    distance_samples: int = DEFAULT_DISTANCE_SAMPLES,
    flatten_method: str = DEFAULT_CUBIC_FLATTEN_METHOD,
    map_threads: int = DEFAULT_THREADS,
    curve_contrib_rgb: torch.Tensor | None = None,
    curve_isect_contrib: torch.Tensor | None = None,
):
    # Kept for API compatibility; no longer used after grouped-edge path removal.
    _ = (tile_bounds, map_threads)
    block_x, block_y, _ = block
    _img_w, _img_h, _ = img_size
    num_primitives = int(colors.shape[0])
    num_segments = int(cubics_px.shape[0])

    v_colors = torch.zeros_like(colors, dtype=torch.float32)
    v_opacities = torch.zeros_like(opacities.view(-1), dtype=torch.float32)
    v_inv_sigma2s = torch.zeros(
        (num_primitives,), dtype=torch.float32, device=colors.device
    )
    v_cubics_px = torch.zeros_like(cubics_px, dtype=torch.float32)
    if (
        int(primitive_ids_sorted.numel()) <= 0
        or num_primitives <= 0
        or num_segments <= 0
    ):
        return v_cubics_px, v_colors, v_opacities, v_inv_sigma2s

    if inv_sigma2s is None:
        if aa_widths is None:
            aa_widths_in = torch.full(
                (num_primitives,),
                float(aa_width),
                device=cubics_px.device,
                dtype=torch.float32,
            )
        else:
            aa_widths_in = aa_widths.to(
                device=cubics_px.device, dtype=torch.float32
            ).view(-1)
            if int(aa_widths_in.numel()) == 1 and num_primitives > 1:
                aa_widths_in = aa_widths_in.expand(num_primitives)
            if int(aa_widths_in.numel()) != num_primitives:
                raise ValueError(
                    f"aa_widths size must match num_primitives ({num_primitives}), got {int(aa_widths_in.numel())}"
                )
        aa_widths_in = torch.clamp(_as_f32_contig(aa_widths_in), min=1e-4)
        inv_sigma2s_in = 1.0 / torch.clamp(aa_widths_in * aa_widths_in, min=1e-6)
    else:
        inv_sigma2s_in = inv_sigma2s.to(
            device=cubics_px.device, dtype=torch.float32
        ).view(-1)
        if int(inv_sigma2s_in.numel()) == 1 and num_primitives > 1:
            inv_sigma2s_in = inv_sigma2s_in.expand(num_primitives)
        if int(inv_sigma2s_in.numel()) != num_primitives:
            raise ValueError(
                f"inv_sigma2s size must match num_primitives ({num_primitives}), got {int(inv_sigma2s_in.numel())}"
            )
        inv_sigma2s_in = _as_f32_contig(inv_sigma2s_in)

    edges_px = _as_f32_contig(edges_px)
    if int(edges_px.shape[0]) <= 0:
        return v_cubics_px, v_colors, v_opacities, v_inv_sigma2s

    isect_edge_ranges_i32 = _as_i32_contig(isect_edge_ranges)
    if isect_edge_ranges_i32.ndim != 2 or int(isect_edge_ranges_i32.shape[1]) != 2:
        raise ValueError(
            f"isect_edge_ranges must be [num_intersects,2], got {tuple(isect_edge_ranges_i32.shape)}"
        )
    if int(isect_edge_ranges_i32.shape[0]) != int(primitive_ids_sorted.shape[0]):
        raise ValueError(
            "isect_edge_ranges first dim must match num_intersects "
            f"({int(primitive_ids_sorted.shape[0])}), got {int(isect_edge_ranges_i32.shape[0])}"
        )

    v_edges = torch.zeros_like(edges_px, dtype=torch.float32)
    kernel = rasterize_cubic_fill_backward_kernel(
        block_x=int(block_x),
        block_y=int(block_y),
        dtype="float32",
    )
    primitive_ids_i32 = _as_i32_contig(primitive_ids_sorted).view(-1)
    tile_bins_i32 = _as_i32_contig(tile_bins)
    edges_px_f32 = _as_f32_contig(edges_px)
    colors_f32 = _as_f32_contig(colors)
    opacities_f32 = _as_f32_contig(opacities).view(-1)
    fill_rules_i32 = _as_i32_contig(fill_rules).view(-1)
    background_f32 = _as_f32_contig(
        background.to(device=cubics_px.device, dtype=torch.float32).view(3)
    )
    final_Ts_f32 = _as_f32_contig(final_Ts)
    final_idx_i32 = _as_i32_contig(final_idx)
    v_output_f32 = _as_f32_contig(v_output)
    if curve_contrib_rgb is None:
        curve_contrib_rgb_tensor = torch.empty_like(colors_f32, dtype=torch.float32)
        collect_curve_contrib_i32 = 0
    else:
        if curve_contrib_rgb.shape != colors.shape:
            raise ValueError(
                "curve_contrib_rgb must match colors shape "
                f"{tuple(colors.shape)}, got {tuple(curve_contrib_rgb.shape)}"
            )
        curve_contrib_rgb_tensor = curve_contrib_rgb.to(
            device=colors.device, dtype=torch.float32
        ).contiguous()
        collect_curve_contrib_i32 = 1
    if curve_isect_contrib is None:
        curve_isect_contrib_tensor = torch.empty(
            (1,), device=colors.device, dtype=torch.float32
        )
        collect_curve_isect_contrib_i32 = 0
    else:
        if curve_isect_contrib.ndim != 1:
            raise ValueError("curve_isect_contrib must be a 1D tensor")
        if int(curve_isect_contrib.numel()) != int(primitive_ids_i32.numel()):
            raise ValueError("curve_isect_contrib size must match primitive_ids_sorted")
        curve_isect_contrib_tensor = curve_isect_contrib.to(
            device=colors.device, dtype=torch.float32
        ).contiguous()
        collect_curve_isect_contrib_i32 = 1

    kernel(
        primitive_ids_i32,
        tile_bins_i32,
        edges_px_f32,
        isect_edge_ranges_i32,
        colors_f32,
        opacities_f32,
        inv_sigma2s_in.view(-1),
        fill_rules_i32,
        background_f32,
        final_Ts_f32,
        final_idx_i32,
        v_output_f32,
        v_edges,
        v_colors,
        v_opacities,
        v_inv_sigma2s,
        int(collect_curve_contrib_i32),
        curve_contrib_rgb_tensor,
        int(collect_curve_isect_contrib_i32),
        curve_isect_contrib_tensor,
    )

    v_cubics_px = _accumulate_edge_grads_to_cubics(
        v_edges,
        num_segments,
        distance_samples=int(distance_samples),
        flatten_method=str(flatten_method),
        segment_edge_offsets=segment_edge_offsets,
        edge_t_starts=edge_t_starts,
        edge_t_ends=edge_t_ends,
    )
    return v_cubics_px, v_colors, v_opacities, v_inv_sigma2s


class _CubicFillSplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cubics_norm: torch.Tensor,
        primitive_seg_offsets: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        fill_rules: torch.Tensor,
        img_h: int,
        img_w: int,
        block_x: int,
        block_y: int,
        background: torch.Tensor,
        project_threads: int,
        map_threads: int,
        aabb_pad: float,
        aniso_intersects: bool,
        aa_widths: torch.Tensor,
        distance_samples: int,
        flatten_method: str = DEFAULT_CUBIC_FLATTEN_METHOD,
        process_fn: Optional[CubicFillProcessFn] = None,
    ):
        flatten_method_norm = normalize_cubic_flatten_method(flatten_method)
        num_segments = int(cubics_norm.shape[0])
        offs = _as_i32_contig(primitive_seg_offsets).view(-1)
        num_primitives = int(max(0, int(offs.numel()) - 1))
        tile_bounds = (
            (int(img_w) + int(block_x) - 1) // int(block_x),
            (int(img_h) + int(block_y) - 1) // int(block_y),
            1,
        )
        block = (int(block_x), int(block_y), 1)
        img_size = (int(img_w), int(img_h), 1)
        bg = background.to(device=cubics_norm.device, dtype=torch.float32).view(3)
        if num_primitives <= 0:
            out = bg.view(1, 1, 3).expand(int(img_h), int(img_w), 3).clone()
            ctx.empty = True
            ctx.img_h = int(img_h)
            ctx.img_w = int(img_w)
            return out

        if int(colors.shape[0]) != num_primitives:
            raise ValueError(
                f"colors first dim must match primitive count ({num_primitives}), got {int(colors.shape[0])}"
            )
        if int(opacities.view(-1).shape[0]) != num_primitives:
            raise ValueError(
                f"opacities size must match primitive count ({num_primitives}), got {int(opacities.view(-1).shape[0])}"
            )
        if int(depths.view(-1).shape[0]) != num_primitives:
            raise ValueError(
                f"depths size must match primitive count ({num_primitives}), got {int(depths.view(-1).shape[0])}"
            )
        if int(fill_rules.view(-1).shape[0]) != num_primitives:
            raise ValueError(
                f"fill_rules size must match primitive count ({num_primitives}), got {int(fill_rules.view(-1).shape[0])}"
            )
        if int(offs[0].item()) != 0 or int(offs[-1].item()) != num_segments:
            raise ValueError(
                "primitive_seg_offsets must start at 0 and end at num_segments "
                f"(got start={int(offs[0].item())}, end={int(offs[-1].item())}, num_segments={num_segments})"
            )

        widths = aa_widths.to(device=cubics_norm.device, dtype=torch.float32).view(-1)
        if int(widths.numel()) != num_primitives:
            raise ValueError(
                f"aa_widths size must match primitive count ({num_primitives}), got {int(widths.numel())}"
            )
        widths = torch.clamp(_as_f32_contig(widths), min=1e-4)
        inv_sigma2s = (1.0 / torch.clamp(widths * widths, min=1e-6)).contiguous()
        support_radii_prim = widths * 3.0

        seg_counts = (offs[1:] - offs[:-1]).to(torch.int64).clamp_min(0)
        support_radii_seg = torch.repeat_interleave(
            support_radii_prim, seg_counts
        ).contiguous()
        if int(support_radii_seg.numel()) != num_segments:
            raise ValueError(
                "primitive_seg_offsets and cubics size mismatch when expanding support radii: "
                f"{int(support_radii_seg.numel())} vs {num_segments}"
            )

        stroke_zeros = torch.zeros(
            (num_segments,), dtype=torch.float32, device=cubics_norm.device
        )
        cubics_px, seg_tile_ranges, _ = project_cubics_2d_forward_tilelang(
            cubics_norm,
            stroke_zeros,
            support_radii_seg,
            int(img_h),
            int(img_w),
            tile_bounds,
            block_x=int(block_x),
            block_y=int(block_y),
            threads=int(project_threads),
            aabb_pad=float(aabb_pad),
            aniso_intersects=bool(aniso_intersects),
        )
        (
            edges_px,
            primitive_edge_offsets,
            segment_edge_offsets,
            edge_t_starts,
            edge_t_ends,
        ) = _flatten_cubics_to_edges(
            cubics_px,
            offs,
            distance_samples=int(distance_samples),
            flatten_method=flatten_method_norm,
        )

        primitive_tile_ranges, num_tiles_hit = (
            _primitive_tile_ranges_from_segment_tile_ranges(
                seg_tile_ranges,
                offs,
                tile_bounds,
            )
        )
        if int(num_tiles_hit.numel()) <= 0:
            out = bg.view(1, 1, 3).expand(int(img_h), int(img_w), 3).clone()
            ctx.empty = True
            ctx.img_h = int(img_h)
            ctx.img_w = int(img_w)
            return out

        cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
        num_intersects = (
            int(cum_tiles_hit[-1].item()) if int(cum_tiles_hit.numel()) > 0 else 0
        )
        if num_intersects <= 0:
            out = bg.view(1, 1, 3).expand(int(img_h), int(img_w), 3).clone()
            ctx.empty = True
            ctx.img_h = int(img_h)
            ctx.img_w = int(img_w)
            return out

        isect_ids, primitive_ids = map_cubic_to_intersects_tilelang(
            primitive_tile_ranges,
            depths.view(-1),
            cum_tiles_hit,
            tile_bounds,
            threads=int(map_threads),
        )
        isect_ids_sorted, sorted_idx = torch.sort(isect_ids)
        primitive_ids_sorted = torch.gather(primitive_ids, 0, sorted_idx)
        isect_edge_ranges = _build_fill_isect_edge_ranges(
            primitive_ids_sorted,
            primitive_edge_offsets,
            num_edges=int(edges_px.shape[0]),
        )
        num_tiles = int(tile_bounds[0] * tile_bounds[1])
        tile_bins = get_tile_bin_edges_tilelang(
            int(num_intersects),
            isect_ids_sorted,
            num_tiles=num_tiles,
            compact=True,
        )

        out_img, final_Ts, final_idx = rasterize_cubic_fill_forward_tilelang(
            tile_bounds,
            block,
            img_size,
            primitive_ids_sorted,
            tile_bins,
            cubics_px,
            offs,
            colors,
            opacities.view(-1),
            fill_rules.view(-1),
            bg,
            inv_sigma2s=inv_sigma2s,
            edges_px=edges_px,
            isect_edge_ranges=isect_edge_ranges,
            distance_samples=int(distance_samples),
            map_threads=int(map_threads),
        )

        ctx.empty = False
        ctx.tile_bounds = tile_bounds
        ctx.block = block
        ctx.img_size = img_size
        ctx.img_h = int(img_h)
        ctx.img_w = int(img_w)
        ctx.map_threads = int(map_threads)
        ctx.distance_samples = int(distance_samples)
        ctx.flatten_method = flatten_method_norm
        ctx.process_fn = process_fn
        # Cache large flattened edge tensors on ctx attributes to avoid
        # save_for_backward packing overhead on every step.
        ctx.cached_edges_px = edges_px
        ctx.cached_isect_edge_ranges = isect_edge_ranges
        ctx.cached_segment_edge_offsets = segment_edge_offsets
        ctx.cached_edge_t_starts = edge_t_starts
        ctx.cached_edge_t_ends = edge_t_ends
        ctx.save_for_backward(
            cubics_norm,
            offs,
            colors,
            opacities.view(-1),
            depths.view(-1),
            _as_i32_contig(fill_rules).view(-1),
            widths,
            inv_sigma2s,
            primitive_ids_sorted,
            tile_bins,
            cubics_px,
            bg,
            final_Ts,
            final_idx,
        )
        return out_img

    @staticmethod
    def backward(ctx, grad_out_img: torch.Tensor):
        if bool(getattr(ctx, "empty", False)):
            img_h = int(getattr(ctx, "img_h", 1))
            img_w = int(getattr(ctx, "img_w", 1))
            _ = (img_h, img_w)
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        (
            cubics_norm,
            primitive_seg_offsets,
            colors,
            opacities,
            depths,
            fill_rules,
            aa_widths,
            inv_sigma2s,
            primitive_ids_sorted,
            tile_bins,
            cubics_px,
            background,
            final_Ts,
            final_idx,
        ) = ctx.saved_tensors
        edges_px = ctx.cached_edges_px
        isect_edge_ranges = ctx.cached_isect_edge_ranges
        segment_edge_offsets = getattr(ctx, "cached_segment_edge_offsets", None)
        edge_t_starts = getattr(ctx, "cached_edge_t_starts", None)
        edge_t_ends = getattr(ctx, "cached_edge_t_ends", None)
        process_fn = getattr(ctx, "process_fn", None)
        pre_bwd_payload: CubicFillProcessPayload = {
            "collect_curve_contrib_rgb": False,
            "collect_curve_isect_contrib": False,
            "num_primitives": int(colors.shape[0]),
            "colors": colors,
        }
        pre_bwd_payload = _run_cubic_process_fn(
            process_fn,
            "pre_backward_rasterize",
            pre_bwd_payload,
        )
        collect_curve_contrib = bool(
            pre_bwd_payload.get("collect_curve_contrib_rgb", False)
        )
        collect_curve_isect_contrib = bool(
            pre_bwd_payload.get("collect_curve_isect_contrib", False)
        )
        curve_contrib_rgb = (
            torch.zeros_like(colors, dtype=torch.float32)
            if collect_curve_contrib
            else None
        )
        curve_isect_contrib = (
            torch.zeros(
                (int(primitive_ids_sorted.numel()),),
                dtype=torch.float32,
                device=colors.device,
            )
            if collect_curve_isect_contrib
            else None
        )

        v_cubics_px, v_colors, v_opacity, v_inv_sigma2 = (
            rasterize_cubic_fill_backward_tilelang(
                ctx.tile_bounds,
                ctx.block,
                ctx.img_size,
                primitive_ids_sorted,
                tile_bins,
                cubics_px,
                primitive_seg_offsets,
                colors,
                opacities,
                fill_rules,
                background,
                final_Ts,
                final_idx,
                grad_out_img,
                inv_sigma2s=inv_sigma2s,
                edges_px=edges_px,
                isect_edge_ranges=isect_edge_ranges,
                segment_edge_offsets=segment_edge_offsets,
                edge_t_starts=edge_t_starts,
                edge_t_ends=edge_t_ends,
                distance_samples=int(ctx.distance_samples),
                flatten_method=str(getattr(ctx, "flatten_method", "bernstein")),
                map_threads=int(ctx.map_threads),
                curve_contrib_rgb=curve_contrib_rgb,
                curve_isect_contrib=curve_isect_contrib,
            )
        )
        post_bwd_payload: CubicFillProcessPayload = {
            "v_cubics_px": v_cubics_px,
            "v_colors": v_colors,
            "v_opacity": v_opacity,
            "curve_contrib_rgb": curve_contrib_rgb,
            "curve_isect_contrib": curve_isect_contrib,
            "primitive_ids_sorted": primitive_ids_sorted,
            "tile_bins": tile_bins,
            "colors": colors,
            "opacities": opacities,
            "primitive_seg_offsets": primitive_seg_offsets,
        }
        post_bwd_payload = _run_cubic_process_fn(
            process_fn,
            "post_backward_rasterize",
            post_bwd_payload,
        )
        v_cubics_px = post_bwd_payload["v_cubics_px"]
        v_colors = post_bwd_payload["v_colors"]
        v_opacity = post_bwd_payload["v_opacity"]

        half_w = 0.5 * float(ctx.img_w)
        half_h = 0.5 * float(ctx.img_h)
        v_cubics_norm = torch.empty_like(cubics_norm, dtype=torch.float32)
        v_cubics_norm[..., 0] = v_cubics_px[..., 0] * half_w
        v_cubics_norm[..., 1] = v_cubics_px[..., 1] * half_h

        v_depths = torch.zeros_like(depths, dtype=torch.float32)
        denom = torch.clamp(aa_widths * aa_widths * aa_widths, min=1e-12)
        v_aa_widths = (-2.0 * v_inv_sigma2 / denom).contiguous()
        return (
            v_cubics_norm,
            None,
            v_colors,
            v_opacity.view_as(opacities),
            v_depths,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            v_aa_widths,
            None,
            None,
            None,
        )


def render_cubic_fill_splat_tilelang(
    cubics_norm: torch.Tensor,
    primitive_seg_offsets: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    img_h: int,
    img_w: int,
    block_x: int,
    block_y: int,
    background: torch.Tensor,
    *,
    fill_rules: torch.Tensor | None = None,
    project_threads: int = DEFAULT_THREADS,
    map_threads: int = DEFAULT_THREADS,
    aabb_pad: float = DEFAULT_AABB_PAD,
    aniso_intersects: bool = False,
    aa_widths: torch.Tensor | None = None,
    aa_width: float = DEFAULT_AA_WIDTH,
    distance_samples: int = DEFAULT_DISTANCE_SAMPLES,
    flatten_method: str = DEFAULT_CUBIC_FLATTEN_METHOD,
    process_fn: Optional[CubicFillProcessFn] = None,
):
    num_segments = int(cubics_norm.shape[0])
    offs = _as_i32_contig(primitive_seg_offsets).view(-1)
    num_primitives = int(max(0, int(offs.numel()) - 1))
    if num_primitives <= 0:
        bg = background.to(device=cubics_norm.device, dtype=torch.float32).view(1, 1, 3)
        return bg.expand(int(img_h), int(img_w), 3).clone()

    if int(colors.shape[0]) != num_primitives:
        raise ValueError(
            f"colors first dim must match primitive count ({num_primitives}), got {int(colors.shape[0])}"
        )
    if int(opacities.view(-1).shape[0]) != num_primitives:
        raise ValueError(
            f"opacities size must match primitive count ({num_primitives}), got {int(opacities.view(-1).shape[0])}"
        )
    if int(depths.view(-1).shape[0]) != num_primitives:
        raise ValueError(
            f"depths size must match primitive count ({num_primitives}), got {int(depths.view(-1).shape[0])}"
        )
    if int(offs[0].item()) != 0 or int(offs[-1].item()) != num_segments:
        raise ValueError(
            "primitive_seg_offsets must start at 0 and end at num_segments "
            f"(got start={int(offs[0].item())}, end={int(offs[-1].item())}, num_segments={num_segments})"
        )
    if fill_rules is None:
        fill_rules_in = torch.ones(
            (num_primitives,), dtype=torch.int32, device=cubics_norm.device
        )
    else:
        fill_rules_in = fill_rules.to(
            device=cubics_norm.device, dtype=torch.int32
        ).view(-1)
        if int(fill_rules_in.numel()) == 1 and num_primitives > 1:
            fill_rules_in = fill_rules_in.expand(num_primitives)
        if int(fill_rules_in.numel()) != num_primitives:
            raise ValueError(
                f"fill_rules size must match primitive count ({num_primitives}), got {int(fill_rules_in.numel())}"
            )

    if aa_widths is None:
        aa_widths_in = torch.full(
            (num_primitives,),
            float(aa_width),
            device=cubics_norm.device,
            dtype=torch.float32,
        )
    else:
        aa_widths_in = aa_widths.to(
            device=cubics_norm.device, dtype=torch.float32
        ).view(-1)
        if int(aa_widths_in.numel()) == 1 and num_primitives > 1:
            aa_widths_in = aa_widths_in.expand(num_primitives)
        if int(aa_widths_in.numel()) != num_primitives:
            raise ValueError(
                f"aa_widths size must match primitive count ({num_primitives}), got {int(aa_widths_in.numel())}"
            )
    aa_widths_in = torch.clamp(_as_f32_contig(aa_widths_in), min=1e-4)
    flatten_method_norm = normalize_cubic_flatten_method(flatten_method)
    return _CubicFillSplatFunction.apply(
        cubics_norm,
        offs.contiguous(),
        colors.contiguous(),
        opacities.view(-1).contiguous(),
        depths.view(-1).contiguous(),
        fill_rules_in.contiguous(),
        int(img_h),
        int(img_w),
        int(block_x),
        int(block_y),
        background,
        int(project_threads),
        int(map_threads),
        float(aabb_pad),
        bool(aniso_intersects),
        aa_widths_in,
        int(distance_samples),
        flatten_method_norm,
        process_fn,
    )


__all__ = [
    "BLOCK_X",
    "BLOCK_Y",
    "DEFAULT_AA_WIDTH",
    "DEFAULT_AABB_PAD",
    "DEFAULT_DISTANCE_SAMPLES",
    "CUBIC_FLATTEN_METHOD_BERNSTEIN",
    "CUBIC_FLATTEN_METHOD_DE_CASTELJAU",
    "DEFAULT_CUBIC_FLATTEN_METHOD",
    "DEFAULT_THREADS",
    "normalize_cubic_flatten_method",
    "closed_cubic_fill_segments_forward_kernel",
    "closed_cubic_fill_segments_backward_kernel",
    "closed_cubic_fill_segments_tilelang",
    "map_cubic_to_intersects_kernel",
    "map_cubic_to_intersects_tilelang",
    "project_cubics_2d_forward_kernel",
    "project_cubics_2d_forward_tilelang",
    "rasterize_cubic_fill_forward_kernel",
    "rasterize_cubic_fill_forward_tilelang",
    "rasterize_cubic_fill_backward_kernel",
    "rasterize_cubic_fill_backward_tilelang",
    "render_cubic_fill_splat_tilelang",
]
