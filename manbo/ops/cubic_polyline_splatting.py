from __future__ import annotations

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from .cubic_splatting import (
    BLOCK_X,
    BLOCK_Y,
    DEFAULT_AA_WIDTH,
    DEFAULT_AABB_PAD,
    DEFAULT_CUBIC_FLATTEN_METHOD,
    DEFAULT_DISTANCE_SAMPLES,
    DEFAULT_THREADS,
    CubicProcessFn,
    CubicProcessPayload,
    _accumulate_edge_grads_to_cubics,
    _as_f32_contig,
    _as_i32_contig,
    _build_fill_isect_edge_ranges,
    _flatten_cubics_to_edges,
    _primitive_tile_ranges_from_segment_tile_ranges,
    _run_cubic_process_fn,
    map_cubic_to_intersects_tilelang,
    normalize_cubic_flatten_method,
    project_cubics_2d_forward_tilelang,
)
from .tile_bins import get_tile_bin_edges_tilelang


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def rasterize_cubic_polyline_forward_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    dtype: str = "float32",
):
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
        stroke_widths: T.Tensor[[num_primitives], dtype],
        inv_sigma2s: T.Tensor[[num_primitives], dtype],
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
                    stroke_widths: T.Cast(dtype, 0.0),
                    inv_sigma2s: T.Cast(dtype, 1.0),
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
            stroke_batch = T.alloc_shared((block_size,), dtype)
            inv_sigma2_batch = T.alloc_shared((block_size,), dtype)
            edge_start_batch = T.alloc_shared((block_size,), "int32")
            edge_end_batch = T.alloc_shared((block_size,), "int32")
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
                    stroke_batch[tr] = stroke_widths[pid]
                    inv_sigma2_batch[tr] = inv_sigma2s[pid]

                T.sync_threads()

                batch_size = T.min(block_size, range_end - batch_start)
                for t in T.serial(batch_size):
                    edge_start = edge_start_batch[t]
                    edge_end = edge_end_batch[t]
                    n_edge = T.max(T.int32(0), edge_end - edge_start)

                    min_d2 = T.alloc_var(dtype)
                    min_d2 = T.Cast(dtype, 1e30)

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

                        T.sync_threads()

                    if done == T.int32(0):
                        dist = T.sqrt(T.max(min_d2, zero_f))
                        half_sw = T.max(zero_f, stroke_batch[t]) * half_f
                        inv_sigma2 = T.max(inv_sigma2_batch[t], eps_f)
                        edge_d = T.max(zero_f, dist - half_sw)
                        sigma = half_f * edge_d * edge_d * inv_sigma2
                        cov = T.exp(-sigma)
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
def rasterize_cubic_polyline_backward_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    collect_curve_contrib: bool = False,
    collect_curve_isect_contrib: bool = False,
    dtype: str = "float32",
):
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
        stroke_widths: T.Tensor[[num_primitives], dtype],
        inv_sigma2s: T.Tensor[[num_primitives], dtype],
        background: T.Tensor[[3], dtype],
        final_Ts: T.Tensor[[img_height, img_width], dtype],
        final_idx: T.Tensor[[img_height, img_width], T.int32],
        v_output: T.Tensor[[img_height, img_width, 3], dtype],
        v_edges: T.Tensor[[num_edges, 2, 2], dtype],
        v_colors: T.Tensor[[num_primitives, 3], dtype],
        v_opacities: T.Tensor[[num_primitives], dtype],
        v_stroke_widths: T.Tensor[[num_primitives], dtype],
        v_inv_sigma2s: T.Tensor[[num_primitives], dtype],
        collect_curve_contrib_i32: T.int32,
        curve_contrib_rgb: T.Tensor[[num_primitives, 3], dtype],
        collect_curve_isect_contrib_i32: T.int32,
        curve_isect_contrib: T.Tensor[[num_isect_contrib], dtype],
    ):
        tile_bound_x = T.ceildiv(img_width, block_x)
        tile_bound_y = T.ceildiv(img_height, block_y)
        alpha_cap = T.Cast(dtype, 0.999)
        alpha_min = T.Cast(dtype, 1.0 / 255.0)
        zero_f = T.Cast(dtype, 0.0)
        one_f = T.Cast(dtype, 1.0)
        half_f = T.Cast(dtype, 0.5)
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
                    stroke_widths: T.Cast(dtype, 0.0),
                    inv_sigma2s: T.Cast(dtype, 1.0),
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
            stroke_batch = T.alloc_shared((block_size,), dtype)
            inv_sigma2_batch = T.alloc_shared((block_size,), dtype)
            edge_start_batch = T.alloc_shared((block_size,), "int32")
            edge_end_batch = T.alloc_shared((block_size,), "int32")
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
            v_stroke_local = T.alloc_var(dtype)
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
                    stroke_batch[tr] = stroke_widths[pid]
                    inv_sigma2_batch[tr] = inv_sigma2s[pid]

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
                    v_stroke_local = zero_f
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
                    stroke_w = T.alloc_var(dtype)
                    inv_sigma2 = T.alloc_var(dtype)
                    rgb0 = zero_f
                    rgb1 = zero_f
                    rgb2 = zero_f
                    opac = zero_f
                    stroke_w = zero_f
                    inv_sigma2 = one_f
                    if valid_i32 != T.int32(0):
                        rgb0 = color0_batch[t]
                        rgb1 = color1_batch[t]
                        rgb2 = color2_batch[t]
                        opac = opacity_batch[t]
                        stroke_w = stroke_batch[t]
                        inv_sigma2 = T.max(inv_sigma2_batch[t], eps_f)

                    edge_start = edge_start_batch[t]
                    edge_end = edge_end_batch[t]
                    n_edge = T.max(T.int32(0), edge_end - edge_start)

                    min_d2 = T.alloc_var(dtype)
                    best_eid = T.alloc_var("int32")
                    best_proj = T.alloc_var(dtype)
                    best_cx = T.alloc_var(dtype)
                    best_cy = T.alloc_var(dtype)
                    min_d2 = T.Cast(dtype, 1e30)
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

                    if valid_i32 != T.int32(0):
                        dist = T.sqrt(T.max(min_d2, zero_f))
                        half_sw = T.max(zero_f, stroke_w) * half_f
                        edge_d = T.max(zero_f, dist - half_sw)
                        sigma = half_f * edge_d * edge_d * inv_sigma2
                        cov = T.exp(-sigma)
                        alpha = T.min(alpha_cap, opac * cov)

                        if alpha >= alpha_min:
                            contrib_i32 = T.int32(1)
                            ra = one_f / (one_f - alpha)
                            T_local = T_local * ra
                            fac = alpha * T_local

                            v_rgb_local[0] = fac * v_out0
                            v_rgb_local[1] = fac * v_out1
                            v_rgb_local[2] = fac * v_out2
                            if (collect_curve_contrib_i32 != T.int32(0)) | (
                                collect_curve_isect_contrib_i32 != T.int32(0)
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

                                v_sigma = -opac * cov * v_alpha
                                v_inv_sigma2_local = half_f * edge_d * edge_d * v_sigma
                                if edge_d > zero_f:
                                    v_edge = v_sigma * edge_d * inv_sigma2
                                    v_stroke_local = -half_f * v_edge

                                    if (dist > eps_f) & (best_eid >= T.int32(0)):
                                        geom_valid_i32 = T.int32(1)
                                        v_dist = v_edge
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
                    if (collect_curve_contrib_i32 != T.int32(0)) | (
                        collect_curve_isect_contrib_i32 != T.int32(0)
                    ):
                        contrib_rgb_local[0] = T.warp_reduce_sum(contrib_rgb_local[0])
                        contrib_rgb_local[1] = T.warp_reduce_sum(contrib_rgb_local[1])
                        contrib_rgb_local[2] = T.warp_reduce_sum(contrib_rgb_local[2])
                    if warp_any_opacity != T.int32(0):
                        v_opacity_local = T.warp_reduce_sum(v_opacity_local)
                        v_stroke_local = T.warp_reduce_sum(v_stroke_local)
                        v_inv_sigma2_local = T.warp_reduce_sum(v_inv_sigma2_local)

                    if lane == 0:
                        pid = id_batch[t]
                        T.assume((pid >= 0) & (pid < num_primitives))
                        T.atomic_add(v_colors[pid, 0], v_rgb_local[0])
                        T.atomic_add(v_colors[pid, 1], v_rgb_local[1])
                        T.atomic_add(v_colors[pid, 2], v_rgb_local[2])
                        if collect_curve_contrib_i32 != T.int32(0):
                            T.atomic_add(
                                curve_contrib_rgb[pid, 0], contrib_rgb_local[0]
                            )
                            T.atomic_add(
                                curve_contrib_rgb[pid, 1], contrib_rgb_local[1]
                            )
                            T.atomic_add(
                                curve_contrib_rgb[pid, 2], contrib_rgb_local[2]
                            )
                        if collect_curve_isect_contrib_i32 != T.int32(0):
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
                            T.atomic_add(v_stroke_widths[pid], v_stroke_local)
                            T.atomic_add(v_inv_sigma2s[pid], v_inv_sigma2_local)

                    if (valid_i32 != T.int32(0)) & (geom_valid_i32 != T.int32(0)):
                        eid = best_eid_local
                        T.assume((eid >= T.int32(0)) & (eid < num_edges))
                        T.atomic_add(v_edges[eid, 0, 0], v_edge_local[0])
                        T.atomic_add(v_edges[eid, 0, 1], v_edge_local[1])
                        T.atomic_add(v_edges[eid, 1, 0], v_edge_local[2])
                        T.atomic_add(v_edges[eid, 1, 1], v_edge_local[3])

    return kernel


def rasterize_cubic_polyline_forward_tilelang(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    primitive_ids_sorted: torch.Tensor,
    tile_bins: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    stroke_widths: torch.Tensor,
    background: torch.Tensor,
    *,
    aa_widths: torch.Tensor | None = None,
    inv_sigma2s: torch.Tensor | None = None,
    edges_px: torch.Tensor,
    isect_edge_ranges: torch.Tensor,
    aa_width: float = DEFAULT_AA_WIDTH,
):
    img_width, img_height, _ = img_size
    num_primitives = int(colors.shape[0])
    num_intersects = int(primitive_ids_sorted.shape[0])
    out_img = torch.empty(
        (img_height, img_width, 3), dtype=torch.float32, device=colors.device
    )
    final_Ts = torch.empty(
        (img_height, img_width), dtype=torch.float32, device=colors.device
    )
    final_idx = torch.empty(
        (img_height, img_width), dtype=torch.int32, device=colors.device
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
                device=colors.device,
                dtype=torch.float32,
            )
        else:
            aa_widths_in = aa_widths.to(device=colors.device, dtype=torch.float32).view(
                -1
            )
            if int(aa_widths_in.numel()) == 1 and num_primitives > 1:
                aa_widths_in = aa_widths_in.expand(num_primitives)
            if int(aa_widths_in.numel()) != num_primitives:
                raise ValueError(
                    f"aa_widths size must match num_primitives ({num_primitives}), got {int(aa_widths_in.numel())}"
                )
        aa_widths_in = torch.clamp(_as_f32_contig(aa_widths_in), min=1e-4)
        inv_sigma2s_in = 1.0 / torch.clamp(aa_widths_in * aa_widths_in, min=1e-6)
    else:
        inv_sigma2s_in = inv_sigma2s.to(device=colors.device, dtype=torch.float32).view(
            -1
        )
        if int(inv_sigma2s_in.numel()) == 1 and num_primitives > 1:
            inv_sigma2s_in = inv_sigma2s_in.expand(num_primitives)
        if int(inv_sigma2s_in.numel()) != num_primitives:
            raise ValueError(
                f"inv_sigma2s size must match num_primitives ({num_primitives}), got {int(inv_sigma2s_in.numel())}"
            )
        inv_sigma2s_in = _as_f32_contig(inv_sigma2s_in)

    edges_px_f32 = _as_f32_contig(edges_px)
    if int(edges_px_f32.shape[0]) <= 0:
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

    kernel = rasterize_cubic_polyline_forward_kernel(
        block_x=int(block[0]),
        block_y=int(block[1]),
        dtype="float32",
    )

    primitive_ids_i32 = _as_i32_contig(primitive_ids_sorted).view(-1)
    tile_bins_i32 = _as_i32_contig(tile_bins)
    colors_f32 = _as_f32_contig(colors)
    opacities_f32 = _as_f32_contig(opacities).view(-1)
    stroke_widths_f32 = _as_f32_contig(stroke_widths).view(-1)
    background_f32 = _as_f32_contig(
        background.to(device=colors.device, dtype=torch.float32).view(3)
    )

    kernel(
        primitive_ids_i32,
        tile_bins_i32,
        edges_px_f32,
        isect_edge_ranges_i32,
        colors_f32,
        opacities_f32,
        stroke_widths_f32,
        inv_sigma2s_in.view(-1),
        background_f32,
        out_img,
        final_Ts,
        final_idx,
    )
    return out_img, final_Ts, final_idx


def rasterize_cubic_polyline_backward_tilelang(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    primitive_ids_sorted: torch.Tensor,
    tile_bins: torch.Tensor,
    cubics_px: torch.Tensor,
    primitive_seg_offsets: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    stroke_widths: torch.Tensor,
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
    curve_contrib_rgb: torch.Tensor | None = None,
    curve_isect_contrib: torch.Tensor | None = None,
):
    _ = tile_bounds
    block_x, block_y, _ = block
    _img_w, _img_h, _ = img_size
    num_primitives = int(colors.shape[0])
    num_segments = int(cubics_px.shape[0])

    v_colors = torch.zeros_like(colors, dtype=torch.float32)
    v_opacities = torch.zeros_like(opacities.view(-1), dtype=torch.float32)
    v_stroke_widths = torch.zeros_like(stroke_widths.view(-1), dtype=torch.float32)
    v_inv_sigma2s = torch.zeros(
        (num_primitives,), dtype=torch.float32, device=colors.device
    )
    v_cubics_px = torch.zeros_like(cubics_px, dtype=torch.float32)
    if (
        int(primitive_ids_sorted.numel()) <= 0
        or num_primitives <= 0
        or num_segments <= 0
    ):
        return v_cubics_px, v_colors, v_opacities, v_stroke_widths, v_inv_sigma2s

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

    edges_px_f32 = _as_f32_contig(edges_px)
    if int(edges_px_f32.shape[0]) <= 0:
        return v_cubics_px, v_colors, v_opacities, v_stroke_widths, v_inv_sigma2s

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

    v_edges = torch.zeros_like(edges_px_f32, dtype=torch.float32)
    kernel = rasterize_cubic_polyline_backward_kernel(
        block_x=int(block_x),
        block_y=int(block_y),
        collect_curve_contrib=bool(curve_contrib_rgb is not None),
        collect_curve_isect_contrib=bool(curve_isect_contrib is not None),
        dtype="float32",
    )
    primitive_ids_i32 = _as_i32_contig(primitive_ids_sorted).view(-1)
    tile_bins_i32 = _as_i32_contig(tile_bins)
    colors_f32 = _as_f32_contig(colors)
    opacities_f32 = _as_f32_contig(opacities).view(-1)
    stroke_widths_f32 = _as_f32_contig(stroke_widths).view(-1)
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
        stroke_widths_f32,
        inv_sigma2s_in.view(-1),
        background_f32,
        final_Ts_f32,
        final_idx_i32,
        v_output_f32,
        v_edges,
        v_colors,
        v_opacities,
        v_stroke_widths,
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
    return v_cubics_px, v_colors, v_opacities, v_stroke_widths, v_inv_sigma2s


class _CubicPolylineSplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cubics_norm: torch.Tensor,
        primitive_seg_offsets: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        stroke_widths: torch.Tensor,
        depths: torch.Tensor,
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
        process_fn: Optional[CubicProcessFn] = None,
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
        if int(stroke_widths.view(-1).shape[0]) != num_primitives:
            raise ValueError(
                f"stroke_widths size must match primitive count ({num_primitives}), got {int(stroke_widths.view(-1).shape[0])}"
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

        stroke_seg = torch.repeat_interleave(
            _as_f32_contig(stroke_widths).view(-1), seg_counts
        ).contiguous()
        if int(stroke_seg.numel()) != num_segments:
            raise ValueError(
                "primitive_seg_offsets and stroke_widths mismatch when expanding stroke widths: "
                f"{int(stroke_seg.numel())} vs {num_segments}"
            )

        cubics_px, seg_tile_ranges, _ = project_cubics_2d_forward_tilelang(
            cubics_norm,
            stroke_seg,
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

        out_img, final_Ts, final_idx = rasterize_cubic_polyline_forward_tilelang(
            tile_bounds,
            block,
            img_size,
            primitive_ids_sorted,
            tile_bins,
            colors,
            opacities.view(-1),
            stroke_widths.view(-1),
            bg,
            inv_sigma2s=inv_sigma2s,
            edges_px=edges_px,
            isect_edge_ranges=isect_edge_ranges,
        )

        ctx.empty = False
        ctx.tile_bounds = tile_bounds
        ctx.block = block
        ctx.img_size = img_size
        ctx.img_h = int(img_h)
        ctx.img_w = int(img_w)
        ctx.distance_samples = int(distance_samples)
        ctx.flatten_method = flatten_method_norm
        ctx.process_fn = process_fn
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
            stroke_widths.view(-1),
            depths.view(-1),
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
            stroke_widths,
            depths,
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
        pre_bwd_payload: CubicProcessPayload = {
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

        v_cubics_px, v_colors, v_opacity, v_stroke, v_inv_sigma2 = (
            rasterize_cubic_polyline_backward_tilelang(
                ctx.tile_bounds,
                ctx.block,
                ctx.img_size,
                primitive_ids_sorted,
                tile_bins,
                cubics_px,
                primitive_seg_offsets,
                colors,
                opacities,
                stroke_widths,
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
                curve_contrib_rgb=curve_contrib_rgb,
                curve_isect_contrib=curve_isect_contrib,
            )
        )
        post_bwd_payload: CubicProcessPayload = {
            "v_cubics_px": v_cubics_px,
            "v_colors": v_colors,
            "v_opacity": v_opacity,
            "v_stroke": v_stroke,
            "curve_contrib_rgb": curve_contrib_rgb,
            "curve_isect_contrib": curve_isect_contrib,
            "primitive_ids_sorted": primitive_ids_sorted,
            "tile_bins": tile_bins,
            "colors": colors,
            "opacities": opacities,
            "stroke_widths": stroke_widths,
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
        v_stroke = post_bwd_payload["v_stroke"]

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
            v_stroke.view_as(stroke_widths),
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
            v_aa_widths,
            None,
            None,
            None,
        )


def render_cubic_polyline_splat_tilelang(
    cubics_norm: torch.Tensor,
    primitive_seg_offsets: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    stroke_widths: torch.Tensor,
    depths: torch.Tensor,
    img_h: int,
    img_w: int,
    block_x: int,
    block_y: int,
    background: torch.Tensor,
    *,
    project_threads: int = DEFAULT_THREADS,
    map_threads: int = DEFAULT_THREADS,
    aabb_pad: float = DEFAULT_AABB_PAD,
    aniso_intersects: bool = False,
    aa_widths: torch.Tensor | None = None,
    aa_width: float = DEFAULT_AA_WIDTH,
    distance_samples: int = DEFAULT_DISTANCE_SAMPLES,
    flatten_method: str = DEFAULT_CUBIC_FLATTEN_METHOD,
    process_fn: Optional[CubicProcessFn] = None,
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
    if int(stroke_widths.view(-1).shape[0]) != num_primitives:
        raise ValueError(
            f"stroke_widths size must match primitive count ({num_primitives}), got {int(stroke_widths.view(-1).shape[0])}"
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
    return _CubicPolylineSplatFunction.apply(
        cubics_norm,
        offs.contiguous(),
        colors.contiguous(),
        opacities.view(-1).contiguous(),
        stroke_widths.view(-1).contiguous(),
        depths.view(-1).contiguous(),
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
    "rasterize_cubic_polyline_forward_kernel",
    "rasterize_cubic_polyline_backward_kernel",
    "rasterize_cubic_polyline_forward_tilelang",
    "rasterize_cubic_polyline_backward_tilelang",
    "render_cubic_polyline_splat_tilelang",
]
