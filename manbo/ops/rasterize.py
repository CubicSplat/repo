import tilelang
import tilelang.language as T
import torch

BLOCK_X = 16
BLOCK_Y = 16
DEFAULT_THREADS = 256


@T.macro
def copy_vector3(src: T.Tensor, src_base: T.int32, dst: T.Tensor, dst_base: T.int32):
    """Copies a 3D vector from src at src_base to dst at dst_base."""
    dst[dst_base, 0] = src[src_base, 0]
    dst[dst_base, 1] = src[src_base, 1]
    dst[dst_base, 2] = src[src_base, 2]


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def rasterize_forward_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    dtype: str = "float32",
):
    """TileLang kernel for rasterize_forward (forward.cu)."""
    num_points = T.dynamic("num_points")
    num_intersects = T.dynamic("num_intersects")
    num_tiles = T.dynamic("num_tiles")
    img_height = T.dynamic("img_height")
    img_width = T.dynamic("img_width")
    block_size = block_x * block_y

    @T.prim_func
    def kernel(
        gaussian_ids_sorted: T.Tensor[[num_intersects], T.int32],
        tile_bins: T.Tensor[[num_tiles, 2], T.int32],
        xys: T.Tensor[[num_points, 2], dtype],
        conics: T.Tensor[[num_points, 3], dtype],
        colors: T.Tensor[[num_points, 3], dtype],
        opacities: T.Tensor[[num_points], dtype],
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
        one_f = T.Cast(dtype, 1.0)
        with T.Kernel(tile_bound_x, tile_bound_y, threads=(block_x, block_y)) as (
            bx,
            by,
        ):
            T.annotate_safe_value(
                {
                    tile_bins: T.int32(0),
                    gaussian_ids_sorted: T.int32(0),
                    xys: T.Cast(dtype, 0.0),
                    conics: T.Cast(dtype, 0.0),
                    colors: T.Cast(dtype, 0.0),
                    opacities: T.Cast(dtype, 0.0),
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
            xy_opacity_batch = T.alloc_shared((block_size, 3), dtype)
            conic_batch = T.alloc_shared((block_size, 3), dtype)
            color_batch = T.alloc_shared((block_size, 3), dtype)
            final_Ts_flat = T.reshape(final_Ts, [img_height * img_width])
            final_idx_flat = T.reshape(final_idx, [img_height * img_width])
            out_img_flat = T.reshape(out_img, [img_height * img_width, 3])

            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)
            i = by * block_y + ty
            j = bx * block_x + tx
            pix_id = T.min(i * img_width + j, img_height * img_width - 1)
            T.assume((pix_id >= 0) & (pix_id < img_height * img_width))
            px = T.Cast(dtype, j)
            py = T.Cast(dtype, i)

            inside_i32 = T.if_then_else(
                (i < img_height) & (j < img_width), T.int32(1), T.int32(0)
            )
            inside = inside_i32 != T.int32(0)

            done = T.alloc_var("int32")
            done = T.if_then_else(inside, T.int32(0), T.int32(1))

            T_local = T.alloc_var(dtype, init=one_f)
            cur_idx = T.alloc_var("int32", init=0)

            pix_out = T.alloc_local((3,), dtype)
            pix_out[0] = 0.0
            pix_out[1] = 0.0
            pix_out[2] = 0.0

            tr = ty * block_x + tx
            for b in T.serial(num_batches):
                done_count = T.call_extern("int32", "__syncthreads_count", done)
                if done_count >= block_size:
                    break

                batch_start = range_start + block_size * b
                idx = batch_start + tr
                if idx < range_end:
                    T.assume(idx < num_intersects)
                    g_id = gaussian_ids_sorted[idx]
                    T.assume((g_id >= 0) & (g_id < num_points))
                    xy_opacity_batch[tr, 0] = xys[g_id, 0]
                    xy_opacity_batch[tr, 1] = xys[g_id, 1]
                    xy_opacity_batch[tr, 2] = opacities[g_id]
                    conic_batch[tr, 0] = conics[g_id, 0]
                    conic_batch[tr, 1] = conics[g_id, 1]
                    conic_batch[tr, 2] = conics[g_id, 2]
                    color_batch[tr, 0] = colors[g_id, 0]
                    color_batch[tr, 1] = colors[g_id, 1]
                    color_batch[tr, 2] = colors[g_id, 2]

                T.sync_threads()

                batch_size = T.min(block_size, range_end - batch_start)
                for t in T.serial(batch_size):
                    if done == T.int32(0):
                        conic_x = conic_batch[t, 0]
                        conic_y = conic_batch[t, 1]
                        conic_z = conic_batch[t, 2]
                        xy_x = xy_opacity_batch[t, 0]
                        xy_y = xy_opacity_batch[t, 1]
                        opac = xy_opacity_batch[t, 2]

                        dx = xy_x - px
                        dy = xy_y - py
                        sigma = (
                            0.5 * (conic_x * dx * dx + conic_z * dy * dy)
                            + conic_y * dx * dy
                        )
                        if sigma < 0.0:
                            continue
                        alpha = T.min(alpha_cap, opac * T.exp(-sigma))
                        if alpha < alpha_min:
                            continue
                        next_T = T_local * (one_f - alpha)
                        if next_T <= trans_stop:
                            done = T.Cast(T.int32, 1)
                        vis = alpha * T_local
                        pix_out[0] += color_batch[t, 0] * vis
                        pix_out[1] += color_batch[t, 1] * vis
                        pix_out[2] += color_batch[t, 2] * vis
                        T_local = next_T
                        cur_idx = batch_start + t

            if inside:
                final_Ts_flat[pix_id] = T_local
                final_idx_flat[pix_id] = cur_idx
                out_img_flat[pix_id, 0] = pix_out[0] + T_local * background[0]
                out_img_flat[pix_id, 1] = pix_out[1] + T_local * background[1]
                out_img_flat[pix_id, 2] = pix_out[2] + T_local * background[2]

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def rasterize_backward_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    dtype: str = "float32",
):
    """TileLang kernel for rasterize_backward (backward.cu)."""
    num_points = T.dynamic("num_points")
    num_intersects = T.dynamic("num_intersects")
    num_tiles = T.dynamic("num_tiles")
    img_height = T.dynamic("img_height")
    img_width = T.dynamic("img_width")
    num_isect_contrib = T.dynamic("num_isect_contrib")
    block_size = block_x * block_y

    @T.prim_func
    def kernel(
        gaussian_ids_sorted: T.Tensor[[num_intersects], T.int32],
        intersect_meta_sorted: T.Tensor[[num_intersects], T.int32],
        tile_bins: T.Tensor[[num_tiles, 2], T.int32],
        xys: T.Tensor[[num_points, 2], dtype],
        conics: T.Tensor[[num_points, 3], dtype],
        rgbs: T.Tensor[[num_points, 3], dtype],
        opacities: T.Tensor[[num_points], dtype],
        background: T.Tensor[[3], dtype],
        final_Ts: T.Tensor[[img_height, img_width], dtype],
        final_idx: T.Tensor[[img_height, img_width], T.int32],
        v_output: T.Tensor[[img_height, img_width, 3], dtype],
        v_output_alpha: T.Tensor[[img_height, img_width], dtype],
        v_xy: T.Tensor[[num_points, 2], dtype],
        v_conic: T.Tensor[[num_points, 3], dtype],
        v_rgb: T.Tensor[[num_points, 3], dtype],
        v_opacity: T.Tensor[[num_points], dtype],
        collect_curve_contrib: T.int32,
        curve_contrib_rgb: T.Tensor[[num_points, 3], dtype],
        collect_curve_isect_contrib: T.int32,
        curve_isect_contrib: T.Tensor[[num_isect_contrib], dtype],
    ):
        # Keep num_points in closure for eager annotation resolution.
        num_points_ref = num_points
        _ = num_points_ref
        tile_bound_x = T.ceildiv(img_width, block_x)
        tile_bound_y = T.ceildiv(img_height, block_y)
        with T.Kernel(tile_bound_x, tile_bound_y, threads=(block_x, block_y)) as (
            bx,
            by,
        ):
            T.annotate_safe_value(
                {
                    tile_bins: T.int32(0),
                    gaussian_ids_sorted: T.int32(0),
                    intersect_meta_sorted: T.int32(0),
                    xys: T.Cast(dtype, 0.0),
                    conics: T.Cast(dtype, 0.0),
                    rgbs: T.Cast(dtype, 0.0),
                    opacities: T.Cast(dtype, 0.0),
                }
            )
            tile_id = by * tile_bound_x + bx
            T.assume(tile_id < num_tiles)
            tx = T.get_thread_binding(0)
            ty = T.get_thread_binding(1)

            i = by * block_y + ty
            j = bx * block_x + tx

            inside_i32 = T.alloc_var("int32")
            inside_i32 = T.if_then_else(
                (i < img_height) & (j < img_width), T.int32(1), T.int32(0)
            )
            pix_id = T.alloc_var("int32")
            pix_id = T.min(i * img_width + j, img_width * img_height - 1)
            T.assume((pix_id >= 0) & (pix_id < img_height * img_width))
            px = T.Cast(dtype, j)
            py = T.Cast(dtype, i)
            zero_f = T.Cast(dtype, 0.0)
            alpha_cap = T.Cast(dtype, 0.99)
            alpha_min = T.Cast(dtype, 1.0 / 255.0)

            range_start = tile_bins[tile_id, 0]
            range_end = tile_bins[tile_id, 1]
            T.assume(range_start >= 0)
            T.assume(range_end >= range_start)
            T.assume(range_end <= num_intersects)
            num_batches = T.ceildiv(range_end - range_start, block_size)

            id_batch = T.alloc_shared((block_size,), "int32")
            meta_batch = T.alloc_shared((block_size,), "int32")
            xy_opacity_batch = T.alloc_shared((block_size, 3), dtype)
            conic_batch = T.alloc_shared((block_size, 3), dtype)
            rgbs_batch = T.alloc_shared((block_size, 3), dtype)
            final_Ts_flat = T.reshape(final_Ts, [img_height * img_width])
            final_idx_flat = T.reshape(final_idx, [img_height * img_width])
            v_output_flat = T.reshape(v_output, [img_height * img_width, 3])
            v_output_alpha_flat = T.reshape(v_output_alpha, [img_height * img_width])

            T_local = T.alloc_var(dtype)
            T_final = T.alloc_var(dtype)
            bin_final = T.alloc_var("int32")
            v_out0 = T.alloc_var(dtype)
            v_out1 = T.alloc_var(dtype)
            v_out2 = T.alloc_var(dtype)
            v_out_alpha = T.alloc_var(dtype)
            T_local = zero_f
            bin_final = T.int32(0)
            v_out0 = zero_f
            v_out1 = zero_f
            v_out2 = zero_f
            v_out_alpha = zero_f
            T_local = final_Ts_flat[pix_id]
            bin_final = T.if_then_else(
                inside_i32 != T.int32(0), final_idx_flat[pix_id], T.int32(0)
            )
            v_out0 = v_output_flat[pix_id, 0]
            v_out1 = v_output_flat[pix_id, 1]
            v_out2 = v_output_flat[pix_id, 2]
            v_out_alpha = v_output_alpha_flat[pix_id]
            T_final = T_local

            buffer = T.alloc_local((3,), dtype)
            T.clear(buffer)
            bg_dot_vout = (
                background[0] * v_out0 + background[1] * v_out1 + background[2] * v_out2
            )

            tr = ty * block_x + tx
            use_bin_final = T.warp_reduce_max(bin_final)

            v_rgb_local = T.alloc_local((3,), dtype)
            contrib_rgb_local = T.alloc_local((3,), dtype)
            v_conic_local = T.alloc_local((3,), dtype)
            v_xy_local = T.alloc_local((2,), dtype)
            v_opacity_local = T.alloc_var(dtype)
            valid_i32 = T.alloc_var("int32")
            geom_valid_i32 = T.alloc_var("int32")
            base_valid_i32 = T.alloc_var("int32")
            geom_enabled_i32 = T.alloc_var("int32")
            conic_x = T.alloc_var(dtype)
            conic_y = T.alloc_var(dtype)
            conic_z = T.alloc_var(dtype)
            opac = T.alloc_var(dtype)
            dx = T.alloc_var(dtype)
            dy = T.alloc_var(dtype)
            vis = T.alloc_var(dtype)
            alpha = T.alloc_var(dtype)
            v_alpha = T.alloc_var(dtype)
            g_id_load = T.alloc_var("int32")
            base_valid_i32 = inside_i32

            for b in T.serial(num_batches):
                T.sync_threads()
                batch_end = range_end - 1 - block_size * b
                batch_size = T.min(block_size, batch_end + 1 - range_start)

                idx = batch_end - tr
                if idx >= range_start:
                    T.assume(idx < num_intersects)
                    g_id_load = gaussian_ids_sorted[idx]
                    T.assume((g_id_load >= 0) & (g_id_load < num_points))
                    id_batch[tr] = g_id_load
                    meta_batch[tr] = intersect_meta_sorted[idx]
                    xy_opacity_batch[tr, 0] = xys[g_id_load, 0]
                    xy_opacity_batch[tr, 1] = xys[g_id_load, 1]
                    xy_opacity_batch[tr, 2] = opacities[g_id_load]
                    conic_batch[tr, 0] = conics[g_id_load, 0]
                    conic_batch[tr, 1] = conics[g_id_load, 1]
                    conic_batch[tr, 2] = conics[g_id_load, 2]
                    rgbs_batch[tr, 0] = rgbs[g_id_load, 0]
                    rgbs_batch[tr, 1] = rgbs[g_id_load, 1]
                    rgbs_batch[tr, 2] = rgbs[g_id_load, 2]

                T.sync_threads()

                t_start = T.max(T.int32(0), batch_end - use_bin_final)
                for t in T.serial(t_start, batch_size):
                    valid_i32 = base_valid_i32
                    geom_valid_i32 = T.int32(0)
                    geom_enabled_i32 = T.if_then_else(
                        meta_batch[t] != T.int32(0), T.int32(1), T.int32(0)
                    )
                    if batch_end - t > bin_final:
                        valid_i32 = T.int32(0)

                    if valid_i32 != T.int32(0):
                        conic_x = conic_batch[t, 0]
                        conic_y = conic_batch[t, 1]
                        conic_z = conic_batch[t, 2]
                        dx = xy_opacity_batch[t, 0] - px
                        dy = xy_opacity_batch[t, 1] - py
                        opac = xy_opacity_batch[t, 2]

                        sigma = (
                            0.5 * (conic_x * dx * dx + conic_z * dy * dy)
                            + conic_y * dx * dy
                        )
                        if sigma < 0.0:
                            valid_i32 = T.int32(0)
                        else:
                            vis = T.exp(-sigma)
                            alpha = T.min(alpha_cap, opac * vis)
                            if alpha < alpha_min:
                                valid_i32 = T.int32(0)

                    warp_any = T.call_extern(
                        "int32",
                        "__any_sync",
                        T.Cast("uint32", T.uint32(0xFFFFFFFF)),
                        valid_i32,
                    )
                    if warp_any == T.int32(0):
                        continue

                    v_rgb_local[0] = 0.0
                    v_rgb_local[1] = 0.0
                    v_rgb_local[2] = 0.0
                    contrib_rgb_local[0] = 0.0
                    contrib_rgb_local[1] = 0.0
                    contrib_rgb_local[2] = 0.0
                    v_conic_local[0] = 0.0
                    v_conic_local[1] = 0.0
                    v_conic_local[2] = 0.0
                    v_xy_local[0] = 0.0
                    v_xy_local[1] = 0.0
                    v_opacity_local = 0.0

                    if valid_i32 != T.int32(0):
                        v_alpha = 0.0
                        ra = 1.0 / (1.0 - alpha)
                        T_local = T_local * ra
                        fac = alpha * T_local

                        v_rgb_local[0] = fac * v_out0
                        v_rgb_local[1] = fac * v_out1
                        v_rgb_local[2] = fac * v_out2

                        rgb0 = rgbs_batch[t, 0]
                        rgb1 = rgbs_batch[t, 1]
                        rgb2 = rgbs_batch[t, 2]
                        if (collect_curve_contrib != T.int32(0)) | (
                            collect_curve_isect_contrib != T.int32(0)
                        ):
                            contrib_rgb_local[0] = fac * rgb0
                            contrib_rgb_local[1] = fac * rgb1
                            contrib_rgb_local[2] = fac * rgb2

                        v_alpha = v_alpha + (rgb0 * T_local - buffer[0] * ra) * v_out0
                        v_alpha = v_alpha + (rgb1 * T_local - buffer[1] * ra) * v_out1
                        v_alpha = v_alpha + (rgb2 * T_local - buffer[2] * ra) * v_out2
                        v_alpha = v_alpha + T_final * ra * (v_out_alpha - bg_dot_vout)

                        buffer[0] += rgb0 * fac
                        buffer[1] += rgb1 * fac
                        buffer[2] += rgb2 * fac

                        v_opacity_local = vis * v_alpha
                        if geom_enabled_i32 != T.int32(0):
                            v_sigma = -opac * vis * v_alpha
                            v_conic_local[0] = 0.5 * v_sigma * dx * dx
                            v_conic_local[1] = 0.5 * v_sigma * dx * dy
                            v_conic_local[2] = 0.5 * v_sigma * dy * dy
                            v_xy_local[0] = v_sigma * (conic_x * dx + conic_y * dy)
                            v_xy_local[1] = v_sigma * (conic_y * dx + conic_z * dy)
                            geom_valid_i32 = T.int32(1)

                    warp_any_geom = T.call_extern(
                        "int32",
                        "__any_sync",
                        T.Cast("uint32", T.uint32(0xFFFFFFFF)),
                        geom_valid_i32,
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
                    v_opacity_local = T.warp_reduce_sum(v_opacity_local)
                    if warp_any_geom != T.int32(0):
                        v_conic_local[0] = T.warp_reduce_sum(v_conic_local[0])
                        v_conic_local[1] = T.warp_reduce_sum(v_conic_local[1])
                        v_conic_local[2] = T.warp_reduce_sum(v_conic_local[2])
                        v_xy_local[0] = T.warp_reduce_sum(v_xy_local[0])
                        v_xy_local[1] = T.warp_reduce_sum(v_xy_local[1])

                    if T.get_lane_idx() == 0:
                        g = id_batch[t]
                        T.assume((g >= 0) & (g < num_points))
                        T.atomic_add(v_rgb[g, 0], v_rgb_local[0])
                        T.atomic_add(v_rgb[g, 1], v_rgb_local[1])
                        T.atomic_add(v_rgb[g, 2], v_rgb_local[2])
                        T.atomic_add(v_opacity[g], v_opacity_local)
                        if collect_curve_contrib != T.int32(0):
                            T.atomic_add(curve_contrib_rgb[g, 0], contrib_rgb_local[0])
                            T.atomic_add(curve_contrib_rgb[g, 1], contrib_rgb_local[1])
                            T.atomic_add(curve_contrib_rgb[g, 2], contrib_rgb_local[2])
                        if collect_curve_isect_contrib != T.int32(0):
                            isect_idx = batch_end - t
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
                        if warp_any_geom != T.int32(0):
                            T.atomic_add(v_conic[g, 0], v_conic_local[0])
                            T.atomic_add(v_conic[g, 1], v_conic_local[1])
                            T.atomic_add(v_conic[g, 2], v_conic_local[2])
                            T.atomic_add(v_xy[g, 0], v_xy_local[0])
                            T.atomic_add(v_xy[g, 1], v_xy_local[1])

    return kernel


def rasterize_forward_tilelang(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    gaussian_ids_sorted: torch.Tensor,
    tile_bins: torch.Tensor,
    xys: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    background: torch.Tensor,
):
    block_x, block_y, _ = block
    img_w, img_h, _ = img_size

    out_img = torch.empty((img_h, img_w, 3), dtype=xys.dtype, device=xys.device)
    final_Ts = torch.empty((img_h, img_w), dtype=xys.dtype, device=xys.device)
    final_idx = torch.empty((img_h, img_w), dtype=torch.int32, device=xys.device)

    kernel = rasterize_forward_kernel(
        block_x=block_x,
        block_y=block_y,
        dtype="float32",
    )

    kernel(
        gaussian_ids_sorted.contiguous(),
        tile_bins.contiguous(),
        xys.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        background.contiguous(),
        out_img,
        final_Ts,
        final_idx,
    )

    return out_img, final_Ts, final_idx


def rasterize_backward_tilelang(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    gaussian_ids_sorted: torch.Tensor,
    tile_bins: torch.Tensor,
    xys: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    background: torch.Tensor,
    final_Ts: torch.Tensor,
    final_idx: torch.Tensor,
    v_output: torch.Tensor,
    v_output_alpha: torch.Tensor,
    *,
    intersect_meta_sorted: torch.Tensor | None = None,
    curve_contrib_rgb: torch.Tensor | None = None,
    curve_isect_contrib: torch.Tensor | None = None,
):
    block_x, block_y, _ = block
    img_w, img_h, _ = img_size

    if opacities.dim() == 1:
        opacities_flat = opacities
        opac_was_2d = False
    elif opacities.dim() == 2 and opacities.shape[1] == 1:
        opacities_flat = opacities.view(-1)
        opac_was_2d = True
    else:
        raise ValueError("opacities must be shape (N,) or (N,1)")

    v_xy = torch.zeros_like(xys)
    v_conic = torch.zeros_like(conics)
    v_rgb = torch.zeros_like(colors)
    v_opacity = torch.zeros_like(opacities_flat)
    if curve_contrib_rgb is None:
        curve_contrib_rgb_tensor = torch.empty_like(colors, dtype=torch.float32)
        collect_curve_contrib_i32 = 0
    else:
        if curve_contrib_rgb.shape != colors.shape:
            raise ValueError(
                "curve_contrib_rgb must match colors shape "
                f"{tuple(colors.shape)}, got {tuple(curve_contrib_rgb.shape)}"
            )
        curve_contrib_rgb_tensor = curve_contrib_rgb.to(
            device=colors.device,
            dtype=torch.float32,
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
        if int(curve_isect_contrib.numel()) != int(gaussian_ids_sorted.numel()):
            raise ValueError("curve_isect_contrib size must match gaussian_ids_sorted")
        curve_isect_contrib_tensor = curve_isect_contrib.to(
            device=colors.device,
            dtype=torch.float32,
        ).contiguous()
        collect_curve_isect_contrib_i32 = 1
    if intersect_meta_sorted is None:
        intersect_meta_sorted_tensor = torch.ones_like(
            gaussian_ids_sorted,
            dtype=torch.int32,
            device=gaussian_ids_sorted.device,
        )
    else:
        if intersect_meta_sorted.ndim != 1:
            raise ValueError("intersect_meta_sorted must be a 1D tensor")
        if intersect_meta_sorted.numel() != gaussian_ids_sorted.numel():
            raise ValueError(
                "intersect_meta_sorted size must match gaussian_ids_sorted"
            )
        intersect_meta_sorted_tensor = intersect_meta_sorted.to(
            device=gaussian_ids_sorted.device,
            dtype=torch.int32,
        ).contiguous()

    kernel = rasterize_backward_kernel(
        block_x=block_x,
        block_y=block_y,
        dtype="float32",
    )

    kernel(
        gaussian_ids_sorted.contiguous(),
        intersect_meta_sorted_tensor,
        tile_bins.contiguous(),
        xys.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities_flat.contiguous(),
        background.contiguous(),
        final_Ts.contiguous(),
        final_idx.contiguous(),
        v_output.contiguous(),
        v_output_alpha.contiguous(),
        v_xy,
        v_conic,
        v_rgb,
        v_opacity,
        int(collect_curve_contrib_i32),
        curve_contrib_rgb_tensor,
        int(collect_curve_isect_contrib_i32),
        curve_isect_contrib_tensor,
    )

    if opac_was_2d:
        v_opacity = v_opacity.view(-1, 1)
    return v_xy, v_conic, v_rgb, v_opacity
