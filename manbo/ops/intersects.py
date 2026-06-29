import tilelang
import tilelang.language as T
import torch

BLOCK_X = 16
BLOCK_Y = 16
DEFAULT_THREADS = 256
ANISO_SIGMA_MULT = 3.0
CONIC_EPS = 1e-12


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def map_gaussian_to_intersects_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
    num_points=T.dynamic("num_points"),
    num_intersects=T.dynamic("num_intersects"),
):
    @T.prim_func
    def kernel(
        xys: T.Tensor[[num_points, 2], dtype],
        conics: T.Tensor[[num_points, 3], dtype],
        depths_i32: T.Tensor[[num_points], T.int32],
        radii: T.Tensor[[num_points], T.int32],
        cum_tiles_hit: T.Tensor[[num_points], T.int32],
        isect_ids: T.Tensor[[num_intersects], T.int64],
        gaussian_ids: T.Tensor[[num_intersects], T.int32],
        tile_bound_x: T.int32,
        tile_bound_y: T.int32,
    ):
        block_x_f = T.Cast(dtype, block_x)
        block_y_f = T.Cast(dtype, block_y)
        one_f = T.Cast(dtype, 1.0)

        grid = T.ceildiv(num_points, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                gid = bx * threads + tx
                if gid < num_points:
                    T.assume(gid >= 0)
                    r = radii[gid]
                    if r > T.int32(0):
                        center_x = xys[gid, 0]
                        center_y = xys[gid, 1]
                        qxx = conics[gid, 0]
                        qxy = conics[gid, 1]
                        qyy = conics[gid, 2]
                        q_det = qxx * qyy - qxy * qxy
                        radius_f = T.Cast(dtype, r)

                        radius_x = T.alloc_var(dtype)
                        radius_y = T.alloc_var(dtype)
                        radius_x = radius_f
                        radius_y = radius_f
                        if (
                            (qxx > T.Cast(dtype, 0.0))
                            & (qyy > T.Cast(dtype, 0.0))
                            & (q_det > T.Cast(dtype, CONIC_EPS))
                        ):
                            radius_x = T.Cast(dtype, ANISO_SIGMA_MULT) * T.sqrt(
                                qyy / q_det
                            )
                            radius_y = T.Cast(dtype, ANISO_SIGMA_MULT) * T.sqrt(
                                qxx / q_det
                            )

                        tile_center_x = center_x / block_x_f
                        tile_center_y = center_y / block_y_f
                        tile_radius_x = radius_x / block_x_f
                        tile_radius_y = radius_y / block_y_f

                        tile_min_x = T.alloc_var("int32")
                        tile_max_x = T.alloc_var("int32")
                        tile_min_y = T.alloc_var("int32")
                        tile_max_y = T.alloc_var("int32")
                        tile_w = T.alloc_var("int32")
                        tile_h = T.alloc_var("int32")
                        tile_area = T.alloc_var("int32")
                        base_out = T.alloc_var("int32")
                        depth_u64 = T.alloc_var("int64")
                        row_base_out = T.alloc_var("int32")
                        row_base_tile = T.alloc_var("int32")
                        out_idx = T.alloc_var("int32")
                        tile_id64 = T.alloc_var("int64")

                        tile_min_x = T.clamp(
                            T.Cast(T.int32, tile_center_x - tile_radius_x),
                            T.int32(0),
                            tile_bound_x,
                        )
                        tile_max_x = T.clamp(
                            T.Cast(T.int32, tile_center_x + tile_radius_x + one_f),
                            T.int32(0),
                            tile_bound_x,
                        )
                        tile_min_y = T.clamp(
                            T.Cast(T.int32, tile_center_y - tile_radius_y),
                            T.int32(0),
                            tile_bound_y,
                        )
                        tile_max_y = T.clamp(
                            T.Cast(T.int32, tile_center_y + tile_radius_y + one_f),
                            T.int32(0),
                            tile_bound_y,
                        )

                        tile_w = tile_max_x - tile_min_x
                        tile_h = tile_max_y - tile_min_y
                        tile_area = tile_w * tile_h
                        if tile_area > T.int32(0):
                            base_out = (
                                T.int32(0) if gid == 0 else cum_tiles_hit[gid - 1]
                            )
                            T.assume(base_out >= 0)
                            T.assume(base_out + tile_area <= num_intersects)
                            depth_u64 = T.Cast(T.int64, depths_i32[gid]) & T.Cast(
                                T.int64, 0xFFFFFFFF
                            )
                            for row in T.serial(tile_h):
                                row_base_out = base_out + row * tile_w
                                row_base_tile = (
                                    tile_min_y + row
                                ) * tile_bound_x + tile_min_x
                                for col in T.serial(tile_w):
                                    out_idx = row_base_out + col
                                    T.assume(
                                        (out_idx >= 0) & (out_idx < num_intersects)
                                    )
                                    tile_id64 = T.Cast(T.int64, row_base_tile + col)
                                    isect_ids[out_idx] = (tile_id64 << 32) | depth_u64
                                    gaussian_ids[out_idx] = gid

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def map_gaussian_to_intersects_isotropic_kernel(
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
    num_points=T.dynamic("num_points"),
    num_intersects=T.dynamic("num_intersects"),
):
    @T.prim_func
    def kernel(
        xys: T.Tensor[[num_points, 2], dtype],
        depths_i32: T.Tensor[[num_points], T.int32],
        radii: T.Tensor[[num_points], T.int32],
        cum_tiles_hit: T.Tensor[[num_points], T.int32],
        isect_ids: T.Tensor[[num_intersects], T.int64],
        gaussian_ids: T.Tensor[[num_intersects], T.int32],
        tile_bound_x: T.int32,
        tile_bound_y: T.int32,
    ):
        block_x_f = T.Cast(dtype, block_x)
        block_y_f = T.Cast(dtype, block_y)
        one_f = T.Cast(dtype, 1.0)

        grid = T.ceildiv(num_points, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                gid = bx * threads + tx
                if gid < num_points:
                    T.assume(gid >= 0)
                    r = radii[gid]
                    if r > T.int32(0):
                        center_x = xys[gid, 0]
                        center_y = xys[gid, 1]
                        radius_f = T.Cast(dtype, r)

                        tile_center_x = center_x / block_x_f
                        tile_center_y = center_y / block_y_f
                        tile_radius_x = radius_f / block_x_f
                        tile_radius_y = radius_f / block_y_f

                        tile_min_x = T.alloc_var("int32")
                        tile_max_x = T.alloc_var("int32")
                        tile_min_y = T.alloc_var("int32")
                        tile_max_y = T.alloc_var("int32")
                        tile_w = T.alloc_var("int32")
                        tile_h = T.alloc_var("int32")
                        tile_area = T.alloc_var("int32")
                        base_out = T.alloc_var("int32")
                        depth_u64 = T.alloc_var("int64")
                        row_base_out = T.alloc_var("int32")
                        row_base_tile = T.alloc_var("int32")
                        out_idx = T.alloc_var("int32")
                        tile_id64 = T.alloc_var("int64")

                        tile_min_x = T.clamp(
                            T.Cast(T.int32, tile_center_x - tile_radius_x),
                            T.int32(0),
                            tile_bound_x,
                        )
                        tile_max_x = T.clamp(
                            T.Cast(T.int32, tile_center_x + tile_radius_x + one_f),
                            T.int32(0),
                            tile_bound_x,
                        )
                        tile_min_y = T.clamp(
                            T.Cast(T.int32, tile_center_y - tile_radius_y),
                            T.int32(0),
                            tile_bound_y,
                        )
                        tile_max_y = T.clamp(
                            T.Cast(T.int32, tile_center_y + tile_radius_y + one_f),
                            T.int32(0),
                            tile_bound_y,
                        )

                        tile_w = tile_max_x - tile_min_x
                        tile_h = tile_max_y - tile_min_y
                        tile_area = tile_w * tile_h
                        if tile_area > T.int32(0):
                            base_out = (
                                T.int32(0) if gid == 0 else cum_tiles_hit[gid - 1]
                            )
                            T.assume(base_out >= 0)
                            T.assume(base_out + tile_area <= num_intersects)
                            depth_u64 = T.Cast(T.int64, depths_i32[gid]) & T.Cast(
                                T.int64, 0xFFFFFFFF
                            )
                            for row in T.serial(tile_h):
                                row_base_out = base_out + row * tile_w
                                row_base_tile = (
                                    tile_min_y + row
                                ) * tile_bound_x + tile_min_x
                                for col in T.serial(tile_w):
                                    out_idx = row_base_out + col
                                    T.assume(
                                        (out_idx >= 0) & (out_idx < num_intersects)
                                    )
                                    tile_id64 = T.Cast(T.int64, row_base_tile + col)
                                    isect_ids[out_idx] = (tile_id64 << 32) | depth_u64
                                    gaussian_ids[out_idx] = gid

    return kernel


def map_gaussian_to_intersects_tilelang(
    num_points: int,
    num_intersects: int,
    xys: torch.Tensor,
    depths: torch.Tensor,
    radii: torch.Tensor,
    cum_tiles_hit: torch.Tensor,
    tile_bounds: tuple[int, int, int],
    *,
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    threads: int = DEFAULT_THREADS,
    conics: torch.Tensor | None = None,
):
    tile_bound_x, tile_bound_y, _ = tile_bounds

    isect_ids = torch.empty((num_intersects,), dtype=torch.int64, device=xys.device)
    gaussian_ids = torch.empty((num_intersects,), dtype=torch.int32, device=xys.device)

    depths_i32 = depths.view(torch.int32)
    if conics is None:
        kernel = map_gaussian_to_intersects_isotropic_kernel(
            block_x=block_x,
            block_y=block_y,
            threads=threads,
            dtype="float32",
        )
        kernel(
            xys.contiguous(),
            depths_i32.contiguous(),
            radii.contiguous(),
            cum_tiles_hit.contiguous(),
            isect_ids,
            gaussian_ids,
            tile_bound_x,
            tile_bound_y,
        )
    else:
        if conics.ndim != 2 or conics.shape[1] != 3:
            raise ValueError("conics must be shaped (num_points, 3)")
        if conics.shape[0] != num_points:
            raise ValueError(
                f"conics num_points mismatch: expected {num_points}, got {conics.shape[0]}"
            )
        if conics.device != xys.device or conics.dtype != xys.dtype:
            conics_in = conics.to(device=xys.device, dtype=xys.dtype)
        else:
            conics_in = conics

        kernel = map_gaussian_to_intersects_kernel(
            block_x=block_x,
            block_y=block_y,
            threads=threads,
            dtype="float32",
        )
        kernel(
            xys.contiguous(),
            conics_in.contiguous(),
            depths_i32.contiguous(),
            radii.contiguous(),
            cum_tiles_hit.contiguous(),
            isect_ids,
            gaussian_ids,
            tile_bound_x,
            tile_bound_y,
        )
    return isect_ids, gaussian_ids


def map_gaussian_meta_to_intersects(
    gaussian_meta: torch.Tensor,
    gaussian_ids: torch.Tensor,
) -> torch.Tensor:
    """Maps per-gaussian int meta flags to per-intersect order via gaussian ids."""
    if gaussian_meta.ndim != 1:
        raise ValueError("gaussian_meta must be a 1D tensor")
    if gaussian_ids.ndim != 1:
        raise ValueError("gaussian_ids must be a 1D tensor")
    if gaussian_meta.device != gaussian_ids.device:
        gaussian_meta = gaussian_meta.to(gaussian_ids.device)
    meta = gaussian_meta.to(dtype=torch.int32).contiguous()
    idx = gaussian_ids.to(dtype=torch.int64).contiguous()
    return torch.gather(meta, 0, idx)
