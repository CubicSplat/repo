import tilelang
import tilelang.language as T
import torch

# Match gsplat_2d/gsplat/cuda/csrc/config.h
BLOCK_X = 16
BLOCK_Y = 16
DEFAULT_THREADS = 256
ANISO_SIGMA_MULT = 3.0
CONIC_EPS = 1e-12


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def project_gaussians_2d_scale_rot_forward_kernel(
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
    num_points=T.dynamic("num_points"),
):
    @T.prim_func
    def kernel(
        means2d: T.Tensor[[num_points, 2], dtype],
        scales2d: T.Tensor[[num_points, 2], dtype],
        rotation: T.Tensor[[num_points], dtype],
        xys: T.Tensor[[num_points, 2], dtype],
        depths: T.Tensor[[num_points], dtype],
        radii: T.Tensor[[num_points], T.int32],
        conics: T.Tensor[[num_points, 3], dtype],
        num_tiles_hit: T.Tensor[[num_points], T.int32],
        tile_bound_x: T.int32,
        tile_bound_y: T.int32,
        img_height: T.int32,
        img_width: T.int32,
        block_x: T.int32,
        block_y: T.int32,
    ):
        offset_x = T.Cast(dtype, 0.5) * T.Cast(dtype, img_width)
        offset_y = T.Cast(dtype, 0.5) * T.Cast(dtype, img_height)

        grid = T.ceildiv(num_points, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                idx = bx * threads + tx
                if idx < num_points:
                    T.assume(idx >= 0)
                    mx = means2d[idx, 0]
                    my = means2d[idx, 1]
                    sx = scales2d[idx, 0]
                    sy = scales2d[idx, 1]
                    rot = rotation[idx]

                    center_x = offset_x * mx + offset_x
                    center_y = offset_y * my + offset_y

                    cosr = T.cos(rot)
                    sinr = T.sin(rot)

                    m00 = cosr * sx
                    m01 = sinr * sy
                    m10 = -sinr * sx
                    m11 = cosr * sy

                    cov_xx = m00 * m00 + m01 * m01
                    cov_xy = m00 * m10 + m01 * m11
                    cov_yy = m10 * m10 + m11 * m11

                    det = cov_xx * cov_yy - cov_xy * cov_xy
                    out_x = T.alloc_var(dtype)
                    out_y = T.alloc_var(dtype)
                    out_depth = T.alloc_var(dtype)
                    out_radius = T.alloc_var("int32")
                    out_conic_x = T.alloc_var(dtype)
                    out_conic_y = T.alloc_var(dtype)
                    out_conic_z = T.alloc_var(dtype)
                    out_tiles = T.alloc_var("int32")
                    out_x = T.Cast(dtype, 0.0)
                    out_y = T.Cast(dtype, 0.0)
                    out_depth = T.Cast(dtype, 0.0)
                    out_radius = T.Cast(T.int32, 0)
                    out_conic_x = T.Cast(dtype, 0.0)
                    out_conic_y = T.Cast(dtype, 0.0)
                    out_conic_z = T.Cast(dtype, 0.0)
                    out_tiles = T.Cast(T.int32, 0)

                    if det != T.Cast(dtype, 0.0):
                        inv_det = T.Cast(dtype, 1.0) / det
                        conic_x = cov_yy * inv_det
                        conic_y = -cov_xy * inv_det
                        conic_z = cov_xx * inv_det

                        b = T.Cast(dtype, 0.5) * (cov_xx + cov_yy)
                        delta = T.max(T.Cast(dtype, 0.1), b * b - det)
                        sqrt_delta = T.sqrt(delta)
                        v1 = b + sqrt_delta
                        v2 = b - sqrt_delta
                        max_v = T.max(v1, v2)
                        radius = T.ceil(T.Cast(dtype, 3.0) * T.sqrt(max_v))

                        tile_center_x = center_x / T.Cast(dtype, block_x)
                        tile_center_y = center_y / T.Cast(dtype, block_y)

                        # Conservative anisotropic bbox from conic (inverse covariance):
                        # rx = k * sqrt(qyy / det(Q)), ry = k * sqrt(qxx / det(Q)).
                        q_det = conic_x * conic_z - conic_y * conic_y
                        radius_x = T.alloc_var(dtype)
                        radius_y = T.alloc_var(dtype)
                        radius_x = radius
                        radius_y = radius
                        if (
                            (conic_x > T.Cast(dtype, 0.0))
                            & (conic_z > T.Cast(dtype, 0.0))
                            & (q_det > T.Cast(dtype, CONIC_EPS))
                        ):
                            radius_x = T.Cast(dtype, ANISO_SIGMA_MULT) * T.sqrt(
                                conic_z / q_det
                            )
                            radius_y = T.Cast(dtype, ANISO_SIGMA_MULT) * T.sqrt(
                                conic_x / q_det
                            )

                        tile_radius_x = radius_x / T.Cast(dtype, block_x)
                        tile_radius_y = radius_y / T.Cast(dtype, block_y)

                        tile_min_x = T.clamp(
                            T.Cast(T.int32, tile_center_x - tile_radius_x),
                            T.Cast(T.int32, 0),
                            tile_bound_x,
                        )
                        tile_max_x = T.clamp(
                            T.Cast(
                                T.int32,
                                tile_center_x + tile_radius_x + T.Cast(dtype, 1.0),
                            ),
                            T.Cast(T.int32, 0),
                            tile_bound_x,
                        )
                        tile_min_y = T.clamp(
                            T.Cast(T.int32, tile_center_y - tile_radius_y),
                            T.Cast(T.int32, 0),
                            tile_bound_y,
                        )
                        tile_max_y = T.clamp(
                            T.Cast(
                                T.int32,
                                tile_center_y + tile_radius_y + T.Cast(dtype, 1.0),
                            ),
                            T.Cast(T.int32, 0),
                            tile_bound_y,
                        )

                        tile_area = (tile_max_x - tile_min_x) * (
                            tile_max_y - tile_min_y
                        )
                        if tile_area > T.Cast(T.int32, 0):
                            out_x = center_x
                            out_y = center_y
                            out_conic_x = conic_x
                            out_conic_y = conic_y
                            out_conic_z = conic_z
                            out_radius = T.Cast(T.int32, radius)
                            out_tiles = tile_area

                    xys[idx, 0] = out_x
                    xys[idx, 1] = out_y
                    depths[idx] = out_depth
                    radii[idx] = out_radius
                    conics[idx, 0] = out_conic_x
                    conics[idx, 1] = out_conic_y
                    conics[idx, 2] = out_conic_z
                    num_tiles_hit[idx] = out_tiles

    return kernel


def project_gaussians_2d_scale_rot_forward_tilelang(
    means2d: torch.Tensor,
    scales2d: torch.Tensor,
    rotation: torch.Tensor,
    img_height: int,
    img_width: int,
    tile_bounds: tuple[int, int, int],
    *,
    block_x: int = BLOCK_X,
    block_y: int = BLOCK_Y,
    threads: int = DEFAULT_THREADS,
):
    if rotation.dim() == 2 and rotation.shape[1] == 1:
        rotation = rotation.view(-1)

    num_points = means2d.shape[0]
    xys = torch.empty((num_points, 2), dtype=means2d.dtype, device=means2d.device)
    depths = torch.empty((num_points,), dtype=means2d.dtype, device=means2d.device)
    radii = torch.empty((num_points,), dtype=torch.int32, device=means2d.device)
    conics = torch.empty((num_points, 3), dtype=means2d.dtype, device=means2d.device)
    num_tiles_hit = torch.empty((num_points,), dtype=torch.int32, device=means2d.device)

    tile_bound_x = (img_width + block_x - 1) // block_x
    tile_bound_y = (img_height + block_y - 1) // block_y

    kernel = project_gaussians_2d_scale_rot_forward_kernel(
        threads=threads,
        dtype="float32",
    )

    kernel(
        means2d.contiguous(),
        scales2d.contiguous(),
        rotation.contiguous(),
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        tile_bound_x,
        tile_bound_y,
        img_height,
        img_width,
        block_x,
        block_y,
    )
    return xys, depths, radii, conics, num_tiles_hit
