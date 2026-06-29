import tilelang
import tilelang.language as T
import torch

DEFAULT_THREADS = 256


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def project_gaussians_2d_scale_rot_backward_kernel(
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
    num_points=T.dynamic("num_points"),
):
    """TileLang kernel for gsplat_2d backward2d (scale+rot version)."""

    @T.prim_func
    def kernel(
        scales2d: T.Tensor[[num_points, 2], dtype],
        rotation: T.Tensor[[num_points], dtype],
        radii: T.Tensor[[num_points], T.int32],
        conics: T.Tensor[[num_points, 3], dtype],
        v_xy: T.Tensor[[num_points, 2], dtype],
        v_depth: T.Tensor[[num_points], dtype],
        v_conic: T.Tensor[[num_points, 3], dtype],
        v_cov2d: T.Tensor[[num_points, 3], dtype],
        v_mean2d: T.Tensor[[num_points, 2], dtype],
        v_scale: T.Tensor[[num_points, 2], dtype],
        v_rot: T.Tensor[[num_points], dtype],
        img_height: T.int32,
        img_width: T.int32,
    ):
        half = T.Cast(dtype, 0.5)
        img_w = T.Cast(dtype, img_width)
        img_h = T.Cast(dtype, img_height)
        grid = T.ceildiv(num_points, threads)
        with T.Kernel(grid, threads=threads) as bx:
            for tx in T.Parallel(threads):
                idx = bx * threads + tx
                if idx < num_points:
                    T.assume(idx >= 0)
                    v_cov2d[idx, 0] = T.Cast(dtype, 0.0)
                    v_cov2d[idx, 1] = T.Cast(dtype, 0.0)
                    v_cov2d[idx, 2] = T.Cast(dtype, 0.0)
                    v_mean2d[idx, 0] = T.Cast(dtype, 0.0)
                    v_mean2d[idx, 1] = T.Cast(dtype, 0.0)
                    v_scale[idx, 0] = T.Cast(dtype, 0.0)
                    v_scale[idx, 1] = T.Cast(dtype, 0.0)
                    v_rot[idx] = T.Cast(dtype, 0.0)

                    if radii[idx] > T.Cast(T.int32, 0):
                        c0 = conics[idx, 0]
                        c1 = conics[idx, 1]
                        c2 = conics[idx, 2]
                        g0 = v_conic[idx, 0]
                        g1 = v_conic[idx, 1]
                        g2 = v_conic[idx, 2]

                        h00 = c0 * g0 + c1 * g1
                        h01 = c0 * g1 + c1 * g2
                        h10 = c1 * g0 + c2 * g1
                        h11 = c1 * g1 + c2 * g2

                        v00 = -(h00 * c0 + h01 * c1)
                        v01 = -(h00 * c1 + h01 * c2)
                        v10 = -(h10 * c0 + h11 * c1)
                        v11 = -(h10 * c1 + h11 * c2)

                        g11 = v00
                        g12 = v10 + v01
                        g22 = v11

                        v_cov2d[idx, 0] = g11
                        v_cov2d[idx, 1] = g12
                        v_cov2d[idx, 2] = g22

                        sx = scales2d[idx, 0]
                        sy = scales2d[idx, 1]
                        rot = rotation[idx]

                        cosr = T.cos(rot)
                        sinr = T.sin(rot)

                        sigma_x_00 = T.Cast(dtype, 2.0) * sx * cosr * cosr
                        sigma_x_10 = -T.Cast(dtype, 2.0) * sx * sinr * cosr
                        sigma_x_11 = T.Cast(dtype, 2.0) * sx * sinr * sinr

                        sigma_y_00 = T.Cast(dtype, 2.0) * sy * sinr * sinr
                        sigma_y_10 = T.Cast(dtype, 2.0) * sy * cosr * sinr
                        sigma_y_11 = T.Cast(dtype, 2.0) * sy * cosr * cosr

                        v_scale[idx, 0] = (
                            g11 * sigma_x_00
                            + T.Cast(dtype, 2.0) * g12 * sigma_x_10
                            + g22 * sigma_x_11
                        )
                        v_scale[idx, 1] = (
                            g11 * sigma_y_00
                            + T.Cast(dtype, 2.0) * g12 * sigma_y_10
                            + g22 * sigma_y_11
                        )

                        r00 = cosr
                        r01 = sinr
                        r10 = -sinr
                        r11 = cosr

                        rg00 = -sinr
                        rg01 = cosr
                        rg10 = -cosr
                        rg11 = -sinr

                        m00 = r00 * sx
                        m01 = r01 * sy
                        m10 = r10 * sx
                        m11 = r11 * sy

                        a00 = rg00 * sx
                        a01 = rg01 * sy
                        a10 = rg10 * sx
                        a11 = rg11 * sy

                        t1_00 = a00 * m00 + a01 * m01
                        t1_10 = a10 * m00 + a11 * m01
                        t1_11 = a10 * m10 + a11 * m11

                        b00 = m00 * sx
                        b01 = m01 * sy
                        b10 = m10 * sx
                        b11 = m11 * sy

                        t2_00 = b00 * rg00 + b01 * rg01
                        t2_10 = b10 * rg00 + b11 * rg01
                        t2_11 = b10 * rg10 + b11 * rg11

                        theta_00 = t1_00 + t2_00
                        theta_10 = t1_10 + t2_10
                        theta_11 = t1_11 + t2_11

                        v_rot[idx] = (
                            g11 * theta_00
                            + T.Cast(dtype, 2.0) * g12 * theta_10
                            + g22 * theta_11
                        )

                        v_mean2d[idx, 0] = v_xy[idx, 0] * (half * img_w)
                        v_mean2d[idx, 1] = v_xy[idx, 1] * (half * img_h)

    return kernel


def project_gaussians_2d_scale_rot_backward_tilelang(
    means2d: torch.Tensor,
    scales2d: torch.Tensor,
    rotation: torch.Tensor,
    img_height: int,
    img_width: int,
    radii: torch.Tensor,
    conics: torch.Tensor,
    v_xy: torch.Tensor,
    v_depth: torch.Tensor,
    v_conic: torch.Tensor,
    *,
    threads: int = DEFAULT_THREADS,
):
    """TileLang implementation of gsplat_2d backward2d (scale+rot version)."""
    if rotation.dim() == 1:
        rotation_flat = rotation
        rot_was_2d = False
    elif rotation.dim() == 2 and rotation.shape[1] == 1:
        rotation_flat = rotation.view(-1)
        rot_was_2d = True
    else:
        raise ValueError("rotation must be shape (N,) or (N,1)")

    if v_depth.dim() == 1:
        v_depth_flat = v_depth
    elif v_depth.dim() == 2 and v_depth.shape[1] == 1:
        v_depth_flat = v_depth.view(-1)
    else:
        raise ValueError("v_depth must be shape (N,) or (N,1)")

    num_points = means2d.shape[0]

    v_cov2d = torch.empty((num_points, 3), dtype=means2d.dtype, device=means2d.device)
    v_mean2d = torch.empty((num_points, 2), dtype=means2d.dtype, device=means2d.device)
    v_scale = torch.empty((num_points, 2), dtype=means2d.dtype, device=means2d.device)
    v_rot = torch.empty((num_points,), dtype=means2d.dtype, device=means2d.device)

    kernel = project_gaussians_2d_scale_rot_backward_kernel(
        threads=threads,
        dtype="float32",
    )

    kernel(
        scales2d.contiguous(),
        rotation_flat.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        v_xy.contiguous(),
        v_depth_flat.contiguous(),
        v_conic.contiguous(),
        v_cov2d,
        v_mean2d,
        v_scale,
        v_rot,
        img_height,
        img_width,
    )

    if rot_was_2d:
        v_rot = v_rot.view(-1, 1)
    return v_cov2d, v_mean2d, v_scale, v_rot
