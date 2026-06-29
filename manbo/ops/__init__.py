from __future__ import annotations

import os


def _patch_tilelang_jit_compile_flags() -> None:
    import tilelang

    if getattr(tilelang.jit, "_cubicsplat_compile_flags_patch", False):
        return

    original_jit = tilelang.jit
    compile_flags = [
        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
    ]
    if os.name == "nt":
        compile_flags.extend(
            [
                "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                "-D_ENABLE_EXTENDED_ALIGNED_STORAGE",
            ]
        )

    def _merge_compile_flags(value):
        if value is None:
            return list(compile_flags)
        if isinstance(value, str):
            flags = [value]
        else:
            flags = list(value)
        for flag in compile_flags:
            if flag not in flags:
                flags.append(flag)
        return flags

    def jit_with_windows_nvcc_flags(func=None, **kwargs):
        kwargs["compile_flags"] = _merge_compile_flags(kwargs.get("compile_flags"))
        if func is None:
            return original_jit(**kwargs)
        return original_jit(func, **kwargs)

    jit_with_windows_nvcc_flags._cubicsplat_compile_flags_patch = True
    tilelang.jit = jit_with_windows_nvcc_flags


_patch_tilelang_jit_compile_flags()

from .backward2d import (
    project_gaussians_2d_scale_rot_backward_tilelang,
)
from .connected_components import (
    connected_components_labels_4_tilelang,
    connected_components_with_stats_4_tilelang,
    select_largest_component_center,
)
from .cubic_polyline_splatting import (
    render_cubic_polyline_splat_tilelang,
)
from .cubic_splatting import (
    closed_cubic_fill_segments_backward_kernel,
    closed_cubic_fill_segments_forward_kernel,
    closed_cubic_fill_segments_tilelang,
    map_cubic_to_intersects_kernel,
    map_cubic_to_intersects_tilelang,
    project_cubics_2d_forward_kernel,
    project_cubics_2d_forward_tilelang,
    rasterize_cubic_fill_backward_kernel,
    rasterize_cubic_fill_backward_tilelang,
    rasterize_cubic_fill_forward_kernel,
    rasterize_cubic_fill_forward_tilelang,
    render_cubic_fill_splat_tilelang,
)
from .forward2d import (
    project_gaussians_2d_scale_rot_forward_tilelang,
)
from .intersects import (
    map_gaussian_meta_to_intersects,
    map_gaussian_to_intersects_tilelang,
)
from .metric import (
    mse_backward_diff_kernel,
    mse_backward_diff_tilelang,
    mse_partial_sum_kernel,
    reduce_sum_1d_tilelang,
    reduce_sum_partial_kernel,
    ssim_map_kernel,
)
from .optim import (
    adan_step_from_grad_tilelang,
)
from .rasterize import (
    rasterize_backward_tilelang,
    rasterize_forward_tilelang,
)
from .regularization import (
    DEFAULT_REG_ITEMS_PER_THREAD,
    DEFAULT_REG_THREADS,
    bezier_open_backward_kernel,
    bezier_open_partial_sum_kernel,
    bezier_open_proj_outside_grad_tilelang,
    bezier_open_proj_outside_sum_tilelang,
    bezier_shape_backward_kernel,
    bezier_shape_partial_sum_kernel,
    bezier_shape_proj_outside_grad_tilelang,
    bezier_shape_proj_outside_sum_tilelang,
    boundary_backward_kernel,
    boundary_joints_penalty_grad_tilelang,
    boundary_joints_penalty_sum_tilelang,
    boundary_partial_sum_kernel,
    curvature_backward_kernel,
    curvature_masked_second_diff_grad_tilelang,
    curvature_masked_second_diff_sum_tilelang,
    curvature_partial_sum_kernel,
    opacity_abs_sigmoid_delta_grad_tilelang,
    opacity_abs_sigmoid_delta_sum_tilelang,
    opacity_backward_kernel,
    opacity_partial_sum_kernel,
)
from .tile_bins import (
    get_tile_bin_edges_tilelang,
)

__all__ = [
    "project_gaussians_2d_scale_rot_forward_tilelang",
    "project_gaussians_2d_scale_rot_backward_tilelang",
    "map_gaussian_to_intersects_tilelang",
    "map_gaussian_meta_to_intersects",
    "get_tile_bin_edges_tilelang",
    "rasterize_forward_tilelang",
    "rasterize_backward_tilelang",
    "project_cubics_2d_forward_kernel",
    "project_cubics_2d_forward_tilelang",
    "closed_cubic_fill_segments_forward_kernel",
    "closed_cubic_fill_segments_backward_kernel",
    "closed_cubic_fill_segments_tilelang",
    "map_cubic_to_intersects_kernel",
    "map_cubic_to_intersects_tilelang",
    "rasterize_cubic_fill_backward_kernel",
    "rasterize_cubic_fill_backward_tilelang",
    "rasterize_cubic_fill_forward_kernel",
    "rasterize_cubic_fill_forward_tilelang",
    "render_cubic_fill_splat_tilelang",
    "render_cubic_polyline_splat_tilelang",
    "connected_components_labels_4_tilelang",
    "connected_components_with_stats_4_tilelang",
    "select_largest_component_center",
    "adan_step_from_grad_tilelang",
    "mse_backward_diff_kernel",
    "mse_backward_diff_tilelang",
    "mse_partial_sum_kernel",
    "reduce_sum_partial_kernel",
    "ssim_map_kernel",
    "reduce_sum_1d_tilelang",
    "DEFAULT_REG_THREADS",
    "DEFAULT_REG_ITEMS_PER_THREAD",
    "bezier_shape_backward_kernel",
    "bezier_open_backward_kernel",
    "opacity_backward_kernel",
    "boundary_backward_kernel",
    "curvature_backward_kernel",
    "bezier_shape_partial_sum_kernel",
    "bezier_open_partial_sum_kernel",
    "opacity_partial_sum_kernel",
    "boundary_partial_sum_kernel",
    "curvature_partial_sum_kernel",
    "bezier_shape_proj_outside_grad_tilelang",
    "bezier_shape_proj_outside_sum_tilelang",
    "bezier_open_proj_outside_grad_tilelang",
    "bezier_open_proj_outside_sum_tilelang",
    "opacity_abs_sigmoid_delta_grad_tilelang",
    "opacity_abs_sigmoid_delta_sum_tilelang",
    "boundary_joints_penalty_grad_tilelang",
    "boundary_joints_penalty_sum_tilelang",
    "curvature_masked_second_diff_grad_tilelang",
    "curvature_masked_second_diff_sum_tilelang",
]
