from __future__ import annotations

from typing import Optional

import torch

from .ops import (
    DEFAULT_REG_ITEMS_PER_THREAD,
    DEFAULT_REG_THREADS,
    bezier_open_proj_outside_grad_tilelang,
    bezier_open_proj_outside_sum_tilelang,
    bezier_shape_proj_outside_grad_tilelang,
    bezier_shape_proj_outside_sum_tilelang,
    boundary_joints_penalty_grad_tilelang,
    boundary_joints_penalty_sum_tilelang,
    curvature_masked_second_diff_grad_tilelang,
    curvature_masked_second_diff_sum_tilelang,
    opacity_abs_sigmoid_delta_grad_tilelang,
    opacity_abs_sigmoid_delta_sum_tilelang,
)

"""
TileLang regularization terms used by GaussianTrace.

Notation:
- N: number of curves.
- cp: control points with shape [N, K, 2].
- op: opacity logits (before sigmoid).
- sigmoid(x): 1 / (1 + exp(-x)).

Total loss in this module:
L = L_shape + L_opacity + I_closed * (L_curvature + L_boundary)

Per-operator formulas:
1) Shape projection penalty (open/closed):
   L_shape = (lambda_proj / N) * sum_{i,j} phi(alpha_{i,j}),
   phi(a) = ReLU(a - 1)^2 + ReLU(-a)^2.
   alpha_{i,j} is the projection ratio of a middle control point on the
   segment from the first endpoint to the second endpoint.

2) Opacity penalty:
   L_opacity = opacity_weight * mean(|sigmoid(op) - 1|).

3) Curvature penalty (closed only):
   L_curvature = mean(||prev - 2*curr + next||_2^2 * mask(angle < thresh)).

4) Boundary joint penalty (closed only):
   L_boundary = mean(ReLU(x - bound) + ReLU(-bound - x)).
"""


def _resolve_shape_bezier_degree(
    *,
    mode: str,
    num_beziers: int,
    bezier_degree: int,
) -> int:
    if mode == "unclosed":
        return int(bezier_degree)
    if int(num_beziers) > 2:
        return int(bezier_degree) * int(num_beziers // 2) + 1
    return int(bezier_degree)


def _regularization_loss_tilelang_forward(
    control_points: torch.Tensor,
    opacity: torch.Tensor,
    xyz: Optional[torch.Tensor],
    *,
    mode: str,
    num_beziers: int,
    bezier_degree: int,
    num_samples: int,
    lambda_proj: float,
    opacity_weight: float,
    boundary_degree: int,
    boundary_bound: float,
    curvature_angle_thresh_deg: float,
    threads: int,
    items_per_thread: int,
) -> torch.Tensor:
    mode_norm = str(mode).lower()
    if mode_norm not in {"closed", "unclosed"}:
        raise ValueError(f"Unsupported mode={mode!r}; expected 'closed' or 'unclosed'")

    n_curves = int(control_points.shape[0])
    if n_curves <= 0:
        raise ValueError("control_points must include at least one curve")

    if mode_norm == "unclosed":
        # L_shape(open) = lambda_proj / N * sum phi(alpha),
        # phi(alpha) = ReLU(alpha - 1)^2 + ReLU(-alpha)^2.
        shape_sum = bezier_open_proj_outside_sum_tilelang(
            control_points,
            threads=int(threads),
            items_per_thread=int(items_per_thread),
        )
        shape_loss = float(lambda_proj) * shape_sum / float(max(1, n_curves))
    else:
        shape_bezier_degree = _resolve_shape_bezier_degree(
            mode=mode_norm,
            num_beziers=int(num_beziers),
            bezier_degree=int(bezier_degree),
        )
        expected = 2 * int(shape_bezier_degree) + 2
        if int(control_points.shape[1]) != int(expected):
            raise ValueError(
                f"control_points shape mismatch for closed regularizer: expected K={expected}, got {control_points.shape[1]}"
            )

        # L_shape(closed) has the same phi(alpha) form as open mode,
        # but middle points are split around the seam split_idx.
        shape_sum = bezier_shape_proj_outside_sum_tilelang(
            control_points,
            split_idx=int(shape_bezier_degree) + 1,
            threads=int(threads),
            items_per_thread=int(items_per_thread),
        )
        shape_loss = float(lambda_proj) * shape_sum / float(n_curves)

    # L_opacity = opacity_weight * mean(|sigmoid(opacity) - 1|).
    op_sum = opacity_abs_sigmoid_delta_sum_tilelang(
        opacity,
        threads=int(threads),
        items_per_thread=int(items_per_thread),
    )
    opacity_loss = float(opacity_weight) * op_sum / float(max(1, int(opacity.numel())))

    loss = shape_loss + opacity_loss

    if mode_norm == "closed":
        if xyz is None:
            raise ValueError("xyz must be provided when mode='closed'")

        # L_curvature = mean(||prev - 2*curr + next||^2 * mask(angle < thresh)).
        curvature_sum, sampled_per_curve = curvature_masked_second_diff_sum_tilelang(
            xyz,
            stride=int(num_samples),
            angle_thresh_deg=float(curvature_angle_thresh_deg),
            threads=int(threads),
            items_per_thread=int(items_per_thread),
        )
        curvature_loss_val = curvature_sum / float(
            max(1, int(n_curves) * int(sampled_per_curve))
        )

        # L_boundary = mean(ReLU(x - bound) + ReLU(-bound - x))
        # over boundary joints and x/y coordinates.
        boundary_sum = boundary_joints_penalty_sum_tilelang(
            control_points,
            degree=int(boundary_degree),
            bound=float(boundary_bound),
            threads=int(threads),
            items_per_thread=int(items_per_thread),
        )
        joint_count = (int(control_points.shape[1]) + int(boundary_degree) - 1) // int(
            boundary_degree
        )
        boundary_denom = max(1, int(n_curves) * int(joint_count) * 2)
        boundary_loss_val = boundary_sum / float(boundary_denom)

        loss = loss + curvature_loss_val + boundary_loss_val

    return loss


def _regularization_backward_tilelang(
    control_points: torch.Tensor,
    opacity: torch.Tensor,
    xyz: Optional[torch.Tensor],
    *,
    mode: str,
    num_beziers: int,
    bezier_degree: int,
    num_samples: int,
    lambda_proj: float,
    opacity_weight: float,
    boundary_degree: int,
    boundary_bound: float,
    curvature_angle_thresh_deg: float,
    threads: int,
    items_per_thread: int,
    needs_control_points: bool,
    needs_opacity: bool,
    needs_xyz: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    mode_norm = str(mode).lower()
    if mode_norm not in {"closed", "unclosed"}:
        raise ValueError(f"Unsupported mode={mode!r}; expected 'closed' or 'unclosed'")

    grad_cp: Optional[torch.Tensor] = None
    grad_opacity: Optional[torch.Tensor] = None
    grad_xyz: Optional[torch.Tensor] = None

    if needs_opacity:
        # d L_opacity / d opacity
        grad_opacity = opacity_abs_sigmoid_delta_grad_tilelang(
            opacity,
            opacity_weight=float(opacity_weight),
            threads=int(threads),
        )

    if needs_control_points:
        grad_cp = torch.zeros_like(control_points, dtype=torch.float32)
        if mode_norm == "unclosed":
            # d L_shape(open) / d control_points
            grad_cp = grad_cp + bezier_open_proj_outside_grad_tilelang(
                control_points,
                lambda_proj=float(lambda_proj),
                threads=int(threads),
                items_per_thread=int(items_per_thread),
            )
        else:
            shape_bezier_degree = _resolve_shape_bezier_degree(
                mode=mode_norm,
                num_beziers=int(num_beziers),
                bezier_degree=int(bezier_degree),
            )
            split_idx = int(shape_bezier_degree) + 1
            # d L_shape(closed) / d control_points
            grad_cp = grad_cp + bezier_shape_proj_outside_grad_tilelang(
                control_points,
                split_idx=int(split_idx),
                lambda_proj=float(lambda_proj),
                threads=int(threads),
                items_per_thread=int(items_per_thread),
            )
            # d L_boundary / d control_points
            grad_cp = grad_cp + boundary_joints_penalty_grad_tilelang(
                control_points,
                degree=int(boundary_degree),
                bound=float(boundary_bound),
                threads=int(threads),
                items_per_thread=int(items_per_thread),
            )

    if mode_norm == "closed" and needs_xyz:
        if xyz is None:
            raise ValueError("xyz must be provided when mode='closed'")
        # d L_curvature / d xyz
        grad_xyz = curvature_masked_second_diff_grad_tilelang(
            xyz,
            stride=int(num_samples),
            angle_thresh_deg=float(curvature_angle_thresh_deg),
            threads=int(threads),
            items_per_thread=int(items_per_thread),
        )

    return grad_cp, grad_opacity, grad_xyz


class _TileLangRegularizationFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        control_points: torch.Tensor,
        opacity: torch.Tensor,
        xyz: torch.Tensor,
        has_xyz: int,
        mode: str,
        num_beziers: int,
        bezier_degree: int,
        num_samples: int,
        lambda_proj: float,
        opacity_weight: float,
        boundary_degree: int,
        boundary_bound: float,
        curvature_angle_thresh_deg: float,
        threads: int,
        items_per_thread: int,
    ) -> torch.Tensor:
        xyz_arg = xyz if int(has_xyz) != 0 else None
        out = _regularization_loss_tilelang_forward(
            control_points,
            opacity,
            xyz_arg,
            mode=str(mode),
            num_beziers=int(num_beziers),
            bezier_degree=int(bezier_degree),
            num_samples=int(num_samples),
            lambda_proj=float(lambda_proj),
            opacity_weight=float(opacity_weight),
            boundary_degree=int(boundary_degree),
            boundary_bound=float(boundary_bound),
            curvature_angle_thresh_deg=float(curvature_angle_thresh_deg),
            threads=int(threads),
            items_per_thread=int(items_per_thread),
        )
        ctx.save_for_backward(control_points, opacity, xyz)
        ctx.has_xyz = bool(int(has_xyz))
        ctx.mode = str(mode)
        ctx.num_beziers = int(num_beziers)
        ctx.bezier_degree = int(bezier_degree)
        ctx.num_samples = int(num_samples)
        ctx.lambda_proj = float(lambda_proj)
        ctx.opacity_weight = float(opacity_weight)
        ctx.boundary_degree = int(boundary_degree)
        ctx.boundary_bound = float(boundary_bound)
        ctx.curvature_angle_thresh_deg = float(curvature_angle_thresh_deg)
        ctx.threads = int(threads)
        ctx.items_per_thread = int(items_per_thread)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        control_points, opacity, xyz = ctx.saved_tensors
        needs_cp = bool(ctx.needs_input_grad[0])
        needs_opacity = bool(ctx.needs_input_grad[1])
        needs_xyz = bool(ctx.needs_input_grad[2]) and bool(ctx.has_xyz)

        if not (needs_cp or needs_opacity or needs_xyz):
            return (None,) * 15

        with torch.no_grad():
            g_cp, g_opacity, g_xyz = _regularization_backward_tilelang(
                control_points,
                opacity,
                xyz if ctx.has_xyz else None,
                mode=ctx.mode,
                num_beziers=ctx.num_beziers,
                bezier_degree=ctx.bezier_degree,
                num_samples=ctx.num_samples,
                lambda_proj=ctx.lambda_proj,
                opacity_weight=ctx.opacity_weight,
                boundary_degree=ctx.boundary_degree,
                boundary_bound=ctx.boundary_bound,
                curvature_angle_thresh_deg=ctx.curvature_angle_thresh_deg,
                threads=ctx.threads,
                items_per_thread=ctx.items_per_thread,
                needs_control_points=needs_cp,
                needs_opacity=needs_opacity,
                needs_xyz=needs_xyz,
            )

        grad_scale = grad_output.to(dtype=control_points.dtype)
        grad_cp = g_cp * grad_scale if needs_cp else None
        grad_opacity = g_opacity * grad_scale if needs_opacity else None
        grad_xyz = g_xyz * grad_scale if (needs_xyz and g_xyz is not None) else None

        return (
            grad_cp,
            grad_opacity,
            grad_xyz,
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


def regularization_loss_tilelang(
    control_points: torch.Tensor,
    opacity: torch.Tensor,
    xyz: Optional[torch.Tensor] = None,
    *,
    mode: str,
    num_beziers: int,
    bezier_degree: int,
    num_samples: int,
    lambda_proj: float = 1e-2,
    opacity_weight: float = 1e-2,
    boundary_degree: Optional[int] = None,
    boundary_bound: float = 1.0,
    curvature_angle_thresh_deg: float = 60.0,
    threads: int = DEFAULT_REG_THREADS,
    items_per_thread: int = DEFAULT_REG_ITEMS_PER_THREAD,
    use_autograd: bool = True,
) -> torch.Tensor:
    """Compute regularization loss with TileLang kernels.

    Formula summary:
    L = L_shape + L_opacity + I_closed * (L_curvature + L_boundary).

    - L_shape: projection-outside penalty on control points.
    - L_opacity: opacity_weight * mean(|sigmoid(opacity) - 1|).
    - L_curvature: masked second-difference energy on sampled paths (closed only).
    - L_boundary: hinge penalty on boundary joints (closed only).
    """
    mode_norm = str(mode).lower()
    if mode_norm not in {"closed", "unclosed"}:
        raise ValueError(f"Unsupported mode={mode!r}; expected 'closed' or 'unclosed'")
    if mode_norm == "closed" and xyz is None:
        raise ValueError("xyz must be provided when mode='closed'")

    boundary_deg = (
        int(bezier_degree) + 1 if boundary_degree is None else int(boundary_degree)
    )
    xyz_tensor = xyz
    has_xyz = int(xyz is not None)
    if xyz_tensor is None:
        xyz_tensor = control_points.new_empty((0, 2, 0, 2))

    if not use_autograd:
        return _regularization_loss_tilelang_forward(
            control_points,
            opacity,
            xyz if has_xyz else None,
            mode=mode_norm,
            num_beziers=int(num_beziers),
            bezier_degree=int(bezier_degree),
            num_samples=int(num_samples),
            lambda_proj=float(lambda_proj),
            opacity_weight=float(opacity_weight),
            boundary_degree=int(boundary_deg),
            boundary_bound=float(boundary_bound),
            curvature_angle_thresh_deg=float(curvature_angle_thresh_deg),
            threads=int(threads),
            items_per_thread=int(items_per_thread),
        )

    return _TileLangRegularizationFn.apply(
        control_points,
        opacity,
        xyz_tensor,
        int(has_xyz),
        mode_norm,
        int(num_beziers),
        int(bezier_degree),
        int(num_samples),
        float(lambda_proj),
        float(opacity_weight),
        int(boundary_deg),
        float(boundary_bound),
        float(curvature_angle_thresh_deg),
        int(threads),
        int(items_per_thread),
    )


__all__ = ["regularization_loss_tilelang"]
