from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_safe = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x_safe / (1.0 - x_safe))


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.float(), target.detach().float())


def bezier_shape_regularizer(
    control_points: torch.Tensor,
    bezier_degree: int = 3,
    lambda_proj: float = 1e-2,
) -> torch.Tensor:
    """Keep intermediate Bezier control points projected between two endpoints."""
    n, k, _ = control_points.shape
    expected = 2 * bezier_degree + 2
    if k != expected:
        raise ValueError(
            f"control_points shape mismatch: expected K={expected}, got K={k}"
        )

    p0 = control_points[:, 0, :]
    pend = control_points[:, bezier_degree + 1, :]

    idx = [i for i in range(k) if i not in (0, bezier_degree + 1)]
    mids = control_points[:, idx, :]

    v = pend - p0
    v_norm_sq = (v**2).sum(dim=1, keepdim=True) + 1e-8
    u = mids - p0[:, None, :]
    alpha = (u * v[:, None, :]).sum(dim=2) / v_norm_sq

    proj_outside = F.relu(alpha - 1.0) ** 2 + F.relu(-alpha) ** 2
    proj_loss = proj_outside.sum() / float(n)
    return lambda_proj * proj_loss


def bezier_open_shape_regularizer(
    control_points: torch.Tensor,
    lambda_proj: float = 1e-2,
) -> torch.Tensor:
    """
    Shape regularizer for open curves with arbitrary number of control points.
    Uses first/last control points as line endpoints and penalizes projected
    coefficients of middle control points outside [0, 1].
    """
    if control_points.dim() != 3 or control_points.shape[-1] != 2:
        raise ValueError(
            f"Expect control_points [N, K, 2], got {tuple(control_points.shape)}"
        )
    n, k, _ = control_points.shape
    if k < 3:
        return control_points.new_tensor(0.0)

    p0 = control_points[:, 0, :]
    pend = control_points[:, -1, :]
    mids = control_points[:, 1:-1, :]

    v = pend - p0
    v_norm_sq = (v**2).sum(dim=1, keepdim=True) + 1e-8
    u = mids - p0[:, None, :]
    alpha = (u * v[:, None, :]).sum(dim=2) / v_norm_sq

    proj_outside = F.relu(alpha - 1.0) ** 2 + F.relu(-alpha) ** 2
    proj_loss = proj_outside.sum() / float(max(1, n))
    return lambda_proj * proj_loss


def curvature_loss(
    paths: torch.Tensor,
    stride: int,
    angle_thresh_deg: float = 60.0,
) -> torch.Tensor:
    """
    paths: [N, 2, M, 2] (closed mode boundary pairs)
    """
    if paths.dim() != 4 or paths.shape[1] != 2:
        raise ValueError(
            f"Expect paths with shape [N, 2, M, 2], got {tuple(paths.shape)}"
        )

    path1 = paths[:, 0, :, :]
    path2 = torch.flip(paths[:, 1, :, :], dims=[1])
    full_path = torch.cat([path1, path2], dim=1)

    total_len = full_path.shape[1]
    indices = torch.arange(0, total_len, stride, device=paths.device)
    prev = torch.roll(full_path, 5, dims=1)[:, indices, :]
    curr = full_path[:, indices, :]
    nex = torch.roll(full_path, -5, dims=1)[:, indices, :]

    second_diff = prev - 2.0 * curr + nex
    curvature = second_diff.pow(2).sum(dim=-1)

    v1 = F.normalize(prev - curr, dim=-1)
    v2 = F.normalize(nex - curr, dim=-1)
    cos_theta = (v1 * v2).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_theta)

    angle_thresh_rad = math.radians(angle_thresh_deg)
    mask = (angle < angle_thresh_rad).float()
    return (curvature * mask).mean()


def boundary_loss_on_joints(
    points: torch.Tensor,
    degree: int,
    bound: float = 1.0,
) -> torch.Tensor:
    if points.dim() != 3 or points.shape[-1] != 2:
        raise ValueError(
            f"Expect points with shape [N, M, 2], got {tuple(points.shape)}"
        )

    joint_indices = torch.arange(0, points.shape[1], degree, device=points.device)
    joints = points[:, joint_indices, :]

    over = F.relu(joints - bound)
    under = F.relu(-bound - joints)
    return (over + under).mean()


def compute_rotated_bbox_vertices(
    cx: torch.Tensor,
    cy: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized rotated rectangle corners.
    Returns shape [..., 4, 2] in clockwise order.
    """
    cx = torch.as_tensor(cx)
    cy = torch.as_tensor(cy, device=cx.device, dtype=cx.dtype)
    width = torch.as_tensor(width, device=cx.device, dtype=cx.dtype)
    height = torch.as_tensor(height, device=cx.device, dtype=cx.dtype)
    angle = torch.as_tensor(angle, device=cx.device, dtype=cx.dtype)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    dx = width * 0.5
    dy = height * 0.5
    corners = torch.stack(
        [
            torch.stack([-dx, -dy], dim=-1),
            torch.stack([dx, -dy], dim=-1),
            torch.stack([dx, dy], dim=-1),
            torch.stack([-dx, dy], dim=-1),
        ],
        dim=-2,
    )

    rot = torch.stack(
        [
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1),
        ],
        dim=-2,
    )
    translated = corners @ rot.transpose(-1, -2)
    translated[..., 0] += cx.unsqueeze(-1)
    translated[..., 1] += cy.unsqueeze(-1)
    return translated


def compute_aabb_xyxy(points: torch.Tensor) -> torch.Tensor:
    """points: [N, M, 2] -> boxes [N, 4] as (x1, y1, x2, y2)."""
    if points.dim() != 3 or points.shape[-1] != 2:
        raise ValueError(f"Expect points [N, M, 2], got {tuple(points.shape)}")
    min_coords = points.min(dim=1).values
    max_coords = points.max(dim=1).values
    return torch.cat([min_coords, max_coords], dim=1)


def compute_outside_area_xyxy(
    boxes: torch.Tensor,
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """boxes: [N,4] in pixel coordinates, return area outside image bounds."""
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expect boxes [N, 4], got {tuple(boxes.shape)}")

    x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    total_area = (x_max - x_min) * (y_max - y_min)

    inter_x_min = torch.clamp(x_min, min=0.0, max=float(img_w))
    inter_y_min = torch.clamp(y_min, min=0.0, max=float(img_h))
    inter_x_max = torch.clamp(x_max, min=0.0, max=float(img_w))
    inter_y_max = torch.clamp(y_max, min=0.0, max=float(img_h))

    inter_w = (inter_x_max - inter_x_min).clamp(min=0.0)
    inter_h = (inter_y_max - inter_y_min).clamp(min=0.0)
    inter_area = inter_w * inter_h
    return total_area - inter_area


def to_pixel_space(points: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Map normalized [-1,1] coordinates to pixel-space (x,y)."""
    if points.dim() < 2 or points.shape[-1] != 2:
        raise ValueError(f"Expect points [..., 2], got {tuple(points.shape)}")
    out = points.clone()
    out[..., 0] = (out[..., 0] * 0.5 + 0.5) * float(img_w)
    out[..., 1] = (out[..., 1] * 0.5 + 0.5) * float(img_h)
    return out


__all__ = [
    "inverse_sigmoid",
    "l2_loss",
    "bezier_shape_regularizer",
    "bezier_open_shape_regularizer",
    "curvature_loss",
    "boundary_loss_on_joints",
    "compute_rotated_bbox_vertices",
    "compute_aabb_xyxy",
    "compute_outside_area_xyxy",
    "to_pixel_space",
]
