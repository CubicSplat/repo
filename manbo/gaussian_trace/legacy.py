from __future__ import annotations

import torch

from visual_debug import emit_visual_debug

from .common import warn_legacy_method


class GaussianTraceLegacyMixin:
    def compute_adjusted_tangents_from_points(self, points, length=1000.0):
        warn_legacy_method(self, "compute_adjusted_tangents_from_points")
        directions = points[:, 1:] - points[:, :-1]
        last_direction = directions[:, -1:]
        directions = torch.cat([directions, last_direction], dim=1)

        norms = torch.norm(directions, dim=-1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        directions = directions / norms

        n, h, _ = points.shape
        mask = torch.ones((n, h, 1), device=points.device)
        midpoint = h // 2
        mask[:, midpoint:] = -1

        tangents = directions * mask * length
        return tangents

    def calculate_shape_grad(self, points, color_grad):
        warn_legacy_method(self, "calculate_shape_grad")
        tangents = self.compute_adjusted_tangents_from_points(points)
        contrib = torch.abs(color_grad.sum(dim=-1, keepdim=True))
        return tangents * contrib

    def visualize_lines_with_tangents(self, points, tangents, colors):
        warn_legacy_method(self, "visualize_lines_with_tangents")
        tangents = self.compute_adjusted_tangents_from_points(points)
        emit_visual_debug(
            self.debug_hook,
            "line_tangent_debug",
            {
                "step": self.iter,
                "points": points.detach(),
                "tangents": tangents.detach(),
                "colors": colors.detach()
                if isinstance(colors, torch.Tensor)
                else colors,
            },
        )

    def bezier_same_side_mask(self, bezier_curves: torch.Tensor) -> torch.Tensor:
        warn_legacy_method(self, "bezier_same_side_mask")
        p0 = bezier_curves[:, 0]
        p4 = bezier_curves[:, 4]
        line_vec = p4 - p0
        control_points = torch.cat(
            [bezier_curves[:, 1:4], bezier_curves[:, 5:8]], dim=1
        )
        vecs_to_points = control_points - p0[:, None, :]
        cross_products = (
            line_vec[:, 0:1] * vecs_to_points[:, :, 1]
            - line_vec[:, 1:2] * vecs_to_points[:, :, 0]
        )
        same_side_mask = ((cross_products > 0).sum(dim=1) > 5) | (
            (cross_products < 0).all(dim=1) > 5
        )
        return same_side_mask

    def calculate_edge_grad(self):
        warn_legacy_method(self, "calculate_edge_grad")
        if not hasattr(self, "_tangents"):
            return
        colors = self.get_features.view(self.num_curves, -1, 3)
        self.visualize_lines_with_tangents(
            self.xys.view(self.num_curves, -1, 2)[0:1, :, :],
            self._tangents[0:1, :, :],
            colors[0:1, :, :],
        )
        if self.xys.grad is not None:
            self.xys.grad += self._tangents.view(-1, 2)

    def project_point_to_line(self, p, a, b):
        warn_legacy_method(self, "project_point_to_line")
        ab = b - a
        ap = p - a
        ab_unit = ab / (torch.norm(ab, dim=-1, keepdim=True) + 1e-8)
        return torch.sum(ap * ab_unit, dim=-1)

    def vector_cross(self, a, b):
        warn_legacy_method(self, "vector_cross")
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def shape_alignment(self):
        warn_legacy_method(self, "shape_alignment")
        control_points = self._control_points.clone().detach()
        control_points_fixed = control_points.clone()
        d = control_points[:, 4] - control_points[:, 0]

        proj_p2 = self.project_point_to_line(
            control_points[:, 1], control_points[:, 0], control_points[:, 4]
        )
        proj_p3 = self.project_point_to_line(
            control_points[:, 2], control_points[:, 0], control_points[:, 4]
        )
        proj_p4 = self.project_point_to_line(
            control_points[:, 3], control_points[:, 0], control_points[:, 4]
        )

        swap_mask_1 = proj_p3 < proj_p2
        if swap_mask_1.any():
            temp = control_points_fixed[swap_mask_1, 1].clone()
            control_points_fixed[swap_mask_1, 1] = control_points_fixed[swap_mask_1, 2]
            control_points_fixed[swap_mask_1, 2] = temp

        swap_mask_2 = proj_p4 < proj_p3
        if swap_mask_1.any():
            temp = control_points_fixed[swap_mask_2, 2].clone()
            control_points_fixed[swap_mask_2, 2] = control_points_fixed[swap_mask_2, 3]
            control_points_fixed[swap_mask_2, 3] = temp

        proj_p5 = self.project_point_to_line(
            control_points[:, 5], control_points[:, 0], control_points[:, 4]
        )
        proj_p6 = self.project_point_to_line(
            control_points[:, 6], control_points[:, 0], control_points[:, 4]
        )
        proj_p7 = self.project_point_to_line(
            control_points[:, 7], control_points[:, 0], control_points[:, 4]
        )
        swap_mask_3 = proj_p5 > proj_p6
        if swap_mask_3.any():
            temp = control_points_fixed[swap_mask_3, 5].clone()
            control_points_fixed[swap_mask_3, 5] = control_points_fixed[swap_mask_3, 6]
            control_points_fixed[swap_mask_3, 6] = temp

        swap_mask_4 = proj_p6 > proj_p7
        if swap_mask_4.any():
            temp = control_points_fixed[swap_mask_4, 6].clone()
            control_points_fixed[swap_mask_4, 6] = control_points_fixed[swap_mask_4, 7]
            control_points_fixed[swap_mask_4, 7] = temp
        with torch.no_grad():
            self._control_points.copy_(control_points_fixed)

    def shape_refinement(self, threshold=0.1):
        warn_legacy_method(self, "shape_refinement")
        for i in range(self.num_beziers):
            ba_vec = (
                self._control_points[:, 3 * i + 1, :]
                - self._control_points[:, 3 * i + 0, :]
            )
            cd_vec = (
                self._control_points[:, 3 * i + 2, :]
                - self._control_points[:, 3 * i + 3, :]
            )
            bc_vec = (
                self._control_points[:, 3 * i + 1, :]
                - self._control_points[:, 3 * i + 2, :]
            )
            cb_vec = (
                self._control_points[:, 3 * i + 2, :]
                - self._control_points[:, 3 * i + 1, :]
            )

            dot = (ba_vec * cd_vec).sum(dim=1)
            eps = 1e-8
            norm_v1 = ba_vec.norm(dim=1) + eps
            norm_v2 = cd_vec.norm(dim=1) + eps

            norm_mask = norm_v1 >= norm_v2
            cos_theta = dot / (norm_v1 * norm_v2)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angles = torch.acos(cos_theta)

            threshold = 0.1
            threshold_turn = 0.1
            mask = angles < threshold
            new_beizers = self._control_points.clone()
            new_beizers[mask & norm_mask][:, [3 * i + 0, 3 * i + 3], :] = (
                self._control_points[(mask & norm_mask)][:, [3 * i + 0, 3 * i + 3], :]
            )
            new_beizers[mask & norm_mask][:, [3 * i + 0, 3 * i + 3], :] = (
                self._control_points[(mask & norm_mask)][:, [3 * i + 2, 3 * i + 3], :]
            )
            new_beizers[mask] = self.modified_control_points(new_beizers[mask])
        with torch.no_grad():
            self._control_points.copy_(new_beizers)


__all__ = ["GaussianTraceLegacyMixin"]
