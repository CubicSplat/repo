from __future__ import annotations

import torch
import torch.nn as nn

from .utils import compute_rotated_bbox_vertices, inverse_sigmoid


class GaussianTraceOptimizerStateMixin:
    def remove_curves_mask(self):
        if self.mode == "closed":
            return self.remove_curves_mask_area()
        if self.mode == "unclosed":
            return self.remove_curves_mask_line()
        raise ValueError(f"Unsupported mode: {self.mode}")

    def remove_curves_mask_line(
        self,
        top_k=0.01,
        iou_threshold=0.2,
        color_threshold=0.03,
        remove_num=None,
        imagesize=None,
    ):
        color_threshold_input = color_threshold
        if self.iter < 7000:
            area_threshold = 5000
            color_threshold = color_threshold_input
        else:
            area_threshold = 500
        num_curves = self._control_points.shape[0]
        xys = self.xys.view(num_curves, -1, 2).detach()
        boxes = self.compute_aabb(xys)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        areas = widths * heights
        outside_area = self.compute_outside_area(boxes)
        ratio = outside_area / areas
        mask_outside = ratio > 0.6
        inter_left = torch.max(boxes[:, None, 0], boxes[None, :, 0])
        inter_top = torch.max(boxes[:, None, 1], boxes[None, :, 1])
        inter_right = torch.min(boxes[:, None, 2], boxes[None, :, 2])
        inter_bottom = torch.min(boxes[:, None, 3], boxes[None, :, 3])

        inter_width = (inter_right - inter_left).clamp(min=0)
        inter_height = (inter_bottom - inter_top).clamp(min=0)
        inter_area = inter_width * inter_height

        ratio_matrix = inter_area / areas.unsqueeze(1)
        ratio_matrix.fill_diagonal_(0)
        color = self._features_dc.clone() * self.opacity_activation(self._opacity)
        color_diff = torch.norm(color.unsqueeze(1) - color.unsqueeze(0), dim=-1)

        keep = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)
        iou_mask = ratio_matrix > iou_threshold
        color_mask = color_diff < color_threshold
        suppress_matrix = iou_mask & color_mask
        suppress_matrix.fill_diagonal_(0)
        keep[mask_outside] = False

        xys = self.xys.clone().detach().view(self._control_points.shape[0], -1, 2)
        line_areas = torch.norm(xys[:, 1:, :] - xys[:, :-1, :], dim=-1).sum(-1)
        line_areas = line_areas * torch.abs(self._scaling.detach()).squeeze(-1)
        opacities = torch.sigmoid(self._opacity).squeeze(-1)
        opacities_threshold = 0.6
        if self.iter > 10000:
            areas_mask = line_areas < area_threshold
            opacities_mask = opacities.sum(-1) < opacities_threshold / 2
            keep[areas_mask & opacities_mask] = False
        else:
            opacities_mask = opacities < opacities_threshold
            keep[opacities.sum(-1) < opacities_threshold] = False

        for idx in range(len(areas)):
            if not keep[idx]:
                continue
            if suppress_matrix[idx][idx + 1 :].sum() > 0:
                slice_part = suppress_matrix[idx, idx + 1 :]
                relative_idx = (slice_part > 0).nonzero(as_tuple=False)
                original_idx = relative_idx + (idx + 1)
                total_match_count = 0.0
                for qualified_idx in original_idx:
                    match_count, matched_distances = self.compute_pairwise_overlap(
                        xys[idx], xys[qualified_idx]
                    )
                    total_match_count += match_count
                if self.iter > 10000:
                    if (
                        total_match_count / float(self.num_samples) > 0.6
                    ) and line_areas[idx] < area_threshold:
                        keep[idx] = False
                else:
                    if total_match_count / float(self.num_samples) > 0.6:
                        keep[idx] = False
            if (
                (widths[idx] < 4)
                & (heights[idx] < 4)
                & (widths[idx] * heights[idx] < 12)
            ):
                keep[idx] = False
        return keep

    def remove_curves_mask_area(
        self,
        top_k=0.01,
        iou_threshold=0.1,
        color_threshold=0.05,
        remove_num=None,
        imagesize=None,
    ):
        color_threshold_input = color_threshold
        if self.iter < 6000:
            area_threshold = 50000
            color_threshold = color_threshold_input
        elif self.iter < 8000:
            area_threshold = 20000
        else:
            area_threshold = 2000
        if remove_num is not None:
            values, indices = torch.topk(
                self._grad
                / torch.min(
                    torch.abs(self._cholesky[:, :1]), torch.abs(self._cholesky[:, 2:])
                ),
                remove_num,
                dim=0,
                largest=True,
            )

        num_curves = self._control_points.shape[0]
        xys = self.xys.view(num_curves, -1, 2).detach()
        boxes = self.compute_aabb(xys)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        areas = widths * heights
        outside_area = self.compute_outside_area(boxes)
        ratio = outside_area / areas
        mask_outside = ratio > 0.6
        inter_left = torch.max(boxes[:, None, 0], boxes[None, :, 0])
        inter_top = torch.max(boxes[:, None, 1], boxes[None, :, 1])
        inter_right = torch.min(boxes[:, None, 2], boxes[None, :, 2])
        inter_bottom = torch.min(boxes[:, None, 3], boxes[None, :, 3])

        inter_width = (inter_right - inter_left).clamp(min=0)
        inter_height = (inter_bottom - inter_top).clamp(min=0)
        inter_area = inter_width * inter_height
        ratio_matrix = inter_area / areas.unsqueeze(1)
        ratio_matrix.fill_diagonal_(0)
        color = torch.sigmoid(self._features_dc.clone()) * self.opacity_activation(
            self._opacity
        )
        color_diff = torch.norm(color.unsqueeze(1) - color.unsqueeze(0), dim=-1)
        keep = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)
        iou_mask = ratio_matrix > iou_threshold
        color_mask = color_diff < color_threshold
        suppress_matrix = iou_mask & color_mask
        suppress_matrix.fill_diagonal_(0)
        remove_by_overlap = 0

        keep[mask_outside] = False
        locked_indices = set()
        for idx in range(len(areas)):
            if not keep[idx]:
                continue
            if idx in locked_indices:
                continue
            if suppress_matrix[idx][idx + 1 :].sum() > 0:
                slice_part = suppress_matrix[idx, idx + 1 :]
                relative_idx = (slice_part > 0).nonzero(as_tuple=False)
                original_idx = relative_idx + (idx + 1)
                iou_total = 0
                weight_iou_total = 0
                for qualified_idx in original_idx:
                    iou = ratio_matrix[idx, qualified_idx]
                    iou_color_diff = color_diff[idx, qualified_idx]
                    weight_iou_total += iou / (
                        torch.sigmoid(iou_color_diff * 100) + 1e-2
                    )
                    iou_total += iou
                if weight_iou_total > 0.5 and areas[idx] < area_threshold:
                    keep[idx] = False
                    remove_by_overlap += 1

                    for qualified_idx in original_idx:
                        locked_indices.add(qualified_idx)

            if remove_num is None:
                if (
                    (widths[idx] < 5)
                    & (heights[idx] < 5)
                    & (widths[idx] * heights[idx] < 16)
                ):
                    keep[idx] = False
        opacities = torch.sigmoid(self._opacity)
        if self.opacity_mode == 1:
            opacities_threshold = 0.1
            opacities_threshold_final = 0.05
        else:
            opacities_threshold = 0.2
            opacities_threshold_final = 0.2
        if self.iter > 7000:
            areas_mask = areas < area_threshold
            if self.opacity_mode == 1:
                opacities_threshold_final = 0.3
            opacities_mask = opacities.sum(-1) < opacities_threshold_final
            keep[areas_mask & opacities_mask] = False
        else:
            if self.opacity_mode == 1:
                opacities_threshold_final = 0.6
            opacities_mask = opacities.sum(-1) < opacities_threshold
            keep[opacities_mask] = False

        return keep

    def compute_rotated_bbox_vertices(self, cx, cy, width, height, angle):
        corners = compute_rotated_bbox_vertices(cx, cy, width, height, angle)
        return corners.tolist() if isinstance(corners, torch.Tensor) else corners

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                stored_state["exp_avg_diff"] = stored_state["exp_avg_diff"][mask]
                stored_state["neg_pre_grad"] = stored_state["neg_pre_grad"][mask]

                del self.optimizer.state[group["params"][0]]
                if group["name"] == "xyz":
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                else:
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
                opacity_mask = mask.detach().cpu()
                group["params"][0] = nn.Parameter(
                    group["params"][0][opacity_mask].requires_grad_(False)
                )
                optimizable_tensors[group["name"]] = group["params"][0].to(mask.device)
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )
                stored_state["exp_avg_diff"] = torch.cat(
                    (stored_state["exp_avg_diff"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )
                stored_state["neg_pre_grad"] = torch.cat(
                    (stored_state["neg_pre_grad"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (
                            group["params"][0],
                            extension_tensor.to(group["params"][0].device),
                        ),
                        dim=0,
                    ).requires_grad_(False)
                )
                optimizable_tensors[group["name"]] = group["params"][0].to(
                    extension_tensor.device
                )
        return optimizable_tensors

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_beizer_curves(self, mask):
        valid_beizer_mask = mask
        optimizable_tensors = self._prune_optimizer(valid_beizer_mask)
        self._control_points = optimizable_tensors["control_points"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._cholesky = optimizable_tensors["cholesky"]
        self._scaling = optimizable_tensors["scaling"]
        self._opacity = optimizable_tensors["opacity"]
        self._depth = optimizable_tensors["depth"]
        xyz_cached = getattr(self, "xyz", None)
        if isinstance(xyz_cached, torch.Tensor) and xyz_cached.ndim == 3:
            num_curves = int(mask.shape[0])
            if int(xyz_cached.shape[0]) == num_curves:
                self.xyz = xyz_cached[mask].contiguous()
        self.num_curves = int(self._control_points.shape[0])

    def densification_postfix(
        self,
        new_control_points,
        new_features,
        new_cholesky,
        new_depth,
        new_opacities,
        new_scaling,
    ):
        d = {
            "control_points": new_control_points,
            "features_dc": new_features,
            "cholesky": new_cholesky,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "depth": new_depth,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._control_points = optimizable_tensors["control_points"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._cholesky = optimizable_tensors["cholesky"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._depth = optimizable_tensors["depth"]
        self.num_curves = int(self._control_points.shape[0])

    def modified_control_points(self, bezier_curves):
        n, m, _ = bezier_curves.shape
        p0 = bezier_curves[:, 0, :].unsqueeze(1)
        p_last = bezier_curves[:, -1, :].unsqueeze(1)

        t = torch.linspace(0, 1, steps=m, device=bezier_curves.device).view(1, m, 1)
        new_bezier_curves = p0 + t * (p_last - p0)

        return new_bezier_curves

    def split_condition_1(self, num, n=2):
        opacities = torch.sigmoid(self._opacity)
        start = opacities[:, 0]
        end = opacities[:, -1]
        middle1 = opacities[:, 0:-1].mean(dim=-1)

        candidate1 = torch.abs(torch.max(start, end) - middle1)
        candidate2 = torch.abs(middle1 - torch.min(start, end))
        max_gap = torch.maximum(candidate1, candidate2)

        diffs = torch.abs(start - end)
        combined = torch.maximum(diffs, max_gap)
        topk_values, topk_indices = torch.topk(combined, k=num)
        selected_pts_mask = torch.zeros_like(diffs, dtype=torch.bool)
        selected_pts_mask.view(-1)[topk_indices] = True
        xyz = self.xyz.view(-1, self.total_num_sample, 2)
        if selected_pts_mask.sum() > 0:
            new_features = self._features_dc[selected_pts_mask].repeat(n, 1)
            new_cholesky = self._cholesky[selected_pts_mask].repeat(n, 1) / 2
            new_control_points = self._control_points[selected_pts_mask].clone()
            new_control_points_split = self._control_points[selected_pts_mask].clone()
            new_depth = self._depth[selected_pts_mask].repeat(n, 1)
            new_opacity = self._opacity[selected_pts_mask].repeat(n, 1)
            new_scaling = self._scaling[selected_pts_mask]
            new_scaling[:, 0] = new_scaling[:, 0] / 2
            new_scaling = new_scaling.repeat(n, 1)

            filtered_opacity = self._opacity[selected_pts_mask]
            k = filtered_opacity.shape[0]

            new_opacity[0:k, 0:2] = filtered_opacity[:, 0].unsqueeze(1).repeat(k, 2)
            new_opacity[0:k, 2:] = filtered_opacity[:, 1].unsqueeze(1)
            new_opacity[k:, 0:1] = filtered_opacity[:, 1].unsqueeze(1)
            new_opacity[k:, 1:] = filtered_opacity[:, 2].unsqueeze(1).repeat(k, 2)

            xyz = xyz[selected_pts_mask]
            new_control_points[:, -1, :] = xyz[:, int(self.num_samples / 2) - 1, :]
            new_control_points_split[:, 0, :] = xyz[:, int(self.num_samples / 2) + 1, :]
            new_control_points = self.modified_control_points(new_control_points)
            new_control_points_split = self.modified_control_points(
                new_control_points_split
            )
            new_control_points = torch.cat(
                (new_control_points, new_control_points_split), dim=0
            )
            self.prune_beizer_curves(~selected_pts_mask)
            self.densification_postfix(
                new_control_points,
                new_features,
                new_cholesky,
                new_depth,
                new_opacity,
                new_scaling,
            )

    def densify(self, num, pos_init_method, gt_image, radii=0.02):
        centers = torch.tensor(
            [pos_init_method() for _ in range(num)], dtype=torch.float32
        ).to(self._control_points.device)
        centers_rounded = centers.round().long()
        centers_rounded[:, 0] = torch.clamp(centers_rounded[:, 0], 0, self.H - 1)
        centers_rounded[:, 1] = torch.clamp(centers_rounded[:, 1], 0, self.W - 1)
        gt_values = (
            gt_image[:, :, centers_rounded[:, 0], centers_rounded[:, 1]].squeeze(0).T
        )
        new_control_points = self._initialize_control_points_with_center(
            centers.unsqueeze(1), radii
        ).to(self._control_points.device)
        new_cholesky = nn.Parameter(torch.rand(centers.shape[0], 3)).to(
            self._control_points.device
        )
        logits = torch.logit(gt_values.clamp(1e-6, 1 - 1e-6))
        new_features = nn.Parameter(logits.to(self._control_points.device))
        new_depth = nn.Parameter(torch.zeros(centers.shape[0], 1)).to(
            self._control_points.device
        )
        new_scaling = nn.Parameter(torch.rand(centers.shape[0], 1)).to(
            self._control_points.device
        )
        new_opacity = nn.Parameter(
            torch.ones(centers.shape[0], self._opacity.shape[1])
        ).to(self._control_points.device)
        self.densification_postfix(
            new_control_points,
            new_features,
            new_cholesky,
            new_depth,
            new_opacity,
            new_scaling,
        )

    def densify_and_split(self, num, grad_threshold=2e-6, n=2):
        opacities = torch.sigmoid(self._opacity)
        start_end = torch.min(opacities[:, 0], opacities[:, 3])
        middle = torch.min(opacities[:, 1], opacities[:, 2])

        grad = torch.abs(self._features_dc.grad.sum(-1)) / (opacities.sum(-1) / 4)

        diffs_abs = torch.abs(start_end - middle)
        score = diffs_abs * grad
        score_flat = score.view(-1)

        topk_values, topk_indices = torch.topk(score_flat, k=num)
        selected_pts_mask = torch.zeros_like(score, dtype=torch.bool)
        selected_pts_mask.view(-1)[topk_indices] = True
        xyz = self.xyz.view(-1, self.num_samples, 2)
        if selected_pts_mask.sum() > 0:
            new_features = self._features_dc[selected_pts_mask].repeat(n, 1)
            new_cholesky = self._cholesky[selected_pts_mask].repeat(n, 1) / 2
            new_control_points = self._control_points[selected_pts_mask].clone()
            new_control_points_split = self._control_points[selected_pts_mask].clone()
            new_depth = self._depth[selected_pts_mask].repeat(n, 1)
            new_opacity = self._opacity[selected_pts_mask].repeat(n, 1)
            new_scaling = self._scaling[selected_pts_mask]
            new_scaling[:, 1] = new_scaling[:, 1] / 2
            new_scaling = new_scaling.repeat(n, 1)

            filtered_opacity = self._opacity[selected_pts_mask]
            k = filtered_opacity.shape[0]

            new_opacity[0:k, 0:2] = filtered_opacity[:, 0].unsqueeze(1).repeat(k, 2)
            new_opacity[0:k, 2:] = filtered_opacity[:, 1].unsqueeze(1).repeat(k, 2)
            new_opacity[k:, 0:2] = filtered_opacity[:, 2].unsqueeze(1).repeat(k, 2)
            new_opacity[k:, 2:] = filtered_opacity[:, 3].unsqueeze(1).repeat(k, 2)

            xyz = xyz[selected_pts_mask]
            new_control_points[:, -1, :] = xyz[:, int(self.num_samples / 2) - 1, :]
            new_control_points_split[:, 0, :] = xyz[:, int(self.num_samples / 2) + 1, :]
            new_control_points = self.modified_control_points(new_control_points)
            new_control_points_split = self.modified_control_points(
                new_control_points_split
            )
            new_control_points = torch.cat(
                (new_control_points, new_control_points_split), dim=0
            )
            self.prune_beizer_curves(~selected_pts_mask)
            self.densification_postfix(
                new_control_points,
                new_features,
                new_cholesky,
                new_depth,
                new_opacity,
                new_scaling,
            )

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(
                torch.sigmoid(self._opacity),
                torch.ones_like(torch.sigmoid(self._opacity)) * 0.01,
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


__all__ = ["GaussianTraceOptimizerStateMixin"]
