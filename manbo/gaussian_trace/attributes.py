from __future__ import annotations

import math

import torch

from .utils import compute_aabb_xyxy, compute_outside_area_xyxy
from .common import FeatureAreaModulator


class GaussianTraceAttributesMixin:
    def compute_valid_aabb(self, points):
        mask = (points >= -1) & (points <= 1)
        inf_val = torch.tensor(float("inf"), device=points.device, dtype=points.dtype)
        ninf_val = torch.tensor(float("-inf"), device=points.device, dtype=points.dtype)

        points_for_min = torch.where(mask, points, inf_val)
        bbox_min = points_for_min.min(dim=1).values

        points_for_max = torch.where(mask, points, ninf_val)
        bbox_max = points_for_max.max(dim=1).values

        bbox_min = torch.where(
            torch.isinf(bbox_min), torch.full_like(bbox_min, -1.0), bbox_min
        )
        bbox_max = torch.where(
            torch.isinf(bbox_max), torch.full_like(bbox_max, -1.0), bbox_max
        )

        return bbox_min.unsqueeze(1), bbox_max.unsqueeze(1)

    def get_scaling(self, factor=1):
        if self.mode == "closed":
            return self.get_scaling_closed(factor)
        return self.get_scaling_open()

    def get_scaling_closed(self, factor):
        xyz = torch.cat([self.xyz, self.xyz_area], dim=1).detach()
        n = xyz.shape[1]
        diffs = torch.abs(xyz[:, :, 1:, :] - xyz[:, :, :-1, :])
        scale = torch.tensor(
            [self.W * factor, self.H * factor], device=diffs.device
        ).view(1, 1, 1, 2)
        diffs = diffs * scale

        sigma = torch.norm(diffs, dim=-1)
        sigma_last = sigma[:, :, -2:-1].clone()
        sigma_x = torch.cat([sigma, sigma_last], dim=-1) / (
            3.0 / torch.sqrt(torch.tensor(factor, dtype=torch.float32))
        )
        scale = torch.tensor([0.4, 0.9, 1.0], device=sigma_x.device).view(1, 1, 3)
        sigma_x[:, :, :3] *= scale
        sigma_x[:, :, -3:] *= scale.flip(dims=[2])
        sigma_x[:, :2, :].clamp_(min=0.3)

        index_order = torch.arange(2, n, device=xyz.device)
        index_order = torch.cat(
            [
                torch.tensor([0], device=xyz.device),
                index_order,
                torch.tensor([1], device=xyz.device),
            ]
        )
        xyz_reordered = xyz[:, index_order, :, :].clone()

        diffs_y = torch.abs(xyz_reordered[:, 1:, :, :] - xyz_reordered[:, :-1, :, :])
        diffs_y[:, :, :, 0] *= self.W * factor
        diffs_y[:, :, :, 1] *= self.H * factor
        sigma_ = torch.norm(diffs_y, dim=-1)
        sigma_first = sigma_[:, :1, :].clone()
        sigma_y = torch.cat([sigma_first, sigma_], dim=1) / (
            3.0 / torch.sqrt(torch.tensor(factor, dtype=torch.float32))
        )

        sigma_y[:, :2, :].clamp_(max=1.0, min=0.75)

        threshold = 0.1
        ratio = 3.0

        sx = sigma_x.clone()
        sy = sigma_y.clone()
        mask = sy < threshold
        mx = mask[:, 2:, :]
        my = mask[:, 2:, :]
        sigma_x[:, 2:, :] = torch.where(
            mx, torch.min(sx[:, 2:, :], sy[:, 2:, :] * ratio), sx[:, 2:, :]
        )
        sigma_y[:, 2:, :] = torch.where(
            my, torch.min(sy[:, 2:, :], sx[:, 2:, :] * ratio), sy[:, 2:, :]
        )
        scaling = torch.cat(
            [sigma_x.unsqueeze(-1), sigma_y.unsqueeze(-1)], dim=-1
        ).contiguous()
        return scaling.view(-1, 2).detach()

    def get_scaling_open(self):
        xyz = self.xyz.view(
            self._control_points.shape[0], self.total_num_sample, 2
        ).detach()
        diffs = torch.abs(xyz[:, 1:, :] - xyz[:, :-1, :])
        diffs[:, :, 0] *= self.W
        diffs[:, :, 1] *= self.H

        sigma_ratio = 2
        sigma = torch.norm(diffs, dim=2)
        sigma_last = sigma[:, -1:].clone()
        sigma_x = (torch.cat([sigma, sigma_last], dim=1)) / sigma_ratio + 0.5
        sigma_y = torch.abs(
            self._scaling.repeat_interleave(self.total_num_sample, dim=1) + 0.5
        )
        scaling = torch.cat([sigma_x.unsqueeze(-1), sigma_y.unsqueeze(-1)], dim=-1)
        return scaling.view(-1, 2)

    @property
    def get_rotation(self):
        return (
            self.rotation_activation(
                self._rotation.repeat_interleave(self.num_samples, dim=0)
            )
            * 2
            * math.pi
        )

    @property
    def get_features(self):
        if self.mode == "closed":
            return self.get_features_closed()
        return self.get_features_open()

    def get_area_weight(self, shape, device, alpha=4.0):
        _, b_area, h, d = shape
        yy = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1)
        xx = torch.linspace(-1, 1, b_area, device=device).view(1, b_area, 1, 1)

        dist = torch.abs(yy) + torch.abs(xx)
        weight = 1.0 - torch.exp(-alpha * dist)
        return weight.repeat(shape[0], 1, 1, d)

    def init_area_weight(self, device, alpha=4.0):
        h = int(self.total_num_sample / 2)
        b_area = self.curves_resolution
        d = self._features_dc.shape[-1]
        yy = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1)
        xx = torch.linspace(-1, 1, b_area, device=device).view(1, b_area, 1, 1)
        dist = torch.abs(yy) + torch.abs(xx)
        weight = 1.0 - torch.exp(-alpha * dist)
        self.area_weight = weight.repeat(self.num_curves, 1, 1, d)

    def get_features_closed(self):
        features_dc = (
            self._features_dc.unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, self.xyz.shape[1], self.xyz.shape[2], 1)
        )
        features_dc_area = (
            self._features_dc.unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, self.xyz_area.shape[1], self.xyz.shape[2], 1)
        )
        area_weight = self.get_area_weight(
            features_dc_area.shape, features_dc_area.device
        )
        features_dc_area = FeatureAreaModulator.apply(features_dc_area, area_weight)
        features = torch.cat([features_dc, features_dc_area], dim=1)
        _features_dc_expanded = torch.sigmoid(features)
        return _features_dc_expanded

    def get_features_open(self):
        _features_dc_expanded = torch.clamp(
            self._features_dc.unsqueeze(1).expand(-1, self.total_num_sample, -1),
            min=0.0,
            max=1.0,
        )
        return _features_dc_expanded.view(-1, 3)

    @property
    def get_depth(self):
        if self.mode == "closed":
            depth = self._depth.unsqueeze(2).repeat(
                1, self.xyz.shape[1] + self.xyz_area.shape[1], self.xyz.shape[2]
            )
            depth_clone = depth.clone().detach()
            depth_clone[:, :2, :] -= 1e-6
            return torch.sigmoid(depth_clone)
        return torch.sigmoid(
            self._depth.repeat_interleave(self.total_num_sample, dim=0)
        )

    def compute_rotations(self, points):
        if self.mode == "closed":
            xyz = torch.cat([self.xyz, self.xyz_area], dim=1).detach()
        else:
            xyz = points.detach().view(
                self._control_points.shape[0], self.num_beziers, -1, 2
            )

        diffs = xyz[:, :, 2:, :] - xyz[:, :, :-2, :]
        diffs[:, :, :, 0] *= self.W
        diffs[:, :, :, 1] *= self.H

        theta = torch.atan2(diffs[..., 1], diffs[..., 0])
        theta_first = theta[..., :1].clone()
        theta_last = theta[..., -1:].clone()
        rotations = torch.cat([theta_first, theta, theta_last], dim=-1)
        return -rotations

    @property
    def get_opacity(self):
        if self.mode == "closed":
            if self.opacity_mode == 1:
                n = self._opacity.shape[0]
                l = self.xyz_area.shape[1]
                m = self.xyz.shape[2]

                opacities_first = self._opacity[:, :1]
                opacities_middle = self._opacity[:, 1:2]
                opacities_last = self._opacity[:, 2:]

                weights_first = torch.linspace(
                    0, 1, steps=l // 2, device=self._opacity.device
                ).view(1, -1)
                weights_second = torch.linspace(
                    0, 1, steps=l - l // 2, device=self._opacity.device
                ).view(1, -1)

                opacities_area_first_half = (
                    1 - weights_first
                ) * opacities_first + weights_first * opacities_middle
                opacities_area_second_half = (
                    1 - weights_second
                ) * opacities_middle + weights_second * opacities_last

                opacities_area = torch.cat(
                    [opacities_area_first_half, opacities_area_second_half], dim=1
                )
                opacities_area = opacities_area.unsqueeze(-1).repeat(1, 1, m)
                opacity = torch.cat(
                    [
                        opacities_first.unsqueeze(1).repeat(1, 1, m),
                        opacities_last.unsqueeze(1).repeat(1, 1, m),
                        opacities_area,
                    ],
                    dim=1,
                )
                return self.opacity_activation(opacity.contiguous().view(-1, 1))

            opacities = self._opacity.unsqueeze(1).repeat(
                1, self.xyz.shape[1] + self.xyz_area.shape[1], self.xyz.shape[2]
            )
            return self.opacity_activation(opacities.contiguous().view(-1, 1))

        if self.opacity_mode == 1:
            n, cols = self._opacity.shape
            base_rep = self.total_num_sample // 3
            remainder = self.num_samples % 3
            parts = []
            for i in range(3):
                rep = base_rep + (remainder if i == 3 else 0)
                part = self._opacity[:, i].unsqueeze(1).repeat(1, rep)
                parts.append(part)
                out = torch.cat(parts, dim=1)
            return self.opacity_activation(out.contiguous().view(-1, 1))

        opacities = self._opacity.repeat(1, self.total_num_sample)
        return self.opacity_activation(opacities.contiguous().view(-1, 1))

    @property
    def get_cholesky_elements(self):
        _cholesky_expanded = self._cholesky.unsqueeze(1).repeat(1, self.num_samples, 1)
        return _cholesky_expanded.view(-1, 3) + self.cholesky_bound

    def compute_aabb(self, points_tensor):
        return compute_aabb_xyxy(points_tensor)

    def compute_aabb_area(self, boxes):
        if boxes.ndim != 2 or boxes.size(1) != 4:
            raise ValueError("boxes must have shape (N, 4).")

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        return areas

    def replace_points(self, points_tensor, target=(0, 0)):
        if points_tensor.ndim != 3 or points_tensor.size(-1) != 2:
            raise ValueError("points_tensor must have shape (N, H, 2).")

        target_tensor = torch.tensor(
            target, device=points_tensor.device, dtype=points_tensor.dtype
        )
        mask = (points_tensor < target_tensor).all(dim=-1)

        for n in range(points_tensor.size(0)):
            valid_points = points_tensor[n, ~mask[n]]
            if valid_points.size(0) == 0:
                return points_tensor
            replacement_value = valid_points[0]
            points_tensor[n, mask[n]] = replacement_value

        return points_tensor

    def compute_outside_area(self, boxes):
        return compute_outside_area_xyxy(boxes, img_h=self.H, img_w=self.W)

    def compute_pairwise_overlap(self, curve_1, curve_2, threshold=10):
        distances = torch.cdist(curve_1, curve_2, p=2)
        min_distances, _ = torch.min(distances, dim=1)
        min_distances, _ = torch.min(distances, dim=1)
        match_mask = min_distances < threshold
        match_count = int(torch.sum(match_mask).item())
        matched_distances = min_distances[match_mask]
        return match_count, matched_distances


__all__ = ["GaussianTraceAttributesMixin"]
