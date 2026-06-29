from __future__ import annotations

import math

import torch
import torch.distributions as dist


class GaussianTraceSamplingMixin:
    @staticmethod
    def _bernstein_device_key(device: torch.device) -> str:
        return device.type

    def _update_bernstein_cache(self, n: int, num_samples: int, device: torch.device):
        key = (n, num_samples, self._bernstein_device_key(device))
        if key in self._bernstein_cache:
            return

        t = torch.linspace(0.007, 0.993, num_samples, device=device)
        comb = torch.tensor(
            [math.comb(n, i) for i in range(n + 1)],
            dtype=torch.float32,
            device=device,
        )
        comb_deriv = torch.tensor(
            [math.comb(n - 1, i) for i in range(n)],
            dtype=torch.float32,
            device=device,
        )
        t_pow = t[:, None] ** torch.arange(n + 1, dtype=torch.float32, device=device)
        one_minus_t_pow = (1 - t[:, None]) ** torch.arange(
            n,
            -1,
            -1,
            dtype=torch.float32,
            device=device,
        )

        t_pow_deriv = t[:, None] ** torch.arange(n, dtype=torch.float32, device=device)
        one_minus_t_pow_deriv = (1 - t[:, None]) ** torch.arange(
            n - 1,
            -1,
            -1,
            dtype=torch.float32,
            device=device,
        )
        bernstein = comb * one_minus_t_pow * t_pow
        bernstein_deriv = comb_deriv * one_minus_t_pow_deriv * t_pow_deriv

        self._bernstein_cache[key] = {
            "bernstein": bernstein,
            "bernstein_deriv": bernstein_deriv,
        }

    def _initialize_control_points(self):
        num_segments = self.num_beziers
        num_points_per_curve = num_segments * (self.bezier_degree + 1)

        angles = torch.linspace(
            0, 2 * torch.pi, num_points_per_curve, device=self.device
        )
        angles = angles.unsqueeze(0).expand(self.num_curves, -1)

        radii = (
            torch.rand(self.num_curves, num_points_per_curve, device=self.device) * 0.5
            + 0.5
        ) * self.radius
        x_center = (torch.rand(self.num_curves, 1, 2, device=self.device) - 0.5) * 2
        x = x_center[:, :, 0] + radii * torch.cos(angles)
        y = x_center[:, :, 1] + radii * torch.sin(angles)

        points = torch.stack([x, y], dim=-1)
        perturbation = torch.randn_like(points) * (self.radius * 0.05)
        points = points + perturbation
        points[:, -1] = points[:, 0]
        control_points = points
        return torch.nn.Parameter(control_points)

    def _initialize_control_points_line(self, order_beizer=2):
        p0 = (torch.rand(self.num_curves, 1, 2, device=self.device) - 0.5) * 2
        offsets = (
            torch.rand(self.num_curves, order_beizer + 1, 2, device=self.device) * 0.5
            + 0.5
        ) * self.radius
        relative_points = torch.cumsum(offsets, dim=1)
        control_points = torch.cat([p0, p0 + relative_points], dim=1)
        return torch.nn.Parameter(control_points)

    def _initialize_control_points_with_center(self, centers, radii=0.02):
        x_centers = centers.clone()
        x_centers[:, :, 1] = (centers[:, :, 0] / self.H - 0.5) * 2
        x_centers[:, :, 0] = (centers[:, :, 1] / self.W - 0.5) * 2
        num_segments = self.num_beziers
        if self.mode == "unclosed":
            num_points_per_curve = num_segments * self.bezier_degree + 1
            num_offsets = num_points_per_curve - 1
            p0 = x_centers
            num_left = num_offsets // 2
            num_right = num_offsets - num_left
            offsets_left = (
                torch.rand(x_centers.shape[0], num_left, 2, device=self.device) * 0.5
                + 0.5
            ) * 0.005
            offsets_right = (
                torch.rand(x_centers.shape[0], num_right, 2, device=self.device) * 0.5
                + 0.5
            ) * 0.005
            relative_left = -torch.cumsum(offsets_left, dim=1)
            relative_right = torch.cumsum(offsets_right, dim=1)
            points_left = p0 + relative_left.flip(dims=[1])
            points_right = p0 + relative_right
            control_points = torch.cat([points_left, p0, points_right], dim=1)
            return torch.nn.Parameter(control_points)

        num_points_per_curve = num_segments * (self.bezier_degree + 1)
        num_curves = centers.shape[0]
        angles = torch.linspace(
            0, 2 * torch.pi, num_points_per_curve, device=self.device
        )
        angles = angles.unsqueeze(0).expand(num_curves, -1)
        x = x_centers[:, :, 0] + radii * torch.cos(angles)
        y = x_centers[:, :, 1] + radii * torch.sin(angles)
        points = torch.stack([x, y], dim=-1)
        points[:, -1] = points[:, 0]
        control_points = points
        return torch.nn.Parameter(control_points)

    def get_xyz_and_depth(self, factor=1, denser_sample=False):
        if self.mode == "line":
            xyz, normals, tangents = self.sample_bezier_curves(
                self._control_points, self.num_samples * factor
            )
        elif self.mode == "unclosed":
            xyz = self.sample_bezier_curves_unclose(
                self._control_points, self.total_num_sample * factor
            )
            return xyz.reshape(-1, 2), torch.zeros(1)
        else:
            if denser_sample:
                sampled_points, area_points = self.sample_bezier_area(
                    self._control_points,
                    resolution=self.curves_resolution * factor,
                    factor=factor,
                )
            else:
                sampled_points, area_points = self.sample_bezier_area(
                    self._control_points,
                    resolution=self.curves_resolution,
                )
            return sampled_points, area_points
        return xyz

    @property
    def get_xyz(self):
        xyz, normals, tangents = self.sample_bezier_curves(
            self._control_points, self.num_samples
        )
        return xyz.view(-1, 2)

    @property
    def get_samples(self):
        xyz = self.sample_bezier_curves_uniform(self._control_points, self.num_samples)
        return xyz

    def bezier_interpolate(self, input_tensor, num_samples):
        t = torch.linspace(0, 1, num_samples, device=input_tensor.device).unsqueeze(1)
        one_minus_t = 1 - t

        b0 = one_minus_t**3
        b1 = 3 * t * (one_minus_t**2)
        b2 = 3 * (t**2) * one_minus_t
        b3 = t**3

        weights = torch.cat([b0, b1, b2, b3], dim=1)
        output = torch.matmul(input_tensor, weights.t())
        return output

    @property
    def get_beizer_curves(self):
        return self._control_points

    def sample_bezier_curves_uniform(
        self, bezier_curves: torch.Tensor, num_samples: int
    ):
        num_curves, num_control_points, dim = bezier_curves.shape
        if dim != 2:
            raise ValueError("Control points must be 2D coordinates.")

        device = bezier_curves.device
        n = num_control_points - 1
        key = (n, num_samples, self._bernstein_device_key(device))
        cache = self._bernstein_cache[key]

        bernstein = cache["bernstein"][None, :, :]
        sampled_points = torch.sum(
            bernstein[..., None] * bezier_curves[:, None, :, :], dim=2
        )
        return sampled_points

    def sample_bezier_curves_unclose(
        self, control_points, num_samples, fine_samples=1000
    ):
        num_curves, total_control_points, _ = control_points.shape
        assert total_control_points == self.num_beziers * self.bezier_degree + 1, (
            f"Expected {self.num_beziers * self.bezier_degree + 1} control points, got {total_control_points}"
        )
        device = control_points.device
        samples_per_segment = int(num_samples / self.num_beziers)

        base = (
            torch.arange(self.num_beziers, device=device).unsqueeze(1)
            * self.bezier_degree
        )
        offsets = torch.arange(self.bezier_degree + 1, device=device).unsqueeze(0)
        indices = (base + offsets) % total_control_points
        indices = indices.unsqueeze(0).expand(num_curves, -1, -1)

        control_points_exp = control_points.unsqueeze(1).expand(
            -1, self.num_beziers, -1, -1
        )
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        segment_control_points = torch.gather(control_points_exp, 2, indices_exp)

        merged_control_points = segment_control_points.reshape(
            -1, self.bezier_degree + 1, 2
        )
        sampled_points = self.sample_bezier_curves_uniform(
            merged_control_points, samples_per_segment
        )
        return sampled_points

    def sample_bezier_curves(self, bezier_curves, num_samples, fine_samples=1000):
        num_curves, num_control_points, dim = bezier_curves.shape
        if dim != 2:
            raise ValueError("控制点必须是 2D 坐标。")

        device = bezier_curves.device
        n = num_control_points - 1

        t_values_fine = torch.linspace(0, 1, fine_samples, device=device)
        comb = torch.tensor(
            [math.comb(n, i) for i in range(n + 1)], dtype=torch.float32, device=device
        )
        t_powers = t_values_fine[:, None] ** torch.arange(
            n + 1, dtype=torch.float32, device=device
        )
        one_minus_t_powers = (1 - t_values_fine[:, None]) ** torch.arange(
            n,
            -1,
            -1,
            dtype=torch.float32,
            device=device,
        )
        bernstein = comb * one_minus_t_powers * t_powers
        bernstein = bernstein.unsqueeze(0)
        fine_points = torch.sum(
            bernstein[..., None] * bezier_curves[:, None, :, :], dim=2
        )

        deltas = torch.norm(fine_points[:, 1:, :] - fine_points[:, :-1, :], dim=-1)
        arc_lengths = torch.cat(
            [torch.zeros(num_curves, 1, device=device), deltas.cumsum(dim=-1)], dim=-1
        )
        total_lengths = arc_lengths[:, -1:]
        normalized_lengths = arc_lengths / total_lengths

        target_lengths = torch.linspace(0, 1, num_samples, device=device)
        target_lengths = target_lengths.unsqueeze(0).expand(
            normalized_lengths.size(0), -1
        )
        indices = torch.searchsorted(normalized_lengths, target_lengths)

        indices = torch.clamp(indices, 1, fine_samples - 1)
        low_indices = indices - 1
        high_indices = indices
        low_lengths = torch.gather(normalized_lengths, 1, low_indices)
        high_lengths = torch.gather(normalized_lengths, 1, high_indices)

        low_t = t_values_fine[low_indices]
        high_t = t_values_fine[high_indices]

        high_low_diff = high_lengths - low_lengths + 1e-8
        t_values_uniform = low_t + (target_lengths - low_lengths) / high_low_diff * (
            high_t - low_t
        )

        t_powers_uniform = t_values_uniform[:, :, None] ** torch.arange(
            n + 1,
            dtype=torch.float32,
            device=device,
        )
        one_minus_t_powers_uniform = (1 - t_values_uniform[:, :, None]) ** torch.arange(
            n,
            -1,
            -1,
            dtype=torch.float32,
            device=device,
        )
        bernstein_uniform = comb * one_minus_t_powers_uniform * t_powers_uniform
        sampled_points = torch.sum(
            bernstein_uniform[..., None] * bezier_curves[:, None, :, :], dim=2
        )

        bezier_derivative = n * (bezier_curves[:, 1:, :] - bezier_curves[:, :-1, :])
        comb_derivative = torch.tensor(
            [math.comb(n - 1, i) for i in range(n)], dtype=torch.float32, device=device
        )
        bernstein_derivative = (
            comb_derivative
            * one_minus_t_powers_uniform[:, :, :-1]
            * t_powers_uniform[:, :, 1:]
        )
        tangents = torch.sum(
            bernstein_derivative[..., None] * bezier_derivative[:, None, :, :], dim=2
        )
        normals = torch.stack([-tangents[..., 1], tangents[..., 0]], dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        return sampled_points, normals, tangents

    def split_bezier_segments(self, ctrl_pts):
        degree = self.bezier_degree + 1
        n, total_pts, _ = ctrl_pts.shape
        assert (total_pts - 1) % degree == 0, (
            "Control points must follow M*D + 1 pattern"
        )
        m = (total_pts - 1) // degree
        segments = [
            ctrl_pts[:, i * degree : i * degree + degree + 1, :].unsqueeze(1)
            for i in range(m)
        ]
        segments = torch.cat(segments, dim=1)
        segments = segments.contiguous().view(-1, degree + 1, 2)
        return segments

    def sample_bezier_closed_boundary(self, control_points, *, num_samples):
        n, total_pts, _ = control_points.shape
        if (total_pts - 2) % 2 != 0:
            raise ValueError("Control point count must be 2M+2 for closed mode")
        m = (total_pts - 2) // 2
        bezier1 = control_points[:, : m + 2, :]
        bezier2 = torch.cat(
            [control_points[:, m + 1 :, :], control_points[:, 0:1, :]], dim=1
        ).flip(dims=[1])

        bezier1_segments = self.split_bezier_segments(bezier1)
        bezier2_segments = self.split_bezier_segments(bezier2)
        boundary_beziers = torch.cat([bezier1_segments, bezier2_segments], dim=0)
        boundary = self.sample_bezier_curves_uniform(boundary_beziers, int(num_samples))

        points_per_boundary = int(self.num_beziers * int(num_samples) / 2)
        bezier1_samples, bezier2_samples = boundary.chunk(2, dim=0)
        sampled_boundary = torch.stack(
            [
                bezier1_samples.reshape(n, points_per_boundary, 2),
                bezier2_samples.reshape(n, points_per_boundary, 2),
            ],
            dim=1,
        )
        return sampled_boundary.contiguous()

    def sample_bezier_area(self, control_points, resolution=20, factor=1):
        n, total_pts, _ = control_points.shape
        num_samples = self.num_samples * factor
        assert (total_pts - 2) % 2 == 0, (
            "Control point count must be 2M+2 for degree M Bézier pairs"
        )
        m = (total_pts - 2) // 2
        bezier1 = control_points[:, : m + 2, :]
        bezier2 = torch.cat(
            [control_points[:, m + 1 :, :], control_points[:, 0:1, :]], dim=1
        ).flip(dims=[1])

        sampled_boundary = self.sample_bezier_closed_boundary(
            control_points, num_samples=int(num_samples)
        )

        bezier1 = bezier1.unsqueeze(1)
        bezier2 = bezier2.unsqueeze(1)
        t_vals = torch.linspace(-2, 2, resolution, device=control_points.device)
        t_vals = dist.Normal(0, 0.85).cdf(t_vals).view(1, resolution, 1, 1)
        interp_cp = (1 - t_vals) * bezier1 + t_vals * bezier2
        interp_cp_flat = interp_cp.view(-1, m + 2, 2)

        interp_segments = self.split_bezier_segments(interp_cp_flat)
        interp_samples = self.sample_bezier_curves_uniform(interp_segments, num_samples)
        return sampled_boundary, interp_samples.view(
            self._control_points.shape[0], resolution, -1, 2
        ).detach()


__all__ = ["GaussianTraceSamplingMixin"]
