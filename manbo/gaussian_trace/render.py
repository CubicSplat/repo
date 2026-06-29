from __future__ import annotations
import os
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from manbo import gs_render
from manbo.ops import (
    closed_cubic_fill_segments_tilelang,
    render_cubic_fill_splat_tilelang,
    render_cubic_polyline_splat_tilelang,
)
from manbo.profile import prof
from visual_debug import emit_visual_debug

_CUBIC_EQUIV_CORE_RATIO = 0.6
_CUBIC_EQUIV_EDGE_RATIO = 0.9
_CLOSED_FILL_SUBDIV = 4


class GaussianTraceRenderMixin:
    @staticmethod
    def _empty_tile_contrib_payload(
        *, num_tiles: int, num_curves: int, primitive_per_curve: int = 0
    ) -> dict[str, Any]:
        return {
            "tile_contrib_tile_ids": torch.empty((0,), dtype=torch.int32),
            "tile_contrib_curve_ids": torch.empty((0,), dtype=torch.int32),
            "tile_contrib_values": torch.empty((0,), dtype=torch.float32),
            "tile_contrib_nnz": 0,
            "tile_contrib_num_tiles": int(max(0, num_tiles)),
            "tile_contrib_num_curves": int(max(0, num_curves)),
            "tile_contrib_primitive_per_curve": int(max(0, primitive_per_curve)),
        }

    @staticmethod
    def _aggregate_sparse_curve_tile_contrib_from_isect(
        *,
        isect_contrib: torch.Tensor,
        primitive_ids_sorted: torch.Tensor,
        tile_bins: torch.Tensor,
        num_curves: int,
        num_primitives: int,
        min_value: float,
    ) -> dict[str, Any]:
        num_curves_i = int(max(0, num_curves))
        num_prims_i = int(max(0, num_primitives))
        if (
            not isinstance(isect_contrib, torch.Tensor)
            or not isinstance(primitive_ids_sorted, torch.Tensor)
            or not isinstance(tile_bins, torch.Tensor)
        ):
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=0,
                num_curves=num_curves_i,
            )

        if tile_bins.ndim != 2 or int(tile_bins.shape[1]) != 2:
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=0,
                num_curves=num_curves_i,
            )

        values = isect_contrib.detach().to(torch.float32).view(-1)
        prim_ids = primitive_ids_sorted.detach().to(torch.int64).view(-1)
        bins = tile_bins.detach().to(torch.int64).contiguous()
        num_tiles = int(bins.shape[0])
        if num_tiles <= 0 or int(values.numel()) <= 0 or int(prim_ids.numel()) <= 0:
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=num_tiles,
                num_curves=num_curves_i,
            )

        counts = (bins[:, 1] - bins[:, 0]).clamp(min=0)
        total_count = int(counts.sum().item())
        if total_count <= 0:
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=num_tiles,
                num_curves=num_curves_i,
            )

        tile_ids = torch.repeat_interleave(
            torch.arange(num_tiles, device=counts.device, dtype=torch.int64),
            counts,
        )
        m = min(int(values.numel()), int(prim_ids.numel()), int(tile_ids.numel()))
        if m <= 0:
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=num_tiles,
                num_curves=num_curves_i,
            )
        values = values[:m]
        prim_ids = prim_ids[:m]
        tile_ids = tile_ids[:m]

        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            min=0.0
        )
        valid = values > float(max(0.0, float(min_value)))
        if num_prims_i > 0:
            valid = valid & (prim_ids >= 0) & (prim_ids < num_prims_i)
        else:
            valid = valid & (prim_ids >= 0)
        if not bool(valid.any().item()):
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=num_tiles,
                num_curves=num_curves_i,
            )
        values = values[valid]
        prim_ids = prim_ids[valid]
        tile_ids = tile_ids[valid]

        primitive_per_curve = 0
        if num_curves_i <= 0:
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=num_tiles,
                num_curves=num_curves_i,
            )
        if num_prims_i == num_curves_i:
            curve_ids = prim_ids
            primitive_per_curve = 1
        elif num_prims_i > 0 and num_prims_i % num_curves_i == 0:
            primitive_per_curve = int(num_prims_i // num_curves_i)
            curve_ids = torch.div(prim_ids, primitive_per_curve, rounding_mode="floor")
        else:
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=num_tiles,
                num_curves=num_curves_i,
            )

        valid_curve = (curve_ids >= 0) & (curve_ids < num_curves_i)
        if not bool(valid_curve.any().item()):
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=num_tiles,
                num_curves=num_curves_i,
                primitive_per_curve=primitive_per_curve,
            )
        values = values[valid_curve]
        curve_ids = curve_ids[valid_curve]
        tile_ids = tile_ids[valid_curve]

        linear = tile_ids * int(num_curves_i) + curve_ids
        uniq_linear, inv = torch.unique(linear, sorted=False, return_inverse=True)
        agg = torch.zeros(
            (int(uniq_linear.numel()),),
            dtype=torch.float32,
            device=values.device,
        )
        agg.scatter_add_(0, inv, values)
        keep = agg > float(max(0.0, float(min_value)))
        if not bool(keep.any().item()):
            return GaussianTraceRenderMixin._empty_tile_contrib_payload(
                num_tiles=num_tiles,
                num_curves=num_curves_i,
                primitive_per_curve=primitive_per_curve,
            )

        uniq_linear = uniq_linear[keep]
        agg = agg[keep]
        tile_out = torch.div(uniq_linear, int(num_curves_i), rounding_mode="floor").to(
            torch.int32
        )
        curve_out = torch.remainder(uniq_linear, int(num_curves_i)).to(torch.int32)
        return {
            "tile_contrib_tile_ids": tile_out.to(device="cpu", dtype=torch.int32),
            "tile_contrib_curve_ids": curve_out.to(device="cpu", dtype=torch.int32),
            "tile_contrib_values": agg.to(device="cpu", dtype=torch.float32),
            "tile_contrib_nnz": int(agg.numel()),
            "tile_contrib_num_tiles": int(num_tiles),
            "tile_contrib_num_curves": int(num_curves_i),
            "tile_contrib_primitive_per_curve": int(max(0, primitive_per_curve)),
        }

    @staticmethod
    def _coords_are_normalized(points: torch.Tensor) -> bool:
        if int(points.numel()) <= 0:
            return True
        pts = torch.nan_to_num(
            points.detach().to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        x = pts[..., 0]
        y = pts[..., 1]
        x_min = float(x.min().item())
        x_max = float(x.max().item())
        y_min = float(y.min().item())
        y_max = float(y.max().item())
        return bool(x_min >= -1.5 and x_max <= 1.5 and y_min >= -1.5 and y_max <= 1.5)

    def _curve_points_to_pixels(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) != 2:
            raise ValueError(f"curve points must be [N,S,2], got {tuple(points.shape)}")
        pts = torch.nan_to_num(
            points.detach().to(torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        x = pts[..., 0]
        y = pts[..., 1]
        h_i = max(1, int(getattr(self, "H", 1)))
        w_i = max(1, int(getattr(self, "W", 1)))
        if self._coords_are_normalized(pts):
            px = (x + 1.0) * 0.5 * float(max(1, w_i - 1))
            py = (y + 1.0) * 0.5 * float(max(1, h_i - 1))
        else:
            px = x
            py = y
        px = px.clamp(0.0, float(max(0, w_i - 1)))
        py = py.clamp(0.0, float(max(0, h_i - 1)))
        return torch.stack([px, py], dim=-1)

    @staticmethod
    def _curve_bbox_area(points_px: torch.Tensor) -> torch.Tensor:
        if points_px.ndim != 3 or int(points_px.shape[-1]) != 2:
            raise ValueError(
                f"pixel curve points must be [N,S,2], got {tuple(points_px.shape)}"
            )
        mins = points_px.min(dim=1).values
        maxs = points_px.max(dim=1).values
        wh = (maxs - mins).clamp(min=0.0)
        return (wh[:, 0] * wh[:, 1]).to(torch.float32)

    def _curve_points_for_area(self, *, num_curves: int) -> Optional[torch.Tensor]:
        if int(num_curves) <= 0:
            return None

        xys = getattr(self, "xys", None)
        if (
            isinstance(xys, torch.Tensor)
            and xys.ndim == 2
            and int(xys.shape[1]) == 2
            and int(xys.shape[0]) >= int(num_curves)
            and int(xys.shape[0]) % int(num_curves) == 0
        ):
            return (
                xys.detach().to(torch.float32).contiguous().view(int(num_curves), -1, 2)
            )

        xyz = getattr(self, "xyz", None)
        xyz_area = getattr(self, "xyz_area", None)
        if isinstance(xyz, torch.Tensor):
            if (
                xyz.ndim == 4
                and int(xyz.shape[0]) == int(num_curves)
                and int(xyz.shape[-1]) == 2
            ):
                pts = xyz.detach().to(torch.float32)
                if (
                    isinstance(xyz_area, torch.Tensor)
                    and xyz_area.ndim == 4
                    and int(xyz_area.shape[0]) == int(num_curves)
                    and int(xyz_area.shape[-1]) == 2
                    and int(xyz_area.shape[2]) == int(xyz.shape[2])
                ):
                    pts = torch.cat([pts, xyz_area.detach().to(torch.float32)], dim=1)
                return pts.contiguous().view(int(num_curves), -1, 2)
            if (
                xyz.ndim == 3
                and int(xyz.shape[0]) == int(num_curves)
                and int(xyz.shape[-1]) == 2
            ):
                return (
                    xyz.detach()
                    .to(torch.float32)
                    .contiguous()
                    .view(int(num_curves), -1, 2)
                )

        cp = getattr(self, "_control_points", None)
        if (
            isinstance(cp, torch.Tensor)
            and cp.ndim == 3
            and int(cp.shape[0]) == int(num_curves)
            and int(cp.shape[-1]) == 2
        ):
            return (
                cp.detach().to(torch.float32).contiguous().view(int(num_curves), -1, 2)
            )

        return None

    def _estimate_curve_coverage_area(
        self, *, num_curves: int
    ) -> Optional[torch.Tensor]:
        pts = self._curve_points_for_area(num_curves=int(num_curves))
        if not isinstance(pts, torch.Tensor):
            return None
        if pts.ndim != 3 or int(pts.shape[0]) != int(num_curves):
            return None
        if int(pts.shape[1]) <= 0:
            return None
        try:
            pts_px = self._curve_points_to_pixels(pts)
        except Exception:
            return None

        mode_norm = str(getattr(self, "mode", "")).lower()
        if mode_norm in {"line", "unclosed"}:
            if int(pts_px.shape[1]) <= 1:
                length = torch.zeros(
                    (int(num_curves),), device=pts_px.device, dtype=torch.float32
                )
            else:
                length = torch.norm(pts_px[:, 1:, :] - pts_px[:, :-1, :], dim=-1).sum(
                    dim=1
                )
            scaling = getattr(self, "_scaling", None)
            if (
                isinstance(scaling, torch.Tensor)
                and scaling.ndim >= 1
                and int(scaling.shape[0]) == int(num_curves)
            ):
                sigma_curve = (
                    torch.abs(
                        scaling.detach()
                        .to(device=pts_px.device, dtype=torch.float32)
                        .view(int(num_curves), -1)[:, 0]
                    )
                    + 0.5
                )
                stroke_width = torch.clamp(
                    2.0 * _CUBIC_EQUIV_CORE_RATIO * sigma_curve,
                    min=0.25,
                )
            else:
                stroke_width = torch.ones(
                    (int(num_curves),), device=pts_px.device, dtype=torch.float32
                )
            area = length * stroke_width
        else:
            area = self._curve_bbox_area(pts_px)

        area = torch.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        return area.to(device=pts.device, dtype=torch.float32).contiguous()

    @staticmethod
    def _summarize_primitive_contrib(
        *,
        curve_contrib_rgb: torch.Tensor,
        topk: int,
        include_rate_tensor: bool = False,
        include_curve_contrib_rgb: bool = False,
        curve_coverage_area: Optional[torch.Tensor] = None,
        area_clamp_min: float = 9.0,
        area_clamp_max: float = 0.0,
    ) -> dict[str, Any]:
        if curve_contrib_rgb.ndim != 2 or int(curve_contrib_rgb.shape[1]) != 3:
            raise ValueError(
                f"curve_contrib_rgb must be [N,3], got {tuple(curve_contrib_rgb.shape)}"
            )
        contrib_abs = curve_contrib_rgb.detach().to(torch.float32).abs()
        score = contrib_abs.sum(dim=1)
        num_curves = int(score.numel())
        area_min = float(max(0.0, float(area_clamp_min)))
        area_max = float(area_clamp_max)
        if area_max > 0.0 and area_max < area_min:
            area_max = area_min
        eta = score
        area_used = False
        safe_area: Optional[torch.Tensor] = None
        if isinstance(curve_coverage_area, torch.Tensor):
            area = curve_coverage_area.detach().to(
                device=score.device, dtype=torch.float32
            )
            area = torch.nan_to_num(area.view(-1), nan=0.0, posinf=0.0, neginf=0.0)
            if int(area.numel()) == num_curves:
                area = area.clamp(min=0.0)
                if area_max > 0.0:
                    area = area.clamp(min=area_min, max=area_max)
                else:
                    area = area.clamp(min=area_min)
                eta = score / area.clamp(min=1e-12)
                safe_area = area
                area_used = True
        eta_sum = float(eta.sum().item())
        score_sum = float(score.sum().item())
        if eta_sum > 0.0:
            rate = eta / eta_sum
        else:
            rate = torch.zeros_like(score)

        if num_curves <= 0:
            out = {
                "num_curves": 0,
                "active_curves": 0,
                "active_ratio": 0.0,
                "curve_contrib_score_sum": 0.0,
                "curve_contrib_r_sum": 0.0,
                "curve_contrib_g_sum": 0.0,
                "curve_contrib_b_sum": 0.0,
                "topk": 0,
                "top1_rate": 0.0,
                "topk_rate_sum": 0.0,
                "effective_curves": 0.0,
                "entropy": 0.0,
                "rate_p50": 0.0,
                "rate_p90": 0.0,
                "rate_p99": 0.0,
                "rate_metric": "eta" if area_used else "ci",
                "eta_sum": 0.0,
                "curve_area_sum": 0.0,
                "curve_area_norm_enabled": bool(area_used),
                "curve_area_clamp_min": float(area_min),
                "curve_area_clamp_max": float(area_max),
                "topk_indices": torch.empty((0,), dtype=torch.int64),
                "topk_rates": torch.empty((0,), dtype=torch.float32),
            }
            if include_rate_tensor:
                out["rate_tensor"] = torch.empty((0,), dtype=torch.float32)
            if include_curve_contrib_rgb:
                out["curve_contrib_rgb_tensor"] = torch.empty(
                    (0, 3), dtype=torch.float32
                )
            return out
        active_curves = int((score > 0).sum().item())
        topk_eff = min(max(1, int(topk)), num_curves)
        topk_rate, topk_idx = torch.topk(rate, k=topk_eff, largest=True, sorted=True)
        sq_sum = float((rate * rate).sum().item())
        effective_curves = 0.0 if score_sum <= 0.0 else (1.0 / max(sq_sum, 1e-12))
        entropy = (
            0.0
            if score_sum <= 0.0
            else float(
                (-rate.clamp(min=1e-12) * torch.log(rate.clamp(min=1e-12))).sum().item()
            )
        )

        out = {
            "num_curves": num_curves,
            "active_curves": active_curves,
            "active_ratio": float(active_curves / max(1, num_curves)),
            "curve_contrib_score_sum": score_sum,
            "curve_contrib_r_sum": float(contrib_abs[:, 0].sum().item()),
            "curve_contrib_g_sum": float(contrib_abs[:, 1].sum().item()),
            "curve_contrib_b_sum": float(contrib_abs[:, 2].sum().item()),
            "topk": int(topk_eff),
            "top1_rate": float(topk_rate[0].item()) if topk_eff > 0 else 0.0,
            "topk_rate_sum": float(topk_rate.sum().item()) if topk_eff > 0 else 0.0,
            "effective_curves": float(effective_curves),
            "entropy": entropy,
            "rate_p50": float(torch.quantile(rate, 0.50).item()),
            "rate_p90": float(torch.quantile(rate, 0.90).item()),
            "rate_p99": float(torch.quantile(rate, 0.99).item()),
            "rate_metric": "eta" if area_used else "ci",
            "eta_sum": float(eta_sum),
            "curve_area_sum": float(safe_area.sum().item()) if area_used else 0.0,
            "curve_area_norm_enabled": bool(area_used),
            "curve_area_clamp_min": float(area_min),
            "curve_area_clamp_max": float(area_max),
            "topk_indices": topk_idx.detach().to(device="cpu", dtype=torch.int64),
            "topk_rates": topk_rate.detach().to(device="cpu", dtype=torch.float32),
        }
        if include_rate_tensor:
            out["rate_tensor"] = rate.detach().contiguous()
        if include_curve_contrib_rgb:
            out["curve_contrib_rgb_tensor"] = curve_contrib_rgb.detach().contiguous()
        return out

    @staticmethod
    def _cubic_polyline_mode_supported(mode: str, *, bezier_degree: int) -> bool:
        return str(mode).lower() == "unclosed" and int(bezier_degree) == 3

    @staticmethod
    def _cubic_fill_mode_supported(mode: str) -> bool:
        return str(mode).lower() == "closed"

    @staticmethod
    def _renderer_backend_name(value: object) -> str:
        return str(value if value is not None else "gaussian").strip().lower()

    def _cubic_polyline_backend_enabled(self) -> bool:
        backend = self._renderer_backend_name(getattr(self, "renderer_backend", None))
        if backend != "cubic":
            return False
        return self._cubic_polyline_mode_supported(
            str(self.mode),
            bezier_degree=int(self.bezier_degree),
        )

    def _cubic_fill_backend_enabled(self) -> bool:
        backend = self._renderer_backend_name(getattr(self, "renderer_backend", None))
        if backend != "cubic_fill":
            return False
        return self._cubic_fill_mode_supported(str(self.mode))

    @staticmethod
    def _cubic_unclosed_aniso_intersects_enabled() -> bool:
        return str(
            os.getenv("MANBO_CUBIC_UNCLOSED_ANISO_INTERSECTS", "1")
        ).strip().lower() not in {"0", "false", "off", "no"}

    def _cubic_segments_per_curve(self) -> int:
        mode_norm = str(self.mode).lower()
        if mode_norm == "line":
            return 1
        if mode_norm == "unclosed":
            return int(self.num_beziers)
        return 1

    def _repeat_curve_values_to_cubics(
        self,
        values: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        seg_n = max(1, int(num_segments))
        vals = values.to(device=self._control_points.device, dtype=torch.float32).view(
            -1
        )
        n_curves = int(self._control_points.shape[0])
        if int(vals.numel()) == n_curves * seg_n:
            return vals.contiguous()
        if int(vals.numel()) == n_curves:
            return vals.repeat_interleave(seg_n).contiguous()
        if int(vals.numel()) == 1:
            return vals.expand(n_curves * seg_n).contiguous()
        raise ValueError(
            f"expected values size 1/{n_curves}/{n_curves * seg_n}, got {int(vals.numel())}"
        )

    def _cubic_gaussian_equiv_curve_sigma(self) -> torch.Tensor:
        return torch.abs(self._scaling[:, 0]) + 0.5

    def _cubic_gaussian_equiv_stroke_per_curve(
        self, sigma_curve: torch.Tensor
    ) -> torch.Tensor:
        stroke = 2.0 * _CUBIC_EQUIV_CORE_RATIO * sigma_curve
        return torch.clamp(stroke, min=0.25)

    def _cubic_gaussian_equiv_aa_widths(
        self,
        sigma_curve: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        aa_curve = sigma_curve * _CUBIC_EQUIV_EDGE_RATIO
        aa_curve = torch.clamp(aa_curve, min=1e-4)
        return self._repeat_curve_values_to_cubics(
            aa_curve,
            num_segments=num_segments,
        )

    @staticmethod
    def _sample_cubic_debug_points(
        cubics: torch.Tensor, samples: int = 16
    ) -> torch.Tensor:
        if cubics.ndim != 3 or tuple(cubics.shape[1:]) != (4, 2):
            raise ValueError(f"cubics must be (N,4,2), got {tuple(cubics.shape)}")
        t = torch.linspace(
            0.0,
            1.0,
            max(2, int(samples)),
            device=cubics.device,
            dtype=cubics.dtype,
        ).view(1, -1, 1)
        omt = 1.0 - t
        pts = (
            cubics[:, 0:1, :] * (omt**3)
            + 3.0 * cubics[:, 1:2, :] * (omt**2) * t
            + 3.0 * cubics[:, 2:3, :] * omt * (t**2)
            + cubics[:, 3:4, :] * (t**3)
        )
        return pts.contiguous().view(-1, 2)

    @staticmethod
    def _cubic_segments_from_open_chain(
        control_points: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        n = int(control_points.shape[0])
        k = int(control_points.shape[1])
        expected_k = int(num_segments) * 3 + 1
        if k != expected_k:
            raise ValueError(
                f"open cubic chain expects K={expected_k}, got {k} (segments={num_segments})"
            )
        base = (
            torch.arange(
                int(num_segments), device=control_points.device, dtype=torch.int64
            ).unsqueeze(1)
            * 3
        )
        offs = torch.arange(
            4, device=control_points.device, dtype=torch.int64
        ).unsqueeze(0)
        idx = (base + offs).unsqueeze(0).expand(n, -1, -1)
        cp_exp = control_points.unsqueeze(1).expand(-1, int(num_segments), -1, -1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, 2)
        seg = torch.gather(cp_exp, dim=2, index=gather_idx)
        return seg.contiguous()

    def _cubic_render_inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mode_norm = str(self.mode).lower()
        if mode_norm != "line":
            raise ValueError(
                f"legacy cubic stroke backend supports only mode='line', got {self.mode!r}"
            )
        color_per_curve = torch.sigmoid(self._features_dc).clamp(0.0, 1.0)
        sigma_per_curve = self._cubic_gaussian_equiv_curve_sigma()
        stroke_per_curve = self._cubic_gaussian_equiv_stroke_per_curve(sigma_per_curve)
        depth_per_curve = torch.sigmoid(self._depth[:, 0])

        cubics = self._control_points
        if self._opacity.ndim == 2 and int(self._opacity.shape[1]) > 0:
            opacity = torch.sigmoid(self._opacity[:, 0])
        else:
            opacity = torch.sigmoid(self._opacity.view(-1))
        endpoint_caps = torch.ones(
            (int(cubics.shape[0]), 2),
            device=cubics.device,
            dtype=torch.int32,
        )
        aa_widths = self._cubic_gaussian_equiv_aa_widths(
            sigma_per_curve,
            num_segments=1,
        )
        return (
            cubics,
            color_per_curve,
            opacity,
            stroke_per_curve,
            depth_per_curve,
            endpoint_caps,
            aa_widths,
        )

    def _cubic_unclosed_polyline_render_inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if str(self.mode).lower() != "unclosed":
            raise ValueError(
                f"polyline cubic inputs support only mode='unclosed', got {self.mode!r}"
            )
        if int(self.bezier_degree) != 3:
            raise ValueError(
                f"unclosed cubic backend requires bezier_degree=3, got {self.bezier_degree}"
            )

        num_segments = int(self.num_beziers)
        seg = self._cubic_segments_from_open_chain(
            self._control_points,
            num_segments=num_segments,
        )
        cubics = seg.view(-1, 4, 2).contiguous()
        n_curves = int(self._control_points.shape[0])
        num_primitives = n_curves * num_segments
        # Keep per-segment attributes by mapping one segment to one primitive.
        seg_offsets = torch.arange(
            0,
            num_primitives + 1,
            1,
            dtype=torch.int32,
            device=self._control_points.device,
        )

        color_src = torch.sigmoid(self._features_dc).clamp(0.0, 1.0)
        if color_src.ndim != 2:
            raise ValueError(
                f"_features_dc must be 2D for cubic polyline backend, got {tuple(color_src.shape)}"
            )
        if int(color_src.shape[0]) != n_curves:
            raise ValueError(
                f"_features_dc first dim must match curve count ({n_curves}), got {int(color_src.shape[0])}"
            )
        if int(color_src.shape[1]) == 3:
            color_seg = color_src.unsqueeze(1).expand(-1, num_segments, -1)
        elif int(color_src.shape[1]) % 3 == 0:
            color_seg_src = color_src.view(n_curves, -1, 3)
            src_segments = int(color_seg_src.shape[1])
            if src_segments == num_segments:
                color_seg = color_seg_src
            elif src_segments == 1:
                color_seg = color_seg_src.expand(-1, num_segments, -1)
            elif src_segments > num_segments:
                color_seg = color_seg_src[:, :num_segments, :]
            else:
                color_seg = F.interpolate(
                    color_seg_src.permute(0, 2, 1),
                    size=num_segments,
                    mode="linear",
                    align_corners=True,
                ).permute(0, 2, 1)
        else:
            raise ValueError(
                "_features_dc second dim must be 3 or a multiple of 3 "
                f"for per-segment colors, got {int(color_src.shape[1])}"
            )
        color_per_segment = color_seg.contiguous().view(-1, 3)
        sigma_per_curve = self._cubic_gaussian_equiv_curve_sigma()
        stroke_per_curve = self._cubic_gaussian_equiv_stroke_per_curve(sigma_per_curve)
        depth_per_curve = torch.sigmoid(self._depth[:, 0])

        opacity_src = torch.sigmoid(self._opacity).view(n_curves, -1)
        if int(opacity_src.shape[1]) == num_segments:
            opacity_seg = opacity_src
        elif int(opacity_src.shape[1]) == 1:
            opacity_seg = opacity_src.expand(-1, num_segments)
        elif int(opacity_src.shape[1]) > num_segments:
            opacity_seg = opacity_src[:, :num_segments]
        else:
            opacity_seg = F.interpolate(
                opacity_src.unsqueeze(1),
                size=num_segments,
                mode="linear",
                align_corners=True,
            ).squeeze(1)
        opacity_per_segment = opacity_seg.contiguous().view(-1)

        stroke_per_segment = self._repeat_curve_values_to_cubics(
            stroke_per_curve, num_segments=num_segments
        )
        depth_per_segment = self._repeat_curve_values_to_cubics(
            depth_per_curve, num_segments=num_segments
        )
        aa_widths = torch.clamp(
            sigma_per_curve * _CUBIC_EQUIV_EDGE_RATIO,
            min=1e-4,
        )
        aa_widths = self._repeat_curve_values_to_cubics(
            aa_widths, num_segments=num_segments
        )
        return (
            cubics,
            seg_offsets.contiguous(),
            color_per_segment.contiguous(),
            opacity_per_segment.contiguous(),
            stroke_per_segment.contiguous(),
            depth_per_segment.contiguous(),
            aa_widths.contiguous(),
        )

    def _closed_cubic_fill_segments(self) -> tuple[torch.Tensor, torch.Tensor]:
        cp = self._control_points
        if cp.ndim != 3 or int(cp.shape[-1]) != 2:
            raise ValueError(
                f"closed control points must be [N,K,2], got {tuple(cp.shape)}"
            )
        n = int(cp.shape[0])
        k = int(cp.shape[1])
        if n <= 0:
            return (
                torch.empty((0, 4, 2), dtype=torch.float32, device=cp.device),
                torch.zeros((1,), dtype=torch.int32, device=cp.device),
            )
        if k < 4 or ((k - 2) % 2) != 0:
            raise ValueError(
                "closed cubic fill expects control point count K that satisfies K>=4 and (K-2)%2==0, "
                f"got K={k}"
            )

        return closed_cubic_fill_segments_tilelang(
            cp,
            subdiv=int(_CLOSED_FILL_SUBDIV),
        )

    def _cubic_fill_render_inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if str(self.mode).lower() != "closed":
            raise ValueError(
                f"cubic fill backend supports only closed mode, got mode={self.mode!r}"
            )

        ring_segments, ring_offsets = self._closed_cubic_fill_segments()
        color_per_curve = torch.sigmoid(self._features_dc)
        if self._opacity.ndim == 2 and int(self._opacity.shape[1]) > 0:
            opacity_per_curve = torch.sigmoid(self._opacity).mean(dim=1)
        else:
            opacity_per_curve = torch.sigmoid(self._opacity.view(-1))
        depth_per_curve = torch.sigmoid(self._depth[:, 0])
        sigma_per_curve = self._cubic_gaussian_equiv_curve_sigma()
        aa_widths = torch.clamp(
            sigma_per_curve * _CUBIC_EQUIV_EDGE_RATIO,
            min=1e-4,
        )
        fill_rules = torch.ones(
            (int(color_per_curve.shape[0]),),
            device=color_per_curve.device,
            dtype=torch.int32,
        )
        return (
            ring_segments,
            ring_offsets,
            color_per_curve,
            opacity_per_curve.view(-1),
            depth_per_curve.view(-1),
            fill_rules,
            aa_widths,
        )

    @staticmethod
    def _closed_geom_cull_enabled() -> bool:
        return str(
            os.getenv("MANBO_CLOSED_GEOM_BWD_CULL", "1")
        ).strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }

    def _closed_gaussian_geometry_meta(self) -> torch.Tensor:
        if self.mode != "closed":
            raise RuntimeError("closed geometry meta is only available in closed mode")
        if not isinstance(self.xyz, torch.Tensor) or not isinstance(
            self.xyz_area, torch.Tensor
        ):
            raise RuntimeError(
                "closed geometry meta requires sampled boundary and area points"
            )
        if self.xyz.ndim != 4 or self.xyz_area.ndim != 4:
            raise RuntimeError(
                "closed geometry meta expects xyz/xyz_area with shape (N, B, S, 2)"
            )
        if (
            self.xyz.shape[0] != self.xyz_area.shape[0]
            or self.xyz.shape[2] != self.xyz_area.shape[2]
        ):
            raise RuntimeError(
                "xyz and xyz_area must share curve and sample dimensions"
            )

        num_curves = int(self.xyz.shape[0])
        num_boundary_bands = int(self.xyz.shape[1])
        num_area_bands = int(self.xyz_area.shape[1])
        num_samples = int(self.xyz.shape[2])
        meta = torch.zeros(
            (num_curves, num_boundary_bands + num_area_bands, num_samples),
            dtype=torch.int32,
            device=self.xyz.device,
        )
        meta[:, :num_boundary_bands, :] = 1
        return meta.contiguous().view(-1)

    def _closed_depth_override(self) -> torch.Tensor:
        with torch.no_grad():
            boxes = self.compute_aabb(self.xyz.view(self.xyz.shape[0], -1, 2))
            ratio = self.W / self.H
            widths = (boxes[:, 2] - boxes[:, 0]) * ratio
            heights = boxes[:, 3] - boxes[:, 1]
            depth = widths * heights
            self._depth.copy_(depth.unsqueeze(-1).contiguous())
        return self.get_depth.view(-1).detach()

    def _update_open_depth_from_projected_xys(self) -> None:
        if self.iter >= 10000 or self.iter % 20 != 0:
            return
        with torch.no_grad():
            xys = self.xys.detach().view(self._control_points.shape[0], -1, 2)
            diffs = torch.norm(xys[:, 1:, :] - xys[:, :-1, :], dim=-1).sum(
                -1, keepdim=True
            )
            diffs = diffs * torch.abs(self._scaling.detach())
            self._depth.copy_(diffs).contiguous()

    def _internal_process_fn(
        self,
        depth_override: Optional[torch.Tensor],
        gaussian_meta: Optional[torch.Tensor],
    ):
        def _fn(stage: str, payload: Mapping[str, Any]):
            if stage == "post_project":
                self.xys = payload["xys"]
                self.radii = payload["radii"]
                self._last_project_payload = {
                    "xys": payload["xys"].detach(),
                    "depths": payload["depths"].detach(),
                    "radii": payload["radii"].detach(),
                    "conics": payload["conics"].detach(),
                    "num_tiles_hit": payload["num_tiles_hit"].detach(),
                }
                updates: dict[str, torch.Tensor] = {}
                if depth_override is not None:
                    updates["depths"] = depth_override.to(payload["depths"].dtype)
                if gaussian_meta is not None:
                    meta = gaussian_meta.to(
                        device=payload["xys"].device, dtype=torch.int32
                    )
                    if meta.numel() != int(payload["xys"].shape[0]):
                        raise RuntimeError(
                            "gaussian_meta size must match projected gaussian count in post_project stage"
                        )
                    updates["gaussian_meta"] = meta.contiguous().view(-1)
                if updates:
                    return updates
            if stage == "pre_backward_rasterize":
                if not bool(getattr(self, "contrib_stats_enabled", False)):
                    return None
                if not bool(getattr(self, "contrib_stats_collect", False)):
                    return None
                collect_isect = bool(
                    getattr(self, "contrib_stats_collect_isect", False)
                )
                return {
                    "collect_curve_contrib_rgb": True,
                    "collect_curve_isect_contrib": collect_isect,
                }
            if stage == "post_backward_rasterize":
                if not bool(getattr(self, "contrib_stats_enabled", False)):
                    return None
                if not bool(getattr(self, "contrib_stats_collect", False)):
                    return None
                topk = int(max(1, getattr(self, "contrib_stats_topk", 16)))
                include_rate_tensor = bool(
                    getattr(self, "contrib_stats_keep_vector", False)
                )
                primitive_contrib_rgb = payload.get("curve_contrib_rgb")
                num_curves = int(self._control_points.shape[0])
                curve_contrib_rgb = torch.zeros(
                    (num_curves, 3),
                    dtype=torch.float32,
                    device=self._control_points.device,
                )
                if isinstance(primitive_contrib_rgb, torch.Tensor):
                    prim_rgb = primitive_contrib_rgb.detach().to(
                        device=self._control_points.device,
                        dtype=torch.float32,
                    )
                    if prim_rgb.ndim == 2 and int(prim_rgb.shape[1]) == 3:
                        if int(prim_rgb.shape[0]) == num_curves:
                            curve_contrib_rgb = prim_rgb.contiguous()
                        elif (
                            num_curves > 0 and int(prim_rgb.shape[0]) % num_curves == 0
                        ):
                            per_curve = int(prim_rgb.shape[0] // num_curves)
                            curve_contrib_rgb = prim_rgb.view(
                                num_curves, per_curve, 3
                            ).sum(dim=1)
                area_norm_enabled = bool(
                    getattr(self, "contrib_stats_area_normalize", True)
                )
                area_clamp_min = float(
                    max(0.0, getattr(self, "contrib_stats_area_clamp_min", 9.0))
                )
                area_clamp_max = float(
                    getattr(self, "contrib_stats_area_clamp_max", 0.0)
                )
                if area_clamp_max <= 0.0:
                    h_i = max(1, int(getattr(self, "H", 1)))
                    w_i = max(1, int(getattr(self, "W", 1)))
                    area_clamp_max = float(max(1, h_i * w_i))
                if area_clamp_max < area_clamp_min:
                    area_clamp_max = area_clamp_min
                curve_area = (
                    self._estimate_curve_coverage_area(num_curves=num_curves)
                    if area_norm_enabled
                    else None
                )
                summary = self._summarize_primitive_contrib(
                    curve_contrib_rgb=curve_contrib_rgb,
                    topk=topk,
                    include_rate_tensor=include_rate_tensor,
                    include_curve_contrib_rgb=include_rate_tensor,
                    curve_coverage_area=curve_area,
                    area_clamp_min=area_clamp_min,
                    area_clamp_max=area_clamp_max,
                )
                collect_isect = bool(
                    getattr(self, "contrib_stats_collect_isect", False)
                )
                if collect_isect:
                    primitive_ids_sorted = payload.get("gaussian_ids_sorted")
                    if not isinstance(primitive_ids_sorted, torch.Tensor):
                        primitive_ids_sorted = payload.get("primitive_ids_sorted")
                    tile_bins = payload.get("tile_bins")
                    isect_contrib = payload.get("curve_isect_contrib")
                    primitive_count = 0
                    colors_tensor = payload.get("colors")
                    if (
                        isinstance(colors_tensor, torch.Tensor)
                        and colors_tensor.ndim >= 1
                    ):
                        primitive_count = int(colors_tensor.shape[0])
                    elif isinstance(primitive_contrib_rgb, torch.Tensor):
                        primitive_count = int(primitive_contrib_rgb.shape[0])
                    tile_payload = self._aggregate_sparse_curve_tile_contrib_from_isect(
                        isect_contrib=isect_contrib
                        if isinstance(isect_contrib, torch.Tensor)
                        else torch.empty((0,), dtype=torch.float32),
                        primitive_ids_sorted=primitive_ids_sorted
                        if isinstance(primitive_ids_sorted, torch.Tensor)
                        else torch.empty((0,), dtype=torch.int32),
                        tile_bins=tile_bins
                        if isinstance(tile_bins, torch.Tensor)
                        else torch.empty((0, 2), dtype=torch.int32),
                        num_curves=num_curves,
                        num_primitives=int(max(0, primitive_count)),
                        min_value=float(
                            max(
                                0.0,
                                getattr(self, "contrib_stats_isect_min_value", 1e-6),
                            )
                        ),
                    )
                    summary.update(tile_payload)
                summary["step"] = int(getattr(self, "iter", 0)) + 1
                self._last_contrib_payload = summary
            return None

        return _fn

    def forward(
        self,
        factor: int = 1,
        denser_sample: bool = False,
        process_fn=None,
    ):
        final_h = self.H * factor
        final_w = self.W * factor
        width_scale = float(max(1, int(factor)))
        backend_name = self._renderer_backend_name(
            getattr(self, "renderer_backend", None)
        )
        use_cubic_polyline_backend = self._cubic_polyline_backend_enabled()
        use_cubic_fill_backend = self._cubic_fill_backend_enabled()
        if backend_name == "cubic" and not use_cubic_polyline_backend:
            raise ValueError(
                "renderer_backend='cubic' currently supports only "
                "mode='unclosed' with bezier_degree=3 (polyline)"
            )
        if backend_name == "cubic_fill" and not use_cubic_fill_backend:
            raise ValueError(
                "renderer_backend='cubic_fill' currently supports only mode='closed'"
            )
        use_gaussian_backend = (not use_cubic_polyline_backend) and (
            not use_cubic_fill_backend
        )
        if use_gaussian_backend:
            with prof("train.step.forward.sample"):
                sampled = self.get_xyz_and_depth(factor, denser_sample)
                if isinstance(sampled, tuple):
                    self.xyz, self.xyz_area = sampled
                else:
                    self.xyz = sampled
                    self.xyz_area = torch.zeros(
                        (1,),
                        device=self.xyz.device,
                        dtype=self.xyz.dtype,
                    )
                if self.mode == "closed":
                    xyz_input = (
                        torch.cat([self.xyz, self.xyz_area], dim=1)
                        .contiguous()
                        .view(-1, 2)
                    )
                else:
                    if self.mode == "line" and self.xyz.ndim == 3:
                        xyz_input = self.xyz.contiguous().view(-1, 2)
                    else:
                        xyz_input = self.xyz
        else:
            xyz_input = None

        if use_cubic_polyline_backend:
            cubic_polyline_process_fn = self._internal_process_fn(
                depth_override=None,
                gaussian_meta=None,
            )
            with prof("train.step.forward.rotation"):
                rotation_input = None
                _ = rotation_input

            with prof("train.step.forward.scaling"):
                scaling = None
                _ = scaling

            with prof("train.step.forward.attr"):
                (
                    cubic_control_points,
                    cubic_seg_offsets,
                    features_dc,
                    opacity,
                    stroke_widths,
                    depths,
                    cubic_aa_widths,
                ) = self._cubic_unclosed_polyline_render_inputs()
                self.tile_bounds = (
                    (final_w + self.BLOCK_W - 1) // self.BLOCK_W,
                    (final_h + self.BLOCK_H - 1) // self.BLOCK_H,
                    1,
                )
                is_training = bool(self.training)
                cubic_distance_samples = int(
                    self.cubic_distance_samples_train
                    if is_training
                    else self.cubic_distance_samples_eval
                )
                if width_scale != 1.0:
                    stroke_widths = stroke_widths * width_scale
                    cubic_aa_widths = cubic_aa_widths * width_scale
                cubic_flatten_method = str(
                    getattr(self, "cubic_flatten_method", "bernstein")
                )
                # Polyline global-edge path is sensitive to per-segment anisotropic
                # AABB tightening (chord-oriented pads may under-cover curved spans).
                # Keep unclosed cubic on conservative isotropic intersects.
                use_aniso_intersects = False

            with prof("train.step.forward.render"):
                with prof("train.step.forward.render.cubic"):
                    out_img = render_cubic_polyline_splat_tilelang(
                        cubic_control_points.contiguous(),
                        cubic_seg_offsets.contiguous(),
                        features_dc.contiguous(),
                        opacity.contiguous(),
                        stroke_widths.contiguous(),
                        depths.contiguous(),
                        int(final_h),
                        int(final_w),
                        int(self.BLOCK_W),
                        int(self.BLOCK_H),
                        self.background,
                        aniso_intersects=use_aniso_intersects,
                        aa_widths=cubic_aa_widths.contiguous(),
                        distance_samples=cubic_distance_samples,
                        flatten_method=cubic_flatten_method,
                        process_fn=cubic_polyline_process_fn,
                    )

            self.xys = self._sample_cubic_debug_points(
                cubic_control_points.contiguous(),
                samples=max(8, int(self.num_samples // 2)),
            )
            self.radii = torch.zeros(
                (int(cubic_control_points.shape[0]),),
                dtype=torch.int32,
                device=cubic_control_points.device,
            )
            self._last_project_payload = {}
            self._last_contrib_payload = {}
        elif use_cubic_fill_backend:
            cubic_fill_process_fn = self._internal_process_fn(
                depth_override=None,
                gaussian_meta=None,
            )
            with prof("train.step.forward.rotation"):
                rotation_input = None
                _ = rotation_input

            with prof("train.step.forward.scaling"):
                scaling = None
                _ = scaling

            with prof("train.step.forward.attr"):
                (
                    fill_segments,
                    fill_seg_offsets,
                    features_dc,
                    opacity,
                    depths,
                    fill_rules,
                    fill_aa_widths,
                ) = self._cubic_fill_render_inputs()
                self.tile_bounds = (
                    (final_w + self.BLOCK_W - 1) // self.BLOCK_W,
                    (final_h + self.BLOCK_H - 1) // self.BLOCK_H,
                    1,
                )
                is_training = bool(self.training)
                cubic_distance_samples = int(
                    self.cubic_distance_samples_train
                    if is_training
                    else self.cubic_distance_samples_eval
                )
                if width_scale != 1.0:
                    fill_aa_widths = fill_aa_widths * width_scale
                cubic_flatten_method = str(
                    getattr(self, "cubic_flatten_method", "bernstein")
                )

            with prof("train.step.forward.render"):
                with prof("train.step.forward.render.cubic_fill"):
                    out_img = render_cubic_fill_splat_tilelang(
                        fill_segments,
                        fill_seg_offsets,
                        features_dc,
                        opacity,
                        depths,
                        int(final_h),
                        int(final_w),
                        int(self.BLOCK_W),
                        int(self.BLOCK_H),
                        self.background,
                        fill_rules=fill_rules,
                        aa_widths=fill_aa_widths,
                        distance_samples=cubic_distance_samples,
                        flatten_method=cubic_flatten_method,
                        process_fn=cubic_fill_process_fn,
                    )

            self.xys = self._sample_cubic_debug_points(
                fill_segments,
                samples=max(8, int(self.num_samples // 2)),
            )
            self.radii = torch.zeros(
                (int(fill_segments.shape[0]),),
                dtype=torch.int32,
                device=fill_segments.device,
            )
            self._last_project_payload = {}
            self._last_contrib_payload = {}
        else:
            with prof("train.step.forward.rotation"):
                with torch.no_grad():
                    rotation_input = (
                        self.compute_rotations(
                            self.xyz.view(self._control_points.shape[0], -1, 2)
                        )
                        .view(-1)
                        .detach()
                    )

            with prof("train.step.forward.scaling"):
                if self.mode == "closed":
                    with torch.no_grad():
                        scaling = self.get_scaling(factor)
                else:
                    scaling = self.get_scaling(factor)

            with prof("train.step.forward.attr"):
                opacity = self.get_opacity.view(-1).contiguous()
                features_dc = self.get_features.contiguous().view(-1, 3)
                self.tile_bounds = (
                    (final_w + self.BLOCK_W - 1) // self.BLOCK_W,
                    (final_h + self.BLOCK_H - 1) // self.BLOCK_H,
                    1,
                )
                depth_override = (
                    self._closed_depth_override() if self.mode == "closed" else None
                )
                gaussian_meta = (
                    self._closed_gaussian_geometry_meta()
                    if self.mode == "closed" and self._closed_geom_cull_enabled()
                    else None
                )
                internal_process_fn = self._internal_process_fn(
                    depth_override, gaussian_meta
                )

            def _chained_process_fn(stage: str, payload: Mapping[str, Any]):
                updates = {}
                internal_updates = internal_process_fn(stage, payload)
                if internal_updates:
                    updates.update(internal_updates)
                    payload = dict(payload)
                    payload.update(internal_updates)
                if process_fn is not None:
                    external_updates = process_fn(stage, payload)
                    if external_updates:
                        updates.update(external_updates)
                return updates or None

            with prof("train.step.forward.render"):
                out_img = gs_render(
                    xyz_input.contiguous(),
                    scaling.contiguous(),
                    rotation_input.contiguous(),
                    features_dc,
                    opacity,
                    final_h,
                    final_w,
                    self.BLOCK_W,
                    self.BLOCK_H,
                    self.background,
                    _chained_process_fn,
                    profile_fwd_prefix="train.step.forward.render.tilelang",
                    profile_bwd_prefix="train.step.backward.tilelang",
                )

        with prof("train.step.forward.post"):
            if (
                self.mode != "closed"
                and not use_cubic_polyline_backend
                and not use_cubic_fill_backend
            ):
                self._update_open_depth_from_projected_xys()

            out_img = torch.clamp(out_img, 0, 1)
            out_img = out_img.permute(2, 0, 1).unsqueeze(0).contiguous()
            if self.debug_hook is not None:
                xys_debug = (
                    self.xys.detach() if isinstance(self.xys, torch.Tensor) else None
                )
                colors_debug = features_dc.detach()
                num_curves = int(self._control_points.shape[0])
                points_per_curve: Optional[int] = None
                if (
                    isinstance(xys_debug, torch.Tensor)
                    and xys_debug.ndim >= 2
                    and xys_debug.shape[0] > 0
                    and num_curves > 0
                    and xys_debug.shape[0] % num_curves == 0
                ):
                    points_per_curve = int(xys_debug.shape[0] // num_curves)
                emit_visual_debug(
                    self.debug_hook,
                    "post_forward",
                    {
                        "step": self.iter,
                        "xys": xys_debug,
                        "colors": colors_debug,
                        "image": out_img.detach(),
                        "num_curves": num_curves,
                        "points_per_curve": points_per_curve,
                        "height": int(final_h),
                        "width": int(final_w),
                    },
                )
        return {"render": out_img}

    def forward_area_boundary(self):
        return self.forward(factor=1, denser_sample=False)


__all__ = ["GaussianTraceRenderMixin"]
