from contextlib import nullcontext
from typing import (
    Any,
    Callable,
    ContextManager,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
)

import torch
from torch import nn

from manbo.ops import (
    get_tile_bin_edges_tilelang,
    map_gaussian_meta_to_intersects,
    map_gaussian_to_intersects_tilelang,
    project_gaussians_2d_scale_rot_backward_tilelang,
    project_gaussians_2d_scale_rot_forward_tilelang,
    rasterize_backward_tilelang,
    rasterize_forward_tilelang,
)

ProfileCtxFactory = Callable[[str], ContextManager]
_PROFILE_CTX_FACTORY: Optional[ProfileCtxFactory] = None
GsProcessPayload = MutableMapping[str, Any]
GsProcessFn = Callable[[str, GsProcessPayload], Optional[Mapping[str, Any]]]


def set_gs_profile_ctx_factory(factory: Optional[ProfileCtxFactory]) -> None:
    global _PROFILE_CTX_FACTORY
    _PROFILE_CTX_FACTORY = factory


def _prof(label: str):
    if _PROFILE_CTX_FACTORY is None:
        return nullcontext()
    return _PROFILE_CTX_FACTORY(label)


def _profile_label(prefix: str, leaf: str) -> str:
    base = str(prefix).strip(".")
    suffix = str(leaf).strip(".")
    if not base:
        return suffix
    if not suffix:
        return base
    return f"{base}.{suffix}"


def _compute_cumulative_intersects(
    num_tiles_hit: torch.Tensor,
) -> Tuple[int, torch.Tensor]:
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def _sort_isects_and_gaussian_ids(
    isect_ids: torch.Tensor,
    gaussian_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    isect_ids_sorted, sorted_idx = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_idx)
    return isect_ids_sorted, gaussian_ids_sorted, sorted_idx


def _tile_bounds(
    img_h: int,
    img_w: int,
    block_x: int,
    block_y: int,
) -> Tuple[int, int, int]:
    return ((img_w + block_x - 1) // block_x, (img_h + block_y - 1) // block_y, 1)


def _normalize_gaussian_meta(
    gaussian_meta: Optional[torch.Tensor],
    *,
    expected_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if gaussian_meta is None:
        return None
    meta = gaussian_meta.to(device=device, dtype=torch.int32).contiguous().view(-1)
    if meta.numel() != int(expected_size):
        raise ValueError(
            f"gaussian_meta size mismatch: expected {expected_size}, got {meta.numel()}"
        )
    return meta


def _project_backward_with_meta(
    means2d: torch.Tensor,
    scales2d: torch.Tensor,
    rotation: torch.Tensor,
    img_h: int,
    img_w: int,
    radii: torch.Tensor,
    conics: torch.Tensor,
    v_xy: torch.Tensor,
    depths: torch.Tensor,
    v_conic: torch.Tensor,
    gaussian_meta: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gaussian_meta is not None:
        meta = _normalize_gaussian_meta(
            gaussian_meta,
            expected_size=means2d.shape[0],
            device=means2d.device,
        )
        if meta is None:
            raise RuntimeError("gaussian_meta normalization unexpectedly returned None")
        keep = meta.to(dtype=v_xy.dtype).unsqueeze(1)
        v_xy = v_xy * keep
        v_conic = v_conic * keep

    _, v_mean2d, v_scales2d, v_rotation = (
        project_gaussians_2d_scale_rot_backward_tilelang(
            means2d,
            scales2d,
            rotation,
            img_h,
            img_w,
            radii,
            conics,
            v_xy,
            depths,
            v_conic,
        )
    )
    return v_mean2d, v_scales2d, v_rotation


def _run_process_fn(
    process_fn: Optional[GsProcessFn],
    stage: str,
    payload: GsProcessPayload,
) -> GsProcessPayload:
    if process_fn is None:
        return payload
    updates = process_fn(stage, payload)
    if updates:
        payload.update(updates)
    return payload


class GsRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        scales2d: torch.Tensor,
        rotation: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        img_h: int,
        img_w: int,
        block_x: int,
        block_y: int,
        bg: torch.Tensor,
        process_fn: Optional[GsProcessFn] = None,
        profile_fwd_prefix: str = "tilelang.fwd",
        profile_bwd_prefix: str = "tilelang.bwd",
    ):
        tile_bounds = _tile_bounds(img_h, img_w, block_x, block_y)
        block = (block_x, block_y, 1)
        img_size = (img_w, img_h, 1)

        with _prof(_profile_label(profile_fwd_prefix, "total")):
            with _prof(_profile_label(profile_fwd_prefix, "proj")):
                xys, depths, radii, conics, num_tiles_hit = (
                    project_gaussians_2d_scale_rot_forward_tilelang(
                        means2d,
                        scales2d,
                        rotation,
                        img_h,
                        img_w,
                        tile_bounds,
                        block_x=block_x,
                        block_y=block_y,
                    )
                )

            xys = xys.contiguous()
            depths = depths.contiguous()
            radii = radii.contiguous()
            conics = conics.contiguous()

            post_proj_payload: GsProcessPayload = {
                "means2d": means2d,
                "scales2d": scales2d,
                "rotation": rotation,
                "xys": xys,
                "depths": depths,
                "radii": radii,
                "conics": conics,
                "num_tiles_hit": num_tiles_hit,
                "colors": colors,
                "opacities": opacities,
                "bg": bg,
                "img_h": img_h,
                "img_w": img_w,
                "block_x": block_x,
                "block_y": block_y,
                "tile_bounds": tile_bounds,
            }
            post_proj_payload = _run_process_fn(
                process_fn, "post_project", post_proj_payload
            )
            xys = post_proj_payload["xys"]
            depths = post_proj_payload["depths"]
            radii = post_proj_payload["radii"]
            conics = post_proj_payload["conics"]
            num_tiles_hit = post_proj_payload["num_tiles_hit"]
            colors = post_proj_payload["colors"]
            opacities = post_proj_payload["opacities"]
            bg = post_proj_payload["bg"]
            gaussian_meta = _normalize_gaussian_meta(
                post_proj_payload.get("gaussian_meta"),
                expected_size=means2d.shape[0],
                device=xys.device,
            )

            with _prof(_profile_label(profile_fwd_prefix, "cums")):
                num_intersects, cum_tiles_hit = _compute_cumulative_intersects(
                    num_tiles_hit
                )

            if num_intersects < 1:
                raise RuntimeError(
                    "No intersections generated; increase num_points or radii"
                )

            with _prof(_profile_label(profile_fwd_prefix, "map")):
                isect_ids, gaussian_ids = map_gaussian_to_intersects_tilelang(
                    means2d.shape[0],
                    num_intersects,
                    xys,
                    depths,
                    radii,
                    cum_tiles_hit,
                    tile_bounds,
                    block_x=block_x,
                    block_y=block_y,
                    conics=conics,
                )

            with _prof(_profile_label(profile_fwd_prefix, "sort")):
                isect_ids_sorted, gaussian_ids_sorted, sorted_idx = (
                    _sort_isects_and_gaussian_ids(isect_ids, gaussian_ids)
                )
                intersect_meta_sorted = None
                if gaussian_meta is not None:
                    intersect_meta = map_gaussian_meta_to_intersects(
                        gaussian_meta,
                        gaussian_ids,
                    )
                    intersect_meta_sorted = torch.gather(intersect_meta, 0, sorted_idx)

            with _prof(_profile_label(profile_fwd_prefix, "bins")):
                num_tiles = tile_bounds[0] * tile_bounds[1]
                tile_bins = get_tile_bin_edges_tilelang(
                    num_intersects,
                    isect_ids_sorted,
                    num_tiles=num_tiles,
                    compact=True,
                )

            pre_rast_payload: GsProcessPayload = {
                "xys": xys,
                "conics": conics,
                "colors": colors,
                "opacities": opacities,
                "bg": bg,
                "gaussian_ids_sorted": gaussian_ids_sorted,
                "intersect_meta_sorted": intersect_meta_sorted,
                "tile_bins": tile_bins,
                "tile_bounds": tile_bounds,
                "block": block,
                "img_size": img_size,
            }
            pre_rast_payload = _run_process_fn(
                process_fn, "pre_rasterize", pre_rast_payload
            )
            xys = pre_rast_payload["xys"]
            conics = pre_rast_payload["conics"]
            colors = pre_rast_payload["colors"]
            opacities = pre_rast_payload["opacities"]
            bg = pre_rast_payload["bg"]
            gaussian_ids_sorted = pre_rast_payload["gaussian_ids_sorted"]
            intersect_meta_sorted = _normalize_gaussian_meta(
                pre_rast_payload.get("intersect_meta_sorted"),
                expected_size=gaussian_ids_sorted.shape[0],
                device=gaussian_ids_sorted.device,
            )
            tile_bins = pre_rast_payload["tile_bins"]

            with _prof(_profile_label(profile_fwd_prefix, "rast")):
                out_img, final_Ts, final_idx = rasterize_forward_tilelang(
                    tile_bounds,
                    block,
                    img_size,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    conics,
                    colors,
                    opacities,
                    bg,
                )

        ctx.save_for_backward(
            means2d,
            scales2d,
            rotation,
            colors,
            opacities,
            xys,
            depths,
            radii,
            conics,
            gaussian_ids_sorted,
            tile_bins,
            final_Ts,
            final_idx,
            bg,
        )
        ctx.img_h = img_h
        ctx.img_w = img_w
        ctx.block_x = block_x
        ctx.block_y = block_y
        ctx.tile_bounds = tile_bounds
        ctx.block = block
        ctx.img_size = img_size
        ctx.profile_bwd_prefix = str(profile_bwd_prefix)
        ctx.gaussian_meta = gaussian_meta
        ctx.intersect_meta_sorted = intersect_meta_sorted
        ctx.process_fn = process_fn

        return out_img

    @staticmethod
    def backward(ctx, grad_out_img: torch.Tensor):
        (
            means2d,
            scales2d,
            rotation,
            colors,
            opacities,
            xys,
            depths,
            radii,
            conics,
            gaussian_ids_sorted,
            tile_bins,
            final_Ts,
            final_idx,
            bg,
        ) = ctx.saved_tensors

        img_h = ctx.img_h
        img_w = ctx.img_w
        block = ctx.block
        img_size = ctx.img_size

        v_out_alpha = torch.zeros_like(final_Ts)
        gaussian_meta = getattr(ctx, "gaussian_meta", None)
        intersect_meta_sorted = getattr(ctx, "intersect_meta_sorted", None)
        process_fn = getattr(ctx, "process_fn", None)

        profile_bwd_prefix = str(getattr(ctx, "profile_bwd_prefix", "tilelang.bwd"))
        with _prof(_profile_label(profile_bwd_prefix, "total")):
            pre_bwd_payload: GsProcessPayload = {
                "collect_curve_contrib_rgb": False,
                "collect_curve_isect_contrib": False,
                "num_primitives": int(colors.shape[0]),
                "colors": colors,
            }
            pre_bwd_payload = _run_process_fn(
                process_fn,
                "pre_backward_rasterize",
                pre_bwd_payload,
            )
            collect_curve_contrib = bool(
                pre_bwd_payload.get("collect_curve_contrib_rgb", False)
            )
            collect_curve_isect_contrib = bool(
                pre_bwd_payload.get("collect_curve_isect_contrib", False)
            )
            curve_contrib_rgb = (
                torch.zeros_like(colors, dtype=torch.float32)
                if collect_curve_contrib
                else None
            )
            curve_isect_contrib = (
                torch.zeros(
                    (int(gaussian_ids_sorted.numel()),),
                    dtype=torch.float32,
                    device=colors.device,
                )
                if collect_curve_isect_contrib
                else None
            )
            with _prof(_profile_label(profile_bwd_prefix, "rast")):
                v_xy, v_conic, v_rgb, v_opacity = rasterize_backward_tilelang(
                    ctx.tile_bounds,
                    block,
                    img_size,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    conics,
                    colors,
                    opacities,
                    bg,
                    final_Ts,
                    final_idx,
                    grad_out_img,
                    v_out_alpha,
                    intersect_meta_sorted=intersect_meta_sorted,
                    curve_contrib_rgb=curve_contrib_rgb,
                    curve_isect_contrib=curve_isect_contrib,
                )
            post_bwd_payload: GsProcessPayload = {
                "v_xy": v_xy,
                "v_conic": v_conic,
                "v_rgb": v_rgb,
                "v_opacity": v_opacity,
                "curve_contrib_rgb": curve_contrib_rgb,
                "curve_isect_contrib": curve_isect_contrib,
                "xys": xys,
                "conics": conics,
                "colors": colors,
                "opacities": opacities,
                "gaussian_ids_sorted": gaussian_ids_sorted,
                "tile_bins": tile_bins,
                "final_Ts": final_Ts,
                "final_idx": final_idx,
                "grad_out_img": grad_out_img,
                "gaussian_meta": gaussian_meta,
                "intersect_meta_sorted": intersect_meta_sorted,
                "v_out_alpha": v_out_alpha,
            }
            post_bwd_payload = _run_process_fn(
                process_fn,
                "post_backward_rasterize",
                post_bwd_payload,
            )
            v_xy = post_bwd_payload["v_xy"]
            v_conic = post_bwd_payload["v_conic"]
            v_rgb = post_bwd_payload["v_rgb"]
            v_opacity = post_bwd_payload["v_opacity"]

            with _prof(_profile_label(profile_bwd_prefix, "proj")):
                v_mean2d, v_scales2d, v_rotation = _project_backward_with_meta(
                    means2d,
                    scales2d,
                    rotation,
                    img_h,
                    img_w,
                    radii,
                    conics,
                    v_xy,
                    depths,
                    v_conic,
                    gaussian_meta,
                )

        if v_opacity is not None and v_opacity.dim() != 1:
            v_opacity = v_opacity.view(-1)
        if v_rotation is not None and v_rotation.dim() != 1:
            v_rotation = v_rotation.view(-1)

        return (
            v_mean2d,
            v_scales2d,
            v_rotation,
            v_rgb,
            v_opacity,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def gs_render(
    means2d: torch.Tensor,
    scales2d: torch.Tensor,
    rotation: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    img_h: int,
    img_w: int,
    block_x: int,
    block_y: int,
    bg: torch.Tensor,
    process_fn: Optional[GsProcessFn] = None,
    *,
    profile_fwd_prefix: str = "tilelang.fwd",
    profile_bwd_prefix: str = "tilelang.bwd",
) -> torch.Tensor:
    return GsRenderFunction.apply(
        means2d,
        scales2d,
        rotation,
        colors,
        opacities,
        img_h,
        img_w,
        block_x,
        block_y,
        bg,
        process_fn,
        str(profile_fwd_prefix),
        str(profile_bwd_prefix),
    )


class GsRenderModule(nn.Module):
    def __init__(
        self,
        img_h: int,
        img_w: int,
        block_x: int = 16,
        block_y: int = 8,
        background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        process_fn: Optional[GsProcessFn] = None,
        profile_fwd_prefix: str = "tilelang.fwd",
        profile_bwd_prefix: str = "tilelang.bwd",
    ):
        super().__init__()
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.block_x = int(block_x)
        self.block_y = int(block_y)
        self.process_fn = process_fn
        self.profile_fwd_prefix = str(profile_fwd_prefix)
        self.profile_bwd_prefix = str(profile_bwd_prefix)
        bg = torch.tensor(background, dtype=torch.float32)
        self.register_buffer("background", bg)

    def forward(
        self,
        means2d: torch.Tensor,
        scales2d: torch.Tensor,
        rotation: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        bg: Optional[torch.Tensor] = None,
        process_fn: Optional[GsProcessFn] = None,
        *,
        profile_fwd_prefix: str | None = None,
        profile_bwd_prefix: str | None = None,
    ) -> torch.Tensor:
        if bg is None:
            bg = self.background
        active_process_fn = self.process_fn if process_fn is None else process_fn
        active_fwd_prefix = (
            self.profile_fwd_prefix
            if profile_fwd_prefix is None
            else str(profile_fwd_prefix)
        )
        active_bwd_prefix = (
            self.profile_bwd_prefix
            if profile_bwd_prefix is None
            else str(profile_bwd_prefix)
        )
        return gs_render(
            means2d,
            scales2d,
            rotation,
            colors,
            opacities,
            self.img_h,
            self.img_w,
            self.block_x,
            self.block_y,
            bg,
            active_process_fn,
            profile_fwd_prefix=active_fwd_prefix,
            profile_bwd_prefix=active_bwd_prefix,
        )
