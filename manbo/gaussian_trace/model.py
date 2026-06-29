from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from manbo import Adan
from visual_debug import VisualDebugHook

from .attributes import GaussianTraceAttributesMixin
from .common import custom_lr_schedule
from .legacy import GaussianTraceLegacyMixin
from .optimizer_state import GaussianTraceOptimizerStateMixin
from .render import GaussianTraceRenderMixin
from .sampling import GaussianTraceSamplingMixin
from .train_ops import GaussianTraceTrainOpsMixin


class GaussianTrace(
    GaussianTraceTrainOpsMixin,
    GaussianTraceRenderMixin,
    GaussianTraceOptimizerStateMixin,
    GaussianTraceAttributesMixin,
    GaussianTraceSamplingMixin,
    GaussianTraceLegacyMixin,
    nn.Module,
):
    def __init__(
        self,
        *,
        loss_type: str = "L2",
        H: int,
        W: int,
        BLOCK_W: int,
        BLOCK_H: int,
        device: torch.device,
        mode: str,
        num_curves: int,
        bezier_degree: int,
        num_samples: int,
        lr: float,
        opt_type: str = "adan",
        use_reg_loss: bool = False,
        renderer_backend: str = "gaussian",
        cubic_distance_samples_train: int = 12,
        cubic_distance_samples_eval: int = 18,
        cubic_flatten_method: str = "bernstein",
        quantize: bool = False,
        debug_hook: Optional[VisualDebugHook] = None,
        **_: Any,
    ):
        super().__init__()
        if loss_type.upper() != "L2":
            raise ValueError("GaussianTrace currently supports only L2 loss.")

        self.H, self.W = H, W
        self.ori_H, self.ori_W = H, W
        self.BLOCK_W, self.BLOCK_H = BLOCK_W, BLOCK_H
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.iter = 0
        self.device = device
        self.mode = mode
        self.use_reg_loss = bool(use_reg_loss)
        self.renderer_backend = str(renderer_backend).strip().lower()
        if self.renderer_backend not in {"gaussian", "cubic", "cubic_fill"}:
            raise ValueError(
                f"Unsupported renderer_backend={renderer_backend!r}; expected 'gaussian', 'cubic', or 'cubic_fill'"
            )
        self.cubic_distance_samples_train = max(2, int(cubic_distance_samples_train))
        self.cubic_distance_samples_eval = max(2, int(cubic_distance_samples_eval))
        flatten_method = str(cubic_flatten_method).strip().lower()
        if flatten_method in {"decasteljau", "de-casteljau"}:
            flatten_method = "de_casteljau"
        if flatten_method not in {"bernstein", "de_casteljau"}:
            raise ValueError(
                "Unsupported cubic_flatten_method="
                f"{cubic_flatten_method!r}; expected 'bernstein' or 'de_casteljau'"
            )
        self.cubic_flatten_method = flatten_method
        self.debug_hook = debug_hook

        self.num_curves_init = 128
        self.num_curves = int(num_curves)
        if self.mode == "closed":
            self.num_beziers = 2 * 1
        else:
            self.num_beziers = 3
        self.opacity_mode = 1
        self.bezier_degree = bezier_degree

        self.curves_resolution = 40
        self.max_sh_degree = 1
        self.radius = 0.01
        if self.mode == "line":
            self.num_samples = num_samples
            self.total_num_sample = self.num_samples
        elif self.mode == "unclosed":
            self.num_samples = 64
            self.total_num_sample = self.num_samples * self.num_beziers
            self.radius = 0.01
        elif self.mode == "closed":
            self.num_samples = num_samples
            self.total_num_sample = self.num_samples * self.num_beziers
        else:
            self.num_samples = 32
            self.total_num_sample = (
                self.num_samples * self.num_beziers + self.curves_resolution**2
            )

        self.rotation_activation = torch.sigmoid

        if self.mode == "line":
            self._control_points = self._initialize_control_points_line()
        elif self.mode == "closed":
            self._control_points = self._initialize_control_points()
        elif self.mode == "unclosed":
            self._control_points = self._initialize_control_points_line(
                (self.bezier_degree * 3) - 1
            )
        else:
            self._control_points = self._initialize_control_points()

        self._features_dc = nn.Parameter(torch.rand(self.num_curves, 3))
        self._cholesky = nn.Parameter(torch.rand(self.num_curves, 3))

        self._scaling = nn.Parameter(torch.ones(self.num_curves, 1) * 2)
        self._rotation = nn.Parameter(torch.zeros(self.num_curves, 1))

        self._xyz = nn.Parameter(torch.zeros(self.num_curves, self.num_samples, 2))
        self._depth = nn.Parameter(torch.ones(self.num_curves, 1))
        if self.opacity_mode == 1:
            self._opacity = nn.Parameter(torch.ones(self.num_curves, 3))
        else:
            self._opacity = nn.Parameter(torch.ones(self.num_curves, 1))
        self._bernstein_cache = {}

        self.last_size = (self.H, self.W)
        self.quantize = quantize
        self.register_buffer("background", torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer("bound", torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer("cholesky_bound", torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self._last_project_payload: dict[str, torch.Tensor] = {}
        self._last_contrib_payload: dict[str, Any] = {}
        self.contrib_stats_enabled = False
        self.contrib_stats_collect = False
        self.contrib_stats_keep_vector = False
        self.contrib_stats_collect_isect = False
        self.contrib_stats_isect_min_value = 1e-6
        self.contrib_stats_topk = 16
        self.contrib_stats_area_normalize = str(self.mode).lower() != "closed"
        self.contrib_stats_area_clamp_min = 9.0
        self.contrib_stats_area_clamp_max = float(max(1, self.H * self.W))

        if self.quantize:
            raise NotImplementedError(
                "Quantization path is not migrated yet. Please set quantize=False."
            )

        if self.mode == "unclosed":
            lr_cp, lr_feat, lr_opacity = 0.02, 1.0, 1.0
        else:
            lr_cp, lr_feat, lr_opacity = 0.02, 1.0, 1.0

        param_groups = [
            {
                "params": [self._control_points],
                "lr": lr * lr_cp,
                "name": "control_points",
            },
            {"params": [self._features_dc], "lr": lr * lr_feat, "name": "features_dc"},
            {"params": [self._cholesky], "lr": lr, "name": "cholesky"},
            {"params": [self._scaling], "lr": lr, "name": "scaling"},
            {"params": [self._opacity], "lr": lr * lr_opacity, "name": "opacity"},
            {"params": [self._depth], "lr": lr, "name": "depth"},
        ]

        self.optimizer = (
            torch.optim.Adam(param_groups, lr=lr)
            if opt_type == "adam"
            else Adan(param_groups, lr=lr)
        )

        if self.mode == "unclosed":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=7500, gamma=0.5
            )
        else:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=custom_lr_schedule)

        for deg in (self.bezier_degree + 1, self.bezier_degree):
            for mul in (1, 2, 4, 8, 16):
                self._update_bernstein_cache(deg, self.num_samples * mul, self.device)

        self.init_area_weight(self.device)

    def _init_data(self):
        if not self.quantize:
            return
        raise NotImplementedError(
            "Quantization path is not migrated yet. Please set quantize=False."
        )


GaussianImage_Cholesky = GaussianTrace


__all__ = [
    "GaussianTrace",
    "GaussianImage_Cholesky",
]
