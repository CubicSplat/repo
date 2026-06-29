from __future__ import annotations

from typing import Optional

import torch

from manbo import mse_loss_tilelang, psnr, regularization_loss_tilelang
from manbo.profile import prof


class GaussianTraceTrainOpsMixin:
    def _regularization_loss(self) -> torch.Tensor:
        mode_for_reg = "unclosed" if str(self.mode).lower() == "line" else self.mode
        xyz = None
        if self.mode == "closed":
            if self._cubic_fill_backend_enabled():
                with prof("train.step.loss.reg.sample"):
                    xyz = self.sample_bezier_closed_boundary(
                        self._control_points,
                        num_samples=int(self.num_samples),
                    )
                    self.xyz = xyz
            else:
                xyz = self.xyz
        return regularization_loss_tilelang(
            self._control_points,
            self._opacity,
            xyz,
            mode=mode_for_reg,
            num_beziers=int(self.num_beziers),
            bezier_degree=int(self.bezier_degree),
            num_samples=int(self.num_samples),
            boundary_degree=int(self.bezier_degree + 1),
        )

    def train_step(self, gt_image: torch.Tensor, *, compute_psnr: bool = True):
        with prof("train.step.optim.zero"):
            self.optimizer.zero_grad(set_to_none=True)

        with prof("train.step.forward"):
            render_pkg = self.forward()
            image = render_pkg["render"]

        with prof("train.step.loss"):
            with prof("train.step.loss.data"):
                data_loss = mse_loss_tilelang(image, gt_image.detach())
            use_reg_loss = bool(getattr(self, "use_reg_loss", False))
            if use_reg_loss:
                with prof("train.step.loss.reg"):
                    reg_loss = self._regularization_loss()
                loss = data_loss + reg_loss
            else:
                loss = data_loss

        with prof("train.step.backward"):
            loss.backward()

        with prof("train.step.optim.step"):
            self.optimizer.step()
            self.scheduler.step()
            self.iter += 1

        step_psnr: Optional[float] = None
        if compute_psnr:
            with prof("train.metric.psnr"):
                with torch.no_grad():
                    step_psnr = float(psnr(image.float(), gt_image.float()).item())
        return loss, step_psnr, image

    def train_iter(self, gt_image: torch.Tensor):
        return self.train_step(gt_image)

    def train_iter_opencurves(self, gt_image: torch.Tensor):
        return self.train_step(gt_image)


__all__ = ["GaussianTraceTrainOpsMixin"]
