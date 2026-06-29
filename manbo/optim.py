import math

import torch
from manbo.ops.optim import adan_step_from_grad_kernel
from torch.optim.optimizer import Optimizer

DEFAULT_TILELANG_THREADS = 256


class Adan(Optimizer):
    """Adan optimizer implemented with TileLang step kernel."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,
        no_prox: bool = False,
        tilelang_threads: int = DEFAULT_TILELANG_THREADS,
    ):
        if not 0.0 <= max_grad_norm:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if tilelang_threads <= 0:
            raise ValueError(f"Invalid tilelang_threads: {tilelang_threads}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            no_prox=no_prox,
            tilelang_threads=tilelang_threads,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("no_prox", False)
            group.setdefault("tilelang_threads", DEFAULT_TILELANG_THREADS)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group["step"] = 0
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["exp_avg_diff"] = torch.zeros_like(p)
                state["neg_pre_grad"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults["max_grad_norm"] > 0:
            device = self.param_groups[0]["params"][0].device
            global_grad_norm = torch.zeros(1, device=device)
            max_grad_norm = torch.tensor(self.defaults["max_grad_norm"], device=device)
            eps = self.param_groups[0]["eps"]

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        global_grad_norm.add_(p.grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)
            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + eps),
                max=1.0,
            ).item()
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            group["step"] = group.get("step", 0) + 1
            step = int(group["step"])

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            bias_correction3_sqrt = math.sqrt(1.0 - beta3**step)

            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue

            threads = int(group["tilelang_threads"])
            no_prox = bool(group["no_prox"])
            if (
                "_tl_kernel" not in group
                or group.get("_tl_kernel_threads") != threads
                or group.get("_tl_kernel_no_prox") != no_prox
            ):
                group["_tl_kernel"] = adan_step_from_grad_kernel(
                    threads=threads,
                    dtype="float32",
                    no_prox=no_prox,
                )
                group["_tl_kernel_threads"] = threads
                group["_tl_kernel_no_prox"] = no_prox
            kernel = group["_tl_kernel"]

            for p in params_with_grad:
                if p.dtype != torch.float32:
                    raise ValueError("TileLang Adan currently expects float32 params")
                if not p.is_contiguous():
                    raise ValueError("TileLang Adan expects contiguous params")

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)
                    state["neg_pre_grad"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_diff = state["exp_avg_diff"]
                neg_pre_grad = state["neg_pre_grad"]

                if (
                    exp_avg.dtype != torch.float32
                    or exp_avg_sq.dtype != torch.float32
                    or exp_avg_diff.dtype != torch.float32
                    or neg_pre_grad.dtype != torch.float32
                    or not exp_avg.is_contiguous()
                    or not exp_avg_sq.is_contiguous()
                    or not exp_avg_diff.is_contiguous()
                    or not neg_pre_grad.is_contiguous()
                ):
                    raise ValueError("TileLang Adan expects contiguous float32 states")

                if (
                    grad.dtype == torch.float32
                    and grad.device == p.device
                    and grad.is_contiguous()
                ):
                    grad_use = grad
                else:
                    grad_buf = state.get("_grad_buf")
                    if grad_buf is None or grad_buf.shape != p.shape:
                        grad_buf = torch.empty_like(p)
                        state["_grad_buf"] = grad_buf
                    grad_buf.copy_(grad)
                    grad_use = grad_buf

                numel = p.numel()
                kernel(
                    p.view(numel),
                    grad_use.view(numel),
                    exp_avg.view(numel),
                    exp_avg_sq.view(numel),
                    exp_avg_diff.view(numel),
                    neg_pre_grad.view(numel),
                    step,
                    float(beta1),
                    float(beta2),
                    float(beta3),
                    float(bias_correction1),
                    float(bias_correction2),
                    float(bias_correction3_sqrt),
                    float(group["lr"]),
                    float(group["weight_decay"]),
                    float(group["eps"]),
                    float(clip_global_grad_norm),
                )

        return loss
