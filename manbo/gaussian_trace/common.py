from __future__ import annotations

import time
import warnings
from typing import Any

import torch
import torch.nn.functional as F


def custom_lr_schedule(step: int) -> float:
    if step < 5000:
        return 1.0
    if step < 6000:
        return 0.5
    if step < 9000:
        return 0.2
    return 0.1


def time_cuda(func, name: str = ""):
    torch.cuda.synchronize()
    start = time.time()
    result = func()
    torch.cuda.synchronize()
    end = time.time()
    elapsed_ms = (end - start) * 1000
    print(f"{name} took {elapsed_ms:.2f} ms")
    return result


def register_gradient_hook(tensor: torch.Tensor, name: str = "tensor") -> None:
    def hook(grad: torch.Tensor) -> torch.Tensor:
        print(f"[Grad Hook] {name} grad stats:")
        print(f"  shape: {grad.shape}")
        print(f"  mean:  {grad.mean().item():.6f}")
        print(f"  std:   {grad.std().item():.6f}")
        print(f"  max:   {grad.max().item():.6f}")
        print(f"  min:   {grad.min().item():.6f}")
        return grad

    tensor.register_hook(hook)


def warn_legacy_method(instance: Any, method_name: str) -> None:
    warned = getattr(instance, "_legacy_warned", None)
    if warned is None:
        warned = set()
        setattr(instance, "_legacy_warned", warned)
    if method_name in warned:
        return
    warnings.warn(
        (
            f"GaussianTrace.{method_name} is deprecated and moved to legacy path. "
            "It remains for compatibility and may be removed in a future refactor."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    warned.add(method_name)


class FeatureAreaModulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight_mask: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(weight_mask)
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight_mask,) = ctx.saved_tensors
        return grad_output * weight_mask, None


class PointGradientSmoother(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        n, b, h, d = grad_output.shape
        grad = grad_output.view(n * b, h, d).permute(0, 2, 1)

        kernel = torch.tensor(
            [[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]],
            device=grad.device,
            dtype=grad.dtype,
        ).view(2, 1, 3)

        smoothed = F.conv1d(grad, kernel, padding=1, groups=2)
        smoothed = smoothed.permute(0, 2, 1).view(n, b, h, d)
        return smoothed


__all__ = [
    "custom_lr_schedule",
    "time_cuda",
    "register_gradient_hook",
    "warn_legacy_method",
    "FeatureAreaModulator",
    "PointGradientSmoother",
]
