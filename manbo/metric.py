from __future__ import annotations

from typing import Dict, Tuple

import torch

from .ops.metric import (
    DEFAULT_ITEMS_PER_THREAD,
    DEFAULT_SSIM_BLOCK_H,
    DEFAULT_SSIM_BLOCK_W,
    DEFAULT_SSIM_THREADS,
    DEFAULT_THREADS,
    mse_backward_diff_tilelang,
    mse_partial_sum_kernel,
    reduce_sum_1d_tilelang,
    ssim_map_kernel,
)

_WINDOW_CACHE: Dict[Tuple[str, int, float], torch.Tensor] = {}


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _as_cuda_float32_contiguous(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def _validate_pair_4d(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}"
        )
    if x.dim() != 4:
        raise ValueError(f"expected input shape [N, C, H, W], got {x.shape}")
    if min(x.shape) <= 0:
        raise ValueError(f"all dimensions must be > 0, got {x.shape}")


def _gaussian_window_2d(
    win_size: int, win_sigma: float, device: torch.device
) -> torch.Tensor:
    key = (str(device), int(win_size), float(win_sigma))
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached

    coords = torch.arange(win_size, dtype=torch.float32, device=device)
    coords -= win_size // 2
    g = torch.exp(-(coords**2) / (2 * (win_sigma**2)))
    g /= g.sum()
    win2d = torch.outer(g, g).contiguous()
    _WINDOW_CACHE[key] = win2d
    return win2d


def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: float = 1.0,
    eps: float = 1e-12,
    threads: int = DEFAULT_THREADS,
    items_per_thread: int = DEFAULT_ITEMS_PER_THREAD,
) -> torch.Tensor:
    _validate_pair_4d(x, y)
    if data_range <= 0.0:
        raise ValueError("data_range must be > 0")
    if eps < 0.0:
        raise ValueError("eps must be >= 0")

    x32 = _as_cuda_float32_contiguous(x, "x")
    y32 = _as_cuda_float32_contiguous(y, "y")
    x_flat = x32.view(-1)
    y_flat = y32.view(-1)

    block_span = threads * items_per_thread
    num_blocks = _ceildiv(x_flat.numel(), block_span)
    partial = torch.empty((num_blocks,), dtype=torch.float32, device=x32.device)

    kernel = mse_partial_sum_kernel(
        threads=threads,
        items_per_thread=items_per_thread,
        dtype="float32",
    )
    kernel(x_flat, y_flat, partial)

    sse = reduce_sum_1d_tilelang(
        partial, threads=threads, items_per_thread=items_per_thread
    )
    mse = sse / float(x_flat.numel())
    return 10.0 * torch.log10((data_range * data_range) / (mse + eps))


def _mse_forward_tilelang(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    threads: int,
    items_per_thread: int,
) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}"
        )

    x32 = x.float().contiguous()
    y32 = y.float().contiguous()
    x_flat = x32.view(-1)
    y_flat = y32.view(-1)

    block_span = threads * items_per_thread
    num_blocks = _ceildiv(x_flat.numel(), block_span)
    partial = torch.empty((num_blocks,), dtype=torch.float32, device=x32.device)

    kernel = mse_partial_sum_kernel(
        threads=threads,
        items_per_thread=items_per_thread,
        dtype="float32",
    )
    kernel(x_flat, y_flat, partial)

    sse = reduce_sum_1d_tilelang(
        partial,
        threads=threads,
        items_per_thread=items_per_thread,
    )
    return sse / float(x_flat.numel())


class _TileLangMSEFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        threads: int,
        items_per_thread: int,
        tilelang_backward: int,
    ):
        loss = _mse_forward_tilelang(
            x,
            y,
            threads=int(threads),
            items_per_thread=int(items_per_thread),
        )
        ctx.save_for_backward(x, y)
        ctx.numel = int(x.numel())
        ctx.threads = int(threads)
        ctx.tilelang_backward = int(tilelang_backward) != 0
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y = ctx.saved_tensors
        numel = float(ctx.numel)

        grad_x = None
        grad_y = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            scale = float((grad_output * (2.0 / numel)).item())
            if ctx.tilelang_backward and x.is_cuda and y.is_cuda:
                if ctx.needs_input_grad[0]:
                    grad_x = mse_backward_diff_tilelang(
                        x,
                        y,
                        scale=scale,
                        threads=ctx.threads,
                    )
                    if grad_x.dtype != x.dtype:
                        grad_x = grad_x.to(dtype=x.dtype)
                if ctx.needs_input_grad[1]:
                    grad_y = mse_backward_diff_tilelang(
                        y,
                        x,
                        scale=scale,
                        threads=ctx.threads,
                    )
                    if grad_y.dtype != y.dtype:
                        grad_y = grad_y.to(dtype=y.dtype)
            else:
                scale_t = grad_output * (2.0 / numel)
                diff = x - y
                if ctx.needs_input_grad[0]:
                    grad_x = scale_t * diff
                if ctx.needs_input_grad[1]:
                    grad_y = -scale_t * diff
        return grad_x, grad_y, None, None, None


def mse_loss_tilelang(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    threads: int = DEFAULT_THREADS,
    items_per_thread: int = DEFAULT_ITEMS_PER_THREAD,
    tilelang_backward: bool = True,
) -> torch.Tensor:
    return _TileLangMSEFn.apply(
        x,
        y,
        int(threads),
        int(items_per_thread),
        int(bool(tilelang_backward)),
    )


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    data_range: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    nonnegative_ssim: bool = False,
    block_h: int = DEFAULT_SSIM_BLOCK_H,
    block_w: int = DEFAULT_SSIM_BLOCK_W,
    threads: int = DEFAULT_SSIM_THREADS,
) -> torch.Tensor:
    _validate_pair_4d(x, y)
    if data_range <= 0.0:
        raise ValueError("data_range must be > 0")
    if win_size <= 0 or win_size % 2 == 0:
        raise ValueError("win_size must be a positive odd integer")
    if win_sigma <= 0.0:
        raise ValueError("win_sigma must be > 0")

    x32 = _as_cuda_float32_contiguous(x, "x")
    y32 = _as_cuda_float32_contiguous(y, "y")

    n, c, h, w = x32.shape
    out_h = h - win_size + 1
    out_w = w - win_size + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"input spatial size {(h, w)} is smaller than win_size {win_size}"
        )

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    x_flat = x32.view(n * c, h, w).contiguous()
    y_flat = y32.view(n * c, h, w).contiguous()
    win2d = _gaussian_window_2d(win_size, win_sigma, x32.device)
    ssim_map = torch.empty(
        (n * c, out_h, out_w), dtype=torch.float32, device=x32.device
    )

    kernel = ssim_map_kernel(
        win_size=win_size,
        block_h=block_h,
        block_w=block_w,
        threads=threads,
    )
    kernel(x_flat, y_flat, win2d, ssim_map, float(c1), float(c2))

    ssim_per_channel = ssim_map.view(n, c, -1).mean(-1)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    return ssim_per_channel.mean(1)


__all__ = [
    "mse_loss_tilelang",
    "psnr",
    "ssim",
]
