import tilelang
import tilelang.language as T
import torch

from .utils import get_warp_size

DEFAULT_THREADS = 256
DEFAULT_ITEMS_PER_THREAD = 8
DEFAULT_SSIM_THREADS = 64
DEFAULT_SSIM_BLOCK_H = 8
DEFAULT_SSIM_BLOCK_W = 16


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@tilelang.jit
def mse_partial_sum_kernel(
    threads: int = DEFAULT_THREADS,
    items_per_thread: int = DEFAULT_ITEMS_PER_THREAD,
    dtype: str = "float32",
    target: str = "auto",
    numel=T.dynamic("numel"),
    num_blocks=T.dynamic("num_blocks"),
):
    """
    Spec:
    - Input tensors are flattened 1D CUDA float32 tensors of the same `numel`.
    - Kernel computes per-block partial SSE (sum((x - y)^2)).
    - Supports non-divisible tail via bounds checks.
    """

    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"mse_partial_sum_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    warp_num = threads // warp_size
    warp_stride = warp_size + 1  # +1 padding avoids shared-memory bank conflicts.
    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        x: T.Tensor[[numel], dtype],
        y: T.Tensor[[numel], dtype],
        partial: T.Tensor[[num_blocks], "float32"],
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            T.annotate_safe_value(
                {
                    x: T.Cast(dtype, 0.0),
                    y: T.Cast(dtype, 0.0),
                }
            )
            warp_partial = T.alloc_shared((warp_num, warp_stride), "float32")
            tx = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            base = bx * block_span + tx
            T.assume(base >= 0)

            acc = T.alloc_var("float32")
            acc = T.Cast("float32", 0.0)
            for i in T.serial(items_per_thread):
                idx = base + i * threads
                T.assume(idx >= 0)
                diff = T.Cast("float32", x[idx] - y[idx])
                acc += diff * diff

            acc = T.warp_reduce_sum(acc)

            if lane_id == 0:
                warp_partial[warp_id, 0] = acc

            T.sync_threads()

            if warp_id == 0:
                block_acc = T.alloc_var("float32")
                block_acc = T.Cast("float32", 0.0)
                if lane_id < warp_num:
                    block_acc = warp_partial[lane_id, 0]
                block_acc = T.warp_reduce_sum(block_acc)
                if lane_id == 0:
                    partial[bx] = block_acc

    return kernel


@tilelang.jit
def reduce_sum_partial_kernel(
    threads: int = DEFAULT_THREADS,
    items_per_thread: int = DEFAULT_ITEMS_PER_THREAD,
    target: str = "auto",
    numel=T.dynamic("numel"),
    num_blocks=T.dynamic("num_blocks"),
):
    """
    Spec:
    - Input is a 1D float32 CUDA tensor.
    - Kernel reduces chunks into per-block partial sums.
    - Supports non-divisible tail via bounds checks.
    """

    if threads <= 0:
        raise ValueError("threads must be > 0")
    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"reduce_sum_partial_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    warp_num = threads // warp_size
    warp_stride = warp_size + 1  # +1 padding avoids shared-memory bank conflicts.
    block_span = threads * items_per_thread

    @T.prim_func
    def kernel(
        x: T.Tensor[[numel], "float32"],
        partial: T.Tensor[[num_blocks], "float32"],
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            T.annotate_safe_value({x: T.Cast("float32", 0.0)})
            warp_partial = T.alloc_shared((warp_num, warp_stride), "float32")
            tx = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            base = bx * block_span + tx
            T.assume(base >= 0)

            acc = T.alloc_var("float32")
            acc = T.Cast("float32", 0.0)
            for i in T.serial(items_per_thread):
                idx = base + i * threads
                T.assume(idx >= 0)
                acc += x[idx]

            acc = T.warp_reduce_sum(acc)

            if lane_id == 0:
                warp_partial[warp_id, 0] = acc

            T.sync_threads()

            if warp_id == 0:
                block_acc = T.alloc_var("float32")
                block_acc = T.Cast("float32", 0.0)
                if lane_id < warp_num:
                    block_acc = warp_partial[lane_id, 0]
                block_acc = T.warp_reduce_sum(block_acc)
                if lane_id == 0:
                    partial[bx] = block_acc

    return kernel


@tilelang.jit
def mse_backward_diff_kernel(
    threads: int = DEFAULT_THREADS,
    dtype: str = "float32",
    numel=T.dynamic("numel"),
):
    """
    Spec:
    - Computes grad = scale * (lhs - rhs) for 1D flattened tensors.
    - Inputs are CUDA tensors, output grad is float32.
    """

    if threads <= 0:
        raise ValueError("threads must be > 0")

    @T.prim_func
    def kernel(
        lhs: T.Tensor[[numel], dtype],
        rhs: T.Tensor[[numel], dtype],
        scale: T.float32,
        grad: T.Tensor[[numel], "float32"],
    ):
        grid = T.ceildiv(numel, threads)
        with T.Kernel(grid, threads=threads) as bx:
            T.annotate_safe_value(
                {
                    lhs: T.Cast(dtype, 0.0),
                    rhs: T.Cast(dtype, 0.0),
                }
            )
            for tx in T.Parallel(threads):
                idx = bx * threads + tx
                if idx < numel:
                    T.assume(idx >= 0)
                    grad[idx] = scale * T.Cast("float32", lhs[idx] - rhs[idx])

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def ssim_map_kernel(
    win_size: int = 11,
    block_h: int = DEFAULT_SSIM_BLOCK_H,
    block_w: int = DEFAULT_SSIM_BLOCK_W,
    threads: int = DEFAULT_SSIM_THREADS,
    target: str = "auto",
    num_channels=T.dynamic("num_channels"),
    height=T.dynamic("height"),
    width=T.dynamic("width"),
    out_height=T.dynamic("out_height"),
    out_width=T.dynamic("out_width"),
):
    """
    Spec:
    - Inputs are CUDA float32 tensors with shape [C_flat, H, W].
    - Output is SSIM map of shape [C_flat, H - win + 1, W - win + 1].
    - Uses valid-window semantics (no padding), matching METRIC_DESIGN.md flow.
    """

    if win_size <= 0 or win_size % 2 == 0:
        raise ValueError("win_size must be a positive odd integer")
    if block_h <= 0 or block_w <= 0:
        raise ValueError("block_h and block_w must be > 0")
    if threads <= 0:
        raise ValueError("threads must be > 0")

    warp_size = get_warp_size(target)
    if threads % warp_size != 0:
        raise ValueError(
            f"ssim_map_kernel requires threads to be a multiple of warp_size ({warp_size})"
        )

    warp_num = threads // warp_size
    tile_h = block_h + win_size - 1
    tile_w = block_w + win_size - 1
    tile_pad = 1
    if (tile_w + tile_pad) % warp_size == 0:
        tile_pad += 1
    tile_w_padded = tile_w + tile_pad
    tile_cols_per_step = warp_size
    tile_row_steps = _ceildiv(tile_h, warp_num)
    tile_col_steps = _ceildiv(tile_w, tile_cols_per_step)

    @T.prim_func
    def kernel(
        x: T.Tensor[[num_channels, height, width], "float32"],
        y: T.Tensor[[num_channels, height, width], "float32"],
        win: T.Tensor[[win_size, win_size], "float32"],
        out: T.Tensor[[num_channels, out_height, out_width], "float32"],
        c1: T.float32,
        c2: T.float32,
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_w),
            T.ceildiv(out_height, block_h),
            num_channels,
            threads=threads,
        ) as (bx, by, bz):
            T.annotate_safe_value(
                {
                    x: T.Cast("float32", 0.0),
                    y: T.Cast("float32", 0.0),
                }
            )
            x_shared = T.alloc_shared((tile_h, tile_w_padded), "float32")
            y_shared = T.alloc_shared((tile_h, tile_w_padded), "float32")
            zero = T.Cast("float32", 0.0)
            two = T.Cast("float32", 2.0)
            T.assume(bz < num_channels)

            for tx in T.Parallel(threads):
                warp_id = T.get_warp_idx()
                lane_id = T.get_lane_idx()

                # Warp-cooperative tile load:
                # - Rows are distributed across warps.
                # - Within each row, lanes load contiguous columns for coalesced global reads.
                for row_step in T.serial(tile_row_steps):
                    iy = row_step * warp_num + warp_id
                    if iy < tile_h:
                        for col_step in T.serial(tile_col_steps):
                            ix = col_step * tile_cols_per_step + lane_id
                            if ix < tile_w:
                                gy = by * block_h + iy
                                gx = bx * block_w + ix
                                T.assume(gy >= 0)
                                T.assume(gx >= 0)
                                in_bound = (gy < height) and (gx < width)
                                x_shared[iy, ix] = T.if_then_else(
                                    in_bound, x[bz, gy, gx], zero
                                )
                                y_shared[iy, ix] = T.if_then_else(
                                    in_bound, y[bz, gy, gx], zero
                                )

            for oy_i, ox_i in T.Parallel(block_h, block_w):
                oy = by * block_h + oy_i
                ox = bx * block_w + ox_i
                if (oy < out_height) and (ox < out_width):
                    mu1 = T.alloc_var("float32")
                    mu2 = T.alloc_var("float32")
                    ex2 = T.alloc_var("float32")
                    ey2 = T.alloc_var("float32")
                    exy = T.alloc_var("float32")
                    mu1 = zero
                    mu2 = zero
                    ex2 = zero
                    ey2 = zero
                    exy = zero

                    for ky in T.serial(win_size):
                        for kx in T.serial(win_size):
                            w = win[ky, kx]
                            xv = x_shared[oy_i + ky, ox_i + kx]
                            yv = y_shared[oy_i + ky, ox_i + kx]

                            mu1 += w * xv
                            mu2 += w * yv
                            ex2 += w * xv * xv
                            ey2 += w * yv * yv
                            exy += w * xv * yv

                    mu1_sq = mu1 * mu1
                    mu2_sq = mu2 * mu2
                    mu1_mu2 = mu1 * mu2
                    sigma1_sq = ex2 - mu1_sq
                    sigma2_sq = ey2 - mu2_sq
                    sigma12 = exy - mu1_mu2

                    luminance = (two * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
                    contrast_structure = (two * sigma12 + c2) / (
                        sigma1_sq + sigma2_sq + c2
                    )
                    out[bz, oy, ox] = luminance * contrast_structure

    return kernel


def mse_backward_diff_tilelang(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    scale: float,
    threads: int = DEFAULT_THREADS,
) -> torch.Tensor:
    lhs_flat = lhs.float().contiguous().view(-1)
    rhs_flat = rhs.float().contiguous().view(-1)
    grad = torch.empty_like(lhs_flat, dtype=torch.float32)

    kernel = mse_backward_diff_kernel(
        threads=threads,
        dtype="float32",
    )
    kernel(lhs_flat, rhs_flat, float(scale), grad)
    return grad.view(lhs.shape)


def reduce_sum_1d_tilelang(
    x: torch.Tensor,
    *,
    threads: int = DEFAULT_THREADS,
    items_per_thread: int = DEFAULT_ITEMS_PER_THREAD,
) -> torch.Tensor:
    kernel = reduce_sum_partial_kernel(
        threads=threads,
        items_per_thread=items_per_thread,
    )
    block_span = threads * items_per_thread
    cur = x.contiguous()

    while cur.numel() > 1:
        num_blocks = _ceildiv(cur.numel(), block_span)
        nxt = torch.empty((num_blocks,), dtype=torch.float32, device=cur.device)
        kernel(cur, nxt)
        cur = nxt

    return cur[0]
