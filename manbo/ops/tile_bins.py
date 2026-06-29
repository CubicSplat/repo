import tilelang
import tilelang.language as T
import torch

BLOCK_X = 16
BLOCK_Y = 16
DEFAULT_THREADS = 256


@tilelang.jit(pass_configs={"tl.enable_fast_math": True})
def get_tile_bin_edges_kernel(
    threads: int = DEFAULT_THREADS,
):
    """TileLang kernel for get_tile_bin_edges (forward.cu)."""

    num_intersects = T.dynamic("num_intersects")
    num_tiles = T.dynamic("num_tiles")

    @T.prim_func
    def kernel(
        isect_ids_sorted: T.Tensor[[num_intersects], T.int64],
        tile_bins: T.Tensor[[num_tiles, 2], T.int32],
    ):
        with T.Kernel(T.ceildiv(num_intersects, threads), threads=threads) as bx:
            for tx in T.Parallel(threads):
                idx = bx * threads + tx
                if idx < num_intersects:
                    cur_tile_idx = T.Cast(T.int32, isect_ids_sorted[idx] >> 32)
                    T.assume(cur_tile_idx >= 0)
                    T.assume(cur_tile_idx < num_tiles)
                    if idx == 0:
                        tile_bins[cur_tile_idx, 0] = T.Cast(T.int32, 0)
                    if idx == num_intersects - 1:
                        tile_bins[cur_tile_idx, 1] = num_intersects
                    if idx > 0:
                        prev_tile_idx = T.Cast(T.int32, isect_ids_sorted[idx - 1] >> 32)
                        T.assume(prev_tile_idx >= 0)
                        T.assume(prev_tile_idx < num_tiles)
                        if prev_tile_idx != cur_tile_idx:
                            tile_bins[prev_tile_idx, 1] = idx
                            tile_bins[cur_tile_idx, 0] = idx

    return kernel


def get_tile_bin_edges_tilelang(
    num_intersects: int,
    isect_ids_sorted: torch.Tensor,
    *,
    num_tiles: int,
    compact: bool = False,
    threads: int = DEFAULT_THREADS,
):
    kernel = get_tile_bin_edges_kernel(threads=threads)
    out_rows = num_tiles if compact else max(int(num_tiles), int(num_intersects))
    tile_bins = torch.zeros(
        (out_rows, 2), dtype=torch.int32, device=isect_ids_sorted.device
    )
    kernel(isect_ids_sorted.contiguous(), tile_bins)
    return tile_bins
