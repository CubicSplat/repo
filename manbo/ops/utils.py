from __future__ import annotations

from functools import lru_cache

from tilelang.utils.target import (
    determine_target,
    target_get_warp_size,
)


@lru_cache(maxsize=16)
def resolve_target_and_warp_size(target: str = "auto") -> tuple[object, int]:
    """
    Resolve TileLang target and its warp size once per target string.
    """

    target_obj = determine_target(target, return_object=True)
    warp_size = int(target_get_warp_size(target_obj))
    if warp_size <= 0:
        raise ValueError(f"invalid warp_size={warp_size} for target={target!r}")
    return target_obj, warp_size


def get_warp_size(target: str = "auto") -> int:
    _, warp_size = resolve_target_and_warp_size(target)
    return warp_size
