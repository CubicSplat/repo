from __future__ import annotations

from typing import Any, Sequence

from rich import box
from rich.console import Console
from rich.table import Table

from .stats import TimingRow, TimingStats


def collect_timing_rows(
    stats: TimingStats,
    *,
    metric: str,
    prefix: str | None = None,
) -> list[TimingRow]:
    return stats.rows(metric=metric, prefix=prefix)


def _sample_info(rows: Sequence[TimingRow]) -> str:
    if not rows:
        return "samples/label=0"
    counts = sorted({row.n for row in rows})
    if len(counts) == 1:
        return f"samples/label={counts[0]}"
    return f"samples/label range=[{counts[0]}, {counts[-1]}]"


def _strip_prefix(label: str, prefix: str | None) -> str:
    display = str(label)
    if prefix and display.startswith(prefix):
        display = display[len(prefix) :]
    return display.strip(".")


def _resolve_parent_label(label: str, labels: set[str]) -> str | None:
    parts = [part for part in str(label).split(".") if part]
    if len(parts) < 2:
        return None
    for cut in range(len(parts) - 1, 0, -1):
        base = ".".join(parts[:cut])
        if base in labels and base != label:
            return base
        total = f"{base}.total"
        if total in labels and total != label:
            return total
    return None


def _hierarchy_order(
    labels: Sequence[str],
) -> tuple[list[str], dict[str, str | None], dict[str, int]]:
    uniq_labels = sorted({str(label) for label in labels if str(label)})
    label_set = set(uniq_labels)
    parent_map: dict[str, str | None] = {
        label: _resolve_parent_label(label, label_set) for label in uniq_labels
    }
    children: dict[str, list[str]] = {label: [] for label in uniq_labels}
    roots: list[str] = []
    for label in uniq_labels:
        parent = parent_map[label]
        if parent is None:
            roots.append(label)
            continue
        children.setdefault(parent, []).append(label)

    for key in children:
        children[key].sort()
    roots.sort()

    ordered: list[str] = []
    for root in roots:
        stack = [root]
        while stack:
            node = stack.pop()
            ordered.append(node)
            kids = children.get(node, [])
            for child in reversed(kids):
                stack.append(child)

    depth_map: dict[str, int] = {}

    def _depth(label: str) -> int:
        depth = depth_map.get(label)
        if depth is not None:
            return depth
        parent = parent_map.get(label)
        if parent is None:
            depth_map[label] = 0
            return 0
        d = _depth(parent) + 1
        depth_map[label] = d
        return d

    for label in ordered:
        _depth(label)
    return ordered, parent_map, depth_map


def render_hotspots_rich(
    console: Console,
    *,
    title: str,
    rows: Sequence[TimingRow],
    metric: str,
    top_k: int,
) -> None:
    if not rows:
        console.print(f"[yellow]{title}: no profiling samples[/yellow]")
        return

    total = sum(row.ms for row in rows)
    hot_rows = sorted(rows, key=lambda row: row.ms, reverse=True)[: max(1, int(top_k))]
    table = Table(
        title=f"{title} (metric={metric}, {_sample_info(rows)})",
        box=box.SIMPLE,
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("label", style="cyan")
    table.add_column("ms", justify="right")
    table.add_column("std", justify="right")
    table.add_column("var", justify="right")
    table.add_column("share%", justify="right")
    table.add_column("n", justify="right")
    for row in hot_rows:
        share = 100.0 * row.ms / max(total, 1e-12)
        table.add_row(
            row.label,
            f"{row.ms:.4f}",
            f"{row.std:.4f}",
            f"{row.var:.4f}",
            f"{share:.2f}%",
            str(row.n),
        )
    console.print(table)


def render_hierarchy_rich(
    console: Console,
    *,
    title: str,
    rows: Sequence[TimingRow],
    metric: str,
    prefix: str | None,
    global_take_label: str | None = None,
) -> None:
    if not rows:
        console.print(f"[yellow]{title}: no profiling samples[/yellow]")
        return

    row_map: dict[str, TimingRow] = {}
    for row in rows:
        display_label = _strip_prefix(row.label, prefix)
        if not display_label:
            continue
        row_map[display_label] = row
    if not row_map:
        console.print(f"[yellow]{title}: no profiling samples[/yellow]")
        return

    ordered, parent_map, depth_map = _hierarchy_order(list(row_map.keys()))
    use_sum_equivalent = str(metric) == "mean"
    children_map: dict[str, list[str]] = {}
    for child_label, parent_label in parent_map.items():
        if parent_label is None:
            continue
        children_map.setdefault(parent_label, []).append(child_label)

    def _take_value(row: TimingRow) -> float:
        return float(row.ms * row.n) if use_sum_equivalent else float(row.ms)

    root_labels = [label for label in ordered if parent_map.get(label) is None]
    if global_take_label and global_take_label in row_map:
        total = _take_value(row_map[global_take_label])
    elif root_labels:
        total = float(sum(_take_value(row_map[label]) for label in root_labels))
    else:
        total = float(sum(_take_value(row) for row in row_map.values()))
    total = max(total, 1e-12)
    table = Table(
        title=f"{title} (metric={metric}, {_sample_info(list(row_map.values()))})",
        box=box.SIMPLE,
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("node", style="cyan")
    table.add_column("ms", justify="right")
    table.add_column("parent%", justify="right")
    table.add_column("take%", justify="right")
    table.add_column("std", justify="right")
    table.add_column("var", justify="right")
    table.add_column("n", justify="right")

    for label in ordered:
        row = row_map[label]
        parent = parent_map.get(label)
        parent_ms = row_map[parent].ms if parent in row_map else None
        ratio_pct = (
            100.0 if parent_ms is None else 100.0 * row.ms / max(parent_ms, 1e-12)
        )
        take_pct = 100.0 * _take_value(row) / total
        parts = [p for p in label.split(".") if p]
        leaf = parts[-1] if parts else label
        if parent is None:
            display_label = label
        elif leaf == "total" and len(parts) >= 2:
            display_label = f"{parts[-2]}.total"
        else:
            display_label = leaf
        node = f"{'  ' * depth_map.get(label, 0)}{display_label}"
        table.add_row(
            node,
            f"{row.ms:.4f}",
            f"{ratio_pct:.2f}%",
            f"{take_pct:.2f}%",
            f"{row.std:.4f}",
            f"{row.var:.4f}",
            str(row.n),
        )
        children = children_map.get(label, [])
        if children:
            child_sum_ms = float(sum(row_map[c].ms for c in children))
            delta_ms = float(row.ms - child_sum_ms)
            tol = max(1e-6, abs(float(row.ms)) * 1e-6)
            if abs(delta_ms) > tol:
                parent_den = max(1e-12, abs(float(row.ms)))
                unk_parent_pct = 100.0 * delta_ms / parent_den
                unk_take_ms = (
                    delta_ms * float(row.n) if use_sum_equivalent else delta_ms
                )
                unk_take_pct = 100.0 * unk_take_ms / total
                unk_node = f"{'  ' * (depth_map.get(label, 0) + 1)}<unk>"
                table.add_row(
                    unk_node,
                    f"{delta_ms:.4f}",
                    f"{unk_parent_pct:.2f}%",
                    f"{unk_take_pct:.2f}%",
                    "0.0000",
                    "0.0000",
                    str(row.n),
                )
    if global_take_label and global_take_label in row_map:
        suffix = " (sum-equivalent for mean)" if use_sum_equivalent else ""
        table.caption = f"take% base={global_take_label}{suffix}"
    elif root_labels:
        suffix = " (sum-equivalent for mean)" if use_sum_equivalent else ""
        table.caption = f"take% base=sum(root nodes){suffix}"

    console.print(table)


def _normalize_compare_sort(sort_by: str) -> str:
    sort = str(sort_by).lower().strip()
    aliases = {
        "ratio": "speedup",
        "speedup": "speedup",
        "diff": "absdiff",
        "absdiff": "absdiff",
        "cuda": "cuda",
        "tilelang": "tilelang",
        "op": "op",
    }
    if sort not in aliases:
        raise ValueError(
            f"Unsupported sort_by={sort_by!r}, expected one of {sorted(aliases)}"
        )
    return aliases[sort]


def _speedup_ratio_and_pct(
    left_ms: float,
    right_ms: float,
    *,
    speedup_as: str,
    eps_ms: float,
) -> tuple[float, float]:
    def _safe_denom(value: float) -> float:
        if abs(float(value)) >= eps_ms:
            return float(value)
        return eps_ms if value >= 0.0 else -eps_ms

    mode = str(speedup_as)
    if mode == "cuda/tilelang":
        ratio = float(left_ms) / _safe_denom(float(right_ms))
    elif mode == "tilelang/cuda":
        ratio = float(right_ms) / _safe_denom(float(left_ms))
    else:
        raise ValueError("speedup_as must be one of: cuda/tilelang, tilelang/cuda")
    return ratio, (ratio - 1.0) * 100.0


def _resolve_compare_parent(
    op: str,
    op_set: set[str],
    *,
    root_op: str | None,
    attach_unparented_to_root: bool,
) -> str | None:
    if root_op and op == root_op:
        return None

    parts = [part for part in str(op).split(".") if part]
    if len(parts) >= 2:
        for cut in range(len(parts) - 1, 0, -1):
            base = ".".join(parts[:cut])
            if base in op_set and base != op:
                return base
            total = f"{base}.total"
            if total in op_set and total != op:
                return total

    if attach_unparented_to_root and root_op and root_op in op_set and op != root_op:
        return root_op
    return None


def _build_compare_tree(
    ops: Sequence[str],
    *,
    root_op: str | None,
    attach_unparented_to_root: bool,
) -> tuple[list[str], dict[str, str | None], dict[str, int], dict[str, list[str]]]:
    ordered_ops: list[str] = []
    seen: set[str] = set()
    for op in ops:
        norm = str(op).strip(".")
        if not norm or norm in seen:
            continue
        ordered_ops.append(norm)
        seen.add(norm)

    if root_op:
        root_norm = str(root_op).strip(".")
        if root_norm and root_norm not in seen:
            ordered_ops.insert(0, root_norm)
            seen.add(root_norm)
        root_op = root_norm

    op_set = set(ordered_ops)
    parent_map: dict[str, str | None] = {}
    for op in ordered_ops:
        parent_map[op] = _resolve_compare_parent(
            op,
            op_set,
            root_op=root_op,
            attach_unparented_to_root=attach_unparented_to_root,
        )

    children_map: dict[str, list[str]] = {op: [] for op in ordered_ops}
    roots: list[str] = []
    for op in ordered_ops:
        parent = parent_map.get(op)
        if parent is None:
            roots.append(op)
            continue
        if parent in children_map:
            children_map[parent].append(op)
        else:
            roots.append(op)

    depth_map: dict[str, int] = {}

    def _depth(op: str) -> int:
        d = depth_map.get(op)
        if d is not None:
            return d
        parent = parent_map.get(op)
        if parent is None:
            depth_map[op] = 0
            return 0
        value = _depth(parent) + 1
        depth_map[op] = value
        return value

    traversal: list[str] = []

    def _walk(op: str) -> None:
        traversal.append(op)
        for child in children_map.get(op, []):
            _walk(child)

    for root in roots:
        _walk(root)
    for op in traversal:
        _depth(op)
    return traversal, parent_map, depth_map, children_map


def compare_ops_report(
    *,
    stats: TimingStats,
    header: str,
    ops: Sequence[str],
    sort_by: str = "speedup",
    speedup_as: str = "cuda/tilelang",
    left_prefix: str = "cuda",
    right_prefix: str = "tilelang",
    root_op: str | None = "total",
    root_display: str = "step.total",
    left_total_override_ms: float | None = None,
    right_total_override_ms: float | None = None,
    attach_unparented_to_root: bool = True,
    top_k: int = 200,
    eps_ms: float = 1e-9,
    console: Console | None = None,
) -> None:
    del top_k
    console = console or Console()
    sort = _normalize_compare_sort(sort_by)

    ordered_ops, parent_map, depth_map, children_map = _build_compare_tree(
        ops,
        root_op=root_op,
        attach_unparented_to_root=bool(attach_unparented_to_root),
    )
    if not ordered_ops:
        console.print(f"[yellow]{header}: no ops to compare[/yellow]")
        return

    raw_rows: dict[str, dict[str, Any]] = {}
    for op in ordered_ops:
        left_label = f"{left_prefix}.{op}"
        right_label = f"{right_prefix}.{op}"
        left_n = int(stats.metric(left_label, "n"))
        right_n = int(stats.metric(right_label, "n"))
        raw_rows[op] = {
            "op": op,
            "left_ms": float(stats.metric(left_label, "mean")) if left_n > 0 else None,
            "right_ms": float(stats.metric(right_label, "mean"))
            if right_n > 0
            else None,
            "left_n": left_n,
            "right_n": right_n,
        }

    resolved_rows: dict[str, dict[str, Any]] = {}

    def _resolve(op: str) -> dict[str, Any]:
        cached = resolved_rows.get(op)
        if cached is not None:
            return cached

        base = raw_rows[op]
        left_ms = base["left_ms"]
        right_ms = base["right_ms"]
        left_n = int(base["left_n"])
        right_n = int(base["right_n"])
        children = children_map.get(op, [])
        if children:
            child_rows = [_resolve(child) for child in children]
            if left_ms is None:
                left_ms = float(sum(float(row["left_ms"]) for row in child_rows))
                left_samples = [
                    int(row["left_n"]) for row in child_rows if int(row["left_n"]) > 0
                ]
                left_n = min(left_samples) if left_samples else 0
            if right_ms is None:
                right_ms = float(sum(float(row["right_ms"]) for row in child_rows))
                right_samples = [
                    int(row["right_n"]) for row in child_rows if int(row["right_n"]) > 0
                ]
                right_n = min(right_samples) if right_samples else 0
        if left_ms is None:
            left_ms = 0.0
        if right_ms is None:
            right_ms = 0.0

        ratio, speedup_pct = _speedup_ratio_and_pct(
            float(left_ms),
            float(right_ms),
            speedup_as=speedup_as,
            eps_ms=eps_ms,
        )
        row = {
            "op": op,
            "left_ms": float(left_ms),
            "right_ms": float(right_ms),
            "diff_ms": float(left_ms) - float(right_ms),
            "ratio": ratio,
            "speedup_pct": speedup_pct,
            "left_n": int(left_n),
            "right_n": int(right_n),
        }
        resolved_rows[op] = row
        return row

    for op in ordered_ops:
        _resolve(op)

    root_ops = [op for op in ordered_ops if parent_map.get(op) is None]
    if not root_ops:
        root_ops = list(ordered_ops)

    if left_total_override_ms is not None:
        total_left = float(left_total_override_ms)
    elif root_op and root_op in resolved_rows:
        total_left = float(resolved_rows[root_op]["left_ms"])
    else:
        total_left = float(sum(float(resolved_rows[op]["left_ms"]) for op in root_ops))

    if right_total_override_ms is not None:
        total_right = float(right_total_override_ms)
    elif root_op and root_op in resolved_rows:
        total_right = float(resolved_rows[root_op]["right_ms"])
    else:
        total_right = float(
            sum(float(resolved_rows[op]["right_ms"]) for op in root_ops)
        )

    total_ratio, total_speedup_pct = _speedup_ratio_and_pct(
        total_left,
        total_right,
        speedup_as=speedup_as,
        eps_ms=eps_ms,
    )

    take_base = "left_ms" if speedup_as == "cuda/tilelang" else "right_ms"
    take_base_label = left_prefix if speedup_as == "cuda/tilelang" else right_prefix
    total_take_base = float(total_left if take_base == "left_ms" else total_right)
    if abs(total_take_base) < 1e-12:
        total_take_base = 1e-12

    console.print()
    console.print(
        f"{header} [mean] sort={sort} speedup_as={speedup_as}",
        style="bold",
        markup=False,
    )
    console.print(
        f"TOTAL: {left_prefix}={total_left:.4f}ms | "
        f"{right_prefix}={total_right:.4f}ms | "
        f"ratio={total_ratio:.4f} | speedup={total_speedup_pct:+.2f}%"
    )

    table = Table(
        box=box.SIMPLE,
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("op", style="cyan", overflow="fold")
    table.add_column("l_ms", justify="right")
    table.add_column("r_ms", justify="right")
    table.add_column("d_ms", justify="right")
    table.add_column("ratio", justify="right")
    table.add_column("spd%", justify="right")
    table.add_column("take%", justify="right")
    table.add_column("n", justify="right")
    table.caption = (
        f"left={left_prefix}, right={right_prefix}, take%=global({take_base_label})"
    )

    for op in ordered_ops:
        row = resolved_rows[op]
        parent = parent_map.get(op)
        take_pct = 100.0 * float(row[take_base]) / total_take_base
        parts = [part for part in op.split(".") if part]
        leaf = parts[-1] if parts else op
        if root_op and op == root_op:
            display_label = str(root_display)
        elif leaf == "total" and len(parts) >= 2:
            display_label = f"{parts[-2]}.total"
        elif parent is None:
            display_label = op
        else:
            display_label = leaf
        display_op = f"{'  ' * depth_map.get(op, 0)}{display_label}"
        sample_text = f"{row['left_n']}/{row['right_n']}"
        table.add_row(
            display_op,
            f"{row['left_ms']:.4f}",
            f"{row['right_ms']:.4f}",
            f"{row['diff_ms']:.4f}",
            f"{row['ratio']:.4f}",
            f"{row['speedup_pct']:+.2f}",
            f"{take_pct:.2f}%",
            sample_text,
        )

        children = children_map.get(op, [])
        if not children:
            continue
        child_left_ms = float(
            sum(float(resolved_rows[child]["left_ms"]) for child in children)
        )
        child_right_ms = float(
            sum(float(resolved_rows[child]["right_ms"]) for child in children)
        )
        unk_left = float(row["left_ms"]) - child_left_ms
        unk_right = float(row["right_ms"]) - child_right_ms
        tol_left = max(1e-6, abs(float(row["left_ms"])) * 1e-6)
        tol_right = max(1e-6, abs(float(row["right_ms"])) * 1e-6)
        if abs(unk_left) <= tol_left and abs(unk_right) <= tol_right:
            continue

        unk_ratio, unk_speedup = _speedup_ratio_and_pct(
            unk_left,
            unk_right,
            speedup_as=speedup_as,
            eps_ms=eps_ms,
        )
        unk_take = (
            100.0
            * (unk_left if take_base == "left_ms" else unk_right)
            / total_take_base
        )
        table.add_row(
            f"{'  ' * (depth_map.get(op, 0) + 1)}<unk>",
            f"{unk_left:.4f}",
            f"{unk_right:.4f}",
            f"{(unk_left - unk_right):.4f}",
            f"{unk_ratio:.4f}",
            f"{unk_speedup:+.2f}",
            f"{unk_take:.2f}%",
            sample_text,
        )
    console.print(table)


def top_rows_for_structlog(
    rows: Sequence[TimingRow], *, top_k: int = 10
) -> list[dict[str, Any]]:
    total = sum(row.ms for row in rows)
    hot_rows = sorted(rows, key=lambda row: row.ms, reverse=True)[: max(1, int(top_k))]
    payload: list[dict[str, Any]] = []
    for row in hot_rows:
        share = 100.0 * row.ms / max(total, 1e-12)
        payload.append(
            {
                "label": row.label,
                "ms": round(float(row.ms), 6),
                "share_pct": round(float(share), 4),
                "n": int(row.n),
            }
        )
    return payload


__all__ = [
    "collect_timing_rows",
    "render_hotspots_rich",
    "render_hierarchy_rich",
    "compare_ops_report",
    "top_rows_for_structlog",
]
