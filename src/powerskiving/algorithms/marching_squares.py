"""Deterministic marching squares (SPEC 6.0.1 / 6.0.2)."""

from __future__ import annotations

from dataclasses import dataclass
import math

from powerskiving.deterministic import q

Pair = tuple[tuple[float, float], tuple[float, float]]

_CASE_SEGMENTS: dict[int, tuple[tuple[int, int], ...]] = {
    0: (),
    1: ((0, 3),),
    2: ((0, 1),),
    3: ((1, 3),),
    4: ((1, 2),),
    6: ((0, 2),),
    7: ((2, 3),),
    8: ((2, 3),),
    9: ((0, 2),),
    11: ((1, 2),),
    12: ((1, 3),),
    13: ((0, 1),),
    14: ((0, 3),),
    15: (),
}

_PAIR_A: tuple[tuple[int, int], ...] = ((0, 3), (1, 2))
_PAIR_B: tuple[tuple[int, int], ...] = ((0, 1), (2, 3))


@dataclass(frozen=True)
class MarchingSquaresComponent:
    points: tuple[tuple[float, float], ...]
    closed: bool
    has_self_intersection: bool


def _nz(x: float) -> float:
    if x == 0.0:
        return 0.0
    return x


def _edge_t(fa: float, fb: float) -> float | None:
    if fa == 0.0 and fb == 0.0:
        return 0.5
    if fa == 0.0:
        return 0.0
    if fb == 0.0:
        return 1.0
    if fa * fb < 0.0:
        return _nz(fa / (fa - fb))
    return None


def _edge_point(edge: int, i: int, j: int, t: float) -> tuple[float, float]:
    if edge == 0:
        return _nz(float(i) + t), _nz(float(j))
    if edge == 1:
        return _nz(float(i + 1)), _nz(float(j) + t)
    if edge == 2:
        return _nz(float(i) + t), _nz(float(j + 1))
    if edge == 3:
        return _nz(float(i)), _nz(float(j) + t)
    raise ValueError(f"invalid edge id: {edge}")


def _pick_ambiguous_pairs(
    *,
    case_id: int,
    f00: float,
    f10: float,
    f11: float,
    f01: float,
    ms_fc_eps: float,
    ms_eps_d: float,
    ms_tie_break: str,
) -> tuple[tuple[int, int], ...]:
    if ms_tie_break != "pair_B":
        raise ValueError("ms_tie_break must be pair_B")
    if case_id not in (5, 10):
        raise ValueError("ambiguous pair requested for non-ambiguous case")
    d = _nz(f00 - f10 - f01 + f11)
    det = _nz(f00 * f11 - f10 * f01)
    if abs(d) <= ms_eps_d:
        return _PAIR_B
    fs = _nz(det / d)
    if abs(fs) <= ms_fc_eps:
        return _PAIR_B
    if case_id == 5:
        return _PAIR_B if fs > 0.0 else _PAIR_A
    return _PAIR_A if fs > 0.0 else _PAIR_B


def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], eps: float) -> int:
    cross = _nz((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    if abs(cross) <= eps:
        return 0
    return 1 if cross > 0.0 else -1


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float], eps: float) -> bool:
    if _orient(a, b, p, eps) != 0:
        return False
    min_x = min(a[0], b[0]) - eps
    max_x = max(a[0], b[0]) + eps
    min_y = min(a[1], b[1]) - eps
    max_y = max(a[1], b[1]) + eps
    return (min_x <= p[0] <= max_x) and (min_y <= p[1] <= max_y)


def _segments_intersect(seg0: Pair, seg1: Pair, eps: float) -> bool:
    a, b = seg0
    c, d = seg1
    o1 = _orient(a, b, c, eps)
    o2 = _orient(a, b, d, eps)
    o3 = _orient(c, d, a, eps)
    o4 = _orient(c, d, b, eps)
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    if o1 == 0 and _on_segment(a, b, c, eps):
        return True
    if o2 == 0 and _on_segment(a, b, d, eps):
        return True
    if o3 == 0 and _on_segment(c, d, a, eps):
        return True
    if o4 == 0 and _on_segment(c, d, b, eps):
        return True
    return False


def _has_self_intersection(points: tuple[tuple[float, float], ...], eps: float) -> bool:
    n = len(points)
    if n < 4:
        return False
    segs: list[Pair] = []
    for i in range(n):
        segs.append((points[i], points[(i + 1) % n]))
    for i in range(n):
        for j in range(i + 1, n):
            if j == i:
                continue
            if j == (i + 1) % n:
                continue
            if i == (j + 1) % n:
                continue
            if _segments_intersect(segs[i], segs[j], eps):
                return True
    return False


def marching_squares_zero(
    *,
    field: tuple[tuple[float, ...], ...],
    ms_iso_eps: float,
    ms_fc_eps: float,
    ms_eps_d: float,
    ms_tie_break: str,
    sort_round_uv_digits: int,
    uv_on_edge_eps: float,
    cell_valid_mask: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[MarchingSquaresComponent, ...]:
    """Extract zero-level components from node scalar field."""
    ny = len(field)
    if ny < 2:
        return ()
    nx = len(field[0])
    if nx < 2:
        return ()
    for row in field:
        if len(row) != nx:
            raise ValueError("field must be rectangular")
    if cell_valid_mask is not None:
        if len(cell_valid_mask) != ny - 1:
            raise ValueError("cell_valid_mask height mismatch")
        for row in cell_valid_mask:
            if len(row) != nx - 1:
                raise ValueError("cell_valid_mask width mismatch")

    point_by_key: dict[tuple[float, float], tuple[float, float]] = {}
    seg_edges: dict[
        tuple[tuple[float, float], tuple[float, float]],
        tuple[tuple[float, float], tuple[float, float]],
    ] = {}

    for j in range(ny - 1):
        for i in range(nx - 1):
            if cell_valid_mask is not None and cell_valid_mask[j][i] != 1:
                continue
            f00 = _nz(0.0 if abs(field[j][i]) <= ms_iso_eps else field[j][i])
            f10 = _nz(0.0 if abs(field[j][i + 1]) <= ms_iso_eps else field[j][i + 1])
            f11 = _nz(0.0 if abs(field[j + 1][i + 1]) <= ms_iso_eps else field[j + 1][i + 1])
            f01 = _nz(0.0 if abs(field[j + 1][i]) <= ms_iso_eps else field[j + 1][i])
            b00 = 1 if f00 > 0.0 else 0
            b10 = 1 if f10 > 0.0 else 0
            b11 = 1 if f11 > 0.0 else 0
            b01 = 1 if f01 > 0.0 else 0
            case_id = (b00 << 0) | (b10 << 1) | (b11 << 2) | (b01 << 3)

            edge_t: dict[int, float] = {}
            t0 = _edge_t(f00, f10)
            if t0 is not None:
                edge_t[0] = t0
            t1 = _edge_t(f10, f11)
            if t1 is not None:
                edge_t[1] = t1
            t2 = _edge_t(f01, f11)
            if t2 is not None:
                edge_t[2] = t2
            t3 = _edge_t(f00, f01)
            if t3 is not None:
                edge_t[3] = t3

            if case_id in (5, 10):
                seg_pairs = _pick_ambiguous_pairs(
                    case_id=case_id,
                    f00=f00,
                    f10=f10,
                    f11=f11,
                    f01=f01,
                    ms_fc_eps=ms_fc_eps,
                    ms_eps_d=ms_eps_d,
                    ms_tie_break=ms_tie_break,
                )
            else:
                seg_pairs = _CASE_SEGMENTS.get(case_id, ())

            for ea, eb in seg_pairs:
                if ea not in edge_t or eb not in edge_t:
                    raise ValueError("marching_squares_error: missing edge intersection")
                p0 = _edge_point(ea, i, j, edge_t[ea])
                p1 = _edge_point(eb, i, j, edge_t[eb])
                k0 = (q(p0[0], sort_round_uv_digits), q(p0[1], sort_round_uv_digits))
                k1 = (q(p1[0], sort_round_uv_digits), q(p1[1], sort_round_uv_digits))
                if k0 == k1:
                    continue
                a, b = (k0, k1) if k0 <= k1 else (k1, k0)
                seg_key = (a, b)
                point_by_key.setdefault(k0, p0)
                point_by_key.setdefault(k1, p1)
                seg_edges.setdefault(seg_key, seg_key)

    sorted_seg_keys = sorted(seg_edges.keys())
    if not sorted_seg_keys:
        return ()

    adjacency: dict[tuple[float, float], list[tuple[tuple[float, float], tuple[float, float]]]] = {}
    for seg_key in sorted_seg_keys:
        a, b = seg_key
        adjacency.setdefault(a, []).append(seg_key)
        adjacency.setdefault(b, []).append(seg_key)
    for key in adjacency:
        adjacency[key].sort()
        if len(adjacency[key]) > 2:
            raise ValueError("marching_squares_error: branching degree > 2")

    components: list[MarchingSquaresComponent] = []
    unvisited = set(sorted_seg_keys)
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        comp_seg_keys: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        comp_points: set[tuple[float, float]] = set()
        while stack:
            seg = stack.pop()
            if seg in comp_seg_keys:
                continue
            comp_seg_keys.add(seg)
            a, b = seg
            comp_points.add(a)
            comp_points.add(b)
            for nxt in adjacency[a]:
                if nxt not in comp_seg_keys:
                    stack.append(nxt)
            for nxt in adjacency[b]:
                if nxt not in comp_seg_keys:
                    stack.append(nxt)

        for seg in comp_seg_keys:
            unvisited.discard(seg)

        degrees: dict[tuple[float, float], int] = {}
        for key in comp_points:
            deg = sum(1 for s in adjacency[key] if s in comp_seg_keys)
            if deg > 2:
                raise ValueError("marching_squares_error: branching degree > 2")
            degrees[key] = deg

        open_endpoints = sorted(key for key, deg in degrees.items() if deg == 1)
        closed = len(open_endpoints) == 0
        if closed:
            start_key = min(comp_points)
        else:
            start_key = open_endpoints[0]

        comp_unvisited = set(comp_seg_keys)
        chain: list[tuple[float, float]] = [start_key]
        current = start_key
        while True:
            candidates = [s for s in adjacency[current] if s in comp_unvisited]
            if not candidates:
                break
            seg = min(candidates)
            comp_unvisited.remove(seg)
            a, b = seg
            current = b if current == a else a
            chain.append(current)

        if comp_unvisited:
            raise ValueError("marching_squares_error: component traversal incomplete")
        if closed:
            if len(chain) < 2 or chain[-1] != chain[0]:
                raise ValueError("marching_squares_error: closed loop reconstruction failed")
            chain = chain[:-1]

        points = tuple(point_by_key[k] for k in chain)
        components.append(
            MarchingSquaresComponent(
                points=points,
                closed=closed,
                has_self_intersection=_has_self_intersection(points, uv_on_edge_eps) if closed else False,
            )
        )

    return tuple(components)
