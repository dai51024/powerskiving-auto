"""Step03A boundary extraction, classification, and wire export."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from powerskiving.algorithms import marching_squares_zero
from powerskiving.deterministic import fixed, q
from powerskiving.json_canon import write_json

STEP03A_REPORT_NAME = "cad_report_step03A.json"
BOUNDARY_PLUS_CSV_NAME = "flank_uv_boundary_plus.csv"
BOUNDARY_MINUS_CSV_NAME = "flank_uv_boundary_minus.csv"
BOUNDARY_WIRE_PLUS_STEP_NAME = "flank_boundary_wire_plus.step"
BOUNDARY_WIRE_MINUS_STEP_NAME = "flank_boundary_wire_minus.step"

_BOUNDARY_HEADER = (
    "loop_id,point_id,is_hole,parent_loop_id,"
    "u_idx,v_idx,u_idx_q,v_idx_q,"
    "area_uv,area_uv_q,"
    "centroid_u,centroid_v,centroid_u_q,centroid_v_q,"
    "length_uv,length_uv_q,"
    "loop_flipped"
)


@dataclass(frozen=True)
class _RawGrid:
    u_mm: tuple[tuple[float, ...], ...]
    v_mm: tuple[tuple[float, ...], ...]
    x_mm: tuple[tuple[float, ...], ...]
    y_mm: tuple[tuple[float, ...], ...]
    z_mm: tuple[tuple[float, ...], ...]
    valid: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _LoopGeom:
    points: tuple[tuple[float, float], ...]
    area_uv: float
    centroid_u: float
    centroid_v: float
    length_uv: float


@dataclass(frozen=True)
class _LoopFeature:
    src_idx: int
    points: tuple[tuple[float, float], ...]
    area_uv: float
    centroid_u: float
    centroid_v: float
    length_uv: float
    p_test: tuple[float, float]
    parent_src_idx: int
    depth: int
    is_hole: int
    loop_flipped: int


@dataclass(frozen=True)
class _LoopFinal:
    loop_id: int
    points: tuple[tuple[float, float], ...]
    area_uv: float
    centroid_u: float
    centroid_v: float
    length_uv: float
    is_hole: int
    parent_loop_id: int
    loop_flipped: int


def _nz(x: float) -> float:
    if x == 0.0:
        return 0.0
    return x


def _read_raw_grid(path: Path, nu: int, nv: int) -> _RawGrid:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != nu * nv:
        raise ValueError(f"unexpected row count in {path.name}: {len(rows)}")

    u_grid = [[0.0 for _ in range(nu)] for _ in range(nv)]
    v_grid = [[0.0 for _ in range(nu)] for _ in range(nv)]
    x_grid = [[0.0 for _ in range(nu)] for _ in range(nv)]
    y_grid = [[0.0 for _ in range(nu)] for _ in range(nv)]
    z_grid = [[0.0 for _ in range(nu)] for _ in range(nv)]
    valid_grid = [[0 for _ in range(nu)] for _ in range(nv)]

    seen: set[tuple[int, int]] = set()
    for row in rows:
        iu = int(row["iu"])
        iv = int(row["iv"])
        if (iu < 0) or (iu >= nu) or (iv < 0) or (iv >= nv):
            raise ValueError(f"invalid index in {path.name}: iu={iu}, iv={iv}")
        key = (iu, iv)
        if key in seen:
            raise ValueError(f"duplicate index in {path.name}: iu={iu}, iv={iv}")
        seen.add(key)

        u_grid[iv][iu] = _nz(float(row["u_mm"]))
        v_grid[iv][iu] = _nz(float(row["v_mm"]))
        x_grid[iv][iu] = _nz(float(row["x_mm"]))
        y_grid[iv][iu] = _nz(float(row["y_mm"]))
        z_grid[iv][iu] = _nz(float(row["z_mm"]))
        valid_grid[iv][iu] = int(row["valid"])

    return _RawGrid(
        u_mm=tuple(tuple(r) for r in u_grid),
        v_mm=tuple(tuple(r) for r in v_grid),
        x_mm=tuple(tuple(r) for r in x_grid),
        y_mm=tuple(tuple(r) for r in y_grid),
        z_mm=tuple(tuple(r) for r in z_grid),
        valid=tuple(tuple(r) for r in valid_grid),
    )


def _build_padded_field(valid_grid: tuple[tuple[int, ...], ...]) -> tuple[tuple[float, ...], ...]:
    nv = len(valid_grid)
    nu = len(valid_grid[0]) if nv > 0 else 0
    out: list[tuple[float, ...]] = []
    for pj in range(nv + 2):
        row: list[float] = []
        for pi in range(nu + 2):
            if (1 <= pi <= nu) and (1 <= pj <= nv):
                row.append(1.0 if valid_grid[pj - 1][pi - 1] == 1 else -1.0)
            else:
                row.append(-1.0)
        out.append(tuple(row))
    return tuple(out)


def _bilerp(grid: tuple[tuple[float, ...], ...], u_idx: float, v_idx: float) -> float:
    nv = len(grid)
    nu = len(grid[0]) if nv > 0 else 0
    if (nu < 2) or (nv < 2):
        raise ValueError("nu/nv must be >= 2")

    uc = u_idx
    vc = v_idx
    if uc < 0.0:
        uc = 0.0
    if uc > float(nu - 1):
        uc = float(nu - 1)
    if vc < 0.0:
        vc = 0.0
    if vc > float(nv - 1):
        vc = float(nv - 1)

    i0 = int(math.floor(uc))
    j0 = int(math.floor(vc))
    if i0 > nu - 2:
        i0 = nu - 2
    if j0 > nv - 2:
        j0 = nv - 2

    du = _nz(uc - float(i0))
    dv = _nz(vc - float(j0))

    x00 = grid[j0][i0]
    x10 = grid[j0][i0 + 1]
    x11 = grid[j0 + 1][i0 + 1]
    x01 = grid[j0 + 1][i0]
    return _nz(
        (1.0 - du) * (1.0 - dv) * x00
        + du * (1.0 - dv) * x10
        + du * dv * x11
        + (1.0 - du) * dv * x01
    )


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


def _segments_intersect(
    seg0: tuple[tuple[float, float], tuple[float, float]],
    seg1: tuple[tuple[float, float], tuple[float, float]],
    eps: float,
) -> bool:
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


def _self_intersection_pairs(points: tuple[tuple[float, float], ...], eps: float) -> tuple[tuple[int, int], ...]:
    n = len(points)
    if n < 4:
        return ()
    segs = [(points[i], points[(i + 1) % n]) for i in range(n)]
    hits: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if j == (i + 1) % n:
                continue
            if i == (j + 1) % n:
                continue
            if _segments_intersect(segs[i], segs[j], eps):
                hits.append((i, j))
    return tuple(hits)


def _polygon_geom(points: tuple[tuple[float, float], ...]) -> _LoopGeom:
    n = len(points)
    if n < 3:
        raise ValueError("LOOP_TOO_SHORT")

    a2 = 0.0
    cx_acc = 0.0
    cy_acc = 0.0
    length_uv = 0.0

    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a2 += cross
        cx_acc += (x0 + x1) * cross
        cy_acc += (y0 + y1) * cross
        dx = x1 - x0
        dy = y1 - y0
        length_uv += math.sqrt(dx * dx + dy * dy)

    area = _nz(0.5 * a2)
    if area == 0.0:
        raise ValueError("LOOP_DEGENERATE")

    denom = 6.0 * area
    centroid_u = _nz(cx_acc / denom)
    centroid_v = _nz(cy_acc / denom)
    return _LoopGeom(
        points=points,
        area_uv=area,
        centroid_u=centroid_u,
        centroid_v=centroid_v,
        length_uv=_nz(length_uv),
    )


def _point_in_polygon_strict(point: tuple[float, float], polygon: tuple[tuple[float, float], ...], eps: float) -> bool:
    n = len(polygon)
    x, y = point
    for i in range(n):
        if _on_segment(polygon[i], polygon[(i + 1) % n], point, eps):
            return False

    inside = False
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            t = (y - y0) / (y1 - y0)
            x_cross = x0 + t * (x1 - x0)
            if x_cross > x + eps:
                inside = not inside
    return inside


def _find_test_point(
    geom: _LoopGeom,
    points: tuple[tuple[float, float], ...],
    hole_test_grid_n: int,
    hole_test_grid_min_inside_points: int,
    uv_on_edge_eps: float,
) -> tuple[float, float]:
    centroid = (geom.centroid_u, geom.centroid_v)
    if _point_in_polygon_strict(centroid, points, uv_on_edge_eps):
        return centroid

    us = [p[0] for p in points]
    vs = [p[1] for p in points]
    u_min = min(us)
    u_max = max(us)
    v_min = min(vs)
    v_max = max(vs)

    n = hole_test_grid_n
    if n < 1:
        raise ValueError("hole_test_grid_n must be >= 1")

    inside_points: list[tuple[float, float]] = []
    for iu in range(n):
        if n == 1:
            u = _nz(0.5 * (u_min + u_max))
        else:
            u = _nz(u_min + (u_max - u_min) * float(iu) / float(n - 1))
        for iv in range(n):
            if n == 1:
                v = _nz(0.5 * (v_min + v_max))
            else:
                v = _nz(v_min + (v_max - v_min) * float(iv) / float(n - 1))
            p = (u, v)
            if _point_in_polygon_strict(p, points, uv_on_edge_eps):
                inside_points.append(p)

    if len(inside_points) < hole_test_grid_min_inside_points:
        raise ValueError("HOLE_CLASSIFICATION_AMBIGUOUS")
    return inside_points[0]


def _rotate_start(points: tuple[tuple[float, float], ...], uv_digits: int) -> tuple[tuple[float, float], ...]:
    q_points = [(q(p[0], uv_digits), q(p[1], uv_digits), idx) for idx, p in enumerate(points)]
    _, _, start_idx = min(q_points)
    if start_idx == 0:
        return points
    return points[start_idx:] + points[:start_idx]


def _extract_side_loops(
    *,
    valid_grid: tuple[tuple[int, ...], ...],
    ms_iso_eps: float,
    ms_fc_eps: float,
    ms_eps_d: float,
    ms_tie_break: str,
    sort_round_uv_digits: int,
    uv_on_edge_eps: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    field = _build_padded_field(valid_grid)
    components = marching_squares_zero(
        field=field,
        ms_iso_eps=ms_iso_eps,
        ms_fc_eps=ms_fc_eps,
        ms_eps_d=ms_eps_d,
        ms_tie_break=ms_tie_break,
        sort_round_uv_digits=sort_round_uv_digits,
        uv_on_edge_eps=uv_on_edge_eps,
    )
    if not components:
        return ()

    loops: list[tuple[tuple[float, float], ...]] = []
    for comp in components:
        if not comp.closed:
            raise ValueError("OPEN_POLYLINE")
        shifted = tuple((_nz(p[0] - 1.0), _nz(p[1] - 1.0)) for p in comp.points)
        loops.append(shifted)
    return tuple(loops)


def _classify_loops(
    *,
    loops: tuple[tuple[tuple[float, float], ...], ...],
    sort_round_uv_digits: int,
    sort_round_area_digits: int,
    hole_test_grid_n: int,
    hole_test_grid_min_inside_points: int,
    uv_on_edge_eps: float,
) -> tuple[tuple[_LoopFinal, ...], tuple[dict[str, Any], ...]]:
    if not loops:
        return (), ()

    geoms: list[_LoopGeom] = []
    self_hits: list[tuple[tuple[int, int], ...]] = []
    for lp in loops:
        geoms.append(_polygon_geom(lp))
        self_hits.append(_self_intersection_pairs(lp, uv_on_edge_eps))

    for hits in self_hits:
        if hits:
            raise ValueError("SELF_INTERSECTION")

    test_points: list[tuple[float, float]] = []
    for geom in geoms:
        test_points.append(
            _find_test_point(
                geom=geom,
                points=geom.points,
                hole_test_grid_n=hole_test_grid_n,
                hole_test_grid_min_inside_points=hole_test_grid_min_inside_points,
                uv_on_edge_eps=uv_on_edge_eps,
            )
        )

    parent_src: list[int] = []
    for i, p_test in enumerate(test_points):
        containers: list[int] = []
        for j, geom_j in enumerate(geoms):
            if i == j:
                continue
            if _point_in_polygon_strict(p_test, geom_j.points, uv_on_edge_eps):
                containers.append(j)
        if not containers:
            parent_src.append(-1)
            continue
        parent = min(containers, key=lambda idx: abs(geoms[idx].area_uv))
        parent_src.append(parent)

    depth_cache: dict[int, int] = {}

    def _depth(idx: int) -> int:
        if idx in depth_cache:
            return depth_cache[idx]
        p = parent_src[idx]
        if p == -1:
            depth_cache[idx] = 0
        else:
            depth_cache[idx] = _depth(p) + 1
        return depth_cache[idx]

    features: list[_LoopFeature] = []
    for idx, geom in enumerate(geoms):
        is_hole = 1 if (_depth(idx) % 2 == 1) else 0
        pts = geom.points
        flipped = 0
        if is_hole == 0 and geom.area_uv < 0.0:
            pts = tuple(reversed(pts))
            flipped = 1
        if is_hole == 1 and geom.area_uv > 0.0:
            pts = tuple(reversed(pts))
            flipped = 1
        pts = _rotate_start(pts, sort_round_uv_digits)
        geom_oriented = _polygon_geom(pts)
        features.append(
            _LoopFeature(
                src_idx=idx,
                points=pts,
                area_uv=geom_oriented.area_uv,
                centroid_u=geom_oriented.centroid_u,
                centroid_v=geom_oriented.centroid_v,
                length_uv=geom_oriented.length_uv,
                p_test=test_points[idx],
                parent_src_idx=parent_src[idx],
                depth=_depth(idx),
                is_hole=is_hole,
                loop_flipped=flipped,
            )
        )

    ordered = sorted(
        features,
        key=lambda f: (
            f.is_hole,
            -q(abs(f.area_uv), sort_round_area_digits),
            q(f.centroid_u, sort_round_uv_digits),
            q(f.centroid_v, sort_round_uv_digits),
            -q(f.length_uv, sort_round_uv_digits),
            -len(f.points),
        ),
    )

    loop_id_by_src = {f.src_idx: idx for idx, f in enumerate(ordered)}
    finals: list[_LoopFinal] = []
    loop_stats: list[dict[str, Any]] = []
    for new_id, feature in enumerate(ordered):
        parent_loop_id = -1
        if feature.parent_src_idx != -1:
            parent_loop_id = loop_id_by_src[feature.parent_src_idx]
        finals.append(
            _LoopFinal(
                loop_id=new_id,
                points=feature.points,
                area_uv=feature.area_uv,
                centroid_u=feature.centroid_u,
                centroid_v=feature.centroid_v,
                length_uv=feature.length_uv,
                is_hole=feature.is_hole,
                parent_loop_id=parent_loop_id,
                loop_flipped=feature.loop_flipped,
            )
        )
        loop_stats.append(
            {
                "loop_id": new_id,
                "is_hole": feature.is_hole,
                "parent": parent_loop_id,
                "area_uv": q(feature.area_uv, sort_round_area_digits),
                "centroid": [
                    q(feature.centroid_u, sort_round_uv_digits),
                    q(feature.centroid_v, sort_round_uv_digits),
                ],
                "length_uv": q(feature.length_uv, sort_round_uv_digits),
                "flipped": feature.loop_flipped,
                "self_intersection_ok": True,
            }
        )
    return tuple(finals), tuple(loop_stats)


def _write_boundary_csv(
    *,
    path: Path,
    loops: tuple[_LoopFinal, ...],
    sort_round_uv_digits: int,
    sort_round_area_digits: int,
) -> None:
    lines = [_BOUNDARY_HEADER]
    for loop in loops:
        for point_id, (u_idx, v_idx) in enumerate(loop.points):
            u_q = q(u_idx, sort_round_uv_digits)
            v_q = q(v_idx, sort_round_uv_digits)
            area_q = q(loop.area_uv, sort_round_area_digits)
            cu_q = q(loop.centroid_u, sort_round_uv_digits)
            cv_q = q(loop.centroid_v, sort_round_uv_digits)
            len_q = q(loop.length_uv, sort_round_uv_digits)
            lines.append(
                ",".join(
                    [
                        str(loop.loop_id),
                        str(point_id),
                        str(loop.is_hole),
                        str(loop.parent_loop_id),
                        fixed(u_idx, sort_round_uv_digits),
                        fixed(v_idx, sort_round_uv_digits),
                        fixed(u_q, sort_round_uv_digits),
                        fixed(v_q, sort_round_uv_digits),
                        fixed(loop.area_uv, sort_round_area_digits),
                        fixed(area_q, sort_round_area_digits),
                        fixed(loop.centroid_u, sort_round_uv_digits),
                        fixed(loop.centroid_v, sort_round_uv_digits),
                        fixed(cu_q, sort_round_uv_digits),
                        fixed(cv_q, sort_round_uv_digits),
                        fixed(loop.length_uv, sort_round_uv_digits),
                        fixed(len_q, sort_round_uv_digits),
                        str(loop.loop_flipped),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _wire_points(
    *,
    loops: tuple[_LoopFinal, ...],
    grid: _RawGrid,
) -> tuple[tuple[tuple[tuple[float, float, float], ...], ...], int]:
    wires: list[tuple[tuple[float, float, float], ...]] = []
    total = 0
    for loop in loops:
        pts: list[tuple[float, float, float]] = []
        for u_idx, v_idx in loop.points:
            x = _bilerp(grid.x_mm, u_idx, v_idx)
            y = _bilerp(grid.y_mm, u_idx, v_idx)
            z = _bilerp(grid.z_mm, u_idx, v_idx)
            pts.append((_nz(x), _nz(y), _nz(z)))
        wires.append(tuple(pts))
        total += len(pts)
    return tuple(wires), total


def _step_point(coords: tuple[float, float, float], mm_digits: int) -> str:
    x, y, z = coords
    return f"({fixed(x, mm_digits)},{fixed(y, mm_digits)},{fixed(z, mm_digits)})"


def _write_wire_step(
    *,
    path: Path,
    side: str,
    wires: tuple[tuple[tuple[float, float, float], ...], ...],
    mm_digits: int,
) -> None:
    lines = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('powerskiving flank boundary wire'),'2;1');",
        f"FILE_NAME('{path.name}','',('powerskiving'),('powerskiving'),'','','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));",
        "ENDSEC;",
        "DATA;",
    ]

    eid = 1
    for loop_id, loop in enumerate(wires):
        point_refs: list[int] = []
        for p in loop:
            lines.append(f"#{eid}=CARTESIAN_POINT('',{_step_point(p, mm_digits)});")
            point_refs.append(eid)
            eid += 1
        refs = ",".join(f"#{rid}" for rid in point_refs + [point_refs[0]])
        lines.append(f"#{eid}=POLYLINE('loop_{side}_{loop_id}',({refs}));")
        eid += 1

    lines.extend(["ENDSEC;", "END-ISO-10303-21;"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_step03A(cfg: dict[str, Any], ctx: dict[str, Any]) -> Path:
    """Extract boundary loops, classify them, and export uv/3d wires."""
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / STEP03A_REPORT_NAME

    status = "failed"
    reason_code = "STEP03A_EXCEPTION"
    message = "step03A failed before boundary extraction"
    payload: dict[str, Any] = {
        "loop_count_plus": None,
        "loop_count_minus": None,
        "point_count_plus": None,
        "point_count_minus": None,
        "boundary_plus_csv": None,
        "boundary_minus_csv": None,
        "boundary_wire_plus_step": None,
        "boundary_wire_minus_step": None,
        "surface_uv_boundary_plus": None,
        "surface_uv_boundary_minus": None,
        "self_intersection_pairs_plus": None,
        "self_intersection_pairs_minus": None,
    }

    try:
        nu = int(cfg["Nu"])
        nv = int(cfg["Nv"])
        mm_digits = int(cfg["csv_float_digits_mm"])
        uv_digits = int(cfg["sort_round_uv_digits"])
        area_digits = int(cfg["sort_round_area_digits"])

        plus_raw = output_dir / "tool_conjugate_grid_plus_raw.csv"
        minus_raw = output_dir / "tool_conjugate_grid_minus_raw.csv"
        if not plus_raw.exists() or not minus_raw.exists():
            raise ValueError("STEP02_RAW_MISSING")

        plus_grid = _read_raw_grid(plus_raw, nu, nv)
        minus_grid = _read_raw_grid(minus_raw, nu, nv)

        loops_plus_raw = _extract_side_loops(
            valid_grid=plus_grid.valid,
            ms_iso_eps=float(cfg["ms_iso_eps"]),
            ms_fc_eps=float(cfg["ms_fc_eps"]),
            ms_eps_d=float(cfg["ms_eps_d"]),
            ms_tie_break=str(cfg["ms_tie_break"]),
            sort_round_uv_digits=uv_digits,
            uv_on_edge_eps=float(cfg["uv_on_edge_eps"]),
        )
        loops_minus_raw = _extract_side_loops(
            valid_grid=minus_grid.valid,
            ms_iso_eps=float(cfg["ms_iso_eps"]),
            ms_fc_eps=float(cfg["ms_fc_eps"]),
            ms_eps_d=float(cfg["ms_eps_d"]),
            ms_tie_break=str(cfg["ms_tie_break"]),
            sort_round_uv_digits=uv_digits,
            uv_on_edge_eps=float(cfg["uv_on_edge_eps"]),
        )

        if len(loops_plus_raw) == 0 or len(loops_minus_raw) == 0:
            raise ValueError("LOOP_COUNT_INVALID")

        loops_plus, stats_plus = _classify_loops(
            loops=loops_plus_raw,
            sort_round_uv_digits=uv_digits,
            sort_round_area_digits=area_digits,
            hole_test_grid_n=int(cfg["hole_test_grid_n"]),
            hole_test_grid_min_inside_points=int(cfg["hole_test_grid_min_inside_points"]),
            uv_on_edge_eps=float(cfg["uv_on_edge_eps"]),
        )
        loops_minus, stats_minus = _classify_loops(
            loops=loops_minus_raw,
            sort_round_uv_digits=uv_digits,
            sort_round_area_digits=area_digits,
            hole_test_grid_n=int(cfg["hole_test_grid_n"]),
            hole_test_grid_min_inside_points=int(cfg["hole_test_grid_min_inside_points"]),
            uv_on_edge_eps=float(cfg["uv_on_edge_eps"]),
        )

        plus_out = output_dir / BOUNDARY_PLUS_CSV_NAME
        minus_out = output_dir / BOUNDARY_MINUS_CSV_NAME
        _write_boundary_csv(
            path=plus_out,
            loops=loops_plus,
            sort_round_uv_digits=uv_digits,
            sort_round_area_digits=area_digits,
        )
        _write_boundary_csv(
            path=minus_out,
            loops=loops_minus,
            sort_round_uv_digits=uv_digits,
            sort_round_area_digits=area_digits,
        )

        plus_wires, plus_wire_points = _wire_points(
            loops=loops_plus,
            grid=plus_grid,
        )
        minus_wires, minus_wire_points = _wire_points(
            loops=loops_minus,
            grid=minus_grid,
        )

        plus_step = output_dir / BOUNDARY_WIRE_PLUS_STEP_NAME
        minus_step = output_dir / BOUNDARY_WIRE_MINUS_STEP_NAME
        _write_wire_step(path=plus_step, side="plus", wires=plus_wires, mm_digits=mm_digits)
        _write_wire_step(path=minus_step, side="minus", wires=minus_wires, mm_digits=mm_digits)

        status = "ok"
        reason_code = "OK"
        message = "step03A boundary extraction completed"
        payload = {
            "loop_count_plus": len(loops_plus),
            "loop_count_minus": len(loops_minus),
            "point_count_plus": sum(len(loop.points) for loop in loops_plus),
            "point_count_minus": sum(len(loop.points) for loop in loops_minus),
            "boundary_plus_csv": plus_out.name,
            "boundary_minus_csv": minus_out.name,
            "boundary_wire_plus_step": plus_step.name,
            "boundary_wire_minus_step": minus_step.name,
            "surface_uv_boundary_plus": list(stats_plus),
            "surface_uv_boundary_minus": list(stats_minus),
            "self_intersection_pairs_plus": [],
            "self_intersection_pairs_minus": [],
            "boundary_wire_point_count_plus": plus_wire_points,
            "boundary_wire_point_count_minus": minus_wire_points,
        }
        ctx["boundary_loop_count_plus"] = len(loops_plus)
        ctx["boundary_loop_count_minus"] = len(loops_minus)
    except Exception as exc:  # keep broad: cad_report must be written even on failures
        message = str(exc)
        if message in {
            "OPEN_POLYLINE",
            "SELF_INTERSECTION",
            "LOOP_COUNT_INVALID",
            "STEP02_RAW_MISSING",
            "HOLE_CLASSIFICATION_AMBIGUOUS",
            "BOUNDARY_WIRE_MAP_FAIL",
            "LOOP_TOO_SHORT",
            "LOOP_DEGENERATE",
        }:
            reason_code = message
            status = "reject"

    cad_report = {
        "step_id": "step03A_boundary",
        "status": status,
        "reason_code": reason_code,
        "message": message,
        "exception_stacktrace": None,
        "ctx": dict(ctx),
        **payload,
    }
    write_json(report_path, cad_report)
    return report_path
