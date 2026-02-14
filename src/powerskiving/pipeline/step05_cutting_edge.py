"""Step05 cutting edge extraction (SPEC Step5.0-5.9)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from powerskiving.algorithms import marching_squares_zero
from powerskiving.deterministic import fixed, q, wrap_rad
from powerskiving.json_canon import write_json

STEP05_REPORT_NAME = "cad_report_step05.json"
CUTTING_EDGE_POINTS_CSV_NAME = "cutting_edge_points.csv"
EDGE_BAND_CELLS_CSV_NAME = "edge_band_cells.csv"

_EDGE_HEADER = (
    "edge_side,edge_component_id,selected,point_id,u_idx,v_idx,"
    "x_mm,y_mm,z_mm,nx,ny,nz,tx,ty,tz,s_mm,plane_dist_mm,valid,reason_code"
)
_EDGE_BAND_HEADER = "edge_side,u_cell,v_cell,valid_4corners"


@dataclass(frozen=True)
class _RawGrid:
    x_mm: tuple[tuple[float, ...], ...]
    y_mm: tuple[tuple[float, ...], ...]
    z_mm: tuple[tuple[float, ...], ...]
    nx: tuple[tuple[float, ...], ...]
    ny: tuple[tuple[float, ...], ...]
    nz: tuple[tuple[float, ...], ...]
    valid: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _EdgePoint:
    u_idx: float
    v_idx: float
    x_mm: float
    y_mm: float
    z_mm: float
    nx: float
    ny: float
    nz: float
    tx: float
    ty: float
    tz: float
    s_mm: float
    plane_dist_mm: float
    valid: int
    reason_code: str


@dataclass(frozen=True)
class _EdgeComponent:
    side: str
    component_id: int
    selected: int
    points: tuple[_EdgePoint, ...]
    in_sector: int
    theta_comp_q: float
    abs_delta_q: float
    length_q: float
    n_points: int
    length_3d_mm: float


def _nz(x: float) -> float:
    if x == 0.0:
        return 0.0
    return x


def _norm3(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize3(v: tuple[float, float, float], eps: float = 1.0e-12) -> tuple[float, float, float] | None:
    n = _norm3(v)
    if (not math.isfinite(n)) or (n <= eps):
        return None
    inv = 1.0 / n
    out = (_nz(v[0] * inv), _nz(v[1] * inv), _nz(v[2] * inv))
    if not (math.isfinite(out[0]) and math.isfinite(out[1]) and math.isfinite(out[2])):
        return None
    return out


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return _nz(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        _nz(a[1] * b[2] - a[2] * b[1]),
        _nz(a[2] * b[0] - a[0] * b[2]),
        _nz(a[0] * b[1] - a[1] * b[0]),
    )


def _sub3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return _nz(a[0] - b[0]), _nz(a[1] - b[1]), _nz(a[2] - b[2])


def _read_raw_grid(path: Path, nu: int, nv: int) -> _RawGrid:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != nu * nv:
        raise ValueError(f"unexpected row count in {path.name}: {len(rows)}")

    x = [[0.0 for _ in range(nu)] for _ in range(nv)]
    y = [[0.0 for _ in range(nu)] for _ in range(nv)]
    z = [[0.0 for _ in range(nu)] for _ in range(nv)]
    nx = [[0.0 for _ in range(nu)] for _ in range(nv)]
    ny = [[0.0 for _ in range(nu)] for _ in range(nv)]
    nz = [[0.0 for _ in range(nu)] for _ in range(nv)]
    valid = [[0 for _ in range(nu)] for _ in range(nv)]

    seen: set[tuple[int, int]] = set()
    for row in rows:
        iu = int(row["iu"])
        iv = int(row["iv"])
        if (iu < 0) or (iu >= nu) or (iv < 0) or (iv >= nv):
            raise ValueError(f"invalid index in {path.name}: iu={iu}, iv={iv}")
        if (iu, iv) in seen:
            raise ValueError(f"duplicate index in {path.name}: iu={iu}, iv={iv}")
        seen.add((iu, iv))

        x[iv][iu] = _nz(float(row["x_mm"]))
        y[iv][iu] = _nz(float(row["y_mm"]))
        z[iv][iu] = _nz(float(row["z_mm"]))
        nx[iv][iu] = _nz(float(row["nx"]))
        ny[iv][iu] = _nz(float(row["ny"]))
        nz[iv][iu] = _nz(float(row["nz"]))
        valid[iv][iu] = int(row["valid"])

    return _RawGrid(
        x_mm=tuple(tuple(r) for r in x),
        y_mm=tuple(tuple(r) for r in y),
        z_mm=tuple(tuple(r) for r in z),
        nx=tuple(tuple(r) for r in nx),
        ny=tuple(tuple(r) for r in ny),
        nz=tuple(tuple(r) for r in nz),
        valid=tuple(tuple(r) for r in valid),
    )


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

    f00 = grid[j0][i0]
    f10 = grid[j0][i0 + 1]
    f11 = grid[j0 + 1][i0 + 1]
    f01 = grid[j0 + 1][i0]
    return _nz(
        (1.0 - du) * (1.0 - dv) * f00
        + du * (1.0 - dv) * f10
        + du * dv * f11
        + (1.0 - du) * dv * f01
    )


def _point_from_uv(grid: _RawGrid, u_idx: float, v_idx: float) -> tuple[float, float, float]:
    return (
        _bilerp(grid.x_mm, u_idx, v_idx),
        _bilerp(grid.y_mm, u_idx, v_idx),
        _bilerp(grid.z_mm, u_idx, v_idx),
    )


def _normal_from_uv(grid: _RawGrid, u_idx: float, v_idx: float) -> tuple[float, float, float] | None:
    n = _normalize3(
        (
            _bilerp(grid.nx, u_idx, v_idx),
            _bilerp(grid.ny, u_idx, v_idx),
            _bilerp(grid.nz, u_idx, v_idx),
        )
    )
    if n is None:
        return None

    p = _point_from_uv(grid, u_idx, v_idx)
    radial = _normalize3((p[0], p[1], 0.0))
    if radial is not None and _dot3(n, radial) < 0.0:
        n = (-n[0], -n[1], -n[2])
    return n


def _compute_rake_plane(cfg: dict[str, Any], plus_grid: _RawGrid) -> tuple[tuple[float, float, float], float, tuple[float, float, float]]:
    ref_mode = str(cfg["ref_point_mode"])
    if ref_mode == "manual_xyz":
        p_ref = (_nz(float(cfg["x_ref_mm"])), _nz(float(cfg["y_ref_mm"])), _nz(float(cfg["z_ref_mm"])))
    elif ref_mode == "grid_uv":
        iu = int(cfg["iu_ref"])
        iv = int(cfg["iv_ref"])
        nv = len(plus_grid.x_mm)
        nu = len(plus_grid.x_mm[0]) if nv > 0 else 0
        if (iu < 0) or (iu >= nu) or (iv < 0) or (iv >= nv):
            raise ValueError("RAKE_REF_INDEX_OUT_OF_RANGE")
        p_ref = (plus_grid.x_mm[iv][iu], plus_grid.y_mm[iv][iu], plus_grid.z_mm[iv][iu])
    else:
        raise ValueError("RAKE_REF_MODE_INVALID")

    theta_ref = wrap_rad(math.atan2(p_ref[1], p_ref[0]))
    e_r = (math.cos(theta_ref), math.sin(theta_ref), 0.0)
    e_t = (-math.sin(theta_ref), math.cos(theta_ref), 0.0)
    gamma = math.radians(float(cfg["rake_angle_deg"]))
    n_rake = _normalize3(
        (
            _nz(math.cos(gamma) * e_t[0] - math.sin(gamma) * e_r[0]),
            _nz(math.cos(gamma) * e_t[1] - math.sin(gamma) * e_r[1]),
            _nz(math.cos(gamma) * e_t[2] - math.sin(gamma) * e_r[2]),
        )
    )
    if n_rake is None:
        raise ValueError("INVALID_RAKE_NORMAL")
    return p_ref, theta_ref, n_rake


def _plane_dist_field(
    grid: _RawGrid,
    n_rake: tuple[float, float, float],
    p_ref: tuple[float, float, float],
) -> tuple[tuple[float, ...], ...]:
    out: list[tuple[float, ...]] = []
    for iv in range(len(grid.x_mm)):
        row: list[float] = []
        for iu in range(len(grid.x_mm[iv])):
            p = (grid.x_mm[iv][iu], grid.y_mm[iv][iu], grid.z_mm[iv][iu])
            d = _dot3(n_rake, _sub3(p, p_ref))
            row.append(_nz(d))
        out.append(tuple(row))
    return tuple(out)


def _cell_mask(valid: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    nv = len(valid)
    nu = len(valid[0]) if nv > 0 else 0
    out: list[tuple[int, ...]] = []
    for iv in range(nv - 1):
        row: list[int] = []
        for iu in range(nu - 1):
            cell_ok = (
                valid[iv][iu] == 1
                and valid[iv][iu + 1] == 1
                and valid[iv + 1][iu + 1] == 1
                and valid[iv + 1][iu] == 1
            )
            row.append(1 if cell_ok else 0)
        out.append(tuple(row))
    return tuple(out)


def _theta_comp(points: list[tuple[float, float, float]]) -> float:
    if not points:
        return 0.0
    thetas = [wrap_rad(math.atan2(p[1], p[0])) for p in points]
    mx = sum(math.cos(t) for t in thetas) / float(len(thetas))
    my = sum(math.sin(t) for t in thetas) / float(len(thetas))
    if (mx * mx + my * my) < 1.0e-12:
        return thetas[0]
    return wrap_rad(math.atan2(my, mx))


def _polyline_length(points: list[tuple[float, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    acc = 0.0
    for i in range(1, len(points)):
        seg = _sub3(points[i], points[i - 1])
        acc = _nz(acc + _norm3(seg))
    return _nz(acc)


def _resample_targets(total_length: float, chord_tol: float) -> list[float]:
    if total_length <= 0.0:
        return [0.0, 0.0]
    n = max(2, int(math.ceil(total_length / chord_tol)) + 1)
    if n <= 1:
        n = 2
    return [_nz(float(k) * total_length / float(n - 1)) for k in range(n)]


def _interpolate_along(s_targets: list[float], s_nodes: list[float], vals: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    out: list[tuple[float, ...]] = []
    if len(vals) == 1:
        return [vals[0] for _ in s_targets]

    seg = 0
    for st in s_targets:
        while seg < len(s_nodes) - 2 and s_nodes[seg + 1] < st:
            seg += 1
        s0 = s_nodes[seg]
        s1 = s_nodes[seg + 1]
        if s1 <= s0:
            t = 0.0
        else:
            t = (st - s0) / (s1 - s0)
        v0 = vals[seg]
        v1 = vals[seg + 1]
        out.append(tuple(_nz((1.0 - t) * v0[d] + t * v1[d]) for d in range(len(v0))))
    return out


def _compute_tangent_and_s(
    *,
    n_rake: tuple[float, float, float],
    uvs: list[tuple[float, float]],
    points: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float] | None],
    z_span_min_mm: float,
    t_cross_min: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]], list[tuple[float, float, float] | None], list[tuple[float, float, float]], list[float], list[int], list[str]]:
    if not points:
        return [], [], [], [], [], [], []

    out_uvs = list(uvs)
    out_points = list(points)
    out_normals = list(normals)

    dz = _nz(out_points[-1][2] - out_points[0][2])
    theta0 = wrap_rad(math.atan2(out_points[0][1], out_points[0][0]))
    theta1 = wrap_rad(math.atan2(out_points[-1][1], out_points[-1][0]))
    dtheta = wrap_rad(theta1 - theta0)

    flip = False
    if abs(dz) >= z_span_min_mm:
        if dz < 0.0:
            flip = True
    elif dtheta < 0.0:
        flip = True

    if flip:
        out_uvs.reverse()
        out_points.reverse()
        out_normals.reverse()

    t_raw: list[tuple[float, float, float]] = []
    valid: list[int] = []
    reason: list[str] = []
    for n in out_normals:
        if n is None:
            t_raw.append((0.0, 0.0, 0.0))
            valid.append(0)
            reason.append("INVALID_NORMAL")
            continue
        c = _cross3(n_rake, n)
        c_norm = _norm3(c)
        if c_norm < t_cross_min:
            t_raw.append((0.0, 0.0, 0.0))
            valid.append(0)
            reason.append("DEGENERATE_CROSS")
            continue
        t = _normalize3(c)
        if t is None:
            t_raw.append((0.0, 0.0, 0.0))
            valid.append(0)
            reason.append("DEGENERATE_CROSS")
            continue
        t_raw.append(t)
        valid.append(1)
        reason.append("OK")

    t_poly: list[tuple[float, float, float]] = []
    npts = len(out_points)
    for i in range(npts):
        if npts == 1:
            t_poly.append((0.0, 0.0, 0.0))
            continue
        if i == 0:
            d = _sub3(out_points[1], out_points[0])
        elif i == npts - 1:
            d = _sub3(out_points[-1], out_points[-2])
        else:
            d = _sub3(out_points[i + 1], out_points[i - 1])
        td = _normalize3(d)
        t_poly.append((0.0, 0.0, 0.0) if td is None else td)

    score = 0.0
    for i in range(npts):
        if valid[i] == 1:
            score = _nz(score + _dot3(t_raw[i], t_poly[i]))

    t_sign = -1.0 if score < 0.0 else 1.0
    t_final = [(_nz(t_sign * t[0]), _nz(t_sign * t[1]), _nz(t_sign * t[2])) for t in t_raw]

    s_mm = [0.0]
    for i in range(1, npts):
        seg = _norm3(_sub3(out_points[i], out_points[i - 1]))
        s_mm.append(_nz(s_mm[-1] + seg))

    return out_uvs, out_points, out_normals, t_final, s_mm, valid, reason


def _build_component_points(
    *,
    grid: _RawGrid,
    uvs: list[tuple[float, float]],
    n_rake: tuple[float, float, float],
    p_ref: tuple[float, float, float],
    z_span_min_mm: float,
    t_cross_min: float,
) -> list[_EdgePoint]:
    points = [_point_from_uv(grid, uv[0], uv[1]) for uv in uvs]
    normals = [_normal_from_uv(grid, uv[0], uv[1]) for uv in uvs]

    out_uvs, out_points, out_normals, tangents, s_mm, valid, reasons = _compute_tangent_and_s(
        n_rake=n_rake,
        uvs=uvs,
        points=points,
        normals=normals,
        z_span_min_mm=z_span_min_mm,
        t_cross_min=t_cross_min,
    )

    out: list[_EdgePoint] = []
    for i, uv in enumerate(out_uvs):
        p = out_points[i]
        n = out_normals[i]
        if n is None:
            n = (0.0, 0.0, 0.0)
        d = _dot3(n_rake, _sub3(p, p_ref))
        out.append(
            _EdgePoint(
                u_idx=uv[0],
                v_idx=uv[1],
                x_mm=p[0],
                y_mm=p[1],
                z_mm=p[2],
                nx=n[0],
                ny=n[1],
                nz=n[2],
                tx=tangents[i][0],
                ty=tangents[i][1],
                tz=tangents[i][2],
                s_mm=s_mm[i],
                plane_dist_mm=d,
                valid=valid[i],
                reason_code=reasons[i],
            )
        )
    return out


def _resample_selected(
    *,
    grid: _RawGrid,
    component_points: list[_EdgePoint],
    n_rake: tuple[float, float, float],
    p_ref: tuple[float, float, float],
    edge_chord_tol_mm: float,
    z_span_min_mm: float,
    t_cross_min: float,
) -> list[_EdgePoint]:
    valid_pts = [p for p in component_points if p.valid == 1]
    if len(valid_pts) < 2:
        return component_points

    src_p = [(p.x_mm, p.y_mm, p.z_mm) for p in valid_pts]
    src_uv = [(p.u_idx, p.v_idx) for p in valid_pts]

    s_nodes = [0.0]
    for i in range(1, len(src_p)):
        s_nodes.append(_nz(s_nodes[-1] + _norm3(_sub3(src_p[i], src_p[i - 1]))))
    total_length = s_nodes[-1]
    if total_length <= 0.0:
        return component_points

    s_targets = _resample_targets(total_length, edge_chord_tol_mm)
    p_targets = _interpolate_along(s_targets, s_nodes, [tuple(v) for v in src_p])
    uv_targets = _interpolate_along(s_targets, s_nodes, [tuple(v) for v in src_uv])

    uvs = [(p[0], p[1]) for p in uv_targets]
    points = [(p[0], p[1], p[2]) for p in p_targets]
    normals = [_normal_from_uv(grid, uv[0], uv[1]) for uv in uvs]

    out_uvs, out_points, out_normals, tangents, s_mm, valid, reasons = _compute_tangent_and_s(
        n_rake=n_rake,
        uvs=uvs,
        points=points,
        normals=normals,
        z_span_min_mm=z_span_min_mm,
        t_cross_min=t_cross_min,
    )

    out: list[_EdgePoint] = []
    for i, uv in enumerate(out_uvs):
        p = out_points[i]
        n = out_normals[i]
        if n is None:
            n = (0.0, 0.0, 0.0)
        d = _dot3(n_rake, _sub3(p, p_ref))
        out.append(
            _EdgePoint(
                u_idx=uv[0],
                v_idx=uv[1],
                x_mm=p[0],
                y_mm=p[1],
                z_mm=p[2],
                nx=n[0],
                ny=n[1],
                nz=n[2],
                tx=tangents[i][0],
                ty=tangents[i][1],
                tz=tangents[i][2],
                s_mm=s_mm[i],
                plane_dist_mm=d,
                valid=valid[i],
                reason_code=reasons[i],
            )
        )
    return out


def _write_edge_csv(
    *,
    path: Path,
    components: list[_EdgeComponent],
    uv_digits: int,
    mm_digits: int,
    unitless_digits: int,
) -> None:
    lines = [_EDGE_HEADER]
    ordered = sorted(
        components,
        key=lambda c: (
            0 if c.side == "plus" else 1,
            c.component_id,
        ),
    )
    for comp in ordered:
        for point_id, p in enumerate(comp.points):
            lines.append(
                ",".join(
                    [
                        comp.side,
                        str(comp.component_id),
                        str(comp.selected),
                        str(point_id),
                        fixed(q(p.u_idx, uv_digits), uv_digits),
                        fixed(q(p.v_idx, uv_digits), uv_digits),
                        fixed(q(p.x_mm, mm_digits), mm_digits),
                        fixed(q(p.y_mm, mm_digits), mm_digits),
                        fixed(q(p.z_mm, mm_digits), mm_digits),
                        fixed(q(p.nx, unitless_digits), unitless_digits),
                        fixed(q(p.ny, unitless_digits), unitless_digits),
                        fixed(q(p.nz, unitless_digits), unitless_digits),
                        fixed(q(p.tx, unitless_digits), unitless_digits),
                        fixed(q(p.ty, unitless_digits), unitless_digits),
                        fixed(q(p.tz, unitless_digits), unitless_digits),
                        fixed(q(p.s_mm, mm_digits), mm_digits),
                        fixed(q(p.plane_dist_mm, mm_digits), mm_digits),
                        str(p.valid),
                        p.reason_code,
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _manhattan_band_cells(
    *,
    points: tuple[_EdgePoint, ...],
    band_width_cells: int,
    nu: int,
    nv: int,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    max_u = nu - 2
    max_v = nv - 2
    for point in points:
        base_u = int(math.floor(point.u_idx))
        base_v = int(math.floor(point.v_idx))
        for du in range(-band_width_cells, band_width_cells + 1):
            for dv in range(-band_width_cells, band_width_cells + 1):
                if abs(du) + abs(dv) > band_width_cells:
                    continue
                iu = base_u + du
                iv = base_v + dv
                if (0 <= iu <= max_u) and (0 <= iv <= max_v):
                    out.add((iu, iv))
    return out


def _cell_valid_4corners(valid_grid: tuple[tuple[int, ...], ...], iu: int, iv: int) -> int:
    return (
        1
        if (
            valid_grid[iv][iu] == 1
            and valid_grid[iv][iu + 1] == 1
            and valid_grid[iv + 1][iu + 1] == 1
            and valid_grid[iv + 1][iu] == 1
        )
        else 0
    )


def _write_edge_band_csv(
    *,
    path: Path,
    rows: list[tuple[str, int, int, int]],
) -> None:
    lines = [_EDGE_BAND_HEADER]
    for row in rows:
        lines.append(f"{row[0]},{row[1]},{row[2]},{row[3]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _compute_od(
    *,
    plus_grid: _RawGrid,
    minus_grid: _RawGrid,
    selected_plus: _EdgeComponent,
    selected_minus: _EdgeComponent,
    used_z_pad_mm: float,
    r_od_margin_mm: float,
    r_od_pad_mm: float,
) -> dict[str, Any]:
    all_edge_points = list(selected_plus.points) + list(selected_minus.points)
    tool_z_min = min(p.z_mm for p in all_edge_points)
    tool_z_max = max(p.z_mm for p in all_edge_points)
    used_z_min = _nz(tool_z_min - used_z_pad_mm)
    used_z_max = _nz(tool_z_max + used_z_pad_mm)

    def _grid_r_max_used(grid: _RawGrid) -> float:
        r_max = 0.0
        nv = len(grid.z_mm)
        nu = len(grid.z_mm[0]) if nv > 0 else 0
        for iv in range(nv):
            for iu in range(nu):
                if grid.valid[iv][iu] != 1:
                    continue
                z = grid.z_mm[iv][iu]
                if z < used_z_min or z > used_z_max:
                    continue
                r = math.sqrt(grid.x_mm[iv][iu] * grid.x_mm[iv][iu] + grid.y_mm[iv][iu] * grid.y_mm[iv][iu])
                if r > r_max:
                    r_max = r
        return _nz(r_max)

    r_grid_max_used = max(_grid_r_max_used(plus_grid), _grid_r_max_used(minus_grid))
    r_edge_max = _nz(
        max(
            math.sqrt(p.x_mm * p.x_mm + p.y_mm * p.y_mm)
            for p in all_edge_points
        )
    )
    r_od = _nz(max(r_grid_max_used, r_edge_max) + r_od_margin_mm + r_od_pad_mm)
    return {
        "tool_z_min": _nz(tool_z_min),
        "tool_z_max": _nz(tool_z_max),
        "used_z_range": [_nz(used_z_min), _nz(used_z_max)],
        "r_grid_max_used": r_grid_max_used,
        "r_edge_max": r_edge_max,
        "r_od": r_od,
    }


def _build_side_components(
    *,
    side: str,
    grid: _RawGrid,
    cfg: dict[str, Any],
    p_ref: tuple[float, float, float],
    theta_ref: float,
    n_rake: tuple[float, float, float],
) -> tuple[list[_EdgeComponent], dict[str, Any]]:
    field = _plane_dist_field(grid, n_rake, p_ref)
    cell_valid_mask = _cell_mask(grid.valid)
    components = marching_squares_zero(
        field=field,
        ms_iso_eps=float(cfg["ms_iso_eps"]),
        ms_fc_eps=float(cfg["ms_fc_eps"]),
        ms_eps_d=float(cfg["ms_eps_d"]),
        ms_tie_break=str(cfg["ms_tie_break"]),
        sort_round_uv_digits=int(cfg["sort_round_uv_digits"]),
        uv_on_edge_eps=float(cfg["uv_on_edge_eps"]),
        cell_valid_mask=cell_valid_mask,
    )

    rad_digits = int(cfg["sort_round_rad_digits"])
    mm_digits = int(cfg["sort_round_mm_digits"])

    half_sector = math.pi / float(int(cfg["z1"]))
    margin = math.radians(float(cfg["theta_sector_margin_deg"]))
    theta_center = float(cfg["theta_tooth_center_rad"])

    built: list[dict[str, Any]] = []
    for idx, comp in enumerate(components):
        uvs = [(p[0], p[1]) for p in comp.points]
        points = _build_component_points(
            grid=grid,
            uvs=uvs,
            n_rake=n_rake,
            p_ref=p_ref,
            z_span_min_mm=float(cfg["z_span_min_mm"]),
            t_cross_min=float(cfg["t_cross_min"]),
        )
        p3 = [(p.x_mm, p.y_mm, p.z_mm) for p in points]
        theta_comp = _theta_comp(p3)
        theta_comp_q = q(theta_comp, rad_digits)
        abs_delta = abs(wrap_rad(theta_comp - theta_ref))
        abs_delta_q = q(abs_delta, rad_digits)
        length_3d = _polyline_length(p3)
        length_q = q(length_3d, mm_digits)
        in_sector = 1 if abs(wrap_rad(theta_comp - theta_center)) <= (half_sector + margin) else 0

        built.append(
            {
                "idx": idx,
                "points": points,
                "in_sector": in_sector,
                "theta_comp_q": theta_comp_q,
                "abs_delta_q": abs_delta_q,
                "length_q": length_q,
                "n_points": len(points),
                "length_3d_mm": length_3d,
            }
        )

    ordered = sorted(
        built,
        key=lambda c: (
            -c["in_sector"],
            c["abs_delta_q"],
            -c["length_q"],
            c["theta_comp_q"],
            -c["n_points"],
        ),
    )

    selected_id: int | None = None
    in_sector_only = [c for c in ordered if c["in_sector"] == 1]
    if in_sector_only:
        best = sorted(
            in_sector_only,
            key=lambda c: (
                c["abs_delta_q"],
                -c["length_q"],
                c["theta_comp_q"],
                -c["n_points"],
            ),
        )[0]
        selected_id = ordered.index(best)

    final: list[_EdgeComponent] = []
    candidates_report: list[dict[str, Any]] = []
    for comp_id, item in enumerate(ordered):
        pts = list(item["points"])
        selected = 1 if (selected_id is not None and comp_id == selected_id) else 0
        if selected == 1:
            pts = _resample_selected(
                grid=grid,
                component_points=pts,
                n_rake=n_rake,
                p_ref=p_ref,
                edge_chord_tol_mm=float(cfg["edge_chord_tol_mm"]),
                z_span_min_mm=float(cfg["z_span_min_mm"]),
                t_cross_min=float(cfg["t_cross_min"]),
            )
            item_len = _polyline_length([(p.x_mm, p.y_mm, p.z_mm) for p in pts])
            item["length_3d_mm"] = item_len
            item["length_q"] = q(item_len, mm_digits)

        final.append(
            _EdgeComponent(
                side=side,
                component_id=comp_id,
                selected=selected,
                points=tuple(pts),
                in_sector=item["in_sector"],
                theta_comp_q=item["theta_comp_q"],
                abs_delta_q=item["abs_delta_q"],
                length_q=item["length_q"],
                n_points=item["n_points"],
                length_3d_mm=item["length_3d_mm"],
            )
        )
        candidates_report.append(
            {
                "edge_component_id": comp_id,
                "in_sector": item["in_sector"],
                "theta_comp_q": item["theta_comp_q"],
                "abs_delta_q": item["abs_delta_q"],
                "length_q": item["length_q"],
                "n_points": item["n_points"],
                "selected": selected,
            }
        )

    selected_reason = "NO_IN_SECTOR_CANDIDATE"
    if selected_id is not None:
        selected_reason = "MIN_ABS_DELTA_MAX_LENGTH"

    meta = {
        "component_count": len(final),
        "selected_reason": selected_reason,
        "candidates": candidates_report,
    }
    return final, meta


def run_step05(cfg: dict[str, Any], ctx: dict[str, Any]) -> Path:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / STEP05_REPORT_NAME

    status = "failed"
    reason_code = "STEP05_EXCEPTION"
    message = "step05 failed before edge extraction"
    payload: dict[str, Any] = {
        "cutting_edge_points_csv": None,
        "edge_band_cells_csv": None,
        "rake_ref_point_T": None,
        "rake_angle_deg": float(cfg["rake_angle_deg"]),
        "theta_ref_rad": None,
        "n_rake": None,
        "edge_plus_ok": None,
        "edge_minus_ok": None,
        "edges_ok": None,
        "selected_plus_length_mm": None,
        "selected_minus_length_mm": None,
        "edge_band": None,
        "tool_z_min": None,
        "tool_z_max": None,
        "od": None,
        "selection_plus": None,
        "selection_minus": None,
    }

    try:
        nu = int(cfg["Nu"])
        nv = int(cfg["Nv"])

        plus_raw = output_dir / "tool_conjugate_grid_plus_raw.csv"
        minus_raw = output_dir / "tool_conjugate_grid_minus_raw.csv"
        if not plus_raw.exists() or not minus_raw.exists():
            raise ValueError("STEP02_RAW_MISSING")

        plus_grid = _read_raw_grid(plus_raw, nu, nv)
        minus_grid = _read_raw_grid(minus_raw, nu, nv)

        p_ref, theta_ref, n_rake = _compute_rake_plane(cfg, plus_grid)

        plus_components, plus_meta = _build_side_components(
            side="plus",
            grid=plus_grid,
            cfg=cfg,
            p_ref=p_ref,
            theta_ref=theta_ref,
            n_rake=n_rake,
        )
        minus_components, minus_meta = _build_side_components(
            side="minus",
            grid=minus_grid,
            cfg=cfg,
            p_ref=p_ref,
            theta_ref=theta_ref,
            n_rake=n_rake,
        )

        selected_plus = next((c for c in plus_components if c.selected == 1), None)
        selected_minus = next((c for c in minus_components if c.selected == 1), None)
        selected_plus_length_mm = 0.0 if selected_plus is None else selected_plus.length_3d_mm
        selected_minus_length_mm = 0.0 if selected_minus is None else selected_minus.length_3d_mm

        edge_min_len = float(cfg["edge_min_length_mm"])
        edge_plus_ok = (selected_plus is not None) and (selected_plus_length_mm >= edge_min_len)
        edge_minus_ok = (selected_minus is not None) and (selected_minus_length_mm >= edge_min_len)
        edges_ok = edge_plus_ok and edge_minus_ok

        csv_path = output_dir / CUTTING_EDGE_POINTS_CSV_NAME
        all_components = plus_components + minus_components
        _write_edge_csv(
            path=csv_path,
            components=all_components,
            uv_digits=int(cfg["sort_round_uv_digits"]),
            mm_digits=int(cfg["csv_float_digits_mm"]),
            unitless_digits=int(cfg["csv_float_digits_unitless"]),
        )

        edge_band_payload: dict[str, Any]
        tool_z_min: float | None = None
        tool_z_max: float | None = None
        od_payload: dict[str, Any] | None = None
        edge_band_cells_csv: str | None = None
        if edges_ok:
            assert selected_plus is not None
            assert selected_minus is not None
            nu = int(cfg["Nu"])
            nv = int(cfg["Nv"])
            band_width_cells = int(cfg["edge_band_width_cells"])
            plus_cells = _manhattan_band_cells(
                points=selected_plus.points,
                band_width_cells=band_width_cells,
                nu=nu,
                nv=nv,
            )
            minus_cells = _manhattan_band_cells(
                points=selected_minus.points,
                band_width_cells=band_width_cells,
                nu=nu,
                nv=nv,
            )

            band_rows: list[tuple[str, int, int, int]] = []
            band_valid = True
            for iu, iv in sorted(plus_cells, key=lambda c: (c[1], c[0])):
                ok = _cell_valid_4corners(plus_grid.valid, iu, iv)
                band_rows.append(("plus", iu, iv, ok))
                if ok != 1:
                    band_valid = False
            for iu, iv in sorted(minus_cells, key=lambda c: (c[1], c[0])):
                ok = _cell_valid_4corners(minus_grid.valid, iu, iv)
                band_rows.append(("minus", iu, iv, ok))
                if ok != 1:
                    band_valid = False

            edge_band_path = output_dir / EDGE_BAND_CELLS_CSV_NAME
            _write_edge_band_csv(path=edge_band_path, rows=band_rows)
            edge_band_cells_csv = edge_band_path.name
            edge_band_payload = {
                "status": "ok" if band_valid else "reject",
                "band_width_cells": band_width_cells,
                "plus_cell_count": len(plus_cells),
                "minus_cell_count": len(minus_cells),
                "invalid_cell_count": sum(1 for row in band_rows if row[3] != 1),
                "skipped_reason": None,
            }
            if not band_valid:
                status = "reject"
                reason_code = "EDGE_BAND_INVALID_CELL"
                message = "step05 edge_band has invalid 4-corner cells"
            else:
                od_payload = _compute_od(
                    plus_grid=plus_grid,
                    minus_grid=minus_grid,
                    selected_plus=selected_plus,
                    selected_minus=selected_minus,
                    used_z_pad_mm=float(cfg["used_z_pad_mm"]),
                    r_od_margin_mm=float(cfg["r_od_margin_mm"]),
                    r_od_pad_mm=float(cfg["r_od_pad_mm"]),
                )
                tool_z_min = od_payload["tool_z_min"]
                tool_z_max = od_payload["tool_z_max"]
        else:
            edge_band_payload = {
                "status": "skipped",
                "band_width_cells": int(cfg["edge_band_width_cells"]),
                "plus_cell_count": 0,
                "minus_cell_count": 0,
                "invalid_cell_count": 0,
                "skipped_reason": "EDGES_NOT_OK",
            }

        if edges_ok:
            if od_payload is None:
                status = "reject"
                reason_code = "EDGE_BAND_INVALID_CELL"
                message = "step05 edge_band has invalid 4-corner cells"
            else:
                status = "ok"
                reason_code = "OK"
                message = "step05 cutting edge extraction completed"
        else:
            status = "ok"
            reason_code = "EDGES_NOT_OK"
            message = "step05 cutting edge extraction completed"
        payload = {
            "cutting_edge_points_csv": csv_path.name,
            "edge_band_cells_csv": edge_band_cells_csv,
            "rake_ref_point_T": [p_ref[0], p_ref[1], p_ref[2]],
            "rake_angle_deg": float(cfg["rake_angle_deg"]),
            "theta_ref_rad": theta_ref,
            "n_rake": [n_rake[0], n_rake[1], n_rake[2]],
            "edge_plus_ok": edge_plus_ok,
            "edge_minus_ok": edge_minus_ok,
            "edges_ok": edges_ok,
            "selected_plus_length_mm": selected_plus_length_mm,
            "selected_minus_length_mm": selected_minus_length_mm,
            "edge_band": edge_band_payload,
            "tool_z_min": tool_z_min,
            "tool_z_max": tool_z_max,
            "od": od_payload,
            "selection_plus": plus_meta,
            "selection_minus": minus_meta,
        }

        ctx["edges_ok"] = edges_ok
        ctx["edge_plus_ok"] = edge_plus_ok
        ctx["edge_minus_ok"] = edge_minus_ok
        ctx["selected_plus_length_mm"] = selected_plus_length_mm
        ctx["selected_minus_length_mm"] = selected_minus_length_mm
        ctx["tool_z_min"] = tool_z_min
        ctx["tool_z_max"] = tool_z_max
        if od_payload is not None:
            ctx["r_od"] = od_payload["r_od"]
    except Exception as exc:
        message = str(exc)
        if message in {
            "STEP02_RAW_MISSING",
            "RAKE_REF_MODE_INVALID",
            "RAKE_REF_INDEX_OUT_OF_RANGE",
            "INVALID_RAKE_NORMAL",
        }:
            status = "reject"
            reason_code = message

    cad_report = {
        "step_id": "step05_cutting_edge",
        "status": status,
        "reason_code": reason_code,
        "message": message,
        "exception_stacktrace": None,
        "ctx": dict(ctx),
        **payload,
    }
    write_json(report_path, cad_report)
    return report_path
