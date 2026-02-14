"""Step03B view mesh export (ASCII STL, deterministic)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from powerskiving.deterministic import fixed
from powerskiving.json_canon import write_json

STEP03B_REPORT_NAME = "cad_report_step03B.json"
VIEW_MESH_PLUS_STL_NAME = "flank_view_mesh_plus.stl"
VIEW_MESH_MINUS_STL_NAME = "flank_view_mesh_minus.stl"


@dataclass(frozen=True)
class _RawGrid:
    u_mm: tuple[tuple[float, ...], ...]
    v_mm: tuple[tuple[float, ...], ...]
    x_mm: tuple[tuple[float, ...], ...]
    y_mm: tuple[tuple[float, ...], ...]
    z_mm: tuple[tuple[float, ...], ...]
    nx: tuple[tuple[float, ...], ...]
    ny: tuple[tuple[float, ...], ...]
    nz: tuple[tuple[float, ...], ...]
    valid: tuple[tuple[int, ...], ...]


def _nz(x: float) -> float:
    if x == 0.0:
        return 0.0
    return x


def _finite3(p: tuple[float, float, float]) -> bool:
    return math.isfinite(p[0]) and math.isfinite(p[1]) and math.isfinite(p[2])


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return _nz(a[0] - b[0]), _nz(a[1] - b[1]), _nz(a[2] - b[2])


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        _nz(a[1] * b[2] - a[2] * b[1]),
        _nz(a[2] * b[0] - a[0] * b[2]),
        _nz(a[0] * b[1] - a[1] * b[0]),
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return _nz(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = _norm(v)
    if not math.isfinite(n) or n <= 1.0e-12:
        return 0.0, 0.0, 0.0
    inv = 1.0 / n
    out = (_nz(v[0] * inv), _nz(v[1] * inv), _nz(v[2] * inv))
    if not _finite3(out):
        return 0.0, 0.0, 0.0
    return out


def _read_raw_grid(path: Path, nu: int, nv: int) -> _RawGrid:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != nu * nv:
        raise ValueError(f"unexpected row count in {path.name}: {len(rows)}")

    u = [[0.0 for _ in range(nu)] for _ in range(nv)]
    v = [[0.0 for _ in range(nu)] for _ in range(nv)]
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

        u[iv][iu] = _nz(float(row["u_mm"]))
        v[iv][iu] = _nz(float(row["v_mm"]))
        x[iv][iu] = _nz(float(row["x_mm"]))
        y[iv][iu] = _nz(float(row["y_mm"]))
        z[iv][iu] = _nz(float(row["z_mm"]))
        nx[iv][iu] = _nz(float(row["nx"]))
        ny[iv][iu] = _nz(float(row["ny"]))
        nz[iv][iu] = _nz(float(row["nz"]))
        valid[iv][iu] = int(row["valid"])

    return _RawGrid(
        u_mm=tuple(tuple(r) for r in u),
        v_mm=tuple(tuple(r) for r in v),
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

    i0 = int(math.floor(u_idx))
    j0 = int(math.floor(v_idx))
    if i0 < 0:
        i0 = 0
    if j0 < 0:
        j0 = 0
    if i0 > nu - 2:
        i0 = nu - 2
    if j0 > nv - 2:
        j0 = nv - 2

    du = _nz(u_idx - float(i0))
    dv = _nz(v_idx - float(j0))

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


def _p(grid: _RawGrid, iu: int, iv: int) -> tuple[float, float, float]:
    return grid.x_mm[iv][iu], grid.y_mm[iv][iu], grid.z_mm[iv][iu]


def _n(grid: _RawGrid, iu: int, iv: int) -> tuple[float, float, float]:
    return grid.nx[iv][iu], grid.ny[iv][iu], grid.nz[iv][iu]


def _orient_tri(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    n_ref: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    tri_n = _normalize(_cross(_sub(p1, p0), _sub(p2, p0)))
    if _dot(tri_n, n_ref) < 0.0:
        return p0, p2, p1
    return p0, p1, p2


def _triangles_from_valid_cells(grid: _RawGrid) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    nv = len(grid.valid)
    nu = len(grid.valid[0]) if nv > 0 else 0
    tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []

    for iv in range(nv - 1):
        for iu in range(nu - 1):
            if not (
                grid.valid[iv][iu] == 1
                and grid.valid[iv][iu + 1] == 1
                and grid.valid[iv + 1][iu + 1] == 1
                and grid.valid[iv + 1][iu] == 1
            ):
                continue

            p00 = _p(grid, iu, iv)
            p10 = _p(grid, iu + 1, iv)
            p11 = _p(grid, iu + 1, iv + 1)
            p01 = _p(grid, iu, iv + 1)

            n00 = _n(grid, iu, iv)
            n10 = _n(grid, iu + 1, iv)
            n11 = _n(grid, iu + 1, iv + 1)
            n01 = _n(grid, iu, iv + 1)
            n_cell = _normalize(
                (
                    _nz(n00[0] + n10[0] + n11[0] + n01[0]),
                    _nz(n00[1] + n10[1] + n11[1] + n01[1]),
                    _nz(n00[2] + n10[2] + n11[2] + n01[2]),
                )
            )

            t1 = _orient_tri(p00, p10, p11, n_cell)
            t2 = _orient_tri(p00, p11, p01, n_cell)
            tris.append(t1)
            tris.append(t2)
    return tris


def _read_boundary_loops(path: Path) -> tuple[tuple[tuple[float, float], ...], ...]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        return ()
    by_loop: dict[int, list[tuple[int, float, float]]] = {}
    for row in rows:
        lid = int(row["loop_id"])
        pid = int(row["point_id"])
        by_loop.setdefault(lid, []).append((pid, float(row["u_idx"]), float(row["v_idx"])))

    loops: list[tuple[tuple[float, float], ...]] = []
    for lid in sorted(by_loop.keys()):
        pts = sorted(by_loop[lid], key=lambda x: x[0])
        loops.append(tuple((u, v) for _, u, v in pts))
    return tuple(loops)


def _triangles_from_boundary_fan(
    loops: tuple[tuple[tuple[float, float], ...], ...],
    grid: _RawGrid,
    mode: str,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
    for loop in loops:
        if len(loop) < 3:
            continue
        p0_uv = loop[0]
        if mode == "xyz":
            p0 = (
                _bilerp(grid.x_mm, p0_uv[0], p0_uv[1]),
                _bilerp(grid.y_mm, p0_uv[0], p0_uv[1]),
                _bilerp(grid.z_mm, p0_uv[0], p0_uv[1]),
            )
        else:
            p0 = (
                _bilerp(grid.u_mm, p0_uv[0], p0_uv[1]),
                _bilerp(grid.v_mm, p0_uv[0], p0_uv[1]),
                0.0,
            )
        n0 = _normalize(
            (
                _bilerp(grid.nx, p0_uv[0], p0_uv[1]),
                _bilerp(grid.ny, p0_uv[0], p0_uv[1]),
                _bilerp(grid.nz, p0_uv[0], p0_uv[1]),
            )
        )
        for i in range(1, len(loop) - 1):
            puv1 = loop[i]
            puv2 = loop[i + 1]
            if mode == "xyz":
                p1 = (
                    _bilerp(grid.x_mm, puv1[0], puv1[1]),
                    _bilerp(grid.y_mm, puv1[0], puv1[1]),
                    _bilerp(grid.z_mm, puv1[0], puv1[1]),
                )
                p2 = (
                    _bilerp(grid.x_mm, puv2[0], puv2[1]),
                    _bilerp(grid.y_mm, puv2[0], puv2[1]),
                    _bilerp(grid.z_mm, puv2[0], puv2[1]),
                )
            else:
                p1 = (
                    _bilerp(grid.u_mm, puv1[0], puv1[1]),
                    _bilerp(grid.v_mm, puv1[0], puv1[1]),
                    0.0,
                )
                p2 = (
                    _bilerp(grid.u_mm, puv2[0], puv2[1]),
                    _bilerp(grid.v_mm, puv2[0], puv2[1]),
                    0.0,
                )
            t = _orient_tri(p0, p1, p2, n0)
            tris.append(t)
    return tris


def _validate_triangles(
    tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    out: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
    for t in tris:
        p0, p1, p2 = t
        if not (_finite3(p0) and _finite3(p1) and _finite3(p2)):
            raise ValueError("INVALID_NUMBER")
        n = _cross(_sub(p1, p0), _sub(p2, p0))
        if _norm(n) <= 1.0e-12:
            continue
        out.append(t)
    return out


def _tri_normal(t: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> tuple[float, float, float]:
    p0, p1, p2 = t
    return _normalize(_cross(_sub(p1, p0), _sub(p2, p0)))


def _write_stl(
    path: Path,
    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
    mm_digits: int,
    unitless_digits: int,
) -> None:
    base = path.stem
    lines = [f"solid {base}"]
    for tri in triangles:
        n = _tri_normal(tri)
        lines.append(
            "  facet normal "
            + f"{fixed(n[0], unitless_digits)} {fixed(n[1], unitless_digits)} {fixed(n[2], unitless_digits)}"
        )
        lines.append("    outer loop")
        for p in tri:
            lines.append(
                "      vertex "
                + f"{fixed(p[0], mm_digits)} {fixed(p[1], mm_digits)} {fixed(p[2], mm_digits)}"
            )
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {base}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def _build_side_mesh(
    *,
    boundary_csv: Path,
    raw_csv: Path,
    nu: int,
    nv: int,
) -> tuple[list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]], str]:
    if not boundary_csv.exists():
        raise ValueError("STEP3A_BOUNDARY_MISSING")
    if not raw_csv.exists():
        raise ValueError("STEP02_RAW_MISSING")

    loops = _read_boundary_loops(boundary_csv)
    if not loops:
        raise ValueError("BOUNDARY_EMPTY")
    grid = _read_raw_grid(raw_csv, nu, nv)

    tris = _validate_triangles(_triangles_from_valid_cells(grid))
    if tris:
        return tris, "valid_cells"

    tris = _validate_triangles(_triangles_from_boundary_fan(loops, grid, mode="xyz"))
    if tris:
        return tris, "boundary_fan_xyz"

    tris = _validate_triangles(_triangles_from_boundary_fan(loops, grid, mode="uv"))
    if tris:
        return tris, "boundary_fan_uv"

    raise ValueError("NO_TRIANGLES")


def run_step03B(cfg: dict[str, Any], ctx: dict[str, Any]) -> Path:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / STEP03B_REPORT_NAME

    status = "failed"
    reason_code = "STEP03B_EXCEPTION"
    message = "step03B failed before view mesh generation"
    payload: dict[str, Any] = {
        "flank_view_mesh_plus_stl": None,
        "flank_view_mesh_minus_stl": None,
        "tri_count_plus": None,
        "tri_count_minus": None,
        "triangulation_mode_plus": None,
        "triangulation_mode_minus": None,
    }

    try:
        nu = int(cfg["Nu"])
        nv = int(cfg["Nv"])
        mm_digits = int(cfg["csv_float_digits_mm"])
        unitless_digits = int(cfg["csv_float_digits_unitless"])

        plus_csv = output_dir / "flank_uv_boundary_plus.csv"
        minus_csv = output_dir / "flank_uv_boundary_minus.csv"
        plus_raw = output_dir / "tool_conjugate_grid_plus_raw.csv"
        minus_raw = output_dir / "tool_conjugate_grid_minus_raw.csv"

        plus_tris, mode_plus = _build_side_mesh(
            boundary_csv=plus_csv,
            raw_csv=plus_raw,
            nu=nu,
            nv=nv,
        )
        minus_tris, mode_minus = _build_side_mesh(
            boundary_csv=minus_csv,
            raw_csv=minus_raw,
            nu=nu,
            nv=nv,
        )

        plus_out = output_dir / VIEW_MESH_PLUS_STL_NAME
        minus_out = output_dir / VIEW_MESH_MINUS_STL_NAME
        _write_stl(plus_out, plus_tris, mm_digits, unitless_digits)
        _write_stl(minus_out, minus_tris, mm_digits, unitless_digits)

        status = "ok"
        reason_code = "OK"
        message = "step03B view mesh export completed"
        payload = {
            "flank_view_mesh_plus_stl": plus_out.name,
            "flank_view_mesh_minus_stl": minus_out.name,
            "tri_count_plus": len(plus_tris),
            "tri_count_minus": len(minus_tris),
            "triangulation_mode_plus": mode_plus,
            "triangulation_mode_minus": mode_minus,
        }
        ctx["tri_count_flank_view_mesh_plus"] = len(plus_tris)
        ctx["tri_count_flank_view_mesh_minus"] = len(minus_tris)
    except Exception as exc:
        message = str(exc)
        if message in {
            "STEP3A_BOUNDARY_MISSING",
            "STEP02_RAW_MISSING",
            "BOUNDARY_EMPTY",
            "NO_TRIANGLES",
            "INVALID_NUMBER",
        }:
            status = "reject"
            reason_code = message

    cad_report = {
        "step_id": "step03B_view_mesh",
        "status": status,
        "reason_code": reason_code,
        "message": message,
        "exception_stacktrace": None,
        "ctx": dict(ctx),
        **payload,
    }
    write_json(report_path, cad_report)
    return report_path
