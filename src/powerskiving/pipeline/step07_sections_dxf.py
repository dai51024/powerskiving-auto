"""Step07 section DXF export (SPEC Step7)."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from powerskiving.algorithms import marching_squares_zero
from powerskiving.deterministic import fixed, q, wrap_rad
from powerskiving.formats import DxfLwPolyline, write_dxf_r2000
from powerskiving.json_canon import write_json

STEP07_REPORT_NAME = "cad_report_step07.json"


@dataclass(frozen=True)
class _RawGrid:
    x_mm: tuple[tuple[float, ...], ...]
    y_mm: tuple[tuple[float, ...], ...]
    z_mm: tuple[tuple[float, ...], ...]
    valid: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _SectionCandidate:
    src_idx: int
    points_xy: tuple[tuple[float, float], ...]
    closed: bool
    in_sector: int
    theta_comp_q: float
    dist_center_q: float
    length_xy_q: float
    n_points: int


def _nz(x: float) -> float:
    if x == 0.0:
        return 0.0
    return x


def _read_raw_grid(path: Path, nu: int, nv: int) -> _RawGrid:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != nu * nv:
        raise ValueError(f"unexpected row count in {path.name}: {len(rows)}")

    x = [[0.0 for _ in range(nu)] for _ in range(nv)]
    y = [[0.0 for _ in range(nu)] for _ in range(nv)]
    z = [[0.0 for _ in range(nu)] for _ in range(nv)]
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
        valid[iv][iu] = int(row["valid"])

    return _RawGrid(
        x_mm=tuple(tuple(r) for r in x),
        y_mm=tuple(tuple(r) for r in y),
        z_mm=tuple(tuple(r) for r in z),
        valid=tuple(tuple(r) for r in valid),
    )


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


def _polyline_length_xy(points_xy: tuple[tuple[float, float], ...], closed: bool) -> float:
    if len(points_xy) < 2:
        return 0.0
    acc = 0.0
    upto = len(points_xy)
    for i in range(1, upto):
        dx = _nz(points_xy[i][0] - points_xy[i - 1][0])
        dy = _nz(points_xy[i][1] - points_xy[i - 1][1])
        acc = _nz(acc + math.sqrt(dx * dx + dy * dy))
    if closed:
        dx = _nz(points_xy[0][0] - points_xy[-1][0])
        dy = _nz(points_xy[0][1] - points_xy[-1][1])
        acc = _nz(acc + math.sqrt(dx * dx + dy * dy))
    return _nz(acc)


def _theta_comp(points_xy: tuple[tuple[float, float], ...]) -> float:
    if not points_xy:
        return 0.0
    thetas = [wrap_rad(math.atan2(p[1], p[0])) for p in points_xy]
    mx = sum(math.cos(t) for t in thetas) / float(len(thetas))
    my = sum(math.sin(t) for t in thetas) / float(len(thetas))
    if (mx * mx + my * my) < 1.0e-12:
        return thetas[0]
    return wrap_rad(math.atan2(my, mx))


def _ztag(z_sec_mm: float, digits: int) -> str:
    z_q = q(z_sec_mm, digits)
    sign = "+" if z_q >= 0.0 else "-"
    text = fixed(abs(z_q), digits)
    int_part, frac_part = text.split(".")
    return f"z{sign}{int(int_part):05d}.{frac_part}"


def _build_field(grid: _RawGrid, z_sec_mm: float) -> tuple[tuple[float, ...], ...]:
    out: list[tuple[float, ...]] = []
    for iv in range(len(grid.z_mm)):
        row: list[float] = []
        for iu in range(len(grid.z_mm[iv])):
            row.append(_nz(grid.z_mm[iv][iu] - z_sec_mm))
        out.append(tuple(row))
    return tuple(out)


def _build_candidates(
    *,
    grid: _RawGrid,
    z_sec_mm: float,
    cfg: dict[str, Any],
) -> tuple[list[_SectionCandidate], int | None]:
    field = _build_field(grid, z_sec_mm)
    comps = marching_squares_zero(
        field=field,
        ms_iso_eps=float(cfg["ms_iso_eps"]),
        ms_fc_eps=float(cfg["ms_fc_eps"]),
        ms_eps_d=float(cfg["ms_eps_d"]),
        ms_tie_break=str(cfg["ms_tie_break"]),
        sort_round_uv_digits=int(cfg["sort_round_uv_digits"]),
        uv_on_edge_eps=float(cfg["uv_on_edge_eps"]),
        cell_valid_mask=_cell_mask(grid.valid),
    )

    rad_digits = int(cfg["sort_round_rad_digits"])
    mm_digits = int(cfg["sort_round_mm_digits"])
    theta_center = float(cfg["theta_tooth_center_rad"])
    half_sector = math.pi / float(int(cfg["z1"]))
    margin = math.radians(float(cfg["theta_sector_margin_deg"]))

    built: list[_SectionCandidate] = []
    for idx, comp in enumerate(comps):
        points_xy = tuple(
            (
                _bilerp(grid.x_mm, uv[0], uv[1]),
                _bilerp(grid.y_mm, uv[0], uv[1]),
            )
            for uv in comp.points
        )
        theta_comp = _theta_comp(points_xy)
        theta_comp_q = q(theta_comp, rad_digits)
        dist_center_q = q(abs(wrap_rad(theta_comp - theta_center)), rad_digits)
        length_xy_q = q(_polyline_length_xy(points_xy, comp.closed), mm_digits)
        in_sector = 1 if abs(wrap_rad(theta_comp - theta_center)) <= (half_sector + margin) else 0

        built.append(
            _SectionCandidate(
                src_idx=idx,
                points_xy=points_xy,
                closed=comp.closed,
                in_sector=in_sector,
                theta_comp_q=theta_comp_q,
                dist_center_q=dist_center_q,
                length_xy_q=length_xy_q,
                n_points=len(points_xy),
            )
        )

    ordered = sorted(
        built,
        key=lambda c: (
            -c.in_sector,
            c.dist_center_q,
            -c.length_xy_q,
            c.theta_comp_q,
            -c.n_points,
            c.src_idx,
        ),
    )

    selected_component_id: int | None = None
    in_sector_only = [c for c in ordered if c.in_sector == 1]
    if in_sector_only:
        best = sorted(
            in_sector_only,
            key=lambda c: (
                c.dist_center_q,
                -c.length_xy_q,
                c.theta_comp_q,
                -c.n_points,
                c.src_idx,
            ),
        )[0]
        selected_component_id = ordered.index(best)

    return ordered, selected_component_id


def run_step07(cfg: dict[str, Any], ctx: dict[str, Any]) -> Path:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / STEP07_REPORT_NAME

    status = "failed"
    reason_code = "STEP07_EXCEPTION"
    message = "step07 failed before section export"
    payload: dict[str, Any] = {
        "sections_dxf_files": [],
        "z_sections_count": 0,
        "z_sections_mm": [],
        "sections": [],
    }

    try:
        nu = int(cfg["Nu"])
        nv = int(cfg["Nv"])
        z_sections = list(cfg.get("z_sections_mm", []))
        if not z_sections:
            raise ValueError("Z_SECTIONS_EMPTY")

        plus_raw = output_dir / "tool_conjugate_grid_plus_raw.csv"
        minus_raw = output_dir / "tool_conjugate_grid_minus_raw.csv"
        if not plus_raw.exists() or not minus_raw.exists():
            raise ValueError("STEP02_RAW_MISSING")

        plus_grid = _read_raw_grid(plus_raw, nu, nv)
        minus_grid = _read_raw_grid(minus_raw, nu, nv)

        dxf_files: list[str] = []
        sections_report: list[dict[str, Any]] = []
        mm_digits = int(cfg["sort_round_mm_digits"])

        for z_raw in z_sections:
            z_sec_mm = float(z_raw)
            ztag = _ztag(z_sec_mm, mm_digits)
            dxf_name = f"sections_{ztag}.dxf"
            dxf_path = output_dir / dxf_name

            plus_candidates, plus_selected_id = _build_candidates(grid=plus_grid, z_sec_mm=z_sec_mm, cfg=cfg)
            minus_candidates, minus_selected_id = _build_candidates(grid=minus_grid, z_sec_mm=z_sec_mm, cfg=cfg)

            entities: list[DxfLwPolyline] = []
            if plus_selected_id is not None:
                c = plus_candidates[plus_selected_id]
                entities.append(DxfLwPolyline(layer="FLANK_PLUS", points_xy=c.points_xy, closed=c.closed))
            if minus_selected_id is not None:
                c = minus_candidates[minus_selected_id]
                entities.append(DxfLwPolyline(layer="FLANK_MINUS", points_xy=c.points_xy, closed=c.closed))

            write_dxf_r2000(
                path=dxf_path,
                layers=("FLANK_PLUS", "FLANK_MINUS", "EDGE_PLUS", "EDGE_MINUS", "RAKE", "OD_REF"),
                lwpolylines=entities,
                mm_digits=mm_digits,
            )
            dxf_files.append(dxf_name)

            section_entry = {
                "z_sec_mm": q(z_sec_mm, mm_digits),
                "ztag": ztag,
                "dxf": dxf_name,
                "plus": {
                    "component_count": len(plus_candidates),
                    "selected_component_id": plus_selected_id,
                    "selected_reason": "MIN_DIST_MAX_LENGTH" if plus_selected_id is not None else "NO_IN_SECTOR_CANDIDATE",
                    "candidates": [
                        {
                            "component_id": cid,
                            "src_idx": c.src_idx,
                            "in_sector": c.in_sector,
                            "theta_comp_q": c.theta_comp_q,
                            "dist_center_q": c.dist_center_q,
                            "length_xy_q": c.length_xy_q,
                            "n_points": c.n_points,
                            "selected": 1 if plus_selected_id is not None and cid == plus_selected_id else 0,
                        }
                        for cid, c in enumerate(plus_candidates)
                    ],
                },
                "minus": {
                    "component_count": len(minus_candidates),
                    "selected_component_id": minus_selected_id,
                    "selected_reason": "MIN_DIST_MAX_LENGTH" if minus_selected_id is not None else "NO_IN_SECTOR_CANDIDATE",
                    "candidates": [
                        {
                            "component_id": cid,
                            "src_idx": c.src_idx,
                            "in_sector": c.in_sector,
                            "theta_comp_q": c.theta_comp_q,
                            "dist_center_q": c.dist_center_q,
                            "length_xy_q": c.length_xy_q,
                            "n_points": c.n_points,
                            "selected": 1 if minus_selected_id is not None and cid == minus_selected_id else 0,
                        }
                        for cid, c in enumerate(minus_candidates)
                    ],
                },
            }
            sections_report.append(section_entry)

        status = "ok"
        reason_code = "OK"
        message = "step07 section dxf export completed"
        payload = {
            "sections_dxf_files": dxf_files,
            "z_sections_count": len(z_sections),
            "z_sections_mm": [q(float(z), mm_digits) for z in z_sections],
            "sections": sections_report,
        }

    except Exception as exc:
        message = str(exc)
        if message in {"STEP02_RAW_MISSING", "Z_SECTIONS_EMPTY"}:
            status = "reject"
            reason_code = message

    cad_report = {
        "step_id": "step07_sections_dxf",
        "status": status,
        "reason_code": reason_code,
        "message": message,
        "exception_stacktrace": None,
        "ctx": dict(ctx),
        **payload,
    }
    write_json(report_path, cad_report)
    return report_path
