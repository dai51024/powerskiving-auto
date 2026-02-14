"""Deterministic ASCII DXF R2000 writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from powerskiving.deterministic import fixed, q


@dataclass(frozen=True)
class DxfLwPolyline:
    layer: str
    points_xy: tuple[tuple[float, float], ...]
    closed: bool = False


def _code(group: int, value: str) -> str:
    return f"{group}\n{value}\n"


def _fmt(x: float, digits: int) -> str:
    return fixed(q(x, digits), digits)


def write_dxf_r2000(
    *,
    path: str | Path,
    layers: Iterable[str],
    lwpolylines: Iterable[DxfLwPolyline],
    mm_digits: int,
) -> None:
    """Write deterministic DXF R2000 (AC1015) with LWPOLYLINE entities."""
    layer_list = sorted(set(layers))
    polys = list(lwpolylines)

    out: list[str] = []

    out.append(_code(0, "SECTION"))
    out.append(_code(2, "HEADER"))
    out.append(_code(9, "$ACADVER"))
    out.append(_code(1, "AC1015"))
    out.append(_code(9, "$INSUNITS"))
    out.append(_code(70, "4"))
    out.append(_code(0, "ENDSEC"))

    out.append(_code(0, "SECTION"))
    out.append(_code(2, "TABLES"))
    out.append(_code(0, "TABLE"))
    out.append(_code(2, "LAYER"))
    out.append(_code(70, str(len(layer_list))))
    for layer in layer_list:
        out.append(_code(0, "LAYER"))
        out.append(_code(2, layer))
        out.append(_code(70, "0"))
        out.append(_code(62, "7"))
        out.append(_code(6, "CONTINUOUS"))
    out.append(_code(0, "ENDTAB"))
    out.append(_code(0, "ENDSEC"))

    out.append(_code(0, "SECTION"))
    out.append(_code(2, "ENTITIES"))
    for poly in polys:
        if not poly.points_xy:
            continue
        out.append(_code(0, "LWPOLYLINE"))
        out.append(_code(8, poly.layer))
        out.append(_code(90, str(len(poly.points_xy))))
        out.append(_code(70, "1" if poly.closed else "0"))
        for x, y in poly.points_xy:
            out.append(_code(10, _fmt(x, mm_digits)))
            out.append(_code(20, _fmt(y, mm_digits)))
    out.append(_code(0, "ENDSEC"))

    out.append(_code(0, "EOF"))
    Path(path).write_text("".join(out), encoding="ascii", newline="\n")
