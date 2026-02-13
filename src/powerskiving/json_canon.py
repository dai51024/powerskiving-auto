"""Canonical JSON writer for deterministic outputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("NaN/Inf is not allowed in JSON output")
    # Fixed-point output to avoid scientific notation.
    text = format(value, ".17f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        return "0"
    if text.startswith("-0.") and float(text) == 0.0:
        return text[1:]
    return text


def _escape_string(value: str) -> str:
    out: list[str] = ['"']
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _to_json(obj: Any, indent: int) -> str:
    pad = " " * indent
    next_pad = " " * (indent + 2)

    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _format_float(obj)
    if isinstance(obj, str):
        return _escape_string(obj)
    if isinstance(obj, list):
        if not obj:
            return "[]"
        body = ",\n".join(f"{next_pad}{_to_json(item, indent + 2)}" for item in obj)
        return "[\n" + body + "\n" + pad + "]"
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        keys = sorted(obj.keys())
        body = ",\n".join(
            f"{next_pad}{_escape_string(str(k))}: {_to_json(obj[k], indent + 2)}"
            for k in keys
        )
        return "{\n" + body + "\n" + pad + "}"
    raise TypeError(f"unsupported type for JSON output: {type(obj)!r}")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    """Write canonical UTF-8/LF JSON with sorted keys and trailing LF."""
    if not isinstance(obj, dict):
        raise TypeError("top-level JSON object must be dict")
    text = _to_json(obj, 0) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")
