#!/usr/bin/env python3
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPEC_PATH = Path("SPEC.md")
GEOM_PATH = Path("GEOM_SPEC.md")

try:
    from powerskiving.config_io import SPEC_VERSION, GEOM_SPEC_VERSION  # type: ignore
except Exception as e:
    print(f"[speclint] ERROR: cannot import powerskiving.config_io: {e}", file=sys.stderr)
    sys.exit(1)

VERSION_RE = re.compile(r"([0-9]+-[0-9]+-[0-9]+)")

def extract_spec_version(text: str) -> str:
    # Typical: "SPECバージョン: 1-1-1"
    m = re.search(r"SPEC\s*バージョン\s*:\s*([0-9]+-[0-9]+-[0-9]+)", text)
    if not m:
        raise ValueError("cannot find SPEC version line")
    return m.group(1)

def extract_geom_version(text: str) -> str:
    # Be tolerant to escapes like "GEOM\_SPECバージョン: 1-1-1"
    # Look for a line that includes "GEOM" and "バージョン" and extract x-y-z.
    for line in text.splitlines():
        if ("GEOM" in line) and ("バージョン" in line):
            m = VERSION_RE.search(line)
            if m:
                return m.group(1)
    raise ValueError("cannot find GEOM_SPEC version line")

def git_has_diff(paths: list[str]) -> bool:
    r = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--"] + paths,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "git diff failed")
    return bool(r.stdout.strip())

def main() -> int:
    if not SPEC_PATH.exists() or not GEOM_PATH.exists():
        print("[speclint] ERROR: SPEC.md / GEOM_SPEC.md missing", file=sys.stderr)
        return 1

    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    geom_text = GEOM_PATH.read_text(encoding="utf-8")

    spec_v = extract_spec_version(spec_text)
    geom_v = extract_geom_version(geom_text)

    ok = True
    if spec_v != SPEC_VERSION:
        print(f"[speclint] ERROR: SPEC.md version={spec_v} != code SPEC_VERSION={SPEC_VERSION}", file=sys.stderr)
        ok = False
    if geom_v != GEOM_SPEC_VERSION:
        print(f"[speclint] ERROR: GEOM_SPEC.md version={geom_v} != code GEOM_SPEC_VERSION={GEOM_SPEC_VERSION}", file=sys.stderr)
        ok = False

    # fail if these are modified vs HEAD
    if git_has_diff(["SPEC.md", "GEOM_SPEC.md", "AGENTS.md"]):
        print("[speclint] ERROR: SPEC/GEOM/AGENTS changed vs HEAD. Freeze update procedure required.", file=sys.stderr)
        ok = False

    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
