import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.config_io import load_config
from powerskiving.pipeline.runner import run
from powerskiving.pipeline.step03A_boundary import run_step03A


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], eps: float) -> int:
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
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


def _assert_closed_and_non_self_intersecting(path: Path, eps: float) -> None:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    by_loop: dict[int, list[tuple[int, float, float]]] = {}
    for row in rows:
        loop_id = int(row["loop_id"])
        point_id = int(row["point_id"])
        u = float(row["u_idx"])
        v = float(row["v_idx"])
        by_loop.setdefault(loop_id, []).append((point_id, u, v))

    if not by_loop:
        raise AssertionError(f"no loops in {path}")

    for loop_id, vals in by_loop.items():
        ordered = sorted(vals, key=lambda x: x[0])
        pts = [(u, v) for _, u, v in ordered]
        if len(pts) < 3:
            raise AssertionError(f"loop {loop_id} has fewer than 3 points")

        segs = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if j == i + 1:
                    continue
                if i == 0 and j == len(segs) - 1:
                    continue
                if _segments_intersect(segs[i], segs[j], eps):
                    raise AssertionError(f"loop {loop_id} self-intersection at segments {i},{j}")


class TestStep03ABoundary(unittest.TestCase):
    def test_runner_generates_step03A_outputs(self):
        smoke_cfg = json.loads((Path("configs") / "smoke.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "out"
            smoke_cfg["output_dir"] = str(output_dir)
            cfg_path = td_path / "smoke.json"
            cfg_path.write_text(
                json.dumps(smoke_cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            result = run(cfg_path)
            self.assertEqual(result.status, "ok")
            self.assertIsNotNone(result.step03A_report_path)
            assert result.step03A_report_path is not None
            self.assertTrue(result.step03A_report_path.exists())
            self.assertEqual(result.step03A_report_path.name, "cad_report_step03A.json")

            report = json.loads(result.step03A_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["step_id"], "step03A_boundary")
            self.assertEqual(report["status"], "ok")
            self.assertGreaterEqual(report["loop_count_plus"], 1)
            self.assertGreaterEqual(report["loop_count_minus"], 1)

            plus_csv = output_dir / "flank_uv_boundary_plus.csv"
            minus_csv = output_dir / "flank_uv_boundary_minus.csv"
            plus_step = output_dir / "flank_boundary_wire_plus.step"
            minus_step = output_dir / "flank_boundary_wire_minus.step"
            self.assertTrue(plus_csv.exists())
            self.assertTrue(minus_csv.exists())
            self.assertTrue(plus_step.exists())
            self.assertTrue(minus_step.exists())
            self.assertGreater(len(plus_csv.read_text(encoding="utf-8").splitlines()), 1)
            self.assertGreater(len(minus_csv.read_text(encoding="utf-8").splitlines()), 1)
            self.assertGreater(plus_step.stat().st_size, 0)
            self.assertGreater(minus_step.stat().st_size, 0)

            eps = float(smoke_cfg["uv_on_edge_eps"])
            _assert_closed_and_non_self_intersecting(plus_csv, eps)
            _assert_closed_and_non_self_intersecting(minus_csv, eps)

    def test_step03A_is_deterministic(self):
        smoke_cfg = json.loads((Path("configs") / "smoke.json").read_text(encoding="utf-8"))
        hashes: list[tuple[str, str, str, str, str]] = []
        for idx in range(2):
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                output_dir = td_path / f"out_{idx}"
                smoke_cfg["output_dir"] = str(output_dir)
                cfg_path = td_path / f"smoke_{idx}.json"
                cfg_path.write_text(
                    json.dumps(smoke_cfg, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                result = run(cfg_path)
                self.assertEqual(result.status, "ok")
                assert result.step03A_report_path is not None
                hashes.append(
                    (
                        _sha256(result.step03A_report_path),
                        _sha256(output_dir / "flank_uv_boundary_plus.csv"),
                        _sha256(output_dir / "flank_uv_boundary_minus.csv"),
                        _sha256(output_dir / "flank_boundary_wire_plus.step"),
                        _sha256(output_dir / "flank_boundary_wire_minus.step"),
                    )
                )
        self.assertEqual(hashes[0], hashes[1])

    def test_step03A_rejects_when_valid_points_are_zero(self):
        smoke_cfg = json.loads((Path("configs") / "smoke.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "out"
            smoke_cfg["output_dir"] = str(output_dir)
            cfg_path = td_path / "smoke.json"
            cfg_path.write_text(
                json.dumps(smoke_cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            cfg, _ = load_config(cfg_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            header = (
                "iu,iv,u_mm,v_mm,x_mm,y_mm,z_mm,nx,ny,nz,theta1_rad,theta2_rad,residual_abs,valid,reason_code"
            )
            lines = [header]
            nu = int(cfg["Nu"])
            nv = int(cfg["Nv"])
            for iv in range(nv):
                for iu in range(nu):
                    lines.append(
                        ",".join(
                            [
                                str(iu),
                                str(iv),
                                "0.000000",
                                "0.000000",
                                "0.000000",
                                "0.000000",
                                "0.000000",
                                "0.00000000",
                                "0.00000000",
                                "0.00000000",
                                "0.000000000000",
                                "0.000000000000",
                                "0.000000",
                                "0",
                                "SOLVER_FAIL",
                            ]
                        )
                    )
            blob = "\n".join(lines) + "\n"
            (output_dir / "tool_conjugate_grid_plus_raw.csv").write_text(blob, encoding="utf-8", newline="\n")
            (output_dir / "tool_conjugate_grid_minus_raw.csv").write_text(blob, encoding="utf-8", newline="\n")

            report_path = run_step03A(cfg, {})
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "reject")
            self.assertEqual(report["reason_code"], "LOOP_COUNT_INVALID")


if __name__ == "__main__":
    unittest.main()
