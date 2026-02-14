import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.pipeline.runner import run
from powerskiving.pipeline.step05_cutting_edge import run_step05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raw_grid_csv(path: Path, nu: int, nv: int, side: str) -> None:
    lines = ["iu,iv,x_mm,y_mm,z_mm,nx,ny,nz,valid"]
    for iv in range(nv):
        for iu in range(nu):
            x_mm = 19.0 + float(iu)
            y_mm = -1.0 + (2.0 * float(iv) / float(nv - 1))
            if side == "minus":
                y_mm += 0.03
            z_mm = float(iv) * 0.4
            lines.append(f"{iu},{iv},{x_mm:.6f},{y_mm:.6f},{z_mm:.6f},1.0,0.0,0.0,1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


class TestStep05CuttingEdge(unittest.TestCase):
    def test_runner_generates_step05_outputs(self):
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
            self.assertIsNotNone(result.step05_report_path)
            assert result.step05_report_path is not None
            self.assertTrue(result.step05_report_path.exists())

            report = json.loads(result.step05_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["step_id"], "step05_cutting_edge")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["reason_code"], "EDGES_NOT_OK")
            self.assertEqual(report["edges_ok"], False)
            self.assertEqual(report["edge_plus_ok"], False)
            self.assertEqual(report["edge_minus_ok"], False)
            self.assertIn("rake_angle_deg", report)
            self.assertEqual(report["tool_z_min"], None)
            self.assertEqual(report["tool_z_max"], None)
            self.assertIn("edge_band", report)
            self.assertEqual(report["edge_band"]["status"], "skipped")
            self.assertEqual(report["edge_band"]["skipped_reason"], "EDGES_NOT_OK")
            self.assertEqual(report["edge_band_cells_csv"], None)

            edge_csv = output_dir / "cutting_edge_points.csv"
            self.assertTrue(edge_csv.exists())
            lines = edge_csv.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 1)
            self.assertEqual(
                lines[0],
                "edge_side,edge_component_id,selected,point_id,u_idx,v_idx,"
                "x_mm,y_mm,z_mm,nx,ny,nz,tx,ty,tz,s_mm,plane_dist_mm,valid,reason_code",
            )
            self.assertFalse((output_dir / "edge_band_cells.csv").exists())

    def test_step05_is_deterministic(self):
        smoke_cfg = json.loads((Path("configs") / "smoke.json").read_text(encoding="utf-8"))
        hashes: list[tuple[str, str]] = []
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
                self.assertIsNotNone(result.step05_report_path)
                assert result.step05_report_path is not None
                hashes.append(
                    (
                        _sha256(result.step05_report_path),
                        _sha256(output_dir / "cutting_edge_points.csv"),
                    )
                )
        self.assertEqual(hashes[0], hashes[1])

    def test_step05_edges_ok_true_generates_edge_band_and_od(self):
        base_cfg = json.loads((Path("configs") / "demo_edges_ok.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg = dict(base_cfg)
            cfg["output_dir"] = str(output_dir)

            _write_raw_grid_csv(output_dir / "tool_conjugate_grid_plus_raw.csv", 4, 4, "plus")
            _write_raw_grid_csv(output_dir / "tool_conjugate_grid_minus_raw.csv", 4, 4, "minus")

            ctx: dict[str, object] = {}
            report_path = run_step05(cfg, ctx)
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["reason_code"], "OK")
            self.assertEqual(report["edges_ok"], True)
            self.assertEqual(report["edge_band"]["status"], "ok")
            self.assertEqual(report["edge_band"]["skipped_reason"], None)
            self.assertIsNotNone(report["edge_band_cells_csv"])
            self.assertTrue((output_dir / "edge_band_cells.csv").exists())
            self.assertIsInstance(report["tool_z_min"], (int, float))
            self.assertIsInstance(report["tool_z_max"], (int, float))
            self.assertIsInstance(report["od"], dict)
            self.assertIn("r_od", report["od"])
            self.assertGreater(report["od"]["r_od"], 0.0)


if __name__ == "__main__":
    unittest.main()
