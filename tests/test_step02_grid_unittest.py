import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.pipeline.runner import run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestStep02Grid(unittest.TestCase):
    def test_runner_generates_step02_report_and_raw_csv(self):
        smoke_cfg = json.loads((Path("configs") / "smoke.json").read_text(encoding="utf-8"))
        expected_rows = int(smoke_cfg["Nu"]) * int(smoke_cfg["Nv"]) + 1
        expected_header = (
            "iu,iv,u_mm,v_mm,x_mm,y_mm,z_mm,nx,ny,nz,theta1_rad,theta2_rad,residual_abs,valid,reason_code"
        )

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
            self.assertIsNotNone(result.step02_report_path)
            assert result.step02_report_path is not None
            self.assertTrue(result.step02_report_path.exists())
            self.assertEqual(result.step02_report_path.name, "cad_report_step02.json")

            report = json.loads(result.step02_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["step_id"], "step02_grid")
            self.assertEqual(report["status"], "ok")
            self.assertGreater(report["valid_ratio_total"], 0.0)

            plus_csv = output_dir / "tool_conjugate_grid_plus_raw.csv"
            minus_csv = output_dir / "tool_conjugate_grid_minus_raw.csv"
            self.assertTrue(plus_csv.exists())
            self.assertTrue(minus_csv.exists())
            plus_lines = plus_csv.read_text(encoding="utf-8").splitlines()
            minus_lines = minus_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(plus_lines), expected_rows)
            self.assertEqual(len(minus_lines), expected_rows)
            self.assertEqual(plus_lines[0], expected_header)
            self.assertEqual(minus_lines[0], expected_header)

    def test_step02_outputs_are_deterministic(self):
        smoke_cfg = json.loads((Path("configs") / "smoke.json").read_text(encoding="utf-8"))

        hashes: list[tuple[str, str, str]] = []
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
                assert result.step02_report_path is not None
                hashes.append(
                    (
                        _sha256(result.step02_report_path),
                        _sha256(output_dir / "tool_conjugate_grid_plus_raw.csv"),
                        _sha256(output_dir / "tool_conjugate_grid_minus_raw.csv"),
                    )
                )

        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
