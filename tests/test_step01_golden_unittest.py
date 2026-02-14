import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.pipeline.runner import run


class TestStep01Golden(unittest.TestCase):
    def test_runner_generates_step01_report_and_updates_ctx(self):
        smoke_path = Path("configs") / "smoke.json"
        smoke_cfg = json.loads(smoke_path.read_text(encoding="utf-8"))

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
            self.assertIsNotNone(result.step01_report_path)
            assert result.step01_report_path is not None
            self.assertTrue(result.step01_report_path.exists())
            self.assertEqual(result.step01_report_path.name, "cad_report_step01.json")
            self.assertIn("s_rot_selected", result.ctx)
            self.assertIn("golden_p95_mm", result.ctx)
            self.assertIn("golden_max_mm", result.ctx)

            report = json.loads(result.step01_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["step_id"], "step01_golden")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["s_rot_selected"], result.ctx["s_rot_selected"])
            self.assertEqual(report["golden_p95_mm"], result.ctx["golden_p95_mm"])
            self.assertEqual(report["golden_max_mm"], result.ctx["golden_max_mm"])

    def test_runner_stops_on_step01_reject(self):
        smoke_path = Path("configs") / "smoke.json"
        smoke_cfg = json.loads(smoke_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "out"
            smoke_cfg["output_dir"] = str(output_dir)
            smoke_cfg["golden_min_points"] = 100000
            cfg_path = td_path / "smoke_reject.json"
            cfg_path.write_text(
                json.dumps(smoke_cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            result = run(cfg_path)
            self.assertEqual(result.status, "reject")
            self.assertIsNotNone(result.step01_report_path)
            assert result.step01_report_path is not None
            report = json.loads(result.step01_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["step_id"], "step01_golden")
            self.assertEqual(report["status"], "reject")
            self.assertNotEqual(report["reason_code"], "OK")
            self.assertNotIn("s_rot_selected", result.ctx)
            self.assertFalse((output_dir / "cad_report_step02.json").exists())


if __name__ == "__main__":
    unittest.main()
