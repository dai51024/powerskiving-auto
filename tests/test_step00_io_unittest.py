import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.pipeline.runner import run


class TestStep00IO(unittest.TestCase):
    def test_step00_smoke_generates_cad_report(self):
        smoke_path = Path("configs") / "smoke.json"
        smoke_cfg = json.loads(smoke_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            stale = output_dir / "stale.txt"
            stale.write_text("stale", encoding="utf-8")

            smoke_cfg["output_dir"] = str(output_dir)
            cfg_path = td_path / "smoke.json"
            cfg_path.write_text(
                json.dumps(smoke_cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            report_path = run(cfg_path)
            self.assertEqual(report_path.name, "cad_report_step00.json")
            self.assertTrue(report_path.exists())
            self.assertFalse(stale.exists())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["step_id"], "step00_io")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["error_code"], "ok")


if __name__ == "__main__":
    unittest.main()
