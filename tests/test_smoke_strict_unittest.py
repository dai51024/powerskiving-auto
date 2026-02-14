import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.pipeline.runner import run


class TestSmokeStrict(unittest.TestCase):
    def test_smoke_strict_step01_behavior_is_stable(self):
        strict_cfg_path = Path("configs") / "smoke_strict.json"
        strict_cfg = json.loads(strict_cfg_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "out"
            strict_cfg["output_dir"] = str(output_dir)
            cfg_path = td_path / "smoke_strict.json"
            cfg_path.write_text(
                json.dumps(strict_cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            result = run(cfg_path)
            self.assertIsNotNone(result.step01_report_path)
            assert result.step01_report_path is not None
            step01 = json.loads(result.step01_report_path.read_text(encoding="utf-8"))

            if step01["status"] == "ok":
                self.assertEqual(result.status, "ok")
                self.assertEqual(result.reason_code, "ok")
                self.assertEqual(step01["reason_code"], "OK")
                self.assertIsNotNone(result.step02_report_path)
            else:
                self.assertEqual(step01["status"], "reject")
                self.assertEqual(step01["reason_code"], "MIN_POINTS_FAIL")
                self.assertEqual(result.status, "reject")
                self.assertEqual(result.reason_code, "MIN_POINTS_FAIL")
                self.assertIsNone(result.step02_report_path)


if __name__ == "__main__":
    unittest.main()

