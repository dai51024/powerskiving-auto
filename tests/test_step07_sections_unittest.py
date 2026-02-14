import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.deterministic import fixed, q
from powerskiving.pipeline.runner import run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ztag(z_sec_mm: float, digits: int) -> str:
    z_q = q(z_sec_mm, digits)
    sign = "+" if z_q >= 0.0 else "-"
    text = fixed(abs(z_q), digits)
    int_part, frac_part = text.split(".")
    return f"z{sign}{int(int_part):05d}.{frac_part}"


class TestStep07Sections(unittest.TestCase):
    def test_runner_generates_step07_dxf_for_all_sections(self):
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
            self.assertIsNotNone(result.step07_report_path)
            assert result.step07_report_path is not None
            report = json.loads(result.step07_report_path.read_text(encoding="utf-8"))

            self.assertEqual(report["step_id"], "step07_sections_dxf")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["reason_code"], "OK")

            z_sections = smoke_cfg["z_sections_mm"]
            self.assertEqual(report["z_sections_count"], len(z_sections))
            self.assertEqual(len(report["sections_dxf_files"]), len(z_sections))

            for name in report["sections_dxf_files"]:
                self.assertTrue((output_dir / name).exists())

    def test_step07_file_names_follow_ztag(self):
        smoke_cfg = json.loads((Path("configs") / "smoke.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "out"
            smoke_cfg["output_dir"] = str(output_dir)
            smoke_cfg["z_sections_mm"] = [0.0, -3.0, 12.5]
            cfg_path = td_path / "smoke.json"
            cfg_path.write_text(
                json.dumps(smoke_cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            _ = run(cfg_path)

            digits = int(smoke_cfg["sort_round_mm_digits"])
            expected = sorted(f"sections_{_ztag(float(z), digits)}.dxf" for z in smoke_cfg["z_sections_mm"])
            actual = sorted(p.name for p in output_dir.glob("sections_*.dxf"))
            self.assertEqual(actual, expected)

    def test_step07_is_deterministic(self):
        smoke_cfg = json.loads((Path("configs") / "smoke.json").read_text(encoding="utf-8"))
        hashes: list[tuple[str, tuple[tuple[str, str], ...]]] = []
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
                assert result.step07_report_path is not None
                dxf_hashes = tuple(
                    sorted((p.name, _sha256(p)) for p in output_dir.glob("sections_*.dxf"))
                )
                hashes.append((_sha256(result.step07_report_path), dxf_hashes))

        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main()
