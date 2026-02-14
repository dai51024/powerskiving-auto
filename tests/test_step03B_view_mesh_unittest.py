import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.config_io import load_config
from powerskiving.pipeline.runner import run
from powerskiving.pipeline.step03B_view_mesh import run_step03B


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _facet_count(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("  facet normal "):
            count += 1
    return count


class TestStep03BViewMesh(unittest.TestCase):
    def test_runner_generates_step03B_outputs(self):
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
            self.assertIsNotNone(result.step03B_report_path)
            assert result.step03B_report_path is not None
            report = json.loads(result.step03B_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["step_id"], "step03B_view_mesh")
            self.assertEqual(report["status"], "ok")

            plus_stl = output_dir / "flank_view_mesh_plus.stl"
            minus_stl = output_dir / "flank_view_mesh_minus.stl"
            self.assertTrue(plus_stl.exists())
            self.assertTrue(minus_stl.exists())

            plus_facets = _facet_count(plus_stl)
            minus_facets = _facet_count(minus_stl)
            self.assertGreater(plus_facets, 0)
            self.assertGreater(minus_facets, 0)
            self.assertEqual(report["tri_count_plus"], plus_facets)
            self.assertEqual(report["tri_count_minus"], minus_facets)

    def test_step03B_is_deterministic(self):
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
                assert result.step03B_report_path is not None
                hashes.append(
                    (
                        _sha256(result.step03B_report_path),
                        _sha256(output_dir / "flank_view_mesh_plus.stl"),
                        _sha256(output_dir / "flank_view_mesh_minus.stl"),
                    )
                )
        self.assertEqual(hashes[0], hashes[1])

    def test_step03B_rejects_when_boundary_missing(self):
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
            cfg, _ = load_config(cfg_path)
            (output_dir / "flank_uv_boundary_plus.csv").unlink()
            report_path = run_step03B(cfg, dict(result.ctx))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "reject")
            self.assertEqual(report["reason_code"], "STEP3A_BOUNDARY_MISSING")


if __name__ == "__main__":
    unittest.main()
