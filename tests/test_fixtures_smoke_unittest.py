import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from powerskiving.pipeline.runner import run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestFixturesSmoke(unittest.TestCase):
    def test_smoke_outputs_match_fixtures(self):
        fixture_dir = Path("fixtures") / "v1-1-1" / "smoke_expected"
        expected_files = sorted(p.name for p in fixture_dir.iterdir() if p.is_file())
        self.assertTrue(expected_files, "fixture directory is empty")

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

            _ = run(cfg_path)
            actual_files = sorted(p.name for p in output_dir.iterdir() if p.is_file())

            issues: list[str] = []
            missing = sorted(set(expected_files) - set(actual_files))
            extra = sorted(set(actual_files) - set(expected_files))
            if missing:
                issues.append(f"missing files: {missing}")
            if extra:
                issues.append(f"extra files: {extra}")

            for name in expected_files:
                expected_path = fixture_dir / name
                actual_path = output_dir / name
                if not actual_path.exists():
                    continue
                expected_hash = _sha256(expected_path)
                actual_hash = _sha256(actual_path)
                if actual_hash != expected_hash:
                    issues.append(
                        f"sha256 mismatch: {name} expected={expected_hash} actual={actual_hash}"
                    )

            if issues:
                self.fail("\n".join(issues))


if __name__ == "__main__":
    unittest.main()
