import tempfile
import unittest
from pathlib import Path

from powerskiving.config_io import ConfigError, load_config


class TestConfigIO(unittest.TestCase):
    def test_rejects_duplicate_keys(self):
        raw = (
            b'{"spec_version":"1-1-1","geom_spec_version":"1-1-1",'
            b'"module_mm":2.0,"module_mm":3.0}\n'
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dup.json"
            p.write_bytes(raw)
            with self.assertRaises(ConfigError):
                load_config(p)

    def test_sha256_from_raw_bytes_is_stable(self):
        raw = b'{"spec_version":"1-1-1","geom_spec_version":"1-1-1"}\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.json"
            p.write_bytes(raw)
            cfg, config_sha256 = load_config(p)
        self.assertEqual(cfg["spec_version"], "1-1-1")
        self.assertEqual(
            config_sha256,
            "1d32a95b4e9fc9d3e03ff32f19a6e1ea292a05db3552468b7a1513e41724e1ef",
        )

    def test_fails_on_spec_version_mismatch(self):
        raw = b'{"spec_version":"9-9-9","geom_spec_version":"1-1-1"}\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad_spec.json"
            p.write_bytes(raw)
            with self.assertRaises(ConfigError):
                load_config(p)

    def test_fails_on_geom_spec_version_mismatch(self):
        raw = b'{"spec_version":"1-1-1","geom_spec_version":"9-9-9"}\n'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad_geom.json"
            p.write_bytes(raw)
            with self.assertRaises(ConfigError):
                load_config(p)


if __name__ == "__main__":
    unittest.main()
