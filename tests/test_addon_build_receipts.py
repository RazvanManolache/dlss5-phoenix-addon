"""A compiler exit alone must not certify an absent or unreadable addon."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_addon


class BuildReceiptTests(unittest.TestCase):
    def run_build(self, fault=None, diagnostic=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sdk = root / "sdk"
            sdk.mkdir()
            for name in ("nvsdk_ngx.h", "nvsdk_ngx_defs.h", "nvsdk_ngx_params.h"):
                (sdk / name).write_text("// test SDK placeholder")
            if diagnostic:
                source = root / "src/integration/reshade/diagnostic_addon.cpp"
                source.parent.mkdir(parents=True)
                source.write_text("// target selection fixture")
            out = root / "build/test"

            def compiler(*args, **kwargs):
                recipe = (out / "build.cmd").read_text()
                import re
                for name in re.findall(r'/Fe:"[^"\r\n]*/([^/"\r\n]+)"', recipe):
                    if not (fault == "missing" and name == "phoenix.addon64"):
                        (out / name).write_bytes(b"compiled fixture")
                    if name.endswith(".exe"):
                        (out / (name[:-4] + ".result.json")).write_text('{"passed":true}')

            original_read = Path.read_bytes

            def read_bytes(path):
                if fault == "locked" and path.name == "phoenix.addon64":
                    raise PermissionError("simulated locked artifact")
                return original_read(path)

            with patch.object(build_addon, "ROOT", root), \
                    patch.object(sys, "argv", ["build_addon.py", "test", "--ngx-sdk", str(sdk)]), \
                    patch.object(build_addon.subprocess, "check_output", return_value="C:/VS"), \
                    patch.object(build_addon.subprocess, "run", side_effect=compiler), \
                    patch.object(Path, "read_bytes", read_bytes):
                if fault:
                    with self.assertRaisesRegex(RuntimeError, "Cannot verify final artifact"):
                        build_addon.main()
                else:
                    build_addon.main()
            return json.loads((out / "report.json").read_text())

    def test_default_subset_and_verified_artifacts(self):
        report = self.run_build()
        self.assertTrue(report["passed"])
        self.assertTrue(report["artifact_verification_completed"])
        self.assertNotIn("phoenix_diagnostic.addon64", report["outputs"])

    def test_diagnostic_source_remains_supported(self):
        self.assertIn("phoenix_diagnostic.addon64", self.run_build(diagnostic=True)["outputs"])

    def test_disappearing_or_locked_output_fails_after_compilation(self):
        for fault in ("missing", "locked"):
            with self.subTest(fault=fault):
                report = self.run_build(fault)
                self.assertFalse(report["passed"])
                self.assertTrue(report["compilation_completed"])
                self.assertTrue(report["contract_tests_passed"])
                self.assertFalse(report["artifact_verification_completed"])
                self.assertEqual(report["failed_stage"], "verify_artifacts")


if __name__ == "__main__":
    unittest.main()
