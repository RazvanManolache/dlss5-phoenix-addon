import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
spec = importlib.util.spec_from_file_location(
    "release_audit", Path(__file__).resolve().parents[1] / "tools/audit_release.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class PublicationScopeTests(unittest.TestCase):
    def test_selected_kernel_is_published_but_reference_and_private_outputs_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = [
                "src/native/kernels/production/optimized/output/reconstruction.cu",
                "src/native/kernels/production/reference/output/reconstruction.cu",
                "src/native/kernels/production/common/async_memory.cuh",
                "output/result.json",
                "build/native/kernel.cubin",
                "runtime/nvngx_dlssnr.dll",
                "models/local/model.json",
                "models/local/resources/WEIGHTS_HT.bin",
                "source_maps/identifiers.json",
                "src/native/kernels/output/reconstruction/kernel.ptx",
                "src/native/model/kernel_tuning.json",
                ".ruff_cache/checks",
            ]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test")
            with patch.object(audit, "ROOT", root):
                published = {
                    source_path.relative_to(root).as_posix() for source_path in audit.candidates()
                }
            self.assertEqual(published, {names[0], names[2]})

    def test_missing_common_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            name = "src/native/kernels/production/optimized/test.cu"
            path = root / name
            path.parent.mkdir(parents=True)
            path.write_text('#include "../common/missing.cuh"\n')
            catalogue = {"kernels": [{"source": name}]}
            with patch.object(audit, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "dependency missing"):
                    audit.kernel_closure(catalogue, {name: path})

    def test_kernel_include_cannot_escape_source_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            name = "src/native/kernels/production/test.cu"
            path = root / name
            path.parent.mkdir(parents=True)
            path.write_text('#include "../private.cuh"\n')
            with patch.object(audit, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "escapes production"):
                    audit.kernel_closure({"kernels": [{"source": name}]}, {name: path})

    def test_disguised_binary_and_all_ptx_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.h"
            path.write_bytes(b"MZbinary")
            with self.assertRaisesRegex(ValueError, "Binary executable"):
                audit.validate_payload("src/source.h", path)
            path = path.with_suffix(".ptx")
            path.write_text(".version 8.0\n.entry test() {}\n")
            with self.assertRaisesRegex(ValueError, "Forbidden publication"):
                audit.validate_payload("src/unselected.ptx", path)
            formerly_selected = "src/native/kernels/production/optimized/input/preparation.ptx"
            with self.assertRaisesRegex(ValueError, "Forbidden publication"):
                audit.validate_payload(formerly_selected, path)


if __name__ == "__main__":
    unittest.main()
