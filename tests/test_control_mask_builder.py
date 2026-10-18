import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("mask_builder", Path(__file__).parents[1] / "tools/build_control_mask.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ControlMaskBuilderTests(unittest.TestCase):
    def test_object_settings_and_linear_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ids.u32").write_bytes(struct.pack("<4I", 7, 8, 7, 9))
            job = {"control_profile": builder.PROFILE, "width": 2, "height": 2,
                   "object_ids": "ids.u32", "default": {"intensity": .5},
                   "objects": {"7": {"intensity": 0, "tone": .25, "structure": .75}}}
            (root / "job.json").write_text(json.dumps(job))
            report = builder.build(root / "job.json", root / "map")
            values = struct.unpack("<16f", (root / "map/control.rgba32f").read_bytes())
            self.assertEqual(values[:4], (0, .25, .75, 1))
            self.assertEqual(values[4:8], (.5, 1, 1, 1))
            self.assertEqual(values[8:12], values[:4])
            self.assertEqual(report["object_pixel_counts"], {"7": 2, "8": 1, "9": 1})
            self.assertFalse(report["vendor_validated"])
            self.assertTrue((root / "map/control-preview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with self.assertRaises(FileExistsError):
                builder.build(root / "job.json", root / "map")

    def test_rejects_unverified_alpha_and_invalid_strengths(self):
        for value in ({"alpha": 0}, {"tone": 2}, {"intensity": float("nan")},
                      {"structure": True}, {"tone": "1"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                builder.strengths(value)

    def test_short_id_plane_produces_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ids.u32").write_bytes(b"\0")
            (root / "job.json").write_text(json.dumps({"control_profile": builder.PROFILE,
                "width": 2, "height": 2, "object_ids": "ids.u32"}))
            with self.assertRaises(ValueError): builder.build(root / "job.json", root / "map")
            self.assertFalse((root / "map").exists())


if __name__ == "__main__":
    unittest.main()
